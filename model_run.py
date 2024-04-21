from util import *
from train import *
from run import run, BaseEvaluator
import mapf
import torch

class Evaluator(BaseEvaluator):
    def __init__(self, net, a):
        self.net = net
        self.a = a
        self.use_shortest = 'shortest' in a.features
        a.setdefaults(G_orig=a.G)

    def get_costs(self, all_subagents):
        if self.a.loss != 'mse': return Namespace()
        return Namespace((k, torch.tensor([sol.subcost(subagents) for subagents in all_subagents], device=self.a.device, dtype=torch.float32)) for k, sol in [('shortest', self.shortest), ('init', self.solution)])

    def predict(self, pred_args):
        with torch.no_grad():
            float16 = torch.bfloat16 if self.a.device == 'cpu' else torch.float16
            with torch.autocast(device_type=self.a.device[:4], dtype=float16, enabled=self.a.use_amp):
                return self.net(pred_args).preds.cpu().numpy()
    
    def evaluate(self, all_subagents):
        raise NotImplementedError

class SubsetEvaluator(Evaluator):
    def __init__(self, net, a):
        super().__init__(net, a)
        self.use_init = 'init' in a.features
        self.use_separate_static = 'separate_static' in a.features
        self.num_channels = 1 + self.use_init + self.use_shortest + self.use_separate_static
        self.T = (a.T - 1) // a.dT + 1
        self.occupancy = np.zeros((a.num_subsets, self.num_channels, a.G.rows, a.G.cols, self.T), dtype=np.float32)
        self.obstacle_channel = a.num_subsets * (self.num_channels - 1) * a.G.size
        # Obstacles
        if self.use_separate_static:
            # Similar to self.occupancy[:, -1, a.obstacle_rows, a.obstacle_cols, :] = 1
            np.add.at(self.occupancy.reshape(-1, self.T), self.obstacle_channel + a.obstacles, 1)

    def evaluate(self, all_subagents):
        a = self.a
        S = len(all_subagents)
        occupancy = self.occupancy[:S]
        if self.use_separate_static:
            occupancy[:, :-1] = 0
        else:
            occupancy[:] = 0
            np.add.at(self.occupancy.reshape(-1, self.T), self.obstacle_channel + a.obstacles, 1)
        augment = np.random.choice(2, size=(S, 2)).astype(bool) if (a.reflection_symmetry or a.rotation_symmetry) else np.zeros((S, 2), dtype=bool)
        if not a.reflection_symmetry:
            augment[:, 1] = augment[:, 0]
        mapf.populate_occupancy(a.G_orig, self.solution, self.shortest, all_subagents, occupancy.reshape(S, self.num_channels, a.G.size, self.T), self.use_shortest, self.use_init, augment, a.prepool, a.dT)
        pred_args = Namespace(occupancy=torch.tensor(occupancy, device=a.device), costs=self.get_costs(all_subagents))
        return self.predict(pred_args)

class MultiSubsetEvaluator(Evaluator):
    def __init__(self, net, a):
        super().__init__(net, a)
        self.arange_T = np.arange(a.T).reshape(1, -1)

    def get_paths(self, solution):
        a = self.a
        locations_ = np.expand_dims(solution.get_path_locations_padded(a.T, a.wait_at_goal)[:, 0: a.T: a.dT], axis=0)
        lengths = np.expand_dims((np.clip(solution.get_path_lengths(), a.T if a.wait_at_goal else -np.inf, np.inf) - 1) // a.dT + 1, axis=0)
        return to_torch(Namespace(locations_=locations_, lengths=lengths), device=a.device, dtype=torch.int64)

    def setup(self, graph, problem, solution, shortest):
        super().setup(graph, problem, solution, shortest)
        self.shortest_paths = self.get_paths(shortest) if self.use_shortest else None

    def evaluate(self, all_subagents):
        a = self.a
        aug_vert, aug_hor = torch.rand(2).numpy() < 0.5 if (a.reflection_symmetry or a.rotation_symmetry) else (False, False)
        if not a.reflection_symmetry: aug_vert = aug_hor # There should only be rotational symmetry, not reflection
        def augment(paths):
            row, col = paths.locations_ // a.G_orig.cols // a.prepool, paths.locations_ % a.G_orig.cols // a.prepool
            if aug_vert: row = a.G.rows - row - 1
            if aug_hor: col = a.G.cols - col - 1
            return Namespace(locations_=row * a.G.cols + col, lengths=paths.lengths)
        return self.predict(Namespace(
            costs=Namespace((k, v.unsqueeze(0)) for k, v in self.get_costs(all_subagents).items()),
            subagents=torch.tensor(all_subagents, dtype=torch.int64, device=a.device).unsqueeze(0),
            init=augment(self.get_paths(self.solution)),
            shortest=augment(self.shortest_paths),
        )).squeeze(0)

class LinearMultiSubsetEvaluator(Evaluator):
    def setup(self, graph, problem, solution, shortest):
        super().setup(graph, problem, solution, shortest)
        a = self.a
        self.shortest_lengths = torch.tensor(shortest.get_path_lengths(), device=a.device)
        starts = torch.tensor(problem.get_start_locations(), device=a.device)
        num_goals = torch.tensor(problem.get_num_goals(), device=a.device)
        last_goal_idxs = num_goals.cumsum(dim=0) - 1
        goals = problem.get_goal_locations()[last_goal_idxs.cpu().numpy()]
        self.goal_degrees = torch.tensor(graph.get_degree(goals), device=a.device, dtype=torch.int64)
        goals = torch.tensor(goals, device=a.device)
        self.static_features = torch.cat([x.reshape(-1, 1) for x in [self.shortest_lengths - 1, starts // a.G.cols, starts % a.G.cols, goals // a.G.cols, goals % a.G.cols, self.goal_degrees]], axis=1).to(dtype=torch.float32, device=a.device)
        self.heats = torch.zeros(a.G.size, device=a.device)
        self.degree_counts = torch.zeros((a.num_agents, 4), device=a.device)
        self.need_update = True
        self.initialized_dynamic_features = False
        self.arange_A = torch.arange(a.num_agents, device=a.device)

    def update_heat(self, new_locations, old_locations=None):
        self.heats.index_add_(0, new_locations, torch.ones(len(new_locations), device=self.a.device))
        if old_locations is not None: self.heats.index_add_(0, old_locations, -torch.ones(len(old_locations), device=self.a.device))

    def update_degree(self, repeated_agents, degrees, agents):
        self.degree_counts[agents] = 0
        self.degree_counts[agents, self.goal_degrees[agents] - 1] = -1
        self.degree_counts.reshape(-1).index_add_(0, repeated_agents * 4 + degrees - 1, torch.ones(len(degrees), device=self.a.device))

    def get_agent_features(self):
        if not self.need_update: return self.prev_features
        a = self.a
        path_lengths = self.solution.get_path_lengths()
        max_path_length = path_lengths.max()
        if a.wait_at_goal:
            path_locations = self.solution.get_path_locations_padded(max_path_length, True).flatten()
        else:
            path_locations = self.solution.get_path_locations()
        path_locations_t = torch.tensor(path_locations, device=a.device, dtype=torch.int64)
        path_lengths = torch.tensor(path_lengths, device=a.device)
        delay = (path_lengths - self.shortest_lengths).to(torch.float32)
        if a.wait_at_goal:
            path_lengths[:] = max_path_length
        repeated_agents = torch.repeat_interleave(self.arange_A, path_lengths)
        if not self.initialized_dynamic_features:
            self.update_heat(path_locations_t)
            self.update_degree(repeated_agents, torch.tensor(self.graph.get_degree(path_locations), device=a.device), self.arange_A)
            self.initialized_dynamic_features = True
        else:
            subagents = self.prev_subset.agents
            subsolution = mapf.Solution(self.solution, subagents)
            if a.wait_at_goal:
                new_locations = subsolution.get_path_locations_padded(max_path_length, True).flatten()
                prev_locations = self.prev_subset.solution.get_path_locations_padded(self.prev_max_path_length, True).flatten()
            else:
                new_locations = subsolution.get_path_locations()
                prev_locations = self.prev_subset.solution.get_path_locations()
            subagents = torch.tensor(subagents, device=a.device)
            self.update_heat(torch.tensor(new_locations, device=a.device), torch.tensor(prev_locations, device=a.device))
            self.update_degree(torch.repeat_interleave(subagents, path_lengths[subagents]), torch.tensor(self.graph.get_degree(new_locations), device=a.device), subagents)
        heats = self.heats[path_locations_t]
        
        mask = path_lengths.view(-1, 1) > torch.arange(max_path_length, device=a.device).reshape(1, -1)
        square_heats = torch.full_like(mask, torch.inf, dtype=torch.float32)
        square_heats[mask] = heats
        mins = square_heats.min(dim=1).values
        square_heats[~mask] = -torch.inf
        maxs = square_heats.max(dim=1).values
        square_heats[~mask] = torch.nan
        means = square_heats.nanmean(dim=1)
        sums = means * path_lengths
        self.prev_features = features = torch.cat([
            self.static_features,
            delay.reshape(-1, 1),
            torch.where(self.shortest_lengths == 1, torch.tensor(0.0, device=a.device), delay / (self.shortest_lengths - 1)).reshape(-1, 1),
            mins.reshape(-1, 1),
            maxs.reshape(-1, 1),
            sums.reshape(-1, 1),
            means.reshape(-1, 1),
            self.degree_counts,
        ], axis=1)
        self.prev_max_path_length = max_path_length
        return features

    def evaluate(self, all_subagents):
        a = self.a
        return self.predict(Namespace(
            costs=Namespace((k, v.unsqueeze(0)) for k, v in self.get_costs(all_subagents).items()),
            features=self.net.preprocess(self.get_agent_features(), torch.tensor(all_subagents, device=a.device)).unsqueeze(0),
        )).squeeze(0)

    def update(self, idx, subset, accepted):
        self.prev_subset, self.need_update = subset, accepted

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)

model_run_defaults = Namespace(
    train_dirs=[],
    model_weights=[],
    **train_defaults
)

if __name__ == '__main__':
    args = parser.parse_args()
    config = load_config(args.run_dir)
    if (args.run_dir / 'stats.ftr').exists():
        print('Results already exist. Skipping')
        exit()
    train_dir = args.run_dir.parent.parent
    a = config.new().setdefaults(**load_config(train_dir, parent=1)).setdefaults(
        train_dir=train_dir,
        run_dir=args.run_dir,
        **model_run_defaults,
    )
    setup_mapf(a)
    net = eval(a.network + 'Network')(a).to(a.device)

    if a.checkpoint_step is None: a.checkpoint_step = a.train_steps
    loaded_step, start_time = restore(a, net)
    assert loaded_step == a.checkpoint_step

    net.eval()
    a.selector = eval(a.network + 'Evaluator')(net, a)
    run(a)
