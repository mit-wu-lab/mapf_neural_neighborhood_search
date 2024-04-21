from util import *
from run import run_defaults
import mapf

def preprocess(a, graph=None, inputs=None, save=True):
    if graph is None:
        config, graph = build_mapf_config_graph(a)
    if 'data_dir' in a:
        out_path = a.data_dir / a.out_name + '.npz'
        if out_path.exists(): return dict(np.load(out_path))

    ## Inputs ##
    if inputs is not None:
        npz, substats = inputs
    else:
        npz = np.load(a.data_dir / 'data.npz')
        substats = pd.read_feather(a.data_dir / 'substats.ftr')
    delimiter = -1 if npz['goals'].dtype == np.int16 else np.iinfo(np.uint16).max
    # List of goal locations for all agents
    goals = unpack(npz['goals'], delimiter=delimiter).reshape(a.num_windows, a.num_agents)
    # Paths for initial solutions
    paths_init = unpack(npz['paths_init'], delimiter=delimiter).reshape(a.num_windows, a.num_agents)
    # Shortest paths for all agents when disregarding conflicts
    paths_shortest = unpack(npz['paths_shortest'], delimiter=delimiter).reshape(a.num_windows, a.num_agents)
    # The index of the subset selected to be replanned at each iteration
    selected_indices = np.zeros((a.num_windows, a.lns_iterations), dtype=int)
    # List of agents (in replan order) for each considered subset
    permutations = npz['permutations'].reshape(a.num_windows, a.lns_iterations, a.num_subsets, a.num_subset_agents)
    # Cost of each considered subset after replanning
    replanned_subset_costs = substats.final_cost.values[1:].reshape(a.num_windows, a.lns_iterations, a.num_subsets)
    # New paths for the selected subsets
    paths_replanned = unpack(npz['paths_replanned'], delimiter=delimiter).reshape(a.num_windows, a.lns_iterations, a.num_subset_agents)

    n_preprocess_windows = a.num_windows // a.keep_every or 1
    ## Output arrays ##
    # Shortest path location indices, shared across all datapoints within each window
    # Shape (n_preprocess_windows, num_agents, length of longest shortest path)
    shortest_paths_indices = []
    # Shortest path lengths
    # Shape (n_preprocess_windows, num_agents, maximum number of goals = maximum number of path segments)
    shortest_paths_segments_lengths = []
    # Concatenation of occupancy indices for every drive
    occupancy_indices = []
    # Ranges within occupancy_indices for all num_agents agents at each iteration
    # Each is shape (n_preprocess_windows * lns_iterations, num_agents)
    occupancy_iteration_starts = []
    occupancy_iteration_lens = []
    # subset cost (absolute and subtracted by the summed shortest path cost). Roughly shape (n_preprocess_windows * lns_iterations * a.num_subsets, 2)
    costs = []
    # Window index (for indexing into shortest_paths) and iteration index (for indexing into occupancy_iteration_*) for each subset.
    # Roughly shape (n_preprocess_windows * lns_iterations * a.num_subsets, 2)
    indices = []
    # subset agents for indexing into shortest_paths
    # Roughly shape (n_preprocess_windows * lns_iterations * a.num_subsets, num_subset_agents)
    agents = []

    # Features from MAPF-ML-LNS https://ojs.aaai.org/index.php/AAAI/article/view/21168
    num_linear_static_features, num_linear_dynamic_features = a.num_linear_features
    linear_static_features = [] # (n_preprocess_windows * num_agents, num_linear_static_features)
    linear_dynamic_features = [] # (n_preprocess_windows * lns_iterations * num_agents, num_linear_dynamic_features)

    a.verbosity >= 0 and print(f'Out of {a.num_windows} windows total, will preprocess {n_preprocess_windows}')
    for w in range(n_preprocess_windows):
        window_index = min((w + 1) * a.keep_every, a.num_windows) - 1
        if w % 5 == 0:
            a.verbosity >= 0 and print(f'Starting {window_index}th window = {w}th preprocessed window')

        curr_iteration_occupancy_starts, curr_iteration_occupancy_lens = [], []
        curr_shortest_costs = np.array([len(path) - 1 for path in paths_shortest[window_index]])
        for agent, (path, cost, gs) in enumerate(zip(paths_shortest[window_index], curr_shortest_costs, goals[window_index])):
            last_goal_time = 0
            segment_lengths = []
            for n, loc in enumerate([gs] if np.isscalar(gs) else gs):
                time_to_goal = np.argmax(path[last_goal_time:] == loc)
                segment_lengths.append(time_to_goal)
                last_goal_time += time_to_goal
            shortest_paths_indices.append(path)
            shortest_paths_segments_lengths.append(segment_lengths)

            linear_static_features.append([cost, path[0] // graph.cols, path[0] % graph.cols, path[-1] // graph.cols, path[-1] % graph.cols, graph.get_degree(path[-1])])

        heats = np.zeros(graph.size)
        curr_paths = paths_init[window_index]
        curr_costs = np.array([len(path) - 1 for path in curr_paths])
        if a.wait_at_goal:
            max_cost = curr_costs.max()
            curr_padded_paths = [np.pad(path, (0, max_cost - cost), mode='edge') for path, cost in zip(curr_paths, curr_costs)]
        for path in (curr_padded_paths if a.wait_at_goal else curr_paths):
            curr_iteration_occupancy_starts.append(len(occupancy_indices))
            curr_iteration_occupancy_lens.append(len(path))
            occupancy_indices.extend(path)
            np.add.at(heats, path, 1)
        curr_iteration_occupancy_starts = np.array(curr_iteration_occupancy_starts, dtype=np.uint32)
        curr_iteration_occupancy_lens = np.array(curr_iteration_occupancy_lens, dtype=np.uint16)

        for s in range(a.lns_iterations):
            occupancy_iteration_starts.append(curr_iteration_occupancy_starts.copy())
            occupancy_iteration_lens.append(curr_iteration_occupancy_lens.copy())

            for agent, path in enumerate(curr_paths):
                delay = curr_costs[agent] - curr_shortest_costs[agent]
                degree_counts = np.zeros(4)
                np.add.at(degree_counts, graph.get_degree(path[:-1]) - 1, 1)
                path_heats = heats[path]
                linear_dynamic_features.append([delay, delay / curr_shortest_costs[agent] if curr_shortest_costs[agent] else 0] + [getattr(path_heats, x)() for x in ('min', 'max', 'sum', 'mean')] + list(degree_counts))

            selected_index = selected_indices[window_index, s]
            selected_agents = None
            improved = True
            for i, p in enumerate(range(a.num_subsets)):
                replanned_sub_cost = replanned_subset_costs[window_index, s, p]
                subagents = permutations[window_index, s, p]
                if not a.include_infeasible and replanned_sub_cost > 2e9:
                    continue

                curr_sub_cost = curr_costs[subagents].sum()
                indices.append([w, w * a.lns_iterations + s])
                costs.append([replanned_sub_cost, curr_shortest_costs[subagents].sum(), curr_sub_cost])
                agents.append(subagents)
                if i == selected_index:
                    improved = replanned_sub_cost < curr_sub_cost
                    selected_agents = subagents

            if not improved: continue
            if a.wait_at_goal:
                changed_agents = []
                for agent, path in zip(selected_agents, paths_replanned[window_index, s]):
                    if len(path) != len(curr_paths[agent]) or (path != curr_paths[agent]).any():
                        curr_paths[agent] = path
                        curr_costs[agent] = len(path) - 1
                        changed_agents.append(agent)

                new_max_cost = curr_costs.max()
                if new_max_cost != max_cost:
                    changed_agents = range(len(curr_paths))
                max_cost = new_max_cost

                for agent in changed_agents:            
                    np.subtract.at(heats, curr_padded_paths[agent], 1)
                    curr_padded_paths[agent] = path = np.pad(curr_paths[agent], (0, max_cost - curr_costs[agent]), mode='edge')
                    curr_iteration_occupancy_starts[agent] = len(occupancy_indices)
                    curr_iteration_occupancy_lens[agent] = len(path)
                    occupancy_indices.extend(path)
                    np.add.at(heats, path, 1)
            else:
                for agent, path in zip(selected_agents, paths_replanned[window_index, s]):
                    np.subtract.at(heats, curr_paths[agent], 1)
                    np.add.at(heats, path, 1)
                    if len(path) != len(curr_paths[agent]) or (path != curr_paths[agent]).any():
                        curr_paths[agent] = path
                        curr_costs[agent] = len(path) - 1
                        curr_iteration_occupancy_starts[agent] = len(occupancy_indices)
                        curr_iteration_occupancy_lens[agent] = len(path)
                        occupancy_indices.extend(path)

    shortest_paths_indices = pad_arrays(shortest_paths_indices, np.iinfo(np.uint16).max).astype(np.uint16).reshape(n_preprocess_windows, a.num_agents, -1)
    shortest_paths_segments_lengths = pad_arrays(shortest_paths_segments_lengths, 0).astype(np.uint16).reshape(n_preprocess_windows, a.num_agents, -1)

    output = dict(
        shortest_paths_indices=shortest_paths_indices,
        shortest_paths_segments_lengths=shortest_paths_segments_lengths,
        occupancy_indices=np.array(occupancy_indices),
        occupancy_iteration_starts=np.array(occupancy_iteration_starts),
        occupancy_iteration_lens=np.array(occupancy_iteration_lens),
        costs=np.array(costs),
        indices=np.array(indices),
        agents=np.array(agents),
        linear_static_features=np.array(linear_static_features, dtype=np.uint16).reshape(n_preprocess_windows, a.num_agents, num_linear_static_features),
        linear_dynamic_features=np.clip(np.array(linear_dynamic_features, dtype=np.float16).reshape(n_preprocess_windows * a.lns_iterations, a.num_agents, num_linear_dynamic_features), None, np.finfo(np.float16).max), # The sum of path heats could be bigger than 2 ** 15 - 1 if the path is very long (e.g. in maze)
    )
    save and np.savez_compressed(out_path, **output)
    return output

def concat_preprocessed(paths_or_dicts, out_path=None):
    chunks = [dict(np.load(x)) if isinstance(x, str) else x for x in paths_or_dicts]
    max_shortest_path_len = max(chunk['shortest_paths_indices'].shape[2] for chunk in chunks)
    max_num_goals = max(chunk['shortest_paths_segments_lengths'].shape[2] for chunk in chunks)
    for name, pad_len, pad_value in ('shortest_paths_indices', max_shortest_path_len, np.iinfo(np.uint16).max), ('shortest_paths_segments_lengths', max_num_goals, 0):
        for chunk in chunks:
            chunk[name] = np.pad(chunk[name], ((0, 0), (0, 0), (0, pad_len - chunk[name].shape[2])), constant_values=pad_value)

    get_offsets = lambda L: np.cumsum(L) - L
    window_offsets = get_offsets([len(chunk['shortest_paths_indices']) for chunk in chunks])
    improvement_iteration_offsets = get_offsets([len(chunk['occupancy_iteration_starts']) for chunk in chunks])
    default_offsets = np.zeros(len(chunks), dtype=bool)
    all_offsets = dict(
        occupancy_iteration_starts=get_offsets([len(chunk['occupancy_indices']) for chunk in chunks]),
        indices=np.stack([window_offsets, improvement_iteration_offsets], axis=1),
    )
    def offset_concat(name):
        offsets = all_offsets.get(name, default_offsets)
        assert len(chunks) == len(offsets)
        assert (offsets[1:] >= offsets[:-1]).all(), 'Overflow may have occured'
        return np.concatenate([chunk.pop(name) + o for chunk, o in zip(chunks, offsets)])
    output = {name: offset_concat(name) for name in list(chunks[0].keys())}
    out_path and np.savez_compressed(out_path, **output)
    return output

def preprocess_linear(x, subagents, features_min, features_range):
    *dims, A, num_agent_features = x.shape
    *dims, S, SA = subagents.shape
    B = int(np.product(dims))
    x, subagents = x.view(B, A, num_agent_features), subagents.view(B, S, SA)
    subagents_mask = torch.zeros((B, S, A), device=x.device, dtype=bool)
    arange_BL = torch.arange(B, device=x.device).view(B, 1, 1)
    arange_S = torch.arange(S, device=x.device).view(1, S, 1)
    subagents_mask[arange_BL, arange_S, subagents] = True

    x = (x - features_min) / features_range
    x = x.view(B, 1, A, num_agent_features).expand(-1, S, -1, -1)
    sub_features = x[subagents_mask].view(B, S, SA, num_agent_features)
    sub_features = [getattr(sub_features, x)(dim=2) for x in ('min', 'max', 'sum', 'mean')]
    non_sub_features = x[~subagents_mask].view(B, S, A - SA, num_agent_features)
    non_sub_features = [getattr(non_sub_features, x)(dim=2) for x in ('min', 'max', 'sum', 'mean')]
    return torch.cat([x if isinstance(x, torch.Tensor) else x.values for x in sub_features + non_sub_features], dim=-1).view(*dims, S, -1)

def preprocess_linear_dataset(data, a, features_min, features_range, B=10):
    N, A, num_static_features = (linear_static_features := data['linear_static_features']).shape
    num_static_features, num_dynamic_features = a.num_linear_features
    L = data['linear_dynamic_features'].shape[0] // N
    S, SA = a.num_subsets, a.num_subset_agents
    linear_dynamic_features = data['linear_dynamic_features'].reshape(N, L, A, num_dynamic_features)
    all_subagents = data['agents'].reshape(N, L, S, SA)
    costs = data['costs'].reshape(N, L, S, 3)

    linear_features = []
    labels = []
    for b in tqdm(range(0, N, B)):
        B = min(B, N - b)
        BL = B * L
        subagents = all_subagents[b: b + B].astype(int).reshape(BL, S, SA)
        dynamic_features = linear_dynamic_features[b: b + B]
        static_features = np.broadcast_to(linear_static_features[b: b + B].reshape(B, 1, A, num_static_features), (B, L, A, num_static_features))
        agent_features = np.concatenate([static_features, dynamic_features], axis=-1, dtype=np.float32).reshape(BL, A, num_static_features + num_dynamic_features)
        sub_costs = costs[b: b + B].reshape(BL, S, 3)

        if a.frac_sampled_problems and a.frac_sampled_problems < 1:
            BL_sampled = np.sort(np.random.choice(BL, size=int(BL * a.frac_sampled_problems), replace=False))
            agent_features, subagents, sub_costs = agent_features[BL_sampled], subagents[BL_sampled], sub_costs[BL_sampled]
        if a.num_sampled_subsets:
            S_sampled = np.sort(np.random.choice(S, size=a.num_sampled_subsets, replace=False))
            subagents, sub_costs = subagents[:, S_sampled], sub_costs[:, S_sampled]
        linear_features.append(preprocess_linear(*to_torch([agent_features, subagents, features_min, features_range], device=a.device)).cpu().numpy())
        labels.append(sub_costs)
    return np.concatenate(linear_features), np.concatenate(labels) # (NL_sampled, num_features), (NL_sampled, 3)

parser = argparse.ArgumentParser('Preprocess MAPF data into training and validation data')
parser.add_argument('data_dir', type=Path, help='Directory to process')
preprocess_defaults = Namespace(
    out_name='preprocessed', # Output will be saved at data_dir/out_name.npz
    keep_every=1, # Keep window window_index if (window_index + 1) % keep_every == 0 and skip the rest of the windows to decorrelate the data
    num_linear_features=[6, 10],
    include_infeasible=True, # Whether to include infeasible subsets
    **run_defaults,
)

if __name__ == '__main__':
    args = parser.parse_args()
    config_path = args.data_dir / 'config.yaml'
    a = Namespace(config_path.load() if config_path.exists() else {}).setdefaults(
        data_dir=args.data_dir,
        **preprocess_defaults,
    )
    preprocess(a)
