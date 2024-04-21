import scipy.special
from util import *

class BaseEvaluator:
    def setup(self, graph, problem, solution, shortest):
        self.graph, self.problem, self.solution, self.shortest = graph, problem, solution, shortest

    def evaluate(self, all_subagents):
        return np.ones(len(all_subagents))

    def update(self, idx, subset, accepted):
        pass

def run(a, config=None, graph=None, runtime_coefs={}, init=None):
    a.setdefaults(run_dir=None)
    if config is None or graph is None:
        config, graph = build_mapf_config_graph(a)
    use_lns = isinstance(config, mapf.LNSConfig)

    np.random.seed(a.seed)
    system = mapf.System(graph, config)

    if a.run_dir and runtime_coefs is not None:
        dir = a.run_dir.mk().real
        runtime_filename = f'{Path(a.map).name.stem}-{a.num_agents}.yaml'
        while not runtime_coefs and dir.parent:
            if (runtime_file := dir / 'runtimes' / runtime_filename).exists():
                runtime_coefs = {(r['solver'], r['subset_fn'], r['num_subset_agents']): (r['w'], r['b']) for r in runtime_file.load()}
                print(f'Loaded subset time coefficients from {runtime_file}')
                break
            dir = dir.parent

    criteria_fn = lambda sub: sub.solution.cost - sub.initial_cost + sub.solution.runtime / max_subruntime # The last term is always <= 1

    if filter := a.filter:
        selection_counts = np.zeros(config.num_agents)
        count_decay = 1

    if a.collect_data: goals, paths_init, paths_shortest, permutations, paths_replanned = [], [], [], [], []
    stats, substats = [], []
    for window in range(a.num_windows):
        if a.scenario:
            scen = Path(__file__).parent / 'scenarios/random' / a.map.replace('.map', '') + '-random-' + a.seed + '.scen'
            prob_df = pd.read_csv(scen, sep='\t', skiprows=1, usecols=[4, 5, 6, 7], header=None, names=['start_col', 'start_row', 'goal_col', 'goal_row'], nrows=a.num_agents)
            starts = prob_df.start_row * graph.cols + prob_df.start_col
            goals = prob_df.goal_row * graph.cols + prob_df.goal_col
            problem = mapf.Problem(starts.values, goals.values.reshape(-1, 1))
            system.set_problem(problem)
        else:
            problem = system.get_problem()

        a.collect_data and goals.extend(split(problem.get_goal_locations(), problem.get_num_goals()))

        occupancy = mapf.ArrayOccupancy(graph)

        if use_lns:
            assert not (np.isinf(a.lns_iterations) and np.isinf(a.time_limit))

            lns = mapf.LNS(occupancy, graph, config)
            w = b = None
            if runtime_coefs:
                w, b = runtime_coefs[a.solver, a.subset_fn, a.num_subset_agents]

            if a.max_init_time is not None:
                config.max_solution_time = a.max_init_time
            if init is None and a.run_dir and (init_dir := a.run_dir / 'init').exists():
                init = (init_dir / 'init.npz').load()

            solution = mapf.Solution()
            if init is None:
                graph.preprocess_heuristics(problem.get_goal_locations())
                if a.init_solver == 'PPS':
                    mapf.run_PPS(problem, solution, config)
                else:
                    mapf.PP(occupancy, graph, config).repeat_solve(problem, solution)
            else:
                assert (init['start_locations'] == problem.get_start_locations()).all()
                assert (init['goal_locations'] == problem.get_goal_locations()).all()
                graph.import_heuristics(problem, init['heuristics'])
                if 'path_locations' in init:
                    solution.set_path_locations(init['path_locations'], init['path_lengths'])
                else:
                    config.max_path_plan_time = np.inf # We already know that this plan order will work, restricting the time isn't necessary
                    mapf.PP(occupancy, graph, config).solve(problem, solution, init['pp_order'])
                    config.max_path_plan_time = a.max_path_plan_time
                solution.runtime = init['init_runtime']
                solution.LL_num_expanded = init.get('LL_num_expanded', 0)
                solution.LL_num_generated = init.get('LL_num_generated', 0)
            if np.isinf(solution.cost): raise ValueError('No solution')
            lns.initialize(problem, solution)
            config.max_solution_time = a.max_solution_time
            solution, shortest = lns.solution, lns.shortest
            initial_cost, shortest_cost = solution.cost, shortest.cost
            init_runtime = solution.runtime
            if np.isinf(solution.cost):
                raise ValueError('No solution')
            if a.selector == 'unguided':
                a.selector = BaseEvaluator()
            a.selector and a.selector.setup(graph, problem, solution, shortest)

            substats.append([window, 0, shortest.cost, solution.cost, solution.runtime, solution.LL_num_expanded, solution.LL_num_generated])
            a.collect_data and paths_shortest.extend(split(shortest.get_path_locations(), shortest.get_path_lengths()))
            a.collect_data and paths_init.extend(split(solution.get_path_locations(), solution.get_path_lengths()))

            last_accept_it = 0
            it, runtime = 1, 0
            while it <= a.lns_iterations and runtime < a.time_limit:
                subsets = np.array([lns.generate_subset() for _ in range(a.num_subsets)])
                all_subagents = np.array([sub.agents for sub in subsets])
                use_count = use_count_accepted = 0
                S = len(subsets)

                keep_mask = np.ones(S, dtype=bool)
                if filter:
                    sub_selection_counts = selection_counts[all_subagents].sum(axis=1)
                    keep_count_threshold = np.random.choice(sub_selection_counts)
                    keep_mask = sub_selection_counts <= keep_count_threshold

                selected_idxs = []
                if a.selector:
                    select_start_time = time()
                    all_selector_scores = np.full(S, np.inf)
                    scores = all_selector_scores
                    evaluate_mask = (keep_mask | a.solve_all) & np.isinf(scores)
                    evaluate_subagents = all_subagents[evaluate_mask]
                    if len(evaluate_subagents): scores[evaluate_mask] = a.selector.evaluate(evaluate_subagents)
                    selector_scores = np.where(keep_mask, scores, np.inf)
                    selected_idxs = np.argsort(selector_scores)[:a.num_select]
                    select_time = time() - select_start_time
                solve_idxs = selected_idxs if a.selector and not a.solve_all else range(S)
                solve_subsets = subsets[solve_idxs]
                [lns.subsolve(sub) for sub in solve_subsets]
                max_subruntime = max(sub.solution.runtime for sub in solve_subsets)

                # sort by {selected, kept, unkept}, then sort by biggest decrease in cost, tiebreak by smaller runtime
                sorted_idxs = sorted(solve_idxs, key=lambda idx: (idx not in selected_idxs, not keep_mask[idx], criteria_fn(subsets[idx])))
                if w: max_subruntime = max([0, *((sum(x * y for x, y in zip(w, get_runtime_stats(sub.solution))) + b) for sub in solve_subsets)])
                if a.select_best_rand_from:
                    i = np.random.choice(S, size=a.select_best_rand_from, replace=False).min()
                    sorted_idxs.insert(0, sorted_idxs.pop(i))
                applied_idx = sorted_idxs[0]

                for idx in sorted_idxs:
                    sub, subagents = subsets[idx], all_subagents[idx]
                    subsol = sub.solution
                    substats.append([window, it, sub.initial_cost, subsol.cost, subsol.runtime, subsol.LL_num_expanded, subsol.LL_num_generated, *get_pbs_stats(subsol), config.subset_fn, config.num_subset_agents, sub.runtime_generate_agents,
                        *([all_selector_scores[idx], select_time] if a.selector else []),
                        *lif(filter, keep_mask[idx]),
                    ])

                    if idx == applied_idx: # best
                        accepted = lns.merge(sub)

                        a.selector and a.selector.update(idx, sub, accepted)
                        if a.collect_data:
                            subsol_path_locations = subsol.get_path_locations()
                            if len(subsol_path_locations) == 0:
                                assert np.isinf(subsol.cost)
                                subsol_paths = [[]] * a.num_subset_agents # TODO handle variable agents
                            else:
                                subsol_paths = split(subsol_path_locations, subsol.get_path_lengths())
                            paths_replanned.extend(subsol_paths)

                        if filter:
                            count_decay = min(count_decay, 1 / (it - last_accept_it))
                            selection_counts *= 1 - (count_decay if accepted else 0)
                            selection_counts[subagents] += 1

                        last_accept_it = it if accepted else last_accept_it
                    a.collect_data and permutations.append(subagents)
                runtime += max_subruntime
                it += 1
        else:
            solver_cls = getattr(mapf, a.solver)
            assert solver_cls in [mapf.PBS]
            solver = solver_cls(occupancy, graph, config)
            solution = mapf.PBSSolution() if isinstance(solver, mapf.PBS) else mapf.Solution()
            solver.solve(problem, solution)

        stats.append([window, system.num_finished_tasks, solution.cost, initial_cost, shortest_cost, solution.runtime, init_runtime, solution.LL_num_expanded, solution.LL_num_generated, *get_full_pbs_stats(solution)])
        if np.isinf(solution.cost): break
        system.advance(solution)
        occupancy.clear()
    df_stats = pd.DataFrame(stats, columns=get_stats_header(solution))
    df_substats = pd.DataFrame(substats, columns=[*get_substats_header(subsol), *lif(a.selector, 'selector_score', 'selection_time'), *lif(filter, 'keep')]) if use_lns else None
    dtype = np.int16 if graph.size <= np.iinfo(np.int16).max + 1 else np.uint16
    if graph.size > np.iinfo(np.uint16).max + 1:
        valid = np.ones(graph.size, dtype=bool)
        valid[graph.get_obstacles()] = False
        assert not valid[np.iinfo(np.uint16).max:].any()
    output = dict(
        goals=pack(goals).astype(dtype),
        paths_init=pack(paths_init).astype(dtype),
        paths_shortest=pack(paths_shortest).astype(dtype),
        permutations=np.array(permutations, dtype=np.int16),
        paths_replanned=pack(paths_replanned).astype(dtype),
    ) if a.collect_data else None
    if a.run_dir:
        df_stats.to_feather(a.run_dir / 'stats.ftr')
        use_lns and df_substats.to_feather(a.run_dir / 'substats.ftr')
        a.collect_data and np.savez_compressed(a.run_dir / 'data.npz', **output)
    return df_stats, *lif(use_lns, df_substats), *lif(a.collect_data, output)

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)

run_defaults = Namespace(
    map=None,
    simulation_window=5,
    num_windows=1,
    num_agents=800,
    verbosity=-1,
    use_cpp_rng=True,
    scenario=False,
    seed=0,
    init_solver=None,
    max_init_time=None,
    solver='PBS', # or 'PP'
    occupancy='Array', # or 'IntervalList'
    wait_at_goal=False,

    reaction_factor=0.01,
    num_subsets=1,
    num_subset_agents=0,
    num_subset_agents_fn=None,
    select_best_rand_from=None,
    subset_fn='random',
    filter=None,
    selector=None,
    num_select=1,
    acceptance_threshold=0,
    solve_all=False,
    lns_iterations=np.inf,
    time_limit=np.inf,
    max_solution_time=np.inf,
    max_path_plan_time=np.inf,
    subset_criteria='cost', # or 'cost/time'
    subset_noise=0,

    collect_data=False,
)

if __name__ == '__main__':
    args = parser.parse_args()
    a = load_config(args.run_dir).setdefaults(
        run_dir=args.run_dir,
        **run_defaults
    )
    print(f'Running MAPF for {a.run_dir.real}')
    if (a.run_dir / 'stats.ftr').exists():
        print('Results already exist. Skipping')
        exit()
    run(a)