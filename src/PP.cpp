#include "PP.h"

void PP::solve(const Problem &problem, Solution &solution, const vector<int> &agent_order, bool avoid_collisions, bool restore_occupancy) {
    auto start_time = get_time();
    solution.cost = 0;
    for (Path &path : solution.paths) path.clear();
    solution.paths.resize(config.num_agents);
    solution.path_costs.resize(config.num_agents);
    for (const int agent : agent_order) {
        PathSolution path_solution = path_planner->solve(problem.starts[agent], problem.goal_locations[agent], true);
        if (path_solution.path.empty()) {
            if (config.verbosity == 2)
                cout << "Failed to find a path for agent " << agent << endl;
            solution.cost = INFINITY;
            break;
        }
        solution.paths[agent] = path_solution.path;
        solution.path_costs[agent] = path_solution.cost;
        solution.LL_num_expanded += path_solution.num_expanded;
        solution.LL_num_generated += path_solution.num_generated;
        solution.cost += path_solution.cost;
        if (avoid_collisions) occupancy.set_path_constraint(path_solution.path, true, agent);
    }
    solution.runtime += get_runtime(start_time);
    if (restore_occupancy) solution.runtime += set_solution_occupancy(solution, agent_order, false);
}
