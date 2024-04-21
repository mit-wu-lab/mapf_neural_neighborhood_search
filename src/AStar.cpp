#include "AStar.h"

PathSolution AStar::solve(const State &start, const vector<int> &goal_locations, bool strict) {
    PathSolution solution;
    auto start_time = get_time();

    if (occupancy.is_constrained(start.location, start.location, 0)) {
        ostringstream ss; ss << "Start state " << start << " already occupied, implying that two agents start from the same location!";
        throw invalid_argument(ss.str());
    }

    double start_h_val = G.compute_heuristics(start.location, goal_locations);
    AStarNode *start_node = new AStarNode(start, 0, start_h_val, nullptr);
    solution.num_generated++;
    if (start_h_val < WEIGHT_MAX) {
        push_open(start_node); // start node
    } else {
        cerr << "The start and goal locations are disconnected!" << endl;
    }
    while (!open.list.empty() && get_runtime(start_time) < config.max_path_plan_time) {
        AStarNode *curr = open.list.front();
        open.pop();
        if (!curr->in_open_list) continue;
        curr->in_open_list = false;
        solution.num_expanded++;

        if (curr->state.location == goal_locations[curr->goal_index]) {
            curr->goal_index++;
            if (curr->goal_index == goal_locations.size()) {
                if (config.wait_at_goal && occupancy.is_constrained_after(curr->state.location, curr->state.timestep)) { // Goal is constrained by another agent's future trajectory, keep searching
                    curr->goal_index--;
                } else {
                    get_solution_path(curr, solution.path);
                    solution.cost = curr->g_val;
                    break;
                }
            }
        }

        for (const State &next_state : G.get_neighbors(curr->state)) {
            if (occupancy.is_constrained(curr->state.location, next_state.location, next_state.timestep)) continue;

            double next_h_val = G.compute_heuristics(next_state.location, vector(goal_locations.begin() + curr->goal_index, goal_locations.end()));
            if (next_h_val >= WEIGHT_MAX) continue; // This vertex cannot reach the goal vertex
            double next_g_val = curr->g_val + G.get_weight(curr->state.location, next_state.location);

            auto next = new AStarNode(next_state, next_g_val, next_h_val, curr);
            solution.num_generated++;
            const auto &it = best_nodes.find(next);
            if (it != best_nodes.end()) {
                AStarNode *existing_next = *it;
                if (AStarNode::compare_f_g(existing_next, next)) { // next is better
                    best_nodes.erase(it);
                    existing_next->in_open_list = false;
                } else {
                    delete (next);
                    continue;
                }
            }
            push_open(next);
        }
        if (open.list.empty() && !strict) { // no path found, try waiting while ignoring conflicts
            solution.has_conflicts = true;
            double wait_cost = G.get_weight(start.location, start.location);
            for (int t_wait : occupancy.get_constrained_timesteps(start.location)) {
                push_open(new AStarNode({start.location, t_wait, start.orientation}, t_wait * wait_cost, start_h_val, start_node));
                solution.num_generated++;
            }
            strict = true; // do not enter this block again
        }
    }
    solution.runtime += get_runtime(start_time);
    release_nodes();
    return solution;
}

void AStar::get_solution_path(const AStarNode *node, Path &path) {
    path.resize(node->state.timestep + 1);
    for (int t = node->state.timestep; t >= 0; t--) {
        if (node->state.timestep > t) {
            node = node->parent;
            assert(node->state.timestep <= t);
        }
        path[t] = {node->state.location, t, node->state.orientation};
    }
}
