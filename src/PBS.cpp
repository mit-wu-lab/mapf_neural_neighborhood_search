#include "PBS.h"

void PBS::solve(const Problem &problem, PBSSolution &solution) {
    auto start_time = get_time();
    this->problem = &problem;
    this->solution = &solution;

    PBSNode *root = generate_root_node();
    if (root && root->conflicts.empty()) {
        solution.cost = root->g_val;
        best_node = root;
    }

    while (!dfs.empty() && isinf(solution.cost) && get_runtime(start_time) < config.max_solution_time) {
        PBSNode *curr = dfs.back();
        dfs.pop_back();
        if (curr->conflicts.empty()) {
            solution.cost = curr->g_val;
            best_node = curr;
            break;
        }

        update_paths(curr);
        choose_conflict(curr);

        if (curr->earliest_collision > best_node->earliest_collision ||
            curr->earliest_collision == best_node->earliest_collision && curr->g_val < best_node->g_val)
            best_node = curr;

        solution.HL_num_expanded++;

        if (config.verbosity == 2)
            cout << "Expand Node " << solution.HL_num_expanded << " ( cost = " << curr->g_val << " , #conflicts = " << curr->num_collisions() << " ) on conflict " << curr->conflict << endl;
        PBSNode *n1 = new PBSNode();
        PBSNode *n2 = new PBSNode();
        const auto [a1, a2, loc1, loc2, t] = curr->conflict;
        n1->priority = {a1, a2};
        n2->priority = {a2, a1};

        vector<Path *> copy_paths(curr_paths);
        vector<double> copy_path_costs(curr_path_costs);
        vector<PBSNode *> to_push;
        for (auto &n : {n1, n2}) {
            if (generate_child_node(n, curr)) {
                solution.HL_num_generated++;
                if (config.verbosity == 2) {
                    cout << "Generate #" << solution.HL_num_generated << " with "
                         << n->paths.size() << " new paths, "
                         << n->g_val - curr->g_val << " delta cost and "
                         << n->num_collisions() << " conflicts " << endl;
                }
                to_push.push_back(n);
            } else delete n;
            curr_paths = copy_paths;
            curr_path_costs = copy_path_costs;
        }

        if (to_push.empty()) nogood.emplace_back(a1, a2);
        else if (to_push.size() == 1) push_node(to_push.front());
        else if (PBSNode::compare_g_collision(n1, n2)) { push_node(n1); push_node(n2); } // n1 is worse
        else { push_node(n2); push_node(n1); } // n2 is worse
        curr->clear();
    }

    solution.runtime += get_runtime(start_time);
    if (!isinf(solution.cost)) {
        update_paths(best_node);
        for (int k = 0; k < config.num_agents; k++)
            solution.paths.push_back(*curr_paths[k]);
        solution.path_costs = curr_path_costs;
    }
    dfs.clear();
    nogood.clear();
    for (PBSNode *it : all_nodes) delete it;
    all_nodes.clear();
}

void PBS::update_paths(PBSNode *curr) {
    vector<bool> updated(config.num_agents, false);
    while (curr) {
        for (auto &[a, path, cost] : curr->paths) {
            if (!updated[a]) {
                curr_paths[a] = &path;
                curr_path_costs[a] = cost;
                updated[a] = true;
            }
        }
        curr = curr->parent;
    }
}

void PBS::find_conflicts(list<Conflict> &conflicts, int a1, int a2) {
    const Path *path1 = curr_paths[a1], *path2 = curr_paths[a2];
    if (path1 == nullptr || path2 == nullptr) return;
    auto start_time = get_time();
    int len1 = path1->size(), len2 = path2->size();
    int min_length = min(len1, len2), max_length = max(len1, len2);
    for (int timestep = 0; timestep < min_length; timestep++) {
        int loc1 = path1->at(timestep).location;
        int loc2 = path2->at(timestep).location;
        if (loc1 == loc2) { // node conflict
            conflicts.emplace_back(a1, a2, loc1, -1, timestep);
            break;
        } else if (timestep < min_length - 1) { // edge conflicts
            if (loc1 == path2->at(timestep + 1).location && loc2 == path1->at(timestep + 1).location) {
                conflicts.emplace_back(a1, a2, loc1, loc2, timestep + 1);  // edge conflict
                break;
            }
        }
    }
    if (config.wait_at_goal) {
        for (int timestep = min_length; timestep < max_length; timestep++) {
            int loc1 = path1->at(min(timestep, len1 - 1)).location;
            int loc2 = path2->at(min(timestep, len2 - 1)).location;
            if (loc1 == loc2) conflicts.emplace_back(a1, a2, loc1, -1, timestep);
        }
    }
    solution->runtime_find_conflicts += get_runtime(start_time);
}

void PBS::find_conflicts(list<Conflict> &conflicts) {
    for (int a1 = 0; a1 < config.num_agents; a1++)
        for (int a2 = a1 + 1; a2 < config.num_agents; a2++)
            find_conflicts(conflicts, a1, a2);
}

void PBS::find_conflicts(list<Conflict> &new_conflicts, int new_agent) {
    for (int a2 = 0; a2 < config.num_agents; a2++) {
        if (new_agent == a2) continue;
        find_conflicts(new_conflicts, new_agent, a2);
    }
}

void PBS::remove_conflicts(list<Conflict> &conflicts, int excluded_agent) {
    for (auto it = conflicts.begin(); it != conflicts.end();) {
        const auto [a1, a2, loc1, loc2, t] = *it;
        if (a1 == excluded_agent || a2 == excluded_agent) it = conflicts.erase(it);
        else ++it;
    }
}

void PBS::choose_conflict(PBSNode *curr) {
    if (curr->conflicts.empty()) return;

    const Conflict *nogood_conflict = nullptr;
    curr->earliest_collision = INT_MAX;
    for (const Conflict &conflict : curr->conflicts) {
        const auto [a1, a2, loc1, loc2, t] = conflict;
        if (t < curr->earliest_collision) {
            curr->earliest_collision = t;
            curr->conflict = conflict;
        }

        if (nogood_conflict) continue;
        for (const auto [ng1, ng2] : nogood)
            if ((a1 == ng1 && a2 == ng2) || (a1 == ng2 && a2 == ng1))
                nogood_conflict = &conflict;
    }
    if (nogood_conflict) curr->conflict = *nogood_conflict;
}

void PBS::set_occupancy(const unordered_set<int> &high_priority_agents, int current_agent, bool insert_or_remove) {
    auto start_time = get_time();

    for (const int i : high_priority_agents) { // hard constraints
        if (curr_paths[i] == nullptr) continue;
        occupancy.set_path_constraint(*curr_paths[i], insert_or_remove, i);
    }

    if (config.prioritize_start) // prioritize waits at start locations
        occupancy.set_constraints_for_starts(curr_paths, current_agent, insert_or_remove);

    solution->runtime_occupancy += get_runtime(start_time);
}

bool PBS::find_path(PBSNode *node, int agent) {
    unordered_set<int> high_priority_agents = node->priorities.get_reachable_nodes(agent);
    set_occupancy(high_priority_agents, agent, true);
    PathSolution path_solution = path_planner->solve(problem->starts[agent], problem->goal_locations[agent], true);
    set_occupancy(high_priority_agents, agent, false);
    if (path_solution.path.empty()) {
        if (config.verbosity == 2)
            cout << "Failed to find a path for agent " << agent << endl;
        return false;
    }

    solution->num_LL++;
    solution->LL_num_expanded += path_solution.num_expanded;
    solution->LL_num_generated += path_solution.num_generated;
    solution->runtime_plan_paths += path_solution.runtime;
    node->g_val += path_solution.cost - curr_path_costs[agent];
    for (auto it = node->paths.begin(); it != node->paths.end(); ++it) {
        if (std::get<0>(*it) == agent) {
            node->paths.erase(it);
            break;
        }
    }
    node->paths.emplace_back(agent, path_solution.path, path_solution.cost);
    curr_paths[agent] = &std::get<1>(node->paths.back());
    curr_path_costs[agent] = path_solution.cost;
    return true;
}

// return true if the agent keeps waiting at its start location until at least min_timestep
bool PBS::wait_at_start(const Path &path, int start_location, int min_timestep) {
    for (const auto &[location, timestep, _] : path) {
        if (timestep > min_timestep) return true;
        else if (location != start_location) return false;
    }
    return false;  // when the path is empty
}

void PBS::find_replan_agents(PBSNode *node, const list<Conflict> &conflicts,
                             unordered_set<int> &replan) {
    auto start_time = get_time();
    for (const auto &[a1, a2, loc1, loc2, t] : conflicts) {
        if (replan.find(a1) != replan.end() || replan.find(a2) != replan.end());
        else if (config.prioritize_start && wait_at_start(*curr_paths[a1], loc1, t)) replan.insert(a2);
        else if (config.prioritize_start && wait_at_start(*curr_paths[a2], loc2, t)) replan.insert(a1);
        else if (node->priorities.connected(a1, a2)) replan.insert(a1);
        else if (node->priorities.connected(a2, a1)) replan.insert(a2);
    }
    solution->runtime_find_replan_agents += get_runtime(start_time);
}

bool PBS::find_consistent_paths(PBSNode *node, int agent) {
    auto start_time = get_time();
    unordered_set<int> replan;
    if (agent != -1) replan.insert(agent);
    find_replan_agents(node, node->conflicts, replan);
    bool no_solution = false;
    int iterations = 0;  // number of loop iterations
    while (!replan.empty()) {
        if (no_solution = (iterations > node->paths.size() * 5)) break;
        int a = *replan.begin();
        replan.erase(a);
        iterations++;
        if (no_solution = !find_path(node, a)) break;
        remove_conflicts(node->conflicts, a);
        list<Conflict> new_conflicts;
        find_conflicts(new_conflicts, a);
        find_replan_agents(node, new_conflicts, replan);

        node->conflicts.splice(node->conflicts.end(), new_conflicts);
    }
    solution->runtime_find_consistent_paths += get_runtime(start_time);
    return !no_solution;
}

bool PBS::generate_child_node(PBSNode *node, PBSNode *parent) {
    node->parent = parent;
    node->g_val = parent->g_val;

    auto start_time = get_time();
    node->priorities.copy(node->parent->priorities);
    node->priorities.add(node->priority.first, node->priority.second);
    solution->runtime_copy_priorities += get_runtime(start_time);
    node->conflicts = node->parent->conflicts;
    return find_consistent_paths(node, node->priority.first);
}

bool PBS::initialize_paths(PBSNode *root) {
    int agent = 0;
    for (; agent < config.num_agents; agent++) {
        PathSolution path_solution = path_planner->solve(problem->starts[agent], problem->goal_locations[agent], true);
        solution->num_LL++;
        solution->LL_num_expanded += path_solution.num_expanded;
        solution->LL_num_generated += path_solution.num_generated;
        solution->runtime_plan_paths += path_solution.runtime;
        root->paths.emplace_back(agent, path_solution.path, path_solution.cost);
        curr_paths[agent] = &std::get<1>(root->paths.back());
        curr_path_costs[agent] = path_solution.cost;
        root->g_val += path_solution.cost;

        if (path_solution.path.empty()) {
            if (config.verbosity == 2)
                cout << "Failed to find initial solution for agent " << agent << " for root node" << endl;
            break;
        }
        occupancy.set_constraint_for_start(path_solution.path, true, agent);
    }
    for (int i = 0; i < agent; i++) occupancy.set_constraint_for_start(*curr_paths[i], false, i);
    return agent == config.num_agents;
}

PBSNode *PBS::generate_root_node() {
    auto start_time = get_time();
    curr_paths.resize(config.num_agents, nullptr);
    curr_path_costs.resize(config.num_agents);
    PBSNode *root = new PBSNode();

    if (config.verbosity == 2)
        cout << "Generate root node ..." << endl;

    if (!initialize_paths(root)) return nullptr;

    find_conflicts(root->conflicts);
    if (!find_consistent_paths(root, -1)) return nullptr;

    push_node(best_node = root);
    if (config.verbosity == 2)
        cout << "Done! (" << get_runtime(start_time) << "s)" << endl;
    return root;
}

void PBS::push_node(PBSNode *node) {
    dfs.push_back(node);
    all_nodes.push_back(node);
}
