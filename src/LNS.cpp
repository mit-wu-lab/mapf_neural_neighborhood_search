#include "LNS.h"

void LNS::generate_subset(Subset &subset) {
    auto start_time = get_time();
    if (config.subset_fn == "agent_neighborhood") LNS::generate_agent_neighborhood(subset);
    else if (config.subset_fn == "intersection_neighborhood") LNS::generate_intersection_neighborhood(subset);
    else if (config.subset_fn == "random") { // Random subset
        subset.agents.resize(config.num_agents);
        std::iota(subset.agents.begin(), subset.agents.end(), 0);
    } else {
        ostringstream ss; ss << "Got unsupported config.subset_fn \"" << config.subset_fn << "\"" << endl;
        throw invalid_argument(ss.str());
    }
    config.use_cpp_rng ? std::shuffle(subset.agents.begin(), subset.agents.end(), rng) : std::random_shuffle(subset.agents.begin(), subset.agents.end());
    subset.agents.resize(config.num_subset_agents);
    subset.runtime_generate_agents = get_runtime(start_time);
    subset.initial_cost = solution->subcost(subset.agents);
    subset.shortest_cost = shortest->subcost(subset.agents);
}

void LNS::generate_agent_neighborhood(Subset &subset) {
    if (array_occupancy == nullptr) throw invalid_argument("generate_agent_neighborhood can only be used with ArrayOccupancy");

    int a = -1;
    auto sample_delayed_agent = [&]() {
        while (true) {
            if (tabu_agents.size() == config.num_agents) tabu_agents.clear();
            double max_delay = -1;
            for (int i = 0; i < config.num_agents; i++) {
                if (tabu_agents.find(i) != tabu_agents.end()) continue;
                double delay = solution->path_costs[i] - shortest->path_costs[i];
                if (delay > max_delay) {
                    max_delay = delay;
                    a = i;
                }
            }
            if (max_delay == 0 && tabu_agents.size() > 0) tabu_agents.clear();
            else {
                tabu_agents.insert(a);
                return;
            }
        }
    };
    if (a == -1) sample_delayed_agent();
    unordered_set<int> subagents = {a};

    unordered_map<int, list<int>> goal_indices;
    // a random walk around the trajectory of agent a, looking for nearby conflicting agents
    auto random_walk = [&](int a, int t) {
        const Path &path = solution->paths[a];
        const vector<int> &gs = problem->goal_locations[a];
        int max_timestep = path.size() - 1;
        list<int> &g_times = goal_indices[a];
        if (g_times.empty()) {
            int j = gs.size() - 1;
            for (int i = max_timestep; i >= 0 && j >= 0; i--) {
                if (path[i].location == gs[j]) {
                    g_times.push_front(i);
                    j--;
                }
            }
        }
        int location = path[t].location;
        for (; t < max_timestep && subagents.size() < config.num_subset_agents; t++) {
            list<int> locations = G.get_neighbors(location);
            while (!locations.empty()) {
                list<int>::iterator it = locations.begin();
                std::advance(it, (config.use_cpp_rng ? rng() : rand()) % locations.size());
                auto gt_it = g_times.begin();
                auto gs_it = gs.begin();
                while (t >= *gt_it) { gt_it++; gs_it++; }
                if (t + 1 + G.compute_heuristics(*it, vector<int>(gs_it, gs.end())) < max_timestep) {
                    array_occupancy->get_constrained_agents(*it, t + 1, subagents);
                    array_occupancy->get_edge_constrained_agents(location, *it, t + 1, subagents);
                    location = *it;
                    break;
                }
                locations.erase(it);
            }
        }
    };
    int failed_tries = 0;
    random_walk(a, 0);
    while (subagents.size() < config.num_subset_agents) {
        int curr_num_subagents = subagents.size();
        random_walk(a, (config.use_cpp_rng ? rng() : rand()) % solution->paths[a].size());
        if (subagents.size() == curr_num_subagents) {
            failed_tries++;
            if (failed_tries == 10) {
                tabu_agents.insert(subagents.begin(), subagents.end());
                if (tabu_agents.size() == config.num_agents) tabu_agents = subagents;
                sample_delayed_agent();
                subagents.insert(a);
                failed_tries = 0;
            }
        } else {
            failed_tries = 0;
            auto it = subagents.begin();
            std::advance(it, (config.use_cpp_rng ? rng() : rand()) % subagents.size());
            a = *it;
        }
    }
    subset.agents = vector<int>(subagents.begin(), subagents.end());
}

void LNS::generate_intersection_neighborhood(Subset &subset) {
    if (array_occupancy == nullptr) throw invalid_argument("generate_agent_neighborhood can only be used with ArrayOccupancy");
    if (intersections.empty()) {
        for (int i = 0; i < G.size; i++)
            if (G.get_degree(i) >= 3)
                intersections.insert(i);
    }

    auto it = intersections.begin();
    std::advance(it, (config.use_cpp_rng ? rng() : rand()) % intersections.size());
    unordered_set<int> subagents;

    unordered_set<int> queued = { *it };
    list<int> queue = { *it }; // FIFO
    while (!queue.empty() && subagents.size() < config.num_subset_agents) {
        int curr = queue.front();
        queue.pop_front();
        if (intersections.find(curr) != intersections.end())
            array_occupancy->get_constrained_agents(curr, subagents);
        for (int next : G.get_neighbors(curr)) {
            if (queued.find(next) != queued.end()) continue;
            queue.push_back(next);
            queued.insert(next);
        }
    }
    subset.agents = vector<int>(subagents.begin(), subagents.end());
}

void LNS::subsolve(Subset &subset) {
    if (subset.solution) {
        subset.solution->runtime = 0;
        return; // Already solved
    }
    config.num_subset_agents = subconfig.num_agents = subset.agents.size();
    subset.solution = subsolver->make_empty_solution();
    subset.solution->runtime += set_solution_occupancy(*solution, subset.agents, false);
    subset.problem = Problem(*problem, subset.agents);
    subsolver->solve(subset.problem, *subset.solution);
    if (!isinf(subset.solution->cost)) {
        subset.solution->runtime += set_solution_occupancy(*subset.solution, true);
        set_solution_occupancy(*subset.solution, false);
    }
    set_solution_occupancy(*solution, subset.agents, true);
}

bool LNS::merge(const Subset &subset) {
    config.num_subset_agents = subconfig.num_agents = subset.agents.size();
    set_solution_occupancy(*solution, subset.agents, false);
    Solution &subsolution = *subset.solution;
    solution->runtime += subsolution.runtime;
    solution->LL_num_expanded += subsolution.LL_num_expanded;
    solution->LL_num_generated += subsolution.LL_num_generated;

    bool accept = subsolution.cost < subset.initial_cost + config.acceptance_threshold;
    if (accept) {
        int index = 0;
        for (const int agent : subset.agents) {
            solution->paths[agent] = subsolution.paths[index];
            solution->path_costs[agent] = subsolution.path_costs[index];
            index++;
        }
        solution->cost += subsolution.cost - subset.initial_cost;
    }
    set_solution_occupancy(*solution, subset.agents, true);
    iteration++;
    return accept;
}