#include "System.h"

Problem System::get_problem() {
    Problem problem;
    for (int k = 0; k < config.num_agents; k++) {
        const auto [location, t, orientation] = paths[k][timestep];
        problem.starts.emplace_back(location, 0, orientation);
        problem.goal_locations.push_back(goal_locations[k]);
    }
    return problem;
}

void System::set_problem(Problem &problem) {
    for (int k = 0; k < config.num_agents; k++) {
        paths[k][0] = problem.starts[k];
        goal_locations[k] = problem.goal_locations[k];
    }
}

const int EMPTY_AGENT = -1;

void System::advance(const Solution &solution) {
    vector<int> new_node_occupancy(G.size);
    vector<int> new_edge_occupancy(G.edge_size());

    int simulation_window = 0;
    for (const Path &path : solution.paths)
        simulation_window = max(simulation_window, int(path.size()));

    for (int t = 0; t <= simulation_window; t++) {
        bool success = true;
        std::fill(new_node_occupancy.begin(), new_node_occupancy.end(), EMPTY_AGENT);
        std::fill(new_edge_occupancy.begin(), new_edge_occupancy.end(), EMPTY_AGENT);
        int new_timestep = timestep + t;
        for (int k = 0; k < config.num_agents; k++) {
            const State &state = solution.paths[k][min(t, int(solution.paths[k].size()) - 1)];
            const auto [location, _, orientation] = state;

            if (t == 0) {
                if (paths[k].back() != State(location, timestep, orientation)) {
                    cerr << "Agent " << k << " solution initial step " << state << " inconsistent with existing state " << paths[k][timestep] << endl;
                    success = false;
                }
                continue;
            }
            if (paths[k].size() != new_timestep) {
                cerr << "Agent " << k << "'s path has " << paths[k].size() << " timesteps instead of " << new_timestep << " timesteps" << endl;
                success = false;
            }

            const State &prev_state = paths[k].back();
            const auto [prev_location, __, prev_orientation] = prev_state;
            int direction = G.get_direction(prev_location, location);
            if (!G.valid_move(prev_location, direction)) {
                cerr << "Agent " << k << " took invalid move " << direction << " to get to " << state << " from " << prev_state << endl;
                success = false;
            }
            if (config.rotation) {
                if (location == prev_location) {
                    if (G.get_rotate_degree(prev_orientation, orientation) == 2) {
                        cerr << "Agent " << k << " rotates 180 degrees from " << prev_state << " to " << state << endl;
                        success = false;
                    }
                } else {
                    if (prev_orientation != orientation) {
                        cerr << "Agent " << k << " rotates while moving from " << prev_state << " to " << state << endl;
                        success = false;
                    } else if (direction != orientation) {
                        cerr << "Agent " << k << "'s orientation " << orientation << " is inconsistent with its travel direction " << direction << " from " << prev_state << " to " << state << endl;
                        success = false;
                    }
                }
            }

            const int other = new_node_occupancy[location];
            if (other != EMPTY_AGENT) {
                cerr << "Agent " << k << " at " << state << " has conflict with agent " << other << " at " << paths[other][new_timestep] << endl;
                success = false;
            }
            new_node_occupancy[location] = k;

            if (prev_location != location) {
                if (int edge = G.locations_to_edge(location, prev_location); new_edge_occupancy[edge] != EMPTY_AGENT) {
                    int other = new_edge_occupancy[edge];
                    cerr << "Agent " << k << " moving from " << prev_state << " to " << state << " has conflict with agent " << other << " moving from " << paths[other][timestep] << " to " << paths[other][new_timestep] << endl;
                    success = false;
                }
                new_edge_occupancy[G.locations_to_edge(prev_location, location)] = k;
            }

            paths[k].emplace_back(location, new_timestep, orientation);
        }
        if (!success) {
            ostringstream ss; ss << "Found error(s) at timestep " << new_timestep << " (simulation window timestep " << t << "), stopping simulation" << endl;
            throw runtime_error(ss.str());
        }
        for (int k = 0; k < config.num_agents; k++) {
            if (goal_locations[k].empty()) continue;
            if (paths[k].back().location == goal_locations[k].front()) {
                goal_locations[k].erase(goal_locations[k].begin());
                finish_task(k, paths[k].back().location, new_timestep);
            }
        }
    }
    timestep += simulation_window;
    update_goal_locations();
}

void System::initialize() {
    goal_locations.resize(config.num_agents);
    paths.resize(config.num_agents);
    finished_tasks.resize(config.num_agents);

    if (config.verbosity >= 0)
        cout << "Randomly generating initial locations with initial seed " << config.seed << endl;
    initialize_start_locations();
    initialize_goal_locations();
    update_goal_locations();
}

void System::initialize_start_locations() {
    // Choose random unique start locations from any non-obstacle locations
    vector<bool> occupied(G.size, false);
    for (int k = 0; k < config.num_agents;) {
        int loc = (config.use_cpp_rng ? rng() : rand()) % G.size;
        if (G.types[loc] != "Obstacle" && !occupied[loc]) {
            paths[k].emplace_back(loc, 0, config.rotation ? (config.use_cpp_rng ? rng() : rand()) % 4 : -1);
            occupied[loc] = true;
            k++;
        }
    }
}

void System::initialize_goal_locations() {
    // Choose random unique goal locations from any non-obstacle locations
    vector<bool> occupied(G.size, false);
    for (int k = 0; k < config.num_agents;) {
        int loc = (config.use_cpp_rng ? rng() : rand()) % G.size;
        if (G.types[loc] != "Obstacle" && !occupied[loc]) {
            goal_locations[k].push_back(loc);
            occupied[loc] = true;
            k++;
        }
    }
}