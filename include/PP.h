#pragma once

#include "Solver.h"

class PP : public Solver {
  public:
    PP(Occupancy &occupancy, const MapGraph &G, const Config &config) : Solver(occupancy, G, config) {}

    void solve(const Problem &problem, Solution &solution, const vector<int> &agent_order, bool avoid_collisions, bool restore_occupancy);
    void solve(const Problem &problem, Solution &solution) { solve(problem, solution, true); }
    void solve(const Problem &problem, Solution &solution, bool restore_occupancy) {
        vector<int> agent_order(config.num_agents);
        std::iota(agent_order.begin(), agent_order.end(), 0);
        solve(problem, solution, agent_order, true, restore_occupancy);
    }
    void solve(const Problem &problem, Solution &solution, const vector<int> &agent_order, bool restore_occupancy = true) {
        solve(problem, solution, agent_order, true, restore_occupancy);
    }
    vector<int> *repeat_solve(const Problem &problem, Solution &solution) {
        vector<int> *agent_order = new vector<int>(config.num_agents);
        std::iota(agent_order->begin(), agent_order->end(), 0);
        solution.cost = INFINITY;
        auto start_time = get_time();
        while (isinf(solution.cost) && get_runtime(start_time) < config.max_solution_time) {
            config.use_cpp_rng ? std::shuffle(agent_order->begin(), agent_order->end(), rng) : std::random_shuffle(agent_order->begin(), agent_order->end());
            solve(problem, solution, *agent_order, true, true);
        }
        solution.runtime = get_runtime(start_time);
        if (isinf(solution.cost)) agent_order->clear();
        return agent_order;
    }
};
