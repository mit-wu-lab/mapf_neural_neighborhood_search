#pragma once

#include "common.h"
#include "AStar.h"

class Solver {
  public:
    Solver(Occupancy &occupancy, const MapGraph &G, const Config &config) : G(G), config(config), path_planner(make_unique<AStar>(occupancy, G, config)), rng(config.seed), occupancy(occupancy) {}

    virtual unique_ptr<Solution> make_empty_solution() { return make_unique<Solution>(); }

    virtual void solve(const Problem &problem, Solution &solution) {}

    double set_solution_occupancy(const Solution &solution, bool insert_or_remove) {
        auto start_time = get_time();
        for (int i = 0; i < solution.size(); i++) occupancy.set_path_constraint(solution.paths[i], insert_or_remove, i);
        return get_runtime(start_time);
    }
    double set_solution_occupancy(const Solution &solution, const vector<int> &agents, bool insert_or_remove) {
        auto start_time = get_time();
        for (int agent : agents) occupancy.set_path_constraint(solution.paths[agent], insert_or_remove, agent);
        return get_runtime(start_time);
    }

  protected:
    const MapGraph &G;
    const Config &config;
    const unique_ptr<PathPlanner> path_planner;
    std::minstd_rand rng;
    Occupancy &occupancy;
    const Problem *problem;
    Solution *solution;
};
