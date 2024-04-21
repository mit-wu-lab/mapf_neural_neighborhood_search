#pragma once
#include "common.h"
#include "MapGraph.h"

class System {
  public:
    int timestep = 0;
    int num_finished_tasks = 0;

    static System make(const MapGraph &G, const Config &config) {
        System system(G, config);
        system.initialize();
        return system;
    }

    const MapGraph &get_graph() { return G; }
    Problem get_problem();
    void set_problem(Problem &problem);
    
    void advance(const Solution &solution);

  protected:
    System(const MapGraph &G, const Config &config) : G(G), config(config), rng(config.seed) {}

    const MapGraph &G;
    const Config &config;
    std::mt19937 rng;
    vector<Path> paths; // history of drive states
    vector<vector<int>> goal_locations;
    vector<vector<pair<int, int>>> finished_tasks;  // [(location, finish time), ...] for each agent

    virtual void initialize();
    virtual void initialize_start_locations(); // called during initialize
    virtual void initialize_goal_locations(); // called during initialize
    virtual void update_goal_locations() {} // called every simulation_window

    virtual void finish_task(int agent, int location, int timestep) {
        finished_tasks[agent].emplace_back(location, timestep);
        num_finished_tasks++;
    }
};
