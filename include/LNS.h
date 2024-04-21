#pragma once

#include "Solver.h"
#include "PP.h"
#include "PBS.h"

struct LNSConfig : public Config {
    int iterations = 0;
    string occupancy = "Array";
    string subset_fn = "";
    int num_subsets = 0;
    int num_subset_agents = 0;
    string subsolver = "PBS";
    double acceptance_threshold = 0.0;
    double time_limit = INFINITY;

    LNSConfig() {}
    LNSConfig(const LNSConfig &config) { *this = config; }
};

struct Subset {
    bool valid = true;
    vector<int> agents;
    double runtime_generate_agents = 0.0;
    Problem problem;
    double initial_cost = 0.0;
    double shortest_cost = 0.0;
    unique_ptr<Solution> solution;

    Subset() {}
    Subset(const vector<int> &agents) {
        this->agents = agents;
    }
    int size() { return agents.size(); }
};

class LNS : public Solver {
  private:
    Occupancy &occupancy;
    ArrayOccupancy *array_occupancy = nullptr;
    PP initial_solver;
    unique_ptr<Solver> subsolver;

    unordered_set<int> tabu_agents;
    unordered_set<int> intersections;

  public:
    LNSConfig &config;
    Config subconfig;

    unique_ptr<Solution> shortest;
    unique_ptr<Solution> solution;
    double initial_cost;

    vector<Subset> curr_subsets;
    int iteration = 0;

    LNS(Occupancy &occupancy, const MapGraph &G, LNSConfig &config) :
        Solver(occupancy, G, config),
        config(config), subconfig(config.with_num_agents(config.num_subset_agents)),
        occupancy(occupancy),
        initial_solver(occupancy, G, config) {
        if (config.occupancy == "Array") array_occupancy = static_cast<ArrayOccupancy *>(&occupancy);

        if (config.subsolver == "PBS") subsolver = make_unique<PBS>(occupancy, G, subconfig);
        else if (config.subsolver == "PP") subsolver = make_unique<PP>(occupancy, G, subconfig);
        else {
            ostringstream ss; ss << "Invalid subsolver " << config.subsolver << endl;
            throw invalid_argument(ss.str());
        }
    }

    void initialize(const Problem &problem, const Solution *init_solution) {
        this->problem = &problem;
        shortest = make_empty_solution();

        vector<int> agent_order(config.num_agents);
        std::iota(agent_order.begin(), agent_order.end(), 0);
        initial_solver.solve(problem, *shortest, agent_order, false, false);

        solution = init_solution->copy();
        initial_solver.set_solution_occupancy(*solution, true);
        initial_cost = solution->cost;
        iteration = 1;
    }
    Subset generate_subset() { Subset subset; generate_subset(subset); return subset; }
    void generate_subset(Subset &subset);
    void generate_agent_neighborhood(Subset &subset);
    void generate_intersection_neighborhood(Subset &subset);

    void update_subset(Subset &subset) {
        subset.runtime_generate_agents = 0;
        subset.initial_cost = solution->subcost(subset.agents);
        subset.solution = nullptr;
    }

    void subsolve(Subset &subset);
    bool merge(const Subset &subset);
};
