#pragma once

#include "Solver.h"
#include "PriorityGraph.h"

struct PBSSolution : public Solution {
    double runtime_occupancy = 0;
    double runtime_plan_paths = 0;
    double runtime_copy_priorities = 0;
    double runtime_find_conflicts = 0;
    double runtime_find_consistent_paths = 0;
    double runtime_find_replan_agents = 0;

    uint64_t HL_num_expanded = 0, HL_num_generated = 0;
    uint64_t num_LL = 0;

    unique_ptr<Solution> copy() { return make_unique<PBSSolution>(*this); }
};

struct PBSNode {
    list<tuple<int, Path, double>> paths;   // <agent_id, path, path_cost>
    list<Conflict> conflicts; // conflicts in the paths
    Conflict conflict; // the chosen conflict to resolve
    PBSNode *parent = nullptr;
    pair<int, int> priority;  // a1 < a2
    PriorityGraph priorities;
    double g_val = 0;
    int earliest_collision = INT_MAX;

    inline int num_collisions() { return conflicts.size(); }

    static inline bool compare_g_collision(PBSNode *n1, PBSNode *n2) {
        return n1->g_val > n2->g_val || n1->g_val == n2->g_val && n1->num_collisions() >= n2->num_collisions();
    }

    void clear() { conflicts.clear(); priorities.clear(); }
};

class PBS : public Solver {
  public:
    PBS(Occupancy &occupancy, const MapGraph &G, const Config &config) : Solver(occupancy, G, config) {}
    unique_ptr<Solution> make_empty_solution() { return make_unique<PBSSolution>(); }
    void solve(const Problem &problem, PBSSolution &solution);
    void solve(const Problem &problem, Solution &solution) { solve(problem, static_cast<PBSSolution &>(solution)); }

  private:
    PBSSolution *solution;
    bool has_other_constraints;
    vector<Path *> curr_paths;
    vector<double> curr_path_costs;

    vector<PBSNode *> all_nodes;
    list<PBSNode *> dfs;
    vector<pair<int, int>> nogood;
    PBSNode *best_node;

    void find_conflicts(list<Conflict> &conflicts, int a1, int a2);
    void find_conflicts(list<Conflict> &conflicts);
    void find_conflicts(list<Conflict> &new_conflicts, int new_agent);
    void remove_conflicts(list<Conflict> &conflicts, int excluded_agent);
    void choose_conflict(PBSNode *curr);

    // high level search
    void push_node(PBSNode *node);
    void update_paths(PBSNode *curr);
    PBSNode *generate_root_node();
    bool initialize_paths(PBSNode *root);
    bool find_path(PBSNode *node, int ag);
    bool find_consistent_paths(PBSNode *node, int a);  // find paths consistent with priorities
    bool generate_child_node(PBSNode *child, PBSNode *curr);

    // util
    void set_occupancy(const unordered_set<int> &high_priority_agents, int current_agent, bool insert_or_remove);
    bool wait_at_start(const Path &path, int start_location, int min_timestep);
    void find_replan_agents(PBSNode *node, const list<Conflict> &conflicts,
                            unordered_set<int> &replan);
};
