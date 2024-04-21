#pragma once
#include "PathPlanner.h"

class AStarNode {
  public:
    State state;
    double g_val = 0, h_val = 0;
    AStarNode *parent = nullptr;
    bool in_open_list = false;
    int goal_index = 0;  // the id of its current goal.

    static bool compare_f_g(AStarNode *n1, AStarNode *n2) {
        return n1->f_val() > n2->f_val() || n1->f_val() == n2->f_val() && n1->g_val < n2->g_val;
    }

    AStarNode(const State &state, double g_val, double h_val, AStarNode *parent) : state(state), g_val(g_val), h_val(h_val), parent(parent), in_open_list(false) {
        goal_index = parent == nullptr ? 0 : parent->goal_index;
    }

    inline double f_val() const { return g_val + h_val; }

    // we say that two nodes are equal iff
    // both agree on the state and the goal id
    struct EqNode {
        bool operator()(const AStarNode *n1, const AStarNode *n2) const {
            return (n1 == n2) ||
                   (n1 && n2 && n1->state == n2->state && n1->goal_index == n2->goal_index);
        }
    };

    struct Hasher {
        size_t operator()(const AStarNode *n) const { return State::Hasher()(n->state); }
    };
};

class AStar : public PathPlanner {
  public:
    AStar(Occupancy &occupancy, const MapGraph &G, const Config &config) : PathPlanner(occupancy, G, config), open(AStarNode::compare_f_g) {}

    PathSolution solve(const State &start, const vector<int> &goal_locations, bool strict = true);
    void get_solution_path(const AStarNode *node, Path &path);
  protected:
    Heap<AStarNode *> open;
    unordered_set<AStarNode *, AStarNode::Hasher, AStarNode::EqNode> best_nodes;
    vector<AStarNode *> all_nodes;

    void push_open(AStarNode *node) {
        open.push(node);
        node->in_open_list = true;
        best_nodes.insert(node);
        all_nodes.push_back(node);
    }

    void release_nodes() {
        open.list.clear();
        for (AStarNode *node : all_nodes) delete node;
        all_nodes.clear();
        best_nodes.clear();
    }
};
