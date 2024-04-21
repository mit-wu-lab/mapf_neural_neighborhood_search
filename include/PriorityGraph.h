#pragma once

#include "common.h"

class PriorityGraph {
  public:
    double runtime = 0;

    void clear() { PG.clear(); }
    bool empty() const { return PG.empty(); }
    void copy(const PriorityGraph &other) { PG = other.PG; }
    void copy(const PriorityGraph &other, const vector<bool> &excluded_nodes);
    void add(int from, int to) { PG[from].insert(to); }     // from is lower than to
    void remove(int from, int to) { const auto &it = PG.find(from); if (it != PG.end()) it->second.erase(to); }  // from is lower than to
    unordered_set<int> dfs(int root, int target = -1) const;
    bool connected(int from, int to) const;
    unordered_set<int> get_reachable_nodes(int root);
  private:
    unordered_map<int, unordered_set<int>> PG;
};
