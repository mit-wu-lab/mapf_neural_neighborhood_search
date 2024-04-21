#include "PriorityGraph.h"

void PriorityGraph::copy(const PriorityGraph &other, const vector<bool> &excluded_nodes) {
    for (const auto &[k, v] : other.PG) {
        if (excluded_nodes[k]) continue;
        for (int i : v)
            if (!excluded_nodes[i]) PG[k].insert(i);
    }
}

bool PriorityGraph::connected(int from, int to) const {
    list<int> open_list = { from };
    unordered_set<int> closed_list = { from };
    while (!open_list.empty()) {
        int curr = open_list.back();
        open_list.pop_back();
        auto neighbors = PG.find(curr);
        if (neighbors == PG.end()) continue;
        for (const int next : neighbors->second) {
            if (next == to) return true;
            if (closed_list.find(next) == closed_list.end()) {
                open_list.push_back(next);
                closed_list.insert(next);
            }
        }
    }
    return false;
}

unordered_set<int> PriorityGraph::get_reachable_nodes(int root) {
    auto start_time = get_time();
    list<int> open_list = { root };
    unordered_set<int> closed_list;
    while (!open_list.empty()) {
        int curr = open_list.back();
        open_list.pop_back();
        auto neighbors = PG.find(curr);
        if (neighbors == PG.end()) continue;
        for (const int next : neighbors->second) {
            if (closed_list.find(next) == closed_list.end()) {
                open_list.push_back(next);
                closed_list.insert(next);
            }
        }
    }
    runtime = get_runtime(start_time);
    return closed_list;
}

