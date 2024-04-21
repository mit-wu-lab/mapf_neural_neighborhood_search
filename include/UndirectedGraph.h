#pragma once
#include "MapGraph.h"

class UndirectedGraph : public MapGraph {
  public:
    UndirectedGraph(const Config &config) : MapGraph(config) { load_map(); };
    void preprocess_heuristics(vector<int> &locations) {
        for (const int loc : locations) heuristics[loc] = compute_all_heuristics(loc);
    }

  private:
    vector<bool> is_obstacle;
    void load_map();
};
