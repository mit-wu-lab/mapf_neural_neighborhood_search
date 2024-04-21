#pragma once
#include "common.h"
#include "MapGraph.h"
#include "Occupancy.h"

struct PathSolution {
    Path path;
    double cost = INFINITY;
    bool has_conflicts = false;
    uint64_t num_expanded = 0, num_generated = 0;
    double runtime = 0;
};

class PathPlanner {
  public:
    PathPlanner(Occupancy &occupancy, const MapGraph &G, const Config &config) : occupancy(occupancy), G(G), config(config) {}
    virtual PathSolution solve(const State &start, const vector<int> &goal_locations, bool strict = true) = 0;
    const Occupancy &occupancy;

  protected:
    const MapGraph &G;
    const Config &config;
};
