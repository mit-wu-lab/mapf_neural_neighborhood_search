#pragma once
#include "common.h"

#define WEIGHT_MAX INT_MAX / 2

class MapGraph {
  public:
    vector<string> types;
    vector<int> obstacles;
    unordered_map<int, vector<double>> heuristics;
    int rows = 0, cols = 0, size = 0;
    vector<int> move;
    vector<vector<double>> weights;  // (directed) weighted 4-neighbor grid
    const Config &config;

    MapGraph(const Config &config) : config(config) {}

    int get_degree(int location) const;
    list<State> get_neighbors(const State &state) const; // neighboring next states
    list<int> get_neighbors(int location) const; // neighboring next locations
    list<pair<int, int>> get_reverse_neighbors(int location, int orientation, bool skip_wait = true) const;  // neighboring prev states
    inline double get_weight(int from_location, int to_location) const { // locations from and to are neighbors
        return weights[from_location][get_direction(from_location, to_location)];
    }
    int get_rotate_degree(int dir1, int dir2) const;  // return 0 if it is 0; return 1 if it is +-90; return 2 if it is 180

    inline bool valid_move(int location, int direction) const { return weights[location][direction] < WEIGHT_MAX; }
    inline int get_move_direction(int from_location, int to_location) const {
        for (int i = 0; i < 4; i++) if (from_location + move[i] == to_location) return i;
        ostringstream ss; ss << "Invalid move from " << from_location << " to " << to_location << endl;
        throw invalid_argument(ss.str());
    }
    inline int get_direction(int from_location, int to_location) const {
        if (from_location == to_location) return 4;
        return get_move_direction(from_location, to_location);
    }

    vector<double> compute_all_heuristics(int goal_location);  // compute distances from all locations to the goal location
    bool load_heuristics_table(ifstream &file);
    void save_heuristics_table(string filename);
    double compute_heuristics(int location, const vector<int> &goal_locations) const;
    double compute_path_cost(const Path &path) const;
    int get_manhattan_distance(int loc1, int loc2) const { return abs(loc1 / cols - loc2 / cols) + abs(loc1 % cols - loc2 % cols); }

    inline int edge_size() const { return size * move.size(); }
    inline int locations_to_edge(int from, int to) const { return from * move.size() + get_move_direction(from, to); }
    inline pair<int, int> edge_to_locations(int index) const {
        int from = index / move.size();
        return {from, from + move[index % move.size()]};
    }

  protected:
    virtual void load_map() = 0;
};
