#pragma once
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <sstream>
#include <list>
#include <map>
#include <math.h>
#include <memory>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stdexcept>
#include <stdlib.h>
#include <utility>

using std::cerr;
using std::cout;
using std::endl;
using std::list;
using std::max;
using std::min;
using std::ostream;
using std::ofstream;
using std::ifstream;
using std::istringstream;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::to_string;
using std::tuple;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::chrono::high_resolution_clock;
using std::invalid_argument;
using std::runtime_error;
using std::make_unique;
using std::unique_ptr;

inline high_resolution_clock::time_point get_time() { return high_resolution_clock::now(); }
inline double get_runtime(high_resolution_clock::time_point &start_time) { return std::chrono::duration<double>(get_time() - start_time).count(); }

struct State {
    int location;
    int timestep;
    int orientation;

    State(int location = -1, int timestep = -1, int orientation = -1) : location(location), timestep(timestep), orientation(orientation) {}

    State wait() const { return State(location, timestep + 1, orientation); }

    struct Hasher {
        std::size_t operator()(const State &n) const {
            size_t loc_hash = std::hash<int>()(n.location);
            size_t time_hash = std::hash<int>()(n.timestep);
            size_t ori_hash = std::hash<int>()(n.orientation);
            return time_hash ^ (loc_hash << 1) ^ (ori_hash << 2);
        }
    };

    bool operator==(const State &other) const {
        return timestep == other.timestep && location == other.location && orientation == other.orientation;
    }
    bool operator!=(const State &other) const { return !(*this == other); }

    static inline void map(const vector<State> &states, vector<int> *array, const int State::*p) {
        for (const State &state : states) array->push_back(state.*p);
    }

    static inline vector<int> *map(const vector<State> &states, const int State::*p) {
        vector<int> *array = new vector<int>();
        map(states, array, p);
        return array;
    }

    static inline vector<int> *map(const vector<vector<State>> &state_sequences, const int State::*p) {
        vector<int> *array = new vector<int>();
        for (const vector<State> &states : state_sequences) map(states, array, p);
        return array;
    }

    static inline vector<int> *map_padded(const vector<vector<State>> &state_sequences, const int State::*p, int max_length) {
        vector<int> *array = new vector<int>();
        for (const vector<State> &states : state_sequences) {
            int length = 0;
            int prev_val;
            for (const State &state : states) {
                array->push_back(prev_val = state.*p);
                length++;
                if (length == max_length) break;
            }
            while (length++ < max_length) array->push_back(prev_val);
        }
        return array;
    }
};

typedef vector<State> Path;

ostream &operator<<(ostream &out, const State &s);
ostream &operator<<(ostream &out, const Path &path);

// agent, loc1, loc2, t, positive
typedef tuple<int, int, int, int, bool> Constraint;
// node conflict: a1, a2, loc1, -1, timestep
// edge conflict: a1, a2, loc1, loc2, timestep
typedef tuple<int, int, int, int, int> Conflict;
// [t_min, t_max), have conflicts or not
typedef tuple<int, int, bool> Interval;
#define INTERVAL_MAX 10000

ostream &operator<<(ostream &out, const Constraint &constraint);
ostream &operator<<(ostream &out, const Conflict &conflict);
ostream &operator<<(ostream &out, const Interval &interval);

template <typename T>
class Heap {
  public:
    Heap(bool (*fn)(T, T)) : comparator(fn) {}

    inline void push(T x) {
        list.push_back(x);
        std::push_heap(list.begin(), list.end(), comparator);
    }

    inline void pop() {
        std::pop_heap(list.begin(), list.end(), comparator);
        list.pop_back();
    }

    vector<T> list;
  private:
    bool (*comparator)(T, T);
};

struct Config {
    // input map file
    string map_file;
    // number of agents
    int num_agents = 800;
    // -1: silent  0: progress   1: results   2: all
    int verbosity = -1;
    // whether to use C++'s <random> library
    bool use_cpp_rng = true;
    // random seed
    unsigned int seed = 0;
    // max solution time in seconds
    double max_solution_time = INFINITY;
    // max path planning time in seconds
    double max_path_plan_time = INFINITY;
    // whether to account for agent rotation times
    bool rotation = false;
    // whether to prioritize agents waiting at start location
    bool prioritize_start = true;
    // whether agent waits at goal until all agents reach goal
    bool wait_at_goal = true;

    Config() {}
    Config(const Config &config) { *this = config; }

    const Config with_num_agents(int num_agents) const {
        Config config = *this;
        config.num_agents = num_agents;
        return config;
    }

    void set_seed(unsigned int seed) {
        this->seed = seed;
        if (use_cpp_rng) return;
        if (seed == 0 || seed == 1) throw invalid_argument("Warning: avoid using 0 or 1 as seed when using srand");
        srand(seed);
    }
};

struct Problem {
    vector<State> starts;
    vector<vector<int>> goal_locations;

    Problem() {} // default constructor
    Problem(const vector<int> &starts, const vector<vector<int>> &goal_locations) { // for debug
        for (const int s : starts) this->starts.push_back(State(s, 0));
        this->goal_locations = goal_locations;
    }
    Problem(const Problem &problem, const vector<int> &subagents) { // subproblem constructor
        for (const int agent : subagents) {
            starts.push_back(problem.starts[agent]);
            goal_locations.push_back(problem.goal_locations[agent]);
        }
    }
    inline size_t size() const { return starts.size(); }
};

struct Solution {
    vector<Path> paths;
    vector<double> path_costs;
    double cost = INFINITY;
    double runtime = 0;
    uint64_t LL_num_expanded = 0, LL_num_generated = 0;

    Solution() {} // default constructor
    Solution(const Solution &solution) : paths(solution.paths), path_costs(solution.path_costs), cost(solution.cost) {}
    Solution(const Solution &solution, const vector<int> &subagents) { // subsolution constructor
        cost = 0;
        for (const int agent : subagents) {
            paths.push_back(solution.paths[agent]);
            path_costs.push_back(solution.path_costs[agent]);
            cost += path_costs.back();
        }
    }
    inline size_t size() const { return paths.size(); }

    double subcost(const vector<int> &subagents) {
        double cost = 0;
        for (const int agent : subagents) cost += path_costs[agent];
        return cost;
    }

    vector<double> sub_path_costs(const vector<int> &subagents) {
        vector<double> sub_path_costs;
        for (const int agent : subagents) sub_path_costs.push_back(path_costs[agent]);
        return sub_path_costs;
    }

    virtual unique_ptr<Solution> copy() const { return make_unique<Solution>(*this); }
};
