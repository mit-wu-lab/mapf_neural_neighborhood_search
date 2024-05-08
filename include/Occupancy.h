#pragma once
#include "MapGraph.h"

class Occupancy {
  public:
    Occupancy(const MapGraph &G) : G(G), config(G.config) {}

    virtual void clear() = 0;

    virtual void set_path_constraint(const Path &path, bool insert_or_remove, int agent) = 0;  // insert/remove the path in the constraint table
    virtual void set_constraint_for_start(const Path &path, bool insert_or_remove, int agent) = 0;
    virtual void set_constraints_for_starts(const vector<Path *> &paths, int current_agent, bool insert_or_remove);

    virtual bool is_constrained(int curr_location, int next_location, int next_timestep) const = 0;
    virtual bool is_constrained_after(int curr_location, int timestep) const = 0;
    virtual int get_last_constrained_timestep(int location) const = 0;
    virtual set<int> get_constrained_timesteps(int location) const = 0;

  protected:
    const MapGraph &G;
    const Config &config;
};

inline void edge_insert_fn(list<int> &list, int t) { list.push_back(t); }
inline void edge_remove_fn(list<int> &list, int t) { list.erase(std::find(list.begin(), list.end(), t)); }
inline void node_insert_fn(list<pair<int, int>> &list, int t1, int t2) { list.emplace_back(t1, t2); }
inline void node_remove_fn(list<pair<int, int>> &list, int t1, int t2) {
    list.erase(std::find(list.begin(), list.end(), pair<int, int>(t1, t2)));
}

const int EMPTY_AGENT = -1;

class ArrayOccupancy : public Occupancy {
  public:
    vector<vector<int>> ct;  // location/edge -> time -> agent
    std::vector<int> wait_at_goals;  // location -> time

    ArrayOccupancy(const MapGraph &G) : Occupancy(G) { ct.resize(G.size + G.edge_size()); wait_at_goals.resize(G.size, INT_MAX); }

    void clear() { wait_at_goals.clear(); for (auto &m : ct) m.clear();}

    void set_path_constraint(const Path &path, bool insert_or_remove, int agent) { // insert/remove the path in the constraint table
        if (path.empty()) return;
        int prev_location = path[0].location;
        for (const auto [location, timestep, _] : path) {
            set_ct(location, timestep, agent, insert_or_remove);
            if (prev_location != location) {
                set_ct(locations_to_edge(prev_location, location), timestep, agent, insert_or_remove);
                set_ct(locations_to_edge(location, prev_location), timestep, agent, insert_or_remove);
            }
            prev_location = location;
        }
        if (config.wait_at_goal) {
            wait_at_goals[path.back().location] = insert_or_remove ? (path.size() - 1) : INT_MAX;
        }
    }
    void set_constraint_for_start(const Path &path, bool insert_or_remove, int agent);

    bool is_constrained(int from, int to, int to_time) const {
        if (config.wait_at_goal && wait_at_goals[to] <= to_time) return true;
        if (ct[to].size() > to_time && ct[to][to_time] != EMPTY_AGENT) return true;  // node conflict
        if (from != to) {
            const vector<int> &edge_occ = ct[locations_to_edge(from, to)];
            return edge_occ.size() > to_time && edge_occ[to_time] != EMPTY_AGENT;
        }
        return false;
    }

    bool is_constrained_after(int curr_location, int timestep) const {
        for (int t = timestep; t < ct[curr_location].size(); t++) {
            if (is_constrained(curr_location, curr_location, t)) return true;
        }
        return false;
    }

    set<int> get_constrained_timesteps(int location) const {
        set<int> timesteps;
        for (int t = 0; t < ct[location].size(); t++) if (ct[location][t] != EMPTY_AGENT) timesteps.insert(t);
        return timesteps;
    }

    int get_last_constrained_timestep(int location) const {
        const vector<int> &ct_loc = ct[location];
        for (int t = ct_loc.size() - 1; t >= 0; t--) {
            if (ct_loc[t] != EMPTY_AGENT) return t;
        }
        return -1;
    }

    void get_constrained_agents(int next_location, int next_timestep, unordered_set<int> &agents) const {
        if (next_timestep >= ct[next_location].size()) return;
        int agent = ct[next_location][next_timestep];
        if (agent != EMPTY_AGENT) agents.insert(agent);
    }
    void get_edge_constrained_agents(int location, int next_location, int next_timestep, unordered_set<int> &agents) const {
        get_constrained_agents(locations_to_edge(location, next_location), next_timestep, agents);
    }
    void get_constrained_agents(int location, unordered_set<int> &agents) const {
        for (int timestep = 0; timestep < ct[location].size(); timestep++)
            get_constrained_agents(location, timestep, agents);
    }

  private:
    void set_ct(int location, int timestep, int agent, bool insert_or_remove);
    inline int locations_to_edge(int from, int to) const { return G.size + G.locations_to_edge(from, to); }
};