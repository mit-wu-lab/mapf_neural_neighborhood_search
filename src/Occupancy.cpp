#include "Occupancy.h"

void Occupancy::set_constraints_for_starts(const vector<Path *> &paths, int current_agent, bool insert_or_remove) {
    for (int i = 0; i < paths.size(); i++) {
        if (paths[i] == nullptr) continue;
        else if (i != current_agent) // prohibit the agent from conflicting with other agents at their start locations
            set_constraint_for_start(*paths[i], insert_or_remove, i);
    }
}


set<int> IntervalListOccupancy::get_constrained_timesteps(int location) const {
    set<int> rst;
    for (const auto [t1, t2] : nodes[location]) {
        if (t2 == INTERVAL_MAX) continue; // skip goal constraint
        for (auto t = t1; t < t2; t++) rst.insert(t);
    }
    return rst;
}

void IntervalListOccupancy::set_path_constraint(const Path &path, bool insert_or_remove, int agent) {
    if (path.empty()) return;
    auto insert_or_remove_node = insert_or_remove ? node_insert_fn : node_remove_fn;
    auto insert_or_remove_edge = insert_or_remove ? edge_insert_fn : edge_remove_fn;
    auto prev = path.begin();
    auto curr = prev;
    while (true) {
        if (++curr == path.end()) {
            // prev->location must equal path.back().location. No need to set edge constraint
            insert_or_remove_node(nodes[prev->location], prev->timestep, path.back().timestep + 1);
            return;
        }
        if (prev->location == curr->location) continue;
        insert_or_remove_node(nodes[prev->location], prev->timestep, curr->timestep); // Set node constraint
        // Set edge constraint
        insert_or_remove_edge(edges[G.locations_to_edge(curr->location, prev->location)], curr->timestep);
        prev = curr;
    }
}

void IntervalListOccupancy::set_constraint_for_start(const Path &path, bool insert_or_remove, int agent) {
    auto insert_or_remove_node = insert_or_remove ? node_insert_fn : node_remove_fn;
    int start_location = path.front().location;
    for (const auto [location, timestep, _] : path) {
        if (location != start_location) { // starts to move
            // the agent waits at its start locations between [appear_time, timestep - 1]
            // so other agents cannot use this location at these times
            insert_or_remove_node(nodes[start_location], 0, timestep);
            break;
        }
    }
}

bool IntervalListOccupancy::is_constrained(int curr_location, int next_location, int next_timestep) const {
    for (const auto [t1, t2] : nodes[next_location])
        if (next_timestep >= t1 && next_timestep < t2)
            return true;

    if (curr_location != next_location)
        for (const int t : edges[G.locations_to_edge(curr_location, next_location)])
            if (next_timestep == t)
                return true;
    return false;
}


void ArrayOccupancy::set_constraint_for_start(const Path &path, bool insert_or_remove, int agent) {
    int start_location = path.front().location;
    for (const auto [location, timestep, _] : path) {
        if (location != start_location) break; // starts to move
        // the agent waits at its start locations between [appear_time, state.timestep - 1]
        // so other agents cannot use this location at these times
        set_ct(location, timestep, agent, insert_or_remove);
    }
}

void ArrayOccupancy::set_ct(int location, int timestep, int agent, bool insert_or_remove) {
    vector<int> &loc_occ = ct[location];
    if (loc_occ.size() <= timestep) {
        int occ_agent = EMPTY_AGENT;
        if (config.wait_at_goal) { // another agent may be waiting at its goal already
            const auto it = wait_at_goals.find(location);
            if (it != wait_at_goals.end()) occ_agent = it->second.second;
        }
        loc_occ.resize(timestep + 1, occ_agent);
    }
    int &curr_agent = loc_occ[timestep];
    if (insert_or_remove) { // insert
        if (curr_agent == EMPTY_AGENT) curr_agent = agent;
        else {
            ostringstream ss; ss << "Trying to set agent " << agent << " at location " << location << " with current agent " << curr_agent << endl;
            throw invalid_argument(ss.str());
        }
    } else { // remove
        if (curr_agent == agent) curr_agent = EMPTY_AGENT;
        else {
            ostringstream ss; ss << "Trying to unset agent " << agent << " at location " << location << " with current agent " << curr_agent << endl;
            throw invalid_argument(ss.str());
        }
    }
}