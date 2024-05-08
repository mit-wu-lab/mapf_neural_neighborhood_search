#include "Occupancy.h"

void Occupancy::set_constraints_for_starts(const vector<Path *> &paths, int current_agent, bool insert_or_remove) {
    for (int i = 0; i < paths.size(); i++) {
        if (paths[i] == nullptr) continue;
        else if (i != current_agent) // prohibit the agent from conflicting with other agents at their start locations
            set_constraint_for_start(*paths[i], insert_or_remove, i);
    }
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
        loc_occ.resize(timestep + 1, EMPTY_AGENT);
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