#include "MapGraph.h"
#include "AStar.h"

int MapGraph::get_rotate_degree(int dir1, int dir2) const {
    if (dir1 == dir2) return 0;
    else if ((dir1 - dir2) % 2 == 1) return 1;
    else return 2;
}

int MapGraph::get_degree(int location) const {
    int degree = 0;
    for (int i = 0; i < 4; i++) { // move
        int to = location + move[i];
        degree += valid_move(location, i) || to >= 0 && to < size && valid_move(to, get_move_direction(to, location));
    }
    return degree;
}

list<int> MapGraph::get_neighbors(int location) const {
    list<int> neighbors;
    for (int i = 0; i < 4; i++)  // move
        if (valid_move(location, i))
            neighbors.push_back(location + move[i]);
    return neighbors;
}

list<State> MapGraph::get_neighbors(const State &s) const {
    const auto [location, timestep, orientation] = s;
    list<State> neighbors;
    neighbors.emplace_back(location, timestep + 1, orientation);  // wait
    if (orientation == -1) {
        for (int i = 0; i < 4; i++)                              // move
            if (valid_move(location, i))
                neighbors.emplace_back(location + move[i], timestep + 1, orientation);
    } else {
        if (valid_move(location, orientation))
            neighbors.emplace_back(location + move[orientation], timestep + 1, orientation);  // move
        neighbors.emplace_back(location, timestep + 1, (orientation + 1) % 4);  // turn left
        neighbors.emplace_back(location, timestep + 1, (orientation + 3) % 4);  // turn right
    }
    return neighbors;
}

list<pair<int, int>> MapGraph::get_reverse_neighbors(int location, int orientation, bool skip_wait) const {
    list<pair<int, int>> rneighbors;
    if (!skip_wait) rneighbors.emplace_back(location, orientation);  // wait
    if (orientation == -1) {
        for (int i = 0; i < 4; i++) { // move
            int prev_location = location - move[i];
            if (prev_location >= 0 && prev_location < size && valid_move(prev_location, i))
                rneighbors.emplace_back(prev_location, orientation);
        }
    } else {
        int prev_location = location - move[orientation];
        if (prev_location >= 0 && prev_location < size && valid_move(prev_location, orientation))
            rneighbors.emplace_back(prev_location, orientation);  // move
        rneighbors.emplace_back(location, (orientation + 3) % 4);  // turned right
        rneighbors.emplace_back(location, (orientation + 1) % 4);  // turned left
    }
    return rneighbors;
}

bool MapGraph::load_heuristics_table(ifstream &file) {
    string line;
    getline(file, line);  // skip "table_size"
    getline(file, line);
    istringstream ss(line);
    string token;
    getline(ss, token, ',');
    int N = atoi(token.c_str());  // read number of cols
    getline(ss, token, ',');
    int M = atoi(token.c_str());  // read number of cols
    if (M != size)
        return false;
    for (int i = 0; i < N; i++) {
        getline(file, line);
        int loc = atoi(line.c_str());
        getline(file, line);
        ss = istringstream(line);
        vector<double> h_table(size);
        for (int j = 0; j < size; j++) {
            getline(ss, token, ',');
            h_table[j] = atof(token.c_str());
            if (h_table[j] >= WEIGHT_MAX && types[j] != "Obstacle")
                types[j] = "Obstacle";
        }
        heuristics[loc] = h_table;
    }
    return true;
}

void MapGraph::save_heuristics_table(string filename) {
    ofstream file;
    file.open(filename);
    file << "table_size" << endl
         << heuristics.size() << "," << size << endl;
    for (auto &[k, v] : heuristics) {
        file << k << endl;
        for (double h : v)
            file << h << ",";
        file << endl;
    }
    file.close();
}

vector<double> MapGraph::compute_all_heuristics(int goal_location) {
    Heap<AStarNode *> heap(AStarNode::compare_f_g);
    unordered_set<AStarNode *, AStarNode::Hasher, AStarNode::EqNode> best_nodes;
    vector<AStarNode *> nodes;
    auto push_node = [&](AStarNode *node) {
        heap.push(node);
        best_nodes.insert(node);
        nodes.push_back(node);
        node->in_open_list = true;
    };

    vector<int> goal_directions = {-1};
    if (config.rotation) goal_directions = {0, 1, 2, 3}; // any direction reaching the goal is fine
    for (int goal_direction : goal_directions)
        push_node(new AStarNode({goal_location, -1, goal_direction}, 0, 0, nullptr));

    while (!heap.list.empty()) {
        AStarNode *curr = heap.list.front();
        heap.pop();
        if (!curr->in_open_list) continue;
        curr->in_open_list = false;
        const auto [location, _, orientation] = curr->state;
        for (const auto [prev_location, prev_orientation] : get_reverse_neighbors(location, orientation)) {
            double prev_g_val = curr->g_val + get_weight(prev_location, location);
            AStarNode *prev = new AStarNode({prev_location, prev_orientation}, prev_g_val, 0, nullptr);
            const auto &it = best_nodes.find(prev);
            if (it != best_nodes.end()) {  // add the newly generated node to heap and hash table
                AStarNode *existing_prev = *it;
                if (existing_prev->g_val > prev->g_val) {
                    best_nodes.erase(it);
                    existing_prev->in_open_list = false;
                } else {
                    delete prev;
                    continue;
                }
            }
            push_node(prev);
        }
    }
    // iterate over all nodes and populate the distances
    vector<double> distances(size, DBL_MAX);
    for (AStarNode *node : best_nodes)
        distances[node->state.location] = min(node->g_val, distances[node->state.location]);
    for (AStarNode *node : nodes) delete node;
    return distances;
}

double MapGraph::compute_heuristics(int location, const vector<int> &goal_locations) const {
    double h = 0;
    for (int goal_location : goal_locations) {
        h += heuristics.at(goal_location)[location];
        location = goal_location;
    }
    return h;
}

double MapGraph::compute_path_cost(const Path &path) const {
    int prev_location = -1;
    double cost = 0;
    for (const auto [location, timestep, orientation] : path) {
        if (prev_location != -1) cost += get_weight(prev_location, location);
        prev_location = location;
    }
    return cost;
}
