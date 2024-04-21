#include "UndirectedGraph.h"

#include "AStar.h"

void UndirectedGraph::load_map() {
    const string &filename = config.map_file;
    ifstream file(config.map_file.c_str());
    if (!file.is_open()) {
        ostringstream ss; ss << "Map file " << filename << " does not exist" << endl;
        throw invalid_argument(ss.str());
    }
    string line;
    getline(file, line);               // skip the words "type octile"
    getline(file, line);
    istringstream ss1(line);
    string token;
    getline(ss1, token, ' ');
    getline(ss1, token, ' ');
    rows = atoi(token.c_str());  // read number of rows
    getline(file, line);
    istringstream ss2(line);
    getline(ss2, token, ' ');
    getline(ss2, token, ' ');
    cols = atoi(token.c_str());  // read number of cols
    size = rows * cols;
    move = {1, -cols, -1, cols};

    getline(file, line);               // skip the word "map"

    if (config.verbosity >= 0)
        cout << "Loading map " << config.map_file << "  Num rows " << rows << "  Num cols " << cols << endl;

    // read types
    is_obstacle.resize(size);
    types.resize(size);
    for (int i = 0, k = 0; i < rows; i++) {
        getline(file, line);
        for (const auto chr : line) {
            is_obstacle[k] = chr != '.';
            types[k] = is_obstacle[k] ? "Obstacle" : "Travel";
            k++;
        }
    }
    file.close();

    weights.resize(size);
    for (int k = 0; k < size; k++) {
        if (is_obstacle[k]) {
            obstacles.push_back(k);
            weights[k] = { WEIGHT_MAX, WEIGHT_MAX, WEIGHT_MAX, WEIGHT_MAX, WEIGHT_MAX };
            continue;
        }
        int row = k / cols;
        int col = k % cols;
        for (const int m : move) {
            int neighbor = k + m;
            int neighbor_row = neighbor / cols;
            int neighbor_col = neighbor % cols;
            bool in_bound = (0 <= neighbor) && (neighbor < size) && (row == neighbor_row || col == neighbor_col);
            weights[k].push_back((!in_bound || is_obstacle[k + m]) ? WEIGHT_MAX : 1);
        }
        weights[k].push_back(1);
    }
    if (config.verbosity >= 0)
        cout << "Map size: " << rows << "x" << cols << endl;
}
