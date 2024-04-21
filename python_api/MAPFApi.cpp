#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "UndirectedGraph.h"
#include "System.h"
#include "PBS.h"
#include "PP.h"
#include "LNS.h"

//pibt related
#include "simplegrid.h"
#include "pibt_agent.h"
#include "problem.h"
#include "mapf.h"
#include "pibt.h"
#include "pps.h"
#include "winpibt.h"

namespace py = pybind11;

inline auto np_array = []<typename T>(vector<T> *array) {
    py::capsule free_when_done(array, [](void *arr) { delete reinterpret_cast<vector<T> *>(arr); });
    return py::array_t<T>({array->size()}, {sizeof(T)}, array->data(), free_when_done);
};

inline auto as_array = []<typename T>(vector<T> *array) {
    return py::array_t<T>({array->size()}, {sizeof(T)}, array->data());
};

inline auto get_sizes = []<typename T>(const vector<vector<T>> &values) {
    vector<int> *sizes = new vector<int>();
    for (const vector<T> vals : values) sizes->push_back(vals.size());
    return sizes;
};

void populate_occupancy(const MapGraph &graph, const Solution &solution, const Solution &shortest_solution, const vector<py::array_t<int>> &all_subagents, py::array_t<float> &occupancy, bool use_shortest, bool use_init, const py::array_t<bool> &augment, const int prepool, const int dT) {
    auto r = occupancy.mutable_unchecked<4>();
    vector<bool> in_subagents(solution.size(), false);
    if (!(use_shortest || use_init)) throw invalid_argument("One of {use_shortest, use_init} must be true");
    const int shortest_channel = 0;
    const int init_channel = shortest_channel + use_shortest;
    const int nonstatic_channel = init_channel + use_init;
    const int rows = (graph.rows - 1) / prepool + 1;
    const int cols = (graph.cols - 1) / prepool + 1;
    auto populate_path = [&](const Path &path, const int batch_idx, const int channel, bool flip_vert, bool flip_hor) {
        for (const auto &[location, timestep, _] : path) {
            if (timestep % dT != 0) continue;
            if (timestep >= r.shape(3)) break;
            int row = location / graph.cols / prepool;
            int col = location % graph.cols / prepool;
            if (flip_vert) row = rows - row - 1;
            if (flip_hor) col = cols - col - 1;
            r(batch_idx, channel, row * cols + col, timestep / dT) += 1;
        }
    };
    auto a = augment.unchecked<2>();
    for (int j = 0; j < all_subagents.size(); j++) {
        const py::array_t<int> &subagents = all_subagents[j];
        auto s = subagents.unchecked<1>();
        for (int i = 0; i < s.shape(0); i++) in_subagents[s(i)] = true;

        for (int i = 0; i < solution.size(); i++) {
            if (!in_subagents[i]) populate_path(solution.paths[i], j, nonstatic_channel, a(j, 0), a(j, 1));
            else if (use_init) populate_path(solution.paths[i], j, init_channel, a(j, 0), a(j, 1));

            if (in_subagents[i]) populate_path(shortest_solution.paths[i], j, shortest_channel, a(j, 0), a(j, 1));
        }

        for (int i = 0; i < s.shape(0); i++) in_subagents[s(i)] = false;
    }
}

PYBIND11_MODULE(mapf, m) {
    m.doc() = "MAPF module";
    m.def("populate_occupancy", &populate_occupancy);
    m.def("run_PPS", [](const Problem &problem, Solution &solution, const Config &config) {
        std::mt19937 rng(config.seed);

        vector<int> agent_order(config.num_agents);
        std::iota(agent_order.begin(), agent_order.end(), 0);
        std::shuffle(agent_order.begin(), agent_order.end(), rng);

        SimpleGrid G(config.map_file, &rng);
        std::vector<Task*> T;
        PIBT_Agents A;
        Task::cntId = PIBT_Agent::cntId = Node::cntIndex = PPS::s_uuid = 0; // Need to reset static variables
        for (int a : agent_order) {
            // PIBT_Agent and Task will be deallocated in ~MAPF, which inherits from ~PProblem
            A.push_back(new PIBT_Agent(G.getNode(problem.starts[a].location)));
            T.push_back(new Task(G.getNode(problem.goal_locations[a].back())));
        }
        MAPF P(&G, A, T, &rng);

        PPS solver(&P, &rng);
        solver.setTimeLimit(config.max_solution_time);
        bool success = solver.solve();
        if (success) {
            solution.paths.resize(A.size());
            solution.path_costs.resize(A.size());
            solution.cost = 0;
            for (int i = 0; i < solution.size(); i++) {
                int a = agent_order[i];
                Path &path = solution.paths[a];
                const auto hist = A[i]->getHist();
                int t = hist.size() - 1;
                while (t > 0 && hist[t]->v->getId() == hist[t - 1]->v->getId()) t--;
                path.resize(t + 1);
                while (t >= 0) {
                    path[t] = { hist[t]->v->getId(), t, -1 };
                    t--;
                }
                solution.path_costs[a] = path.size() - 1;
                solution.cost += solution.path_costs[a];
            }
        } else solution.cost = INFINITY;
        solution.runtime = solver.getElapsed();
        return success;
    });

    py::class_<Config>(m, "Config")
        .def(py::init<>()) // Constructor
        .def(py::init<const Config &>()) // Copy constructor
        .def_readwrite("map_file", &Config::map_file)
        .def_readwrite("num_agents", &Config::num_agents)
        .def_readonly("seed", &Config::seed)
        .def("set_seed", &Config::set_seed)
        .def_readwrite("verbosity", &Config::verbosity)
        .def_readwrite("wait_at_goal", &Config::wait_at_goal)
        .def_readwrite("max_path_plan_time", &Config::max_path_plan_time)
        .def_readwrite("max_solution_time", &Config::max_solution_time)
        .def_readwrite("use_cpp_rng", &Config::use_cpp_rng);

    py::class_<Problem>(m, "Problem")
        .def(py::init<const Problem &, const vector<int> &>())
        .def(py::init<const vector<int> &, const vector<vector<int>> &>())
        .def("get_start_locations", [](const Problem &problem) { return np_array(State::map(problem.starts, &State::location)); })
        .def("get_start_orientations", [](const Problem &problem) { return np_array(State::map(problem.starts, &State::orientation)); })
        .def("get_goal_locations", [](const Problem &problem) {
            vector<int> *goals = new vector<int>();
            for (const vector<int> &gs : problem.goal_locations) goals->insert(goals->end(), gs.begin(), gs.end());
            return np_array(goals);
        })
        .def("get_num_goals", [](const Problem &problem) { return np_array(get_sizes(problem.goal_locations)); });

    py::class_<Solution>(m, "Solution")
        .def(py::init<>())
        .def(py::init<const Solution &>())
        .def(py::init<const Solution &, const vector<int> &>())
        .def_readonly("path_costs", &Solution::path_costs)
        .def_readonly("cost", &Solution::cost)
        .def_readwrite("runtime", &Solution::runtime)
        .def_readwrite("LL_num_expanded", &Solution::LL_num_expanded)
        .def_readwrite("LL_num_generated", &Solution::LL_num_generated)
        .def("get_path_locations", [](const Solution &solution) { return np_array(State::map(solution.paths, &State::location)); })
        .def("get_path_locations_padded", [](const Solution &solution, int max_length) { return np_array(State::map_padded(solution.paths, &State::location, max_length)).reshape({-1, max_length}); })
        .def("get_path_locations", [](const Solution &solution, int agent) { return np_array(State::map(solution.paths[agent], &State::location)); })
        .def("get_path_timesteps", [](const Solution &solution) { return np_array(State::map(solution.paths, &State::timestep)); })
        .def("get_path_orientations", [](const Solution &solution) { return np_array(State::map(solution.paths, &State::orientation)); })
        .def("get_path_lengths", [](const Solution &solution) { return np_array(get_sizes(solution.paths)); })
        .def("set_path_locations", [](Solution &solution, vector<vector<int>> paths) { // for debug
            solution.paths.clear(); solution.path_costs.clear();
            solution.paths.resize(paths.size()); solution.path_costs.resize(paths.size());
            solution.cost = 0;
            for (int i = 0; i < paths.size(); i++) {
                solution.path_costs[i] = paths[i].size() - 1;
                solution.cost += solution.path_costs[i];
                for (int timestep = 0; timestep < paths[i].size(); timestep++)
                    solution.paths[i].emplace_back(paths[i][timestep], timestep, -1);
            }
        })
        .def("set_path_locations", [](Solution &solution, const py::array_t<int> &locations, const py::array_t<int> &lengths) {
            auto locs = locations.unchecked<1>();
            auto lens = lengths.unchecked<1>();
            solution.paths.clear(); solution.path_costs.clear();
            solution.paths.resize(lens.shape(0)); solution.path_costs.resize(lens.shape(0));
            solution.cost = 0;
            int offset = 0;
            for (int i = 0; i < lens.shape(0); i++) {
                int path_cost = lens(i) - 1;
                solution.path_costs[i] = path_cost;
                solution.cost += path_cost;
                for (int timestep = 0; timestep <= path_cost; timestep++)
                    solution.paths[i].emplace_back(locs(offset++), timestep, -1);
            }
        })
        .def("subcost", &Solution::subcost)
        .def("sub_path_costs", &Solution::sub_path_costs);
    py::class_<PBSSolution, Solution>(m, "PBSSolution")
        .def(py::init<>())
        .def_readonly("runtime_occupancy", &PBSSolution::runtime_occupancy)
        .def_readonly("runtime_plan_paths", &PBSSolution::runtime_plan_paths)
        .def_readonly("runtime_copy_priorities", &PBSSolution::runtime_copy_priorities)
        .def_readonly("runtime_find_conflicts", &PBSSolution::runtime_find_conflicts)
        .def_readonly("runtime_find_consistent_paths", &PBSSolution::runtime_find_consistent_paths)
        .def_readonly("runtime_find_replan_agents", &PBSSolution::runtime_find_replan_agents)
        .def_readonly("HL_num_expanded", &PBSSolution::HL_num_expanded)
        .def_readonly("HL_num_generated", &PBSSolution::HL_num_generated)
        .def_readonly("num_LL", &PBSSolution::num_LL);

    py::class_<MapGraph>(m, "MapGraph")
        .def_readonly("size", &MapGraph::size)
        .def_readonly("rows", &MapGraph::rows)
        .def_readonly("cols", &MapGraph::cols)
        .def("get_obstacles", [](const MapGraph &graph) { return np_array(new vector<int>(graph.obstacles)); })
        .def("get_degree", &MapGraph::get_degree)
        .def("get_degree", [](const MapGraph &graph, vector<int> locations) {
            vector<int> *degrees = new vector<int>();
            for (int location : locations) degrees->push_back(graph.get_degree(location));
            return np_array(degrees);
        })
        .def("compute_heuristics", &MapGraph::compute_heuristics);
    py::class_<UndirectedGraph, MapGraph>(m, "UndirectedGraph")
        .def(py::init<const Config &>())
        .def("preprocess_heuristics", &UndirectedGraph::preprocess_heuristics)
        .def("import_heuristics", [](UndirectedGraph &graph, const Problem &problem, const py::array_t<double> &heuristics) {
            auto heurs = heuristics.unchecked<2>();
            int i = 0;
            for (const vector<int> &locations : problem.goal_locations)
                for (const int location : locations) {
                    vector<double> &loc_heuristics = graph.heuristics[location];
                    loc_heuristics.clear();
                    for (int j = 0; j < graph.size; j++)
                        loc_heuristics.push_back(heurs(i, j));
                    i++;
                }
        })
        .def("export_heuristics", [](const UndirectedGraph &graph, const Problem &problem) {
            vector<double> *heuristics = new vector<double>();
            for (const vector<int> &locations : problem.goal_locations)
                for (const int location : locations)
                    for (const double h : graph.heuristics.at(location))
                        heuristics->push_back(h);
            return np_array(heuristics).reshape({-1, graph.size});
        });

    py::class_<System>(m, "System")
        .def(py::init(&System::make))
        .def_readonly("timestep", &System::timestep)
        .def_readonly("num_finished_tasks", &System::num_finished_tasks)
        .def("get_graph", &System::get_graph)
        .def("get_problem", &System::get_problem)
        .def("set_problem", &System::set_problem)
        .def("advance", &System::advance);

    py::class_<Occupancy>(m, "Occupancy")
        .def("clear", &Occupancy::clear)
        .def("is_constrained", &Occupancy::is_constrained);
    py::class_<ArrayOccupancy, Occupancy>(m, "ArrayOccupancy")
        .def(py::init<const MapGraph &>());

    py::class_<PathPlanner>(m, "PathPlanner");
    py::class_<AStar, PathPlanner>(m, "AStar")
        .def(py::init<Occupancy &, const MapGraph &, const Config &>());

    py::class_<Solver>(m, "Solver")
        .def(py::init<Occupancy &, const MapGraph &, const Config &>())
        .def("set_solution_occupancy", static_cast<double (Solver::*)(const Solution &, bool)>(&Solver::set_solution_occupancy))
        .def("set_solution_occupancy", static_cast<double (Solver::*)(const Solution &, const vector<int> &, bool)>(&Solver::set_solution_occupancy));
    py::class_<PBS, Solver>(m, "PBS")
        .def(py::init<Occupancy &, const MapGraph &, const Config &>())
        .def("solve", [](PBS &pbs, const Problem &problem, PBSSolution &solution) { pbs.solve(problem, solution); } );
    py::class_<PP, Solver>(m, "PP")
        .def(py::init<Occupancy &, const MapGraph &, const Config &>())
        .def("solve", [](PP &pp, const Problem &problem, Solution &solution) { pp.solve(problem, solution); } )
        .def("solve", [](PP &pp, const Problem &problem, Solution &solution, const vector<int> &agent_order) { pp.solve(problem, solution, agent_order); })
        .def("repeat_solve", [](PP &pp, const Problem &problem, Solution &solution) { return np_array(pp.repeat_solve(problem, solution)); });

    py::class_<LNSConfig, Config>(m, "LNSConfig")
        .def(py::init<>()) // Constructor
        .def_readwrite("iterations", &LNSConfig::iterations)
        .def_readwrite("time_limit", &LNSConfig::time_limit)
        .def_readwrite("occupancy", &LNSConfig::occupancy)
        .def_readwrite("subset_fn", &LNSConfig::subset_fn)
        .def_readwrite("num_subsets", &LNSConfig::num_subsets)
        .def_readwrite("num_subset_agents", &LNSConfig::num_subset_agents)
        .def_readwrite("subsolver", &LNSConfig::subsolver)
        .def_readwrite("acceptance_threshold", &LNSConfig::acceptance_threshold);
    py::class_<Subset>(m, "Subset")
        .def(py::init<const vector<int> &>()) // Constructor
        .def_readonly("valid", &Subset::valid)
        .def_readonly("agents", &Subset::agents)
        .def_readonly("runtime_generate_agents", &Subset::runtime_generate_agents)
        .def_readonly("initial_cost", &Subset::initial_cost)
        .def_readonly("shortest_cost", &Subset::shortest_cost)
        .def_readonly("problem", &Subset::problem)
        .def_property_readonly("solution", [](Subset &subset) { return subset.solution.get(); })
        .def("size", &Subset::size);
    py::class_<LNS, Solver>(m, "LNS")
        .def(py::init<Occupancy &, const MapGraph &, LNSConfig &>())
        .def_readonly("subconfig", &LNS::subconfig)
        .def_readonly("iteration", &LNS::iteration)
        .def_readonly("initial_cost", &LNS::initial_cost)
        .def_readonly("curr_subsets", &LNS::curr_subsets)
        .def_property_readonly("shortest", [](LNS &lns) { return lns.shortest.get(); })
        .def_property_readonly("solution", [](LNS &lns) { return lns.solution.get(); })
        .def("initialize", [](LNS &lns, const Problem &problem, const Solution *solution) { return lns.initialize(problem, solution); })
        .def("generate_subset", [](LNS &lns) { return lns.generate_subset(); })
        .def("update_subset", &LNS::update_subset)
        .def("subsolve", [](LNS &lns, Subset &subset) { return lns.subsolve(subset); })
        .def("merge", [](LNS &lns, Subset &subset) { return lns.merge(subset); });
}