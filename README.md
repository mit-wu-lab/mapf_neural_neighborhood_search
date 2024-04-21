# Neural Neighborhood Search for Multi-agent Path Finding
This repository contains the code, run instructions, and trained models for our ICLR 2024 paper: Neural Neighborhood Search for Multi-agent Path Finding.

```
@inproceedings{yan2024neural,
title={Neural Neighborhood Search for Multi-agent Path Finding},
author={Zhongxia Yan and Cathy Wu},
booktitle={International Conference on Learning Representations},
year={2024}
}
```

## Installation
Our C++ MAPF implement is relatively lightweight and does not require dependencies like Boost, Eigen, etc. The only dependency is PyBind11 which is cloned recursively already.
```
git clone --recursive https://github.com/mit-wu-lab/mapf_neural_neighborhood_search.git
cd mapf_neural_neighborhood_search

mkdir build
cd build

cmake ..
```
If needed, you may find the one or more of the following flags useful to add to the previous `cmake ..` command:
- `-DPYTHON_LIBRARY=<path_to_python>/lib/libpython3.so`
- `-DPYTHON_EXECUTABLE=<path_to_python>/bin/python`
- `-DCMAKE_C_COMPILER=/usr/bin/gcc`
- `-DCMAKE_CXX_COMPILER=/usr/bin/g++`

Run `make -j8`
