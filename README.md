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
Our C++ MAPF implementation is relatively lightweight and mostly only depends on PyBind11, which is cloned recursively already. Unfortunately Boost and Eigen are required for the PPS initialization for one case. You can remove everything PIBT and PPS related if you cannot use Boost and Eigen.
```
git clone --recursive https://github.com/mit-wu-lab/mapf_neural_neighborhood_search.git
cd mapf_neural_neighborhood_search

wget https://movingai.com/benchmarks/mapf/mapf-map.zip
unzip mapf-map.zip -d maps
rm mapf-map.zip

wget https://movingai.com/benchmarks/mapf/mapf-scen-random.zip
unzip mapf-scen-random.zip
rm mapf-scen-random.zip
mkdir scenarios
mv scen-random scenarios/random

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
