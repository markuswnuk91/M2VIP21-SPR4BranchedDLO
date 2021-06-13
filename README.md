# Modified SPR for branched DLO - C++

C++ version of Modified Strucure Preserved Registration (SPR) to register a pointcloud representing a branched deformable linear object (DLO) onto a given nodes structure (e.g. joint locations of a multibody or mass-spring-system representation).

The code follows the approach of [Tang et al. 2019](https://doi.org/10.1177/0278364919841431), their [MATLAB example code on Github](https://github.com/thomastangucb/SPR) and publications on [MLLE (dimensionality reductuion on manifolds)](http://papers.nips.cc/paper/3132-mlle-modified-locally-linear-embedding-using-multiple-weights.pdf) and [CPD (registration)](http://papers.nips.cc/paper/2962-non-rigid-point-set-registration-coherent-point-drift.pdf).

> **Note:** 
> - The code is currently constrained to 3D data.
> - The notation is tried to be kept consistently with Tang et al. 2019 throughout the repository, i.e. it may slightly differ from the other literature

## Install:

Dependencies:

- CMake, Make, Git
- Eigen 3 (e.g. Ubuntu `sudo apt install libeigen3-dev`)
- Pointcloud-Library (libpcl-dev). Only Modules `pcl-common` and `pcl-io` (for reading PCD file of measurements) are needed.
- Compiler which is gcc-8 or newer, due to thirdparty library argpase which needs to be build. To get the compiler run:
```sh
sudo apt install gcc-8 g++-8
export CC=/usr/bin/gcc-8
export CXX=/usr/bin/g++-8
# then cmake .. again
```


Get this repository with submodules:

```sh
# get it
git clone --recursive git@git.isw.uni-stuttgart.de:projekte/forschung/2017_DFG_IRTG_SoftTissueRobotics/spr.git
cd M2VIP21-SPR4BranchedDLO

# build it
mkdir build
cd build
cmake ..
make -j
```

## Use: 

### Use in your code

The needed API of the library is:

```cpp
Eigen::MatrixX3f jointLocations;
pcl::PointCloud<pcl::PointXYZ> measuredPointCloud;
 
SPR myspr(); // pass parameters here, if needed, see documentation
Eigen::MatrixX3f Xregistered = myspr.computeEM(jointLocations, measuredPointCloud);

SPRBranched myspr2(); // pass parameters here, if needed, see documentation
Eigen::MatrixX3f Xregistered = myspr2.computeEM(jointLocations, measuredPointCloud, branchArray);

```

### Demo C++

Demo programs for the C++ implementation are contained in the `test` folder.
To see a comparison of SPR and modified SPR run the demo.sh script.
It executes the test programs for SPR and modified SPR automatically, or you can jsut run them individually as expained below:
```sh
# navigate to the test folder
cd ../test
sh demo.sh
```
A MALTAB script is provided for visualization in `test/plotDemo.m`

#### SPR Implementation according to [Tang et al. 2019]

```sh
# use it (also has --verbose flag for more output)
cd ../test
../build/spr_test -o ../data/Xregistered.csv ../data/Xinit.csv ../data/test_cloud.pcd

# Get help on syntax
../build/spr_test
```

#### Modified SPR Implementation for branched DLO:
```sh
# see performance on fully visible dataset
cd ../test
../build/sprbranched_test -o ../data/Xregistered_branched.csv ../data/Xinit.csv ../data/test_cloud.pcd
```

#### See performance on occluded dataset:
```sh
# see performance on occluded branched DLO
cd ../test
../build/sprbranched_test -o ../data/Xregistered_branched.csv ../data/Xinit.csv ../data/test_cloud_occluded.pcd
```

MALTAB scripts are provided for visualization in `test/plotResult_SPR.m` and `test/plotResult_SPRBranched.m`


### Python Bindings
The library comes with python bindings for python3.
To use them the library spr_py.so msut be located in the `python` folder.
It sould be copied in the folder automatically when building.
If it is not, you can copy it from the build folder manually.

The `python/examles` folder contains several examples how the library can be used with python3
```sh
# use the library from python
cd ../python/examples
python3 spr_branched_example.py
```

### Debug

- Build it with CMake in Debug mode. 
- Vor VSCode, a `launch.json` already exists. It executes the demo application for debugging.

## Contents:

```
spr/
├── CMakeLists.txt  # Build recipe
├── .vscode/        # Configuration file for VScode (debugging, building, ...)
├── resources/      # external libraries as Git submodules
├── python/         # python bindings
├── src/            # the main SPR library
├── data/           # folder for files with init data and result for the test programs
└── test/           # Demo application
```
