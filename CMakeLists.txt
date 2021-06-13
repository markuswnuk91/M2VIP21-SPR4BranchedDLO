CMAKE_MINIMUM_REQUIRED(VERSION 3.8)

project(spr
        VERSION 0.1
        LANGUAGES CXX)

message("----------------------------------------------------------")
message(" lib${PROJECT_NAME} ${${PROJECT_NAME}_VERSION}               ")
message("Build with: make mlle")
message("----------------------------------------------------------")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0069 NEW)

find_package(PCL 1.8 QUIET REQUIRED COMPONENTS common)

if(PCL_FOUND)
    message("PCL is found in ${PCL_INCLUDE_DIRS}")
    include_directories( ${PCL_INCLUDE_DIRS} )
endif()

find_package(Eigen3 REQUIRED)
message("Eigen3 is found in ${EIGEN3_INCLUDE_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W0)
else()
  # add_compile_options(-Wall -Wextra)
  add_compile_options(-Wall -O3 -fPIC)
endif()

add_library(${PROJECT_NAME} src/spr.cpp src/mlle.cpp src/gmm.cpp src/spr_branched.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} libpcl_io.so)


# executables for testing:
add_executable(spr_test test/spr_test.cpp)
target_include_directories(spr_test PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/resources) # for inih
target_link_libraries(spr_test ${PROJECT_NAME} argparse)

add_executable(sprbranched_test test/sprbranched_test.cpp)
target_include_directories(sprbranched_test PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/resources) # for inih
target_link_libraries(sprbranched_test ${PROJECT_NAME} argparse)

# python bindings:
add_subdirectory(resources/argparse)
add_subdirectory(python)