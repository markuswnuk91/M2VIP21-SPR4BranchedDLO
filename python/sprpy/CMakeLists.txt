SET(CMAKE_CXX_FLAGS "-std=c++14 -Wall -O3 -fPIC")

message(" *** Current working path : "${CMAKE_CURRENT_SOURCE_DIR})

find_package(PythonLibs 3 REQUIRED)
MESSAGE( " *** PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS} )
MESSAGE( " *** PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES} )
MESSAGE( " *** PYTHON_LIBRARIES : " ${PYTHON_INCLUDE_DIRS} )

find_package(pybind11 CONFIG REQUIRED)
MESSAGE( " *** PYBIND11_INCLUDE_DIRS : " ${pybind11_INCLUDE_DIR} )

find_package(PCL REQUIRED COMPONENTS common)
MESSAGE( " *** PCL_INCLUDE_DIRS : " ${PCL_INCLUDE_DIRS} )
MESSAGE( " *** PCL_LIB_DIRS : " ${PCL_LIBRARY_DIRS} )
MESSAGE( " *** PCL_LIBRARIES : " ${PCL_LIBRARIES} )
add_definitions(${PCL_DEFINITIONS})


include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})

file(GLOB_RECURSE spr_py_cpps "*.cpp")
file(GLOB_RECURSE spr_py_hs "*.h")

MESSAGE( " *** Link information : ")
MESSAGE( " *** Python : " ${PYTHON_LIBRARIES} )
MESSAGE( " *** PCL_LIBRARIES : " ${PCL_LIBRARIES})

add_library(spr_py SHARED ${spr_py_cpps} ${spr_py_hs})

target_include_directories(spr_py PUBLIC  ${CMAKE_SOURCE_DIR}/resources) # for inih
target_link_libraries(spr_py ${PYTHON_LIBRARIES} 
${PROJECT_NAME} 
argparse 
libpcl_common.so
libpcl_io.so
)


SET_TARGET_PROPERTIES(spr_py PROPERTIES PREFIX "")

# copy spr_py.so into the folder python
add_custom_command(TARGET spr_py POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                       ${CMAKE_BINARY_DIR}/python/sprpy/spr_py.so ${CMAKE_BINARY_DIR}/../python)
