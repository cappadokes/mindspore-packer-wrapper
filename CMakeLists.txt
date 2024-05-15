cmake_minimum_required(VERSION 3.27)
project(mindspore_packers)

set(CMAKE_CXX_STANDARD 17)
set(ENABLE_FAST_HASH_TABLE ON)

include_directories(robin-hood-hashing/src)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake) # Include Conan-generated file
conan_basic_setup(TARGETS) # Introduce Conan-generated targets

add_executable(mindspore_packers main.cpp
        somas_solver_core.cc
        somas_solver_core.h
        somas_solver_alg.cc
        somas_solver_alg.h
        somas_solver_pre.cc
        somas_solver_pre.h
        thread_pool.cc
        thread_pool.h
        somas_tensor.cc
        somas_tensor.h)

target_link_libraries(${PROJECT_NAME} CONAN_PKG::robin-hood-hashing)
