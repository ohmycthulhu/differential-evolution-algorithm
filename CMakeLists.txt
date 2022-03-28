cmake_minimum_required(VERSION 3.21)
project(Project CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Project main.cu)

set_target_properties(Project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
