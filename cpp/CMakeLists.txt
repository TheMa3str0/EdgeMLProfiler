cmake_minimum_required(VERSION 3.14)
project(inference
        LANGUAGES CXX
        VERSION 1.0.0
        DESCRIPTION "Profiler"
)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3 ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 ${TORCH_CUDA_FLAGS}")

add_executable(profiler main.cpp CustomNetworks.h InferenceProfiler.cpp TrainingProfiler.cpp CustomDataset.h ConfigParser.h ConfigParser.cpp)

target_link_libraries(profiler torch)

target_compile_features(profiler PUBLIC cxx_std_14)
target_compile_options(profiler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_XX>) # Replace XX with your desired GPU architecture

set_property(TARGET profiler PROPERTY CUDA_STANDARD 14)

target_link_libraries(profiler ${TORCH_CUDA_LIBRARIES})

set_target_properties(profiler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
