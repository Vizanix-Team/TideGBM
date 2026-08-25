cmake_minimum_required(VERSION 3.16)
project(tidegbm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP)

add_library(tidegbm SHARED
    src/histogram.cpp
    src/binning.cpp
    src/tree.cpp
    src/objective.cpp
    src/booster.cpp
    src/serialize.cpp
    src/c_api.cpp
)
target_include_directories(tidegbm PUBLIC include)
if (OpenMP_CXX_FOUND)
    target_link_libraries(tidegbm PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_options(tidegbm PRIVATE -O3)
# Optional: cmake -DTIDEGBM_CUDA=ON and add src/histogram_cuda.cu with nvcc.

