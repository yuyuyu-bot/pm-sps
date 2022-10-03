cmake_minimum_required(VERSION 3.1)

project(pm-sps)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if (OpenCV_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# libSGM
find_package(CUDA REQUIRED)
find_package(libSGM REQUIRED)
include_directories(${LIBSGM_INCLUDE_DIRS})

file(GLOB srcs ${CMAKE_SOURCE_DIR}/src/*.cpp main.cpp)

cuda_add_executable(${PROJECT_NAME} ${srcs})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${LIBSGM_LIBRARY})
