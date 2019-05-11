cmake_minimum_required(VERSION 2.8)
project(GameOfLife)

if(WIN32)
	message("[BUILD] PC Windows")
else()
	message("[BUILD] Linux")
	add_definitions(-D_UNIX)	# for AntTweakBar
	set(CMAKE_C_FLAGS "-Wall")
	set(CMAKE_C_FLAGS_DEBUG "-g -O0")
	set(CMAKE_C_FLAGS_RELEASE "-O3 -s")
	set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14")
	set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
	set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release")
	endif()
endif()


find_package(CUDA)
if(CUDA_FOUND)
	message("Cuda version: " ${CUDA_VERSION})
	option(ENABLE_CUDA "ENABLE_CUDA" ON)
	add_definitions(-DENABLE_CUDA)
endif()

# Include projects
add_subdirectory(${PROJECT_SOURCE_DIR}/GameOfLife GameOfLife)

