cmake_minimum_required(VERSION 2.8)
project(GameOfLife)

# Switch build target
set(BUILD_TARGET PC CACHE STRING "Build target?")

# Common build settings
include(cmakes/common.cmake)

# Build settings for each target
if(${BUILD_TARGET} STREQUAL PC)
	if(WIN32)
		message("[BUILD] PC Windows")
		include(cmakes/PC_WIN.cmake)
	else()
		message("[BUILD] PC Linux")
		include(cmakes/PC_LINUX.cmake)
	endif()
elseif(${BUILD_TARGET} STREQUAL RASPI_NATIVE)
	message("[BUILD] Raspberry Pi Native")
	include(cmakes/RASPI_NATIVE.cmake)
elseif(${BUILD_TARGET} STREQUAL RASPI_CROSS)
	message("[BUILD] Raspberry Pi Cross")
	include(cmakes/RASPI_CROSS.cmake)
elseif(${BUILD_TARGET} STREQUAL JETSON)
	message("[BUILD] Jetson Nano Native")
	include(cmakes/JETSON_NATIVE.cmake)
else()
	message(FATAL_ERROR "[BUILD] Invalid target")
endif()

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	message("Cuda version: " ${CUDA_VERSION})
	option(ENABLE_CUDA "ENABLE_CUDA" ON)
	add_definitions(-DENABLE_CUDA)
endif()

# Include projects
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/output)
add_subdirectory(${PROJECT_SOURCE_DIR}/GameOfLife GameOfLife)

