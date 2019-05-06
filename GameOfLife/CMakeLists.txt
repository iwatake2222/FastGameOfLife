cmake_minimum_required(VERSION 2.8.10)

# Create executable file
set(SOURCES
	AnalView.cpp
	AnalViewAge.cpp
	AnalViewGroup.cpp
	ControllerView.cpp
	FileAccessor.cpp
	LifeGameFast.cpp
	LogicBase.cpp
	LogicGroup.cpp
	LogicNormal.cpp
	LogicNormalMP.cpp
	LogicNormalNonTorus.cpp
	LogicNormalNonTorusMP.cpp
	WindowManager.cpp
	World.cpp
	WorldContext.cpp
	WorldView.cpp
)

set(HEADERS
	AnalView.h
	AnalViewAge.h
	AnalViewGroup.h
	ControllerView.h
	FileAccessor.h
	ILogic.h
	IView.h
	LogicBase.h
	LogicGroup.h
	LogicNormal.h
	LogicNormalMP.h
	LogicNormalNonTorus.h
	LogicNormalNonTorusMP.h
	Values.h
	WindowManager.h
	World.h
	WorldContext.h
	WorldView.h
)

if(ENABLE_CUDA)
	set(SOURCES 
		${SOURCES}
		LogicGroupCuda.cpp
		LogicNormalCuda.cpp
	)
	set(HEADERS 
		${HEADERS}
		LogicGroupCuda.h
		LogicNormalCuda.h
	)
endif()

add_executable(GameOfLife
	${SOURCES}
	${HEADERS}
)

# Add sub project
if(ENABLE_CUDA)
	add_subdirectory(AlgorithmCudaNormal)
	target_include_directories(GameOfLife PUBLIC AlgorithmCudaNormal)
	target_link_libraries(GameOfLife AlgorithmCudaNormal)
	add_subdirectory(AlgorithmCudaGroup)
	target_include_directories(GameOfLife PUBLIC AlgorithmCudaGroup)
	target_link_libraries(GameOfLife AlgorithmCudaGroup)
endif()

# For FreeGlut
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_definitions(-DFREEGLUT_STATIC)
add_subdirectory(${PROJECT_SOURCE_DIR}/ExternalLibs/freeglut freeglut)
target_include_directories(GameOfLife PUBLIC ${PROJECT_SOURCE_DIR}/ExternalLibs/freeglut/include)
target_link_libraries(GameOfLife freeglut_static)

# For AntTweakBar
add_subdirectory(${PROJECT_SOURCE_DIR}/ExternalLibs/AntTweakBar AntTweakBar)
target_include_directories(GameOfLife PUBLIC ${PROJECT_SOURCE_DIR}/ExternalLibs/AntTweakBar/include)
target_link_libraries(GameOfLife AntTweakBar)

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

