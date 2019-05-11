cmake_minimum_required(VERSION 2.8.10)

find_package(CUDA REQUIRED)

set(SOURCES
	algorithmCudaGroup.h
	algorithmCudaGroupInternal.h
	kernel.cu
	kernel_0.cu
)

CUDA_ADD_LIBRARY(AlgorithmCudaGroup STATIC
	${SOURCES}
)


CUDA_ADD_EXECUTABLE(AlgorithmCudaGroupTest
	${SOURCES}
	main.cpp
	referenceCode.cpp
)
