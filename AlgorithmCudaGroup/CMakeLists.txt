cmake_minimum_required(VERSION 2.8.10)

find_package(CUDA REQUIRED)

CUDA_ADD_LIBRARY(AlgorithmCudaGroup STATIC
	algorithmCudaGroup.h
	algorithmCudaGroupInternal.h
	kernel.cu
	kernel_0.cu
)

