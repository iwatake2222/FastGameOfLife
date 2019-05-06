cmake_minimum_required(VERSION 2.8.10)

find_package(CUDA REQUIRED)

CUDA_ADD_LIBRARY(AlgorithmCudaNormal STATIC
	algorithmCudaNormal.h
	algorithmCudaNormalInternal.h
	kernel.cu
	kernel_0.cu
	kernel_0_stream.cu
	kernel_1.cu
	kernel_2.cu
	kernel_2_stream.cu
	kernel_3_stream.cu
)

