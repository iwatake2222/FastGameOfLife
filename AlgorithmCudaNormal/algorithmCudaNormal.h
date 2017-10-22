#pragma once
namespace AlgorithmCudaNormal
{
#if 0
}	// indent guard
#endif

#ifdef DLL_EXPORT
__declspec(dllexport) void allocManaged(int **p, int size);
__declspec(dllexport) void freeManaged(int *p);
__declspec(dllexport) void cudaDeviceSynchronizeWrapper();
__declspec(dllexport) void logicForOneGeneration(int* matDst, int* matSrc, int width, int height);
#else
__declspec(dllimport) void allocManaged(int **p, int size);
__declspec(dllimport) void freeManaged(int *p);
__declspec(dllimport) void cudaDeviceSynchronizeWrapper();
__declspec(dllimport) void logicForOneGeneration(int* matDst, int* matSrc, int width, int height);
#endif

}