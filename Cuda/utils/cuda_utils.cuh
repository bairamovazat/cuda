#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifndef cuda_utils_def
#define cuda_utils_def

__device__ int getCurrentThreadId();

__device__ int getCurrentThreadNumber();

__device__ curandState initState();

__device__ float getRandFloat(curandState* state);

__device__ float getRandFloat();

#endif cuda_utils_def