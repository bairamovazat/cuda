#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>

//__device__ int getCurrentThreadId() {
//	int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
//	int threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
//	return threadId;
//}

