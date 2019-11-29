#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
#include <curand_kernel.h>

//Вычисляет уникальный идентификатор треда во всей сетке
__device__ unsigned long long getCurrentThreadId() {
	const unsigned long long int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z; 

	const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;

	return threadId;
}

//Получает текущий линейый номер треда (2 * 2 * 2 ) - 0...7
//Надо переделать под 3 мерный случай
__device__ int getCurrentThreadNumber() {
	return threadIdx.x;
}

__device__ curandState initState() {
	int id = getCurrentThreadId();
	curandState state;
	curand_init(1234, id, 0, &state);
	return state;
}

__device__ float getRandFloat(curandState* state) {
	return curand_uniform(state);
}

__device__ float getRandFloat() {
	curandState state = initState();
	return getRandFloat(&state);
}


