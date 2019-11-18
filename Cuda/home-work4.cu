//подключение библиотек
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
using namespace std;

#define WARP_SIZE 32;

__device__ int getCurrentThreadId() {
	int blockId = (gridDim.x * blockIdx.y) + blockIdx.x;
	int threadId = (blockId * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ curandState initState() {
	int id = getCurrentThreadId();
	curandState state;
	curand_init(1234, id, 0, &state);
	return state;
}

__device__ float getRandFloat(curandState *state) {
	return curand_uniform(state);
}

__device__ float getRandFloat() {
	curandState state = initState();
	return getRandFloat(&state);
}

__global__ void monteCarlo(int* inCircle, int* inSquare)
{
	curandState state = initState();

	float x = getRandFloat(&state) * 1;
	float y = getRandFloat(&state) * 1;

	if (sqrt(float(pow(x,2) + pow(y,2))) < 1) {
		atomicAdd(inCircle, 1);
	}
	else {
		atomicAdd(inSquare, 1);
	}
}

int homeWork4() {
	dim3 gridSize(300);
	dim3 blockSize(50);
	int hostInCircle = 0;
	int hostInSquare = 0;
	int *deviceInCircle;
	int *deviceInSquare;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)& deviceInCircle, sizeof(int));
	cudaMalloc((void**)& deviceInSquare, sizeof(int));

	cudaEventRecord(start, 0);

	monteCarlo << <300, 200 >> > (deviceInCircle, deviceInSquare);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(&hostInCircle, deviceInCircle, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostInSquare, deviceInSquare, sizeof(int), cudaMemcpyDeviceToHost);

	printf("hostInCircle: %d\n", hostInCircle);
	printf("hostInSquare: %d\n", hostInSquare);

	float pi = (4 * float(hostInCircle) / (float(hostInSquare) + float(hostInCircle)));
	printf("%f\n", pi);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}