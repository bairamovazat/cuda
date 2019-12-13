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
#include "../utils/cuda_utils.cuh"

using namespace std;

__global__ void monteCarlo(int* inCircle, int maxSize)
{
	curandState state = initState();

	float x = getRandFloat(&state) * 1;
	float y = getRandFloat(&state) * 1;

	if (sqrt(float(pow(x,2) + pow(y,2))) < 1) {
		atomicAdd((inCircle + (getCurrentThreadNumber())), 1);
	}
}

int homeWork4() {
	dim3 gridSize(256);
	dim3 blockSize(256);

	int threadsCount = blockSize.x * blockSize.y * blockSize.z;

	int * hostInCircle = new int[threadsCount];

	int hostInSquare = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z;

	int * deviceInCircle;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)& deviceInCircle, threadsCount * sizeof(int));

	cudaEventRecord(start, 0);

	monteCarlo << <gridSize, blockSize >> > (deviceInCircle, threadsCount);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(hostInCircle, deviceInCircle, threadsCount * sizeof(int), cudaMemcpyDeviceToHost);

	int hostInCircleTotal = 0;

	for (int i = 0; i < threadsCount; i++) {
		hostInCircleTotal += *(hostInCircle + i);
	}

	float pi = (4 * float(hostInCircleTotal) / float(hostInSquare));

	printf("Pi: %f\n", pi);

	float error = abs(4 * atan(1) - pi);
	printf("\nError: %f\n", error);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	cudaFree(deviceInCircle);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}