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

int classWork4() {
	dim3 gridSize(1024);
	dim3 blockSize(2048);

	int hostInCircle = 0;
	int hostInSquare = 0;

	int * deviceInCircle;
	int * deviceInSquare;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)& deviceInCircle, sizeof(int));
	cudaMalloc((void**)& deviceInSquare, sizeof(int));

	cudaEventRecord(start, 0);

	monteCarlo << <gridSize, blockSize >> > (deviceInCircle, deviceInSquare);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(&hostInCircle, deviceInCircle, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&hostInSquare, deviceInSquare, sizeof(int), cudaMemcpyDeviceToHost);


	float pi = (4 * float(hostInCircle) / (float(hostInSquare) + float(hostInCircle)));

	printf("Pi: %f\n", pi);
	
	float error = abs(4 * atan(1) - pi);
	printf("\nError: %f\n", error);


	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	cudaFree(deviceInCircle);
	cudaFree(deviceInSquare);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}