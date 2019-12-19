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
#include <cmath>
#include <device_functions.h>

#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "../utils/cuda_utils.cuh"

using namespace std;


__global__ void deviceVectorRateWithShared(float** firstMatrix, float* target) {
	int threadId = threadIdx.x;
	int vectorId = blockIdx.x * blockDim.x + threadId;
	extern __shared__ float sharedMemory[];

	//Это эквивалентно записи просто в локальную переменную
	sharedMemory[threadId] = firstMatrix[vectorId][0];

	__syncthreads();

	atomicAdd(target, sharedMemory[threadId] * sharedMemory[threadId]);
}

__global__ void deviceVectorRate(float** firstMatrix, float* target) {
	int threadId = threadIdx.x;
	int vectorId = blockIdx.x * blockDim.x + threadId;
	atomicAdd(target, firstMatrix[vectorId][0] *  firstMatrix[vectorId][0]);
}

void vectorRate(float** hostFirstMatrix, float* result, unsigned rows) {
	float** deviceFirstArray = initArrayToDevice(rows, 1);
	сopyArrayToDevice(hostFirstMatrix, deviceFirstArray, rows, 1);

	float* deviceResult;
	cudaMalloc((void**) &deviceResult, sizeof(float));
	cudaMemcpy(deviceResult, result, sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridSize(rows / 16);
	dim3 blockSize(16);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//В первый блок будем записывать результат
	deviceVectorRateWithShared << <gridSize, blockSize, (16) * sizeof(float) >> > (deviceFirstArray, deviceResult);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("Вычисление элемента через shared: %f millseconds\n", KernelTime);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	//В первый блок будем записывать результат
	deviceVectorRate << <gridSize, blockSize, (16) * sizeof(float) >> > (deviceFirstArray, deviceResult);
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float KernelTime2;
	cudaEventElapsedTime(&KernelTime, start2, stop2);
	printf("Вычисление элемента: %f millseconds\n", KernelTime2);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	cudaMemcpy(result, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);
}

int homeWork7() {
	unsigned arrayRows = 32;
	unsigned arrayColumns = 1;

	unsigned secondArrayRows = 32;
	unsigned secondArrayColumns = 1;

	float** hostFirstArray = initArrayWithGenerator(arrayRows, arrayColumns, [](int i, int j, int rows, int columns) {return i * columns + j; });
	float result = 0;
	float* target = &result;

	//cout << "---------Норма--------\n";
	//printMatrix(hostFirstArray, arrayRows, arrayColumns);
	//cout << "----------------------\n";

	vectorRate(hostFirstArray, target, arrayRows);
	//cout << "----------------------\n";
	cout << *target;

	return 0;
}