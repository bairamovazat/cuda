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
#include "../models/Vector.h";
#include "../utils/cuda_utils.cuh"

using namespace std;

#define BASE_TYPE float
#define M_PI 3.141592653


__global__ void sinMass(BASE_TYPE* A, int arraySize)
{
	int index = getCurrentThreadId();
	
	//*(A + index) = sin((BASE_TYPE)((index % 360) * M_PI / 180));
	
	// другие варианты вычисления синуса
	//*(A + index) = sinf((index % 360) * M_PI / 180);
	*(A + index) = __sinf((index % 360) * M_PI / 180);
}

int example5() {
	Vector<int> v (12);
	Vector<int>& v2 = v;
	v.getElement(1);

	dim3 gridSize(64, 64);
	dim3 blockSize(32);

	int elementCount = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z;

	BASE_TYPE* hostData = new BASE_TYPE[elementCount];

	BASE_TYPE* deviceData;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&deviceData, elementCount * sizeof(BASE_TYPE));

	cudaEventRecord(start, 0);

	sinMass << <gridSize, blockSize >> > (deviceData, elementCount);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(hostData, deviceData, elementCount * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);

	float error = 0;
	for (int i = 0; i < elementCount; i++) {
		error += (abs(sin((i % 360) * M_PI / 180) - *(hostData + i)));
	}

	printf("\nError: %f\n", error);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f millseconds\n", KernelTime);

	int zeroCount = 0;
	for (int i = 9; i < elementCount; i++) {
		if (*(hostData + i) == 0) {
			zeroCount++;
		}
	}

	printf("%d", zeroCount);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	cudaFree(deviceData);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}