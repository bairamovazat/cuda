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
#include <assert.h>
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "../utils/cuda_utils.cuh"

using namespace std;

//#define VECTOR_SIZE 4096
#define VECTOR_SIZE 256
#define GRID_SIZE 32

__device__ __constant__ float constFirstVector[VECTOR_SIZE];
__device__ __constant__ float constSecondVector[VECTOR_SIZE];

texture<float> firstTexture;
texture<float> secondTexture;

__global__ void deviceMultiply(float* firstVector, float* secondVector, float* result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(result, *(firstVector + id) * *(secondVector + id));
}

void multiply(float* firstVector, float* secondVector, float* result) {

	float* deviceFirstVector;
	cudaMalloc(&deviceFirstVector, VECTOR_SIZE * sizeof(float));
	cudaMemcpy(deviceFirstVector, firstVector, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	float* deviceSecondVector;
	cudaMalloc(&deviceSecondVector, VECTOR_SIZE * sizeof(float));
	cudaMemcpy(deviceSecondVector, secondVector, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	float* deviceResult;
	cudaMalloc(&deviceResult, sizeof(float));
	cudaMemcpy(deviceResult, result, sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	deviceMultiply << <VECTOR_SIZE / GRID_SIZE, GRID_SIZE >> > (deviceFirstVector, deviceSecondVector, deviceResult);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(result, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("¬ычисление скал€рного произведени€: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
}

__global__ void deviceMultiply(float* result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(result, constFirstVector[id] * constSecondVector[id]);
}

void constMultiply(float* firstVector, float* secondVector, float* result) {
	cudaMemcpyToSymbol(constFirstVector, firstVector, VECTOR_SIZE * sizeof(float));

	cudaMemcpyToSymbol(constSecondVector, secondVector, VECTOR_SIZE * sizeof(float));

	float* deviceResult;
	cudaMalloc(&deviceResult, sizeof(float));
	cudaMemcpy(deviceResult, result, sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	deviceMultiply << <VECTOR_SIZE / GRID_SIZE, GRID_SIZE >> > (deviceResult);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(result, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("¬ычисление скал€рного произведени€ через __constant__: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
}

void printVector(float* vector, int size) {
	for (int i = 0; i < size; i++) {
		printf("%f, ", *(vector + i));
	}
	printf("\n");
}
void constMemoryTest() {
	float* firstVector = new float[VECTOR_SIZE];
	float* secondVector = new float[VECTOR_SIZE];

	for (int i = 0; i < VECTOR_SIZE; i++) {
		*(firstVector + i) = i;
		*(secondVector + i) = i;
	}
	printVector(firstVector, VECTOR_SIZE);
	printVector(secondVector, VECTOR_SIZE);
	cout << "-----------------constMultiply-----------------\n";
	float result = 0;
	constMultiply(firstVector, secondVector, &result);
	printf("result: %f\n", result);
	cout << "-----------------multiply-----------------\n";
	float simpleResult = 0;
	multiply(firstVector, secondVector, &simpleResult);
	printf("result: %f\n", simpleResult);
}
__global__ void textureDeviceMultiply(float* result) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	float x2 = tex1D(secondTexture, float(id));
	float x1 = tex1D(firstTexture, float(id));

	atomicAdd(result, x1 * x1);
}

void textureMultiply(float* hostFirstVector, float* hostSecondVector, float* result, int size) {
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(size, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaArray* firstDeviceCudaArray;
	cudaMallocArray(&firstDeviceCudaArray, &firstTexture.channelDesc, size, 1);
	cudaMemcpyToArray(firstDeviceCudaArray, 0, 0, hostFirstVector, sizeof(float) * size, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(firstTexture, firstDeviceCudaArray, firstTexture.channelDesc);

	float* secondDeviceArray;
	cudaMalloc((void**)&secondDeviceArray, size * sizeof(float));
	cudaMemcpy(secondDeviceArray, hostSecondVector, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTexture(0, secondTexture, secondDeviceArray, firstTexture.channelDesc, size * sizeof(float));

	float* deviceResult;
	cudaMalloc(&deviceResult, sizeof(float));
	cudaMemcpy(deviceResult, result, sizeof(float), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	textureDeviceMultiply << <VECTOR_SIZE / GRID_SIZE, GRID_SIZE >> > (deviceResult);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaMemcpy(result, deviceResult, sizeof(float), cudaMemcpyDeviceToHost);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("¬ычисление скал€рного произведени€ через текстурную пам€ть использу€ cudaArray и линейную пам€ть: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));
}

void textureMemoryTest() {
	float* firstVector = new float[VECTOR_SIZE];
	float* secondVector = new float[VECTOR_SIZE];

	for (int i = 0; i < VECTOR_SIZE; i++) {
		*(firstVector + i) = i;
		*(secondVector + i) = i;
	}
	printVector(firstVector, VECTOR_SIZE);
	printVector(secondVector, VECTOR_SIZE);

	float result = 0;
	textureMultiply(firstVector, secondVector, &result, VECTOR_SIZE);
	printf("result: %f\n", result);
}


int homeWork8() {
	cout << "-----------------constMemoryTest-----------------\n";
	constMemoryTest();
	cout << "-----------------textureMemoryTest-----------------\n";
	textureMemoryTest();
	return 0;
}