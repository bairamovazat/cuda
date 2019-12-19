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

#include "../utils/cuda_utils.cuh"

using namespace std;

#define BASE_TYPE float
#define M_PI 3.141592653

//__global__ void deviceMatrixMultiplicationWithForShared(float** firstMatrix, float** secondMatrix, float** targetMatrix, int n, const int blockSize) {
//	int i = threadIdx.x;
//	int j = threadIdx.y;
//	__shared__ float sharedMemory[blockSize];
//	for (int k = 0; k < n; k++) {
//		targetMatrix[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
//	}
//}

__global__ void deviceMatrixMultiplicationWithFor(float** firstMatrix, float** secondMatrix, float** targetMatrix, int n) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	for (int k = 0; k < n; k++) {
		targetMatrix[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
	}
}

__global__ void deviceMatrixAdditional(float** firstMatrix, float** secondMatrix, float** targetMatrix) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	targetMatrix[i][j] = firstMatrix[i][j] + secondMatrix[i][j];
}

__global__ void deviceMatrixMultiplication(float** firstMatrix, float** secondMatrix, float** targetMatrix) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = blockIdx.x;
	atomicAdd(&targetMatrix[i][j], firstMatrix[i][k] * secondMatrix[k][j]);
}

//Сравнение умножения через фор и attomicAdd
void matrixMultiplication(float** hostFirstMatrix, float** hostSecondMatrix, float** hostTargetMatrix, unsigned rows, unsigned columns, unsigned n) {
	float** deviceFirstArray = initArrayToDevice(rows, n);
	сopyArrayToDevice(hostFirstMatrix, deviceFirstArray, rows, n);

	float** deviceSecondArray = initArrayToDevice(n, columns);
	сopyArrayToDevice(hostSecondMatrix, deviceSecondArray, n, columns);

	float** deviceResultedArray = initArrayToDevice(rows, columns);
	сopyArrayToDevice(hostTargetMatrix, deviceResultedArray, rows, columns);

	dim3 gridSize(n);
	dim3 blockSize(rows, columns);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	deviceMatrixMultiplication << <gridSize, blockSize >> > (deviceFirstArray, deviceSecondArray, deviceResultedArray);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("Вычисление элемента на каждую нить через atomicAdd: %f millseconds\n", KernelTime);

	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);
	deviceMatrixMultiplicationWithFor << <1, blockSize >> > (deviceFirstArray, deviceSecondArray, deviceResultedArray, n);
	cudaEventRecord(stop2, 0);
	cudaEventSynchronize(stop2);
	float KernelTime2;
	cudaEventElapsedTime(&KernelTime2, start2, stop2);
	printf("Вычисление элемента через for: %f millseconds\n", KernelTime2);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	сopyArrayToHost(hostTargetMatrix, deviceResultedArray, rows, columns);
}

void matrixAdditional(float** hostFirstMatrix, float** hostSecondMatrix, float** hostTargetMatrix, unsigned rows, unsigned columns) {
	float** deviceFirstArray = initArrayToDevice(rows, columns);
	сopyArrayToDevice(hostFirstMatrix, deviceFirstArray, rows, columns);

	float** deviceSecondArray = initArrayToDevice(rows, columns);
	сopyArrayToDevice(hostSecondMatrix, deviceSecondArray, rows, columns);

	float** deviceResultedArray = initArrayToDevice(rows, columns);
	сopyArrayToDevice(hostTargetMatrix, deviceResultedArray, rows, columns);

	dim3 gridSize(1);
	dim3 blockSize(rows, columns);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	deviceMatrixAdditional << <gridSize, blockSize >> > (deviceFirstArray, deviceSecondArray, deviceResultedArray);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("Вычисление элемента через for: %f millseconds\n", KernelTime);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	сopyArrayToHost(hostTargetMatrix, deviceResultedArray, rows, columns);
}


int homeWork6() {
	unsigned firstArrayRows = 32;
	unsigned firstArrayColumns = 12;

	unsigned secondArrayRows = 12;
	unsigned secondArrayColumns = 32;

	unsigned n = firstArrayColumns;
	unsigned targerArrayRows = firstArrayRows;
	unsigned targerArrayColumns = secondArrayColumns;

	float** hostFirstArray = initArrayWithGenerator(firstArrayRows, firstArrayColumns, [](int i, int j, int rows, int columns) {return i * columns + j; });
	float** hostSecondArray = initArrayWithGenerator(secondArrayRows, secondArrayColumns, [](int i, int j, int rows, int columns) {return i * columns + j; });
	float** hostResultedArray = initArrayWithGenerator(targerArrayRows, targerArrayColumns, [](int i, int j, int rows, int columns) {return 0; });

	cout << "-------Умножение------\n";
	printMatrix(hostFirstArray, firstArrayRows, firstArrayColumns);
	cout << "----------------------\n";
	printMatrix(hostSecondArray, secondArrayRows, secondArrayColumns);
	cout << "----------------------\n";
	printMatrix(hostResultedArray, targerArrayRows, targerArrayColumns);
	cout << "----------------------\n";

	matrixMultiplication(hostFirstArray, hostSecondArray, hostResultedArray, firstArrayRows, secondArrayColumns, n);

	printMatrix(hostResultedArray, targerArrayRows, targerArrayColumns);
	cout << "----------------------\n";

	float** hostFirstArrayAdd = initArrayWithGenerator(firstArrayRows, firstArrayColumns, [](int i, int j, int rows, int columns) {return i * columns + j; });
	float** hostSecondArrayAdd = initArrayWithGenerator(firstArrayRows, firstArrayColumns, [](int i, int j, int rows, int columns) {return i * columns + j; });
	float** hostResultedArrayAdd = initArrayWithGenerator(firstArrayRows, firstArrayColumns, [](int i, int j, int rows, int columns) {return 0; });

	cout << "-------Сложение-------\n";
	printMatrix(hostFirstArrayAdd, firstArrayRows, firstArrayColumns);
	cout << "----------------------\n";
	printMatrix(hostSecondArrayAdd, firstArrayRows, firstArrayColumns);
	cout << "----------------------\n";
	printMatrix(hostResultedArrayAdd, firstArrayRows, firstArrayColumns);
	cout << "----------------------\n";
	matrixAdditional(hostFirstArrayAdd, hostSecondArrayAdd, hostResultedArrayAdd, firstArrayRows, firstArrayColumns);
	printMatrix(hostResultedArrayAdd, firstArrayRows, firstArrayColumns);
	cout << "----------------------\n";
	return 0;
}