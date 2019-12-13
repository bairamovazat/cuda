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
#include <cmath>

using namespace std;

#define BASE_TYPE float
#define M_PI 3.141592653

__global__ void calculateVector(int row, float* sourceAMatrix, float* targetBMatrix, float* buffetProjMatrix, int rows, int columns) {
	int vectorNumber = threadIdx.x;
	int vectorElement = threadIdx.y;
	float* bVector = targetBMatrix + (row * columns);
	float* aVector = sourceAMatrix + (row * columns);
	if (vectorNumber == 0) {
		atomicAdd(bVector + vectorElement, *(aVector + vectorElement));
	}
	else if (vectorNumber > row) {
		//Ничего не делаем, дальше элементы не считаем
	}
	else {
		float upper = *(buffetProjMatrix + vectorNumber);
		float lower = *(buffetProjMatrix + columns + vectorNumber);
		printf("Upper: %f, Lower: %f, %d, %d, %d\n", upper, lower, row, vectorNumber, vectorElement);
		atomicAdd(bVector + vectorElement, (*(aVector + vectorElement) * -upper / lower));
	}

}

__global__ void calculateProj(int row, float* sourceAMatrix, float* targetBMatrix, float* buffetProjMatrix, int rows, int columns) {
	int bVectorIndex = threadIdx.x - 1;
	int vectorElement = threadIdx.y;
	//Если 0, то верхнее <a,b>
	//Если 1, то нижнее  <b,b>
	int lowerExpr = blockIdx.x;

	float* target = buffetProjMatrix + (columns * lowerExpr) + (bVectorIndex + 1);

	float* result = 0;

	//Если это нулевой элемент proj(b-1,a_row)
	if (bVectorIndex == -1)
	{
		//atomicAdd(target, *(sourceAMatrix + (row * columns) + vectorElement));
	}
	else if (bVectorIndex >= row) {
		//Ничего не делаем, дальше элементы не считаем
	}
	else {
		float* vectorA;

		if (lowerExpr == 1) {
			vectorA = targetBMatrix + (bVectorIndex * columns);
		}
		else {
			vectorA = sourceAMatrix + (row * columns);
		}

		float* vectorB = targetBMatrix + (bVectorIndex * columns);

		atomicAdd(target, (*(vectorB + vectorElement) * *(vectorA + vectorElement)));
	}

	return;
}

void printMatrix(float* matrix, int rows, int columns) {
	for (int row = 0; row < rows; row++) {
		for (int column = 0; column < columns; column++) {
			cout << *(matrix + (row * columns) + column) << ", ";
		}
		cout << endl;
	}
}

void fillMatrix(float* matrix, int rows, int columns, bool allZero, bool upperTreangle) {
	for (int row = 0; row < rows; row++) {
		for (int column = 0; column < columns; column++) {
			if (upperTreangle && row <= column) {
				*(matrix + (row * columns) + column) = 1.0;
			}
			else if(upperTreangle && row < column || allZero) {
				*(matrix + (row * columns) + column) = 0.0;
			}
		}
	}
}

int homeWork5() {
	size_t rows = 3;
	size_t columns = 3;

	dim3 gridSizeProj(2);
	dim3 gridSize(1);

	dim3 blockSize(rows, columns);

	int elementCount = rows * columns;

	float* sourceAMatrix = new float[elementCount];

	fillMatrix(sourceAMatrix, rows, columns, true, false);
	printMatrix(sourceAMatrix, rows, columns);
	cout << "--------------------------" << endl;
	fillMatrix(sourceAMatrix, rows, columns, false, true);
	printMatrix(sourceAMatrix, rows, columns);
	cout << "--------------------------" << endl;

	float* deviceSourceAMatrix;
	cudaMalloc((void**)&deviceSourceAMatrix, elementCount * sizeof(float));

	float* targetBMatrix = new float[elementCount];
	fillMatrix(targetBMatrix, rows, columns, true, false);
	float* deviceTargetBMatrix;
	cudaMalloc((void**)&deviceTargetBMatrix, elementCount * sizeof(float));

	float* buffetProjMatrix = new float[columns * 2];
	fillMatrix(targetBMatrix, 2, columns, true, false);
	float* deviceBuffetProjMatrix;
	cudaMalloc((void**)&deviceBuffetProjMatrix, columns * 2 * sizeof(float));

	float* buffetProjMatrixToPrint = new float[columns * 2];

	cudaMemcpy(deviceSourceAMatrix, sourceAMatrix, elementCount * sizeof(float), cudaMemcpyHostToDevice);

	for (int i = 0; i < rows; i++) {
		cudaMemcpy(deviceBuffetProjMatrix, buffetProjMatrix, columns * 2 * sizeof(float), cudaMemcpyHostToDevice);

		calculateProj << <gridSizeProj, blockSize >> > (i, deviceSourceAMatrix, deviceTargetBMatrix, deviceBuffetProjMatrix, rows, columns);
		cudaDeviceSynchronize();
		calculateVector << <gridSize, blockSize >> > (i, deviceSourceAMatrix, deviceTargetBMatrix, deviceBuffetProjMatrix, rows, columns);

		cudaMemcpy(buffetProjMatrixToPrint, deviceBuffetProjMatrix, 2 * columns * sizeof(float), cudaMemcpyDeviceToHost);
		printMatrix(buffetProjMatrixToPrint, 2, columns);
		cout << "--------------------------" << endl;
		cudaMemcpy(targetBMatrix, deviceTargetBMatrix, elementCount * sizeof(float), cudaMemcpyDeviceToHost);
		printMatrix(targetBMatrix, rows, columns);
		cout << "--------------------------" << endl;

	}

	cudaMemcpy(targetBMatrix, deviceTargetBMatrix, elementCount * sizeof(float), cudaMemcpyDeviceToHost);

	printMatrix(targetBMatrix, rows, columns);

	cudaFree(deviceSourceAMatrix);
	cudaFree(deviceTargetBMatrix);
	cudaFree(deviceBuffetProjMatrix);

	delete[] sourceAMatrix;
	delete[] targetBMatrix;
	delete[] buffetProjMatrix;

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	return 0;
}