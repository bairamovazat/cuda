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

////Записывает в элемент targer[index] данные из (source[index] * coeff)
//__device__ int appendToVector(double* targer, double* source, double coeff, int index) {
//
//}
//
////Записывает в строку targer данные из (source * coeff)
//__device__ int appendToVector(double* targer, double* source, double coeff, int vectorLength) {
//	//Потенциально можно сделать через потоки
//}

__device__ double multiple(double* first, double* second, size_t length) {
	double result = 0;
	for (int i = 0; i < length; i++) {
		result += (*(first + i) * *(second + i));
	}

	return result;	
}
__device__ int getElementNumber(size_t row, size_t column, size_t columns) {
	return row * columns + column;
}

__device__ void orthogonalization(double* matrix, double* resultMatrix, size_t rows, size_t columns) {
	int row = threadIdx.x;
	int column = threadIdx.y;
	return;
}

int homeWork5() {
	size_t rows = 5;
	size_t columns = 5;

	dim3 gridSize(64, 64);
	dim3 blockSize(16);

	int elementCount = rows * columns;

	double* hostDataDouble = new double[elementCount];
	double* deviceDataDouble;

	cudaMalloc((void**)&deviceDataDouble, elementCount * sizeof(double));



	cudaFree(deviceDataDouble);
	delete[] hostDataDouble;



	return 0;
}