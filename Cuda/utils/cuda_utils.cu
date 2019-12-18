#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
#include <curand_kernel.h>

using namespace std;

//��������� ���������� ������������� ����� �� ���� �����
__device__ int getCurrentThreadId() {
	const unsigned long long int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

	const unsigned long long int threadId = blockId * blockDim.x + threadIdx.x;

	return threadId;
}

//�������� ������� ������� ����� ����� (2 * 2 * 2 ) - 0...7
//���� ���������� ��� 3 ������ ������
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

//template <typename T>
//T** initArrayToDevice(unsigned row, unsigned column) {
//	T** array;
//	cudaMalloc((void**) & array, row * sizeof(T*));
//	for (int i = 0; i < row; i++) {   // (3)
//		T* arrayElement;
//		cudaMalloc((void**)&arrayElement, column * sizeof(T));
//		array[i] = arrayElement;     // ������������� ����������
//	}
//
//	return array;
//}
//
//template <typename T>
//void �opyArrayToDevice(T** hostArray, T** deviceArray, unsigned row, unsigned column) {
//	for (int i = 0; i < row; i++) {   // (3)
//		cudaMemcpy(deviceArray[i], hostArray[i], column * sizeof(T), cudaMemcpyHostToDevice);
//	}
//}
//
//template <typename T>
//void �opyArrayToHost(T** hostArray, T** deviceArray, unsigned row, unsigned column) {
//	for (int i = 0; i < row; i++) {   // (3)
//		cudaMemcpy(hostArray[i], deviceArray[i], column * sizeof(T), cudaMemcpyDeviceToHost);
//	}
//}
//
//template <typename T>
//T** initAndCopyArrayToDevice(T** hostArray, unsigned row, unsigned column) {
//	T** deviceArray = initArrayToDevice<T>(row, column);
//	�opyArrayToDevice(hostArray, deviceArray, row, column);
//	return deviceArray;
//}
//
//template <typename T>
//void freeArray(T** deviceArray) {
//	cudaFree(deviceArray);
//}
//
//
//int generatorZero(int i, int j, int row, int column) {
//	return 0;
//}
//
//template <typename T>
//T** initArrayWithGenerator(unsigned row, unsigned column, int (generateValue)(int, int, int, int)) {
//	// ��������
//	T** array = new T * [row];    // ������ ���������� (2)
//	for (int i = 0; i < row; i++) {   // (3)
//		array[i] = new T[column];     // ������������� ����������
//		for (int j = 0; j < column; j++) {
//			array[i][j] = (generateValue)(i, j, row, column);
//		}
//	}
//	return array;
//}
//
//template <typename T>
//T** initArray(unsigned row, unsigned column) {
//	return initArrayWithGenerator<T>(row, column, generatorZero);
//}
//
//
//template <typename T>
//void freeArray(T** array, unsigned row, unsigned column) {
//	for (int i = 0; i < row; i++) {
//		delete[] array[i];
//	}
//	delete[] array;
//}


float** initArrayToDevice(unsigned row, unsigned column) {
	float** tempArray = new float* [row];
	float** array;
	cudaMalloc((void**) & array, row * sizeof(float*));
	for (int i = 0; i < row; i++) {
		cudaMalloc(&tempArray[i], column * sizeof(float));
	}
	cudaMemcpy(array, tempArray, row * sizeof(float*), cudaMemcpyHostToDevice);

	return array;
}

void �opyArrayToDevice(float** hostArray, float** deviceArray, unsigned row, unsigned column) {
	float** tempArray = new float* [row];
	cudaMemcpy(tempArray, deviceArray, row * sizeof(float*), cudaMemcpyDeviceToHost);
	for (int i = 0; i < row; i++) {
		cudaMemcpy(tempArray[i], hostArray[i], column * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void �opyArrayToHost(float** hostArray, float** deviceArray, unsigned row, unsigned column) {
	float** tempArray = new float* [row];
	cudaMemcpy(tempArray, deviceArray, row * sizeof(float*), cudaMemcpyDeviceToHost);

	for (int i = 0; i < row; i++) {
		cudaMemcpy(hostArray[i], tempArray[i], column * sizeof(float), cudaMemcpyDeviceToHost);
	}
}

float** initAndCopyArrayToDevice(float** hostArray, unsigned row, unsigned column) {
	float** deviceArray = initArrayToDevice(row, column);
	�opyArrayToDevice(hostArray, deviceArray, row, column);
	return deviceArray;
}

void freeArray(float** deviceArray) {
	cudaFree(deviceArray);
}

int generatorZero(int i, int j, int row, int column) {
	return 0;
}

float** initArrayWithGenerator(unsigned row, unsigned column, int (generateValue)(int, int, int, int)) {
	// ��������
	float** array = new float* [row];    // ������ ���������� (2)
	for (int i = 0; i < row; i++) {   // (3)
		array[i] = new float[column];     // ������������� ����������
		for (int j = 0; j < column; j++) {
			array[i][j] = (generateValue)(i, j, row, column);
		}
	}
	return array;
}

float** initArray(unsigned row, unsigned column) {
	return initArrayWithGenerator(row, column, generatorZero);
}

void freeArray(float** array, unsigned row, unsigned column) {
	for (int i = 0; i < row; i++) {
		delete[] array[i];
	}
	delete[] array;
}

void printMatrix(float** matrix, unsigned rows, unsigned columns) {
	for (int row = 0; row < rows; row++) {
		float* element = *(matrix + row);
		for (int column = 0; column < columns; column++) {
			cout << *(element + column) << ", ";
		}
		cout << endl;
	}
}