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
#include <math_functions.h>

using namespace std;

#define BASE_TYPE float
#define M_PI 3.141592653

template <typename T>
__global__ void sinOfDouble(T* A, int arraySize)
{
	int index = getCurrentThreadId();

	*(A + index) = sin((index % 360) * M_PI / 180);
}

template <typename T>
float sinOfDoubleError(int index, T* hostData, int arraySize)
{
	return abs(sin((index % 360) * M_PI / 180) - *(hostData + index));
}

//---------------------------------------------------------------------//

template <typename T>
__global__ void sinOfFloat(T* A, int arraySize)
{
	int index = getCurrentThreadId();

	*(A + index) = __sinf((index % 360) * M_PI / 180);
}

template <typename T>
float sinOfFloatError(int index, T* hostData, int arraySize)
{
	return abs(sinf((index % 360) * M_PI / 180) - *(hostData + index));
}

//---------------------------------------------------------------------//

template <typename T>
__global__ void expOfDouble(T* A, int arraySize)
{
	int index = getCurrentThreadId();
	int maxNumber = 5;
	*(A + index) = exp((double)((index - (arraySize / 2)) % maxNumber));
}

template <typename T>
float expOfDoubleError(int index, T* hostData, int arraySize)
{
	int maxNumber = 5;
	return abs(exp((double)((index - (arraySize / 2)) % 5)) - *(hostData + index));
}

//---------------------------------------------------------------------//

template <typename T>
__global__ void expOfFloat(T* A, int arraySize)
{
	int index = getCurrentThreadId();
	int maxNumber = 5;
	*(A + index) = __expf((index - (arraySize / 2)) % maxNumber);
}

template <typename T>
float expOfFloatError(int index, T* hostData, int arraySize)
{
	int maxNumber = 5;
	return abs(expf((index - (arraySize / 2)) % 5) - *(hostData + index));
}

//---------------------------------------------------------------------//
template <typename T>
void testFunction(dim3* gridSize, dim3* blockSize, int* elementCount, T* deviceData, T* hostData, void (*func)(T*, int), float (*errorFunc)(int, T*, int)) {
	//------sinus test-------//
	cudaEvent_t start, stop;
	cudaError_t err;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	(*func) << <*gridSize, * blockSize >> > (deviceData, *elementCount);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(hostData, deviceData, *elementCount * sizeof(T), cudaMemcpyDeviceToHost);

	float error = 0;
	for (int i = 0; i < *elementCount; i++) {
		error += (*errorFunc)(i, hostData, *elementCount);
	}

	printf("Error: %f\n", error);

	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTime: %f millseconds\n", KernelTime);

	err = cudaGetLastError();
	if (err != cudaSuccess) printf("%s ", cudaGetErrorString(err));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//------sinus test end-------//
}
int classWork5() {
	dim3 gridSize(64, 64);
	dim3 blockSize(128);

	int elementCount = gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z;
	std::cout << "/////////////////////////////////////" << endl;
	std::cout << "/////////Тестирование double/////////" << endl;
	std::cout << "/////////////////////////////////////" << endl << endl;

	double* hostDataDouble = new double[elementCount];
	double* deviceDataDouble;

	cudaMalloc((void**)&deviceDataDouble, elementCount * sizeof(double));

	void (*funcDouble)(double* array, int size);
	float (*errorFuncDouble)(int, double*, int);

	funcDouble = sinOfDouble;
	errorFuncDouble = sinOfDoubleError;
	std::cout << "----------Тестирование синуса----------" << endl;
	testFunction<double>(&gridSize, &blockSize, &elementCount, deviceDataDouble, hostDataDouble, funcDouble, errorFuncDouble);
	std::cout << "---------------------------------------" << endl << endl;

	cudaDeviceSynchronize();

	funcDouble = expOfDouble;
	errorFuncDouble = expOfDoubleError;
	std::cout << "-----------Тестирование expr-----------" << endl;
	testFunction<double>(&gridSize, &blockSize, &elementCount, deviceDataDouble, hostDataDouble, funcDouble, errorFuncDouble);
	std::cout << "---------------------------------------" << endl << endl;

	cudaDeviceSynchronize();
	/*func = exp10;
	errorFunc = expr10Error;
	std::cout << "----------Тестирование expr10----------" << endl;
	testFunction(&gridSize, &blockSize, &elementCount, deviceData, hostData, func, errorFunc);
	std::cout << "---------------------------------------" << endl;*/

	//func = sinMass2;
	//std::cout << "----------Тестирование синуса----------" << endl;
	//testFunction(&gridSize, &blockSize, &elementCount, deviceData, hostData, func);
	//std::cout << "---------------------------------------" << endl;

	cudaFree(deviceDataDouble);
	delete[] hostDataDouble;

	std::cout << "/////////////////////////////////////" << endl;
	std::cout << "/////////Тестирование float//////////" << endl;
	std::cout << "/////////////////////////////////////" << endl << endl;

	float* hostDataFloat = new float[elementCount];
	float* deviceDataFloat;

	cudaMalloc((void**)&deviceDataFloat, elementCount * sizeof(float));
	
	void (*funcFloat)(float* array, int size);
	float (*errorFuncFloat)(int, float*, int);

	funcFloat = sinOfFloat;
	errorFuncFloat = sinOfFloatError;
	std::cout << "----------Тестирование синуса----------" << endl;
	testFunction<float>(&gridSize, &blockSize, &elementCount, deviceDataFloat, hostDataFloat, funcFloat, errorFuncFloat);
	std::cout << "---------------------------------------" << endl << endl;

	cudaDeviceSynchronize();

	funcFloat = expOfFloat;
	errorFuncFloat = expOfDoubleError;
	std::cout << "-----------Тестирование expr-----------" << endl;
	testFunction<float>(&gridSize, &blockSize, &elementCount, deviceDataFloat, hostDataFloat, funcFloat, errorFuncFloat);
	std::cout << "---------------------------------------" << endl << endl;

	cudaDeviceSynchronize();

	cudaFree(deviceDataFloat);
	delete[] hostDataFloat;

	return 0;
}