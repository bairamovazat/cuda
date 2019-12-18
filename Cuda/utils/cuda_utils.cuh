#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#ifndef cuda_utils_def
#define cuda_utils_def


__device__ int getCurrentThreadId();

__device__ int getCurrentThreadNumber();

__device__ curandState initState();

__device__ float getRandFloat(curandState* state);

__device__ float getRandFloat();

//template <typename T>
//T** initArrayToDevice(unsigned row, unsigned column);
//
//template <typename T>
//T** ñopyArrayToDevice(T** hostArray, T** deviceArray, unsigned row, unsigned column);
//
//template <typename T>
//T** ñopyArrayToHost(T** hostArray, T** deviceArray, unsigned row, unsigned column);
//
//template <typename T>
//T** initAndCopyArrayToDevice(T** hostArray, unsigned row, unsigned column);
//
//template <typename T>
//void freeArray(T** deviceArray);
//
//template <typename T>
//T** initArray(unsigned row, unsigned column);
//
//template <typename T>
//T** initArrayWithGenerator(unsigned row, unsigned column, int (generateValue)(int, int, int, int));
//
//template <typename T>
//T** freeArray(T** array, unsigned row, unsigned column);

float** initArrayToDevice(unsigned row, unsigned column);

void ñopyArrayToDevice(float** hostArray, float** deviceArray, unsigned row, unsigned column);

void ñopyArrayToHost(float** hostArray, float** deviceArray, unsigned row, unsigned column);

float** initAndCopyArrayToDevice(float** hostArray, unsigned row, unsigned column);

void freeArray(float** deviceArray);

float** initArray(unsigned row, unsigned column);

float** initArrayWithGenerator(unsigned row, unsigned column, int (generateValue)(int, int, int, int));

void freeArray(float** array, unsigned row, unsigned column);

void printMatrix(float** array, unsigned row, unsigned column);
#endif


