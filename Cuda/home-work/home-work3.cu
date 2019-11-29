//подключение библиотек
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
using namespace std;

#define N 1000
#define S 2
#define BLOCK_SIZE 1

__global__ void zeta(float* c)
{
	int tid = threadIdx.x;
	int idx = blockIdx.x;
	int ind = blockDim.x * idx + tid + blockDim.y * blockIdx.y + threadIdx.y;
	if (ind > N - 1) return;
	float res = float(1) / pow(double(ind + 1), S);
	c[ind] = res;
}

int homeWork3() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float* host_c;
	float* dev_c;
	cout << endl;

	host_c = (float*)malloc(N * sizeof(float));
	cudaMalloc((void**)& dev_c, N * sizeof(float));


	cudaEventRecord(start, 0);

	dim3 threadPerBlock = dim3(N / BLOCK_SIZE, 1);
	dim3 blockPerGrid = dim3(BLOCK_SIZE, BLOCK_SIZE);

	zeta << <blockPerGrid, threadPerBlock >> > (dev_c);

	// Время работы ядра
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float KernelTime;
	cudaEventElapsedTime(&KernelTime, start, stop);
	printf("KernelTme: %f millseconds\n", KernelTime);

	// Проверка на ошибку выполнения программы на девайсе
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("%s ", cudaGetErrorString(err));

	cudaMemcpy(host_c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		//cout << host_c[i] << " ";
		sum += host_c[i];
	}
	cout << endl;
	cout << "Value Zeta Function: " << sum << " ";

	float error = abs(4 * atan(1) * 4 * atan(1) / float(6) - sum);
	printf("\nError: %f\n", error);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}