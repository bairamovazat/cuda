#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
using namespace std;

// ядро
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int classWork1() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	//Количество мультипроцессоров на устройстве
	printf("Количество мультипроцессоров на устройстве: %d\n", deviceProp.multiProcessorCount);
	//Глобальная память доступна на устройстве в байтах
	printf("Глобальная память доступна на устройстве в байтах : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
	//Тактовая частота в килогерцах
	printf("Тактовая частота в килогерцах: %d\n", deviceProp.clockRate);
	//Пиковая тактовая частота памяти в килогерцах
	printf("Пиковая тактовая частота памяти в килогерцах: %d\n", deviceProp.memoryClockRate);
	//Ширина шины глобальной памяти в битах
	printf("Ширина шины глобальной памяти в битах: %d\n", deviceProp.memoryBusWidth);

	// переменные на CPU
	int a, b, c;

	// переменные на GPU
	int *dev_a, *dev_b, *dev_c;
	int size = sizeof(int); //размерность
	// выделяем память на GPU
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	// инициализация переменных
	a = 2;
	b = 7;
	// копирование информации с CPU на GPU
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);
	// вызов ядра
	add << < 1, 1 >> > (dev_a, dev_b, dev_c);
	// копирование результата работы ядра с GPU на CPU
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	// вывод информации
	printf("%d + %d = %d\n", a, b, c);
	// очищение памяти на GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}