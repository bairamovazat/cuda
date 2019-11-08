#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iomanip>
#include <time.h>
#include <iostream>
using namespace std;

// ����
__global__ void add(int *a, int *b, int *c) {
	*c = *a + *b;
}

int classWork1() {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	//���������� ����������������� �� ����������
	printf("���������� ����������������� �� ����������: %d\n", deviceProp.multiProcessorCount);
	//���������� ������ �������� �� ���������� � ������
	printf("���������� ������ �������� �� ���������� � ������ : %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
	//�������� ������� � ����������
	printf("�������� ������� � ����������: %d\n", deviceProp.clockRate);
	//������� �������� ������� ������ � ����������
	printf("������� �������� ������� ������ � ����������: %d\n", deviceProp.memoryClockRate);
	//������ ���� ���������� ������ � �����
	printf("������ ���� ���������� ������ � �����: %d\n", deviceProp.memoryBusWidth);

	// ���������� �� CPU
	int a, b, c;

	// ���������� �� GPU
	int *dev_a, *dev_b, *dev_c;
	int size = sizeof(int); //�����������
	// �������� ������ �� GPU
	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);
	// ������������� ����������
	a = 2;
	b = 7;
	// ����������� ���������� � CPU �� GPU
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);
	// ����� ����
	add << < 1, 1 >> > (dev_a, dev_b, dev_c);
	// ����������� ���������� ������ ���� � GPU �� CPU
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);
	// ����� ����������
	printf("%d + %d = %d\n", a, b, c);
	// �������� ������ �� GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}