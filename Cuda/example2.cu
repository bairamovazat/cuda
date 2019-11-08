#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
__global__ void HelloWorld()
{
	printf("Hello world, %d, %d\n", blockIdx.x,
		threadIdx.x);
}

int example2()
{
	HelloWorld << <4, 4 >> > ();
	// ������� ������� ����� �������
	getchar();
	return 0;
}