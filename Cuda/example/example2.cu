#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void warpTest()
{
	printf("BlockId: %d, ThreadId: %d\n", blockIdx.x, threadIdx.x);
}

int example2()
{
	warpTest << <5, 32 >> > ();
	// ожидаем нажатия любой клавиши
	getchar();
	return 0;
}