#include <stdio.h>
#include "example1.cuh"
#include "class-work1.cuh"
#include "class-work2.cuh"
#include "home-work2.cuh"
#include "home-work3.cuh"
#include <iostream>
#include <stdexcept>
using namespace std;

int main()
{
	setlocale(LC_ALL, "Russian");
	//class or home or example
	string type = "class";
	//string type = "home";
	//string type = "example";
	int prograNumber = 2;

	if (type == "class") {
		if (prograNumber == 1) {
			classWork1();
		} else if (prograNumber == 2) {
			classWork2();
		}
	}
	else if (type == "home") {
		if (prograNumber == 2) {
			homeWork2();
		}
		else if (prograNumber == 3) {
			homeWork3();
		}
	}
	else if (type == "example") {
		if (prograNumber == 1) {
			example1();
		}
	}
}