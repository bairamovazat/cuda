#include <stdio.h>
#include "example\example1.cuh"
#include "example\example5.cuh"
#include "class\class-work1.cuh"
#include "class\class-work2.cuh"
#include "class\class-work4.cuh"
#include "class\class-work5.cuh"

#include "home-work\home-work2.cuh"
#include "home-work\home-work3.cuh"
#include "home-work\home-work4.cuh"
#include "home-work\home-work5.cuh"
#include "home-work\home-work6.cuh"
#include "home-work\home-work7.cuh"

#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace std;

int main()
{
	setlocale(LC_ALL, "Russian");
	ifstream is;
	is.open("config.ini");
	//class or home or example
	//string type = "class";
	//string type = "home";
	//string type = "example";
	string type;
	int prograNumber;

	is >> type >> prograNumber;
	if (type == "class") {
		if (prograNumber == 1) {
			classWork1();
		}
		else if (prograNumber == 2) {
			classWork2();
		}
		else if (prograNumber == 4) {
			classWork4();
		}
		else if (prograNumber == 5) {
			classWork5();
		}
	}
	else if (type == "home") {
		if (prograNumber == 2) {
			homeWork2();
		}
		else if (prograNumber == 3) {
			homeWork3();
		}
		else if (prograNumber == 4) {
			homeWork4();
		}
		else if (prograNumber == 5) {
			homeWork5();
		}
		else if (prograNumber == 6) {
			homeWork6();
		}
		else if (prograNumber == 7) {
			homeWork7();
		}
	}
	else if (type == "example") {
		if (prograNumber == 1) {
			example1();
		}
		else if (prograNumber == 5) {
			example5();
		}
	}
}