#include <iostream>
#include <fstream>
#include <string>
#include "config-reader.h"
using namespace std;
//
//void processLine(string line, char **type, int *number) {
//	size_t pos = 0;
//	std::string token;
//	std::string previousToken;
//
//	while ((pos = line.find("=")) != std::string::npos) {
//		token = line.substr(0, pos);
//		if (previousToken.compare("programType") == 0) {
//			*type = token;
//		}
//		else if (previousToken.compare("number") == 0) {
//			*number = std::stoi(token);
//		}
//		line.erase(0, pos + 1);
//	}
//}
//
//void readConfigPoperties(char **type, int *number) {
//	fstream newfile;
//	newfile.open("config.ini", ios::in); //open a file to perform read operation using file object
//	if (newfile.is_open()) {   //checking whether the file is open
//		string tp;
//		while (getline(newfile, tp)) { //read data from file object and put it into string.
//			processLine(tp, type, number);
//		}
//		newfile.close(); //close the file object.
//	}
//}