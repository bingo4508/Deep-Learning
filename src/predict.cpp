#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "Net.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;


int main(int argc, char** argv){
	double learning_rate;
	vector<int> layers;
	string test_fname;
	string model;
	string result;
	int has_answer;
	Net d;

	//Set parameters
	//ex: ./predict test_file_name model_name output_name has_answer
	if(argc == 5){
		test_fname.assign(argv[1]);
		model.assign(argv[2]);
		result.assign(argv[3]);
		has_answer = atoi(argv[4]);
	}else{
		printf("Usage:\n");
		printf("./predict test_file model_name output_name has_answer(1/0)\n");
		return 0;
	}
	

	//Initialize neural network
	d.load_model(model);

	puts("Start predicting...");
	d.predict(test_fname,result,has_answer);

	return 0;
}
