#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "Net.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;


int ___main(int argc, char** argv){
	double learning_rate;
	vector<int> layers;
	string test_fname;
	string model;
	string result;
	Net d;

	//Set parameters
	//ex: ./predict test_file_name model_name output_name
	assert(argc == 4);
	test_fname.assign(argv[1]);
	model.assign(argv[2]);
	result.assign(argv[3]);
	

	//Initialize neural network
	d.load_model(model);

	puts("Start predicting...");
	d.predict(test_fname,result);

	return 0;
}
