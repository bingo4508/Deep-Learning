#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "rnnet.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;


int main(int argc, char** argv){
	double learning_rate;
	vector<int> layers;
	string test_fname;
	string train_fvec;
	string train_fclass;
	string model;
	string result;
	int has_answer;
	RNNet d;

	//Set parameters
	//ex: ./predict test_file_name model_name output_name has_answer
	if(argc == 6){
		test_fname.assign(argv[1]);
		train_fvec.assign(argv[2]);
		train_fclass.assign(argv[3]);
		model.assign(argv[4]);
		result.assign(argv[5]);
	}else{
		printf("Usage:\n");
		printf("./predict test_file train.vec train.class model_name output_name\n");
		return 0;
	}
	

	//Initialize neural network
	d.load_model(model);

	puts("Start predicting...");
	d.predict(test_fname,train_fvec,train_fclass,result,d.map_vec, 5, '[');

	return 0;
}
