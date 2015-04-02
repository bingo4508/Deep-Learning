#include <iostream>
#include <algorithm>
#include <stdio.h>
#include "Net.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;


int main(int argc, char** argv){
	double learning_rate;
	vector<int> layers;
	string train_fname;
	int max_epoch;

	//Set parameters
	//ex: ./run 0.01 5-4-3 300 train_file_name
	if(argc != 5){
		puts("Wrong parameters!");
		return 0;
	}else{
		learning_rate = atof(argv[1]);
		string lyr(argv[2]);
		vector<string> x = split(lyr,"-");
		for(int i=0;i<x.size();i++){
			layers.push_back(atoi(x[i].c_str()));
		}
		max_epoch = atoi(argv[3]);
		train_fname.assign(argv[4]);
	}

	//Initialize neural network
	Net d(layers, learning_rate);
	
	puts("Loading training data...");
	d.load(train_fname,d.data,d.label,d.index);

	//Training
	puts("Start training...");
	for(int epoch=0;epoch<max_epoch;epoch++){
		random_shuffle(d.index.begin(), d.index.end());
		for(vector<int>::iterator it=d.index.begin();it!=d.index.end();++it){
			mat y = zeros<mat>(layers.back(),1);
			d.feedforward(d.data[*it]);
			y(d.label[*it],0) = 1;
			d.backprop(y);
			d.update();
		}
		float train_err = d.report_error_rate(d.data,d.label);
		printf("epoch %d\ttrain err:%f\n", epoch,train_err);
	}
}
