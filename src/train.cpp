#include <iostream>
#include <algorithm>
#include <stdio.h>
#include "Net.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);

using namespace std;


int main(int argc, char** argv){
	double learning_rate;
	int batch_size;
	vector<int> layers;
	string train_fname;
	int max_epoch;
	float valid_ratio=0.1;
	string output_model;
	string structure;
	Net d;

	//Set parameters
	//ex: ./run 0.01 5-4-3 300 train_file_name output_model_name model_name
	if(argc < 7){
		printf("Usage:\n");
		printf("./train learning_rate(0.01) batch_size(10) structure(5-4-3) max_epoch(100) train_file output_model [load_model]\n");
		return 0;
	}else{
		learning_rate = atof(argv[1]);
		batch_size = atoi(argv[2]);
		string lyr(argv[3]);
		structure = lyr.c_str();
		vector<string> x = split(lyr,"-");
		for(int i=0;i<x.size();i++){
			layers.push_back(atoi(x[i].c_str()));
		}
		max_epoch = atoi(argv[4]);
		train_fname.assign(argv[5]);
		output_model.assign(argv[6]);
	}

	//Initialize neural network
	if(argc == 7){
		d.load_model(layers);
	}else if(argc == 8){
		string m_name(argv[7]);
		d.load_model(m_name);
	}else{
		printf("wrong parameters\n");
		return 0;
	}
	d.learning_rate = learning_rate;
	d.batch_size = batch_size;


	//Loading data
	puts("Loading training data...");
	d.load_train_data(train_fname,d.data,d.label,d.index);
	vector<int> valid_index(d.index.begin(), d.index.begin()+d.index.size()*valid_ratio);
	vector<int> train_index(d.index.begin()+d.index.size()*valid_ratio, d.index.end());

	//Training
	puts("Start training...");
	for(int epoch=0;epoch<max_epoch;epoch++){
		random_shuffle(train_index.begin(), train_index.end());
		int j=0;
		for(vector<int>::iterator it=train_index.begin();it!=train_index.end();++it,++j){
			mat y = zeros<mat>(layers.back(),1);
			d.feedforward(d.data[*it]);
			y(d.label[*it],0) = 1;
			d.backprop(y);
			if((j % d.batch_size == 0) && j != 0)
				d.update();
		}
		float train_err = d.report_error_rate(d.data,d.label, train_index);
		float valid_err = d.report_error_rate(d.data,d.label, valid_index);
		printf("epoch %d\ttrain err:%f\tvalid err:%f\n", epoch,train_err,valid_err);
		d.save_model(output_model, structure);
	}
}
