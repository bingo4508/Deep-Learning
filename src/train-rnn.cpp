#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <time.h>
#include "nnet.h"
#include "rnnet.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define GET_SECS(t1,t2) (t2-t1)/(double)(CLOCKS_PER_SEC)

using namespace std;


int main(int argc, char** argv){
        clock_t t1, t2;
        vector<int> layers;
        string train_fname;
        int max_epoch;
        float valid_ratio=0;
        string output_model;
        string structure;
        RNNet d;

        //Set parameters
        //ex: ./run 0.01 5-4-3 300 train_file_name output_model_name model_name
        if(argc < 9){
                printf("Usage:\n");
                printf("./train learning_rate(0.01) learning_rate_decay(0.8) batch_size(10) rnn_depth(5) structure(5-4-3) max_epoch(100) train_file output_model [load_model]\n");
                return 0;
        }else{
                d.learning_rate = atof(argv[1]);
                d.learning_rate_decay = atof(argv[2]);
                d.batch_size = atoi(argv[3]);
                string lyr(argv[4]);
                structure = lyr.c_str();
                vector<string> x = split(lyr,"-");
                for(int i=0;i<x.size();i++){
                        layers.push_back(atoi(x[i].c_str()));
                }
		layers[layers.size()-1];
                max_epoch = atoi(argv[5]);
		d.back_t = atoi(argv[6]);
                train_fname.assign(argv[7]);
                output_model.assign(argv[8]);
        }
        //Initialize neural network
        if(argc == 9){
		puts("Initializing model...");
                d.load_model(layers);
        }else if(argc == 10){
		printf("Load model from %s\n",argv[9]);
                string m_name(argv[9]);
                d.load_model(m_name);
        }else{
                printf("wrong parameters\n");
                return 0;
        }
        //Loading data
        puts("Loading training data...");
        t1 = clock();
        d.load_train_data(train_fname+".text", train_fname+".vec", train_fname+".class", d.map_vec, d.data_text, d.index);
        t2 = clock();
        printf("spent %f secs\n", GET_SECS(t1,t2));

        vector<int> valid_index(d.index.begin(), d.index.begin()+d.index.size()*valid_ratio);
        vector<int> train_index(d.index.begin()+d.index.size()*valid_ratio, d.index.end());

        //Training
        puts("Start training...");
        for(int epoch=0;epoch<max_epoch;epoch++){
                t1 = clock();
		printf("eopch %d: ", epoch);
		int one_percent = train_index.size()*0.01;
                for(int i=0;i<train_index.size()-1;i++){
			if(i % one_percent == 0){
				printf(".");
				fflush(stdout);
		                d.save_model(output_model, structure);
			}
			if(d.map_vec.find(d.data_text[train_index[i]]) != d.map_vec.end() &&
			   d.map_vec.find(d.data_text[train_index[i+1]]) != d.map_vec.end()){
				d.feedforward(d.map_vec[d.data_text[train_index[i]]]);
				mat y = zeros<mat>(layers.back()+1,1);
				if(d.map_class.find(d.data_text[train_index[i+1]]) != d.map_class.end())
					y(d.map_class[d.data_text[train_index[i+1]]],0) = 1;
				else
					y(layers.back(),0) = 1;	//Others
				d.backprop(y);
				// Batch has bug, need fix
				if((i % d.batch_size == 0))
					d.update();
			}
                }
/*                float train_err = d.report_error_rate(d.data,d.label, train_index);
                float valid_err = d.report_error_rate(d.data,d.label, valid_index);

                t2 = clock();
                printf("epoch %d\ttrain err:%f\tvalid err:%f\t%f secs\n", epoch,train_err,valid_err,GET_SECS(t1,t2));
		*/
		printf("\n");
                d.save_model(output_model, structure);
                d.learning_rate *= d.learning_rate_decay;
        }
}

