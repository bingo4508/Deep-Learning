#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <time.h>
#include "nnet.h"

#define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);
#define GET_SECS(t1,t2) (t2-t1)/(double)(CLOCKS_PER_SEC)

using namespace std;


int main(int argc, char** argv){
        clock_t t1, t2;
        vector<int> layers;
        string train_fname;
        int max_epoch;
        float valid_ratio=0.1;
        string output_model;
        string structure;
        NNet d;

        //Set parameters
        //ex: ./run 0.01 5-4-3 300 train_file_name output_model_name model_name
        if(argc < 8){
                printf("Usage:\n");
                printf("./train learning_rate(0.01) learning_rate_decay(0.8) batch_size(10) structure(5-4-3) max_epoch(100) train_file output_model [load_model]\n");
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
                max_epoch = atoi(argv[5]);
                train_fname.assign(argv[6]);
                output_model.assign(argv[7]);
        }
        //Initialize neural network
        if(argc == 8){
                d.load_model(layers);
        }else if(argc == 9){
                string m_name(argv[7]);
                d.load_model(m_name);
        }else{
                printf("wrong parameters\n");
                return 0;
        }
        //Loading data
        puts("Loading training data...");
        t1 = clock();
        d.load_train_data(train_fname,d.data,d.label,d.index);
        t2 = clock();
        printf("spent %f secs\n", GET_SECS(t1,t2));

        vector<int> valid_index(d.index.begin(), d.index.begin()+d.index.size()*valid_ratio);
        vector<int> train_index(d.index.begin()+d.index.size()*valid_ratio, d.index.end());

        //Training
        puts("Start training...");
        for(int epoch=0;epoch<max_epoch;epoch++){
                t1 = clock();

                random_shuffle(train_index.begin(), train_index.end());
                int j=0;
                for(vector<int>::iterator it=train_index.begin();it!=train_index.end();++it,++j){
                        mat y = zeros<mat>(layers.back(),1);
                        d.feedforward(d.data[*it]);
                        y(d.label[*it],0) = 1;
                        d.backprop(y);
                        if((j % d.batch_size == 0))
                                d.update();
                }
                float train_err = d.report_error_rate(d.data,d.label, train_index);
                float valid_err = d.report_error_rate(d.data,d.label, valid_index);

                t2 = clock();
                printf("epoch %d\ttrain err:%f\tvalid err:%f\t%f secs\n", epoch,train_err,valid_err,GET_SECS(t1,t2));
                d.save_model(output_model, structure);
                d.learning_rate *= d.learning_rate_decay;
        }
}

