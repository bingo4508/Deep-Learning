#ifndef NN_H
#define NN_H

#include "armadillo"
#include <vector>
#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>
#include "utility.h"

using namespace arma;
using namespace std;

class NNet {
        public:
                /* Parameters */
                double learning_rate;
                double learning_rate_decay;
                int batch_size;

                /* Network data structure */
                vector<mat> data;
                vector<int> label;
                vector<int> index;

                /* Constructors */
                NNet();
                NNet(string);

                /* Load data  */
                void load_model(vector<int>);
                void load_model(string);
                void load_train_data(string, vector<mat>&, vector<int>&, vector<int>&);
                void save_model(string, string);

                /* Core algorithms */
                int feedforward(mat);
                void backprop(mat);
                void update();

                /* Prediction and validation */
                void predict(string,string,int);
                float report_error_rate(vector<mat>&,vector<int>&,vector<int>&);
        protected:
                /* Network data structure */
                vector<mat> weights;
                vector<mat> bias;
                vector<mat> inputs;
                vector<mat> outputs;
                vector<mat> deltas;

                bool batch_start;

                /* Activation function */
                mat sigmoid_mat(mat m);
                mat sigmoid_prime_mat(mat m);
                double sigmoid(double x);
                double sigmoid_prime(double x);

                int max(mat&);

		private:
				mat ReLU_mat(mat m);
				mat ReLU_prime_mat(mat m);
				double ReLU(double x);
				double ReLU_prime(double x);

				mat softmax_mat(mat m);
};

#endif
