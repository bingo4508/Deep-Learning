#ifndef RNN_H
#define RNN_H

#include "armadillo"
#include <vector>
#include <deque>
#include <map>
#include <math.h>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <stdlib.h>
#include "utility.h"
#include "nnet.h"

using namespace arma;
using namespace std;

class RNNet: public NNet{
	public:
		map<string, mat> map_vec;
		map<string, int> map_class;
		map<int, string> map_class2;
		map<int, int> map_cluster;
		vector<deque<int>> map_cluster_i;
		int numClusters;
		vector<string> data_text;

                void feedforward(mat);
				void feedforwardF(mat);
                void backprop(mat);
				void backpropF(mat, int);
                void update();
				void updateF(int);

		void load_model(vector<int>);
		void load_model(string);
		void save_model(string, string);
		void predict(string fname, string fvec, string fclass, string oname, map<string, mat> &map_vec, int n_choice, char symbol_choice);
		void load_train_data(string, string, string, map<string, mat>&, vector<string>&, vector<int>&);
		

		int back_t;
	private:
		vector<mat> mem_weights;
		vector<deque<mat> > mem;
		vector<deque<mat> > mem_deltas;
		vector<deque<mat> > mem_inputs;
		vector<deque<mat> > mem_outputs;

	protected:
		/* Activation function */
		mat ReLU_mat(mat m);
		mat ReLU_prime_mat(mat m);
		double ReLU(double x);
		double ReLU_prime(double x);

		mat softmax_mat(mat m, bool fac = false);

		mat subMatCopy(mat& m, int);
		void subMatCopy(mat source, mat &dest, int);
};

#endif
