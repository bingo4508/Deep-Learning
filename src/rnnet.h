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
		vector<string> data_text;

                void feedforward(mat);
                void backprop(mat);
                void update();

		void load_model(vector<int>);
		void load_model(string);
		void save_model(string, string);
		void predict(string, string, int);
		void load_train_data(string, string, string, map<string, mat>&, vector<string>&, vector<int>&);

		int back_t;
	private:
		vector<mat> mem_weights;
		vector<deque<mat> > mem;
		vector<deque<mat> > mem_deltas;
		vector<deque<mat> > mem_inputs;
		vector<deque<mat> > mem_outputs;
};

#endif
