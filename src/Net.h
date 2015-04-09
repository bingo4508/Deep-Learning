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

class Net {
	public:
		vector<mat> data;
		vector<int> label;
		vector<int> index;
		double learning_rate;
		int batch_size;

		Net();
		void load_model(vector<int>);
		void load_model(string);
		Net(string);
	        void load_train_data(string, vector<mat>&, vector<int>&, vector<int>&);
		void save_model(string, string);

		int feedforward(mat);
		void backprop(mat);
		void update();

		void predict(string,string,int);
		float report_error_rate(vector<mat>&,vector<int>&,vector<int>&);
	private:
		vector<mat> weights;
		vector<mat> bias;
		vector<mat> inputs;
		vector<mat> outputs;
		vector<mat> deltas;

		bool batch_start;

		mat sigmoid_mat(mat m);
		mat sigmoid_prime_mat(mat m);
		double sigmoid(double x);
		double sigmoid_prime(double x);
		int max(mat&);
};
