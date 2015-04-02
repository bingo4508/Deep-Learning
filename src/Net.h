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

		Net(vector<int> layers, double learning_rate);
	        void load(string, vector<mat>&, vector<int>&, vector<int>&);
		int feedforward(mat);
		void backprop(mat);
		void update();
		float report_error_rate(vector<mat>&,vector<int>&);

	private:
		double learning_rate;

		vector<mat> weights;
		vector<mat> bias;
		vector<mat> inputs;
		vector<mat> outputs;
		vector<mat> deltas;


		mat sigmoid_mat(mat m);
		mat sigmoid_prime_mat(mat m);
		double sigmoid(double x);
		double sigmoid_prime(double x);
		int max(mat&);
};
