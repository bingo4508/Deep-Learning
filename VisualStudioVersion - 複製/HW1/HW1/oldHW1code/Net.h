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

		void gibbSample(int, mat);
		void updateRBM(int layer);
		void initDeltaRBM(int layer);

		void predict(string,string);
		float report_error_rate(vector<mat>&,vector<int>&,vector<int>&);
	private:
		vector<mat> weights;
		vector<mat> bias;
		vector<mat> inputs;
		vector<mat> outputs;
		vector<mat> deltas;

		mat vd;
		mat hd;
		mat vm;
		mat hm;

		mat delta_w;
		mat delta_b;
		mat delta_back_b;
		
		vector<mat> back_bias;

		bool batch_start;

		mat sigmoid_mat(mat m);
		mat sigmoid_prime_mat(mat m);
		double sigmoid(double x);
		double sigmoid_prime(double x);
		double gaussian(double x, double mean, double sigma);
		mat gaussian_mat(mat x, mat mean, double sigma);
		int max(mat&);
};
