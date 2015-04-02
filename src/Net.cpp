#include "Net.h"

void print_size(mat &m){
	printf("(%d,%d)\n",m.n_rows,m.n_cols);
}

//m: Nx1 matrix
int Net::max(mat &m){
	uword index;
	m.max(index);
	return index;
}

Net::Net(vector<int> layers, double learning_rate){
	//Set learning rate
	this->learning_rate = learning_rate;

	//Initialize weights
	for(int i=1;i<layers.size();i++){
		mat W = 2*randu<mat>(layers[i], layers[i-1])-1;
		mat B = 2*randu<mat>(layers[i], 1)-1;
		this->weights.push_back(W);
		this->bias.push_back(B);
	}
}

//input: Nx1
int Net::feedforward(mat input){
	this->outputs.clear();
	this->inputs.clear();
	this->inputs.push_back(input);
	this->outputs.push_back(input);

	for(int i=0;i<this->weights.size();i++){
		mat Z = (this->weights[i] * this->outputs[i]) + this->bias[i];
		this->inputs.push_back(Z);
		mat A = this->sigmoid_mat(Z);
		this->outputs.push_back(A);
	}
	return this->max(this->outputs.back());	
}

void Net::backprop(mat y){
	mat error = this->outputs.back() - y;
	
	this->deltas.clear();
	this->deltas.push_back(error % this->sigmoid_prime_mat(this->outputs.back()));	
	//%: element-wise dot
	
	for(int i=this->outputs.size()-2;i>0;i--){
		mat m = this->sigmoid_prime_mat(this->inputs[i]) % (this->weights[i].t()*this->deltas.back());
		this->deltas.push_back(m);
	}
}

void Net::update(){
	int last = this->weights.size()-1;
	for(int i=0;i<=last;i++){
		this->weights[i] -= this->learning_rate*(this->deltas[last-i]*this->outputs[i].t());
		this->bias[i] -= this->learning_rate*deltas[last-i];
	}
}

double Net::sigmoid(double x){
	return 1/(1+exp(-x));
}

double Net::sigmoid_prime(double x){
	double y = sigmoid(x);
	return y*(1.0-y);
}

mat Net::sigmoid_mat(mat m){
	for(mat::iterator i=m.begin();i!=m.end();i++){
		*i = sigmoid(*i);
	}
	return m;
}
mat Net::sigmoid_prime_mat(mat m){
	for(mat::iterator i=m.begin();i!=m.end();i++){
		*i = sigmoid_prime(*i);
	}
	return m;
}

//Format: sample_name feature1(double) feature2 ... label(int)
void Net::load(string fname, vector<mat> &data, vector<int> &label, vector<int> &index){
	fstream fin;
	ifstream input(fname.c_str(), ifstream::in);

	for(string line; getline(input, line);)
	{
		vector<string> x = split(line, " ");
		int feature_size = x.size()-2;
		mat feature(feature_size, 1);

		for(int i=1;i<x.size()-1;i++){
			feature(i-1, 0) = atof(x[i].c_str());
		}
		data.push_back(feature);
		label.push_back(atoi(x[x.size()-1].c_str()));
	}
	for(int i=0;i<data.size();i++){
		index.push_back(i);
	}
	return;
}

float Net::report_error_rate(vector<mat>& data, vector<int> &label){
	float err = 0;
	for(int i=0;i<data.size();i++){
		int p = this->feedforward(data[i]);
		if(label[i] != p){
			err ++;
		}
	}
	return err/data.size();
}

