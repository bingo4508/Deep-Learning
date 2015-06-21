#include "nnet.h"

#define SOFTMAX
#define RELU

void print_size(mat &m){
	printf("(%d,%d)\n",m.n_rows,m.n_cols);
}

//m: Nx1 matrix
int NNet::max(mat &m){
	uword index;
	m.max(index);
	return index;
}

NNet::NNet(){
	batch_start = false;
}

//input: Nx1
int NNet::feedforward(mat input){
	this->outputs.clear();
	this->inputs.clear();
	this->inputs.push_back(input);
	this->outputs.push_back(input);
	int i;
	for(i=0;i<this->weights.size()-1;i++){
		mat Z = (this->weights[i] * this->outputs[i]) + this->bias[i];
		this->inputs.push_back(Z);
#ifdef RELU
		mat A = this->ReLU_mat(Z);
#else
		mat A = this->sigmoid_mat(Z);
#endif
		this->outputs.push_back(A);
	}
	//output layer
	mat Z = (this->weights[i] * this->outputs[i]) + this->bias[i];
	this->inputs.push_back(Z);
#ifdef SOFTMAX
	mat A = this->softmax_mat(Z);
#else
	mat A = this->sigmoid_mat(Z);
#endif
	this->outputs.push_back(A);
	return this->max(this->outputs.back());	
}

void NNet::backprop(mat y){
	vector<mat> delta;
	mat error = this->outputs.back() - y;
#ifdef SOFTMAX
	mat D = error;
#else
	mat D = error % this->sigmoid_prime_mat(this->inputs.back());
#endif

	this->deltas.clear();
	this->deltas.push_back(D);
	delta.push_back(D);

	//%: element-wise do
	for(int i=this->outputs.size()-2, j=1;i>0;i--,j++){
#ifdef RELU
		mat m = this->ReLU_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#else
		mat m = this->sigmoid_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#endif
		delta.push_back(m);

		this->deltas.push_back(m);
	}
}

void NNet::update(int iter){
	int last = this->weights.size()-1;

	if(iter != 0 && iter % this->batch_size == 0){
		//update
		for(int i=0;i<=last;i++){
			this->weights[i] -= this->learning_rate*(this->batch_weight[i]/this->batch_weight.size());
			this->bias[i] -= this->learning_rate*(this->batch_bias[i]/this->batch_bias.size());
		}
		batch_start = false;
		this->batch_weight.clear();
		this->batch_bias.clear();	
	}
	//batching
	for(int i=0;i<=last;i++){
		if(batch_start == false){
			this->batch_weight.push_back(this->deltas[last-i]*this->outputs[i].t());
			this->batch_bias.push_back(this->deltas[last-i]);
		}else{
			this->batch_weight[i] += this->deltas[last-i]*this->outputs[i].t();
			this->batch_bias[i] += this->deltas[last-i];	
		}
	}
	if(batch_start == false)
		batch_start = true;
}


/***********************************************************************/
double NNet::sigmoid(double x){
	return 1/(1+exp(-x));
}

double NNet::sigmoid_prime(double x){
	double y = sigmoid(x);
	return y*(1.0-y);
}

mat NNet::sigmoid_mat(mat m){
	for(mat::iterator i=m.begin();i!=m.end();i++){
		*i = sigmoid(*i);
	}
	return m;
}
mat NNet::sigmoid_prime_mat(mat m){
	for(mat::iterator i=m.begin();i!=m.end();i++){
		*i = sigmoid_prime(*i);
	}
	return m;
}

double NNet::ReLU(double x)
{
	return x > 0 ? x : 0;
}
double NNet::ReLU_prime(double x)
{
	return x > 0 ? 1 : 0;
}
mat NNet::ReLU_mat(mat m)
{
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i = ReLU(*i);
	}
	return m;
}
mat NNet::ReLU_prime_mat(mat m)
{
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i = ReLU_prime(*i);
	}
	return m;
}

mat NNet::softmax_mat(mat m)
{
	double total = 0;
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i = exp(*i);
		total += *i;
	}
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i /= total;
	}
	return m;
}

//Format: sample_name feature1(double) feature2 ... label(int)
void NNet::load_train_data(string fname, vector<mat> &data, vector<int> &label, vector<int> &index){
	fstream fin;
	ifstream input(fname.c_str(), ifstream::in);

	for(string line; getline(input, line);){
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


void NNet::load_model(vector<int> layers){
	//Initialize weights
	for(int i=1;i<layers.size();i++){
		mat W = 2*randu<mat>(layers[i], layers[i-1])-1;
		mat B = 2*randu<mat>(layers[i], 1)-1;
		this->weights.push_back(W);
		this->bias.push_back(B);
	}
}

void NNet::load_model(string fname){
	ifstream fi(fname.c_str(), ifstream::in);
	string line;
	vector<int> layers;

	getline(fi, line);
   	vector<string> x = split(line,"-");
	for(int i=0;i<x.size();i++){
		layers.push_back(atoi(x[i].c_str()));
	}

	for(int i=1;i<layers.size();i++){
		mat W(layers[i], layers[i-1]);
		mat B(layers[i], 1);

		//Weights
		for(int j=0;j<layers[i];j++){
			getline(fi, line);
			vector<string> x = split(line, " ");
			for(int k=0;k<x.size();k++){
				W(j,k) = atof(x[k].c_str());
			}
		}
		//Bias
		for(int j=0;j<layers[i];j++){
			getline(fi, line);
			B(j,0) = atof(line.c_str());
		}

		this->weights.push_back(W);
		this->bias.push_back(B);

//		printf("(%d,%d)\n",W.n_rows,W.n_cols);
//		printf("(%d,%d)\n",B.n_rows,B.n_cols);
	}
}

void NNet::save_model(string fname, string structure){
	ofstream fo(fname.c_str());

	fo << structure << "\n";
	for(int k=0;k<this->weights.size();k++){
		int n_rows = this->weights[k].n_rows;
		int n_cols = this->weights[k].n_cols;

		//Weights
		for(int i=0;i<n_rows;i++){
			for(int j=0;j<n_cols;j++){
				fo << this->weights[k](i,j) << " ";
			}
			fo << "\n";
		}
		//Bias
		for(int i=0;i<n_rows;i++){
			fo << this->bias[k](i,0) << "\n";
		}
	}
}

void NNet::predict(string fname, string oname, int has_answer){
	fstream fin;
	ifstream fi(fname.c_str(), ifstream::in);
	ofstream fo(oname.c_str());
	ofstream fo2((oname+".out_layer").c_str());

	int correct = 0;
	int i=0;
	for(string line; getline(fi, line);i++){
		vector<string> x = split(line, " ");
		int feature_size = x.size()-1-has_answer;
		mat feature(feature_size, 1);

		for(int i=1;i<x.size()-1-has_answer;i++){
			feature(i-1, 0) = atof(x[i].c_str());
		}
		int p = this->feedforward(feature);
		//Output result
		if(has_answer && atoi(x.back().c_str())==p)
			correct++;
		fo << x[0] << "," << p << "\n";
		mat m = this->outputs.back();
		fo2 << x[0] << " ";
		for(mat::iterator i=m.begin();i!=m.end();i++)
		    fo2 << *i << " ";
		fo2 << "\n";
	}
	if(has_answer)
		printf("Accuracy: %f\n",float(correct)/i);

	return;
}

float NNet::report_error_rate(vector<mat>& data, vector<int> &label, vector<int> &index){
	float err = 0;
	for(vector<int>::iterator it=index.begin();it!=index.end();++it){
		int p = this->feedforward(data[*it]);
		if(label[*it] != p){
			err ++;
		}
	}
	return err/index.size();
}

