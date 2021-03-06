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

Net::Net(){
	this->batch_start = false;
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
	vector<mat> delta;
	mat error = this->outputs.back() - y;
	mat D = error % this->sigmoid_prime_mat(this->outputs.back());

	if(!this->batch_start){
		this->deltas.clear();
		this->deltas.push_back(D);
	}else{
		this->deltas[0] += D;
	}
	delta.push_back(D);

	//%: element-wise do
	for(int i=this->outputs.size()-2, j=1;i>0;i--,j++){
		mat m = this->sigmoid_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
		delta.push_back(m);

		if(!this->batch_start)
			this->deltas.push_back(m);
		else
			this->deltas[j] += m;
	}
	if(!this->batch_start)
		this->batch_start = true;
}

void Net::initDeltaRBM(int layer)
{
	delta_w = zeros(this->weights[layer].n_rows, this->weights[layer].n_cols);
	delta_b = zeros(this->bias[layer].n_rows, this->bias[layer].n_cols);
	delta_back_b = zeros(this->back_bias[layer].n_rows, this->back_bias[layer].n_cols);
}

void Net::gibbSample(int l, mat v)
{
	if (l == 0)
	{
		vd = v;

		hd = (this->weights[l] * vd) + this->bias[l];
		hd = this->sigmoid_mat(hd);
		for (mat::iterator i = hd.begin(); i != hd.end(); i++){
			*i = ((double)rand()/RAND_MAX <= (*i) ? 1 : 0);
		}

		vm = (this->weights[l].t() * hd) + this->back_bias[l];
		vm = this->gaussian_mat(vd, vm, 1);
		for (mat::iterator i = vm.begin(); i != vm.end(); i++){
			*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
		}

		hm = (this->weights[l] * vm) + this->bias[l];
		hm = this->sigmoid_mat(hm);
		for (mat::iterator i = hm.begin(); i != hm.end(); i++){
			*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
		}
	}
	else
	{
		//feed forward until the current layer
		vd = v;
		for (int i = 0; i < l; i++)
		{
			vd = (this->weights[i] * vd) + this->bias[i];
			vd = this->sigmoid_mat(vd);
			for (mat::iterator i = vd.begin(); i != vd.end(); i++){
				*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
			}
		}

		hd = (this->weights[l] * vd) + this->bias[l];
		hd = this->sigmoid_mat(hd);
		for (mat::iterator i = hd.begin(); i != hd.end(); i++){
			*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
		}

		vm = (this->weights[l].t() * hd) + this->back_bias[l];
		vm = this->sigmoid_mat(vm);
		for (mat::iterator i = vm.begin(); i != vm.end(); i++){
			*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
		}

		hm = (this->weights[l] * vm) + this->bias[l];
		hm = this->sigmoid_mat(hm);
		for (mat::iterator i = hm.begin(); i != hm.end(); i++){
			*i = ((double)rand() / RAND_MAX <= *i ? 1 : 0);
		}
	}
	delta_w += hd*vd.t() - hm*vm.t();
	delta_b += hd - hm;
	delta_back_b += vd - vm;
}

void Net::update(){
	this->batch_start = false;
	int last = this->weights.size()-1;

	for(int i=0;i<=last;i++){
		this->weights[i] -= this->learning_rate*((this->deltas[last-i]/this->batch_size)*this->outputs[i].t());
		this->bias[i] -= this->learning_rate*(this->deltas[last-i]/this->batch_size);
	}
}

void Net::updateRBM(int l)
{
	this->batch_start = false;
	this->weights[l] += this->learning_rate*(this->delta_w / this->batch_size);
	this->bias[l] += this->learning_rate*(this->delta_b / this->batch_size);
	this->back_bias[l] += this->learning_rate*(this->delta_back_b / this->batch_size);

	cout << norm(delta_b, 1) + norm(delta_back_b) << endl;
	//cout << bias[0];

	delta_w.fill(0);
	delta_b.fill(0);
	delta_back_b.fill(0);
}


/***********************************************************************/
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
double Net::gaussian(double x, double mean, double sigma)
{
	return 1 / (sigma*sqrt(2 * math::pi())) * exp(-(x - mean)*(x - mean) / (2 * sigma*sigma));
}
mat Net::gaussian_mat(mat m, mat mean, double sigma)
{
	mat::iterator i = m.begin();
	mat::iterator im = mean.begin();
	while (i != m.end())
	{
		*i = gaussian(*i, *im, sigma);
		i++;
		im++;
	}
	return m;
}

//Format: sample_name feature1(double) feature2 ... label(int)
void Net::load_train_data(string fname, vector<mat> &data, vector<int> &label, vector<int> &index){
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


void Net::load_model(vector<int> layers){
	//Initialize weights
	for(int i=1;i<layers.size();i++){
		mat W = 2*randu<mat>(layers[i], layers[i-1])-1;
		mat B = 2*randu<mat>(layers[i], 1)-1;
		mat B2 = 2*randu<mat>(layers[i - 1], 1)-1;
		this->weights.push_back(W);
		this->bias.push_back(B);
		this->back_bias.push_back(B2);
	}
}

void Net::load_model(string fname){
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

void Net::save_model(string fname, string structure){
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

void Net::predict(string fname, string oname){
	fstream fin;
	ifstream fi(fname.c_str(), ifstream::in);
	ofstream fo(oname.c_str());

	for(string line; getline(fi, line);){
		vector<string> x = split(line, " ");
		int feature_size = x.size()-1;
		mat feature(feature_size, 1);

		for(int i=1;i<x.size()-1;i++){
			feature(i-1, 0) = atof(x[i].c_str());
		}
		int p = this->feedforward(feature);
		//Output result
		fo << x[0] << "," << p << "\n";
	}
	return;
}

float Net::report_error_rate(vector<mat>& data, vector<int> &label, vector<int> &index){
	float err = 0;
	for(vector<int>::iterator it=index.begin();it!=index.end();++it){
		int p = this->feedforward(data[*it]);
		if(label[*it] != p){
			err ++;
		}
	}
	return err/index.size();
}

