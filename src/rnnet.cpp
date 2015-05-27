#include "rnnet.h"

#define SOFTMAX
//#define RELU

//#define DEBUG

//input: Nx1
void RNNet::feedforward(mat input){
#ifdef DEBUG
	puts("forward");
#endif
        this->outputs.clear();
        this->inputs.clear();
        this->inputs.push_back(input);
        this->outputs.push_back(input);
        // Hidden layers with memory
	int i;
        for(i=0;i<this->weights.size()-1;i++){
                mat Z = (this->weights[i] * this->outputs[i])+(this->mem_weights[i]*this->mem[i].back())+ this->bias[i];
#ifdef RELU
                mat A = this->ReLU_mat(Z);
#else
				mat A = this->sigmoid_mat(Z);
#endif
                this->inputs.push_back(Z);
                this->outputs.push_back(A);
        }
        // Output layer
        mat Z = (this->weights[i] * this->outputs[i]) + this->bias[i];
#ifdef SOFTMAX
        mat A = this->softmax_mat(Z);
#else
		mat A = this->sigmoid_mat(Z);
#endif
        this->inputs.push_back(Z);
        this->outputs.push_back(A);

        return;
}

void RNNet::backprop(mat y){
#ifdef DEBUG
	puts("backprop");
#endif
        vector<mat> delta;
        mat error = this->outputs.back() - y;
#ifdef SOFTMAX
		mat D = error;
#else
		mat D = error % this->sigmoid_prime_mat(this->inputs.back());
#endif

        if(!this->batch_start){
                this->deltas.clear();
                this->deltas.push_back(D);
        }else{
                this->deltas[0] += D;
        }
        delta.push_back(D);
        //%: element-wise dot
        for(int i=this->outputs.size()-2, j=1;i>0;i--,j++){
#ifdef RELU
                mat m = this->ReLU_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#else
				mat m = this->sigmoid_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#endif
                delta.push_back(m);

		// Propagate delta to memory deltas
		while(!this->mem_deltas[i-1].empty()) this->mem_deltas[i-1].pop_front();
		this->mem_deltas[i-1].push_back(m);
		int size = this->mem_inputs[i-1].size()-1;
		for(int k=0; k<=size; k++){
#ifdef RELU
			mat mm = this->ReLU_prime_mat(this->mem_inputs[i-1][size-k]) % (this->mem_weights[i-1] * this->mem_deltas[i-1].back());
#else
			mat mm = this->sigmoid_prime_mat(this->mem_inputs[i-1][size-k]) % (this->mem_weights[i-1] * this->mem_deltas[i-1].back());
#endif
			this->mem_deltas[i-1].push_back(mm);
		}
		this->mem_deltas[i-1].pop_front();	//pop current delta
		// Batching
                if(!this->batch_start)
                        this->deltas.push_back(m);
                else
                        this->deltas[j] += m;
        }
        if(!this->batch_start)
                this->batch_start = true;
	delta.clear();
}

void RNNet::update(){
#ifdef DEBUG
	puts("update");
#endif
        this->batch_start = false;
        int last = this->weights.size()-1;

	// Update output layer
	this->weights[last] -= this->learning_rate*((this->deltas[0]/this->batch_size)*this->outputs[last].t());
        this->bias[last] -= this->learning_rate*(this->deltas[0]/this->batch_size);

	// Update Hidden layers
        for(int i=0;i<last;i++){
		int delta_s = this->mem_deltas[i].size()-1;
		// Weights
                mat tmp = this->deltas[last-i]*this->outputs[i].t();
		for(int j=0;j<=delta_s;j++)
			tmp += this->mem_deltas[i][delta_s-j]*this->mem_outputs[i][j].t();
		this->weights[i] -= this->learning_rate*(tmp/batch_size);
		// Biases
		tmp = this->deltas[last-i];
		for(int j=0;j<=delta_s;j++)
			tmp += this->mem_deltas[i][j];
                this->bias[i] -= this->learning_rate*(tmp/batch_size);
		// Update memory
		tmp = (this->deltas[last-i]/this->batch_size)*this->mem[i].back().t();
		for(int j=0;j<=delta_s;j++)
			tmp += this->mem_deltas[i][delta_s-j]*this->mem[i][j].t();
                this->mem_weights[i] -= this->learning_rate*tmp;
        }
	// Replace memory with current hidden layer state
	for(int i=0;i<last;i++){
		this->mem[i].push_back(this->outputs[i+1]);
		this->mem_inputs[i].push_back(this->inputs[i+1]);
		this->mem_outputs[i].push_back(this->outputs[i]);
		if(this->mem[i].size() > this->back_t){
			this->mem[i].pop_front();
			this->mem_deltas[i].pop_front();
			this->mem_inputs[i].pop_front();
			this->mem_outputs[i].pop_front();
		}
	}
}

/***********************************************************************/
double RNNet::ReLU(double x)
{
	return x > 0 ? x : 0;
}
double RNNet::ReLU_prime(double x)
{
	return x > 0 ? 1 : 0;
}
mat RNNet::ReLU_mat(mat m)
{
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i = ReLU(*i);
	}
	return m;
}
mat RNNet::ReLU_prime_mat(mat m)
{
	for (mat::iterator i = m.begin(); i != m.end(); i++){
		*i = ReLU_prime(*i);
	}
	return m;
}

mat RNNet::softmax_mat(mat m)
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

void RNNet::load_train_data(string ftext, string fvec, string fclass, map<string, mat> &map_vec, vector<string> &data_text, vector<int> &index){
        ifstream input_vec(fvec.c_str(), ifstream::in);
        ifstream input_text(ftext.c_str(), ifstream::in);
        ifstream input_class(fclass.c_str(), ifstream::in);
        int n_sample,n_feature;

        string line;
        getline(input_vec, line);
        vector<string> x = split(line, " ");
        n_sample = atoi(x[0].c_str());
        n_feature = atoi(x[1].c_str());

	// word vectors
        for(; getline(input_vec, line);){
                vector<string> x = split(line, " ");
                mat feature(n_feature, 1);

                for(int i=1;i<x.size()-1;i++){
                        feature(i-1, 0) = atof(x[i].c_str());
                }
                map_vec[x[0]] = feature;
        }
	// text
	for(; getline(input_text, line);){
		vector<string> x = split(line, " ");
		data_text.insert(data_text.end(), x.begin(), x.end());
	}
	// class of words
	for(int i=0; getline(input_class, line);i++){
		map_class[line] = i;
	}

        for(int i=0;i<data_text.size();i++){
                index.push_back(i);
        }
        return;
}


void RNNet::load_model(vector<int> layers){
	this->mem.resize(layers.size()-2);
	this->mem_deltas.resize(layers.size()-2);
	this->mem_inputs.resize(layers.size()-2);
	this->mem_outputs.resize(layers.size()-2);

	layers[layers.size()-1] += 1;	//For "Others"

	//Initialize weights
	for(int i=1;i<layers.size();i++){
		mat W = 2*randu<mat>(layers[i], layers[i-1])-1;
		mat B = 2*randu<mat>(layers[i], 1)-1;

		this->weights.push_back(W);
		this->bias.push_back(B);
	}
	//Initialize RNN memory for hidden layers
	for(int i=1;i<layers.size()-1;i++){
		mat W = 2*randu<mat>(layers[i], layers[i])-1;
		mat M = 2*randu<mat>(layers[i], 1)-1;

		this->mem_weights.push_back(W);
		this->mem[i-1].push_back(M);
	}
}

void RNNet::load_model(string fname){
	ifstream fi(fname.c_str(), ifstream::in);
	string line;
	vector<int> layers;

	getline(fi, line);
   	vector<string> x = split(line,"-");
	for(int i=0;i<x.size();i++){
		layers.push_back(atoi(x[i].c_str()));
	}
	layers[layers.size()-1] += 1; //For "Others"

	this->mem.resize(layers.size()-2);
	this->mem_deltas.resize(layers.size()-2);
	this->mem_inputs.resize(layers.size()-2);
	this->mem_outputs.resize(layers.size()-2);


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

		//Only hidden layer has memory
		if(i < layers.size()-1){
			mat MW(layers[i], layers[i]);
			mat M(layers[i], 1);

			//Mem weights
			for(int j=0;j<layers[i];j++){
				getline(fi, line);
				vector<string> x = split(line, " ");
				for(int k=0;k<x.size();k++){
					MW(j,k) = atof(x[k].c_str());
				}
			}
			//Mem of layer
			for(int j=0;j<layers[i];j++){
				getline(fi, line);
				M(j,0) = atof(line.c_str());
			}

			this->mem_weights.push_back(MW);
			this->mem[i-1].push_back(M);
		}

		this->weights.push_back(W);
		this->bias.push_back(B);
//		printf("(%d,%d)\n",W.n_rows,W.n_cols);
//		printf("(%d,%d)\n",B.n_rows,B.n_cols);
	}
}

void RNNet::save_model(string fname, string structure){
	ofstream fo(fname.c_str());

	fo << structure << "\n";
	for(int k=0;k<this->weights.size();k++){
		int n_rows = this->weights[k].n_rows;
		int n_cols = this->weights[k].n_cols;

		//Weights
		for(int i=0;i<n_rows;i++){
			for(int j=0;j<n_cols;j++)
				fo << this->weights[k](i,j) << " ";
			fo << "\n";
		}
		//Bias
		for(int i=0;i<n_rows;i++)
			fo << this->bias[k](i,0) << "\n";
		
		//Only hidden layer has memory
		if(k < this->weights.size()-1){
			//Mem weights
			for(int i=0;i<n_rows;i++){
				for(int j=0;j<n_rows;j++)
					fo << this->mem_weights[k](i,j) << " ";
				fo << "\n";
			}
			//Mem of layer
			for(int i=0;i<n_rows;i++)
				fo << this->mem[k].back()(i,0) << "\n";
		}	
	}
}

void RNNet::predict(string fname, string oname, int has_answer){
	fstream fin;
	ifstream fi(fname.c_str(), ifstream::in);
	ofstream fo(oname.c_str());

	int correct = 0;
	int i=0;
	for(string line; getline(fi, line);i++){
		//TODO
	}
	if(has_answer)
		printf("Accuracy: %f\n",float(correct)/i);

	return;
}


