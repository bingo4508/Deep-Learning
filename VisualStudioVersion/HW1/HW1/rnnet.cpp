#include "rnnet.h"

#define SOFTMAX
//#define RELU

//#define TOO_SMALL 0.0000001
#define TOO_BIG 9999999

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
#ifdef TOO_SMALL
				double norm = arma::norm(m);
				if (norm < TOO_SMALL)
					m = m *0;
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
#ifdef TOO_SMALL
			double norm = arma::norm(mm);
			if (norm < TOO_SMALL)
				mm = mm *0;
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
	int last = this->weights.size() - 1;
	mat *updateW = new mat[last + 1];
	mat *updateB = new mat[last + 1];
	mat *updateM = new mat[last];
	// Update output layer
	//this->weights[last] -= this->learning_rate*((this->deltas[0] / this->batch_size)*this->outputs[last].t());
	//this->bias[last] -= this->learning_rate*(this->deltas[0] / this->batch_size);
	updateW[last] = (this->deltas[0] / this->batch_size)*this->outputs[last].t();
	updateB[last] = this->deltas[0] / this->batch_size;

	// Update Hidden layers
	for (int i = 0; i<last; i++){
		int delta_s = this->mem_deltas[i].size() - 1;
		// Weights
		mat tmp = this->deltas[last - i] * this->outputs[i].t();
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][delta_s - j] * this->mem_outputs[i][j].t();
		//this->weights[i] -= this->learning_rate*(tmp / batch_size);
		updateW[i] = tmp / batch_size;
		// Biases
		tmp = this->deltas[last - i];
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][j];
		//this->bias[i] -= this->learning_rate*(tmp / batch_size);
		updateB[i] = tmp / batch_size;
		// Update memory
		tmp = (this->deltas[last - i] / this->batch_size)*this->mem[i].back().t();
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][delta_s - j] * this->mem[i][j].t();
		//this->mem_weights[i] -= this->learning_rate*tmp;
		updateM[i] = tmp;
	}

	double norm = 0;
	/*norm += arma::norm(updateW[last]);
	norm += arma::norm(updateB[last]);
	for (int i = 0; i < last; i++)
	{
		norm += arma::norm(updateW[i]);
		norm += arma::norm(updateB[i]);
		norm += arma::norm(updateM[i]);
	}
	cout << norm << endl;*/
	double threshold = 100;
	double factor = threshold < norm ? threshold / norm : 1;
	weights[last] -= learning_rate*factor*updateW[last];
	bias[last] -= learning_rate*factor*updateB[last];
	for (int i = 0; i < last; i++)
	{
		weights[i] -= learning_rate*factor*updateW[i];
		bias[i] -= learning_rate*factor*updateB[i];
		mem_weights[i] -= learning_rate*factor*updateM[i];
	}
	delete[] updateW;
	delete[] updateB;
	delete[] updateM;

	// Replace memory with current hidden layer state
	for (int i = 0; i<last; i++){
		this->mem[i].push_back(this->outputs[i + 1]);
		this->mem_inputs[i].push_back(this->inputs[i + 1]);
		this->mem_outputs[i].push_back(this->outputs[i]);
		if (this->mem[i].size() > this->back_t){
			this->mem[i].pop_front();
			this->mem_deltas[i].pop_front();
			this->mem_inputs[i].pop_front();
			this->mem_outputs[i].pop_front();
		}
	}
}

void RNNet::feedforwardF(mat input){
#ifdef DEBUG
	puts("forward");
#endif
	this->outputs.clear();
	this->inputs.clear();
	this->inputs.push_back(input);
	this->outputs.push_back(input);
	// Hidden layers with memory
	int i;
	for (i = 0; i<this->weights.size() - 1; i++){
		mat Z = (this->weights[i] * this->outputs[i]) + (this->mem_weights[i] * this->mem[i].back()) + this->bias[i];
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
	mat A = this->softmax_mat(Z, true);
#else
	printf("error: sigmoid + factorization\n");
	mat A = this->sigmoid_mat(Z);
#endif
	this->inputs.push_back(Z);
	this->outputs.push_back(A);

	return;
}

void RNNet::backpropF(mat y, int wordCluster){
#ifdef DEBUG
	puts("backprop");
#endif
	vector<mat> delta;
	mat tempOut = subMatCopy(outputs.back(), wordCluster);
	mat tempY = subMatCopy(y, wordCluster);
	mat error = tempOut - tempY;
	if (arma::norm(error) != arma::norm(error))
	{
		for (int i = 0; i < error.size(); i++)
		if (error[i] != error[i])
			cout << error[i] << endl;
		cout << "WTF" << endl;
		exit(-1);
	}
#ifdef SOFTMAX
	mat D = error;
#else
	mat D = error % this->sigmoid_prime_mat(this->inputs.back());
#endif

	if (!this->batch_start){
		this->deltas.clear();
		this->deltas.push_back(D);
	}
	else{
		this->deltas[0] += D;
	}
	delta.push_back(D);
	//%: element-wise dot
	for (int i = this->outputs.size() - 2, j = 1; i>0; i--, j++){
		mat m;
		if (i == this->outputs.size() - 2)
		{
			mat tempW = subMatCopy(this->weights[i], wordCluster);
#ifdef RELU
			m = this->ReLU_prime_mat(this->inputs[i]) % (tempW.t()*delta.back());
#else
			m = this->sigmoid_prime_mat(this->inputs[i]) % (tempW.t()*delta.back());
#endif
		}
		else
		{
#ifdef RELU
			m = this->ReLU_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#else
			m = this->sigmoid_prime_mat(this->inputs[i]) % (this->weights[i].t()*delta.back());
#endif
		}
#ifdef TOO_SMALL
		double norm = arma::norm(m);
		if (norm < TOO_SMALL)
			m = m * 0;
#endif
		delta.push_back(m);

		// Propagate delta to memory deltas
		while (!this->mem_deltas[i - 1].empty()) this->mem_deltas[i - 1].pop_front();
		this->mem_deltas[i - 1].push_back(m);
		int size = this->mem_inputs[i - 1].size() - 1;
		for (int k = 0; k <= size; k++){
#ifdef RELU
			mat mm = this->ReLU_prime_mat(this->mem_inputs[i - 1][size - k]) % (this->mem_weights[i - 1] * this->mem_deltas[i - 1].back());
#else
			mat mm = this->sigmoid_prime_mat(this->mem_inputs[i - 1][size - k]) % (this->mem_weights[i - 1] * this->mem_deltas[i - 1].back());
#endif
#ifdef TOO_SMALL
			double norm = arma::norm(mm);
			if (norm < TOO_SMALL)
				mm = mm * 0;
#endif
			this->mem_deltas[i - 1].push_back(mm);
		}
		this->mem_deltas[i - 1].pop_front();	//pop current delta
		// Batching
		if (!this->batch_start)
			this->deltas.push_back(m);
		else
			this->deltas[j] += m;
	}
	if (!this->batch_start)
		this->batch_start = true;
	delta.clear();
}

void RNNet::updateF(int wordCluster){
#ifdef DEBUG
	puts("update");
#endif
	this->batch_start = false;
	int last = this->weights.size() - 1;
	mat *updateW = new mat[last + 1];
	mat *updateB = new mat[last + 1];
	mat *updateM = new mat[last];
	// Update output layer
	
	updateW[last] = (this->deltas[0] / this->batch_size)*this->outputs[last].t();
	updateB[last] = this->deltas[0] / this->batch_size;

	// Update Hidden layers
	for (int i = 0; i<last; i++){
		int delta_s = this->mem_deltas[i].size() - 1;
		// Weights
		mat tmp = this->deltas[last - i] * this->outputs[i].t();
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][delta_s - j] * this->mem_outputs[i][j].t();
		//this->weights[i] -= this->learning_rate*(tmp / batch_size);
		updateW[i] = tmp / batch_size;
		// Biases
		tmp = this->deltas[last - i];
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][j];
		//this->bias[i] -= this->learning_rate*(tmp / batch_size);
		updateB[i] = tmp / batch_size;
		// Update memory
		tmp = (this->deltas[last - i] / this->batch_size)*this->mem[i].back().t();
		for (int j = 0; j <= delta_s; j++)
			tmp += this->mem_deltas[i][delta_s - j] * this->mem[i][j].t();
		//this->mem_weights[i] -= this->learning_rate*tmp;
		updateM[i] = tmp;
	}

	double norm = 0;
	/*norm += arma::norm(updateW[last]);
	norm += arma::norm(updateB[last]);
	for (int i = 0; i < last; i++)
	{
		norm += arma::norm(updateW[i]);
		norm += arma::norm(updateB[i]);
		norm += arma::norm(updateM[i]);
	}
	cout << norm << endl;*/
	double threshold = 100;
	double factor = threshold < norm ? threshold / norm : 1;

	mat tempW = subMatCopy(this->weights[last], wordCluster);
	mat tempBias = subMatCopy(this->bias[last], wordCluster);
	tempW -= this->learning_rate*factor*updateW[last];
	tempBias -= this->learning_rate*factor*updateB[last];
	subMatCopy(tempW, this->weights[last], wordCluster);
	subMatCopy(tempBias, this->bias[last], wordCluster);

	for (int i = 0; i < last; i++)
	{
		weights[i] -= learning_rate*factor*updateW[i];
		bias[i] -= learning_rate*factor*updateB[i];
		mem_weights[i] -= learning_rate*factor*updateM[i];
	}
	delete[] updateW;
	delete[] updateB;
	delete[] updateM;
	// Replace memory with current hidden layer state
	for (int i = 0; i<last; i++){
		this->mem[i].push_back(this->outputs[i + 1]);
		this->mem_inputs[i].push_back(this->inputs[i + 1]);
		this->mem_outputs[i].push_back(this->outputs[i]);
		if (this->mem[i].size() > this->back_t){
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
#ifdef TOO_SMALL
	if (x < TOO_SMALL && x > -TOO_SMALL)
		return 0;
#endif
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

mat RNNet::softmax_mat(mat m, bool fac)
{
	if (fac)
	{
		double total = 0;
		for (int i = 0; i < map_class.size(); i++)
		{
			m[i] = exp(m[i]);
			if (m[i] != m[i])
				cout << "softmax overflow?" << m[i] << endl;
			total += m[i];
		}
		if (total > TOO_BIG)
			total = TOO_BIG;
		for (int i = 0; i < map_class.size(); i++)
		{
			m[i] /= total;
#ifdef TOO_SMALL
			if (m[i] < TOO_SMALL && m[i] > -TOO_SMALL)
				m[i] = 0;
#endif
			if (m[i] > 1)
				m[i] = 1;
			if (m[i] != m[i])
				cout << "softmax overflow?" << m[i] << endl;
		}
		total = 0;
		for (int i = map_class.size(); i < m.size(); i++)
		{
			m[i] = exp(m[i]);
			if (m[i] != m[i])
				cout << "softmax overflow?" << m[i] << endl;
			total += m[i];
		}
		if (total > TOO_BIG)
			total = TOO_BIG;
		for (int i = map_class.size(); i < m.size(); i++)
		{
			m[i] /= total;
#ifdef TOO_SMALL
			if (m[i] < TOO_SMALL && m[i] > -TOO_SMALL)
				m[i] = 0;
#endif
			if (m[i] > 1)
				m[i] = 1;
			if (m[i] != m[i])
				cout << "softmax overflow?" << m[i] << endl;
		}
	}
	else
	{
		double total = 0;
		for (mat::iterator i = m.begin(); i != m.end(); i++){
			*i = exp(*i);
			total += *i;
		}
		for (mat::iterator i = m.begin(); i != m.end(); i++){
			*i /= total;
#ifdef TOO_SMALL
			if (*i < TOO_SMALL && *i > -TOO_SMALL)
				*i = 0;
#endif
		}
	}
	return m;
}

mat RNNet::subMatCopy(mat& m, int wordCluster)
{
	mat tempW(map_cluster_i[wordCluster].size()+numClusters, m.n_cols);
	for (int j = 0; j < map_cluster_i[wordCluster].size(); j++)
		tempW.row(j) = m.row(map_cluster_i[wordCluster][j]);
	for (int j = 0; j < numClusters; j++)
		tempW.row(map_cluster_i[wordCluster].size() + j) = m.row(map_class.size() + j);
	return tempW;
}

void RNNet::subMatCopy(mat source, mat &dest, int wordCluster)
{
	for (int j = 0; j < map_cluster_i[wordCluster].size(); j++)
		dest.row(map_cluster_i[wordCluster][j]) = source.row(j);
	for (int j = 0; j < numClusters; j++)
		dest.row(map_class.size() + j) = source.row(map_cluster_i[wordCluster].size() + j);
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
	getline(input_class, line);
	map_cluster_i.resize(atoi(line.c_str())+1);
	numClusters = atoi(line.c_str()) + 1;
	
	for(int i=0; getline(input_class, line) && line != "";i++){
		vector<string> x = split(line, " ");
		map_class[x[0]] = i;
		map_class2[i] = x[0];
		if (x.size() > 1)
		{
			map_cluster[i] = atoi(x[1].c_str());
			map_cluster_i[atoi(x[1].c_str())].push_back(i);
		}
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

void RNNet::predict(string fname, string fvec, string fclass, string oname, map<string, mat> &map_vec, int n_choice, char symbol_choice){
	ifstream input_vec(fvec.c_str(), ifstream::in);
	ifstream input_class(fclass.c_str(), ifstream::in);
	ifstream fi(fname.c_str(), ifstream::in);
	ofstream fo(oname.c_str());
	string line;
	int n_lines = 0;

	// Get number of lines
	while (getline(fi, line))
		++n_lines;
	fi.close();
	fi.open(fname.c_str(), ifstream::in);
	int ten_percent = n_lines*0.1;

	// load vector and class
	getline(input_vec, line);
	vector<string> x = split(line, " ");
	int n_feature = atoi(x[1].c_str());
	// word vectors
	for (; getline(input_vec, line);){
		vector<string> x = split(line, " ");
		mat feature(n_feature, 1);

		for (int i = 1; i<x.size() - 1; i++){
			feature(i - 1, 0) = atof(x[i].c_str());
		}
		map_vec[x[0]] = feature;
	}
	// class of words
	getline(input_class, line);
	map_cluster_i.resize(atoi(line.c_str()) + 1);
	numClusters = atoi(line.c_str()) + 1;

	for (int i = 0; getline(input_class, line) && line != ""; i++){
		vector<string> x = split(line, " ");
		map_class[x[0]] = i;
		map_class2[i] = x[0];
		if (x.size() > 1)
		{
			map_cluster[i] = atoi(x[1].c_str());
			map_cluster_i[atoi(x[1].c_str())].push_back(i);
		}
	}

	// predict
	// format: a b c d [e] f g
	vector<string> choices;
	int index_choice;

	for (int i = 0; getline(fi, line); i++){
		// print progress
		if (i % ten_percent == 0){
			printf(".");
			fflush(stdout);
		}

		x = split(line, " ");
		for (int j = 0; j<x.size(); j++){
			if (x[j][0] == symbol_choice){
				index_choice = j;
				choices.push_back(x[j].substr(1, x[j].length() - 2));
				break;
			}
		}
		// Start predict and choose best guess
		if ((i + 1) % n_choice == 0){
			double max_p = -1;
			int ans;
			// Check each choie and see whose probability is max
			for (int j = 0; j<n_choice; j++){
				double p = 1;
				x[index_choice] = choices[j];
				mat input;
				int out;
				// Forward a whole sentance
				for (int k = 0; k<x.size() - 1; k++){
					input = map_vec[x[k]];
					if (input.n_rows == 0)
						input = zeros<mat>(n_feature, 1);	//Others has no vector, so why not also zeros...
					this->feedforward(input);
					if (map_class.find(x[k + 1]) != map_class.end()){
						out = map_class[x[k + 1]];
					}
					else{
						out = this->outputs.back().n_rows - 1;
					}
					if (numClusters == 1)
						p *= this->outputs.back()(out, 0);
					else
					{
						p *= this->outputs.back()(out, 0)*this->outputs.back()(map_cluster[out]+map_class.size(), 0);
					}
				}
				//cout << p << endl;
				if (p > max_p){
					max_p = p;
					ans = j;
				}
			}
			fo << ans << "\n";
			choices.clear();
		}
	}

	return;
}



