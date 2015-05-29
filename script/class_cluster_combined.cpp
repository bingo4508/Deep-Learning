#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string.h>
#include <stdlib.h>

using namespace std;

vector<string> split(string str,string sep){
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<string> arr;
    current=strtok(cstr,sep.c_str());
    while(current!=NULL){
        arr.push_back(current);
        current=strtok(NULL,sep.c_str());
    }   
    return arr;
}


int main()
{
	map<string, int> map_cluster;
	ifstream in_cluster("result_classes_20.txt");
	string line;
	while(getline(in_cluster, line))
	{
		vector<string> x = split(line, " ");
		map_cluster[x[0]] = atoi(x[1].c_str());
	}
	in_cluster.close();
	ifstream in_class("train_small.class");
	vector<string> classes;
	int size = 0;
	while(getline(in_class, line))
	{
		classes.push_back(line);
		size++;
	}
	in_class.close();
	
	ofstream out_class_cluster("output.class");
	for (int i = 0; i < size; i++)
		out_class_cluster << classes[i] << " " << map_cluster[classes[i]] << endl;
	out_class_cluster.close();
}
