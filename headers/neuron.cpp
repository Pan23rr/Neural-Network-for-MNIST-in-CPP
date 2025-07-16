#include <bits/stdc++.h>
#include <vector>
#include <random>
using namespace std;



class Neuron{    
   vector<double> weights;
    public:
    Neuron(int inputSize){
        weights=std::vector<double> (inputSize);
    }

    double relu(double input){
        return max(0.0,input);
    }

    
};
