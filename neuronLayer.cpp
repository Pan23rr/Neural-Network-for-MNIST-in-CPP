#include <bits/stdc++.h>
#include <random>
#include "./neuron.cpp"


class Layer {
    
    vector<Neuron> neurons;
    double bias;

    void generateNeurons( uint32_t &layerSize, uint32_t &inputSize){

        for(uint32_t neuron=0; neuron < layerSize; neuron++ ){
            neurons [neuron] = Neuron (inputSize);
        }
    }

    void generateBias(){
        default_random_engine e;
        uniform_real_distribution<> dis(1e-5, 1e5);
        bias=dis(e);
    }
    
    public:

    Layer(uint32_t layerSize, uint32_t inputSize){
        neurons=vector<Neuron> (layerSize);
        generateNeurons(layerSize,inputSize);
        generateBias();
    }

    double getActivation( vector<double> inputs){
        double currActivation=0.0;

        for(Neuron &neuron: neurons){
            double neuronActivation=0.0;
            for(double &input: inputs){
                neuronActivation=neuronActivation+neuron.getActivation(input);
            }
            currActivation+=neuronActivation;
        }
    }
};