#include <bits/stdc++.h>
#include <vector>
#include <random>
using namespace std;


class Neuron{    

   vector<double> weights;

   double relu ( double input ) {
        return max(0.0 , input) ;
    } 

    void generateRandomWeights(){
        uint32_t inputs = weights.size();
        
        for(uint32_t inp=0 ; inp < inputs ; inp++ ){
            static default_random_engine eng ;
            static uniform_int_distribution<> dis (-1e7 , 1e7) ;
            weights [inp] = dis (eng) ;  
        }
    }
   public:
    Neuron ( uint32_t inputSize ) {
        weights = std::vector <double> (inputSize) ;
        generateRandomWeights() ;
    }

    double getActivation ( double input ) {
        return relu(input) ;
    }

};
