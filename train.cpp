#include <bits/stdc++.h>
#include <cmath>
using namespace std;







const int MAXN = 6e4;
double image[MAXN][30][30];
unsigned int num, magic, rows, cols;
int currBatch=0;
unsigned int label[MAXN];
unsigned int in(ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}
void input() {
    ifstream icin;
    icin.open("C:/Users/Asus/Desktop/NN/data/train-images-idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (int i = 0; i < num; i++) {
        for (int x = 0; x < rows; x++) {
            for (int y = 0; y < cols; y++) {
                image[i][x][y] = in(icin, 1);
                image[i][x][y]/=(255*1.0);
            }
        }
    }
    icin.close();
    icin.open("C:/Users/Asus/Desktop/NN/data/train-labels-idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (int i = 0; i < num; i++) {
        label[i] = in(icin, 1);
    }
}



class Neuron{    

   vector<double> weights;


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
    
double relu ( double input ) {
        return max(0.0 , input) ;
    } 

 double getActivation ( double input ) {
        return relu(input) ;
    }

    vector<double> weight(){
        return weights;
    }
};



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
    double biases(){
        return bias;
    }
    uint32_t size(){
        return neurons.size();
    }

    Neuron &operator[](size_t index){
        return neurons[index];
    }
};




class NeuralNetwork{
    vector<Layer> model;
    public:
        NeuralNetwork(uint32_t layers, vector<uint32_t> neuronCount ,vector<uint32_t> layerSize){
            model=vector<Layer> (layers);
            for(int layer=0;layer<layers;layer++){

                model[layer]= Layer(neuronCount[layer], layerSize[layer]);
            
            }
        }
        vector<vector<double>> feedForward(uint32_t batchSize=1000){
            vector<vector<double>> activations(model.size());

            //Processing for the firstLayer
            for(int datapoint=currBatch;datapoint<MAXN && datapoint<currBatch+batchSize;datapoint++){
                int firstLayersize=model[0].size();
                for(int neuron=0;neuron<firstLayersize;neuron++){
                    vector<double> weights=model[0][neuron].weight();
                    double currActivation=0.0;
                     for(int x=0;x<30;x++){
                    
                        for(int y=0;y<30;y++){
                            currActivation+=max(0.0,weights[neuron]*image[datapoint][x][y]);
                        }
                
                    }
                    currActivation+=model[0].biases();
                    currActivation=currActivation/(firstLayersize*1.0);
                    activations[0].push_back(currActivation);
                }
            }


            //Applying Feed Forward for the subsequent layers with Relu
            for(int layer=1;layer<model.size();layer++){
                vector<double> Input=activations[layer-1];
                uint32_t layerSize=model[layer].size();
                for(uint32_t neuron;neuron<layerSize;neuron++){
                    double currActivation=0.0;
                    vector<double> weights=model[0][neuron].weight();
                    for(int i=0;i<Input.size();i++){
                        currActivation+=(weights[i]*Input[i]);
                    }
                    currActivation+=(model[layer].biases());
                    currActivation/=(Input.size()*1.0);
                    currActivation=max(0.0,currActivation);
                    activations[layer].push_back(currActivation);
                }
            }

            return activations;
        }


        double calculateProbability(vector<double> activations, unsigned int correctLabel){
            double ex,totalex=0;
            ex=exp(activations[correctLabel]);
            for(int i=0;i<10;i++){
                totalex+=exp(activations[i]);
            }
            return ex/(1.0*totalex);
        }

        double getLoss(double probability){
            return -log(probability);
        }
        

};