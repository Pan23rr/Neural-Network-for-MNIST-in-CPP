#include <bits/stdc++.h>
#include <cmath>
using namespace std;


double MAX_ACCURACY=0.0;

const int MAXN = 6e4;
vector<vector<vector<double>>> image(MAXN, vector<vector<double>>(28, vector<double>(28, 0)));
unsigned int num, magic, rows, cols;
int currBatch=0;
unsigned int correctLabel[MAXN];
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
	icin.open("FileLocation", ios::binary);
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
	icin.open("FileLocation", ios::binary);
	magic = in(icin, 4), num = in(icin, 4);
	for (int i = 0; i < num; i++) {
		correctLabel[i] = in(icin, 1);
	}
}

vector<vector<double>> Images;

void flattenImage() {
	Images.clear();
	for (int i = 0; i < 6e4; i++) {
		vector<double> singleImage;
		for (int j = 0; j < 28; j++) {
			for (int k = 0; k < 28; k++) {
				singleImage.push_back(image[i][j][k]);
			}
		}
		Images.push_back(singleImage);
	}
}




class Neuron {

	vector<double> weights;


	void generateRandomWeights() {
		uint32_t inputs = weights.size();
		for(uint32_t inp = 0; inp < inputs; inp++) {
			static std::default_random_engine eng(std::random_device{}());
			// Better initialization for ReLU networks
			uniform_real_distribution<double> dis(-sqrt(2.0 / inputs), sqrt(2.0 / inputs));
			weights[inp] = dis(eng);
		}
	}
public:
	Neuron ( uint32_t inputSize ) {
		weights = std::vector <double> (inputSize) ;
		generateRandomWeights() ;
	}

	double relu ( double input ) {
		return max(0.0, input) ;
	}

	double getActivation ( double input ) {
		return relu(input) ;
	}

	vector<double> weight() {
		return weights;
	}
	void updateW(vector<double> newW) {
		weights=newW;
	}
};



class Layer {

	vector<Neuron> neurons;
	vector<double> bias;


	void generateNeurons( uint32_t &layerSize, uint32_t &inputSize) {

		for(uint32_t neuron=0; neuron < layerSize; neuron++ ) {
			neurons.emplace_back (inputSize);
		}
	}

	void generateBias() {
		default_random_engine e(random_device{}());
		uniform_real_distribution<> dis(-0.5, 0.5);
		bias.resize(neurons.size());
		for (double &b : bias) b = dis(e);
	}


public:

	Layer(uint32_t layerSize, uint32_t inputSize) {
		neurons.reserve(layerSize);
		generateNeurons(layerSize, inputSize);
		generateBias();
	}

	vector<double> biases() {
		return bias;
	}



	uint32_t size() {
		return neurons.size();
	}


	vector<vector<double>> weights() {
		int input=neurons[0].weight().size();
		int neuronCount=neurons.size();
		vector<vector<double>> weightMatrix(neurons[0].weight().size(),vector<double>(neurons.size()));
		for (size_t i = 0; i < neurons.size(); ++i)
			for (size_t j = 0; j < neurons[i].weight().size(); ++j)
				weightMatrix[j][i] = neurons[i].weight()[j];
		return weightMatrix;
	}

	void updatebias(const vector<double>& newbias) {
		bias = newbias;
	}

	void updateweights(const vector<vector<double>>& deltaW) {
		for (size_t i = 0; i < neurons.size(); ++i) {
			auto wOld = neurons[i].weight();
			for (size_t j = 0; j < wOld.size(); ++j) {
				wOld[j] -= deltaW[j][i];
			}
			neurons[i].updateW(wOld);
		}
	}

	Neuron &operator[](size_t index) {
		return neurons[index];
	}


};

void Transpose(vector<vector<double>> & Mat) {
	if (Mat.empty() || Mat[0].empty()) return;
	vector<vector<double>> Transposed(Mat[0].size(),vector<double> (Mat.size(),0));
	for(int r=0; r<Mat.size(); r++) {
		for(int c=0; c<Mat[0].size(); c++) {
			Transposed[c][r]=Mat[r][c];
		}
	}
	Mat=Transposed;
}



vector<vector<double>> MatMul(vector<vector<double>> &A,vector<vector<double>> &B ) {
	vector<vector<double>> Result(A.size(),vector<double> (B[0].size(),0.0));
	int first=A.size(),second=A[0].size(),third=B[0].size();
	for(int i=0; i<first; i++) {
		for(int j=0; j<third; j++) {
			for(int k=0; k<second; k++) {
				Result[i][j]+=(A[i][k]*B[k][j]);
			}
		}
	}
	return Result;
}


class NeuralNetwork {
	vector<Layer> model;
	double lR=1e-2;
public:
	NeuralNetwork(uint32_t layers, vector<uint32_t> neuronCount,vector<uint32_t> layerSize) {
		for (int layer = 0; layer < layers; layer++) {
			model.emplace_back(neuronCount[layer], layerSize[layer]);
		}

	}

	vector<vector<int>> labels(int batchSize, int classSize=10) {
		vector<vector<int>> label;
        //if(currBatch>59967) cout<<"REACHED";
		for(int image=currBatch; image<Images.size() && image<currBatch+batchSize; image++) {
            //if(image>59967) cout<<"REACHED "<<image<<endl;
			vector<int> t (classSize,0);
			t[correctLabel[image]]=1;
			label.push_back(t);
		}
		return label;
	}

	vector<vector<double>> LayerActivation(vector<vector<double>> Input,Layer layer) {
		vector<vector<double>> Mat=layer.weights();
		vector<vector<double>> Result=MatMul(Input,Mat);
		vector<double> bias=layer.biases();
		for(int r=0; r<Result.size(); r++) {
			for(int c=0; c<Result[r].size(); c++) {
				Result[r][c]+=bias[c];
			}
		}
		return Result;
	}

	void Relu(vector<vector<double>> &result) {
		for(int r=0; r<result.size(); r++) {
			for(int c=0; c<result[r].size(); c++) {
				result[r][c]=max(0.0,result[r][c]);
			}
		}
	}
	vector<vector<vector<double>>> FeedForward( int batchSize) {
		vector<vector<vector<double>>> activations(model.size());

		vector<vector<double>> currInput(batchSize);
        
		for (int datapoint = currBatch; datapoint < Images.size() && datapoint < currBatch + batchSize; datapoint++) {
			currInput[datapoint - currBatch] = Images[datapoint];
		}

		for (int layer = 0; layer < model.size(); layer++) {
			currInput = LayerActivation(currInput, model[layer]);

			if (layer != model.size() - 1) {
				Relu(currInput);
			}
			activations[layer] = currInput;
		}

		return activations;
	}


	vector<double> softmax(vector<double> finalOutput) {
		vector<double> result(finalOutput.size());
		double total = 0.0;
		double maxLogit = *max_element(finalOutput.begin(), finalOutput.end());

		for (double activation : finalOutput) {
			total += exp(activation - maxLogit);
		}

		if (total == 0.0) total = 1e-12; 

		for (int i = 0; i < finalOutput.size(); i++) {
			result[i] = exp(finalOutput[i] - maxLogit) / total;
		}

		return result;
	}


	vector<vector<double>> Probabilities(vector<vector<double>> outputs) {
		vector<vector<double>> pred(outputs.size());
		int i=0;
		for(vector<double> final: outputs) {
			pred[i]=softmax(final);
			i++;
		}
		return pred;
	}
	vector<vector<double>> cost(vector<vector<double>> predY, vector<vector<int>> trueY) {
		vector<vector<double>> gradients = predY;
		for (size_t i = 0; i < gradients.size(); ++i) {
			for (size_t j = 0; j < gradients[i].size(); ++j) {
				gradients[i][j] -= trueY[i][j];
			}
		}
		return gradients;
	}
	void unitwiseMulti(vector<vector<double>> &result, vector<vector<double>> activations) {
		for(int r=0; r<result.size(); r++) {
			for(int c=0; c<result[r].size(); c++) {
				result[r][c]*=(activations[r][c]>0?1.0:0.0);
			}
		}
	}
	vector<vector<double>> getError(vector<vector<double>> nextLayerError, int layer, vector<vector<double>> activations) {

		vector<vector<double>> weights=model[layer+1].weights();
		Transpose(weights);
		vector<vector<double>> Result=MatMul(nextLayerError,weights);
		unitwiseMulti(Result,activations);
		return Result;

	}
	void UpdateBias(int layer, vector<vector<double>> layerError) {
		vector<double> biasUpdate(model[layer].size(), 0.0);
		int batchSize = layerError.size();

		for(int r = 0; r < layerError.size(); r++) {
			for(int c = 0; c < layerError[0].size(); c++) {
				biasUpdate[c] += layerError[r][c];
			}
		}

		for(int c = 0; c < biasUpdate.size(); c++) {
			biasUpdate[c] = lR * biasUpdate[c] / batchSize;  
		}

		vector<double> currentBias = model[layer].biases();
		for (int i = 0; i < currentBias.size(); i++) {
			currentBias[i] -= biasUpdate[i];
		}
		model[layer].updatebias(currentBias);
	}

	void UpdateWeights(int layer, vector<vector<double>> prevActivations, vector<vector<double>> currError) {
		Transpose(prevActivations);
		vector<vector<double>> Result = MatMul(prevActivations, currError);
		int batchSize = currError.size();

		for(int r = 0; r < Result.size(); r++) {
			for(int c = 0; c < Result[r].size(); c++) {
				Result[r][c] *= lR / batchSize; 
			}
		}
		model[layer].updateweights(Result);
	}



	void backPropagate(int batchSize, const vector<vector<double>> lastLayerError, const vector<vector<vector<double>>>& currActivations){
		int L = model.size();
		vector<vector<vector<double>>> errors(L);
		UpdateBias(L - 1, lastLayerError);
		vector<vector<double>> prevAct;
		if (L > 1) {
			prevAct = currActivations[L - 2];
		} else {
			prevAct.resize(batchSize);
			for (int i = 0; i < batchSize; ++i) {
				prevAct[i] = Images[currBatch + i];
			}
		}
		UpdateWeights(L - 1, prevAct, lastLayerError);

		errors[L - 1] = lastLayerError;

		for (int layer = L - 2; layer >= 0; --layer) {
			errors[layer] = getError(errors[layer + 1], layer, currActivations[layer]);
			UpdateBias(layer, errors[layer]);
			if (layer == 0) {
				vector<vector<double>> inputBatch(batchSize);
				for (int i = 0; i < batchSize; ++i) {
					inputBatch[i] = Images[currBatch + i];
				}
				UpdateWeights(0, inputBatch, errors[0]);
			} else {
				UpdateWeights(layer,currActivations[layer - 1],errors[layer]);
			}
		}
	}


	void train(vector<Layer> &best) {
		int batch = 64;
		for (currBatch = 0; currBatch < num; currBatch += batch) {
            //cout<<"TEST "<<currBatch;
			vector<vector<int>> trueY = labels(batch);
			vector<int> actualLabels(trueY.size());
			for (int j = 0; j < trueY.size(); ++j) {
				for (int k = 0; k < 10; ++k) {
					if (trueY[j][k] == 1) {
						actualLabels[j] = k;
						break;
					}
				}
			}
			vector<vector<vector<double>>> currActivations = FeedForward(trueY.size());
            vector<vector<double>> predY = Probabilities(currActivations.back());
            vector<vector<double>> LastlayerError = cost(predY, trueY);
			backPropagate(trueY.size(), LastlayerError, currActivations);
			double totalLoss = 0.0;
			for (int j = 0; j < predY.size(); ++j) {
				totalLoss -= log(max(predY[j][actualLabels[j]], 1e-15));
			}
			double avgLoss = totalLoss / batch;

			int correct = 0;
			for (int j = 0; j < predY.size(); ++j) {
				int predicted = max_element(predY[j].begin(), predY[j].end()) - predY[j].begin();
				if (predicted == actualLabels[j]) correct++;
			}
			double accuracy = (correct * 100.0) / batch;
            if(accuracy>MAX_ACCURACY){
                best=model;
                MAX_ACCURACY=accuracy;
            }
			cout << "Batch " << (currBatch / batch + 1) << " / " << num/64<< " -> ";
			cout << "Loss: " << avgLoss << " | Accuracy: " << fixed << setprecision(2) << accuracy << "%\n";
		}
	}
};





void saveModel( vector<Layer>& Best, string filename) {
    ofstream fout(filename);

    for (int i = 0; i < Best.size(); ++i) {
        vector<double> bias = Best[i].biases();
        for (double b : bias) fout << b << " ";
        fout<<"\n";
        vector<vector<double>> weights = Best[i].weights(); 
        for (const auto& row : weights) {
            for (double w : row) fout << w << " ";
            fout << "\n";
        }
        fout << "\n";
    }

    fout.close();
}




int main() {
	srand(time(0));
	input();
	flattenImage();
    vector<Layer> Best;
    //NeuralNetwork maxAccuracy(1,{0},{0});
	NeuralNetwork train(3, {100,50,10}, {784,100,50});
	train.train(Best);
    cout<<"======Finished training the Neural Network======\n";
    cout<<"----Maximum Accuracy obtained: "<<MAX_ACCURACY<<" ----";
    saveModel(Best,"model.txt");
}
