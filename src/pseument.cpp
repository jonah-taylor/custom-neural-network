// Filename: pseument.cpp
// Author: Jonah Taylor
// Date: December 25, 2024
// Description: Pseument (Pseudo Mantis), my first neural network

#include "pseument.hpp"

NeuralNetwork::NeuralNetwork(const vector<MakeLayer>& l_info) {

    if(debugging) cout << "started constructor\n";

    layers.push_back(unique_ptr<Layer>(new DenseLayer(l_info[0].l_size[0], l_info[0].l_size[0], l_info[0].a_func_name)));

    for (size_t l = 1; l < l_info.size(); ++l) {
        switch(l_info[l].l_type) {
            case 0:
                layers.push_back(unique_ptr<Layer>(new DenseLayer(l_info[l].l_size[0], layers[l - 1]->a.rows() * layers[l - 1]->a.cols(), l_info[l].a_func_name)));
                break;
            case 1:
                layers.push_back(unique_ptr<ConvoLayer>(new ConvoLayer(l_info[l].l_size[0], l_info[l].l_size[1], l_info[l].a_func_name, 3)));
                break;
            default:
                layers.push_back(unique_ptr<Layer>(new DenseLayer(l_info[l].l_size[0], layers[l - 1]->a.rows() * layers[l - 1]->a.cols(), l_info[l].a_func_name)));
                break;
        }
    }

    if(debugging) cout << "finished constructor\n";

}

vector<double> NeuralNetwork::forward(const vector<double>& input) {

    if(debugging) cout << "Started forward\n";

    // convert vector<double> to vectorxd
    VectorXd in(input.size());
    for (size_t i = 0; i < input.size(); ++i)
        in[i] = input[i];

    forward(in);

    vector<double> aws(layers.back()->a.data(), layers.back()->a.data() + layers.back()->a.size());

    if(debugging) cout << "Finished forward\n";

    return aws;

}

VectorXd NeuralNetwork::forward(const VectorXd& in) {

    if(debugging) cout << "Started forward\n";

    layers[0]->a = in;
    MatrixXd out = in;
    for (size_t l = 1; l < layers.size(); ++l) {
        out = layers[l]->forward(out);
    }

    if(debugging) cout << "Finished forward " << "\n";

    return out;

}

void NeuralNetwork::getOutputDeltas(const VectorXd& target) {


    if(debugging) cout << "Started getOutputDeltas\n";

    layers.back()->getOutputDeltas(target);
    layers.back()->updateGrads(layers[layers.size() - 2]->a);

    if(debugging) cout << "Start accuracy testing\n";

    tested++;
    if(layers.back()->a(0) > 0.5 ? 1 : 0 == target(0))
        correct++;

    if(debugging) cout << "Finished accuracy testing\n";

    if(debugging) cout << "Finished getOutputDeltas\n";

}

void NeuralNetwork::backward() {

    if(debugging) cout << "Started backward\n";

    for (size_t l = layers.size() - 2; l > 0; --l) {

        layers[l]->backward(layers[l + 1]->w, layers[l + 1]->dz);
        layers[l]->updateGrads(layers[l - 1]->a);

    }

    if(debugging) cout << "Finished backward\n";

}

void NeuralNetwork::stepSGD(double& lr, size_t& bs) {

    if(debugging) cout << "Started updateNeurons\n";

    for (size_t l = 1; l < layers.size(); ++l)
        layers[l]->stepSGD(lr, bs);

     if(debugging) cout << "Finished updateNeurons\n";

}

void NeuralNetwork::stepAdamW(double& lr, size_t& bs, size_t& t) {

    if(debugging) cout << "Started updateNeurons\n";

    for (size_t l = 1; l < layers.size(); ++l)
        layers[l]->stepAdamW(lr, bs, t);

     if(debugging) cout << "Finished updateNeurons\n";

}


void NeuralNetwork::train(vector<vector<double>>& X, vector<vector<double>>& Y, 
        size_t& epochs, size_t& bs, double& lr, string da, bool print) {

    if(debugging) cout << "Started train\n";

    if(da == "sgd")
        descent = 0;
    else if(da == "adamw")
        descent = 1;
    else
        descent = 0;

    for(size_t l = 1; l < layers.size(); ++l) {
        layers[l]->lambda = lr / 100; // Weight decay
    }

    size_t numSamples = X.size();

    // Convert inputs to MatrixXd
    MatrixXd inputs(numSamples, X[0].size());
    for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            inputs(i, j) = X[i][j];
        }
    }

    // Convert targets to MatrixXd
    MatrixXd targets(numSamples, Y[0].size());
    for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < Y[i].size(); ++j) {
            targets(i, j) = Y[i][j];
        }
    }

    vector<int> shuffled(inputs.rows());
    iota(shuffled.begin(), shuffled.end(), 0);
    random_device rd;
    mt19937 g(rd());

    for (size_t epoch = 0; epoch < epochs; ++epoch) {

        shuffle(shuffled.begin(), shuffled.end(), g);
        t++;
        correct = 0;
        tested = 0;

        for (size_t i = 0; i < size_t(inputs.rows()); i += bs) {

            for (size_t b = 0; b < bs; ++b) {

                forward(inputs.row(shuffled[i + b]));
                getOutputDeltas(targets.row(shuffled[i + b]));
                backward();

            }
            switch(descent) {
                case 0:
                    stepSGD(lr, bs);
                    break;
                case 1:
                    stepAdamW(lr, bs, t);
                    break;
            }
        }

        if (print) cout << "Epoch " << epoch + 1 << ": " << correct << " / " << tested << "\n";
    }

    if(debugging) cout << "Finished train\n";

}


void NeuralNetwork::save(const string& fn) {

    if(debugging) cout << "Started save\n";

    // Open/create file
    ofstream file(fn);
    if(!file) {
        cerr << "File couldn't be accessed for saving\n";
        return;
    }

    // Save layer sizes
    file << layers.size() << "\n";
    for (size_t l = 0; l < layers.size(); l++) {
        if (dynamic_cast<DenseLayer*>(layers[l].get())) {
            file << "dense ";
        } else if (dynamic_cast<ConvoLayer*>(layers[l].get())) {
            file << "convolutional ";
        } else {
            file << "unknown ";
        }

        file << layers[l]->size().first << " ";
        file << layers[l]->size().second << " ";
        file << layers[l]->a_func_name << "\n";
    }
    file << "\n";

    // Save weight data in file
    for (size_t l = 1; l < layers.size(); l++) {
        file << layers[l]->w << "\n\n"; // Saves each matrix in Eigen format (row-major)
    }

    // Save bias data in file 
    for (size_t l = 1; l < layers.size(); l++) {
        file << layers[l]->b.transpose() << "\n\n";  // Save biases as a row (transpose to save as row)
    }
    file << "\n";

    file.close();

    if(debugging) cout << "Finished save\n";

}


void NeuralNetwork::load(const string& fn) {

    if(debugging) cout << "Started load\n";

    // Open File
    ifstream file(fn);
    if (!file) {
        cerr << "File couldn't be accessed for loading\n";
        return;
    }

    // Load number of layers
    size_t l_count;
    file >> l_count;
    layers.resize(l_count);

    // Load layer sizes
    string lt;
    size_t lsx, lsy;
    string afn;
    for (size_t l = 0; l < l_count; l++) {

        file >> lt;
        file >> lsx;
        file >> lsy;
        file >> afn;
        if(l == 0) 
            layers[l] = unique_ptr<DenseLayer>(new DenseLayer(lsx, 0, afn));
        else if(lt == "dense")
            layers[l] = unique_ptr<DenseLayer>(new DenseLayer(lsx, layers[l - 1]->size().first, afn));
        else if(lt == "convolutional")
            layers[l] = unique_ptr<ConvoLayer>(new ConvoLayer(lsx, lsy, afn, 3));
        else
            layers[l] = unique_ptr<DenseLayer>(new DenseLayer(lsx, layers[l - 1]->size().first, afn));
    }

    // Load weights
    for (size_t l = 1; l < layers.size(); l++) {
        for (size_t row = 0; row < layers[l]->size().first; row++) {
            for (size_t col = 0; col < layers[l - 1]->size().first; col++) {
                file >> layers[l]->w(row, col);  // Read individual element into matrix
            }
        }
    }

    // Load biases
    for (size_t l = 1; l < layers.size(); l++) {
        for (size_t i = 0; i < layers[l]->size().first; i++) {
            file >> layers[l]->b(i);  // Read individual element into vector
        }
    }

    file.close();

    if(debugging) cout << "Finished load\n";

}

vector<size_t> NeuralNetwork::getLayerSizes() {

    vector<size_t> layer_sizes(layers.size());
    for(size_t i = 0; i < layers.size(); i++)
        layer_sizes[i] = layers[i]->size().first;
    return layer_sizes;

}


