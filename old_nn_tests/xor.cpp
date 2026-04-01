// Filename: main.cpp
// Author: Jonah Taylor
// Date: December 20, 2024
// Description: A xor program to test the Neural Network

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <deque>

#include "pseument.hpp"

using namespace std;

int window_width = 1120;
int window_height = 1120;
const float squareWidth = 40;

int guess = -1;
int detectFPS = 0; // Frames Per Second
float FPS = 1000;

bool A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, Z = false; // Save X and Y for Inputs and Predictions
bool training = false;
bool fast = false;
bool dSDisplay = false;
bool printEpochs = true;

NeuralNetwork nn({
    MakeLayer({2}, "dense", "leakyrelu"), 
    MakeLayer({2}, "dense", "leakyrelu"),
    MakeLayer({1}, "dense", "leakyrelu")
});
vector<double> inputs;
vector<vector<double>> X;
vector<vector<double>> Y;
size_t epochs = 1000;
size_t batchSize = 1;
double trainingSpeed = 0.01;

int main3() {

    X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    Y = {{0}, {1}, {1}, {0}};

    //nn.load("./save");
    nn.train(X, Y, epochs, batchSize, trainingSpeed, "adamw", true);

    for (uint i = 0; i < X.size(); i++) {
        vector<double> prediction = nn.forward(X[i]);
        cout << "Prediction: " << (double)prediction[0] << " Ans: " << Y[i][0] << endl;
    }

    nn.save("./save");

}