// Filename: VisionAI.cpp
// Author: Jonah Taylor
// Date: November 25, 2024
// Description: A Neural Network that sees pixels and identifies numbers

#include <fstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <deque>

#include "pseument.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <SFML/System.hpp>

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

vector<vector<double>> images;
vector<double> labels;
int trainingSamples = 60000;
int trained = 0;

int draw_radius = 4;
const int largeGridSize = 140;
const int smallGridSize = 28;
const int ratio = largeGridSize / smallGridSize;
const int cellSize = window_width / largeGridSize;
const int dSCellSize = window_width / smallGridSize;
vector<vector<double>> largeGrid(largeGridSize, vector<double>(largeGridSize, 0.0));
vector<vector<double>> dSGrid(smallGridSize, vector<double>(smallGridSize, 0.0));
vector<double> dSInput;

NeuralNetwork nn({
    MakeLayer({784}, "dense", "leakyrelu"),
    MakeLayer({28, 28}, "convo", "leakyrelu"),
    MakeLayer({784}, "dense", "leakyrelu"),
    MakeLayer({30}, "dense", "leakyrelu"),
    MakeLayer({10}, "dense", "leakyrelu")

});
vector<double> inputs;
vector<vector<double>> X;
vector<vector<double>> Y;
size_t epochs = 5;
size_t batchSize = 1;
double trainingSpeed = 0.0001;

void keyBoardInputs();
vector<vector<double>> getMnistImages(const string& file_path, int num_images);
vector<double> getMnistLabels(const string& file_path, int num_labels);
int getNum(vector<double> outputs);

int main2(int argc, char* argv[]) {
    
    // Load network if used as argument
    if (argc > 1) {
        string filename = argv[1];
        cout << "Loading network: " << filename << "\n";
        nn.load("../data/arc/" + filename + ".txt");
    }

    sf::RenderWindow window(sf::VideoMode(window_width, window_height), "", sf::Style::None);
    window.setTitle("VisionAI Guess = " + to_string(guess) + " | FPS = " + to_string(detectFPS));

    sf::Vector2i mousePos = sf::Mouse::getPosition(window);

    // sf::Image icon;
    // if (!icon.loadFromFile("Images/icon.png")) {
    //     cerr << "Error loading icon\n";
    // }
    // else {
    //     window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());
    // }

    // Create random seed
    srand(time(0));

    // Get Mnist Data
    images = getMnistImages("../data/imgs/mnist/train-images.idx3-ubyte", trainingSamples);
    labels = getMnistLabels("../data/imgs/mnist/train-labels.idx1-ubyte", trainingSamples);

    // cout << labels[0] << "\n";
    // for(int i = 0; i < 28; i++) {
    //     for(int j = 0; j < 28; j++) {
    //         cout << round(images[0][28 * i + j]);
    //     }
    //     cout << "\n";
    // }

    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            dSGrid[i][j] = images[images.size() - 3][i * 28 + j];
        }
    }

    dSInput = images[images.size() - 3];
    
    // Set up clock for frame timing
    sf::Clock clock;
    const sf::Time targetTime = sf::seconds(1.f / FPS);
    static int frameCount = 0;

    // Main game loop
    while (window.isOpen()) {
        sf::Time elapsed = clock.getElapsedTime();
        if (elapsed >= targetTime || fast) {
            detectFPS = 1 / (elapsed.asSeconds());
            clock.restart();
            frameCount++;

            keyBoardInputs();

            if(frameCount == 100 || training) {
                guess = getNum(nn.forward(dSInput));
                // for(int i = 0; i < 28; i++) {
                //     for(int j = 0; j < 28; j++) {
                //         cout << round(dSInput[28 * i + j]);
                //     }
                //     cout << "\n";
                // }
                window.setTitle("VisionAI Guess = " + to_string(guess) + " | FPS = " + to_string(detectFPS));
                frameCount = 0;
            }

            // When window is closed
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    // Auto Save
                    int id = rand() % 101;
                    cout << "Saving network: " << "exit" + to_string(id) + "_" << to_string(int(epochs)) << ".txt" << "\n";
                    nn.save("../data/arc/exit" + to_string(id) + "_" + to_string(int(epochs)) + ".txt");
                    window.close();
                }
            }

            if(training) {
                for(int i = 0; i < 1000; i++) {
                    if((uint)(i + trained) >= images.size()) {
                        trained = 0;
                    }
                    X.push_back(images[i + trained]);
                    Y.push_back({0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                    Y[Y.size() - 1][labels[i + trained]] = 1;
                    trained++;
                }
                nn.train(X, Y, epochs, batchSize, trainingSpeed, "adamw", printEpochs);
                X.clear();
                Y.clear();
            }
            else {
                
                mousePos = sf::Mouse::getPosition(window);
                int gridX = mousePos.x / cellSize;
                int gridY = mousePos.y / cellSize;

                // Clear screen on right click
                if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
                    for (int i = 0; i < largeGridSize; ++i) {
                        for (int j = 0; j < largeGridSize; ++j) {
                            largeGrid[i][j] = 0;
                        }
                    }

                    for (int i = 0; i < smallGridSize; ++i) {
                        for (int j = 0; j < smallGridSize; ++j) {
                            dSGrid[i][j] = 0;
                        }
                    }
                }

                // Draw on left click
                if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                    for (int dy = -draw_radius; dy <= draw_radius; ++dy) {
                        for (int dx = -draw_radius; dx <= draw_radius; ++dx) {
                            // Check if within circle radius
                            if (dx * dx + dy * dy < draw_radius * draw_radius && gridX + dx >= 0 && gridX + dx < largeGridSize && gridY + dy >= 0 && gridY + dy < largeGridSize) {
                                largeGrid[gridY + dy][gridX + dx] = 1.0;
                            }
                        }
                    }
                    
                    for (int i = 0; i < smallGridSize; ++i) {
                        for (int j = 0; j < smallGridSize; ++j) {
                            dSGrid[i][j] = 0;
                        }
                    }
                    
                    // Update downscaled grid
                    for (int i = 0; i < largeGridSize; ++i) {
                        for (int j = 0; j < largeGridSize; ++j) {
                            dSGrid[i / ratio][j / ratio] += largeGrid[i][j];
                        }
                    }

                    // Normalize the 5x5 block sum to [0,1] by dividing by 25
                    for (int i = 0; i < smallGridSize; ++i) {
                        for (int j = 0; j < smallGridSize; ++j) {
                            dSGrid[i][j] /= ratio * ratio;
                        }
                    }

                    dSInput.clear();
                    for (int i = 0; i < smallGridSize; ++i) {
                        for (int j = 0; j < smallGridSize; ++j) {
                            dSInput.push_back(dSGrid[i][j]);
                        }
                    }
                }

                //cout << "X: " << mouse_x << " Y: " << mouse_y << "\n";
                

                window.clear(sf::Color::Black);

                // Render objects to screen
                if(!dSDisplay) {
                    sf::RectangleShape pixel(sf::Vector2f(cellSize, cellSize));
                    for (int i = 0; i < largeGridSize; ++i) {
                        for (int j = 0; j < largeGridSize; ++j) {
                            pixel.setPosition(j * cellSize, i * cellSize);
                            pixel.setFillColor(largeGrid[i][j] > 0 ? sf::Color::White : sf::Color::Black);
                            window.draw(pixel);
                        }
                    }
                }
                else {
                    sf::RectangleShape pixel(sf::Vector2f(dSCellSize, dSCellSize));
                    for (int i = 0; i < smallGridSize; ++i) {
                        for (int j = 0; j < smallGridSize; ++j) {
                            pixel.setPosition(j * dSCellSize, i * dSCellSize);
                            pixel.setFillColor(sf::Color(255, 255, 255, dSGrid[i][j] * 255));
                            window.draw(pixel);
                        }
                    }
                }
                window.display();
            }
        }
    }

    return 0;
};

void keyBoardInputs() {
    

    // Toggle dSDisplay for Faster Training
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
        if(!D)
            dSDisplay = !dSDisplay;
        D = true;
    }
    else {
        D = false;
    }

    // Toggle Epoch Feedback
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::E)) {
        if(!E)
            printEpochs = !printEpochs;
        E = true;
    }
    else {
        E = false;
    }

    // Toggle Fast Mode
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::F)) {
        if(!F)
            fast = !fast;
        F = true;
    }
    else {
        F = false;
    }

    // Print Network Information
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::N)) {
        if(!N) {
            cout << "\nName: " << "nw" << "_" << to_string(int(epochs)) << ".txt" << "\n";
            //vector<int> layers = nn.getLayers();
            //cout << "Layer Count: " << layers.size() << "\n";
            // cout << "Layers: ";
            // for(int layer : layers) {
            //     cout << layer << " ";
            // }
            cout << "\n\n";
        }
        N = true;
    }
    else {
        N = false;
    }

    // Save Network
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
        if(!S) {
            int id = rand() % 101;
            cout << "Saving network: " << "nw" << id << "_" << to_string(int(epochs)) << ".txt" << "\n";
            nn.save("../data/arc/nw" + to_string(id) + "_" + to_string(int(epochs)) + ".txt");
        }
        S = true;
    }
    else {
        S = false;
    }

    // Toggle Training Mode
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
        if(!T)
            training = !training;
        if(training)
            cout << "Training!\n";
        else {
            fast = false;
        }
        T = true;
    }
    else {
        T = false;
    }
}

vector<vector<double>> getMnistImages(const string& file_path, int num_images) {
    const int image_size = 28 * 28; // Each image is 28x28 pixels
    ifstream file(file_path, ios::binary);
    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_path);

    // Read the header
    file.ignore(16); // Skip the 16-byte header

    // Prepare a container for the images
    vector<vector<double>> images(num_images, vector<double>(image_size));

    // Read each image
    for (int i = 0; i < num_images; ++i) {
        for (int j = 0; j < image_size; ++j) {
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
            images[i][j] = pixel / 255.0f; // Normalize pixel values to [0, 1]
        }
    }
    file.close();
    return images;
}

vector<double> getMnistLabels(const string& file_path, int num_labels) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_path);

    // Read the header
    file.ignore(8); // Skip the 8-byte header

    // Prepare a container for the labels
    vector<double> labels(num_labels);

    // Read each label
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label; // Store the label
    }
    file.close();
    return labels;
}

int getNum(vector<double> outputs) {
    double largestVal = outputs[0];
    int largestIndex = 0;
    for(int i = 1; i < 10; i++) {
        if(outputs[i] > largestVal) {
            largestVal = outputs[i];
            largestIndex = i;
        }
    }
    return largestIndex;
}