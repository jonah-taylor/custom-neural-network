// Filename: PongAI.cpp
// Author: Jonah Taylor
// Date: 9 Nov 2024
// Description: A Neural Network that does Pong

#include <iostream>
#include <deque>

#include "pseument.hpp"
#include "SFML/Graphics.hpp"
#include "SFML/Window.hpp"
#include "SFML/System.hpp"

const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 1000;
const float PADDLE_WIDTH = 10.f;
const float PADDLE_HEIGHT = 100.f;
const float BALL_SIZE = 10.f;
const float BALL_SPEED_X = 16;

int leftScore = 0;
int rightScore = 0;
int winRateLast100 = 0;
std::deque<int> last100(100, 0);
int detectFPS = 0; // Frames Per Second
float FPS = 60;

bool A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, Z = false; // Save X and Y for Inputs and Predictions
bool training = true;
bool fast = false;
bool display = true;
bool printEpochs = true;

NeuralNetwork nn({
    MakeLayer({6}, "dense", "leakyrelu"), 
    MakeLayer({30}, "dense", "leakyrelu"),
    MakeLayer({10}, "dense", "leakyrelu"),
    MakeLayer({1}, "dense", "leakyrelu")
});
std::vector<double> inputs;
std::vector<std::vector<double>> X;
std::vector<std::vector<double>> Y;
size_t epochs = 1;
size_t batchSize = 1;
double trainingSpeed = 0.0001;

void keyBoardInputs();


class Paddle : public sf::RectangleShape {
public:
    float velocity = 0;

    Paddle(float x, float y) {
        setSize(sf::Vector2f(PADDLE_WIDTH, PADDLE_HEIGHT));
        setPosition(x, y);
    }
    void update() {
        move({0, velocity});
    }

};

class Ball : public sf::CircleShape {
public:
    sf::Vector2f velocity = sf::Vector2f(BALL_SPEED_X, 3);

    Ball(float radius) : sf::CircleShape(radius) {
        setFillColor(sf::Color::White);
        setPosition(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2);
    }

    void update() {
        move(velocity);

        if (getPosition().y <= 0) {
            velocity.y = abs(velocity.y);  // Bounce off top and bottom walls
        }
        else if(getPosition().y + BALL_SIZE * 2 >= WINDOW_HEIGHT) {
            velocity.y = -abs(velocity.y);
        }
    }

    void reset() {
        setPosition(WINDOW_WIDTH / 10 - BALL_SIZE / 2, WINDOW_HEIGHT / 2 - BALL_SIZE / 2);
        velocity = sf::Vector2f(BALL_SPEED_X, std::rand() % 30 - 15);  // Reset velocity
    }
};

// Screen objects
Paddle leftPaddle(50.f, WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2);
Paddle rightPaddle(WINDOW_WIDTH - 50.f - PADDLE_WIDTH, WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2);
Ball ball(BALL_SIZE);


int main(int argc, char* argv[]) {

    // Load network if used as argument
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Loading network: " << filename << "\n";
        nn.load("../data/arc/" + filename + ".txt");
    }

    sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "", sf::Style::None);
    window.setTitle("Pong " + std::to_string(leftScore) + " to " + std::to_string(rightScore) + "  |  AI WR: " + 
                    std::to_string(winRateLast100) + " / 100  |  detectFPS = " + std::to_string(detectFPS));

    sf::Image icon;
    if (!icon.loadFromFile("Images/icon.png")) {
        std::cerr << "Error loading icon\n";
    }
    else {
        window.setIcon(icon.getSize().x, icon.getSize().y, icon.getPixelsPtr());
    }

    // Create random seed
    std::srand(std::time(0));

    // Give AI unique color
    rightPaddle.setFillColor(sf::Color::Green);

    // Set up clock for frame timing
    sf::Clock clock;
    const sf::Time targetTime = sf::seconds(1.f / FPS);

    // Main game loop
    while (window.isOpen()) {
        sf::Time elapsed = clock.getElapsedTime();
 
        // Process frame every 1/FPS of a second or, if training, as fast as possible
        if (elapsed >= targetTime || fast) {
            detectFPS = 1 / (elapsed.asSeconds());
            clock.restart();

            // Close window when exiting the program
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    // Auto Save
                    std::cout << "Saving network: " << "exit" << std::to_string(winRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
                    nn.save("../data/arc/exit" + std::to_string(winRateLast100) + "_" + std::to_string(int(epochs)) + ".txt");
                    window.close();
                }
            }

            keyBoardInputs();

            leftPaddle.velocity = 0;
            rightPaddle.velocity = 0;
            double error = ((std::rand() % int(PADDLE_HEIGHT)) - PADDLE_HEIGHT / 2); // Gives algorithm paddle a slightly unique position to target

            // Handle left paddle movement
            if(training) {
                // Algorithm Player
                if (leftPaddle.getPosition().y > ball.getPosition().y + BALL_SIZE / 2 - PADDLE_HEIGHT / 2 + error && leftPaddle.getPosition().y > 0)
                    leftPaddle.velocity = -10;
                if (leftPaddle.getPosition().y < ball.getPosition().y + BALL_SIZE / 2 - PADDLE_HEIGHT / 2 + error && leftPaddle.getPosition().y + PADDLE_HEIGHT <= WINDOW_HEIGHT)
                    leftPaddle.velocity = 10;
            }
            else {
                // Human Player
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up) && leftPaddle.getPosition().y > 0)
                    leftPaddle.velocity = -10;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down) && leftPaddle.getPosition().y + PADDLE_HEIGHT < WINDOW_HEIGHT)
                    leftPaddle.velocity = 10;
            }

            // Handle right paddle movement
            inputs = {
                (rightPaddle.getPosition().y + PADDLE_HEIGHT / 2) / WINDOW_HEIGHT, // right paddle position y
                rightPaddle.getPosition().x / WINDOW_WIDTH, // right paddle position x
                (ball.getPosition().y + BALL_SIZE / 2) / WINDOW_HEIGHT, // ball position y
                (ball.getPosition().x + BALL_SIZE / 2) / WINDOW_WIDTH, // ball position x
                (ball.getPosition().y + ball.velocity.y) / WINDOW_HEIGHT - ball.getPosition().y / WINDOW_HEIGHT, // ball velocity y
                (ball.getPosition().x + ball.velocity.x) / WINDOW_WIDTH - ball.getPosition().x / WINDOW_WIDTH // ball velocity x
            };

            // Store inputs for training
            if(ball.velocity.x > 0) {
                X.push_back(inputs);
            }

            // Decide which direction to move
            std::vector<double> output = nn.forward(inputs);

            if (output[0] > 0.5 && rightPaddle.getPosition().y > 0) {
                rightPaddle.velocity = -10;
            }
            else if (output[0] <= 0.5 && rightPaddle.getPosition().y + PADDLE_HEIGHT < WINDOW_HEIGHT) {
                rightPaddle.velocity = 10;
            }

            // Paddle movement
            leftPaddle.update();
            rightPaddle.update();

            // Ball collides with left paddle
            ball.update();
            if (ball.getGlobalBounds().intersects(leftPaddle.getGlobalBounds()) && ball.velocity.x < 0) {
                ball.velocity.x = -ball.velocity.x;
                ball.velocity.y = (ball.getPosition().y + BALL_SIZE / 2 - PADDLE_HEIGHT / 2 - leftPaddle.getPosition().y) / 4;
            }

            // Ball collides with right paddle
            if(ball.getGlobalBounds().intersects(rightPaddle.getGlobalBounds()) && ball.velocity.x > 0) {
                ball.velocity.x = -ball.velocity.x;
                ball.velocity.y = (ball.getPosition().y + BALL_SIZE / 2 - PADDLE_HEIGHT / 2 - rightPaddle.getPosition().y) / 4;
            }

            // Left paddle misses ball
            if (ball.getPosition().x < 0) {

                // Score Adjustment
                rightScore++;
                winRateLast100++;
                winRateLast100 -= last100.front();
                last100.pop_front();
                last100.push_back(1);
                window.setTitle("Pong " + std::to_string(leftScore) + " to " + std::to_string(rightScore) + "  |  AI WR: " + 
                    std::to_string(winRateLast100) + " / 100  |  detectFPS = " + std::to_string(detectFPS));

                ball.reset(); 
            }

            // Right paddle misses ball
            if(ball.getPosition().x + BALL_SIZE > WINDOW_WIDTH) {

                // Score Adjustment
                leftScore++;
                winRateLast100 -= last100.front();
                last100.pop_front();
                last100.push_back(0);
                window.setTitle("Pong " + std::to_string(leftScore) + " to " + std::to_string(rightScore) + "  |  AI WR: " + 
                    std::to_string(winRateLast100) + " / 100  |  detectFPS = " + std::to_string(detectFPS));

                // Train the neural network
                if(training) {
                    for(std::vector<double> input: X) {
                        if(input[0] > (ball.getPosition().y + BALL_SIZE / 2) / WINDOW_HEIGHT) // If right paddle height > ball height
                            Y.push_back({1});
                        else
                            Y.push_back({0});
                    }
                    nn.train(X, Y, epochs, batchSize, trainingSpeed, "adamw", printEpochs);
                    X.clear();
                    Y.clear();
                }

                ball.reset();
            }

            // Render objects to screen
            if(display) {
                window.clear(sf::Color::Black);
                window.draw(leftPaddle);
                window.draw(rightPaddle);
                window.draw(ball);
                window.display();
            }
        }
    }

    return 0;
};

void keyBoardInputs() {

    // Toggle Display for Faster Training
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
        if(!D)
            display = !display;
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
            std::cout << "\nName: " << "nw" << std::to_string(winRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
            std::vector<size_t> layer_sizes = nn.getLayerSizes();
            std::cout << "Layer Count: " << layer_sizes.size() << "\n";
            std::cout << "Layers: ";
            for(int layer : layer_sizes) {
                std::cout << layer << " ";
            }
            std::cout << "\n\n";
        }
        N = true;
    }
    else {
        N = false;
    }

    // Save Network
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
        if(!S) {
            std::cout << "Saving network: " << "nw" << std::to_string(winRateLast100) << "_" << std::to_string(int(epochs)) << ".txt" << "\n";
            nn.save("../data/arc/nw" + std::to_string(winRateLast100) + "_" + std::to_string(int(epochs)) + ".txt");
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
            leftPaddle.setFillColor(sf::Color::Red);
        else {
            leftPaddle.setFillColor(sf::Color::White);
            fast = false;
        }
        T = true;
    }
    else {
        T = false;
    }

}
