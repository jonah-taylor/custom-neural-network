#ifndef LAYER_HPP
#define LAYER_HPP

#include "Eigen/Dense"

#include <functional>
#include <iostream>

using namespace Eigen;
using namespace std;

class Layer {

public:

    MatrixXd w, b, z, a;
    MatrixXd dz, avg_grad_w, avg_grad_b;
    MatrixXd m_w, v_w, m_b, v_b;

    double beta1 = 0.9;       // Exponential decay rate for first moment
    double beta2 = 0.999;     // Exponential decay rate for second moment
    double epsilon = 1e-8;   // Small constant to avoid division by zero
    double lambda = 0.0;     // Weight decay coefficient (Set to 1% of learning rate)

    string a_func_name = "leakyrelu";
    function<MatrixXd(const MatrixXd&)> a_func;
    function<MatrixXd(const MatrixXd&)> a_func_deri;

    void setActFunc(const std::string& func_name) {

        a_func_name = func_name;

        if (func_name == "leakyrelu") {
            a_func = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) {
                    return v > 0 ? v : 0.01 * v;
                });
            };
            a_func_deri = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) { 
                    return v > 0 ? 1 : 0.01; 
                });
            };
        } 

        else if (func_name == "sigmoid") {
            a_func = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) {
                    return 1.0 / (1.0 + exp(-v)); 
                });
            };
            a_func_deri = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) {
                    double sig = 1.0 / (1.0 + exp(-v));
                    return sig * (1.0 - sig);
                });
            };
        } 

        else if (func_name == "tanh") {
            a_func = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) { 
                    return std::tanh(v); 
                });
            };
            a_func_deri = [](const MatrixXd& x) {
                return x.unaryExpr([](double v) {
                    double t = std::tanh(v);
                    return 1.0 - t * t;
                });
            };
        }

        else {
            throw std::invalid_argument("Unknown activation function");
        }

    }

    virtual MatrixXd forward(const MatrixXd& in) = 0;
    virtual void getOutputDeltas(const MatrixXd& target) = 0;
    virtual void backward(const MatrixXd& w_next, const MatrixXd& d_next) = 0;
    virtual void updateGrads(const MatrixXd& in) = 0;
    virtual void stepSGD(const double& lr, const size_t& bs) = 0;
    virtual void stepAdamW(const double& lr, const size_t& bs, size_t& t) = 0;
    virtual pair<size_t, size_t> size() = 0;

    virtual ~Layer() = default;
    
};

#endif