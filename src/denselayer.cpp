// Filename: denselayer.cpp
// Author: Jonah Taylor
// Date: December 25, 2024
// Description: DenseLayer class implementation

#include "denselayer.hpp"

DenseLayer::DenseLayer() {

    w = MatrixXd::Random(0, 0);
    b = MatrixXd::Zero(0, 0);
    z = MatrixXd::Zero(0, 0);
    a = MatrixXd::Zero(0, 0);

    dz = MatrixXd::Zero(0, 0);
    avg_grad_w = MatrixXd::Zero(0, 0);
    avg_grad_b = MatrixXd::Zero(0, 0);

    m_w = MatrixXd::Zero(0, 0);
    v_w = MatrixXd::Zero(0, 0);
    m_b = MatrixXd::Zero(0, 0);
    v_b = MatrixXd::Zero(0, 0);

};

DenseLayer::DenseLayer(size_t ls, size_t in_size, string afn) : l_size(ls) {

    w = MatrixXd::Random(l_size, in_size) * sqrt(2.0 / in_size);
    b = MatrixXd::Zero(l_size, 1);
    a = MatrixXd::Zero(l_size, 1);
    z = MatrixXd::Zero(l_size, 1);

    dz = MatrixXd::Zero(l_size, 1);
    avg_grad_w = MatrixXd::Zero(l_size, in_size);
    avg_grad_b = MatrixXd::Zero(l_size, 1);

    m_w = MatrixXd::Zero(l_size, in_size);
    v_w = MatrixXd::Zero(l_size, in_size);
    m_b = MatrixXd::Zero(l_size, 1);
    v_b = MatrixXd::Zero(l_size, 1);

    setActFunc(afn);

};

MatrixXd DenseLayer::forward(const MatrixXd& in) {

    z = w * in + b;
    a = a_func(z);
    return a;

}

void DenseLayer::getOutputDeltas(const MatrixXd& target) {

    MatrixXd error = a - target;
    dz = error.array().cwiseProduct(a_func_deri(z).array());

}

void DenseLayer::backward(const MatrixXd& w_next, const MatrixXd& d_next) {
    
    dz = (w_next.transpose() * d_next).array() * a_func_deri(z).array();

}

void DenseLayer::updateGrads(const MatrixXd& in) {
    
    avg_grad_w += dz * in.transpose();
    avg_grad_b += dz;

}

void DenseLayer::stepSGD(const double& lr, const size_t& bs) {

    w -= (lr * avg_grad_w.array() / bs).matrix();
    b -= (lr * avg_grad_b.array() / bs).matrix();

    avg_grad_w = MatrixXd::Zero(avg_grad_w.rows(), avg_grad_w.cols());
    avg_grad_b = MatrixXd::Zero(avg_grad_b.rows(), avg_grad_b.cols());

}

void DenseLayer::stepAdamW(const double& lr, const size_t& bs, size_t& t) {

    MatrixXd grad_w = avg_grad_w / bs;
    MatrixXd grad_b = avg_grad_b / bs;

    m_w = (beta1 * m_w + (1 - beta1) * grad_w).matrix(); 
    v_w = beta2 * v_w + (1 - beta2) * grad_w.array().square().matrix();
    
    m_b = (beta1 * m_b + (1 - beta1) * grad_b).matrix(); 
    v_b = beta2 * v_b + (1 - beta2) * grad_b.array().square().matrix(); 
    
    MatrixXd m_hat_w = m_w / (1 - pow(beta1, t));
    MatrixXd v_hat_w = v_w / (1 - pow(beta2, t));

    MatrixXd m_hat_b = m_b / (1 - pow(beta1, t));
    MatrixXd v_hat_b = v_b / (1 - pow(beta2, t));

    w -= (lr * (m_hat_w.array() / (v_hat_w.array().sqrt() + epsilon)).matrix() + lambda * w).matrix();
    b -= (lr * (m_hat_b.array() / (v_hat_b.array().sqrt() + epsilon)).matrix() + lambda * b).matrix();

    avg_grad_w = MatrixXd::Zero(avg_grad_w.rows(), avg_grad_w.cols());
    avg_grad_b = MatrixXd::Zero(avg_grad_b.rows(), avg_grad_b.cols());
    
}

pair<size_t, size_t> DenseLayer::size() {

    return {l_size, l_size};

}