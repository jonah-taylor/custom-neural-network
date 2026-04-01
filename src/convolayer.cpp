// Filename: convolayer.cpp
// Author: Jonah Taylor
// Date: December 21, 2024
// Description: My first layer based Neural Network

#include "convolayer.hpp"

ConvoLayer::ConvoLayer() {

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

ConvoLayer::ConvoLayer(size_t r, size_t c, string afn, size_t ks) : rows(r), cols(c), k_size(ks) {

    w = MatrixXd::Random(k_size, k_size) * sqrt(2.0 / (k_size * k_size));
    b = MatrixXd::Zero(rows, cols);
    z = MatrixXd::Zero(rows, cols);
    a = MatrixXd::Zero(rows, cols);

    dz = MatrixXd::Zero(rows, cols);
    avg_grad_w = MatrixXd::Zero(k_size, k_size);
    avg_grad_b = MatrixXd::Zero(rows, cols);

    m_w = MatrixXd::Zero(k_size, k_size);
    v_w = MatrixXd::Zero(k_size, k_size);
    m_b = MatrixXd::Zero(rows, cols);
    v_b = MatrixXd::Zero(rows, cols);

    setActFunc(afn);

};

MatrixXd ConvoLayer::convolve(const MatrixXd& in, const MatrixXd& k) {

    if(w.rows() % 2 == 0) std::cerr << "Kernel must have an odd number of rows and columns!" << std::endl;
    if(w.rows() != w.cols()) std::cerr << "The kernel must be a square!" << std::endl;

    size_t padding = (k.rows()) / 2;
    MatrixXd padded_input = MatrixXd::Zero(in.rows() + 2 * padding, in.cols() + 2 * padding);
    padded_input.block(padding, padding, in.rows(), in.cols()) = in;
    MatrixXd out = MatrixXd::Zero(in.rows(), in.cols());
    
    for (int r = 0; r < out.rows(); r++) {
        for (int c = 0; c < out.cols(); c++) {
            out(r, c) = (padded_input.block(r, c, k.rows(), k.cols()).array() * k.array()).sum();
        }
    }

    return out;

}

MatrixXd ConvoLayer::forward(const MatrixXd& in) {

    MatrixXd in_mapped = Map<const Matrix<double, 28, 28>>(in.data());

    MatrixXd z_rect = convolve(in_mapped, w) + b;
    // cout << b << endl;
    // MatrixXd z_rect = b;
    z = Map<Matrix<double, 784, 1>>(z_rect.data());

    MatrixXd a_rect = a_func(z);
    a = Map<Matrix<double, 784, 1>>(a_rect.data());

    return a;

}

void ConvoLayer::getOutputDeltas(const MatrixXd& target) {

    // MatrixXd a_rect = Map<Matrix<double, 28, 28>>(a.data());
    // MatrixXd t_rect = Map<const Matrix<double, 28, 28>>(target.data());

    MatrixXd error = a - target;
    dz = error.array().cwiseProduct(a_func_deri(z).array());
    
}

void ConvoLayer::backward(const MatrixXd& w_next, const MatrixXd& d_next) {

    dz = (w_next.transpose() * d_next).array() * a_func_deri(z).array();

}

void ConvoLayer::updateGrads(const MatrixXd& in) {

    MatrixXd in_rect = Map<const Matrix<double, 28, 28>>(in.data());
    MatrixXd dz_rect = Map<const Matrix<double, 28, 28>>(in.data());
    MatrixXd dz_rect_flipped = dz_rect.colwise().reverse().rowwise().reverse();

    // Perform the convolution to calculate the gradients
    for (int i = 0; i <= in_rect.rows() - k_size; ++i) {
        for (int j = 0; j <= in_rect.cols() - k_size; ++j) {
            avg_grad_w += (in_rect.block(i, j, k_size, k_size).array() * dz_rect_flipped.block(i, j, k_size, k_size).array()).matrix();
        }
    }
    avg_grad_b += dz_rect;

}

void ConvoLayer::stepSGD(const double& lr, const size_t& bs) {

    // cout << "Dimensions " << w.rows() << " " << w.cols() << endl;
    // cout << "Dimensions " << avg_grad_w.rows() << " " << avg_grad_w.cols() << endl;
    // cout << "Dimensions " << b.rows() << " " << b.cols() << endl;
    // cout << "Dimensions " << avg_grad_b.rows() << " " << avg_grad_b.cols() << endl;

    w -= (lr * avg_grad_w.array() / bs).matrix();
    b -= (lr * avg_grad_b.array() / bs).matrix();

    avg_grad_w = MatrixXd::Zero(k_size, k_size);
    avg_grad_b = MatrixXd::Zero(rows, cols);

}

void ConvoLayer::stepAdamW(const double& lr, const size_t& bs, size_t& t) {

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

    avg_grad_w = MatrixXd::Zero(k_size, k_size);
    avg_grad_b = MatrixXd::Zero(rows, cols);
    
}

pair<size_t, size_t> ConvoLayer::size() {

    return {rows, cols};

}