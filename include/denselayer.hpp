#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "Eigen/Dense"
#include "layer.hpp"

using namespace std;
using namespace Eigen;

class DenseLayer : public Layer {

public:

    size_t l_size = 0;

    DenseLayer();
    DenseLayer(size_t ls, size_t in_size, string afn);

    MatrixXd forward(const MatrixXd& in) override;
    void getOutputDeltas(const MatrixXd& target) override;
    void backward(const MatrixXd& w_next, const MatrixXd& d_next) override;
    void updateGrads(const MatrixXd& in) override;
    void stepSGD(const double& lr, const size_t& bs) override;
    void stepAdamW(const double& lr, const size_t& bs, size_t& t) override;
    pair<size_t, size_t> size() override;
    
    ~DenseLayer() = default;
};

#endif