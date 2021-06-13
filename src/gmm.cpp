#include "gmm.h"

Eigen::MatrixXf computeSquaredDistances(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y) {
  auto N = X.rows();
  auto M = Y.rows();
  // auto D = X.cols(); // unused
  assert(Y.cols() == X.cols());

  Eigen::MatrixXf dSqr = Eigen::MatrixXf(N, M);
  // ([x1-y1, x2-y1, ..., x_N -y1, x1-y2, ..., x_N-y2, ... x_(N-1)-y_M, x_N -y_M]).squareNorm()
  Eigen::Matrix3Xf diffs = (X.transpose().replicate(1, M) -
                            Eigen::kroneckerProduct(Y.transpose(), Eigen::MatrixXf::Ones(1, N)));
  Eigen::VectorXf dSqr_vec = diffs.colwise().squaredNorm();
  dSqr = Eigen::Map<Eigen::MatrixXf>(
      dSqr_vec.data(), N,
      M); // TODO: just use reshaping with Eigen3.4 (not yet released, 20.08.2020)
  return dSqr;
}

Eigen::MatrixXf computeGaussianKernels(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y,
                                       float beta) {
  //   auto N = X.rows();
  //   auto M = Y.rows();
  auto D = X.cols();
  assert(Y.cols() == D);

  Eigen::MatrixXf G = (-computeSquaredDistances(X, Y) / (2 * beta * beta)).array().exp();
  return G;
}