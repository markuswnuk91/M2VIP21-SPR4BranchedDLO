#pragma once

#include <eigen3/Eigen/Eigen>
#include <eigen3/unsupported/Eigen/KroneckerProduct>

/** Computes squared distances between M measurements and the N RBF centers (each of
 * dimensionality D).
 *
 * \arg X (NxD) Matrix of gaussian centroids
 * \arg Y (MxD) Matrix of measuements
 * \returns Matrix of (N x M) squared distances
 * */
Eigen::MatrixXf computeSquaredDistances(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y);

/** Computes gaussian Kernel matrix G, s.t. G_ij = exp(-||x_i -y_j||^2/(2*beta^2) )
 *
 * \arg X (NxD) Matrix of gaussian centroids
 * \arg Y (MxD) Matrix of measuements
 * \arg beta kernel width
 *
 * \returns Gaussian kernel matrix G \in |R^(N x M)
 *
 * \note If X==Y, this method returns a Gaussian Gram matrix.
 * */
Eigen::MatrixXf computeGaussianKernels(const Eigen::MatrixXf &X, const Eigen::MatrixXf &Y,
                                       float beta);
