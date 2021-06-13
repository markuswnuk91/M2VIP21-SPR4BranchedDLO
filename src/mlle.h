#pragma once

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>
#include <eigen3/Eigen/StdVector>

#include <algorithm> // std::sort, std::stable_sort
#include <numeric>   // std::iota
#include <utility>
#include <iostream>

/** For a row-wise dataset X, this method returns the row indices of nearest neighbours for X(i,:)
 * in order of increasing distance.
 *
 * \arg X N data points of dimension 3
 * \arg i row  of X to compute the distance against
 *
 * \returns indices of nearest neighbours of point i with increasing distance \in |R^(N-1)
 * \note the index i itself is excluded from the list of indices.
 * */
std::vector<Eigen::Index> findIndicesOfNearest(const Eigen::MatrixX3f &X, Eigen::Index i);

/** Returns sorted eigenvalues and corresponding eigenvectors of (G^T*G)
 *
 * The problem is calculated via SVD. G^T*G = V*Sigma^T*U^T*U*Sigma*V^T = V*(Sigma^T*Sigma)*V^T =
 * V*S*V Where G = U*S*V^T and G^T*G = X*S*X^T (eigenvalue decomposition) I.e. S = Sigma^T*Sigma and
 * X = V
 *
 * https://en.wikipedia.org/wiki/Singular_value_decomposition
 * https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix
 *
 * \note This mehtod is only efficient for small D, i.e. D<=5, though, in general D would be 3
 * (xyz).
 *
 * \arg G (D,N) matrix to decompose, where D<N
 * \returns [S,X] the eigenvalues (sorted in descending order) and eigenvectors matrix of (G^T*G)
 * */
std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecompositionSquared(const Eigen::Matrix3Xf &G);

std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecomposition(const Eigen::MatrixXf &G);

std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecompositionES(const Eigen::MatrixXf &G);
/** MLLE constructing nonlinear d-dimensional embedding in set of D-dimensional data.
 *
 * Following [Zhang2007]: Zhang, Wang (2007): MLLE - Modified Locally Linear Embedding Using
 * Multiple Weights
 *
 * \arg X data set (N x D)  (N measured points of dimension D)
 * \arg k nearest neighbours to incorporate in calculation of MLLE
 * \arg d reduced dimensionality
 * \returns Phi
 * */
Eigen::MatrixXf MLLE(const Eigen::MatrixX3f &X, std::size_t k, std::size_t d);