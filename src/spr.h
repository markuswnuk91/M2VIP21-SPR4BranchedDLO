#pragma once
#include <eigen3/Eigen/Eigen>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


#include "gmm.h"
#include "mlle.h"

#include <iostream>
/** Performs the SPR algorithm accoring to Tang Te et al. (2019): Track deformable objects from
 * point clouds with structure preserved registration
 *
 * https://journals.sagepub.com/doi/pdf/10.1177/0278364919841431
 *
 * Use as
 *
 * ```{.cpp}
 * Eigen::MatrixX3f jointLocations;
 * pcl::PointCloud<pcl::PointXYZ> measuredPointCloud;
 *
 * SPR myspr(); // pass parameters, if needed
 * Xregistered = myspr.computeEM(jointLocations, measuredPointCloud);
 * ```
 * */
class SPR {
public:
  /**
   *
   * \arg beta width of Gaussian kernel
   * \arg lambda0 (1..5) regularization of global topology (lambda >> 1 => less motion)
   * \arg tau0 (1..5) regularization of local topology
   * \arg nMaxIterations maximum number of iterations before termination
   * \arg tolerance stopping tolerance to decide if converged (>0)
   * \arg kNN number of nearest neighbours to include in MLLE algorithm (should be < number RBF
   * centroids)
   * \arg wOutliers weightign for noise and outliers \in [0,1]
   * \arg annealingFactor \in
   * (0,1] (e.g. 0.97) lambda and tau are multiplied by this factor in each iteration
   * \arg normalization (true / false) weather the data should be normalized to zero mean and unit variance before registration
   *  * */
  SPR(float beta = 1, float lambda0 = 1, float tau0 = 5, std::size_t nMaxIterations = 100,
      float tolerance = 1e-5, std::size_t kNN = 20, float wOutliers = 0.1,
      float annealingFactor = 0.97, bool normalization = true);

  /** Computes the SPR.
   *
   * \arg X RBF centers (N x D)
   * \arg Y measured pointcloud (M x D)
   * \returns points X_t fitted to measurement data Y
   * */
  Eigen::MatrixX3f computeEM(Eigen::MatrixX3f &X,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr Y);

  inline void setVerbose(bool verbosity) { isVerbose = verbosity; }

protected:
  float beta;
  float tau0;
  float lambda0;
  std::size_t nMaxIterations;
  float tolerance;
  float kNN;
  float wOutliers;
  float annealingFactor;
  bool normalization;

  bool isVerbose = false;

  /** returns the Probabilities matrix P, where P(n,m) = p(n|y_m) = RBF(x_n, y_m, sigma²)/
   * (sum_k=1^N RBF(x_k, y_m, sigma^2)  + outlier)
   *
   * Source: Tang Te et al. (2019): Track deformable objects from point clouds with structure
   * preserved registration
   * */
  Eigen::MatrixXf getProbabilities(const Eigen::MatrixX3f &X, const Eigen::MatrixX3f &Y,
                                   float sigmaSquared);
  Eigen::MatrixX3f normalize(const Eigen::MatrixX3f &X ,const Eigen::RowVector3f &mean, 
                                   double scaleFactor);
  std::pair<Eigen::RowVector3f,double> calculateMeanAndScaleFactor(const Eigen::MatrixX3f &X);
};