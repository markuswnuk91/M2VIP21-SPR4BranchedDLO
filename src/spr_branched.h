#include "spr.h"
/** Performs the SPR algorithm for branched deformable objects
 * Use as
 *
 * ```{.cpp}
 * Eigen::MatrixX3f jointLocations;
 * pcl::PointCloud<pcl::PointXYZ> measuredPointCloud;
 * std::vector<std::vector<Eigen::Index>> branchMatrix; (each std::vector holds a list of indices as
 * std:vector for every branch)
 *
 * SPR mysprbranched(); // pass parameters, if needed
 * Xregistered = mysprbranched.computeEM(jointLocations, measuredPointCloud, branchMatrix);
 * ```
 * */
class SPRBranched : public SPR {
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
   * \arg normalization (true / false) weather the data should be normalized to zero mean and unit
   * variance before registration
   *  * */
  SPRBranched(float beta = 1, float lambda0 = 1, float tau0 = 5, std::size_t nMaxIterations = 100,
              float tolerance = 1e-5, std::size_t kNN = 20, float wOutliers = 0.1,
              float annealingFactor = 0.97, bool normalization = true);

  /** Computes the SPR.
   *
   * \arg X RBF centers (N x D)
   * \arg Y measured pointcloud (M x D)
   * \arg B branch array std::vector<std::vector<Eigen::Index>> where the indices of RBF centers
   * beloning to one branch are stored in the inner std::vector \returns points X_t fitted to
   * measurement data Y
   * */
  Eigen::MatrixX3f computeEM(Eigen::MatrixX3f &X, const pcl::PointCloud<pcl::PointXYZ>::Ptr Y,
                             std::vector<std::vector<Eigen::Index>> B);
  Eigen::MatrixX3f computeEM(Eigen::MatrixX3f &X, Eigen::MatrixX3f &Ymat,
                             std::vector<std::vector<Eigen::Index>> B);

protected:
  /** returns the modified Probabilities as P_mod.*P (elemt wise multiplication) , where P(n,m) =
   * p(n|y_m) = RBF(x_n, y_m, sigma²)/ (sum_n=1^N RBF(x_n, y_m, sigma^2)  + outlier) and P_mod(n,m)
   * = \tilde{p}(n,y_m) = sum_n_in_k RBF(x_n, y_m, sigma^2)/sum_k=1^K p(k) sum_n_in_k RBF(x_n, y_m,
   * sigma^2)
   * */
  Eigen::MatrixXf getModifiedProbabilities(const Eigen::MatrixX3f &X, const Eigen::MatrixX3f &Y,
                                           float sigmaSquared,
                                           std::vector<std::vector<Eigen::Index>> B);
};