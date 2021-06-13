#include "spr.h"
// #include "csv_utils.h" // uncomment, if csv files have to be loaded, e.g. for testing an external Phi matrix

SPR::SPR(float beta, float lambda0, float tau0, std::size_t nMaxIterations, float tolerance,
         std::size_t kNN, float wOutliers, float annealingFactor, bool normalization)
    : beta(beta), tau0(tau0), lambda0(lambda0), nMaxIterations(nMaxIterations),
      tolerance(tolerance), kNN(kNN), wOutliers(wOutliers), annealingFactor(annealingFactor),
      normalization(normalization) {}

Eigen::MatrixX3f SPR::computeEM(Eigen::MatrixX3f &X,
                                const pcl::PointCloud<pcl::PointXYZ>::Ptr Y) {

  if (isVerbose) {
    std::cout << std::endl
              << "--------------------------------------------------------" << std::endl
              << "SPR algorithm start. " << std::endl
              << "Params: - beta = " << beta << std::endl
              << "        - tau0 = " << tau0 << std::endl
              << "        - lambda0 = " << lambda0 << std::endl
              << "        - nMaxIterations = " << nMaxIterations << std::endl
              << "        - tolerance = " << tolerance << std::endl
              << "        - kNN = " << kNN << std::endl
              << "        - wOutliers = " << wOutliers << std::endl
              << "        - annealingFactor = " << annealingFactor << std::endl
              << "        - normalization = " << normalization << std::endl
              << "--------------------------------------------------------" << std::endl;
  }

  const auto N = X.rows();
  const auto M = Y->size();
  const Eigen::Index D = X.cols();

  Eigen::MatrixX3f Ymat = Y->getMatrixXfMap(3, 4, 0).transpose();

  Eigen::RowVector3f X_mean;
  Eigen::RowVector3f Y_mean;
  double X_scaleFactor;
  double Y_scaleFactor;
  if (normalization){
    //normalize datasets
    std::pair<Eigen::RowVector3f,double> X_norm = calculateMeanAndScaleFactor(X);
    X_mean = X_norm.first;
    X_scaleFactor = X_norm.second;

    std::pair<Eigen::RowVector3f,double> Y_norm = calculateMeanAndScaleFactor(Ymat);
    Y_mean = Y_norm.first;
    Y_scaleFactor = Y_norm.second;
    
    X = normalize(X, X_mean, X_scaleFactor);
    Ymat = normalize(Ymat, Y_mean, Y_scaleFactor);
  }



  std::size_t nIterations = 0;                       // number of iterations
  float nTol = tolerance + 10;                       // relative convergence factor
  Eigen::MatrixX3f W = Eigen::MatrixX3f::Zero(N, 3); // weights
  Eigen::MatrixX3f X_t = X;                          // registered gaussian centroid

  float sigmaSquared = (N * (Ymat.transpose() * Ymat).trace() + M * (X.transpose() * X).trace() -
                        2 * X.colwise().sum() * Ymat.colwise().sum().transpose()) /
                       (M * N * D);

  auto G = computeGaussianKernels(X, X, beta);
  auto Phi = MLLE(X, kNN, D);
  // auto Phi = load_csv_to_eigen<Eigen::MatrixXf>("data/Phi_ml.csv"); // use this for external Phi matrix instead of calculating via MLLE

  float Q_tilde = 1;

  while ((nIterations < nMaxIterations) && (nTol > tolerance) && (sigmaSquared > 1e-8)) {

    if (isVerbose) {
      std::cout << "it: " << nIterations << "/" << nMaxIterations;
    }

    // annealing of tau, lambda
    float tau = tau0 * (float)std::pow(annealingFactor, nIterations);
    float lambda = lambda0 * (float)std::pow(annealingFactor, nIterations);

    float Q_tilde_old = Q_tilde;
    // P-Step
    auto P = getProbabilities(X_t, Ymat, sigmaSquared);
    Eigen::MatrixXf PY = P * Ymat;
    Eigen::VectorXf P1 = P.rowwise().sum();       // (N x 1)
    Eigen::VectorXf Ptransp1 = P.colwise().sum(); // (M x 1)
    float Np = P1.sum();

    Q_tilde = -1 / (2 * sigmaSquared) * (Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace() +
              1 / sigmaSquared * (X_t.transpose()  * P * Ymat).trace() -
              1 / (2 * sigmaSquared) * (X_t.transpose() * P1.asDiagonal() * X_t).trace() -
              Np * D / 2 * log(sigmaSquared) - lambda / 2 * (W.transpose() * G * W).trace() -
              tau / 2 * ( X_t.transpose() * Phi * X_t).trace();
    nTol = abs((Q_tilde - Q_tilde_old) / Q_tilde);

    // M-Step
    // solve LHS*W = RHS
    Eigen::MatrixXf leftHandSide = (P1.asDiagonal() * G + lambda * sigmaSquared * Eigen::MatrixXf::Ones(N, N) +
         tau * sigmaSquared * Phi * G);
    Eigen::MatrixXf rightHandSide = PY - P1.asDiagonal() * X  - tau * sigmaSquared * Phi * X;
    W = leftHandSide.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rightHandSide);

    X_t = X + G * W;

    sigmaSquared = abs((( Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace() 
                        - 2 * (X_t.transpose() * PY).trace() +
                        (X_t.transpose() * P1.asDiagonal() * X_t).trace()) /
                       (Np * D));

    if (isVerbose) {
      std::cout << ", Q_tilde: " << Q_tilde << ", nTol: " << nTol
                << ", sigmaSquared: " << sigmaSquared << std::endl;
    }
    nIterations++;
  }

  if (isVerbose) {
    std::cout << "SPR terminated.";
    if (nIterations >= nMaxIterations)
      std::cout << "Maximum number of iterations (" << nMaxIterations << ") reached." << std::endl;
    else if (nTol <= tolerance)
      std::cout << "Q_tilde converged (abs(Q(t) - Q(t-1))/Q(t)<=" << tolerance << ")." << std::endl;
    else if (sigmaSquared <= 1e-8)
      std::cout << "sigma^2=" << sigmaSquared << " was too small (<1e-8)." << std::endl;
    else
      std::cout << "Unknown reason." << std::endl;
  }

  if (normalization){
    // denormalization
    X_t = X_t*Y_scaleFactor;
    X_t += Y_mean.replicate(N,1);
  }


  return X_t;
}

Eigen::MatrixXf SPR::getProbabilities(const Eigen::MatrixX3f &X, const Eigen::MatrixX3f &Y,
                                      float sigmaSquared) {
  const auto N = X.rows();
  const auto M = Y.rows();
  const Eigen::Index D = 3;
  float mu = wOutliers;

  // Gaussian kernel matrix G \in |R^(N x M)
  auto rbfEval =
      computeGaussianKernels(X, Y, sqrt(sigmaSquared)); // this is the nominator of matrix P
  Eigen::RowVectorXf P_normalization =
      rbfEval.colwise().sum().array() +
      (float)std::pow(2 * M_PI * sigmaSquared, D / 2) * mu * N / ((1 - mu) * M);
  return rbfEval.cwiseQuotient(P_normalization.replicate(N, 1)); // = P
}

std::pair<Eigen::RowVector3f,double> SPR::calculateMeanAndScaleFactor(const Eigen::MatrixX3f &X){
  // variable decalaration
  Eigen::RowVector3f mean;
  double scaleFactor;

  const auto N = X.rows();
  mean = X.colwise().mean();
  Eigen::MatrixX3f x = X - mean.replicate(N, 1);
  scaleFactor = sqrt(x.cwiseProduct(x).sum()/N);
  std::pair<Eigen::RowVector3f,double> meanAndScaleFactor(mean, scaleFactor);
  return meanAndScaleFactor;
}

Eigen::MatrixX3f SPR::normalize(const Eigen::MatrixX3f &X ,const Eigen::RowVector3f &mean, double scaleFactor){
  Eigen::MatrixX3f X_scaled;
  const auto N = X.rows();
  X_scaled = X;
  X_scaled -= mean.replicate(N, 1);
  X_scaled /= scaleFactor;
  return X_scaled;
}