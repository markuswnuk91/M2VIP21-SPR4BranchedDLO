#include "spr_branched.h"

SPRBranched::SPRBranched(float beta, float lambda0, float tau0, std::size_t nMaxIterations,
                         float tolerance, std::size_t kNN, float wOutliers, float annealingFactor,
                         bool normalization)
    : SPR(beta, lambda0, tau0, nMaxIterations, tolerance, kNN, wOutliers, annealingFactor,
          normalization) {}

Eigen::MatrixX3f SPRBranched::computeEM(Eigen::MatrixX3f &X,
                                        const pcl::PointCloud<pcl::PointXYZ>::Ptr Y,
                                        std::vector<std::vector<Eigen::Index>> B) {

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
    std::cout << "This is the stored branch array:" << std::endl;
    std::vector<Eigen::Index> branchIndices;
    for (std::size_t i = 0; i < B.size(); ++i) {
      branchIndices = B[i];
      std::cout << i + 1 << ". Branch" << std::endl;
      for (std::size_t j = 0; j < branchIndices.size(); ++j) {
        std::cout << branchIndices[j] << " ";
      }
      std::cout << "/" << std::endl;
    }
  }

  const auto N = X.rows();
  const auto M = Y->size();
  const Eigen::Index D = X.cols();

  Eigen::MatrixX3f Ymat = Y->getMatrixXfMap(3, 4, 0).transpose();

  Eigen::RowVector3f X_mean;
  Eigen::RowVector3f Y_mean;
  double X_scaleFactor;
  double Y_scaleFactor;
  if (normalization) {
    // normalize datasets
    std::pair<Eigen::RowVector3f, double> X_norm = calculateMeanAndScaleFactor(X);
    X_mean = X_norm.first;
    X_scaleFactor = X_norm.second;

    std::pair<Eigen::RowVector3f, double> Y_norm = calculateMeanAndScaleFactor(Ymat);
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
  // auto Phi = load_csv_to_eigen<Eigen::MatrixXf>("data/Phi_ml.csv"); // use this for external Phi
  // matrix instead of calculating via MLLE

  // std::cout << "MLLE:\n" << Phi << std::endl;
  float Q_tilde = 1;

  while ((nIterations < nMaxIterations) && (nTol > tolerance) && (sigmaSquared > 1e-8)) {

    if (isVerbose) {
      std::cout << "it: " << nIterations << "/" << nMaxIterations << std::endl;
    }

    // annealing of tau, lambda
    float tau = tau0 * (float)std::pow(annealingFactor, nIterations);
    float lambda = lambda0 * (float)std::pow(annealingFactor, nIterations);
    float Q_tilde_old = Q_tilde;
    // P-Step
    auto P = getModifiedProbabilities(X_t, Ymat, sigmaSquared, B);
    Eigen::MatrixXf PY = P * Ymat;
    Eigen::VectorXf P1 = P.rowwise().sum();       // (N x 1)
    Eigen::VectorXf Ptransp1 = P.colwise().sum(); // (M x 1)
    float Np = P1.sum();

    Q_tilde = -1 / (2 * sigmaSquared) * (Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace() +
              1 / sigmaSquared * (X_t.transpose() * P * Ymat).trace() -
              1 / (2 * sigmaSquared) * (X_t.transpose() * P1.asDiagonal() * X_t).trace() -
              Np * D / 2 * log(sigmaSquared) - lambda / 2 * (W.transpose() * G * W).trace() -
              tau / 2 * (X_t.transpose() * Phi * X_t).trace();
    nTol = abs((Q_tilde - Q_tilde_old) / Q_tilde);

    // M-Step
    // solve LHS*W = RHS
    Eigen::MatrixXf leftHandSide =
        (P1.asDiagonal() * G + lambda * sigmaSquared * Eigen::MatrixXf::Identity(N, N) +
         tau * sigmaSquared * Phi * G);
    Eigen::MatrixXf rightHandSide = PY - P1.asDiagonal() * X - tau * sigmaSquared * Phi * X;
    W = leftHandSide.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rightHandSide);

    X_t = X + G * W;

    // implementation hze
    // sigmaSquared = abs((( Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace()
    //                     - 2 * (X_t.transpose() * PY).trace() +
    //                     (X_t.transpose() * P1.asDiagonal() * X_t).trace()) /
    //                    (Np * D));

    // implementation xwk
    // sigmaSquared = abs((( Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace()
    //                       - 2 * (W.transpose() * G * PY).trace()
    //                       + (W.transpose() * G * P1.asDiagonal() * G * W).trace()) /
    //                       (Np * D));

    // implementation spr matlab

    sigmaSquared = abs((((Ymat.cwiseAbs2()).cwiseProduct(Ptransp1.replicate(1, D))).sum() +
                        ((X_t.cwiseAbs2()).cwiseProduct(P1.replicate(1, D))).sum() -
                        2 * (PY.transpose() * X_t).trace()) /
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

  if (normalization) {
    // denormalization
    X_t = X_t * Y_scaleFactor;
    X_t += Y_mean.replicate(N, 1);
  }

  return X_t;
}

Eigen::MatrixX3f SPRBranched::computeEM(Eigen::MatrixX3f &X, Eigen::MatrixX3f &Ymat,
                                        std::vector<std::vector<Eigen::Index>> B) {

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
    std::cout << "This is the stored branch array:" << std::endl;
    std::vector<Eigen::Index> branchIndices;
    for (std::size_t i = 0; i < B.size(); ++i) {
      branchIndices = B[i];
      std::cout << i + 1 << ". Branch" << std::endl;
      for (std::size_t j = 0; j < branchIndices.size(); ++j) {
        std::cout << branchIndices[j] << " ";
      }
      std::cout << "/" << std::endl;
    }
  }

  const auto N = X.rows();
  const auto M = Ymat.rows();
  const Eigen::Index D = X.cols();

  Eigen::RowVector3f X_mean;
  Eigen::RowVector3f Y_mean;
  double X_scaleFactor;
  double Y_scaleFactor;
  if (normalization) {
    // normalize datasets
    std::pair<Eigen::RowVector3f, double> X_norm = calculateMeanAndScaleFactor(X);
    X_mean = X_norm.first;
    X_scaleFactor = X_norm.second;

    std::pair<Eigen::RowVector3f, double> Y_norm = calculateMeanAndScaleFactor(Ymat);
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
  // auto Phi = load_csv_to_eigen<Eigen::MatrixXf>("data/Phi_ml.csv"); // use this for external Phi
  // matrix instead of calculating via MLLE

  // std::cout << "MLLE:\n" << Phi << std::endl;
  float Q_tilde = 1;

  if (isVerbose) {
    std::cout << Ymat.colwise().sum() << std::endl;
    // std::cout <<"X is : "<< X << std::endl;
  }
  while ((nIterations < nMaxIterations) && (nTol > tolerance) && (sigmaSquared > 1e-8)) {

    if (isVerbose) {
      std::cout << "it: " << nIterations << "/" << nMaxIterations << std::endl;
    }

    // annealing of tau, lambda
    float tau = tau0 * (float)std::pow(annealingFactor, nIterations);
    float lambda = lambda0 * (float)std::pow(annealingFactor, nIterations);
    float Q_tilde_old = Q_tilde;
    // P-Step
    auto P = getModifiedProbabilities(X_t, Ymat, sigmaSquared, B);
    Eigen::MatrixXf PY = P * Ymat;
    Eigen::VectorXf P1 = P.rowwise().sum();       // (N x 1)
    Eigen::VectorXf Ptransp1 = P.colwise().sum(); // (M x 1)
    float Np = P1.sum();

    Q_tilde = -1 / (2 * sigmaSquared) * (Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace() +
              1 / sigmaSquared * (X_t.transpose() * P * Ymat).trace() -
              1 / (2 * sigmaSquared) * (X_t.transpose() * P1.asDiagonal() * X_t).trace() -
              Np * D / 2 * log(sigmaSquared) - lambda / 2 * (W.transpose() * G * W).trace() -
              tau / 2 * (X_t.transpose() * Phi * X_t).trace();
    nTol = abs((Q_tilde - Q_tilde_old) / Q_tilde);

    // M-Step
    // solve LHS*W = RHS
    Eigen::MatrixXf leftHandSide =
        (P1.asDiagonal() * G + lambda * sigmaSquared * Eigen::MatrixXf::Identity(N, N) +
         tau * sigmaSquared * Phi * G);
    Eigen::MatrixXf rightHandSide = PY - P1.asDiagonal() * X - tau * sigmaSquared * Phi * X;
    W = leftHandSide.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rightHandSide);

    X_t = X + G * W;

    // implementation hze
    // sigmaSquared = abs((( Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace()
    //                     - 2 * (X_t.transpose() * PY).trace() +
    //                     (X_t.transpose() * P1.asDiagonal() * X_t).trace()) /
    //                    (Np * D));

    // implementation xwk
    // sigmaSquared = abs((( Ymat.transpose() * Ptransp1.asDiagonal() * Ymat).trace()
    //                       - 2 * (W.transpose() * G * PY).trace()
    //                       + (W.transpose() * G * P1.asDiagonal() * G * W).trace()) /
    //                       (Np * D));

    // implementation spr matlab (short)

    sigmaSquared = abs((((Ymat.cwiseAbs2()).cwiseProduct(Ptransp1.replicate(1, D))).sum() +
                        ((X_t.cwiseAbs2()).cwiseProduct(P1.replicate(1, D))).sum() -
                        2 * (PY.transpose() * X_t).trace()) /
                       (Np * D));

    // implementation spr matlab (long)
    /*
    sigmaSquared = abs(
                        (
                          (
                            ( Ymat.cwiseAbs2() ).cwiseProduct(Ptransp1.replicate(1,D))
                          ).sum() +
                          (
                            (X_t.cwiseAbs2()).cwiseProduct(P1.replicate(1,D))
                          ).sum() -
                          2*(PY.transpose() * X_t).trace()
                        ) / (Np * D)
                      );
    */
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

  if (normalization) {
    // denormalization
    X_t = X_t * Y_scaleFactor;
    X_t += Y_mean.replicate(N, 1);
  }

  return X_t;
}

Eigen::MatrixXf SPRBranched::getModifiedProbabilities(const Eigen::MatrixX3f &X,
                                                      const Eigen::MatrixX3f &Y, float sigmaSquared,
                                                      std::vector<std::vector<Eigen::Index>> B) {
  const auto N = X.rows();
  const auto M = Y.rows();
  const Eigen::Index D = 3;
  const auto K = B.size();
  float mu = wOutliers;

  // Gaussian kernel matrix G \in |R^(N x M)
  auto rbfEval =
      computeGaussianKernels(X, Y, sqrt(sigmaSquared)); // this is the nominator of matrix P
  Eigen::RowVectorXf P_normalization =
      rbfEval.colwise().sum().array() +
      (float)std::pow(2 * M_PI * sigmaSquared, D / 2) * mu * N / ((1 - mu) * M);
  auto P = rbfEval.cwiseQuotient(P_normalization.replicate(N, 1));

  // Probability modification based on branch likeliness
  Eigen::MatrixXf P_mod = Eigen::MatrixXf::Zero(N, M);
  Eigen::MatrixXf P_k_nom =
      Eigen::MatrixXf::Zero(K, M); // matrix to store nominator of p(k) = sum over the centroids
                                   // belonging to branches 1,2,...,k
  Eigen::RowVectorXf P_k_den =
      Eigen::RowVectorXf::Zero(M); // denominator of p(k) = summed probability over all branches
  Eigen::MatrixXf P_k =
      Eigen::MatrixXf::Zero(K, M); // matrix to store branch probability p(k) for each branch

  // asselmble P_mod directly from P
  // calculate branchwise probabilities
  for (std::size_t k = 0; k < K; ++k) {
    auto branch_indices = B[k];
    for (std::size_t l = 0; l < branch_indices.size(); ++l) {
      P_k_nom.row((Eigen::Index)k) +=
          P.row(branch_indices[l]); // this is the nominator of matrix P_mod
    }
    // TODO: with Eigen 3.4 use vector indexing here
  }
  // Assemble P_mod
  for (std::size_t k = 0; k < K; ++k) {
    auto branch_indices = B[k];
    for (std::size_t l = 0; l < branch_indices.size(); ++l) {
      P_mod.row(branch_indices[l]) = P_k_nom.row((Eigen::Index)k);
    }
  }

  return P_mod.cwiseProduct(P); // = modified probability matrix
}
