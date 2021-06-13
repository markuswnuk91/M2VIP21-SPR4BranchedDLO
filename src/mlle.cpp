#include "mlle.h"

std::vector<Eigen::Index> findIndicesOfNearest(const Eigen::MatrixX3f &X, Eigen::Index i) {
  Eigen::MatrixX3f X_Xi_diff = X - X.row(i).replicate(X.rows(), 1);
  Eigen::VectorXf d = X_Xi_diff.rowwise().squaredNorm(); // distances (unsorted)

  // folowing: https://stackoverflow.com/a/12399290 to sort indices
  std::vector<Eigen::Index> idx(X.rows());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&d](size_t i1, size_t i2) { return d(i1) < d(i2); });

  return idx;
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecompositionSquared(const Eigen::Matrix3Xf &G) {
  auto D = G.rows();
  auto k_i = G.cols();
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(G, Eigen::ComputeFullV);
  Eigen::VectorXf S = Eigen::VectorXf::Zero(k_i);
  S.head(D) = svd.singularValues().array().pow(2.0);
  return std::make_pair<Eigen::VectorXf, Eigen::MatrixXf>(std::move(S),
                                                          Eigen::MatrixXf(svd.matrixV()));
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecompositionES(const Eigen::MatrixXf &G) {
//  auto D = G.rows();
 // auto k_i = G.cols();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es(G);
  Eigen::VectorXf eVals = es.eigenvalues();
  Eigen::MatrixXf eVecs = es.eigenvectors();

//sort the eigenvalues in descending order = diagS[JIi]
// following: https://stackoverflow.com/a/12399290 to sort indices
std::vector<Eigen::Index> JIi(eVals.size());
std::iota(JIi.begin(), JIi.end(), 0);
stable_sort(JIi.begin(), JIi.end(), [&eVals](size_t i1, size_t i2) { return eVals(i1) > eVals(i2); });
Eigen::VectorXf Ev = Eigen::VectorXf(eVals.size());
Eigen::MatrixXf Theta = Eigen::MatrixXf(eVecs.rows(),eVals.size());
//sort the eigenvalues in descending order
for (Eigen::Index j = 0; j < eVals.size(); ++j)
{
  Ev[j] = eVals(JIi[j]); 
  Theta.col(j) = eVecs.col(JIi[j]);
} 
  return std::make_pair(Ev, Theta);
}

std::pair<Eigen::VectorXf, Eigen::MatrixXf> eigenDecomposition(const Eigen::MatrixXf &G) {
//  auto D = G.rows();
 // auto k_i = G.cols();

  Eigen::RealSchur<Eigen::MatrixXf> schur(G);
  Eigen::MatrixXf V = schur.matrixU();
//  Eigen::MatrixXf S = schur.matrixT();
  Eigen::VectorXf diagS = schur.matrixT().diagonal();

//sort the eigenvalues in descending order = diagS[JIi]
// following: https://stackoverflow.com/a/12399290 to sort indices
std::vector<Eigen::Index> JIi(diagS.size());
std::iota(JIi.begin(), JIi.end(), 0);
stable_sort(JIi.begin(), JIi.end(), [&diagS](size_t i1, size_t i2) { return diagS(i1) > diagS(i2); });
Eigen::VectorXf Ev = Eigen::VectorXf(diagS.size());
Eigen::MatrixXf Theta = Eigen::MatrixXf(V.rows(),diagS.size());
//sort the eigenvalues in descending order
for (Eigen::Index j = 0; j < diagS.size(); ++j)
{
  Ev[j] = diagS(JIi[j]); 
  Theta.col(j) = V.col(JIi[j]);
} 
  return std::make_pair(Ev, Theta);
}

Eigen::MatrixXf MLLE(const Eigen::MatrixX3f &X, std::size_t k, std::size_t d) {
  // X: NxD Matrix with N Datapoints and D dimension of a data point
  /// tolearance parameter
  float tol = 1e-3;
  auto N = X.rows();

  Eigen::VectorXf ratio = Eigen::VectorXf(N); // vector of ratios of eigenvalues

  // std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> S(
  //     N); // N vectors containing descendingly sorted eigenvalues of k nearest neighbours of X[:,i]
  //         // in element i.
  std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> Ev(
      N); // N vectors containing descendingly sorted eigenvalues of k nearest neighbours of X[:,i]
          // in element i.

  // std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> V(
  //     N); // N  (k x k)-matrices containing eigenvectors for each S[i].
  std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> Theta(
      N); // N  (k x k)-matrices containing eigenvectors for each Ev[i].


  std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> W(
      N); // N normalized solutions for w_i(\gamma)
  std::vector<std::vector<Eigen::Index>> nearestIndices(
      N); // nearest indices (k_i nearest neighbours for x_i = X(i,:) ).

  Eigen::MatrixXf Phi = Eigen::MatrixXf::Zero(N, N); // matrix to return

  Eigen::Index k_i = std::min(N-1, (Eigen::Index)k);

  for (Eigen::Index i = 0; i < N; ++i) { // [Zhang2007] Algorithm Step 1

    auto nearestIdx = findIndicesOfNearest(X, i); // [Zhang2007] Algorithm Step 1.1
    

    Eigen::MatrixXf G_i = Eigen::MatrixXf(X.cols(), k_i);
    for (Eigen::Index j = 0; j < G_i.cols(); ++j) {
      G_i.col(j) = X.row(nearestIdx[j+1]) - X.row(i); // // exclude i itself (first index of nearestIdx)
    }

    //auto S_V = eigenDecompositionSquared(G_i); // [Zhang2007] Algorithm Step 1.3

    //auto S_V = eigenDecomposition(C);
    //ratio(i) = S_V.first.tail(k_i - d).sum() / S_V.first.head(d).sum();
    //ratio(i) = S_V.first.tail(k_i - d).sum() / S_V.first.head(d).sum();
    //S[i] = S_V.first;
    //V[i] = S_V.second;
    //nearestIndices[i] = nearestIdx;

    // [Zhang2007] Algorithm Step 1.3 with Schur decomposition
    Eigen::MatrixXf C = G_i.transpose() * G_i;
    auto Ev_Theta = eigenDecompositionES(C);
    float rho_num = Ev_Theta.first.segment(d,k_i - d).sum();
    float rho_denom = Ev_Theta.first.head(d).sum();
    ratio(i) = rho_num / rho_denom;

    Ev[i] = Ev_Theta.first;
    Theta[i] = Ev_Theta.second;
    nearestIndices[i] = nearestIdx;

    // [Zhang2007] Algorithm Step 1.2
    C += Eigen::MatrixXf::Identity(k_i, k_i) * tol * C.trace(); // regularization
    //Eigen::VectorXf wi = C.llt().solve(Eigen::VectorXf::Ones(k_i)).normalized();
    //Eigen::VectorXf wi = C.ldlt().solve(Eigen::VectorXf::Ones(k_i)).normalized();
    Eigen::VectorXf wi = C.householderQr().solve(Eigen::VectorXf::Ones(k_i));
    // if (Eigen::VectorXf::Ones(k_i).isApprox(C*wi) == false){
    //   std::cout << "WARNING: MLLE did not find optimal weights for" << i << std::endl;
    // }
    wi /= wi.sum();
    W[i] = wi;
  }

  // [Zhang2007] Algorithm Step 2
  auto ratioSorted = ratio;
  //sort in ascending order
  std::sort(ratioSorted.data(), ratioSorted.data() + ratioSorted.size());
  //std::sort(ratioSorted.data(), ratioSorted.data() + ratioSorted.size(),std::greater<float>()); --> this sorts in desending order!
  float eta = ratioSorted(ceil(N / 2));

  // [Zhang2007] Algorithm Step 3
  for (Eigen::Index i = 0; i < N; ++i) {

    // [Zhang2007] Algorithm Step 3.1
    auto nearestIdx = nearestIndices[i];

    Eigen::Index ell = k_i - Eigen::Index(d);

    while ((Ev[i].tail(k_i - ell).sum() / Ev[i].head(k_i - ell).sum()) > eta && ell > 1) {
      ell -= 1;
    }
    Eigen::Index s_i = ell;

    Eigen::MatrixXf Vi = Theta[i].rightCols(s_i);
    Eigen::VectorXf ve = Vi.colwise().sum();
    float alpha = ve.norm() / sqrt(s_i);

    // [Zhang2007] Algorithm Step 3.2
    Eigen::VectorXf u = ve.array() - alpha;
    Eigen::MatrixXf Wi;

    if (u.norm() > 1e-5) {
      u.normalize();
      Wi = (1 - alpha) * (1 - alpha) * W[i] * Eigen::MatrixXf::Ones(1, s_i) +
           (2 - alpha) * (Vi - (Vi * (2 * u)) * u.transpose());
    } else {
      Wi = (1 - alpha) * (1 - alpha) * W[i] * Eigen::MatrixXf::Ones(1, s_i) + (2 - alpha) * Vi;
    }
    Eigen::MatrixXf WiExt = Eigen::MatrixXf(Wi.rows() + 1, Wi.cols());
    WiExt << -Eigen::RowVectorXf::Ones(s_i), Wi;

    // TODO: with Eigen 3.4 use index slicing here
    // i.e. `Phi(nearestIdx, nearestIdx) = WiExt.transpose()*WiExt;`
    Eigen::MatrixXf Phi_i = WiExt * WiExt.transpose();
    for (Eigen::Index j = 0; j < k_i + 1; ++j) {
      for (Eigen::Index l = 0; l < k_i + 1; ++l) {
        Phi(nearestIdx[j], nearestIdx[l]) += Phi_i(j, l);
      }
    }
  }

  return Phi;
}