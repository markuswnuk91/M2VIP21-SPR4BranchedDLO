#pragma once
#include <fstream>
#include <string>
#include <eigen3/Eigen/Eigen>

template <typename M> M load_csv_to_eigen(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<float> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stof(cell));
    }
    ++rows;
  }
  return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                                        M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

template <typename M> void write_eigen_to_csv(const std::string &path, const M &matrixToWrite) {
  std::ofstream outdata;
  outdata.open(path);
  for (Eigen::Index r = 0; r < matrixToWrite.rows(); ++r) {
    for (Eigen::Index c = 0; c < matrixToWrite.cols()-1; ++c) {
      outdata << matrixToWrite(r, c) << ", ";
    }
    outdata << matrixToWrite(r, matrixToWrite.cols()-1);
    outdata << "\n";
  }
  outdata.close();
}