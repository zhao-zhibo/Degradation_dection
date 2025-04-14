#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>
#include "data.h"
#include "degeneracy.h"

// 论文为：Probabilistic Degeneracy Detection for  Point-to-Plane Error Minimization

// 计算Hessian矩阵，对应论文中的公式5
Eigen::Matrix<double, 6, 6> ComputeHessian(const degeneracy::VectorVector3<double>& points, const degeneracy::VectorVector3<double>& normals, const std::vector<double>& weights) {
  const size_t nPoints = points.size();
  Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(6, 6);
  for (size_t i = 0; i < nPoints; i++) {
    // 提取点、法线和权重
    const Eigen::Vector3d point = points[i];
    const Eigen::Vector3d normal = normals[i];
    const Eigen::Vector3d pxn = point.cross(normal); // 叉乘之后是3乘1的向量
    const double w = std::sqrt(weights[i]);
    // 构造向量 v = w * [ pxn; n ]
    Eigen::Matrix<double, 6, 1> v; // v是6乘1的向量
    v.head(3) = w * pxn;
    v.tail(3) = w * normal;
    H += v * v.transpose(); // 乘完之后和公式5是对应关系
  }
  return H;
}

// 计算法线的协方差矩阵，但是代码中也提供了另一个函数EstimateNormal，可以计算法线的协方差矩阵，这个函数更全面
// 下面这个计算是简化版本的计算法线协方差矩阵的方法
degeneracy::VectorMatrix3<double> GetIsotropicCovariances(const size_t& N, const double stdev) {
  degeneracy::VectorMatrix3<double> covariances;
  covariances.reserve(N);
  for (size_t i = 0; i < N; i++) {
    covariances.push_back(Eigen::Matrix3d::Identity() * std::pow(stdev, 2));
  }
  return covariances;
}

int main() {
  // Points, normals and covariances must be expressed in the same frame of reference
  // For the conditioning of the Hessian, it is preferable to use the LiDAR frame (and not the world frame)
  // 提前提供了120个点云点，120点法线，120点对应的权重
  const auto points = data::points;
  const auto normals = data::normals;
  const auto weights_squared = data::weights_squared;
  const auto normal_covariances = GetIsotropicCovariances(data::normals.size(), data::stdev_normals);

  const auto H = ComputeHessian(points, normals, weights_squared);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(H);

  const auto eigenvectors = eigensolver.eigenvectors();
  const auto eigenvalues = eigensolver.eigenvalues();

  Eigen::Matrix<double, 6, 6> noise_mean;
  Eigen::Matrix<double, 6, 1> noise_variance;
  const double snr_factor = 10.0;

  std::tie(noise_mean, noise_variance) = degeneracy::ComputeNoiseEstimate<double, double>(points, normals, weights_squared, normal_covariances, eigenvectors, data::stdev_points);
  Eigen::Matrix<double, 6, 1> non_degeneracy_probabilities = degeneracy::ComputeSignalToNoiseProbabilities<double>(H, noise_mean, noise_variance, eigenvectors, snr_factor);

  std::cout << "The non-degeneracy probabilities are: " << std::endl;
  std::cout << non_degeneracy_probabilities.transpose() << std::endl;

  std::cout << "For the eigenvectors of the Hessian: " << std::endl;
  std::cout << eigenvectors << std::endl;

  // The following exemplifies how to solve the system of equations using the probabilities
  // Dummy right hand side rhs = Jtb  给了一个临时的-Jb，然后后面代入函数中计算出最终的优化量
  const Eigen::Matrix<double, 6, 1> rhs = Eigen::Matrix<double, 6, 1>::Zero(6, 1);
  const auto estimate = degeneracy::SolveWithSnrProbabilities(eigenvectors, eigenvalues, rhs, non_degeneracy_probabilities);

  return 0;
}
