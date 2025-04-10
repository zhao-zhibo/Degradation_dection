#include <boost/math/distributions/normal.hpp>
#include <cmath>
#include "Eigen/Eigenvalues"

namespace degeneracy {

template <typename T>
using VectorVector3 = std::vector<Eigen::Matrix<T, 3, 1>>;

template <typename T>
using VectorMatrix3 = std::vector<Eigen::Matrix<T, 3, 3>>;

// 将三维向量转换为反对称矩阵
template <typename T>
inline Eigen::Matrix<T, 3, 3> VectorToSkew(const Eigen::Matrix<T, 3, 1>& vector) {
  Eigen::Matrix<T, 3, 3> skew;
  skew << 0, -vector.z(), vector.y(), vector.z(), 0, -vector.x(), -vector.y(), vector.x(), 0;
  return skew;
}

//// @brief 对应论文中的公式15和16，最终返回的是均值和方差。这两个统计量用于后续计算每个方向的退化概率
//// @return mean：表示​​点噪声​​（如 LiDAR 测距误差）和法线噪声​​（如表面拟合误差）对 Hessian 的平均影响
//// @return variance​​：表示噪声在特定方向 u 上的波动强度。
//// @param normal_covariances：法线的协方差矩阵
//// @param U：Hessian矩阵的特征向量矩阵，用于投影噪声到各方向。 
//// @param stdevPoints：点噪声标准差
template <typename T, typename Q>
auto ComputeNoiseEstimate(const VectorVector3<Q>& points, const VectorVector3<Q>& normals, const std::vector<Q>& weights, const VectorMatrix3<T>& normal_covariances, const Eigen::Matrix<T, 6, 6>& U, const T& stdevPoints) {
  using Vector3 = Eigen::Matrix<T, 3, 1>;
  using Matrix3 = Eigen::Matrix<T, 3, 3>;
  using Vector6 = Eigen::Matrix<T, 6, 1>;
  using Matrix6 = Eigen::Matrix<T, 6, 6>;

  Matrix6 mean = Matrix6::Zero();
  Vector6 variance = Vector6::Zero();
  // points是所有的点云
  const size_t nPoints = points.size();

  for (size_t i = 0; i < nPoints; i++) {
    // 取出点云中的一个点，对应这个点的法线和对应这个点的权重
    const Vector3 point = points[i].template cast<T>();
    const Vector3 normal = normals[i].template cast<T>();
    const Matrix3 nx = VectorToSkew<T>(normal);
    const Matrix3 px = VectorToSkew<T>(point);
    const T w = weights[i];

    // Coefficient matrix for epsilon and eta 
    // 对应论文中的公式13中的B矩阵
    Matrix6 B = Matrix6::Zero();
    B.block(0, 0, 3, 3) = -nx;
    B.block(0, 3, 3, 3) = px * nx;
    B.block(3, 3, 3, 3) = nx;

    // Covariance matrix for epsilon and eta
    // 对应论文中的公式14中的N矩阵
    Matrix6 N = Matrix6::Zero();
    N.block(0, 0, 3, 3) = Matrix3::Identity() * std::pow(stdevPoints, 2);  // 点噪声Σ_p = σ_p²I
    N.block(3, 3, 3, 3) = normal_covariances[i]; // 法线噪声Σ_n（来自外部输入）
    // 对应论文公式14中的噪声协方差矩阵Σ  公式15
    Matrix6 contribution_to_mean = (B * N * B.transpose()) * w;

    mean.noalias() += contribution_to_mean.eval();

    // v hat weighted by w 对应公式12中的vi
    Vector6 v = Vector6::Zero();
    v.head(3) = std::sqrt(w) * px * normal;
    v.tail(3) = std::sqrt(w) * normal;

    // Compute variance in the directions given by U 计算各个方向上的方差贡献，对应公式17
    for (size_t k = 0; k < 6; k++) {
      const Vector6 u = U.col(k);
      const T a = (u.transpose() * contribution_to_mean * u).value();
      const T b = (u.transpose() * v).value();
      const T contribution_to_variance = 2 * std::pow(a, 2) + 4 * a * std::pow(b, 2);
      variance[k] += contribution_to_variance;
    }
  }

  return std::make_tuple(mean, variance);
}

// ​​对应论文章节​​: III.A (信号与噪声比概率计算)，计算每个方向uk的退化概率p
// measurement：信号值，表示Hessian矩阵在方向uk上的特征值λ

//// @brief 计算每个特征向量方向uk的退化概率 puk,即在该方向上，信号（Hessian 矩阵的特征值）是否足够强于噪声的概率。
//// @param measured_information_matrix：Hessian矩阵
//// @param estimated_noise_mean：噪声均值,即为​点噪声​​（如 LiDAR 测距误差）和法线噪声​​（如表面拟合误差）对 Hessian 的平均影响
//// @param estimated_noise_variances：噪声方差，表示噪声在特定方向 u 上的波动强度
//// @param U：Hessian矩阵的特征向量矩阵，6x6的矩阵，每一列表示特征向量
//// @param snr_factor：信噪比因子
template <typename T>
Eigen::Matrix<T, 6, 1> ComputeSignalToNoiseProbabilities(const Eigen::Matrix<T, 6, 6>& measured_information_matrix,
                                                         const Eigen::Matrix<T, 6, 6>& estimated_noise_mean,
                                                         const Eigen::Matrix<T, 6, 1>& estimated_noise_variances,
                                                         const Eigen::Matrix<T, 6, 6>& U,
                                                         const T& snr_factor) {
  typedef Eigen::Matrix<T, 6, 1> Vector6;

  Vector6 probabilities = Vector6::Zero();

  for (size_t k = 0; k < 6; k++) {
    // ​​步骤 1：计算信号与噪声的统计量​​, 测量值大于10倍的噪声值，表示信号足够强于噪声
    const Vector6 u = U.col(k);
    const T measurement = (u.transpose() * measured_information_matrix * u).value(); // 计算 u^T H u，得到的特征向量u所对应的特征值λ。因为特征向量u满足Hu = λu，也就是信号值，测量值
    const T expected_noise = (u.transpose() * estimated_noise_mean * u).value(); // 计算 u^T Σ u,对应公式18,estimated_noise_mean对应公式中的Σ，最终得到方向 u 的噪声期望值
    const T stdev = std::sqrt(estimated_noise_variances[k]); // 噪声标准差​​：表示Hessian矩阵在方向 uk上的噪声强度，这个量从ComputeNoiseEstimate返回
    const T test_point = measurement / (T(1.0) + snr_factor);  // 退化阈值点​​：判断信号是否足够强于噪声的临界值，噪声大于这个值就表明发生了退化

    const bool any_nan = std::isnan(expected_noise) || std::isnan(stdev) || std::isnan(test_point);

    if (!any_nan) {
      // normal_distribution<T>(expected_noise, stdev): 定义噪声的正态分布
      // test_point: 计算该处的CDF值
      const T probability = boost::math::cdf(
          boost::math::normal_distribution<T>(expected_noise, stdev), test_point);

      probabilities[k] = probability;
    } else {
      std::cout << "NaN value in probability calculation - stDev: " << stdev << " test point: " << test_point << " expected noise: " << expected_noise << std::endl;
      probabilities[k] = 0.0;
    }
  }

  return probabilities;
}

template <typename T>
Eigen::Matrix<T, 6, 1> SolveWithSnrProbabilities(
    const Eigen::Matrix<T, 6, 6>& U,
    const Eigen::Matrix<T, 6, 1>& eigenvalues,
    const Eigen::Matrix<T, 6, 1>& rhs,
    const Eigen::Matrix<T, 6, 1>& snr_probabilities) {
  typedef typename Eigen::Matrix<T, 6, 1> Vector6;

  Vector6 d_psinv = Vector6::Zero();

  for (size_t i = 0; i < 6; i++) {
    const T eigenvalue = eigenvalues[i];
    const T p = snr_probabilities[i];
    d_psinv[i] = p / eigenvalue;
  }

  Vector6 perturbation = U * d_psinv.asDiagonal() * U.transpose() * rhs;

  return perturbation;
}

// 估计点云法线及其协方差矩阵，用于噪声模型中的法线不确定性计算。
// 功能​​: 输入一个包含 N 个三维点的矩阵points(一个平面区域的点)，返回这个平面的法线方向、方差、到原点的距离及法线的协方差矩阵。
template <size_t N, typename T>
auto EstimateNormal(const Eigen::Matrix<T, 3, N>& points, const T& stDevPoint, const bool& robust) {
  using Vector3 = Eigen::Matrix<T, 3, 1>;
  using Matrix3 = Eigen::Matrix<T, 3, 3>;

  Vector3 mean = Vector3::Zero();
  Matrix3 covariance = Matrix3::Zero();

  for (size_t i = 0; i < N; i++) {
    mean += points.col(i); // 累加所有点
    covariance += points.col(i) * points.col(i).transpose(); // 累加外积
  }
  mean /= N; // 计算均值
  covariance /= N; // 计算未中心化的协方差
  covariance -= mean * mean.transpose(); // 中心化协方差 论文公式30

  Eigen::SelfAdjointEigenSolver<Matrix3> solver(covariance);
  Vector3 eigenvalues = solver.eigenvalues(); // 协方差矩阵的特征值
  Matrix3 eigenvectors = solver.eigenvectors(); // 协方差矩阵的特征向量

  const Vector3 normal = eigenvectors.col(0); // 最小特征值对应的特征向量为法线方向，论文公式31

  T mid_eigenvalue = eigenvalues(1);
  T max_eigenvalue = eigenvalues(2);
  // 鲁棒性调整，抑制噪声，对应论文公式34(不确定是不是这个公式),​​目的​​: 消除点噪声对协方差矩阵特征值的影响，确保法线估计的鲁棒性。
  if (robust) {
    mid_eigenvalue = std::max(mid_eigenvalue - stDevPoint * stDevPoint, 1e-7);
    max_eigenvalue = std::max(max_eigenvalue - stDevPoint * stDevPoint, 1e-7);
  }
  // 计算法线方差，法线方向的不确定性与点噪声和协方差矩阵中的特征值成反比，对应于论文公式34
  const T variance = stDevPoint * stDevPoint * (1 / T(N)) * (1 / mid_eigenvalue);
  // ​​计算到原点的距离,平面方程n.trans *  q = d.​​
  const T distance_to_origin = normal.transpose() * mean;
  // 计算法线的协方差矩阵，对应论文中的公式34，但是公式和论文中特征值的排序不同
  const Matrix3 covariance_of_normal = stDevPoint * stDevPoint * (1 / T(N)) * eigenvectors * Vector3(0, 1 / mid_eigenvalue, 1 / max_eigenvalue).asDiagonal() * eigenvectors.transpose();
  // 
  return std::make_tuple(normal, variance, distance_to_origin, covariance_of_normal);
}

}  // namespace degeneracy