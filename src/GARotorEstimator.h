#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <vector>

Eigen::Quaterniond GAFastRotorEstimator(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Quaterniond GAFastRotorEstimatorAVX(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Quaterniond GAFastRotorEstimatorIncr(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w, const Eigen::Quaterniond& Qprev);
Eigen::Quaterniond GANewtonRotorEstimator(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Quaterniond GAFastRotorEstimatorAprox(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w, double epsilon=1e-6, size_t steps=2);
