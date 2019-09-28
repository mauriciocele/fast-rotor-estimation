#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

Eigen::Matrix3d SVDEigen(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);

Eigen::Matrix3d SVDMcAdams(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
