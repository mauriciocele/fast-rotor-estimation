#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <Eigen/LU>
#include <Eigen/SVD>
#include <vector>

Eigen::Matrix3d FA3R_int(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Matrix3d FA3R_double(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
