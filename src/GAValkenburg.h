#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion
#include <vector>

Eigen::Quaterniond GAValkenburg(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
