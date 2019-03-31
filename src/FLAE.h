#pragma once

#include <Eigen\core>
#include <Eigen\Dense>
#include <Eigen\Geometry> // For Quaternion
#include <vector>

Eigen::Quaterniond FlaeSymbolic(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Quaterniond Flae(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
Eigen::Quaterniond FlaeNewton(const std::vector<Eigen::Vector3d>& P, const std::vector<Eigen::Vector3d>& Q, const std::vector<double>& w);
