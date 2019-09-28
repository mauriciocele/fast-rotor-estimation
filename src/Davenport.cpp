#include "Davenport.h"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Quaterniond;
using std::vector;

Quaterniond Davenport(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d K;
	Matrix3d B;
	const size_t N = P.size();
    B.setZero();
	for (size_t j = 0; j < N; ++j) {
		B.noalias() += (w[j] * P[j]) * Q[j].transpose();
	}
    const double trace = B.trace();
	K(0, 0) = trace;
	K(1, 0) = B(1, 2) - B(2, 1);
	K(2, 0) = B(2, 0) - B(0, 2);
	K(3, 0) = B(0, 1) - B(1, 0);
	K(1, 1) = 2.0 * B(0, 0) - trace;
	K(2, 1) = B(0, 1) + B(1, 0); 
	K(3, 1) = B(2, 0) + B(0, 2);
	K(2, 2) = 2.0 * B(1, 1) - trace; 
	K(3, 2) = B(1, 2) + B(2, 1);
	K(3, 3) = 2.0 * B(2, 2) - trace;

	K.selfadjointView<Eigen::Lower>().evalTo(K);
	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen(K);
	const Vector4d V = eigen.eigenvectors().col(3);
	return Quaterniond(V(0), V(1), V(2), V(3));
}
