#include "Horn.h"

#include <Eigen\Eigenvalues>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d; 
using Eigen::Quaterniond;
using std::vector;

Quaterniond Horn(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d M;
	Matrix4d N;
	const size_t n = P.size();
	M.setZero();
	for (size_t j = 0; j < n; ++j)
	{
		M += w[j] * P[j] * Q[j].transpose();
	}

	// on-diagonal elements
	N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
	N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
	N(2, 2) = -M(0, 0) + M(1, 1) - M(2, 2);
	N(3, 3) = -M(0, 0) - M(1, 1) + M(2, 2);

	// off-diagonal elements
	N(0, 1) = N(1, 0) = M(1, 2) - M(2, 1);
	N(0, 2) = N(2, 0) = M(2, 0) - M(0, 2);
	N(0, 3) = N(3, 0) = M(0, 1) - M(1, 0);
	N(1, 2) = N(2, 1) = M(0, 1) + M(1, 0);
	N(1, 3) = N(3, 1) = M(2, 0) + M(0, 2);
	N(2, 3) = N(3, 2) = M(1, 2) + M(2, 1);

	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen(N);
	Vector4d V = eigen.eigenvectors().col(3);
	Quaterniond quaternion(V(0), V(1), V(2), V(3));
	return quaternion;
}
