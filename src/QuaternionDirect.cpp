#include "QuaternionDirect.h"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Quaterniond;
using std::vector;

/*
    For product L*X
*/
inline Matrix4d QuaternionLeft(const Vector3d& q)
{
    Matrix4d L;
    L(0,0) = L(1,1) = L(2,2) = L(3,3) = 0;
    L(0,1) = -q.x(); L(1,0) = q.x(); 
    L(0,2) = -q.y(); L(2,0) = q.y();
    L(0,3) = -q.z(); L(3,0) = q.z();
    L(1,2) =  q.z(); L(2,1) = -q.z();
    L(1,3) = -q.y(); L(3,1) = q.y();
    L(2,3) =  q.x(); L(3,2) = -q.x();
    return L;
}

/*
    For product X*R
*/
inline Matrix4d QuaternionRight(const Vector3d& q)
{
    Matrix4d R;
    R(0,0) = R(1,1) = R(2,2) = R(3,3) = 0;
    R(0,1) = -q.x(); R(1,0) = q.x(); 
    R(0,2) = -q.y(); R(2,0) = q.y();
    R(0,3) = -q.z(); R(3,0) = q.z();
    R(1,2) = -q.z(); R(2,1) = q.z();
    R(1,3) =  q.y(); R(3,1) = -q.y();
    R(2,3) = -q.x(); R(3,2) = q.x();
    return R;
}

Quaterniond QuaternionDirect(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d M;
	const size_t N = P.size();
    M.setZero();
	for (size_t j = 0; j < N; ++j) {
        Matrix4d A = QuaternionRight(Q[j]) - QuaternionLeft(P[j]);
		M.noalias() += w[j] * A.transpose() * A;
	}
	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen(M);
	Vector4d V = eigen.eigenvectors().col(0);
	Quaterniond quaternion(V(0), V(1), V(2), V(3));
	return quaternion;
}
