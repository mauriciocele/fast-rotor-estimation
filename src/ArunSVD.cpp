#include "ArunSVD.h"
#include "svd3.h"

using Eigen::Vector3d;
using Eigen::Matrix3d;
using std::vector;

Matrix3d SVDEigen(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d m;
	const size_t N = P.size();
	m.setZero();
	for (size_t j = 0; j < N; ++j)
	{
		m += (w[j] * P[j]) * Q[j].transpose();
	}
	Eigen::JacobiSVD<Matrix3d> svdjac(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Matrix3d& U = svdjac.matrixU();
	Matrix3d V = svdjac.matrixV();
	// V U^T is a reflection when det(V U^T) = -1 (e.g. planar or very noisy data);
	// flipping the singular vector of the smallest singular value gives the
	// closest rotation instead (Arun et al. 1987, Umeyama 1991).
	if ((V * U.transpose()).determinant() < 0.0)
		V.col(2) *= -1.0;
	return V * U.transpose();
}

Matrix3d SVDMcAdams(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Vector3d eij, teij;
	Matrix3d A;
	double a11; double a12; double a13;
	double a21; double a22; double a23;
	double a31; double a32; double a33;

	const size_t N = P.size();

	a11 = a12 = a13 = 0.0;
	a21 = a22 = a23 = 0.0;
	a31 = a32 = a33 = 0.0;
	for (size_t j = 0; j < N; ++j)
	{
		eij = Q[j] * w[j];
		teij = P[j];
		a11 += eij.x() * teij.x();
		a12 += eij.x() * teij.y();
		a13 += eij.x() * teij.z();
		a21 += eij.y() * teij.x();
		a22 += eij.y() * teij.y();
		a23 += eij.y() * teij.z();
		a31 += eij.z() * teij.x();
		a32 += eij.z() * teij.y();
		a33 += eij.z() * teij.z();
	}

	double w11, w12, w13, w21, w22, w23, w31, w32, w33;
	double s11, s12, s13, s21, s22, s23, s31, s32, s33;
	double v11, v12, v13, v21, v22, v23, v31, v32, v33;

	svd(a11, a12, a13, a21, a22, a23, a31, a32, a33,
		w11, w12, w13, w21, w22, w23, w31, w32, w33,
		s11, s12, s13, s21, s22, s23, s31, s32, s33,
		v11, v12, v13, v21, v22, v23, v31, v32, v33);
	// A = sum w Q P^T = W S V^T, so the rotation taking P to Q is R = W V^T.
	// W and V are proper rotations (the sign of det(A) is kept in s33), so R is too.
	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	multAB(w11, w12, w13, w21, w22, w23, w31, w32, w33,
		v11, v21, v31, v12, v22, v32, v13, v23, v33,
		r11, r12, r13, r21, r22, r23, r31, r32, r33);
	Matrix3d R;
	R << r11, r12, r13,
	     r21, r22, r23,
	     r31, r32, r33;
	return R;
}
