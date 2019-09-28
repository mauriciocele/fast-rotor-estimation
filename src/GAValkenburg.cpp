#include "GAValkenburg.h"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;

Vector4d Lmap(const Vector3d& X, const vector<Vector3d>& Ps, const vector<Vector3d>& Qs, const vector<double>& w)
{
	//X = X1*e1*e2 + X2*e1*e3 + X3*e2*e3;
	//P = P1*e1 + P2*e2 + P3*e3;
	//Q = Q1*e1 + Q2*e2 + Q3*e3;
	//?QX = Q*X;
	//?Y = QX*P;
	Vector4d Y;
	Vector4d QX;
	Vector4d R;
	R.setZero();
	const size_t N = Ps.size();
	for (size_t k = 0; k < N; ++k)
	{
		const Vector3d& P = Ps[k];
		const Vector3d& Q = Qs[k];
		QX[0] = (-(Q[2] * X[1])) - Q[1] * X[0]; //e1
		QX[1] = Q[0] * X[0] - Q[2] * X[2]; //e2
		QX[2] = Q[1] * X[2] + Q[0] * X[1]; //e3
		QX[3] = Q[0] * X[2] - Q[1] * X[1] + Q[2] * X[0]; // e1^e2^e3
		Y[0] = P[2] * QX[2] + P[1] * QX[1] + P[0] * QX[0]; // 1
		Y[1] = P[2] * QX[3] - P[0] * QX[1] + P[1] * QX[0]; // e1^e2
		Y[2] = (-(P[1] * QX[3])) - P[0] * QX[2] + P[2] * QX[0]; // e1^e3
		Y[3] = P[0] * QX[3] - P[1] * QX[2] + P[2] * QX[1]; // e2^e3
		R += w[k] * Y;
	}
	return R;
}

Vector4d Lmap(const vector<Vector3d>& Ps, const vector<Vector3d>& Qs, const vector<double>& w)
{
	//P = P1*e1 + P2*e2 + P3*e3;
	//Q = Q1*e1 + Q2*e2 + Q3*e3;
	//?Y = Q*P;
	Vector4d Y;
	Vector4d R;
	R.setZero();
	const size_t N = Ps.size();
	for (size_t k = 0; k < N; ++k)
	{
		const Vector3d& P = Ps[k];
		const Vector3d& Q = Qs[k];
		Y[0] = P[2] * Q[2] + P[1] * Q[1] + P[0] * Q[0]; // 1.0
		Y[1] = P[1] * Q[0] - P[0] * Q[1]; // e1 ^ e2
		Y[2] = P[2] * Q[0] - P[0] * Q[2]; // e1 ^ e3
		Y[3] = P[2] * Q[1] - P[1] * Q[2]; // e2 ^ e3
		R += w[k] * Y;
	}
	return R;
}

double BivectorXRotor(const Vector3d& X, const Vector4d& R)
{
	//R = R1 + R2*e1*e2 + R3*e1*e3 + R4*e2*e3;
	//X = X1*e1*e2 + X2*e1*e3 + X3*e2*e3;
	//?Lij = X * R;
	//Vector4d Lij;
	//Lij[0] = (-(R[3] * X[2])) - R[2] * X[1] - R[1] * X[0]; // 1.0
	//Lij[1] = R[2] * X[2] - R[3] * X[1] + R[0] * X[0]; // e1 ^ e2
	//Lij[2] = (-(R[1] * X[2])) + R[0] * X[1] + R[3] * X[0]; // e1 ^ e3
	//Lij[3] = (R[0] * X[2] + R[1] * X[1]) - R[2] * X[0]; // e2 ^ e3
	//return Lij;
	return (-(R[3] * X[2])) - R[2] * X[1] - R[1] * X[0]; // 1.0
}

Quaterniond GAValkenburg(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	static const Vector3d IM[4] = { Vector3d(0,0,0), Vector3d(1,0,0), Vector3d(0,-1,0), Vector3d(0,0,1) };
	static const Vector3d IMi[4] = { Vector3d(0,0,0), Vector3d(-1,0,0), Vector3d(0,1,0), Vector3d(0,0,-1) };

	Matrix4d L;
	for (int j = 0; j < 4; ++j)
	{
		Vector4d Lj = j == 0 ? Lmap(P, Q, w) : Lmap(IM[j], P, Q, w);
		L(j, j) = j == 0 ? Lj[0] : BivectorXRotor(IMi[j], Lj);
		for (int i = j + 1; i < 4; ++i)
		{
			L(i, j) = BivectorXRotor(IMi[i], Lj);
			L(j, i) = L(i, j);
		}
	}

	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen(L);
	Vector4d Rot = eigen.eigenvectors().col(3);
	return Quaterniond(Rot(0), -Rot(3), -Rot(2), -Rot(1));
}
