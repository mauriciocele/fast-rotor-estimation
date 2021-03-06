#include "FLAE.h"
#include <complex>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;
using std::complex;
using std::max;

Quaterniond FlaeSymbolic(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d  sigma_;

	const size_t N = P.size();
	sigma_.setZero();

	for (size_t i = 0; i < N; ++i) {
		sigma_.noalias() += (w[i] * P[i]) * Q[i].transpose();
	}

	Matrix3d A = sigma_ - sigma_.transpose();
	Matrix3d tmp = sigma_ + sigma_.transpose();
	Vector3d D(A(1, 2), A(2, 0), A(0, 1));
	Matrix4d QQ;
	QQ(0, 0) = sigma_(0, 0) + sigma_(1, 1) + sigma_(2, 2);
	tmp(0, 0) -= QQ(0, 0);    tmp(1, 1) -= QQ(0, 0);    tmp(2, 2) -= QQ(0, 0);
	QQ(0, 1) = D.x();     QQ(0, 2) = D.y();     QQ(0, 3) = D.z();
	QQ(1, 0) = D.x();     QQ(2, 0) = D.y();     QQ(3, 0) = D.z();

	QQ(1, 1) = tmp(0, 0); QQ(1, 2) = tmp(0, 1); QQ(1, 3) = tmp(0, 2);
	QQ(2, 1) = tmp(1, 0); QQ(2, 2) = tmp(1, 1); QQ(2, 3) = tmp(1, 2);
	QQ(3, 1) = tmp(2, 0); QQ(3, 2) = tmp(2, 1); QQ(3, 3) = tmp(2, 2);

	double c = QQ.determinant();
	double b = -8.0 * sigma_.determinant();
	double a = -2.0 * (sigma_(0, 0) * sigma_(0, 0) + sigma_(0, 1) * sigma_(0, 1) + sigma_(0, 2) * sigma_(0, 2) +
		sigma_(1, 0) * sigma_(1, 0) + sigma_(1, 1) * sigma_(1, 1) + sigma_(1, 2) * sigma_(1, 2) +
		sigma_(2, 0) * sigma_(2, 0) + sigma_(2, 1) * sigma_(2, 1) + sigma_(2, 2) * sigma_(2, 2));

	double T0 = 2.0 * a * a * a + 27.0 * b * b - 72.0 * a * c;
	double tt = a * a + 12.0 * c;
	double theta = atan2(sqrt(4.0 * tt * tt * tt - T0 * T0), T0);
	double aT1 = 1.259921049894873 * sqrt(tt) * cos(theta * 0.333333333333333333);
	double T2 = sqrt(-4.0 * a + 3.174802103936399 * aT1);
	double lambda = 0.204124145231932 * (T2 + sqrt(-T2 * T2 - 12.0 * a - 29.393876913398135 * b / T2));

	double G11 = QQ(0, 0) - lambda, G12 = QQ(0, 1), G13 = QQ(0, 2), G14 = QQ(0, 3);
	double G22 = QQ(1, 1) - lambda, G23 = QQ(1, 2), G24 = QQ(1, 3);
	double G33 = QQ(2, 2) - lambda, G34 = QQ(2, 3);
	double G44 = QQ(3, 3);

	Quaterniond qRes = Quaterniond(
		G14 * G23 * G23 - G13 * G23 * G24 - G14 * G22 * G33 + G12 * G24 * G33 + G13 * G22 * G34 - G12 * G23 * G34,
		G13 * G13 * G24 + G12 * G14 * G33 - G11 * G24 * G33 + G11 * G23 * G34 - G13 * G14 * G23 - G13 * G12 * G34,
		G13 * G14 * G22 - G12 * G14 * G23 - G12 * G13 * G24 + G11 * G23 * G24 + G12 * G12 * G34 - G11 * G22 * G34,
		-(G13 * G13 * G22 - 2 * G12 * G13 * G23 + G11 * G23 * G23 + G12 * G12 * G33 - G11 * G22 * G33));
	qRes.normalize();
	return qRes;
}

Quaterniond Flae(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d MM;
	Matrix4d W;
	double Hx1, Hx2, Hx3, Hy1, Hy2, Hy3, Hz1, Hz2, Hz3;
	double c, b, a, lambda, T0, s;
	complex<double> T1, T2, lambda1, lambda2, lambda3, lambda4;
	Vector4d W0, W1, W2;// , W3;
	Vector3d wQ;
	const double pow243 = 2.5198420997897463295344212145565; //pow(2, 4.0 / 3.0)
	const double pow223 = 1.5874010519681994747517056392723; //pow(2, 2.0 / 3.0)
	const size_t N = P.size();

	MM.setZero();
	for (size_t j = 0; j < N; ++j)
	{
		MM.noalias() += (w[j] * Q[j]) * P[j].transpose();
	}

	Hx1 = MM(0, 0);    Hx2 = MM(0, 1);    Hx3 = MM(0, 2);
	Hy1 = MM(1, 0);    Hy2 = MM(1, 1);    Hy3 = MM(1, 2);
	Hz1 = MM(2, 0);    Hz2 = MM(2, 1);    Hz3 = MM(2, 2);

	W(0, 0) = Hx1 + Hy2 + Hz3; W(0, 1) = -Hy3 + Hz2; W(0, 2) = -Hz1 + Hx3; W(0, 3) = -Hx2 + Hy1;
	W(1, 1) = Hx1 - Hy2 - Hz3; W(1, 2) = Hx2 + Hy1;  W(1, 3) = Hx3 + Hz1;
	W(2, 2) = Hy2 - Hx1 - Hz3; W(2, 3) = Hy3 + Hz2;
	W(3, 3) = Hz3 - Hy2 - Hx1;
	W(1, 0) = W(0, 1);
	W(2, 0) = W(0, 2);
	W(3, 0) = W(0, 3);
	W(2, 1) = W(1, 2);
	W(3, 1) = W(1, 3);
	W(3, 2) = W(2, 3);

	c = W.determinant();
	b = 8 * Hx3*Hy2*Hz1 - 8 * Hx2*Hy3*Hz1 - 8 * Hx3*Hy1*Hz2 + 8 * Hx1*Hy3*Hz2 + 8 * Hx2*Hy1*Hz3 - 8 * Hx1*Hy2*Hz3;
	a = -2 * Hx1*Hx1 - 2 * Hx2*Hx2 - 2 * Hx3*Hx3 - 2 * Hy1*Hy1 - 2 * Hy2*Hy2 - 2 * Hy3*Hy3 - 2 * Hz1*Hz1 - 2 * Hz2*Hz2 - 2 * Hz3*Hz3;

	double a2 = a*a;
	T0 = 2 * a2*a + 27 * b*b - 72 * a * c;
	s = a2 + 12 * c;
	T1 = pow(T0 + sqrt(complex<double>(-4.0 * s*s*s + T0*T0, 0)), 0.33333333333333333333333333333333);
	complex<double> T22 = -4.0*a + pow243*(a2 + 12 * c) / T1 + pow223*T1;
	T2 = sqrt(T22);

	complex<double> u = -T22 - 12 * a;
	complex<double> v = 12 * 2.4494897427831780981972840747059 * b / T2;
	complex<double> Suv = sqrt(u + v);
	complex<double> Duv = sqrt(u - v);
	lambda1 = 0.20412414523193150818310700622549*(T2 - Duv);
	lambda2 = 0.20412414523193150818310700622549*(T2 + Duv);
	lambda3 = -0.20412414523193150818310700622549*(T2 + Suv);
	lambda4 = -0.20412414523193150818310700622549*(T2 - Suv);
	//lambda = lambda2.real();
	lambda = max(lambda1.real(), max(lambda2.real(), max(lambda3.real(), lambda4.real())));

	W(0, 0) -= lambda;	W(1, 1) -= lambda;	W(2, 2) -= lambda;	W(3, 3) -= lambda;
	W0 = W.row(0);	W1 = W.row(1);	W2 = W.row(2);	//W3 = W.row(3);

	W0.noalias() = W0 / W0(0);
	W1.noalias() = W1 - W1(0)*W0;
	W2.noalias() = W2 - W2(0)*W0;
	//W3 = W3 - W3(0)*W0;

	W1.noalias() = W1 / W1(1);
	W0.noalias() = W0 - W0(1)*W1;
	W2.noalias() = W2 - W2(1)*W1;
	//W3 = W3 - W3(1)*W1;

	W2.noalias() = W2 / W2(2);
	W0.noalias() = W0 - W0(2)*W2;
	W1.noalias() = W1 - W1(2)*W2;
	//W3 = W3 - W3(2)*W2;

	Quaterniond quaternion(W0(3), W1(3), W2(3), -1);
	quaternion.normalize();
	return quaternion;
}

Quaterniond FlaeNewton(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d MM;
	Matrix4d W;
	double Hx1, Hx2, Hx3, Hy1, Hy2, Hy3, Hz1, Hz2, Hz3;
	double c, b, a, lambda;
	Vector4d W0, W1, W2;// , W3;
	const size_t N = P.size();

	MM.setZero();
	for (size_t j = 0; j < N; ++j)
	{
		MM.noalias() += (w[j] * Q[j]) * P[j].transpose();
	}

	Hx1 = MM(0, 0);    Hx2 = MM(0, 1);    Hx3 = MM(0, 2);
	Hy1 = MM(1, 0);    Hy2 = MM(1, 1);    Hy3 = MM(1, 2);
	Hz1 = MM(2, 0);    Hz2 = MM(2, 1);    Hz3 = MM(2, 2);

	W(0, 0) = Hx1 + Hy2 + Hz3; W(0, 1) = -Hy3 + Hz2; W(0, 2) = -Hz1 + Hx3; W(0, 3) = -Hx2 + Hy1;
	W(1, 1) = Hx1 - Hy2 - Hz3; W(1, 2) = Hx2 + Hy1;  W(1, 3) = Hx3 + Hz1;
	W(2, 2) = Hy2 - Hx1 - Hz3; W(2, 3) = Hy3 + Hz2;
	W(3, 3) = Hz3 - Hy2 - Hx1;
	W.selfadjointView<Eigen::Upper>().evalTo(W);

	c = W.determinant();
	b = 8 * Hx3*Hy2*Hz1 - 8 * Hx2*Hy3*Hz1 - 8 * Hx3*Hy1*Hz2 + 8 * Hx1*Hy3*Hz2 + 8 * Hx2*Hy1*Hz3 - 8 * Hx1*Hy2*Hz3;
	a = -2 * Hx1*Hx1 - 2 * Hx2*Hx2 - 2 * Hx3*Hx3 - 2 * Hy1*Hy1 - 2 * Hy2*Hy2 - 2 * Hy3*Hy3 - 2 * Hz1*Hz1 - 2 * Hz2*Hz2 - 2 * Hz3*Hz3;

	lambda = 1.0;
	double old_lambda = 0.0;
	double lamnda2, lamnda3;
	while (fabs(old_lambda - lambda) > 1e-5) {
		old_lambda = lambda;
		lamnda2 = lambda*lambda;
		lamnda3 = lamnda2 * lambda;
		lambda = lambda - ((lamnda3*lambda + a*lamnda2 + b*lambda + c) / (4 * lamnda3 + 2 * a*lambda + b));
	}

	W(0, 0) -= lambda;	W(1, 1) -= lambda;	W(2, 2) -= lambda;	W(3, 3) -= lambda;
	W0 = W.row(0);	W1 = W.row(1);	W2 = W.row(2);	//W3 = W.row(3);

	W0.noalias() = W0 / W0(0);
	W1.noalias() = W1 - W1(0)*W0;
	W2.noalias() = W2 - W2(0)*W0;
	//W3 = W3 - W3(0)*W0;

	W1.noalias() = W1 / W1(1);
	W0.noalias() = W0 - W0(1)*W1;
	W2.noalias() = W2 - W2(1)*W1;
	//W3 = W3 - W3(1)*W1;

	W2.noalias() = W2 / W2(2);
	W0.noalias() = W0 - W0(2)*W2;
	W1.noalias() = W1 - W1(2)*W2;
	//W3 = W3 - W3(2)*W2;

	Quaterniond quaternion(W0(3), W1(3), W2(3), -1);
	quaternion.normalize();
	return quaternion;
}
