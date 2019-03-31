#include "GARotorEstimator.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;

#include "GARotorEstimator.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;

Quaterniond GAFastRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Vector4d R, R_i;
	Vector3d S, D;
	double S1, S2, S3, D1, D2, D3, wj;
	constexpr double epsilon = 5e-2;
	const size_t N = P.size();

	H.setZero();
	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D3*S2 - D2*S3); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(1, 2) += wj*(D1*D2 - S2*S1); H(1, 3) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(2, 3) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(H);
	H = epsilon * H.inverse().eval();
	R(0) = 1; R(1) = R(2) = R(3) = epsilon;
	do {
		R_i = R;
		R = H * R;
	} while ((R_i - R).lpNorm<1>() > 1e-13);
	R.normalize();
	return Quaterniond(R(0), R(1), R(2), R(3));
}


Quaterniond GAFastRotorEstimatorAprox(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, double epsilon, size_t steps)
{
	Matrix4d H;
	Vector4d R;
	Vector3d S, D;
	double S1, S2, S3, D1, D2, D3, wj;
	const size_t N = P.size();

	H.setZero();
	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D3*S2 - D2*S3); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(1, 2) += wj*(D1*D2 - S2*S1); H(1, 3) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(2, 3) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(H);
	H = epsilon * H.inverse().eval();
	R(0) = 1; R(1) = R(2) = R(3) = epsilon;
	while (steps--) {
		R = H * R;
	}
	R.normalize();
	return Quaterniond(R(0), R(1), R(2), R(3));
}

Quaterniond GAFastRotorEstimatorIncr(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const Quaterniond& Qprev)
{
	Matrix4d H;
	Vector4d R(Qprev.w(), Qprev.x(), Qprev.y(), Qprev.z());
	Vector3d S, D;
	constexpr double epsilon = 1e-6;
	double S1, S2, S3, D1, D2, D3, wj;
	size_t N = P.size();

	H.setZero();
	for (int j = 0; j < N; ++j)
	{
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D3*S2 - D2*S3); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(1, 2) += wj*(D1*D2 - S2*S1); H(1, 3) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(2, 3) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(H);
	R = H.inverse().selfadjointView<Eigen::Upper>() * R;
	R.normalize();
	return Quaterniond(R(0), R(1), R(2), R(3));
}

Quaterniond GANewtonRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	const size_t N = P.size();
	Matrix4d H, Hinv;
	Vector4d Ri, R(1, 0, 0, 0);
	Vector3d D, S;
	double wj;
	double S1, S2, S3, D1, D2, D3;
	constexpr double epsilon = 1e-6;
	H.setZero();
	for (register size_t i = 0; i < N; ++i) {
		wj = w[i];
		S = Q[i] + P[i];
		D = P[i] - Q[i];
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D3*S2 - D2*S3); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(1, 2) += wj*(D1*D2 - S2*S1); H(1, 3) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(2, 3) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H.selfadjointView<Eigen::Upper>().evalTo(H);
	Hinv = H;
	Hinv(0, 0) += epsilon; Hinv(1, 1) += epsilon; Hinv(2, 2) += epsilon; Hinv(3, 3) += epsilon;
	Hinv = -Hinv.inverse() * H;
	do {
		Ri = R;
		R += Hinv * R;
	} while ((Ri - R).lpNorm<1>() > 1e-8);
	R.normalize();
	return Quaterniond(R[0], R[1], R[2], R[3]);
}
