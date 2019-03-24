#include "GARotorEstimator.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;

Quaterniond GAFastRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H, Htemp;
	Vector4d g, R, R_i;
	Vector3d S, D;
	double S1, S2, S3, D1, D2, D3, wj;
	const double epsilon = 1e-6;
	const size_t N = P.size();

	H.setZero();
	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D1*S2 - D2*S1); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S3 - D3*S2);
		H(1, 1) += wj*(S2*S2 + S1*S1 + D3*D3); H(1, 2) += wj*(S2*S3 - D3*D2); H(1, 3) += wj*(D3*D1 - S1*S3);
		H(2, 2) += wj*(S3*S3 + S1*S1 + D2*D2); H(2, 3) += wj*(S1*S2 - D2*D1);
		H(3, 3) += wj*(S3*S3 + S2*S2 + D1*D1);
	}
	g = -H.row(0);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(Htemp);
	Htemp.inverse().selfadjointView<Eigen::Upper>().evalTo(H);
	R(0) = 1; R(1) = 0; R(2) = 0; R(3) = 0;
	do {
		R_i = R;
		R(0) -= 1.0;
		R.noalias() = H * (g + epsilon * R);
		R(0) += 1.0;
		R.normalize();
	} while ((R_i - R).dot(R_i - R) > 1e-6);
	return Quaterniond(R(0), -R(3), R(2), -R(1));
}

Quaterniond GAFastRotorEstimatorAprox(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, double epsilon, size_t steps)
{
	Matrix4d H, Htemp;
	Vector4d g, R;
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
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D1*S2 - D2*S1); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S3 - D3*S2);
		H(1, 1) += wj*(S2*S2 + S1*S1 + D3*D3); H(1, 2) += wj*(S2*S3 - D3*D2); H(1, 3) += wj*(D3*D1 - S1*S3);
		H(2, 2) += wj*(S3*S3 + S1*S1 + D2*D2); H(2, 3) += wj*(S1*S2 - D2*D1);
		H(3, 3) += wj*(S3*S3 + S2*S2 + D1*D1);
	}
	g = -H.row(0);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(Htemp);
	Htemp.inverse().selfadjointView<Eigen::Upper>().evalTo(H);
	R(0) = 1; R(1) = 0; R(2) = 0; R(3) = 0;
	while(steps--)
	{
		R(0) -= 1.0;
		R.noalias() = H * (g + epsilon * R);
		R(0) += 1.0;
		R.normalize();
	}
	return Quaterniond(R(0), -R(3), R(2), -R(1));
}

Quaterniond GAFastRotorEstimatorIncr(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const Quaterniond& Qprev)
{
	Matrix4d H, Htemp;
	Vector4d g, R, R_i(Qprev.w() - 1.0, -Qprev.z(), Qprev.y(), -Qprev.x());
	Vector3d S, D;
	const double epsilon = 1e-6;
	double S1, S2, S3, D1, D2, D3, wj;
	const size_t N = P.size();

	H.setZero();
	for (register size_t j = 0; j < N; ++j)
	{
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(0, 1) += wj*(D1*S2 - D2*S1); H(0, 2) += wj*(D1*S3 - D3*S1); H(0, 3) += wj*(D2*S3 - D3*S2);
		H(1, 1) += wj*(S2*S2 + S1*S1 + D3*D3); H(1, 2) += wj*(S2*S3 - D3*D2); H(1, 3) += wj*(D3*D1 - S1*S3);
		H(2, 2) += wj*(S3*S3 + S1*S1 + D2*D2); H(2, 3) += wj*(S1*S2 - D2*D1);
		H(3, 3) += wj*(S3*S3 + S2*S2 + D1*D1);
	}
	g = -H.row(0);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Upper>().evalTo(Htemp);
	R.noalias() = Htemp.inverse().selfadjointView<Eigen::Upper>() * (g + epsilon * R_i);
	R(0) += 1.0;
	R.normalize();
	return Quaterniond(R(0), -R(3), R(2), -R(1));
}

Quaterniond GANewtonRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	const size_t MAX_VECTORS = 32;
	const size_t N = P.size() < MAX_VECTORS ? P.size() : MAX_VECTORS;
	Matrix4d Jt[MAX_VECTORS], H, Hinv;
	Vector4d Fi, g, Ri, R(1, 0, 0, 0);
	Vector3d D, S;
	double wj;
	double S1, S2, S3, D1, D2, D3;

	H.setZero();
	for (register size_t i = 0; i < N; ++i) {
		wj = w[i];
		S = Q[i] + P[i];
		D = P[i] - Q[i];
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		Matrix4d& Jti = Jt[i];
		Jti(0, 0) = wj*D1; Jti(0, 1) = wj*D2;  Jti(0, 2) = wj*D3;  Jti(0, 3) = 0;
		Jti(1, 0) = wj*S2; Jti(1, 1) = -wj*S1; Jti(1, 2) = 0;      Jti(1, 3) = wj*D3;
		Jti(2, 0) = wj*S3; Jti(2, 1) = 0;      Jti(2, 2) = -wj*S1; Jti(2, 3) = -wj*D2;
		Jti(3, 0) = 0;     Jti(3, 1) = wj*S3;  Jti(3, 2) = -wj*S2; Jti(3, 3) = wj*D1;
		H(0, 0) += wj * (D1*D1 + D2 * D2 + D3 * D3); H(0, 1) += wj * (D1*S2 - D2 * S1); H(0, 2) += wj * (D1*S3 - D3 * S1); H(0, 3) += wj * (D2*S3 - D3 * S2);
		H(1, 1) += wj * (S2*S2 + S1 * S1 + D3 * D3); H(1, 2) += wj * (S2*S3 - D3 * D2); H(1, 3) += wj * (D3*D1 - S1 * S3);
		H(2, 2) += wj * (S3*S3 + S1 * S1 + D2 * D2); H(2, 3) += wj * (S1*S2 - D2 * D1);
		H(3, 3) += wj * (S3*S3 + S2 * S2 + D1 * D1);
	}
	H(0, 0) += 1e-6; H(1, 1) += 1e-6; H(2, 2) += 1e-6; H(3, 3) += 1e-6;
	H.selfadjointView<Eigen::Upper>().evalTo(Hinv);
	Hinv.inverse().selfadjointView<Eigen::Upper>().evalTo(H);
	do {
		Ri = R;
		g.setZero();
		for (register size_t i = 0; i < N; ++i) {
			S = Q[i] + P[i];
			D = P[i] - Q[i];
			S1 = S.x(); S2 = S.y(); S3 = S.z();
			D1 = D.x(); D2 = D.y(); D3 = D.z();
			Fi(0) = -(S3 * R[2] + S2 * R[1] + D1 * R[0]);
			Fi(1) = -(S3 * R[3] - S1 * R[1] + D2 * R[0]);
			Fi(2) = -(-S2 * R[3] - S1 * R[2] + D3 * R[0]);
			Fi(3) = -(D1 * R[3] - D2 * R[2] + D3 * R[1]);
			g += Jt[i] * Fi;
		}
		R += H * g;
		R.normalize();
	} while ((Ri - R).dot(Ri - R) > 1e-6);
	return Quaterniond(R[0], -R[3], R[2], -R[1]);
}
