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
	Vector4d g, R, R_i;
	Vector3d S, D;
	const double epsilon = 1e-6;
	double wj;
	const size_t N = P.size();

	H.setZero();
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		S = Q[j] + P[j];
		D = P[j] - Q[j];
		H(0, 0) += wj*(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);
		H(0, 1) += wj*(D[0] * S[1] - D[1] * S[0]);
		H(0, 2) += wj*(D[0] * S[2] - D[2] * S[0]);
		H(0, 3) += wj*(D[1] * S[2] - D[2] * S[1]);
		H(1, 1) += wj*(S[1] * S[1] + S[0] * S[0] + D[2] * D[2]);
		H(1, 2) += wj*(S[1] * S[2] - D[2] * D[1]);
		H(1, 3) += wj*(D[2] * D[0] - S[0] * S[2]);
		H(2, 2) += wj*(S[2] * S[2] + S[0] * S[0] + D[1] * D[1]);
		H(2, 3) += wj*(S[0] * S[1] - D[1] * D[0]);
		H(3, 3) += wj*(S[2] * S[2] + S[1] * S[1] + D[0] * D[0]);
	}
	H(1, 0) = H(0, 1);
	H(2, 0) = H(0, 2); H(2, 1) = H(1, 2);
	H(3, 0) = H(0, 3); H(3, 1) = H(1, 3); H(3, 2) = H(2, 3);
	g = -H.col(0);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H = H.inverse();
	R(0) = 1; R(1) = 0; R(2) = 0; R(3) = 0;
	do {
		R_i = R;
		R(0) -= 1.0;
		R.noalias() = H * (g + epsilon * R);
		R(0) += 1.0;
		R /= sqrt(R.dot(R));
	} while ((R_i - R).dot(R_i - R) > 1e-6);
	return Quaterniond(R(0), -R(3), R(2), -R(1));
}

Quaterniond GAFastRotorEstimatorIncr(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const Quaterniond& Qprev)
{
	Matrix4d H;
	Vector4d g, R, R_i(Qprev.w() - 1.0, -Qprev.z(), Qprev.y(), -Qprev.x());
	Vector3d S, D;
	const double epsilon = 1e-6;
	double S1, S2, S3, D1, D2, D3, wj;
	int N = P.size();

	H.setZero();
	for (int j = 0; j < N; ++j)
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
	H(1, 0) = H(0, 1);
	H(2, 0) = H(0, 2); H(2, 1) = H(1, 2);
	H(3, 0) = H(0, 3); H(3, 1) = H(1, 3); H(3, 2) = H(2, 3);
	g = -H.row(0);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	R.noalias() = H.inverse() * (g + epsilon * R_i);
	R(0) += 1.0;
	R.noalias() = R / sqrt(R.dot(R));
	return Quaterniond(R(0), -R(3), R(2), -R(1));
}

Quaterniond GANewtonRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	const size_t MAX_VECTORS = 32;
	const size_t N = P.size() < MAX_VECTORS ? P.size() : MAX_VECTORS;
	Matrix4d Jt[MAX_VECTORS], H;
	Vector4d Fi, g, Ri, R(1, 0, 0, 0);
	Vector3d D, S;
	double wi;

	H.setZero();
	for (size_t i = 0; i < N; ++i) {
		wi = w[i];
		S = Q[i] + P[i];
		D = P[i] - Q[i];
		Matrix4d& Jti = Jt[i];
		Jti(0, 0) = wi*D[0]; Jti(0, 1) = wi*D[1];  Jti(0, 2) = wi*D[2];  Jti(0, 3) = 0;
		Jti(1, 0) = wi*S[1]; Jti(1, 1) = -wi*S[0]; Jti(1, 2) = 0;        Jti(1, 3) = wi*D[2];
		Jti(2, 0) = wi*S[2]; Jti(2, 1) = 0;        Jti(2, 2) = -wi*S[0]; Jti(2, 3) = -wi*D[1];
		Jti(3, 0) = 0;       Jti(3, 1) = wi*S[2];  Jti(3, 2) = -wi*S[1]; Jti(3, 3) = wi*D[0];
		H(0, 0) += wi*(D[0] * D[0] + D[1] * D[1] + D[2] * D[2]);
		H(0, 1) += wi*(D[0] * S[1] - D[1] * S[0]);
		H(0, 2) += wi*(D[0] * S[2] - D[2] * S[0]);
		H(0, 3) += wi*(D[1] * S[2] - D[2] * S[1]);
		H(1, 1) += wi*(S[1] * S[1] + S[0] * S[0] + D[2] * D[2]);
		H(1, 2) += wi*(S[1] * S[2] - D[2] * D[1]);
		H(1, 3) += wi*(D[2] * D[0] - S[0] * S[2]);
		H(2, 2) += wi*(S[2] * S[2] + S[0] * S[0] + D[1] * D[1]);
		H(2, 3) += wi*(S[0] * S[1] - D[1] * D[0]);
		H(3, 3) += wi*(S[2] * S[2] + S[1] * S[1] + D[0] * D[0]);
	}
	H(1, 0) = H(0, 1);
	H(2, 0) = H(0, 2); H(2, 1) = H(1, 2);
	H(3, 0) = H(0, 3); H(3, 1) = H(1, 3); H(3, 2) = H(2, 3);
	H(0, 0) += 1e-6; H(1, 1) += 1e-6; H(2, 2) += 1e-6; H(3, 3) += 1e-6;
	H = H.inverse();
	do {
		Ri = R;
		g.setZero();
		for (size_t i = 0; i < N; ++i) {
			S = Q[i] + P[i];
			D = P[i] - Q[i];
			Fi(0) = -(S[2] * R[2] + S[1] * R[1] + D[0] * R[0]);
			Fi(1) = -(S[2] * R[3] - S[0] * R[1] + D[1] * R[0]);
			Fi(2) = -(-S[1] * R[3] - S[0] * R[2] + D[2] * R[0]);
			Fi(3) = -(D[0] * R[3] - D[1] * R[2] + D[2] * R[1]);
			g += Jt[i] * Fi;
		}
		R += H * g;
		R /= sqrt(R.dot(R));
	} while ((Ri - R).dot(Ri - R) > 1e-6);
	return Quaterniond(R[0], -R[3], R[2], -R[1]);
}
