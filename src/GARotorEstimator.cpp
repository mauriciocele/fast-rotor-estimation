#include "GARotorEstimator.h"

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;

Quaterniond GAFastRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Eigen::Matrix4d H;
	Eigen::Vector4d R;
	Eigen::Vector3d S, D;
	double S0, S1, S2, S3, D1, D2, D3, wj;
	const double epsilon = 1e-13;
	const size_t N = P.size(), MAX_STEPS = 12;

	H.setZero();
	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(1, 0) += wj*(D3*S2 - D2*S3); H(2, 0) += wj*(D1*S3 - D3*S1); H(3, 0) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(2, 1) += wj*(D1*D2 - S2*S1); H(3, 1) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(3, 2) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H = H.inverse().eval();
	for (register size_t j = 0; j < MAX_STEPS; ++j) {
		H *= H;
		H *= 1.0 / H.trace();
	}
	S0 = H.col(0).lpNorm<1>();
	S1 = H.col(1).lpNorm<1>();
	S2 = H.col(2).lpNorm<1>();
	S3 = H.col(3).lpNorm<1>();
	wj = std::max(S3, std::max(S2, std::max(S0, S1)));
	if (wj == S0) R = H.col(0);
	else if (wj == S1) R = H.col(1);
	else if (wj == S2) R = H.col(2);
	else R = H.col(3);
	R.normalize();
	return Quaterniond(R(0), R(1), R(2), R(3));
}

Quaterniond GAFastRotorEstimatorAVX(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Vector4d R;
	Vector3d S, D;
	double S0, S1, S2, S3, D1, D2, D3, wj;
	const double epsilon = 1e-13;
	const size_t N = P.size(), MAX_STEPS = 12;
	const double* DP = nullptr;

	H.setZero();
	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		S1 = S.x(); S2 = S.y(); S3 = S.z();
		D1 = D.x(); D2 = D.y(); D3 = D.z();
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(1, 0) += wj*(D3*S2 - D2*S3); H(2, 0) += wj*(D1*S3 - D3*S1); H(3, 0) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(2, 1) += wj*(D1*D2 - S2*S1); H(3, 1) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(3, 2) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H = H.inverse().eval();
	for (register size_t j = 0; j < MAX_STEPS; ++j) {
		wj = 1.0 / H.trace();
		__m256d C1 = _mm256_setr_pd(H(0, 0), H(1, 0), H(2, 0), H(3, 0));
		__m256d C2 = _mm256_setr_pd(H(1, 0), H(1, 1), H(2, 1), H(3, 1));
		__m256d C3 = _mm256_setr_pd(H(2, 0), H(2, 1), H(2, 2), H(3, 2));
		__m256d C4 = _mm256_setr_pd(H(3, 0), H(3, 1), H(3, 2), H(3, 3));
		__m256d weights = _mm256_setr_pd(wj, wj, wj, wj);

		__m256d xy0 = _mm256_mul_pd(C1, C1);
		__m256d xy1 = _mm256_mul_pd(C1, C2);
		__m256d xy2 = _mm256_mul_pd(C1, C3);
		__m256d xy3 = _mm256_mul_pd(C1, C4);

		// low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
		__m256d temp01 = _mm256_hadd_pd(xy0, xy1);
		// low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
		__m256d temp23 = _mm256_hadd_pd(xy2, xy3);
		// low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
		__m256d swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
		// low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
		__m256d blended = _mm256_blend_pd(temp01, temp23, 0b1100);
		__m256d dotproduct = _mm256_mul_pd(_mm256_add_pd(swapped, blended), weights);
		DP = (double*)& dotproduct;
		H(0, 0) = DP[0]; H(1, 0) = DP[1]; H(2, 0) = DP[2]; H(3, 0) = DP[3];

		xy0 = _mm256_mul_pd(C2, C2);
		xy1 = _mm256_mul_pd(C2, C3);
		xy2 = _mm256_mul_pd(C2, C4);
		xy3 = _mm256_mul_pd(C3, C3);

		// low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
		temp01 = _mm256_hadd_pd(xy0, xy1);
		// low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
		temp23 = _mm256_hadd_pd(xy2, xy3);
		// low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
		swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
		// low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
		blended = _mm256_blend_pd(temp01, temp23, 0b1100);
		dotproduct = _mm256_mul_pd(_mm256_add_pd(swapped, blended), weights);
		DP = (double*)& dotproduct;
		H(1, 1) = DP[0]; H(2, 1) = DP[1]; H(3, 1) = DP[2]; H(2, 2) = DP[3];

		xy0 = _mm256_mul_pd(C3, C4);
		xy1 = _mm256_mul_pd(C4, C4);
		temp01 = _mm256_mul_pd(_mm256_hadd_pd(xy0, xy1), weights);
		__m128d hi128 = _mm256_extractf128_pd(temp01, 1);
		__m128d dotproduct2 = _mm_add_pd(_mm256_castpd256_pd128(temp01), hi128);
		DP = (double*)& dotproduct2;

		H(3, 2) = DP[0]; H(3, 3) = DP[1];
	}
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	S0 = H.col(0).lpNorm<1>();
	S1 = H.col(1).lpNorm<1>();
	S2 = H.col(2).lpNorm<1>();
	S3 = H.col(3).lpNorm<1>();
	wj = std::max(S3, std::max(S2, std::max(S0, S1)));
	if (wj == S0) R = H.col(0);
	else if (wj == S1) R = H.col(1);
	else if (wj == S2) R = H.col(2);
	else R = H.col(3);
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
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(1, 0) += wj*(D3*S2 - D2*S3); H(2, 0) += wj*(D1*S3 - D3*S1); H(3, 0) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(2, 1) += wj*(D1*D2 - S2*S1); H(3, 1) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(3, 2) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	H = H.inverse().eval();
	H *= H;
	H *= 1.0 / H.trace();
	H *= H;
	H *= 1.0 / H.trace();
	D1 = H.col(0).lpNorm<1>();
	S1 = H.col(1).lpNorm<1>();
	S2 = H.col(2).lpNorm<1>();
	S3 = H.col(3).lpNorm<1>();
	wj = std::max(S3, std::max(S2, std::max(D1, S1)));
	if (wj == D1) R = H.col(0);
	else if (wj == S1) R = H.col(1);
	else if (wj == S2) R = H.col(2);
	else R = H.col(3);
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
	const double epsilon = -1e-6;
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
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(1, 0) += wj*(D3*S2 - D2*S3); H(2, 0) += wj*(D1*S3 - D3*S1); H(3, 0) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(2, 1) += wj*(D1*D2 - S2*S1); H(3, 1) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(3, 2) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H(0, 0) += epsilon; H(1, 1) += epsilon; H(2, 2) += epsilon; H(3, 3) += epsilon;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	R = H.inverse() * R;
	R.normalize();
	return Quaterniond(R(0), R(1), R(2), R(3));
}

Quaterniond GANewtonRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	const double epsilon = 5e-2;
	const size_t N = P.size();
	Matrix4d H, Hinv;
	Vector4d R_i, R(1, epsilon, epsilon, epsilon);
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
		H(0, 0) += wj*(D1*D1 + D2*D2 + D3*D3); H(1, 0) += wj*(D3*S2 - D2*S3); H(2, 0) += wj*(D1*S3 - D3*S1); H(3, 0) += wj*(D2*S1 - D1*S2);
		H(1, 1) += wj*(D1*D1 + S3*S3 + S2*S2); H(2, 1) += wj*(D1*D2 - S2*S1); H(3, 1) += wj*(D1*D3 - S3*S1);
		H(2, 2) += wj*(D2*D2 + S3*S3 + S1*S1); H(3, 2) += wj*(D2*D3 - S3*S2);
		H(3, 3) += wj*(D3*D3 + S2*S2 + S1*S1);
	}
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	Hinv = H;
	Hinv(0, 0) += epsilon; Hinv(1, 1) += epsilon; Hinv(2, 2) += epsilon; Hinv(3, 3) += epsilon;
	Hinv = Hinv.inverse() * H;
	do {
		R_i = R;
		R = R - Hinv * R;
	} while ((R_i - R).lpNorm<1>() > 1e-13);
	R.normalize();
	return Quaterniond(R[0], R[1], R[2], R[3]);
}
