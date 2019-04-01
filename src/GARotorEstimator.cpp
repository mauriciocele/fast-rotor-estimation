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

Quaterniond GAFastRotorEstimatorAVX2(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Vector4d R, R_i;
	Vector3d S, D;
	double wj;
	constexpr double epsilon = 5e-2;
	const size_t N = P.size();

	const double *DP;
	__m256d H0, H1;
	__m128d H2;

	H0 = _mm256_setr_pd(0, 0, 0, 0);
	H1 = _mm256_setr_pd(0, 0, 0, 0);
	H2 = _mm_setr_pd(0, 0);

	for (register size_t j = 0; j < N; ++j) {
		wj = (w[j]);
		S = (Q[j] + P[j]);
		D = (P[j] - Q[j]);
		__m256d J1 = _mm256_setr_pd(D.x(), D.y(), D.z(), 0);
		__m256d J2 = _mm256_setr_pd(S.y(), -S.x(), 0, D.z());
		__m256d J3 = _mm256_setr_pd(S.z(), 0, -S.x(), -D.y());
		__m256d J4 = _mm256_setr_pd(0, S.z(), -S.y(), D.x());
		__m256d weights = _mm256_setr_pd(wj, wj, wj, wj);

		__m256d xy0 = _mm256_mul_pd(J1, J1);
		__m256d xy1 = _mm256_mul_pd(J1, J2);
		__m256d xy2 = _mm256_mul_pd(J1, J3);
		__m256d xy3 = _mm256_mul_pd(J1, J4);

		// low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
		__m256d temp01 = _mm256_hadd_pd(xy0, xy1);
		// low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
		__m256d temp23 = _mm256_hadd_pd(xy2, xy3);
		// low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
		__m256d swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
		// low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
		__m256d blended = _mm256_blend_pd(temp01, temp23, 0b1100);
		__m256d dotproduct = _mm256_mul_pd(_mm256_add_pd(swapped, blended), weights);
		//DP = (double *)&dotproduct;
		
		H0 = _mm256_add_pd(H0, dotproduct);

		xy0 = _mm256_mul_pd(J2, J2);
		xy1 = _mm256_mul_pd(J2, J3);
		xy2 = _mm256_mul_pd(J2, J4);
		xy3 = _mm256_mul_pd(J3, J3);

		// low to high: xy00+xy01 xy10+xy11 xy02+xy03 xy12+xy13
		temp01 = _mm256_hadd_pd(xy0, xy1);
		// low to high: xy20+xy21 xy30+xy31 xy22+xy23 xy32+xy33
		temp23 = _mm256_hadd_pd(xy2, xy3);
		// low to high: xy02+xy03 xy12+xy13 xy20+xy21 xy30+xy31
		swapped = _mm256_permute2f128_pd(temp01, temp23, 0x21);
		// low to high: xy00+xy01 xy10+xy11 xy22+xy23 xy32+xy33
		blended = _mm256_blend_pd(temp01, temp23, 0b1100);
		dotproduct = _mm256_mul_pd(_mm256_add_pd(swapped, blended), weights);

		H1 = _mm256_add_pd(H1, dotproduct);

		xy0 = _mm256_mul_pd(J3, J4);
		xy1 = _mm256_mul_pd(J4, J4);
		temp01 = _mm256_mul_pd(_mm256_hadd_pd(xy0, xy1), weights);
		__m128d hi128 = _mm256_extractf128_pd(temp01, 1);
		__m128d dotproduct2 = _mm_add_pd(_mm256_castpd256_pd128(temp01), hi128);

		H2 = _mm_add_pd(H2, dotproduct2);
	}
	DP = (double *)&H0;
	H(0, 0) = DP[0] + epsilon; H(0, 1) = DP[1]; H(0, 2) = DP[2]; H(0, 3) = DP[3];
	DP = (double *)&H1;
	H(1, 1) = DP[0] + epsilon; H(1, 2) = DP[1]; H(1, 3) = DP[2]; H(2, 2) = DP[3] + epsilon;
	DP = (double *)&H2;
	H(2, 3) = DP[0]; H(3, 3) = DP[1] + epsilon;

	H.selfadjointView<Eigen::Upper>().evalTo(H);
	H = epsilon * H.inverse().eval();
	R(0) = 1; R(1) = R(2) = R(3) = epsilon;
	do {
		R_i = R;
		R = H * R;
	} while ((R_i - R).lpNorm<1>() > 1e-13);
	R.normalize();
	return Quaterniond(R(0), -R(3), R(2), -R(1));
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
	constexpr double epsilon = 5e-3;
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
	H.selfadjointView<Eigen::Upper>().evalTo(Hinv);
	Hinv(0, 0) += epsilon; Hinv(1, 1) += epsilon; Hinv(2, 2) += epsilon; Hinv(3, 3) += epsilon;
	Hinv = -Hinv.inverse() * H.selfadjointView<Eigen::Upper>();
	do {
		Ri = R;
		R += Hinv * R;
	} while ((Ri - R).lpNorm<1>() > 1e-13);
	R.normalize();
	return Quaterniond(R[0], R[1], R[2], R[3]);
}
