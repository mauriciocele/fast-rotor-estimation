#include "GARotorEstimator.h"
#include "Multivector.h"
#include <array>
#include <iostream>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using std::vector;
using std::array;

/*
	a = H4 * e1 + H5 * e2 + H[6] * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	? det = (a - lambda * e1)^(b - lambda * e2)^(c - lambda * e3)^(d - lambda * e4);
*/
double characteristic(const array<double, 10>& H, double lambda)
{
	return (((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * (H[9] - lambda) 
		+ (-(((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[8])) 
		+ (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[6]) * (H[0] - lambda) 
		+ (-((((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * H[3] + (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * H[8])) 
		+ (H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * H[6]) * H[3])) + (((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[3] 
		+ (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * (H[9] - lambda))) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[6]) * H[2] 
		+ (-(((H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[3] + (-((H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * (H[9] - lambda))) 
		+ (H[6] * H[2] + (-(H[1] * H[8]))) * H[8]) * H[1])); // e1 ^ (e2 ^ (e3 ^ e4))	
}

/*
	a = H4 * e1 + H5 * e2 + H6 * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	? ddet = ((-1 * e1)^(b - lambda * e2)^(c - lambda * e3)^(d - lambda * e4)) +
	((a - lambda * e1)^(-1 * e2)^(c - lambda * e3)^(d - lambda * e4)) +
	((a - lambda * e1)^(b - lambda * e2)^(-1 * e3)^(d - lambda * e4)) +
	((a - lambda * e1)^(b - lambda * e2)^(c - lambda * e3)^(-1 * e4));
*/
double deriv_characteristic(const array<double, 10>& H, double lambda)
{
	return ((-(H[7] - lambda)) * (H[9] - lambda) + (-((-H[8]) * H[8]))) * (H[0] - lambda) 
		+ (-(((-(H[7] - lambda)) * H[3] + (-((-H[2]) * H[8]))) * H[3])) + ((-H[8]) * H[3] 
		+ (-((-H[2]) * (H[9] - lambda)))) * H[2] + ((-(H[4] - lambda)) * (H[9] - lambda) + H[6] * H[6]) * (H[0] - lambda) 
		+ (-(((-(H[4] - lambda)) * H[3] + H[1] * H[6]) * H[3])) + (-((H[6] * H[3] + (-(H[1] * (H[9] - lambda)))) * H[1])) 
		+ (-((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5])))) * (H[0] - lambda) + ((H[4] - lambda) * H[2] 
		+ (-(H[1] * H[5]))) * H[2] + (-((H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * H[1])) + (-(((H[4] - lambda) * (H[7] - lambda) 
		+ (-(H[5] * H[5]))) * (H[9] - lambda) + (-(((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[8])) 
		+ (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[6])); // e1 ^ (e2 ^ (e3 ^ e4))	
}

/*
	a = H4 * e1 + H5 * e2 + H6 * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	? R0 = ((a - lambda * e1)^(c - lambda * e3)^(d - lambda * e4)
	+ (a - lambda * e1)^(b - lambda * e2)^(d - lambda * e4) 
	+ (a - lambda * e1)^(b - lambda * e2)^(c - lambda * e3) 
	+ (b - lambda * e2)^(c - lambda * e3)^(d - lambda * e4)) . (e1^e2^e3^e4);
*/
Vector4d hyperplanes_intersection(const array<double, 10>& H, double lambda)
{
	Vector4d R0;
	R0[0] = (H[5] * (H[9] - lambda) + (-(H[6] * H[8]))) * (H[0] - lambda) + (-((H[5] * H[3] + (-(H[1] * H[8]))) * H[3])) + (H[6] * H[3] + (-(H[1] * (H[9] - lambda)))) * H[2] + (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * (H[0] - lambda) + (-((H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * H[3])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[2] + (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[3] + (-((H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * (H[9] - lambda))) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[8] + ((H[7] - lambda) * (H[9] - lambda) + (-(H[8] * H[8]))) * (H[0] - lambda) + (-(((H[7] - lambda) * H[3] + (-(H[2] * H[8]))) * H[3])) + (H[8] * H[3] + (-(H[2] * (H[9] - lambda)))) * H[2]; // e1
	R0[1] = (-(((H[4] - lambda) * (H[9] - lambda) + (-(H[6] * H[6]))) * (H[0] - lambda) + (-(((H[4] - lambda) * H[3] + (-(H[1] * H[6]))) * H[3])) + (H[6] * H[3] + (-(H[1] * (H[9] - lambda)))) * H[1] + ((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * (H[0] - lambda) + (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * H[3])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[1] + ((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[3] + (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * (H[9] - lambda))) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[6] + (H[5] * (H[9] - lambda) + (-(H[8] * H[6]))) * (H[0] - lambda) + (-((H[5] * H[3] + (-(H[2] * H[6]))) * H[3])) + (H[8] * H[3] + (-(H[2] * (H[9] - lambda)))) * H[1])); // e2
	R0[2] = ((H[4] - lambda) * H[8] + (-(H[5] * H[6]))) * (H[0] - lambda) + (-(((H[4] - lambda) * H[3] + (-(H[1] * H[6]))) * H[2])) + (H[5] * H[3] + (-(H[1] * H[8]))) * H[1] + ((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * (H[0] - lambda) + (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * H[2])) + (H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * H[1] + ((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * H[3] + (-(((H[4] - lambda) * H[2] + (-(H[1] * H[5]))) * H[8])) + (H[5] * H[2] + (-(H[1] * (H[7] - lambda)))) * H[6] + (H[5] * H[8] + (-((H[7] - lambda) * H[6]))) * (H[0] - lambda) + (-((H[5] * H[3] + (-(H[2] * H[6]))) * H[2])) + ((H[7] - lambda) * H[3] + (-(H[2] * H[8]))) * H[1]; // e3
	R0[3] = (-(((H[4] - lambda) * H[8] + (-(H[5] * H[6]))) * H[3] + (-(((H[4] - lambda) * (H[9] - lambda) + (-(H[6] * H[6]))) * H[2])) + (H[5] * (H[9] - lambda) + (-(H[6] * H[8]))) * H[1] + ((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * H[3] + (-(((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[2])) + (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[1] + ((H[4] - lambda) * (H[7] - lambda) + (-(H[5] * H[5]))) * (H[9] - lambda) + (-(((H[4] - lambda) * H[8] + (-(H[6] * H[5]))) * H[8])) + (H[5] * H[8] + (-(H[6] * (H[7] - lambda)))) * H[6] + (H[5] * H[8] + (-((H[7] - lambda) * H[6]))) * H[3] + (-((H[5] * (H[9] - lambda) + (-(H[8] * H[6]))) * H[2])) + ((H[7] - lambda) * (H[9] - lambda) + (-(H[8] * H[8]))) * H[1])); // e4
	return R0;
}

Quaterniond GARotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Matrix3d Sx;
	const size_t N = P.size();
	Sx.setZero();
	double S = 0;
	for (size_t j = 0; j < N; ++j) {
		const double wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S += wj * (Pj.dot(Pj) + Qj.dot(Qj));
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}
	H(3, 3) = S + 2.0 * Sx.trace(); 
	S = S - 2.0 * Sx.trace();
	H(3, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(3, 1) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 2) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(0, 0) = 4.0 * Sx(0, 0) + S;
	H(1, 0) = 2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(2, 0) = 2.0 * (Sx(2, 0) + Sx(0, 2));
	H(1, 1) = 4.0 * Sx(1, 1) + S; 
	H(2, 1) = 2.0 * (Sx(1, 2) + Sx(2, 1));
	H(2, 2) = 4.0 * Sx(2, 2) + S;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	Eigen::SelfAdjointEigenSolver<Matrix4d> eigen(H);
	Vector4d V = eigen.eigenvalues();

	static Multivector e0 = getBasisVector(0), e1 = getBasisVector(1), e2 = getBasisVector(2), e3 = getBasisVector(3), e4 = getBasisVector(4);
	static Multivector e0123 = e0^e1^e2^e3;

	Multivector a = H(0,0) * e0 + H(1,0) * e1 + H(2,0) * e2 + H(3,0) * e3;
	Multivector b = H(0,1) * e0 + H(1,1) * e1 + H(2,1) * e2 + H(3,1) * e3;
	Multivector c = H(0,2) * e0 + H(1,2) * e1 + H(2,2) * e2 + H(3,2) * e3;
	Multivector d = H(0,3) * e0 + H(1,3) * e1 + H(2,3) * e2 + H(3,3) * e3;

	double lambda = H.trace();
	double lambda_prev;
	double det;
	double ddet;
	do {
		lambda_prev = lambda;
		det = scp(e0123, (a - lambda * e0)^(b - lambda * e1)^(c - lambda * e2)^(d - lambda * e3));
		ddet = scp(e0123, 
			((-1 * e0)^(b - lambda * e1)^(c - lambda * e2)^(d - lambda * e3)) +
			((a - lambda * e0)^(-1 * e1)^(c - lambda * e2)^(d - lambda * e3)) +
			((a - lambda * e0)^(b - lambda * e1)^(-1 * e2)^(d - lambda * e3)) +
			((a - lambda * e0)^(b - lambda * e1)^(c - lambda * e2)^(-1 * e3))
		);
		lambda = lambda - det / ddet;
	} while(std::abs(lambda_prev - lambda) > 1e-5);

	//Multivector R0;
	//R0 = (a - lambda * e0 + e4)^(b - lambda * e1 + e4)^(c - lambda * e2 + e4)^(d - lambda * e3 + e4);
	//R0 = (a - lambda * e0 + 1)^(b - lambda * e1 + 1)^(c - lambda * e2 + 1)^(d - lambda * e3 + 1);
	// R0 = 
	//   ((a - lambda * e0)^(b - lambda * e1)^(c - lambda * e2)^(d - lambda * e3))
	// + ((a - lambda * e0)^(c - lambda * e2)^(d - lambda * e3))
	// + ((a - lambda * e0)^(b - lambda * e1)^(d - lambda * e3))
	// + ((a - lambda * e0)^(b - lambda * e1)^(c - lambda * e2))
	// + ((b - lambda * e1)^(c - lambda * e2)^(d - lambda * e3))
	// + ((a - lambda * e0)^(d - lambda * e3))
	// + ((b - lambda * e1)^(d - lambda * e3))
	// + ((a - lambda * e0)^(c - lambda * e2))
	// + ((a - lambda * e0)^(b - lambda * e1))
	// + ((c - lambda * e2)^(d - lambda * e3))
	// + ((b - lambda * e1)^(c - lambda * e2))
	// + (d - lambda * e3)
	// + (c - lambda * e2)
	// + (b - lambda * e1)
	// + (a - lambda * e0)
	// + 1;
	Multivector R0 = ((a - lambda * e0)^(c - lambda * e2)^(d - lambda * e3))
		+ ((a - lambda * e0)^(b - lambda * e1)^(d - lambda * e3))
		+ ((a - lambda * e0)^(b - lambda * e1)^(c - lambda * e2))
		+ ((b - lambda * e1)^(c - lambda * e2)^(d - lambda * e3));
			
	Multivector R3 = dual(R0, 4);
	std::cout << "R3: " << R3.toString() << std::endl;
	Vector4d RR( scp(e0,R3), scp(e1,R3), scp(e2,R3), scp(e3,R3) );
	Quaterniond QQ( RR );
	QQ.normalize();
	return QQ;
}

Quaterniond LARotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	array<double, 10> H;
	Matrix3d Sx;
	Vector3d S;
	double wj;
	const size_t N = P.size();
	Sx.setZero();
	S.setZero();
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S.noalias() += wj * (Pj + Qj);
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}
	wj = S.dot(S);
 	H[0] = wj + 2.0 * Sx.trace();       // (3,3)
	wj = wj - 2.0 * Sx.trace();
	H[1] = 2.0 * (Sx(1, 2) - Sx(2, 1)); // (3,0)
	H[2] = 2.0 * (Sx(2, 0) - Sx(0, 2)); // (3,1)
	H[3] = 2.0 * (Sx(0, 1) - Sx(1, 0)); // (3,2)
	H[4] = 4.0 * Sx(0, 0) + wj;         // (0,0)
	H[5] = 2.0 * (Sx(0, 1) + Sx(1, 0)); // (1,0) 
	H[6] = 2.0 * (Sx(2, 0) + Sx(0, 2)); // (2,0)
	H[7] = 4.0 * Sx(1, 1) + wj;         // (1,1)
	H[8] = 2.0 * (Sx(1, 2) + Sx(2, 1)); // (2,1)
	H[9] = 4.0 * Sx(2, 2) + wj;         // (2,2)

	double lambda = H[0]+H[4]+H[7]+H[9];
	double lambda_prev;
	do {
		lambda_prev = lambda;
		lambda = lambda - characteristic(H, lambda) / deriv_characteristic(H, lambda);
	} while(std::abs(lambda_prev - lambda) > 1e-5);

	Quaterniond R( hyperplanes_intersection(H, lambda) );
	R.normalize();
	return R;
}


/*

max | p + R* q R |^2

p + R* q R = 0

p R* + R* q = 0

p (w - L) + (w - L) q = 0

(Remember: inner product of bivectors is negative and commutator anticommute)
w (p + q) + L . (p + q) + (q - p) x L = 0

| 0     s^T  | | w | = | s^T L       |  
| s    [d]_x | | L |   | w s + d x L |

| 0     s^T  | | 0    s^T  | = | s^T s    s^T [d]_x       |
| s   -[d]_x | | s   [d]_x |   |-[d]_x s  s s^T - [d]^2_x |

| 0     s^T  | | 0    s^T  | = | s^T s    s x d         | = | s^T s    s x d                 |
| s   -[d]_x | | s   [d]_x |   | s x d  s s^T - [d]^2_x |   | s x d  s s^T - d d^t + d^T d I |

*/

Quaterniond GAFastRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Matrix3d Sx;
	Vector3d S;
	double wj;
	const size_t N = P.size();
	const size_t MAX_STEPS = 12;
	Sx.setZero();
	S.setZero();
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S.noalias() += wj * (Pj + Qj);
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}
	wj = S.dot(S);
	H(3, 3) = wj + 2.0 * Sx.trace(); 
	wj = wj - 2.0 * Sx.trace();
	H(3, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(3, 1) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 2) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(0, 0) = 4.0 * Sx(0, 0) + wj;
	H(1, 0) = 2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(2, 0) = 2.0 * (Sx(2, 0) + Sx(0, 2));
	H(1, 1) = 4.0 * Sx(1, 1) + wj; 
	H(2, 1) = 2.0 * (Sx(1, 2) + Sx(2, 1));
	H(2, 2) = 4.0 * Sx(2, 2) + wj;
	H.selfadjointView<Eigen::Lower>().evalTo(H);	
	for (size_t j = 0; j < MAX_STEPS; ++j) {
		H *= H;
		H *= 1.0 / H.trace();
	}
	return Quaterniond(H.col(0).normalized());
}

Quaterniond GAFastRotorEstimatorAVX(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix4d H;
	Matrix3d Sx;
	Vector3d S;
	double wj;
	const size_t N = P.size();
	const size_t MAX_STEPS = 12;
	const double* DP = nullptr;

	Sx.setZero();
	S.setZero();
	for (size_t j = 0; j < N; ++j) {
		const double wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S.noalias() += wj * (Pj + Qj);
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}
	wj = S.dot(S);
	H(3, 3) = wj + 2.0 * Sx.trace(); 
	wj = wj - 2.0 * Sx.trace();
	H(3, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(3, 1) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 2) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(0, 0) = 4.0 * Sx(0, 0) + wj;
	H(1, 0) = 2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(2, 0) = 2.0 * (Sx(2, 0) + Sx(0, 2));
	H(1, 1) = 4.0 * Sx(1, 1) + wj; 
	H(2, 1) = 2.0 * (Sx(1, 2) + Sx(2, 1));
	H(2, 2) = 4.0 * Sx(2, 2) + wj;
	for (size_t j = 0; j < MAX_STEPS; ++j) {
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
	return Quaterniond(H.col(0).normalized());
}

Quaterniond GAFastRotorEstimatorAprox(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const double epsilon, size_t steps)
{
	Matrix4d H;
	Vector4d R;
	Matrix3d Sx;
	Vector3d S;
	double wj;
	const size_t N = P.size();
	Sx.setZero();
	S.setZero();
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S.noalias() += wj * (Pj + Qj);
		Sx.noalias() += (wj * Qj) * Pj.transpose();
	}
	wj = S.dot(S);
	H(3, 3) = wj - 2.0 * Sx.trace(); 
	wj = wj + 2.0 * Sx.trace();
	H(3, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(3, 1) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 2) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(0, 0) = -4.0 * Sx(0, 0) + wj;
	H(1, 0) = -2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(2, 0) = -2.0 * (Sx(2, 0) + Sx(0, 2));
	H(1, 1) = -4.0 * Sx(1, 1) + wj; 
	H(2, 1) = -2.0 * (Sx(1, 2) + Sx(2, 1));
	H(2, 2) = -4.0 * Sx(2, 2) + wj;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	H = H.inverse().eval();
	H *= H;
	H *= 1.0 / H.trace();
	H *= H;
	H *= 1.0 / H.trace();
	R = H.col(0);
	while (steps--) {
		R = H * R;
	}
	R.normalize();
	return Quaterniond(R);
}

Quaterniond GAFastRotorEstimatorIncr(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const Quaterniond& Qprev)
{
	Matrix4d H;
	Matrix3d Sx;
	Vector3d S;
	double wj;
	const size_t N = P.size();
	Sx.setZero();
	S.setZero();
	for (size_t j = 0; j < N; ++j)
	{
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S.noalias() += wj * (Pj + Qj);
		Sx.noalias() += (wj * Qj) * Pj.transpose();
	}
	wj = S.dot(S);
	H(3, 3) = wj - 2.0 * Sx.trace(); 
	wj = wj + 2.0 * Sx.trace();
	H(3, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(3, 1) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 2) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(0, 0) = -4.0 * Sx(0, 0) + wj;
	H(1, 0) = -2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(2, 0) = -2.0 * (Sx(2, 0) + Sx(0, 2));
	H(1, 1) = -4.0 * Sx(1, 1) + wj; 
	H(2, 1) = -2.0 * (Sx(1, 2) + Sx(2, 1));
	H(2, 2) = -4.0 * Sx(2, 2) + wj;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	return Quaterniond((H.inverse() * Qprev.coeffs()).normalized());
}

Quaterniond GANewtonRotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	const double epsilon = 5e-2;
	const size_t N = P.size();
	Matrix4d H, Hinv;
	Matrix3d Sx;
	Vector4d R_i, R(1, epsilon, epsilon, epsilon);
	double wj;
	double S0, S1, S3;
	Sx.setZero();
	S0 = S1 = 0;
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S1 += wj * Qj.dot(Qj);
		S0 += wj * Pj.dot(Pj);
		Sx.noalias() += (wj * Qj) * Pj.transpose();
	}
	S3 = (S0 + S1);
	H(0, 0) = S3 - 2.0 * Sx.trace(); 
	S3 = S3 + 2.0 * Sx.trace();
	H(1, 0) = 2.0 * (Sx(1, 2) - Sx(2, 1));
	H(2, 0) = 2.0 * (Sx(2, 0) - Sx(0, 2));
	H(3, 0) = 2.0 * (Sx(0, 1) - Sx(1, 0));
	H(1, 1) = -4.0 * Sx(0, 0) + S3;
	H(2, 1) = -2.0 * (Sx(0, 1) + Sx(1, 0)); 
	H(3, 1) = -2.0 * (Sx(2, 0) + Sx(0, 2));
	H(2, 2) = -4.0 * Sx(1, 1) + S3; 
	H(3, 2) = -2.0 * (Sx(1, 2) + Sx(2, 1));
	H(3, 3) = -4.0 * Sx(2, 2) + S3;
	H.selfadjointView<Eigen::Lower>().evalTo(H);
	Hinv = H;
	Hinv(0, 0) += epsilon; Hinv(1, 1) += epsilon; Hinv(2, 2) += epsilon; Hinv(3, 3) += epsilon;
	Hinv = Hinv.inverse() * H;
	do {
		R_i = R;
		R = R - Hinv * R;
	} while ((R_i - R).lpNorm<1>() > 1e-6);
	R.normalize();
	return Quaterniond(R[0], R[1], R[2], R[3]);
}
