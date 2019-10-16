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
	double wj;
	const size_t N = P.size();
	Sx.setZero();
	double S = 0;
	for (size_t j = 0; j < N; ++j) {
		wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		S += wj * Pj.dot(Pj);
		S += wj * Qj.dot(Qj);
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}

	wj = 0.5 * S;
 	H[0] = wj + Sx.trace();       // (3,3)
	wj = wj - Sx.trace();
	H[1] = (Sx(1, 2) - Sx(2, 1)); // (3,0)
	H[2] = (Sx(2, 0) - Sx(0, 2)); // (3,1)
	H[3] = (Sx(0, 1) - Sx(1, 0)); // (3,2)
	H[4] = 2.0 * Sx(0, 0) + wj;   // (0,0)
	H[5] = (Sx(0, 1) + Sx(1, 0)); // (1,0) 
	H[6] = (Sx(2, 0) + Sx(0, 2)); // (2,0)
	H[7] = 2.0 * Sx(1, 1) + wj;   // (1,1)
	H[8] = (Sx(1, 2) + Sx(2, 1)); // (2,1)
	H[9] = 2.0 * Sx(2, 2) + wj;   // (2,2)

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
