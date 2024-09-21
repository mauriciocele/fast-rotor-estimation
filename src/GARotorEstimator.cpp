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
	a = H4 * e1 + H5 * e2 + H6 * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	x1 = (a - (lambda * e1));
	x2 = (b - (lambda * e2));
	x3 = (c - (lambda * e3));
	x4 = (d - (lambda * e4));

	? det = x1^x2^x3^x4;
*/
double characteristic(const array<double, 10>& H, double lambda)
{
	Vector4d x;
	x[0] = H[4] - lambda; // e1
	x[1] = H[7] - lambda; // e2
	x[2] = H[9] - lambda; // e3
	x[3] = H[0] - lambda; // e4
	return ((x[0] * x[1] + (-(H[5] * H[5]))) * x[2] + (-((x[0] * H[8] + (-(H[6] * H[5]))) * H[8])) 
		+ (H[5] * H[8] + (-(H[6] * x[1]))) * H[6]) * x[3] + (-(((x[0] * x[1] + (-(H[5] * H[5]))) * H[3] 
		+ (-((x[0] * H[2] + (-(H[1] * H[5]))) * H[8])) + (H[5] * H[2] + (-(H[1] * x[1]))) * H[6]) * H[3])) 
		+ ((x[0] * H[8] + (-(H[6] * H[5]))) * H[3] + (-((x[0] * H[2] + (-(H[1] * H[5]))) * x[2])) 
		+ (H[6] * H[2] + (-(H[1] * H[8]))) * H[6]) * H[2] + (-(((H[5] * H[8] + (-(H[6] * x[1]))) * H[3] 
		+ (-((H[5] * H[2] + (-(H[1] * x[1]))) * x[2])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[8]) * H[1])); // e1 ^ (e2 ^ (e3 ^ e4))		
}

/*
	a = H4 * e1 + H5 * e2 + H6 * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	x1 = (a - (lambda * e1));
	x2 = (b - (lambda * e2));
	x3 = (c - (lambda * e3));
	x4 = (d - (lambda * e4));

	? ddet = -e1^x2^x3^x4 - x1^e2^x3^x4 - x1^x2^e3^x4 - x1^x2^x3^e4;	
*/
double deriv_characteristic(const array<double, 10>& H, double lambda)
{
	Vector4d x;
	x[0] = H[4] - lambda; // e1
	x[1] = H[7] - lambda; // e2
	x[2] = H[9] - lambda; // e3
	x[3] = H[0] - lambda; // e4
	return (((-x[1]) * x[2] + (-((-H[8]) * H[8]))) * x[3] 
		+ (-(((-x[1]) * H[3] + (-((-H[2]) * H[8]))) * H[3])) + ((-H[8]) * H[3] + (-((-H[2]) * x[2]))) * H[2]) 
		- ((x[0] * x[2] + (-H[6]) * H[6]) * x[3] + (-((x[0] * H[3] + (-H[1]) * H[6]) * H[3])) 
		+ (-(((-H[6]) * H[3] + (-((-H[1]) * x[2]))) * H[1]))) - ((x[0] * x[1] + (-(H[5] * H[5]))) * x[3] 
		+ (-(x[0] * H[2] + (-(H[1] * H[5])))) * H[2] + (-((-(H[5] * H[2] + (-(H[1] * x[1])))) * H[1]))) 
		- ((x[0] * x[1] + (-(H[5] * H[5]))) * x[2] + (-((x[0] * H[8] + (-(H[6] * H[5]))) * H[8])) 
		+ (H[5] * H[8] + (-(H[6] * x[1]))) * H[6]); // e1 ^ (e2 ^ (e3 ^ e4))
}

/*
	a = H4 * e1 + H5 * e2 + H6 * e3 + H1 * e4;
	b = H5 * e1 + H7 * e2 + H8 * e3 + H2 * e4;
	c = H6 * e1 + H8 * e2 + H9 * e3 + H3 * e4;
	d = H1 * e1 + H2 * e2 + H3 * e3 + H0 * e4;

	x1 = (a - (lambda * e1));
	x2 = (b - (lambda * e2));
	x3 = (c - (lambda * e3));
	x4 = (d - (lambda * e4));

	? R0 = (x1^x3^x4 + x1^x2^x4 + x1^x2^x3 + x2^x3^x4) . (e1^e2^e3^e4);
*/
Vector4d hyperplanes_intersection(const array<double, 10>& H, double lambda)
{
	Vector4d R0, x;
	x[0] = H[4] - lambda; // e1
	x[1] = H[7] - lambda; // e2
	x[2] = H[9] - lambda; // e3
	x[3] = H[0] - lambda; // e4
	R0[0] = (H[5] * x[2] + (-(H[6] * H[8]))) * x[3] + (-((H[5] * H[3] + (-(H[1] * H[8]))) * H[3])) + (H[6] * H[3] + (-(H[1] * x[2]))) * H[2] + (H[5] * H[8] + (-(H[6] * x[1]))) * x[3] + (-((H[5] * H[2] + (-(H[1] * x[1]))) * H[3])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[2] + (H[5] * H[8] + (-(H[6] * x[1]))) * H[3] + (-((H[5] * H[2] + (-(H[1] * x[1]))) * x[2])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[8] + (x[1] * x[2] + (-(H[8] * H[8]))) * x[3] + (-((x[1] * H[3] + (-(H[2] * H[8]))) * H[3])) + (H[8] * H[3] + (-(H[2] * x[2]))) * H[2]; // e1
	R0[1] = (-((x[0] * x[2] + (-(H[6] * H[6]))) * x[3] + (-((x[0] * H[3] + (-(H[1] * H[6]))) * H[3])) + (H[6] * H[3] + (-(H[1] * x[2]))) * H[1] + (x[0] * H[8] + (-(H[6] * H[5]))) * x[3] + (-((x[0] * H[2] + (-(H[1] * H[5]))) * H[3])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[1] + (x[0] * H[8] + (-(H[6] * H[5]))) * H[3] + (-((x[0] * H[2] + (-(H[1] * H[5]))) * x[2])) + (H[6] * H[2] + (-(H[1] * H[8]))) * H[6] + (H[5] * x[2] + (-(H[8] * H[6]))) * x[3] + (-((H[5] * H[3] + (-(H[2] * H[6]))) * H[3])) + (H[8] * H[3] + (-(H[2] * x[2]))) * H[1])); // e2
	R0[2] = (x[0] * H[8] + (-(H[5] * H[6]))) * x[3] + (-((x[0] * H[3] + (-(H[1] * H[6]))) * H[2])) + (H[5] * H[3] + (-(H[1] * H[8]))) * H[1] + (x[0] * x[1] + (-(H[5] * H[5]))) * x[3] + (-((x[0] * H[2] + (-(H[1] * H[5]))) * H[2])) + (H[5] * H[2] + (-(H[1] * x[1]))) * H[1] + (x[0] * x[1] + (-(H[5] * H[5]))) * H[3] + (-((x[0] * H[2] + (-(H[1] * H[5]))) * H[8])) + (H[5] * H[2] + (-(H[1] * x[1]))) * H[6] + (H[5] * H[8] + (-(x[1] * H[6]))) * x[3] + (-((H[5] * H[3] + (-(H[2] * H[6]))) * H[2])) + (x[1] * H[3] + (-(H[2] * H[8]))) * H[1]; // e3
	R0[3] = (-((x[0] * H[8] + (-(H[5] * H[6]))) * H[3] + (-((x[0] * x[2] + (-(H[6] * H[6]))) * H[2])) + (H[5] * x[2] + (-(H[6] * H[8]))) * H[1] + (x[0] * x[1] + (-(H[5] * H[5]))) * H[3] + (-((x[0] * H[8] + (-(H[6] * H[5]))) * H[2])) + (H[5] * H[8] + (-(H[6] * x[1]))) * H[1] + (x[0] * x[1] + (-(H[5] * H[5]))) * x[2] + (-((x[0] * H[8] + (-(H[6] * H[5]))) * H[8])) + (H[5] * H[8] + (-(H[6] * x[1]))) * H[6] + (H[5] * H[8] + (-(x[1] * H[6]))) * H[3] + (-((H[5] * x[2] + (-(H[8] * H[6]))) * H[2])) + (x[1] * x[2] + (-(H[8] * H[8]))) * H[1])); // e4

	return R0;
}

Vector4d MultivectorDerivative(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, const Multivector& R) {

	static Multivector e1 = getBasisVector(0), e2 = getBasisVector(1), e3 = getBasisVector(2);
	static Multivector e12 = e1^e2;
	static Multivector e31 = e3^e1;
	static Multivector e23 = e2^e3;
	const size_t N = P.size();
	Multivector derivativeAccum;
	for (size_t j = 0; j < N; ++j) {
		const double wj = w[j];
		const Vector3d& Qj = Q[j];
		const Vector3d& Pj = P[j];
		Multivector qBivector =  Qj.z() * e12 + Qj.y() * e31 + Qj.x() * e23;
		Multivector pBivector =  Pj.z() * e12 + Pj.y() * e31 + Pj.x() * e23;
		//q q ~R + 2 q ~R p + ~R p p
		Multivector derivative = -1.0 * (qBivector * qBivector * R) + 
			-2.0 * (qBivector * R * pBivector) +
			-1.0 * (R * pBivector * pBivector);
		if (j == 0) {
			derivativeAccum = wj * derivative;	
		} else {
			derivativeAccum = derivativeAccum + wj * derivative;
		}
	}
	Vector4d column( scp(e23,derivativeAccum), scp(e31,derivativeAccum), scp(e12,derivativeAccum), scalarPart(derivativeAccum));

	return column;
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
	const size_t N = P.size();

	static Multivector e0 = getBasisVector(0), e1 = getBasisVector(1), e2 = getBasisVector(2), e3 = getBasisVector(3);
	static Multivector e0123 = e0^e1^e2^e3;
	static Multivector e12 = e0^e1;
	static Multivector e31 = e2^e0;
	static Multivector e23 = e1^e2;
	static Multivector one = BasisBlade(1.0);

	Vector4d H_0 = MultivectorDerivative(P, Q, w, one);
	Vector4d H_1 = MultivectorDerivative(P, Q, w, -1.0*e12);
	Vector4d H_2 = MultivectorDerivative(P, Q, w, -1.0*e31);
	Vector4d H_3 = MultivectorDerivative(P, Q, w, -1.0*e23);

	Multivector a = H_3(0) * e0 + H_3(1) * e1 + H_3(2) * e2 + H_3(3) * e3;
	Multivector b = H_2(0) * e0 + H_2(1) * e1 + H_2(2) * e2 + H_2(3) * e3;
	Multivector c = H_1(0) * e0 + H_1(1) * e1 + H_1(2) * e2 + H_1(3) * e3;
	Multivector d = H_0(0) * e0 + H_0(1) * e1 + H_0(2) * e2 + H_0(3) * e3;

	double lambda = H_3(0) + H_2(1) + H_1(2) + H_0(3); // trace of matrix
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

	/**
	 * Multivector derivative is: 
	 *   d_R |q + ~R p R|^2 = d_R (R q + p R)~(R q + p R) 
	 *   d_R |q + ~R p R|^2 = -2 q q ~R - 4 q ~R p - 2 ~R p p
	 * 
	 *   Which implies the following eigen-rotor problem
	 * 
	 *   2 q ~R p  = lambda ~R
	 * 
	 *   where lambda = -(q q + p p)
	 *   
	 *   NOTE: lambda is positive value since square of bivectors p and q is negative
	 *   
	 *   Lambda is the exact eigenvalue when input bivectors Ps and Qs don't have noise
	 *   In general, in presence of noise, Lambda is the best initial guess to the newton iteration.
	 */
	double lambda = S;
	double lambda_prev;
	do {
		lambda_prev = lambda;
		lambda = lambda - characteristic(H, lambda) / deriv_characteristic(H, lambda);
	} while(std::abs(lambda_prev - lambda) > 1e-5);

	Quaterniond R( hyperplanes_intersection(H, lambda) );
	R.normalize();
	return R;
}
