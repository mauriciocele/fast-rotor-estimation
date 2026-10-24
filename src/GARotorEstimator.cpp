#include "GARotorEstimator.h"
#include "Multivector.h"
#include <array>

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

	are the columns x1..x4 of M - lambda I, M being the 4x4 matrix of LARotorEstimator.

	Blades of the 4D algebra with basis e1..e4: a bivector holds its coefficients on
	e12, e13, e14, e23, e24, e34 and a trivector on e123, e124, e134, e234.
*/
struct Bivector4 { double e12, e13, e14, e23, e24, e34; };
struct Trivector4 { double e123, e124, e134, e234; };

/* a ^ b */
static inline Bivector4 wedge(const Vector4d& a, const Vector4d& b)
{
	return { a[0] * b[1] - a[1] * b[0], a[0] * b[2] - a[2] * b[0], a[0] * b[3] - a[3] * b[0],
	         a[1] * b[2] - a[2] * b[1], a[1] * b[3] - a[3] * b[1], a[2] * b[3] - a[3] * b[2] };
}

/* B ^ c (= c ^ B) */
static inline Trivector4 wedge(const Bivector4& B, const Vector4d& c)
{
	return { B.e12 * c[2] - B.e13 * c[1] + B.e23 * c[0],
	         B.e12 * c[3] - B.e14 * c[1] + B.e24 * c[0],
	         B.e13 * c[3] - B.e14 * c[2] + B.e34 * c[0],
	         B.e23 * c[3] - B.e24 * c[2] + B.e34 * c[1] };
}

/* (A ^ C) . (e1^e2^e3^e4)^-1, the scalar of the 4-vector A ^ C */
static inline double wedge(const Bivector4& A, const Bivector4& C)
{
	return A.e12 * C.e34 - A.e13 * C.e24 + A.e14 * C.e23 + A.e23 * C.e14 - A.e24 * C.e13 + A.e34 * C.e12;
}

/* m ? a : b, component by component, so that the compiler can use conditional selects */
static inline Vector4d select(bool m, const Vector4d& a, const Vector4d& b)
{
	return Vector4d(m ? a[0] : b[0], m ? a[1] : b[1], m ? a[2] : b[2], m ? a[3] : b[3]);
}

static inline Bivector4 select(bool m, const Bivector4& a, const Bivector4& b)
{
	return { m ? a.e12 : b.e12, m ? a.e13 : b.e13, m ? a.e14 : b.e14,
	         m ? a.e23 : b.e23, m ? a.e24 : b.e24, m ? a.e34 : b.e34 };
}

/* T . (e1^e2^e3^e4)^-1, the vector dual to T */
static inline Vector4d dual(const Trivector4& T)
{
	return Vector4d(T.e234, -T.e134, T.e124, -T.e123);
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

	// The dual of each triple wedge is one column of adj(M - lambda I), parallel to the
	// eigenrotor and scaled by its component along the missing column. Summing them
	// cancels for a whole family of rotations (see AdjugateNullVector.h), so keep the one
	// with the largest component along its own missing column.
	const Multivector x[4] = { a - lambda * e0, b - lambda * e1, c - lambda * e2, d - lambda * e3 };
	const Multivector* basis[4] = { &e0, &e1, &e2, &e3 };
	Multivector R3;
	double bestScale = -1.0;
	for (int k = 0; k < 4; ++k) {
		const Multivector& u = x[(k + 1) % 4];
		const Multivector& v = x[(k + 2) % 4];
		const Multivector& t = x[(k + 3) % 4];
		Multivector candidate = dual(u^v^t, 4);
		const double scale = std::abs(scp(*basis[k], candidate));
		if (scale > bestScale) {
			bestScale = scale;
			R3 = candidate;
		}
	}
	Vector4d RR( scp(e0,R3), scp(e1,R3), scp(e2,R3), scp(e3,R3) );
	Quaterniond QQ( RR );
	QQ.normalize();
	return QQ;
}

Quaterniond LARotorEstimator(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	// S = sum w (|p|^2 + |q|^2) and Sx = sum w p q^T
	Matrix3d Sx;
	Sx.setZero();
	double S = 0;
	const size_t N = P.size();
	for (size_t j = 0; j < N; ++j) {
		const double wj = w[j];
		const Vector3d& Pj = P[j];
		const Vector3d& Qj = Q[j];
		S += wj * (Pj.squaredNorm() + Qj.squaredNorm());
		Sx.noalias() += (wj * Pj) * Qj.transpose();
	}

	/*
		M = K + (S / 2) I, where K is Davenport's matrix of Sx, with columns k1..k4 (basis
		e1..e4). The shift does not change the eigenvectors, so we work with K and
		mu = lambda - S / 2.
	*/
	const double tr = Sx.trace();
	const Vector4d k1(2.0 * Sx(0, 0) - tr, Sx(0, 1) + Sx(1, 0), Sx(2, 0) + Sx(0, 2), Sx(1, 2) - Sx(2, 1));
	const Vector4d k2(Sx(0, 1) + Sx(1, 0), 2.0 * Sx(1, 1) - tr, Sx(1, 2) + Sx(2, 1), Sx(2, 0) - Sx(0, 2));
	const Vector4d k3(Sx(2, 0) + Sx(0, 2), Sx(1, 2) + Sx(2, 1), 2.0 * Sx(2, 2) - tr, Sx(0, 1) - Sx(1, 0));
	const Vector4d k4(Sx(1, 2) - Sx(2, 1), Sx(2, 0) - Sx(0, 2), Sx(0, 1) - Sx(1, 0), tr);

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
	/*
		The characteristic outer product (x1 ^ x2 ^ x3 ^ x4) . (e1^e2^e3^e4)^-1 = det(K - mu I)
		is the quartic mu^4 + a mu^2 + b mu + c, since K is traceless: a = -2 |Sx|^2,
		b = -8 det(Sx) and c = det K = (k1 ^ k2) ^ (k3 ^ k4). With the coefficients computed
		once, each Newton step is a few operations. lambda_0 = S, i.e. mu_0 = S / 2.
	*/
	const double a = -2.0 * Sx.squaredNorm();
	const double b = -8.0 * Sx.determinant();
	const double c = wedge(wedge(k1, k2), wedge(k3, k4));
	double mu = 0.5 * S;
	double mu_prev;
	do {
		mu_prev = mu;
		const double mu2 = mu * mu;
		mu -= ((mu2 + a) * mu2 + b * mu + c) / ((4.0 * mu2 + 2.0 * a) * mu + b);
	} while(std::abs(mu_prev - mu) > 1e-5);

	/*
		Eigenrotor, as in GARotorEstimator: the dual of each triple wedge of the columns
		x1..x4 of K - mu I (= M - lambda I) is parallel to the eigenrotor, scaled by its component along
		the missing column. Keep the one where that component is largest (summing them
		cancels for a family of rotations).

		? A = x1 ^ x2;  C = x3 ^ x4;
		? T1 = x2 ^ C;  T2 = x1 ^ C;  T3 = A ^ x4;  T4 = A ^ x3;   (Tk misses xk)
		? R = dual(Tk) for the k with the largest |dual(Tk) . ek|
	*/
	const Vector4d x1 = k1 - mu * Vector4d::UnitX();
	const Vector4d x2 = k2 - mu * Vector4d::UnitY();
	const Vector4d x3 = k3 - mu * Vector4d::UnitZ();
	const Vector4d x4 = k4 - mu * Vector4d::UnitW();
	const Bivector4 A = wedge(x1, x2);
	const Bivector4 C = wedge(x3, x4);

	// dual(Tk) . ek is the coefficient of Tk on the blade without ek
	const double own[4] = {
		C.e23 * x2[3] - C.e24 * x2[2] + C.e34 * x2[1],   // (x2 ^ C) . e234
		C.e13 * x1[3] - C.e14 * x1[2] + C.e34 * x1[0],   // (x1 ^ C) . e134
		A.e12 * x4[3] - A.e14 * x4[1] + A.e24 * x4[0],   // (A ^ x4) . e124
		A.e12 * x3[2] - A.e13 * x3[1] + A.e23 * x3[0] }; // (A ^ x3) . e123
	// Branch-free selection: the largest meet is T = B ^ v with (B, v) = (C, x2), (C, x1),
	// (A, x4) or (A, x3); select B and v, then compute a single trivector. The winner depends
	// on the rotation, so a branch here would often be mispredicted.
	const bool m1 = std::abs(own[1]) > std::abs(own[0]);
	double best = m1 ? std::abs(own[1]) : std::abs(own[0]);
	Bivector4 Bs = C;
	Vector4d v = select(m1, x1, x2);
	const bool m2 = std::abs(own[2]) > best;
	best = m2 ? std::abs(own[2]) : best;
	Bs = select(m2, A, Bs);
	v = select(m2, x4, v);
	const bool m3 = std::abs(own[3]) > best;
	Bs = select(m3, A, Bs);
	v = select(m3, x3, v);
	const Trivector4 T = wedge(Bs, v);

	// basis order e1, e2, e3, e4, which Quaterniond(Vector4d) reads as x, y, z, w
	Quaterniond R( dual(T) );
	R.normalize();
	return R;
}
