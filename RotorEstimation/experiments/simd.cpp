// Batched SIMD comparison of the proposed method, QUEST, FOAM and FLAE Newton: each SIMD lane
// solves an independent problem. All methods share the accumulation code, the Newton start (S/2)
// and the stopping rule; QUEST uses the method of sequential rotations near 180 degrees, and
// FLAE is run as published (q_z = -1) and with the adjugate fallback.
// The same templates instantiated with T = double give the scalar versions.
// Requires Clang (ext_vector_type). Writes, into the directory given as argument (../data):
//   simd_timing.csv    ns per problem for each method and number of vectors
//   simd_accuracy.csv  worst angle error for rotations by pi - delta
//   simd_accuracy_flae.csv  worst angle error for axes at angle delta from the xy-plane
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include "GARotorEstimator.h"
#include "Quest.h"
#include "FLAE.h"

typedef double V4 __attribute__((ext_vector_type(4)));
typedef long M4 __attribute__((ext_vector_type(4)));   // the type of V4 comparisons

// lane helpers: select, absolute value, "any lane true"
static inline double sel(bool m, double a, double b) { return m ? a : b; }
static inline V4 sel(M4 m, V4 a, V4 b) { return (V4)(((M4)a & m) | ((M4)b & ~m)); }
static inline double vabs(double x) { return std::abs(x); }
static inline V4 vabs(V4 x) { return __builtin_elementwise_abs(x); }
static inline bool notm(bool m) { return !m; }
static inline M4 notm(M4 m) { return ~m; }
static inline bool any(bool m) { return m; }
static inline bool any(M4 m) { return m[0] | m[1] | m[2] | m[3]; }
static inline double vsqrt(double x) { return std::sqrt(x); }
static inline V4 vsqrt(V4 x) { return __builtin_elementwise_sqrt(x); }

template <class T> struct Problem { const T *px, *py, *pz, *qx, *qy, *qz, *w; int n; };
template <class T> struct Quat { T w, x, y, z; };
template <class T> struct Acc { T S, b[3][3]; };   // b[i][k] = sum w q_i p_k

template <class T>
static inline Acc<T> accumulate(const Problem<T>& P)
{
	Acc<T> A;
	A.S = 0;
	for (auto& r : A.b) for (auto& v : r) v = 0;
	for (int j = 0; j < P.n; ++j) {
		const T w = P.w[j];
		const T p[3] = { P.px[j], P.py[j], P.pz[j] }, q[3] = { P.qx[j], P.qy[j], P.qz[j] };
		A.S += w * (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
		const T wp[3] = { w * p[0], w * p[1], w * p[2] };
		for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) A.b[i][k] += q[i] * wp[k];
	}
	return A;
}

template <class T>
static inline T det3(const T m[3][3])
{
	return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
	     + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

template <class T>
static inline Quat<T> normalized(Quat<T> q)
{
	const T s = 1.0 / vsqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
	return { q.w * s, q.x * s, q.y * s, q.z * s };
}

// ---------------------------------------------------------------- proposed (Algorithm 1)
template <class T> struct Biv { T e01, e02, e03, e12, e13, e23; };
template <class T> struct Tri { T e012, e013, e023, e123; };
template <class T> static inline Biv<T> wedge(const T a[4], const T b[4])
{
	return { a[0] * b[1] - a[1] * b[0], a[0] * b[2] - a[2] * b[0], a[0] * b[3] - a[3] * b[0],
	         a[1] * b[2] - a[2] * b[1], a[1] * b[3] - a[3] * b[1], a[2] * b[3] - a[3] * b[2] };
}
template <class T> static inline Tri<T> wedge(const Biv<T>& B, const T c[4])
{
	return { B.e01 * c[2] - B.e02 * c[1] + B.e12 * c[0], B.e01 * c[3] - B.e03 * c[1] + B.e13 * c[0],
	         B.e02 * c[3] - B.e03 * c[2] + B.e23 * c[0], B.e12 * c[3] - B.e13 * c[2] + B.e23 * c[1] };
}
template <class T> static inline T wedge(const Biv<T>& A, const Biv<T>& C)
{
	return A.e01 * C.e23 - A.e02 * C.e13 + A.e03 * C.e12 + A.e12 * C.e03 - A.e13 * C.e02 + A.e23 * C.e01;
}

template <class T>
static Quat<T> Proposed(const Problem<T>& P)
{
	const Acc<T> A = accumulate(P);
	const auto& b = A.b;
	const T tr = b[0][0] + b[1][1] + b[2][2];
	const T z[3] = { b[2][1] - b[1][2], b[0][2] - b[2][0], b[1][0] - b[0][1] };
	T L[3][3];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) L[i][k] = b[i][k] + b[k][i];
	for (int i = 0; i < 3; ++i) L[i][i] -= tr;
	T k[4][4] = { { tr, z[0], z[1], z[2] } };
	for (int i = 1; i < 4; ++i) { k[i][0] = z[i - 1]; for (int r = 0; r < 3; ++r) k[i][r + 1] = L[r][i - 1]; }
	T sq = 0;
	for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) sq += b[i][j] * b[i][j];
	const T a = -2.0 * sq, bb = -8.0 * det3(b), c = wedge(wedge(k[0], k[1]), wedge(k[2], k[3]));
	T mu = 0.5 * A.S, prev;
	do {
		prev = mu;
		const T mu2 = mu * mu;
		mu -= ((mu2 + a) * mu2 + bb * mu + c) / ((4.0 * mu2 + 2.0 * a) * mu + bb);
	} while (any(vabs(prev - mu) > 1e-5));
	for (int i = 0; i < 4; ++i) k[i][i] -= mu;
	const Biv<T> Ab = wedge(k[0], k[1]), Cb = wedge(k[2], k[3]);
	const T own[4] = {
		vabs(Cb.e12 * k[1][3] - Cb.e13 * k[1][2] + Cb.e23 * k[1][1]),   // (x1 ^ C)_123
		vabs(Cb.e02 * k[0][3] - Cb.e03 * k[0][2] + Cb.e23 * k[0][0]),   // (x0 ^ C)_023
		vabs(Ab.e01 * k[3][3] - Ab.e03 * k[3][1] + Ab.e13 * k[3][0]),   // (A ^ x3)_013
		vabs(Ab.e01 * k[2][2] - Ab.e02 * k[2][1] + Ab.e12 * k[2][0]) }; // (A ^ x2)_012
	// branch-free selection of the trivector T_k = B ^ v: choose B and v, then one wedge
	auto m1 = own[1] > own[0];
	T best = sel(m1, own[1], own[0]);
	Biv<T> Bs = Cb;   // T_0 and T_1 both use C
	T v[4]; for (int i = 0; i < 4; ++i) v[i] = sel(m1, k[0][i], k[1][i]);
	for (int kk = 2; kk < 4; ++kk) {
		auto m = own[kk] > best;
		best = sel(m, own[kk], best);
		Bs = { sel(m, Ab.e01, Bs.e01), sel(m, Ab.e02, Bs.e02), sel(m, Ab.e03, Bs.e03), sel(m, Ab.e12, Bs.e12), sel(m, Ab.e13, Bs.e13), sel(m, Ab.e23, Bs.e23) };
		const int vi = kk == 2 ? 3 : 2;
		for (int i = 0; i < 4; ++i) v[i] = sel(m, k[vi][i], v[i]);
	}
	const Tri<T> t = wedge(Bs, v);
	return normalized(Quat<T>{ t.e123, -t.e023, t.e013, -t.e012 });
}

// ---------------------------------------------------------------- QUEST (Quest.cpp)
// Q[i][k] = sum w p_i q_k is the transpose of the accumulated b.
template <class T>
static inline void questVector(const T Q[3][3], T lambda, T& gamma, T x[3])
{
	const T s = Q[0][0] + Q[1][1] + Q[2][2];
	T S[3][3];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) S[i][k] = Q[i][k] + Q[k][i];
	const T Z[3] = { Q[1][2] - Q[2][1], Q[2][0] - Q[0][2], Q[0][1] - Q[1][0] };
	const T kappa = S[1][1] * S[2][2] - S[1][2] * S[2][1] + S[0][0] * S[2][2] - S[0][2] * S[2][0] + S[0][0] * S[1][1] - S[0][1] * S[1][0];
	const T alpha = lambda * lambda - (s * s - kappa), beta = lambda - s;
	gamma = alpha * (lambda + s) - det3(S);
	T SZ[3], SSZ[3];
	for (int i = 0; i < 3; ++i) SZ[i] = S[i][0] * Z[0] + S[i][1] * Z[1] + S[i][2] * Z[2];
	for (int i = 0; i < 3; ++i) SSZ[i] = S[i][0] * SZ[0] + S[i][1] * SZ[1] + S[i][2] * SZ[2];
	for (int i = 0; i < 3; ++i) x[i] = beta * SZ[i] + SSZ[i] + alpha * Z[i];
}

template <class T>
static Quat<T> Quest4(const Problem<T>& P, double threshold)
{
	const Acc<T> A = accumulate(P);
	T Q[3][3];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) Q[i][k] = A.b[k][i];
	const T s = Q[0][0] + Q[1][1] + Q[2][2];
	T S[3][3];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) S[i][k] = Q[i][k] + Q[k][i];
	const T Z[3] = { Q[1][2] - Q[2][1], Q[2][0] - Q[0][2], Q[0][1] - Q[1][0] };
	const T kappa = S[1][1] * S[2][2] - S[1][2] * S[2][1] + S[0][0] * S[2][2] - S[0][2] * S[2][0] + S[0][0] * S[1][1] - S[0][1] * S[1][0];
	const T s2 = s * s, Sdet = det3(S);
	T SZ[3], SSZ[3];
	for (int i = 0; i < 3; ++i) SZ[i] = S[i][0] * Z[0] + S[i][1] * Z[1] + S[i][2] * Z[2];
	for (int i = 0; i < 3; ++i) SSZ[i] = S[i][0] * SZ[0] + S[i][1] * SZ[1] + S[i][2] * SZ[2];
	const T a = s2 - kappa, b = s2 + (Z[0] * Z[0] + Z[1] * Z[1] + Z[2] * Z[2]);
	const T c = Sdet + (Z[0] * SZ[0] + Z[1] * SZ[1] + Z[2] * SZ[2]), d = Z[0] * SSZ[0] + Z[1] * SSZ[1] + Z[2] * SSZ[2];
	T lambda = 0.5 * A.S, prev;
	do {
		prev = lambda;
		const T l2 = lambda * lambda, l3 = l2 * lambda;
		lambda -= (l3 * lambda - (a + b) * l2 - c * lambda + (a * b + c * s - d)) / (4.0 * l3 - 2.0 * (a + b) * lambda - c);
	} while (any(vabs(prev - lambda) > 1e-5));
	const T alpha = lambda * lambda - a, beta = lambda - s;
	T gamma = alpha * (lambda + s) - Sdet;
	T x[3];
	for (int i = 0; i < 3; ++i) x[i] = beta * SZ[i] + SSZ[i] + alpha * Z[i];
	Quat<T> q = { gamma, x[0], x[1], x[2] };
	// method of sequential rotations, when q_w^2 = gamma / f'(lambda) is small in any lane
	const T df = 4.0 * lambda * lambda * lambda - 2.0 * (a + b) * lambda - c;
	auto need = vabs(gamma) < threshold * vabs(df);
	if (any(need)) {
		T bestG = vabs(gamma);
		for (int k = 0; k < 3; ++k) {
			T Qk[3][3];
			for (int r = 0; r < 3; ++r) for (int col = 0; col < 3; ++col) Qk[r][col] = (r == k) ? Q[r][col] : -Q[r][col];
			T gk, xk[3];
			questVector(Qk, lambda, gk, xk);
			auto m = need & (vabs(gk) > bestG);
			bestG = sel(m, vabs(gk), bestG);
			// q = (gk, xk) * (0, e_k)
			const T e[3] = { T(k == 0), T(k == 1), T(k == 2) };
			const Quat<T> qk = { -(xk[0] * e[0] + xk[1] * e[1] + xk[2] * e[2]),
			                     gk * e[0] + xk[1] * e[2] - xk[2] * e[1],
			                     gk * e[1] + xk[2] * e[0] - xk[0] * e[2],
			                     gk * e[2] + xk[0] * e[1] - xk[1] * e[0] };
			q = { sel(m, qk.w, q.w), sel(m, qk.x, q.x), sel(m, qk.y, q.y), sel(m, qk.z, q.z) };
		}
	}
	return normalized(q);
}

// ---------------------------------------------------------------- FOAM (Quest.cpp)
// Returns the rotation matrix, as the library does. B = sum w q p^T is the accumulated b.
template <class T> struct Mat { T m[3][3]; };

template <class T>
static Mat<T> Foam4(const Problem<T>& P)
{
	const Acc<T> A = accumulate(P);
	const auto& B = A.b;
	T cof[3][3];   // cofactor matrix = adj(B)^T
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) {
		const int i1 = (i + 1) % 3, i2 = (i + 2) % 3, k1 = (k + 1) % 3, k2 = (k + 2) % 3;
		cof[i][k] = B[i1][k1] * B[i2][k2] - B[i1][k2] * B[i2][k1];
	}
	T Bnorm = 0, Badj = 0;
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) { Bnorm += B[i][k] * B[i][k]; Badj += cof[i][k] * cof[i][k]; }
	Badj = 4.0 * Badj;
	const T Bdet = 8.0 * det3(B);
	T lambda = 0.5 * A.S, prev;
	do {
		prev = lambda;
		const T kappa = lambda * lambda - Bnorm;
		lambda -= (kappa * kappa - lambda * Bdet - Badj) / (4.0 * lambda * kappa - Bdet);
	} while (any(vabs(prev - lambda) > 1e-5));
	const T kappa = 0.5 * (lambda * lambda - Bnorm);
	T BBt[3][3], BBtB[3][3];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) BBt[i][k] = B[i][0] * B[k][0] + B[i][1] * B[k][1] + B[i][2] * B[k][2];
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) BBtB[i][k] = BBt[i][0] * B[0][k] + BBt[i][1] * B[1][k] + BBt[i][2] * B[2][k];
	const T inv = 1.0 / (lambda * kappa - Bdet / 8.0);
	Mat<T> R;
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) R.m[i][k] = ((kappa + Bnorm) * B[i][k] + lambda * cof[i][k] - BBtB[i][k]) * inv;
	return R;
}

// FOAM followed by the conversion to a quaternion: the columns of 4 q q^T are linear in R,
// and the one with the largest diagonal entry (>= 1) is chosen branch-free, as Shepperd does.
template <class T>
static Quat<T> FoamQuat4(const Problem<T>& P)
{
	const Mat<T> M = Foam4(P);
	const auto& R = M.m;
	const T col[4][4] = {
		{ 1.0 + R[0][0] + R[1][1] + R[2][2], R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1] },
		{ R[2][1] - R[1][2], 1.0 + R[0][0] - R[1][1] - R[2][2], R[0][1] + R[1][0], R[0][2] + R[2][0] },
		{ R[0][2] - R[2][0], R[0][1] + R[1][0], 1.0 - R[0][0] + R[1][1] - R[2][2], R[1][2] + R[2][1] },
		{ R[1][0] - R[0][1], R[0][2] + R[2][0], R[1][2] + R[2][1], 1.0 - R[0][0] - R[1][1] + R[2][2] } };
	T best = col[0][0], v[4] = { col[0][0], col[0][1], col[0][2], col[0][3] };
	for (int k = 1; k < 4; ++k) {
		auto m = col[k][k] > best;
		best = sel(m, col[k][k], best);
		for (int i = 0; i < 4; ++i) v[i] = sel(m, col[k][i], v[i]);
	}
	return normalized(Quat<T>{ v[0], v[1], v[2], v[3] });
}

// ---------------------------------------------------------------- FLAE Newton (FLAE.cpp)
// The 4x4 matrix W in the order (w, x, y, z), Newton on its characteristic polynomial,
// then Gauss-Jordan on the first three rows of W - lambda I with q_z = -1 (as published).
// With adjugate = true, the lanes whose leading minor is small (q_z^2 < 0.01) take the
// best adjugate column instead (AdjugateNullVector.h), computed when any lane needs it.
template <class T>
static Quat<T> FlaeNewton4(const Problem<T>& P, bool adjugate)
{
	const Acc<T> A = accumulate(P);
	const auto& H = A.b;   // H[i][k] = sum w q_i p_k, the library's MM
	T W[4][4];
	W[0][0] = H[0][0] + H[1][1] + H[2][2]; W[0][1] = -H[1][2] + H[2][1]; W[0][2] = -H[2][0] + H[0][2]; W[0][3] = -H[0][1] + H[1][0];
	W[1][1] = H[0][0] - H[1][1] - H[2][2]; W[1][2] = H[0][1] + H[1][0]; W[1][3] = H[0][2] + H[2][0];
	W[2][2] = H[1][1] - H[0][0] - H[2][2]; W[2][3] = H[1][2] + H[2][1];
	W[3][3] = H[2][2] - H[1][1] - H[0][0];
	for (int i = 0; i < 4; ++i) for (int k = 0; k < i; ++k) W[i][k] = W[k][i];
	T sq = 0;
	for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) sq += H[i][k] * H[i][k];
	const T a = -2.0 * sq, b = -8.0 * det3(H), c = wedge(wedge(W[0], W[1]), wedge(W[2], W[3]));
	T lambda = 0.5 * A.S, prev;
	do {
		prev = lambda;
		const T l2 = lambda * lambda, l3 = l2 * lambda;
		lambda -= (l3 * lambda + a * l2 + b * lambda + c) / (4.0 * l3 + 2.0 * a * lambda + b);
	} while (any(vabs(prev - lambda) > 1e-5));
	for (int i = 0; i < 4; ++i) W[i][i] -= lambda;
	T G[4][4];
	for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k) G[i][k] = W[i][k];
	// Gauss-Jordan on rows 0..2
	T r0[4], r1[4], r2[4];
	for (int k = 0; k < 4; ++k) { r0[k] = W[0][k]; r1[k] = W[1][k]; r2[k] = W[2][k]; }
	auto eliminate = [](T* piv, int col, T* x, T* y) {
		const T s = 1.0 / piv[col];
		for (int k = 0; k < 4; ++k) piv[k] = piv[k] * s;
		const T fx = x[col], fy = y[col];
		for (int k = 0; k < 4; ++k) { x[k] = x[k] - fx * piv[k]; y[k] = y[k] - fy * piv[k]; }
	};
	eliminate(r0, 0, r1, r2);
	eliminate(r1, 1, r0, r2);
	eliminate(r2, 2, r0, r1);
	Quat<T> q = { r0[3], r1[3], r2[3], T(-1.0) };
	if (adjugate) {
		const T lead = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1]) - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
		             + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
		const T dp = 4.0 * lambda * lambda * lambda + 2.0 * a * lambda + b;
		auto need = notm(vabs(lead) >= 0.01 * vabs(dp));   // NaN fails the comparison
		if (any(need)) {
			// AdjugateNullVector: 2x2 minors of rows 0-1 (s) and rows 2-3 (c)
			const T s0 = G[0][0] * G[1][1] - G[1][0] * G[0][1], s1 = G[0][0] * G[1][2] - G[1][0] * G[0][2];
			const T s2 = G[0][0] * G[1][3] - G[1][0] * G[0][3], s3 = G[0][1] * G[1][2] - G[1][1] * G[0][2];
			const T s4 = G[0][1] * G[1][3] - G[1][1] * G[0][3], s5 = G[0][2] * G[1][3] - G[1][2] * G[0][3];
			const T c0 = G[2][0] * G[3][1] - G[3][0] * G[2][1], c1 = G[2][0] * G[3][2] - G[3][0] * G[2][2];
			const T c2 = G[2][0] * G[3][3] - G[3][0] * G[2][3], c3 = G[2][1] * G[3][2] - G[3][1] * G[2][2];
			const T c4 = G[2][1] * G[3][3] - G[3][1] * G[2][3], c5 = G[2][2] * G[3][3] - G[3][2] * G[2][3];
			const T d[4] = { G[1][1] * c5 - G[1][2] * c4 + G[1][3] * c3, G[0][0] * c5 - G[0][2] * c2 + G[0][3] * c1,
			                 G[3][0] * s4 - G[3][1] * s2 + G[3][3] * s0, G[2][0] * s3 - G[2][1] * s1 + G[2][2] * s0 };
			const T col[4][4] = {
				{ d[0], -G[0][1] * c5 + G[0][2] * c4 - G[0][3] * c3, G[3][1] * s5 - G[3][2] * s4 + G[3][3] * s3, -G[2][1] * s5 + G[2][2] * s4 - G[2][3] * s3 },
				{ -G[1][0] * c5 + G[1][2] * c2 - G[1][3] * c1, d[1], -G[3][0] * s5 + G[3][2] * s2 - G[3][3] * s1, G[2][0] * s5 - G[2][2] * s2 + G[2][3] * s1 },
				{ G[1][0] * c4 - G[1][1] * c2 + G[1][3] * c0, -G[0][0] * c4 + G[0][1] * c2 - G[0][3] * c0, d[2], -G[2][0] * s4 + G[2][1] * s2 - G[2][3] * s0 },
				{ -G[1][0] * c3 + G[1][1] * c1 - G[1][2] * c0, G[0][0] * c3 - G[0][1] * c1 + G[0][2] * c0, -G[3][0] * s3 + G[3][1] * s1 - G[3][2] * s0, d[3] } };
			T best = vabs(d[0]), v[4] = { col[0][0], col[0][1], col[0][2], col[0][3] };
			for (int kk = 1; kk < 4; ++kk) {
				auto m = vabs(d[kk]) > best;
				best = sel(m, vabs(d[kk]), best);
				for (int i = 0; i < 4; ++i) v[i] = sel(m, col[kk][i], v[i]);
			}
			q = { sel(need, v[0], q.w), sel(need, v[1], q.x), sel(need, v[2], q.y), sel(need, v[3], q.z) };
		}
	}
	return normalized(q);
}

// sum of all output components, so that the compiler cannot drop any of them when timing
static inline double sum(const Quat<double>& q) { return q.w + q.x + q.y + q.z; }
static inline double sum(const Quat<V4>& q) { const V4 v = q.w + q.x + q.y + q.z; return v[0] + v[1] + v[2] + v[3]; }
static inline double sum(const Mat<double>& m) { double r = 0; for (auto& row : m.m) for (double v : row) r += v; return r; }
static inline double sum(const Mat<V4>& m) { V4 r = 0; for (auto& row : m.m) for (const V4& v : row) r += v; return r[0] + r[1] + r[2] + r[3]; }

// ---------------------------------------------------------------- benchmark
int main(int argc, char** argv)
{
	using namespace Eigen;
	const std::string dir = argc > 1 ? argv[1] : "../data";
	std::mt19937_64 rng(5);
	std::normal_distribution<double> g;
	const int problems = 1024;
	FILE* out = fopen((dir + "/simd_timing.csv").c_str(), "w");
	fprintf(out, "n,proposed_scalar,quest_scalar,foam_scalar,foamq_scalar,flae_scalar,flae_adj_scalar,proposed_simd,quest_simd,quest_simd_1e-4,foam_simd,foamq_simd,flae_simd,flae_adj_simd,max_angle_vs_library\n");
	for (int n : { 3, 10, 30, 100, 1000 }) {
		// a pool of problems with uniformly random rotations, small noise
		std::vector<std::vector<Vector3d>> Ps(problems), Qs(problems);
		std::vector<std::vector<double>> Ws(problems);
		for (int t = 0; t < problems; ++t) {
			const Quaterniond R = Quaterniond::UnitRandom();
			for (int j = 0; j < n; ++j) {
				const Vector3d p(g(rng), g(rng), g(rng));
				Ps[t].push_back(p);
				Qs[t].push_back(R._transformVector(p) + 0.01 * Vector3d(g(rng), g(rng), g(rng)));
				Ws[t].push_back(1.0 / n);
			}
		}
		// structure-of-arrays layouts: scalar, and 4 problems per lane group
		std::vector<double> s1(7 * problems * n);
		std::vector<V4> s4(7 * (problems / 4) * n);
		for (int t = 0; t < problems; ++t)
			for (int j = 0; j < n; ++j) {
				const double vals[7] = { Ps[t][j].x(), Ps[t][j].y(), Ps[t][j].z(), Qs[t][j].x(), Qs[t][j].y(), Qs[t][j].z(), Ws[t][j] };
				for (int f = 0; f < 7; ++f) {
					s1[(t * 7 + f) * n + j] = vals[f];
					s4[((t / 4) * 7 + f) * n + j][t % 4] = vals[f];
				}
			}
		auto prob1 = [&](int t) { const double* b = &s1[t * 7 * n]; return Problem<double>{ b, b + n, b + 2 * n, b + 3 * n, b + 4 * n, b + 5 * n, b + 6 * n, n }; };
		auto prob4 = [&](int t) { const V4* b = &s4[t * 7 * n]; return Problem<V4>{ b, b + n, b + 2 * n, b + 3 * n, b + 4 * n, b + 5 * n, b + 6 * n, n }; };
		auto angle = [](const Quaterniond& a, const Quaterniond& b) { const Quaterniond d = a * b.conjugate(); return 2 * std::atan2(d.vec().norm(), std::abs(d.w())); };
		auto mangle = [&](const Matrix3d& a, const Matrix3d& b) { return angle(Quaterniond(a * b.transpose()), Quaterniond::Identity()); };
		auto toM = [](const Mat<double>& m) { Matrix3d r; for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) r(i, k) = m.m[i][k]; return r; };
		auto toM4 = [](const Mat<V4>& m, int l) { Matrix3d r; for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) r(i, k) = m.m[i][k][l]; return r; };

		// correctness against the library implementations
		// (published FLAE is compared only where the library does not take its fallback, |q_z| > 0.1)
		double err = 0;
		auto Qd = [](const Quat<double>& q) { return Quaterniond(q.w, q.x, q.y, q.z); };
		for (int t = 0; t < problems; ++t) {
			const Quaterniond lf = FlaeNewton(Ps[t], Qs[t], Ws[t]);
			err = std::max(err, angle(Qd(Proposed(prob1(t))), LARotorEstimator(Ps[t], Qs[t], Ws[t])));
			err = std::max(err, angle(Qd(Quest4(prob1(t), 0.1)), Quest(Ps[t], Qs[t], Ws[t])));
			err = std::max(err, mangle(toM(Foam4(prob1(t))), Foam(Ps[t], Qs[t], Ws[t])));
			err = std::max(err, angle(Qd(FoamQuat4(prob1(t))), Quaterniond(Foam(Ps[t], Qs[t], Ws[t]))));
			err = std::max(err, angle(Qd(FlaeNewton4(prob1(t), true)), lf));
			if (std::abs(lf.normalized().z()) > 0.1) err = std::max(err, angle(Qd(FlaeNewton4(prob1(t), false)), lf));
		}
		for (int t = 0; t < problems / 4; ++t) {
			auto p4 = Proposed(prob4(t)); auto q4 = Quest4(prob4(t), 0.1); auto f4 = Foam4(prob4(t)); auto fq4 = FoamQuat4(prob4(t));
			auto a4 = FlaeNewton4(prob4(t), true), n4 = FlaeNewton4(prob4(t), false);
			auto Q4 = [](const Quat<V4>& q, int l) { return Quaterniond(q.w[l], q.x[l], q.y[l], q.z[l]); };
			for (int l = 0; l < 4; ++l) {
				const int u = 4 * t + l;
				const Quaterniond lf = FlaeNewton(Ps[u], Qs[u], Ws[u]);
				err = std::max(err, angle(Q4(p4, l), LARotorEstimator(Ps[u], Qs[u], Ws[u])));
				err = std::max(err, angle(Q4(q4, l), Quest(Ps[u], Qs[u], Ws[u])));
				err = std::max(err, mangle(toM4(f4, l), Foam(Ps[u], Qs[u], Ws[u])));
				err = std::max(err, angle(Q4(fq4, l), Quaterniond(Foam(Ps[u], Qs[u], Ws[u]))));
				err = std::max(err, angle(Q4(a4, l), lf));
				if (std::abs(lf.normalized().z()) > 0.1) err = std::max(err, angle(Q4(n4, l), lf));
			}
		}

		const long reps = std::max(1L, 20000000L / (long(problems) * (n + 10)));
		auto time = [&](auto&& f) {
			double best = 1e300;
			for (int r = 0; r < 5; ++r) {
				const auto t0 = std::chrono::steady_clock::now();
				for (long k = 0; k < reps; ++k) f();
				best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (reps * problems));
			}
			return best;
		};
		volatile double sink = 0;
		const double tP1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(Proposed(prob1(t))); });
		const double tQ1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(Quest4(prob1(t), 0.1)); });
		const double tP4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(Proposed(prob4(t))); });
		const double tQ4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(Quest4(prob4(t), 0.1)); });
		const double tQ4b = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(Quest4(prob4(t), 1e-4)); });
		const double tF1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(Foam4(prob1(t))); });
		const double tG1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(FoamQuat4(prob1(t))); });
		const double tL1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(FlaeNewton4(prob1(t), false)); });
		const double tA1 = time([&] { for (int t = 0; t < problems; ++t) sink = sink + sum(FlaeNewton4(prob1(t), true)); });
		const double tF4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(Foam4(prob4(t))); });
		const double tG4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(FoamQuat4(prob4(t))); });
		const double tL4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(FlaeNewton4(prob4(t), false)); });
		const double tA4 = time([&] { for (int t = 0; t < problems / 4; ++t) sink = sink + sum(FlaeNewton4(prob4(t), true)); });
		if (err > 1e-8) { fprintf(stderr, "SIMD implementations differ from the library by %g\n", err); return 1; }
		fprintf(out, "%d,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1e\n", n, tP1, tQ1, tF1, tG1, tL1, tA1, tP4, tQ4, tQ4b, tF4, tG4, tL4, tA4, err);
	}
	fclose(out);

	// accuracy near rotations by pi, noise-free unit vectors, N = 100
	out = fopen((dir + "/simd_accuracy.csv").c_str(), "w");
	fprintf(out, "delta,qw,proposed,quest_0.1,quest_1e-4,foam,flae,flae_adj\n");
	for (double delta : { 1e-1, 3e-2, 2.1e-2, 1.9e-2, 1e-2, 1e-3, 1e-5, 1e-7 }) {
		double e[6] = { 0, 0, 0, 0, 0, 0 };
		for (int rep = 0; rep < 200; ++rep) {
			const Quaterniond R(AngleAxisd(M_PI - delta, Vector3d(g(rng), g(rng), g(rng)).normalized()));
			const int n = 100;
			std::vector<double> s(7 * n);
			for (int j = 0; j < n; ++j) {
				const Vector3d p = Vector3d(g(rng), g(rng), g(rng)).normalized(), q = R._transformVector(p);
				const double v[7] = { p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), 1.0 / n };
				for (int f = 0; f < 7; ++f) s[f * n + j] = v[f];
			}
			const Problem<double> P{ &s[0], &s[n], &s[2 * n], &s[3 * n], &s[4 * n], &s[5 * n], &s[6 * n], n };
			auto ang = [&](Quat<double> q) { const Quaterniond d = Quaterniond(q.w, q.x, q.y, q.z) * R.conjugate(); return 2 * std::atan2(d.vec().norm(), std::abs(d.w())); };
			auto angM = [&](const Mat<double>& m) { Matrix3d r; for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) r(i, k) = m.m[i][k];
				const Quaterniond d(r * R.toRotationMatrix().transpose()); return 2 * std::atan2(d.vec().norm(), std::abs(d.w())); };
			e[0] = std::max(e[0], ang(Proposed(P)));
			e[1] = std::max(e[1], ang(Quest4(P, 0.1)));
			e[2] = std::max(e[2], ang(Quest4(P, 1e-4)));
			e[3] = std::max(e[3], angM(Foam4(P)));
			e[4] = std::max(e[4], ang(FlaeNewton4(P, false)));
			e[5] = std::max(e[5], ang(FlaeNewton4(P, true)));
		}
		fprintf(out, "%.1e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e,%.2e\n", delta, std::sin(delta / 2), e[0], e[1], e[2], e[3], e[4], e[5]);
	}
	fclose(out);

	// accuracy near FLAE's singular set q_z = 0: random angle, axis at angle delta from the xy-plane
	out = fopen((dir + "/simd_accuracy_flae.csv").c_str(), "w");
	fprintf(out, "delta,proposed,quest_0.1,foam,flae,flae_adj\n");
	for (double delta : { 1e-1, 1e-2, 1e-3, 1e-5, 1e-7, 0.0 }) {
		double e[5] = { 0, 0, 0, 0, 0 };
		std::uniform_real_distribution<double> u(0.1, M_PI - 0.1), phi(0, 2 * M_PI);
		for (int rep = 0; rep < 200; ++rep) {
			const double f = phi(rng);
			const Quaterniond R(AngleAxisd(u(rng), Vector3d(std::cos(delta) * std::cos(f), std::cos(delta) * std::sin(f), std::sin(delta))));
			const int n = 100;
			std::vector<double> s(7 * n);
			for (int j = 0; j < n; ++j) {
				const Vector3d p = Vector3d(g(rng), g(rng), g(rng)).normalized(), q = R._transformVector(p);
				const double v[7] = { p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), 1.0 / n };
				for (int k = 0; k < 7; ++k) s[k * n + j] = v[k];
			}
			const Problem<double> P{ &s[0], &s[n], &s[2 * n], &s[3 * n], &s[4 * n], &s[5 * n], &s[6 * n], n };
			auto ang = [&](Quat<double> q) { const Quaterniond d = Quaterniond(q.w, q.x, q.y, q.z) * R.conjugate(); return 2 * std::atan2(d.vec().norm(), std::abs(d.w())); };
			auto angM = [&](const Mat<double>& m) { Matrix3d r; for (int i = 0; i < 3; ++i) for (int k = 0; k < 3; ++k) r(i, k) = m.m[i][k];
				const Quaterniond d(r * R.toRotationMatrix().transpose()); return 2 * std::atan2(d.vec().norm(), std::abs(d.w())); };
			e[0] = std::max(e[0], ang(Proposed(P)));
			e[1] = std::max(e[1], ang(Quest4(P, 0.1)));
			e[2] = std::max(e[2], angM(Foam4(P)));
			e[3] = std::max(e[3], ang(FlaeNewton4(P, false)));
			e[4] = std::max(e[4], ang(FlaeNewton4(P, true)));
		}
		fprintf(out, "%.1e,%.2e,%.2e,%.2e,%.2e,%.2e\n", delta, e[0], e[1], e[2], e[3], e[4]);
	}
	fclose(out);
	return 0;
}
