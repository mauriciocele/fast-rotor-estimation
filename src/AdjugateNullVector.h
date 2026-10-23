#pragma once

#include <cmath>
#include <Eigen/Core>

/* Determinant of G without row 'row' and column 'col'. */
inline double Minor3(const Eigen::Matrix4d& G, int row, int col)
{
	int r[3], c[3];
	for (int i = 0, k = 0; i < 4; ++i) if (i != row) r[k++] = i;
	for (int j = 0, k = 0; j < 4; ++j) if (j != col) c[k++] = j;
	return G(r[0], c[0]) * (G(r[1], c[1]) * G(r[2], c[2]) - G(r[1], c[2]) * G(r[2], c[1]))
		- G(r[0], c[1]) * (G(r[1], c[0]) * G(r[2], c[2]) - G(r[1], c[2]) * G(r[2], c[0]))
		+ G(r[0], c[2]) * (G(r[1], c[0]) * G(r[2], c[1]) - G(r[1], c[1]) * G(r[2], c[0]));
}

/*
	Null vector v of a symmetric 4x4 matrix G of rank 3, such as M - lambda I at an
	eigenvalue lambda of M. Since adj(G) = c v v^T, every column k of adj(G) is v scaled by
	c v_k, and its diagonal entry c v_k^2 is the principal minor of G without row and
	column k. The column with the largest diagonal entry is the best conditioned one.
	Combining columns instead can cancel: their sum with alternating signs, which is what
	the sum of the four triple wedges a_i ^ a_j ^ a_k computes, vanishes for every v with
	v_1 - v_2 + v_3 - v_4 = 0.
*/
inline Eigen::Vector4d AdjugateNullVector(const Eigen::Matrix4d& G)
{
	// 2x2 minors of rows 0-1 (s) and rows 2-3 (c), shared by all 3x3 cofactors
	const double s0 = G(0, 0) * G(1, 1) - G(1, 0) * G(0, 1);
	const double s1 = G(0, 0) * G(1, 2) - G(1, 0) * G(0, 2);
	const double s2 = G(0, 0) * G(1, 3) - G(1, 0) * G(0, 3);
	const double s3 = G(0, 1) * G(1, 2) - G(1, 1) * G(0, 2);
	const double s4 = G(0, 1) * G(1, 3) - G(1, 1) * G(0, 3);
	const double s5 = G(0, 2) * G(1, 3) - G(1, 2) * G(0, 3);
	const double c0 = G(2, 0) * G(3, 1) - G(3, 0) * G(2, 1);
	const double c1 = G(2, 0) * G(3, 2) - G(3, 0) * G(2, 2);
	const double c2 = G(2, 0) * G(3, 3) - G(3, 0) * G(2, 3);
	const double c3 = G(2, 1) * G(3, 2) - G(3, 1) * G(2, 2);
	const double c4 = G(2, 1) * G(3, 3) - G(3, 1) * G(2, 3);
	const double c5 = G(2, 2) * G(3, 3) - G(3, 2) * G(2, 3);

	// diagonal of adj(G): the principal minors
	const double d[4] = {
		G(1, 1) * c5 - G(1, 2) * c4 + G(1, 3) * c3,
		G(0, 0) * c5 - G(0, 2) * c2 + G(0, 3) * c1,
		G(3, 0) * s4 - G(3, 1) * s2 + G(3, 3) * s0,
		G(2, 0) * s3 - G(2, 1) * s1 + G(2, 2) * s0 };
	int best = 0;
	for (int k = 1; k < 4; ++k)
		if (std::abs(d[k]) > std::abs(d[best]))
			best = k;

	// row 'best' of adj(G), which equals its column since G is symmetric
	switch (best) {
	case 0:
		return Eigen::Vector4d(d[0],
			-G(0, 1) * c5 + G(0, 2) * c4 - G(0, 3) * c3,
			G(3, 1) * s5 - G(3, 2) * s4 + G(3, 3) * s3,
			-G(2, 1) * s5 + G(2, 2) * s4 - G(2, 3) * s3);
	case 1:
		return Eigen::Vector4d(-G(1, 0) * c5 + G(1, 2) * c2 - G(1, 3) * c1,
			d[1],
			-G(3, 0) * s5 + G(3, 2) * s2 - G(3, 3) * s1,
			G(2, 0) * s5 - G(2, 2) * s2 + G(2, 3) * s1);
	case 2:
		return Eigen::Vector4d(G(1, 0) * c4 - G(1, 1) * c2 + G(1, 3) * c0,
			-G(0, 0) * c4 + G(0, 1) * c2 - G(0, 3) * c0,
			d[2],
			-G(2, 0) * s4 + G(2, 1) * s2 - G(2, 3) * s0);
	default:
		return Eigen::Vector4d(-G(1, 0) * c3 + G(1, 1) * c1 - G(1, 2) * c0,
			G(0, 0) * c3 - G(0, 1) * c1 + G(0, 2) * c0,
			-G(3, 0) * s3 + G(3, 1) * s1 - G(3, 2) * s0,
			d[3]);
	}
}
