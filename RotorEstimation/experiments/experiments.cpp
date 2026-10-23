// Experiments of the paper "Singularity-Free Rotor Estimation with Geometric Algebra".
//
// One implementation of the method of the paper, in the basis eps0..eps3 = (1, E1, E2, E3)
// of the even subalgebra (coordinates = Hamilton quaternion w, x, y, z), whose last step
// can read the eigenrotor out in different ways ("charts"). It writes, into ../data:
//   charts.csv            Table 2: angle errors of each chart on special rotations
//   near_singular_pi.csv  Figure 3a: rotations by pi - delta
//   near_singular_sigma.csv  Figure 3b: rotations approaching the sigma chart's sphere
//   accuracy.csv          Table 3: the proposed method against an eigensolver
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "GARotorEstimator.h"

using namespace Eigen;
using std::vector;

struct Bivector4 { double e01, e02, e03, e12, e13, e23; };
struct Trivector4 { double e012, e013, e023, e123; };

static Bivector4 wedge(const Vector4d& a, const Vector4d& b)
{
	return { a[0] * b[1] - a[1] * b[0], a[0] * b[2] - a[2] * b[0], a[0] * b[3] - a[3] * b[0],
	         a[1] * b[2] - a[2] * b[1], a[1] * b[3] - a[3] * b[1], a[2] * b[3] - a[3] * b[2] };
}
static Trivector4 wedge(const Bivector4& B, const Vector4d& c)
{
	return { B.e01 * c[2] - B.e02 * c[1] + B.e12 * c[0],
	         B.e01 * c[3] - B.e03 * c[1] + B.e13 * c[0],
	         B.e02 * c[3] - B.e03 * c[2] + B.e23 * c[0],
	         B.e12 * c[3] - B.e13 * c[2] + B.e23 * c[1] };
}
static double wedge(const Bivector4& A, const Bivector4& C)
{
	return A.e01 * C.e23 - A.e02 * C.e13 + A.e03 * C.e12 + A.e12 * C.e03 - A.e13 * C.e02 + A.e23 * C.e01;
}
static Vector4d dual(const Trivector4& T) { return Vector4d(T.e123, -T.e023, T.e013, -T.e012); }

enum class Chart { Scalar, Z, Alternating, Best };
static const char* chartName[] = { "scalar", "z", "alternating", "best" };

struct Result { Quaterniond q; int newtonSteps; };

static Result Estimate(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& c, Chart chart)
{
	Matrix3d B = Matrix3d::Zero();
	double S = 0;
	for (size_t j = 0; j < P.size(); ++j) {
		S += c[j] * (P[j].squaredNorm() + Q[j].squaredNorm());
		B.noalias() += (c[j] * Q[j]) * P[j].transpose();
	}
	const Vector3d z(B(2, 1) - B(1, 2), B(0, 2) - B(2, 0), B(1, 0) - B(0, 1));   // sum c p x q
	const double tr = B.trace();
	const Matrix3d L = B + B.transpose() - tr * Matrix3d::Identity();
	const Vector4d k0(tr, z[0], z[1], z[2]);
	const Vector4d k1(z[0], L(0, 0), L(1, 0), L(2, 0));
	const Vector4d k2(z[1], L(0, 1), L(1, 1), L(2, 1));
	const Vector4d k3(z[2], L(0, 2), L(1, 2), L(2, 2));
	const double a = -2.0 * B.squaredNorm(), b = -8.0 * B.determinant();
	const double cc = wedge(wedge(k0, k1), wedge(k2, k3));
	double mu = 0.5 * S, prev;
	int steps = 0;
	do {
		prev = mu; ++steps;
		const double mu2 = mu * mu;
		mu -= ((mu2 + a) * mu2 + b * mu + cc) / ((4.0 * mu2 + 2.0 * a) * mu + b);
	} while (std::abs(prev - mu) > 1e-5);
	const Vector4d x0 = k0 - mu * Vector4d::Unit(0), x1 = k1 - mu * Vector4d::Unit(1);
	const Vector4d x2 = k2 - mu * Vector4d::Unit(2), x3 = k3 - mu * Vector4d::Unit(3);
	const Bivector4 A = wedge(x0, x1), C = wedge(x2, x3);
	const Trivector4 T[4] = { wedge(C, x1), wedge(C, x0), wedge(A, x3), wedge(A, x2) };
	Vector4d r;
	switch (chart) {
	case Chart::Scalar: r = dual(T[0]); break;
	case Chart::Z: r = dual(T[3]); break;
	case Chart::Alternating: r = dual(T[0]) + dual(T[1]) + dual(T[2]) + dual(T[3]); break;
	case Chart::Best: {
		const double own[4] = { T[0].e123, T[1].e023, T[2].e013, T[3].e012 };
		int k = 0;
		for (int i = 1; i < 4; ++i) if (std::abs(own[i]) > std::abs(own[k])) k = i;
		r = dual(T[k]);
	} break;
	}
	Quaterniond q(r[0], r[1], r[2], r[3]);
	q.normalize();
	return { q, steps };
}

// reference: largest eigenvector of the same 4x4 matrix with Eigen's eigensolver
static Quaterniond Reference(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& c)
{
	Matrix3d B = Matrix3d::Zero();
	for (size_t j = 0; j < P.size(); ++j) B.noalias() += (c[j] * Q[j]) * P[j].transpose();
	const Vector3d z(B(2, 1) - B(1, 2), B(0, 2) - B(2, 0), B(1, 0) - B(0, 1));
	Matrix4d K;
	K(0, 0) = B.trace(); K.block<1, 3>(0, 1) = z.transpose(); K.block<3, 1>(1, 0) = z;
	K.block<3, 3>(1, 1) = B + B.transpose() - B.trace() * Matrix3d::Identity();
	SelfAdjointEigenSolver<Matrix4d> es(K);
	const Vector4d v = es.eigenvectors().col(3);
	return Quaterniond(v[0], v[1], v[2], v[3]);
}

static double angle(const Quaterniond& a, const Quaterniond& b)   // angle of the rotation a b^-1
{
	const Quaterniond d = a * b.conjugate();
	return 2.0 * std::atan2(d.vec().norm(), std::abs(d.w()));
}

static double loss(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& c, const Quaterniond& q)
{
	double e = 0;
	for (size_t j = 0; j < P.size(); ++j) e += c[j] * (q._transformVector(P[j]) - Q[j]).squaredNorm();
	return e;
}

static std::mt19937_64 rng(2026);
static Vector3d gaussian() { std::normal_distribution<double> n; return Vector3d(n(rng), n(rng), n(rng)); }
static double uniform(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

static void NoiseFree(const Quaterniond& R, int n, vector<Vector3d>& P, vector<Vector3d>& Q, vector<double>& c)
{
	P.clear(); Q.clear(); c.assign(n, 1.0 / n);
	for (int j = 0; j < n; ++j) { P.push_back(gaussian().normalized()); Q.push_back(R._transformVector(P.back())); }
}

static Quaterniond FromAngleAxis(double theta, const Vector3d& axis) { return Quaterniond(AngleAxisd(theta, axis.normalized())); }

int main(int argc, char** argv)
{
	const std::string dir = argc > 1 ? argv[1] : "../data";
	vector<Vector3d> P, Q; vector<double> c;

	// Table 2: special rotations, noise-free unit vectors, N = 1000
	{
		struct Case { const char* name; Quaterniond R; };
		const Case cases[] = {
			{ "identity", Quaterniond::Identity() },
			{ "pi about (1,2,3)", FromAngleAxis(M_PI, Vector3d(1, 2, 3)) },
			{ "pi/2 about z", FromAngleAxis(M_PI / 2, Vector3d(0, 0, 1)) },
			{ "2pi/3 about (1,1,0)", FromAngleAxis(2 * M_PI / 3, Vector3d(1, 1, 0)) },
			{ "1 rad about (1,2,3)", FromAngleAxis(1.0, Vector3d(1, 2, 3)) },
		};
		FILE* f = fopen((dir + "/charts.csv").c_str(), "w");
		fprintf(f, "rotation,w,x,y,z,scalar,z_chart,alternating,best,eigensolver,newton_steps\n");
		for (const Case& cs : cases) {
			NoiseFree(cs.R, 1000, P, Q, c);
			double e[4];
			int steps = 0;
			for (int k = 0; k < 4; ++k) { Result r = Estimate(P, Q, c, Chart(k)); e[k] = angle(r.q, cs.R); steps = r.newtonSteps; }
			const double ref = angle(Reference(P, Q, c), cs.R);
			// the library implementation must agree with the best chart
			const double lib = angle(LARotorEstimator(P, Q, c), Estimate(P, Q, c, Chart::Best).q);
			if (lib > 1e-9) { fprintf(stderr, "LARotorEstimator differs from the best chart by %g\n", lib); return 1; }
			fprintf(f, "\"%s\",%.6f,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e,%.3e,%.3e,%d\n", cs.name, cs.R.w(), cs.R.x(), cs.R.y(), cs.R.z(), e[0], e[1], e[2], e[3], ref, steps);
		}
		fclose(f);
	}

	// Figure 3: approaching a chart's singular sphere, noise-free unit vectors, N = 1000
	{
		FILE* fa = fopen((dir + "/near_singular_pi.csv").c_str(), "w");
		FILE* fb = fopen((dir + "/near_singular_sigma.csv").c_str(), "w");
		fprintf(fa, "delta,scalar,z_chart,alternating,best\n");
		fprintf(fb, "delta,scalar,z_chart,alternating,best\n");
		for (int i = 0; i <= 48; ++i) {
			const double delta = std::pow(10.0, -12.0 + 0.25 * i);
			// (a) rotation by pi - delta about (1,2,3): q_w = sin(delta / 2)
			// (b) rotation by pi/2 + delta about z: q_w - q_x + q_y - q_z = cos - sin of (pi/4 + delta/2)
			const Quaterniond Ra = FromAngleAxis(M_PI - delta, Vector3d(1, 2, 3));
			const Quaterniond Rb = FromAngleAxis(M_PI / 2 + delta, Vector3d(0, 0, 1));
			for (int s = 0; s < 2; ++s) {
				const Quaterniond& R = s == 0 ? Ra : Rb;
				double worst[4] = { 0, 0, 0, 0 };
				for (int rep = 0; rep < 20; ++rep) {
					NoiseFree(R, 1000, P, Q, c);
					for (int k = 0; k < 4; ++k) worst[k] = std::max(worst[k], angle(Estimate(P, Q, c, Chart(k)).q, R));
				}
				fprintf(s == 0 ? fa : fb, "%.6e,%.6e,%.6e,%.6e,%.6e\n", delta, std::max(worst[0], 1e-17), std::max(worst[1], 1e-17), std::max(worst[2], 1e-17), std::max(worst[3], 1e-17));
			}
		}
		fclose(fa); fclose(fb);
	}

	// Table 3: random problems, the proposed method (library implementation) against the eigensolver
	{
		FILE* f = fopen((dir + "/accuracy.csv").c_str(), "w");
		fprintf(f, "noise,trials,max_angle_vs_eigensolver,max_rel_excess_loss,max_angle_vs_truth,mean_newton_steps,max_newton_steps\n");
		const double noises[] = { 0.0, 1e-6, 1e-3, 1e-1, 1.0 };
		for (double sigma : noises) {
			const int trials = 20000;
			double maxAngle = 0, maxExcess = 0, maxTruth = 0, sumSteps = 0; int maxSteps = 0;
			for (int t = 0; t < trials; ++t) {
				const int n = int(std::exp(uniform(std::log(3.0), std::log(1000.0))));
				const bool unit = t % 2 == 0, planar = t % 5 == 0;
				const Quaterniond R = Quaterniond::UnitRandom();
				P.clear(); Q.clear(); c.clear();
				for (int j = 0; j < n; ++j) {
					Vector3d p = gaussian().normalized() * (unit ? 1.0 : uniform(1.0, 1000.0));
					if (planar) p[2] = 0.0;
					P.push_back(p);
					Q.push_back(R._transformVector(p) + sigma * p.norm() * gaussian());
					c.push_back(uniform(0.1, 1.0));
				}
				const Quaterniond est = LARotorEstimator(P, Q, c);
				const Quaterniond ref = Reference(P, Q, c);
				maxAngle = std::max(maxAngle, angle(est, ref));
				const double le = loss(P, Q, c, est), lr = loss(P, Q, c, ref);
				double S = 0; for (int j = 0; j < n; ++j) S += c[j] * (P[j].squaredNorm() + Q[j].squaredNorm());
				maxExcess = std::max(maxExcess, (le - lr) / std::max(lr, 1e-300 + 1e-16 * S));
				if (sigma == 0) maxTruth = std::max(maxTruth, angle(est, R));
				const int steps = Estimate(P, Q, c, Chart::Best).newtonSteps;
				sumSteps += steps; maxSteps = std::max(maxSteps, steps);
			}
			fprintf(f, "%g,%d,%.3e,%.3e,%.3e,%.2f,%d\n", sigma, trials, maxAngle, maxExcess, sigma == 0 ? maxTruth : -1.0, sumSteps / trials, maxSteps);
		}
		fclose(f);
	}
	return 0;
}
