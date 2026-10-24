// The rotor estimator with a prior rotor (PriorRotorEstimator): accuracy against a long-double reference, Newton
// steps, where the multiplier falls in its bracket, and cost against LARotorEstimator. Writes, into the directory
// given as argument (../data):
//   prior.csv         per prior angle and weight: steps, error, bracket position, rotation between data and result
//   prior_timing.csv  ns per call of LARotorEstimator and PriorRotorEstimator
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "GARotorEstimator.h"

using namespace Eigen;
using std::vector;
typedef long double ld;

static std::mt19937_64 rng(77);
static Vector3d gaussian() { std::normal_distribution<double> n; return Vector3d(n(rng), n(rng), n(rng)); }

static double angle(const Quaterniond& a, const Quaterniond& b)
{
	const Quaterniond d = a * b.conjugate();
	return 2.0 * std::atan2(d.vec().norm(), std::abs(d.w()));
}

// Davenport's matrix of the data, in the basis (x, y, z, w), as in LARotorEstimator
static Matrix4d Davenport(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w, double& S)
{
	Matrix3d Sx = Matrix3d::Zero();
	S = 0;
	for (size_t j = 0; j < P.size(); ++j) {
		S += w[j] * (P[j].squaredNorm() + Q[j].squaredNorm());
		Sx += (w[j] * P[j]) * Q[j].transpose();
	}
	const double tr = Sx.trace();
	Matrix4d K;
	K.col(0) = Vector4d(2.0 * Sx(0, 0) - tr, Sx(0, 1) + Sx(1, 0), Sx(2, 0) + Sx(0, 2), Sx(1, 2) - Sx(2, 1));
	K.col(1) = Vector4d(Sx(0, 1) + Sx(1, 0), 2.0 * Sx(1, 1) - tr, Sx(1, 2) + Sx(2, 1), Sx(2, 0) - Sx(0, 2));
	K.col(2) = Vector4d(Sx(2, 0) + Sx(0, 2), Sx(1, 2) + Sx(2, 1), 2.0 * Sx(2, 2) - tr, Sx(0, 1) - Sx(1, 0));
	K.col(3) = Vector4d(Sx(1, 2) - Sx(2, 1), Sx(2, 0) - Sx(0, 2), Sx(0, 1) - Sx(1, 0), tr);
	return K;
}

// Reference: eigendecomposition of K in long double, and bisection of the secular equation on its bracket
static Vector4d Reference(const Matrix4d& K, const Vector4d& h, double& muOut, double& mu0Out)
{
	SelfAdjointEigenSolver<Matrix<ld, 4, 4>> es(K.cast<ld>());
	const Matrix<ld, 4, 1> m = es.eigenvalues();   // increasing
	const Matrix<ld, 4, 4> V = es.eigenvectors();
	const Matrix<ld, 4, 1> c = V.transpose() * h.cast<ld>();
	const ld mu0 = m(3);
	auto norm2 = [&](ld mu) { ld s = 0; for (int k = 0; k < 4; ++k) s += c(k) * c(k) / ((mu - m(k)) * (mu - m(k))); return s; };
	ld lo = mu0 + std::abs(c(3)), hi = mu0 + h.cast<ld>().norm();
	for (int it = 0; it < 300; ++it) {
		const ld mid = 0.5L * (lo + hi);
		if (norm2(mid) > 1.0L) lo = mid; else hi = mid;
	}
	const ld mu = 0.5L * (lo + hi);
	Matrix<ld, 4, 1> r = Matrix<ld, 4, 1>::Zero();
	for (int k = 0; k < 4; ++k) r += c(k) / (mu - m(k)) * V.col(k);
	if (r.norm() < 0.999L) {   // hard case: the root is at mu0 and r is completed along the top eigenvector
		r.setZero();
		for (int k = 0; k < 3; ++k) r += c(k) / (mu0 - m(k)) * V.col(k);
		r += std::sqrt(std::max(0.0L, 1.0L - r.squaredNorm())) * V.col(3);
	}
	muOut = double(mu);
	mu0Out = double(mu0);
	return r.normalized().cast<double>();
}

int main(int argc, char** argv)
{
	const std::string dir = argc > 1 ? argv[1] : "../data";
	FILE* f = fopen((dir + "/prior.csv").c_str(), "w");
	fprintf(f, "theta_deg,beta_rel,steps_mean,steps_max,max_error_rad,max_deficit_rel,bracket_mean,result_from_data_deg\n");
	const int n = 20, trials = 500;
	for (double thetaDeg : { 0.0, 30.0, 90.0, 150.0, 179.0, 180.0 }) {
		for (double betaRel : { 0.01, 0.1, 1.0 }) {
			double stepsSum = 0, maxErr = 0, maxDeficit = 0, bracketSum = 0, fromData = 0;
			int stepsMax = 0;
			for (int t = 0; t < trials; ++t) {
				const Quaterniond R = Quaterniond::UnitRandom();
				vector<Vector3d> P(n), Q(n);
				vector<double> w(n, 1.0 / n);
				for (int j = 0; j < n; ++j) { P[j] = gaussian(); Q[j] = R._transformVector(P[j]) + 0.05 * gaussian(); }
				double S;
				const Matrix4d K = Davenport(P, Q, w, S);
				const Quaterniond data = LARotorEstimator(P, Q, w);
				// a prior thetaDeg away from the data's best rotor, on the same side
				const Quaterniond delta(AngleAxisd(thetaDeg * M_PI / 180.0, gaussian().normalized()));
				const Quaterniond prior = data * delta;
				const double beta = betaRel * 0.5 * S;
				int steps;
				const Quaterniond r = PriorRotorEstimator(P, Q, w, prior, beta, &steps);
				const Vector4d h = beta * Vector4d(prior.x(), prior.y(), prior.z(), prior.w());
				double mu, mu0;
				const Vector4d ref = Reference(K, h, mu, mu0);
				const Vector4d rv(r.x(), r.y(), r.z(), r.w());
				// at 180 degrees the two solutions y +- tau r0 tie, so compare objective values as well as rotations
				auto objective = [&](const Vector4d& x) { return x.dot(K * x) + 2.0 * h.dot(x); };
				const double deficit = (objective(ref) - objective(rv)) / (0.5 * S + h.norm());
				maxDeficit = std::max(maxDeficit, deficit);
				if (thetaDeg < 180.0) { const double sg = rv.dot(ref) < 0 ? -1.0 : 1.0; maxErr = std::max(maxErr, 2.0 * std::atan2((rv - sg * ref).norm(), (rv + sg * ref).norm())); }
				stepsSum += steps;
				stepsMax = std::max(stepsMax, steps);
				// where mu* falls in [mu0 + |h.r0|, mu0 + |h|], as a fraction (0 = left end)
				const Vector4d r0(data.x(), data.y(), data.z(), data.w());
				const double lo = mu0 + std::abs(h.dot(r0)), hi = mu0 + h.norm();
				bracketSum += hi > lo ? (mu - lo) / (hi - lo) : 0.0;
				fromData += angle(r, data) * 180.0 / M_PI;
			}
			fprintf(f, "%.0f,%.2f,%.2f,%d,%.2e,%.2e,%.3f,%.2f\n", thetaDeg, betaRel, stepsSum / trials, stepsMax, maxErr, maxDeficit, bracketSum / trials, fromData / trials);
			printf("theta %5.1f beta %.2f: steps %.2f (max %d), error %.2e, deficit %.1e, bracket %.3f, from data %.2f deg\n", thetaDeg, betaRel, stepsSum / trials, stepsMax, maxErr, maxDeficit, bracketSum / trials, fromData / trials);
		}
	}
	fclose(f);

	// cost
	f = fopen((dir + "/prior_timing.csv").c_str(), "w");
	fprintf(f, "n,la_ns,prior_ns\n");
	for (int nn : { 3, 10, 100 }) {
		const int pool = 256;
		vector<vector<Vector3d>> Ps(pool), Qs(pool);
		vector<vector<double>> Ws(pool);
		vector<Quaterniond> priors(pool);
		for (int t = 0; t < pool; ++t) {
			const Quaterniond R = Quaterniond::UnitRandom();
			for (int j = 0; j < nn; ++j) { Ps[t].push_back(gaussian()); Qs[t].push_back(R._transformVector(Ps[t].back()) + 0.01 * gaussian()); Ws[t].push_back(1.0 / nn); }
			priors[t] = R * Quaterniond(AngleAxisd(1.0, gaussian().normalized()));
		}
		volatile double sink = 0;
		auto time = [&](auto&& fn) {
			double best = 1e300;
			const long reps = std::max(1L, 2000000L / (long(pool) * (nn + 10)));
			for (int r = 0; r < 5; ++r) {
				const auto t0 = std::chrono::steady_clock::now();
				for (long k = 0; k < reps; ++k) for (int t = 0; t < pool; ++t) sink = sink + fn(t).coeffs().sum();
				best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (reps * pool));
			}
			return best;
		};
		const double tLa = time([&](int t) { return LARotorEstimator(Ps[t], Qs[t], Ws[t]); });
		const double tPr = time([&](int t) { return PriorRotorEstimator(Ps[t], Qs[t], Ws[t], priors[t], 0.05); });
		fprintf(f, "%d,%.1f,%.1f\n", nn, tLa, tPr);
		printf("n %d: LARotorEstimator %.1f ns, PriorRotorEstimator %.1f ns\n", nn, tLa, tPr);
	}
	fclose(f);
	return 0;
}
