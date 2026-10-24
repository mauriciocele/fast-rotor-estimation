// Timing of all estimators (Table 3, Figure 5). Writes CSV to stdout: method, N, ns per call.
// Each size uses a pool of problems with uniformly random rotations and small noise, so that
// data-dependent branches (e.g. QUEST's sequential rotations) are exercised as in practice.
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <Eigen/Geometry>
#include "FLAE.h"
#include "GARotorEstimator.h"
#include "ArunSVD.h"
#include "Horn.h"
#include "GAValkenburg.h"
#include "FA3R.h"
#include "Davenport.h"
#include "Quest.h"
#include "QuaternionDirect.h"
using namespace Eigen;
using std::vector;
typedef vector<Vector3d> V;
typedef vector<double> W;
int main()
{
	std::mt19937_64 rng(1);
	std::normal_distribution<double> g;
	const vector<std::pair<std::string, std::function<double(const V&, const V&, const W&)>>> methods = {
		{ "Proposed", [](const V& p, const V& q, const W& w) { return LARotorEstimator(p, q, w).coeffs().sum(); } },
		{ "QUEST", [](const V& p, const V& q, const W& w) { return Quest(p, q, w).coeffs().sum(); } },
		{ "FOAM", [](const V& p, const V& q, const W& w) { return Foam(p, q, w).sum(); } },
		{ "FLAE Newton", [](const V& p, const V& q, const W& w) { return FlaeNewton(p, q, w).coeffs().sum(); } },
		{ "FLAE", [](const V& p, const V& q, const W& w) { return Flae(p, q, w).coeffs().sum(); } },
		{ "Symbolic", [](const V& p, const V& q, const W& w) { return FlaeSymbolic(p, q, w).coeffs().sum(); } },
		{ "FA3R", [](const V& p, const V& q, const W& w) { return FA3R_double(p, q, w).sum(); } },
		{ "FA3R Int", [](const V& p, const V& q, const W& w) { return FA3R_int(p, q, w).sum(); } },
		{ "Q-Method", [](const V& p, const V& q, const W& w) { return Davenport(p, q, w).coeffs().sum(); } },
		{ "Horn", [](const V& p, const V& q, const W& w) { return Horn(p, q, w).coeffs().sum(); } },
		{ "SVD", [](const V& p, const V& q, const W& w) { return SVDEigen(p, q, w).sum(); } },
		{ "SVD McAdams", [](const V& p, const V& q, const W& w) { return SVDMcAdams(p, q, w).sum(); } },
		{ "GA Valkenburg", [](const V& p, const V& q, const W& w) { return GAValkenburg(p, q, w).coeffs().sum(); } },
	};
	const int sizes[] = { 3, 10, 30, 100, 300, 1000, 3000 };
	printf("method,n,ns\n");
	const int pool = 256;
	for (int n : sizes) {
		vector<V> P(pool), Q(pool);
		vector<W> w(pool);
		for (int t = 0; t < pool; ++t) {
			const Quaterniond R = Quaterniond::UnitRandom();
			for (int i = 0; i < n; ++i) {
				const Vector3d p(g(rng), g(rng), g(rng));
				P[t].push_back(p);
				Q[t].push_back(R._transformVector(p) + 0.01 * Vector3d(g(rng), g(rng), g(rng)));
				w[t].push_back(1.0 / n);
			}
		}
		for (const auto& m : methods) {
			const long rounds = std::max(1L, 4000000L / ((n + 10) * long(pool)));
			double best = 1e300;
			volatile double sink = 0;
			for (int rep = 0; rep < 5; ++rep) {
				const auto t0 = std::chrono::steady_clock::now();
				for (long i = 0; i < rounds; ++i)
					for (int t = 0; t < pool; ++t) sink = sink + m.second(P[t], Q[t], w[t]);
				best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (rounds * pool));
			}
			printf("%s,%d,%.1f\n", m.first.c_str(), n, best);
		}
	}
	return 0;
}
