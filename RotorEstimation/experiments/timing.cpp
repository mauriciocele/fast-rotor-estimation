// Timing of all estimators (Table 4, Figure 4). Writes CSV to stdout: method, N, ns per call.
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
	const Quaterniond R(AngleAxisd(1.1, Vector3d(1, 2, 3).normalized()));
	const vector<std::pair<std::string, std::function<double(const V&, const V&, const W&)>>> methods = {
		{ "Proposed", [](const V& p, const V& q, const W& w) { return LARotorEstimator(p, q, w).w(); } },
		{ "QUEST", [](const V& p, const V& q, const W& w) { return Quest(p, q, w).w(); } },
		{ "FOAM", [](const V& p, const V& q, const W& w) { return Foam(p, q, w)(0, 0); } },
		{ "FLAE Newton", [](const V& p, const V& q, const W& w) { return FlaeNewton(p, q, w).w(); } },
		{ "FLAE", [](const V& p, const V& q, const W& w) { return Flae(p, q, w).w(); } },
		{ "Symbolic", [](const V& p, const V& q, const W& w) { return FlaeSymbolic(p, q, w).w(); } },
		{ "FA3R", [](const V& p, const V& q, const W& w) { return FA3R_double(p, q, w)(0, 0); } },
		{ "FA3R Int", [](const V& p, const V& q, const W& w) { return FA3R_int(p, q, w)(0, 0); } },
		{ "Q-Method", [](const V& p, const V& q, const W& w) { return Davenport(p, q, w).w(); } },
		{ "Horn", [](const V& p, const V& q, const W& w) { return Horn(p, q, w).w(); } },
		{ "SVD", [](const V& p, const V& q, const W& w) { return SVDEigen(p, q, w)(0, 0); } },
		{ "SVD McAdams", [](const V& p, const V& q, const W& w) { return SVDMcAdams(p, q, w)(0, 0); } },
		{ "GA Valkenburg", [](const V& p, const V& q, const W& w) { return GAValkenburg(p, q, w).w(); } },
	};
	const int sizes[] = { 3, 10, 30, 100, 300, 1000, 3000 };
	printf("method,n,ns\n");
	for (int n : sizes) {
		V P, Q; W w;
		for (int i = 0; i < n; ++i) {
			const Vector3d p(g(rng), g(rng), g(rng));
			P.push_back(p);
			Q.push_back(R._transformVector(p) + 0.01 * Vector3d(g(rng), g(rng), g(rng)));
			w.push_back(1.0 / n);
		}
		for (const auto& m : methods) {
			const long iters = 4000000L / (n + 10);
			double best = 1e300;
			volatile double sink = 0;
			for (int rep = 0; rep < 5; ++rep) {
				const auto t0 = std::chrono::steady_clock::now();
				for (long i = 0; i < iters; ++i) sink = sink + m.second(P, Q, w);
				best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / iters);
			}
			printf("%s,%d,%.1f\n", m.first.c_str(), n, best);
		}
	}
	return 0;
}
