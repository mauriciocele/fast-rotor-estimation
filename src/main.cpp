// Flatten.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <time.h>
#include <iostream>

#include "FLAE.h"
#include "GARotorEstimator.h"
#include "ArunSVD.h"
#include "Horn.h"
#include "GAValkenburg.h"

using namespace std;
using namespace Eigen;

Vector3d getRandomVector()
{
	auto e0 = (double)rand() / (double)RAND_MAX;
	auto e1 = (double)rand() / (double)RAND_MAX;
	auto e2 = (double)rand() / (double)RAND_MAX;

	auto v = Vector3d(e0, e1, e2);
	return v.normalized();
}

double getRandom()
{
	return (double)rand() / (double)RAND_MAX;
}

double WahbaError(const vector<Vector3d>& P, const vector<Vector3d>& Q, const Quaterniond& R)
{
	const size_t N = P.size();
	double error = 0;
	for (size_t i = 0; i < N; ++i)
	{
		error += (R._transformVector(P[i]) - Q[i]).squaredNorm();
	}
	return error;
}

double WahbaError(const vector<Vector3d>& P, const vector<Vector3d>& Q, const Matrix3d& R)
{
	const size_t N = P.size();
	double error = 0;
	for (size_t i = 0; i < N; ++i)
	{
		error += (R*P[i] - Q[i]).squaredNorm();
	}
	return error;
}

template<typename ResultType>
double benchmark(
	const vector<Vector3d>& P, 
	const vector<Vector3d>& Q, 
	const std::vector<double>& weights, 
	function<ResultType(const vector<Vector3d>&, const vector<Vector3d>&, const vector<double>&)> estimator)
{
	ResultType result;
	double total = 0;
	for (int i = 0; i < 5000000; ++i) {
		clock_t time1 = clock();
		result = estimator(P, Q, weights);
		clock_t time2 = clock();
		total += time2 - time1;
	}
	return total;
}

double benchmarkIncr(
	const vector<Vector3d>& P, 
	const vector<Vector3d>& Q, 
	const std::vector<double>& weights, 
	function<Quaterniond(const vector<Vector3d>&, const vector<Vector3d>&, const vector<double>&, const Quaterniond&)> estimator)
{
	Quaterniond QI;
	QI.setIdentity();
	Quaterniond result;
	double total = 0;
	for (int i = 0; i < 5000000; ++i) {
		clock_t time1 = clock();
		result = estimator(P, Q, weights, QI);
		clock_t time2 = clock();
		total += time2 - time1;
	}
	return total;
}

int main(int argc, char* argv[])
{
	srand(static_cast<unsigned>(time(NULL)));

	vector<Vector3d> pointsOriginal;
	vector<Vector3d> pointsTransformed;
	vector<double> weights;
	const size_t N = 10;

	// Initialize the lines
	for(size_t i = 0 ; i < N ; ++i)
	{
		auto v = getRandomVector();
		pointsOriginal.push_back(v);
		weights.push_back(1.0 / double(N));
	}

	auto axis = getRandomVector();
	auto angle = 2 * M_PI*getRandom();
	Quaterniond Q;
	Q = AngleAxisd(angle, axis);

	for(size_t i = 0 ; i < N ; ++i)
	{
		pointsTransformed.push_back(Q._transformVector(pointsOriginal[i]));
	}

	Quaterniond QI;
	QI.setIdentity();

	Quaterniond flaeQ = Flae(pointsOriginal, pointsTransformed, weights);
	Quaterniond flaeNewtonQ = FlaeNewton(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAFastQ = GAFastRotorEstimator(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAFastQInc = GAFastRotorEstimatorIncr(pointsOriginal, pointsTransformed, weights, QI);
	Quaterniond GANewtonQ = GANewtonRotorEstimator(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdM = SVDMcAdams(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdE = SVDEigen(pointsOriginal, pointsTransformed, weights);
	Quaterniond hornQ = Horn(pointsOriginal, pointsTransformed, weights);
	Quaterniond GAValkenburgQ = GAValkenburg(pointsOriginal, pointsTransformed, weights);

	double errorGroundTruth = WahbaError(pointsOriginal, pointsTransformed, Q);
	double errorFlae = WahbaError(pointsOriginal, pointsTransformed, flaeQ);
	double errorFlaeNewton = WahbaError(pointsOriginal, pointsTransformed, flaeNewtonQ);
	double errorGAFast = WahbaError(pointsOriginal, pointsTransformed, GAFastQ);
	double errorGAFastInc = WahbaError(pointsOriginal, pointsTransformed, GAFastQInc);
	double errorSVD = WahbaError(pointsOriginal, pointsTransformed, svdM);
	double errorSVDE = WahbaError(pointsOriginal, pointsTransformed, svdE);
	double errorHorn = WahbaError(pointsOriginal, pointsTransformed, hornQ);
	double errorGAValkenburg = WahbaError(pointsOriginal, pointsTransformed, GAValkenburgQ);
	double errorGANewton = WahbaError(pointsOriginal, pointsTransformed, GANewtonQ);

	std::cout << "Ground Truth error " << errorGroundTruth << endl;
	std::cout << "FLAE error " << errorFlae << endl;
	std::cout << "FLAE Newton error " << errorFlaeNewton << endl;
	std::cout << "GA Fast Rotor Estimator error " << errorGAFast << endl;
	std::cout << "GA Fast Rotor Estimator Incremental error " << errorGAFastInc << endl;
	std::cout << "GA Rotor Estimator Newton error " << errorGANewton << endl;
	std::cout << "GA Valkenburg error " << errorGAValkenburg << endl;
	std::cout << "SVD McAdams error " << errorSVD << endl;
	std::cout << "SVD error " << errorSVDE << endl;
	std::cout << "Horn error " << errorHorn << endl;

	double total;
	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Flae);
	std::cout << "Exec time FLAE: "<< total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, FlaeNewton);
	std::cout << "Exec time FLAE Newton: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimator);
	std::cout << "Exec time GAFastRotorEstimator: " << total / double(CLOCKS_PER_SEC)  << " sec." << endl;

	total = benchmarkIncr(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorIncr);
	std::cout << "Exec time GAFastRotorEstimator Incremental: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GANewtonRotorEstimator);
	std::cout << "Exec time GARotorEstimator Newton: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAValkenburg);
	std::cout << "Exec time GA Valkenburg: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDMcAdams);
	std::cout << "Exec time SVD McAdams: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDEigen);
	std::cout << "Exec time SVD: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Horn);
	std::cout << "Exec time Horn: " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	std::cout << "FINISHED..." << endl;
	int key;
	std::cin >> key;
	return 0;
}