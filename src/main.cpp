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
#include "FA3R.h"
#include "Davenport.h"
#include "Quest.h"

using namespace std;
using namespace Eigen;
using namespace std::placeholders;

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
	for (int i = 0; i < 1000000; ++i) {
		clock_t time1 = clock();
		result = estimator(P, Q, weights);
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
	const size_t N = 1000;

	double totalNorms = 0;
	// Initialize the lines
	for(size_t i = 0 ; i < N ; ++i)
	{
		double norm = (rand() % 1000) + 1.0; // 1 to 10 continuous
		auto v = getRandomVector() * norm;
		pointsOriginal.push_back(v);
		weights.push_back(norm);
		totalNorms += norm;
	}
	for(size_t i = 0 ; i < N ; ++i)
	{
		weights[i] /= totalNorms;
	}

	auto axis = getRandomVector();
	auto angle = 4 * EIGEN_PI * getRandom();
	Quaterniond Q;
	Q = AngleAxisd(angle, axis);

	for(size_t i = 0 ; i < N ; ++i)
	{
		pointsTransformed.push_back((Q * AngleAxisd(2*EIGEN_PI * getRandom(), getRandomVector()))._transformVector(pointsOriginal[i]));
		//pointsTransformed.push_back(Q._transformVector(pointsOriginal[i]));
	}

	Quaterniond QI;
	QI.setIdentity();

	Quaterniond flaeQ = Flae(pointsOriginal, pointsTransformed, weights);
	Quaterniond flaeSymbolicQ = FlaeSymbolic(pointsOriginal, pointsTransformed, weights);
	Quaterniond flaeNewtonQ = FlaeNewton(pointsOriginal, pointsTransformed, weights);
	Matrix3d FA3EDoubleM = FA3R_double(pointsOriginal, pointsTransformed, weights);
	Matrix3d FA3EIntM = FA3R_int(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAFastQ = GAFastRotorEstimator(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAFastQAVX = GAFastRotorEstimatorAVX(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAQ = GARotorEstimator(pointsOriginal, pointsTransformed, weights);
	Quaterniond  LAQ = LARotorEstimator(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAFastQInc = GAFastRotorEstimatorIncr(pointsOriginal, pointsTransformed, weights, QI);
	for(size_t i = 0 ; i < 24 ; ++i) 
		GAFastQInc = GAFastRotorEstimatorIncr(pointsOriginal, pointsTransformed, weights, GAFastQInc);
	Quaterniond  GAFastQ2 = GAFastRotorEstimatorAprox(pointsOriginal, pointsTransformed, weights, 1e-13, 2);
	Quaterniond  GAFastQ4 = GAFastRotorEstimatorAprox(pointsOriginal, pointsTransformed, weights, 1e-13, 4);
	Quaterniond  GAFastQ8 = GAFastRotorEstimatorAprox(pointsOriginal, pointsTransformed, weights, 1e-13, 8);
	Quaterniond  GAFastQ15 = GAFastRotorEstimatorAprox(pointsOriginal, pointsTransformed, weights,1e-13, 15);
	Quaterniond GANewtonQ = GANewtonRotorEstimator(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdM = SVDMcAdams(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdE = SVDEigen(pointsOriginal, pointsTransformed, weights);
	Quaterniond hornQ = Horn(pointsOriginal, pointsTransformed, weights);
	Quaterniond GAValkenburgQ = GAValkenburg(pointsOriginal, pointsTransformed, weights);
	Quaterniond davenportQ = Davenport(pointsOriginal, pointsTransformed, weights);
	Quaterniond questQ = Quest(pointsOriginal, pointsTransformed, weights);
	Matrix3d foamQ = Foam(pointsOriginal, pointsTransformed, weights);

	double errorGroundTruth = WahbaError(pointsOriginal, pointsTransformed, Q);
	double errorFlae = WahbaError(pointsOriginal, pointsTransformed, flaeQ);
	double errorFlaeSymbolic = WahbaError(pointsOriginal, pointsTransformed, flaeSymbolicQ);
	double errorFlaeNewton = WahbaError(pointsOriginal, pointsTransformed, flaeNewtonQ);
	double errorFA3RDouble = WahbaError(pointsOriginal, pointsTransformed, FA3EDoubleM);
	double errorFA3RInt = WahbaError(pointsOriginal, pointsTransformed, FA3EIntM);
	double errorGAFast = WahbaError(pointsOriginal, pointsTransformed, GAFastQ);
	double errorGAFastAVX = WahbaError(pointsOriginal, pointsTransformed, GAFastQAVX);
	double errorGA = WahbaError(pointsOriginal, pointsTransformed, GAQ);
	double errorLA = WahbaError(pointsOriginal, pointsTransformed, LAQ);
	double errorGAFastInc = WahbaError(pointsOriginal, pointsTransformed, GAFastQInc);
	double errorGAFast2 = WahbaError(pointsOriginal, pointsTransformed, GAFastQ2);
	double errorGAFast4 = WahbaError(pointsOriginal, pointsTransformed, GAFastQ4);
	double errorGAFast8 = WahbaError(pointsOriginal, pointsTransformed, GAFastQ8);
	double errorGAFast15 = WahbaError(pointsOriginal, pointsTransformed, GAFastQ15);
	double errorSVD = WahbaError(pointsOriginal, pointsTransformed, svdM);
	double errorSVDE = WahbaError(pointsOriginal, pointsTransformed, svdE);
	double errorHorn = WahbaError(pointsOriginal, pointsTransformed, hornQ);
	double errorGAValkenburg = WahbaError(pointsOriginal, pointsTransformed, GAValkenburgQ);
	double errorGANewton = WahbaError(pointsOriginal, pointsTransformed, GANewtonQ);
	double errorDavenport = WahbaError(pointsOriginal, pointsTransformed, davenportQ);
	double errorQuest = WahbaError(pointsOriginal, pointsTransformed, questQ);
	double errorFoam = WahbaError(pointsOriginal, pointsTransformed, foamQ);

	std::cout.precision(15);
	//std::cout << "Ground Truth error                        " << errorGroundTruth << endl;
	std::cout << "FLAE error                                " << errorFlae << endl;
	std::cout << "FLAE Symbolic error                       " << errorFlaeSymbolic << endl;
	std::cout << "FLAE Newton error                         " << errorFlaeNewton << endl;
	std::cout << "FA3R Double error                         " << errorFA3RDouble << endl;
	std::cout << "FA3R Int error                            " << errorFA3RInt << endl;
	std::cout << "GA Fast Rotor Estimator AVX error         " << errorGAFastAVX << endl;
	std::cout << "GA Fast Rotor Estimator error             " << errorGAFast << endl;
	std::cout << "GA Rotor Estimator error                  " << errorGA << endl;
	std::cout << "LA Rotor Estimator error                  " << errorLA << endl;
	std::cout << "GA Fast Rotor Estimator Incremental error " << errorGAFastInc << endl;
	std::cout << "GA Fast Rotor Estimator Aprox 2 error     " << errorGAFast2 << endl;
	std::cout << "GA Fast Rotor Estimator Aprox 4 error     " << errorGAFast4 << endl;
	std::cout << "GA Fast Rotor Estimator Aprox 8 error     " << errorGAFast8 << endl;
	std::cout << "GA Fast Rotor Estimator Aprox 15 error    " << errorGAFast15 << endl;
	std::cout << "GA Rotor Estimator Newton error           " << errorGANewton << endl;
	std::cout << "Davenport error                           " << errorDavenport << endl;
	std::cout << "Quest error                               " << errorQuest << endl;
	std::cout << "Foam error                                " << errorFoam << endl;
	std::cout << "GA Valkenburg error                       " << errorGAValkenburg << endl;
	std::cout << "SVD McAdams error                         " << errorSVD << endl;
	std::cout << "SVD error                                 " << errorSVDE << endl;
	std::cout << "Horn error                                " << errorHorn << endl;

	auto GAFastRotorEstimatorIncrQI = std::bind(GAFastRotorEstimatorIncr, _1, _2, _3, QI);
	auto GAFastRotorEstimatorAprox2 = std::bind(GAFastRotorEstimatorAprox, _1, _2, _3, 1e-13, 2);
	auto GAFastRotorEstimatorAprox4 = std::bind(GAFastRotorEstimatorAprox, _1, _2, _3, 1e-13, 4);
	auto GAFastRotorEstimatorAprox8 = std::bind(GAFastRotorEstimatorAprox, _1, _2, _3, 1e-13, 8);
	auto GAFastRotorEstimatorAprox15 = std::bind(GAFastRotorEstimatorAprox, _1, _2, _3,1e-13, 15);

	double total;
	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Flae);
	std::cout << "Exec time FLAE:                           "<< total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, FlaeSymbolic);
	std::cout << "Exec time FLAE Symbolic:                  " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, FlaeNewton);
	std::cout << "Exec time FLAE Newton:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, FA3R_double);
	std::cout << "Exec time FA3R Double:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, FA3R_int);
	std::cout << "Exec time FA3R Int:                       " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorAVX);
	std::cout << "Exec time GAFastRotorEstimatorAVX:        " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimator);
	std::cout << "Exec time GAFastRotorEstimator:           " << total / double(CLOCKS_PER_SEC)  << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, LARotorEstimator);
	std::cout << "Exec time LARotorEstimator:               " << total / double(CLOCKS_PER_SEC)  << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorIncrQI);
	std::cout << "Exec time GAFastRotorEstimator Incr.:     " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorAprox2);
	std::cout << "Exec time GAFastRotorEstimator Aprox 2:   " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorAprox4);
	std::cout << "Exec time GAFastRotorEstimator Aprox 4:   " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorAprox8);
	std::cout << "Exec time GAFastRotorEstimator Aprox 8:   " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAFastRotorEstimatorAprox15);
	std::cout << "Exec time GAFastRotorEstimator Aprox 15:  " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GANewtonRotorEstimator);
	std::cout << "Exec time GARotorEstimator Newton:        " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Davenport);
	std::cout << "Exec time Davenport Q-Method:             " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Quest);
	std::cout << "Exec time QUEST:                          " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, Foam);
	std::cout << "Exec time FOAM:                           " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAValkenburg);
	std::cout << "Exec time GA Valkenburg:                  " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDMcAdams);
	std::cout << "Exec time SVD McAdams:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDEigen);
	std::cout << "Exec time SVD:                            " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Horn);
	std::cout << "Exec time Horn:                           " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	std::cout << "FINISHED..." << endl;
	int key;
	std::cin >> key;
	return 0;
}
