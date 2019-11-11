// Flatten.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <optional>
#include <string>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry> // For Quaternion

#include "FLAE.h"
#include "GARotorEstimator.h"
#include "ArunSVD.h"
#include "Horn.h"
#include "GAValkenburg.h"
#include "FA3R.h"
#include "Davenport.h"
#include "Quest.h"
#include "QuaternionDirect.h"

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
	const vector<double>& weights, 
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


class log_stream {
public:
	std::ofstream log;
	
	log_stream(const string& filename) : log(filename)
	{
		log.precision(15);
		std::cout.precision(15);
	}

	log_stream& operator << (ostream& (*pfun)(ostream&)) {
		pfun(log);
		pfun(std::cout);
		return *this;
	}
};

template <typename T>
log_stream& operator << (log_stream& st, T val) {
	st.log << val;
	std::cout << val;
	return st;
};

enum class DataPlanes { NONE, X, Y, Z };
class CommandLineParams {
public:
	string filename;
	DataPlanes dataPlane;
	size_t N;
	bool noise;
	optional<double> norm;
	optional<Vector3d> axis;
	optional<double> angle;

	CommandLineParams(int argc, char* argv[]) {
		set_defaults();
		for(size_t i = 1 ; i < argc ; ++i) {
			string param(argv[i]);
			string key, value;
			int pos = param.find('=', 0);
			if(pos > 0) {
				key = param.substr(0, pos);
				std::transform(key.begin(), key.end(), key.begin(), ::tolower);
				value = param.substr(pos+1, param.length()-pos-1);
				std::transform(value.begin(), value.end(), value.begin(), ::tolower);
			} else {
				key = param;
			}
			if(key == "--name") {
				filename = value;
			} else if(key == "--plane") {
				if(value=="x") dataPlane = DataPlanes::X;
				if(value=="y") dataPlane = DataPlanes::Y;
				if(value=="z") dataPlane = DataPlanes::Z;
			} else if(key == "--n") {
				 N = std::stoul(value);
			} else if(key == "--noise") {
				noise = value == "true";
			} else if(key == "--norm") {
				norm = std::stod(value);				
			} else if(key == "--axis") {
				int prev = 0;
				pos = value.find(',', 0);
				double x = std::stod(value.substr(prev, pos));
				prev = pos;
				pos = value.find(',', pos+1);
				double y = std::stod(value.substr(prev+1, pos-prev-1));
				prev = pos;
				double z = std::stod(value.substr(prev+1, value.length()-prev-1));
				axis = Vector3d(x, y, z);
			} else if(key == "--angle") {
				angle = std::stod(value);
			}
		}
	}

	void set_defaults() {
		filename = "out.txt";
		dataPlane = DataPlanes::NONE;
		N = 1000;
		noise = true;
	}
};

void show_usage(log_stream& log, CommandLineParams& params)
{
	log << "Params: " << std::endl;
	log << "      "
		<< " --name=" << params.filename
		<< " --plane=" << (params.dataPlane == DataPlanes::X ? "x" : params.dataPlane == DataPlanes::Y ? "y" : params.dataPlane == DataPlanes::Z ? "z" : "[x|y|z]")
		<< " --n=" << params.N
		<< " --noise=" << (params.noise ? "true" : "false");
	if(params.norm.has_value())
		log << " --norm=" << params.norm.value_or(1);
	else
		log << " --norm=[double]";
	if(params.axis.has_value()) {
		auto axis = params.axis.value_or(Vector3d(0,0,1));
		log << " --axis=" << axis.x() << "," << axis.y() << "," << axis.z();
	}
	else
		log << " --axis=[double,double,double]";
	if(params.angle.has_value())
		log << " --angle=" << params.angle.value_or(EIGEN_PI);
	else
		log << " --angle=[double]";
	log << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	srand(static_cast<unsigned>(time(NULL)));

	CommandLineParams params(argc, argv);

	log_stream log(params.filename);

	show_usage(log, params);
	
	vector<Vector3d> pointsOriginal;
	vector<Vector3d> pointsTransformed;
	vector<double> weights;
	const size_t N = params.N;

	double totalNorms = 0;
	// Initialize the lines
	for(size_t i = 0 ; i < N ; ++i)
	{
		double norm = params.norm.value_or((rand() % 1000) + 1.0); // 1 to 1000 continuous
		Eigen::Vector3d v = getRandomVector() * norm;
		switch(params.dataPlane) {
			case DataPlanes::X: v[0] = 0; break;
			case DataPlanes::Y: v[1] = 0; break;
			case DataPlanes::Z: v[2] = 0; break;
			case DataPlanes::NONE:
			default: break;
		}
		pointsOriginal.push_back(v);
		weights.push_back(norm);
		totalNorms += norm;
	}
	for(size_t i = 0 ; i < N ; ++i)
	{
		weights[i] /= totalNorms;
	}

	Eigen::Vector3d axis = params.axis.value_or(getRandomVector());
	auto angle = params.angle.value_or(4 * EIGEN_PI * getRandom());
	Quaterniond Q;
	Q = AngleAxisd(angle, axis);

	for(size_t i = 0 ; i < N ; ++i)
	{
		if(params.noise)
			pointsTransformed.push_back((Q * AngleAxisd(2*EIGEN_PI * getRandom(), getRandomVector()))._transformVector(pointsOriginal[i]));
		else
			pointsTransformed.push_back(Q._transformVector(pointsOriginal[i]));
	}

	Quaterniond QI;
	QI.setIdentity();

	Quaterniond flaeQ = Flae(pointsOriginal, pointsTransformed, weights);
	Quaterniond flaeSymbolicQ = FlaeSymbolic(pointsOriginal, pointsTransformed, weights);
	Quaterniond flaeNewtonQ = FlaeNewton(pointsOriginal, pointsTransformed, weights);
	Matrix3d FA3EDoubleM = FA3R_double(pointsOriginal, pointsTransformed, weights);
	Matrix3d FA3EIntM = FA3R_int(pointsOriginal, pointsTransformed, weights);
	Quaterniond  GAQ = GARotorEstimator(pointsOriginal, pointsTransformed, weights);
	Quaterniond  LAQ = LARotorEstimator(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdM = SVDMcAdams(pointsOriginal, pointsTransformed, weights);
	Matrix3d  svdE = SVDEigen(pointsOriginal, pointsTransformed, weights);
	Quaterniond hornQ = Horn(pointsOriginal, pointsTransformed, weights);
	Quaterniond GAValkenburgQ = GAValkenburg(pointsOriginal, pointsTransformed, weights);
	Quaterniond davenportQ = Davenport(pointsOriginal, pointsTransformed, weights);
	Quaterniond questQ = Quest(pointsOriginal, pointsTransformed, weights);
	Matrix3d foamQ = Foam(pointsOriginal, pointsTransformed, weights);
	Quaterniond qDirect = QuaternionDirect(pointsOriginal, pointsTransformed, weights);

	double errorGroundTruth = WahbaError(pointsOriginal, pointsTransformed, Q);
	double errorFlae = WahbaError(pointsOriginal, pointsTransformed, flaeQ);
	double errorFlaeSymbolic = WahbaError(pointsOriginal, pointsTransformed, flaeSymbolicQ);
	double errorFlaeNewton = WahbaError(pointsOriginal, pointsTransformed, flaeNewtonQ);
	double errorFA3RDouble = WahbaError(pointsOriginal, pointsTransformed, FA3EDoubleM);
	double errorFA3RInt = WahbaError(pointsOriginal, pointsTransformed, FA3EIntM);
	double errorGA = WahbaError(pointsOriginal, pointsTransformed, GAQ);
	double errorLA = WahbaError(pointsOriginal, pointsTransformed, LAQ);
	double errorSVD = WahbaError(pointsOriginal, pointsTransformed, svdM);
	double errorSVDE = WahbaError(pointsOriginal, pointsTransformed, svdE);
	double errorHorn = WahbaError(pointsOriginal, pointsTransformed, hornQ);
	double errorGAValkenburg = WahbaError(pointsOriginal, pointsTransformed, GAValkenburgQ);
	double errorDavenport = WahbaError(pointsOriginal, pointsTransformed, davenportQ);
	double errorQuest = WahbaError(pointsOriginal, pointsTransformed, questQ);
	double errorFoam = WahbaError(pointsOriginal, pointsTransformed, foamQ);
	double errorQDirect = WahbaError(pointsOriginal, pointsTransformed, qDirect);

	log << "Axis                                      : " << axis.x() << ", " << axis.y() << ", " << axis.z() << std::endl;
	log << "Angle                                     : " << angle << std::endl;

	log << "Comparisons: " << std::endl << std::endl;
	//log << "Ground Truth error                        " << errorGroundTruth << endl;
	log << "FLAE error                                " << errorFlae << endl;
	log << "FLAE Symbolic error                       " << errorFlaeSymbolic << endl;
	log << "FLAE Newton error                         " << errorFlaeNewton << endl;
	log << "FA3R Double error                         " << errorFA3RDouble << endl;
	log << "FA3R Int error                            " << errorFA3RInt << endl;
	log << "GA Rotor Estimator error                  " << errorGA << endl;
	log << "LA Rotor Estimator error                  " << errorLA << endl;
	log << "Davenport error                           " << errorDavenport << endl;
	log << "Quest error                               " << errorQuest << endl;
	log << "Foam error                                " << errorFoam << endl;
	log << "GA Valkenburg error                       " << errorGAValkenburg << endl;
	log << "SVD McAdams error                         " << errorSVD << endl;
	log << "SVD error                                 " << errorSVDE << endl;
	log << "Horn error                                " << errorHorn << endl;
	log << "Quaternion Direct error                   " << errorQDirect << endl;

	double total;
	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Flae);
	log << "Exec time FLAE:                           "<< total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, FlaeSymbolic);
	log << "Exec time FLAE Symbolic:                  " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, FlaeNewton);
	log << "Exec time FLAE Newton:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, FA3R_double);
	log << "Exec time FA3R Double:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, FA3R_int);
	log << "Exec time FA3R Int:                       " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, LARotorEstimator);
	log << "Exec time LARotorEstimator:               " << total / double(CLOCKS_PER_SEC)  << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Davenport);
	log << "Exec time Davenport Q-Method:             " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Quest);
	log << "Exec time QUEST:                          " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, Foam);
	log << "Exec time FOAM:                           " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, GAValkenburg);
	log << "Exec time GA Valkenburg:                  " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDMcAdams);
	log << "Exec time SVD McAdams:                    " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Matrix3d>(pointsOriginal, pointsTransformed, weights, SVDEigen);
	log << "Exec time SVD:                            " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	total = benchmark<Quaterniond>(pointsOriginal, pointsTransformed, weights, Horn);
	log << "Exec time Horn:                           " << total / double(CLOCKS_PER_SEC) << " sec." << endl;

	log << "FINISHED..." << endl;
	return 0;
}
