#include "Davenport.h"

using Eigen::Matrix4d;
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::Quaterniond;
using std::vector;

/*
    Davenport Matrix
    K = | p^T q            p x q            |
        | p x q    p q^T + q p^T - (p^T q)I |
    
    K q = lambda q
    K q - lambda q = 0
    (K - lambda I) q = 0

    det(K - lambda I) q = 0
    lambda^4 + C1 lambda^3 + C2 lambda^2 + C3 lambda + C4 = 0

    How to Estimate Attitude From Vector Observations
    http://www.malcolmdshuster.com/FC_MarkleyMortari_Girdwood_1999_AAS.pdf
    An Improvement to the QUEST Algorithm
    https://pdfs.semanticscholar.org/657f/afa702198881e4f9584e84b8f26c7bf83185.pdf
*/
Quaterniond Quest(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d B;
	const size_t N = P.size();
    B.setZero();
	for (size_t j = 0; j < N; ++j) {
		B.noalias() += (w[j] * P[j]) * Q[j].transpose();
	}
    double s = B.trace();
    Matrix3d S = B + B.transpose();
    Vector3d Z = Vector3d(B(1, 2) - B(2, 1), B(2, 0) - B(0, 2), B(0, 1) - B(1, 0));
    // double kappa = S.adjoint().trace(); We reaaly need the adjugate not the adjoint. 
    double kappa = S(1,1)*S(2,2) - S(1,2)*S(2,1) + S(0,0)*S(2,2) - S(0,2)*S(2,0) + S(0,0)*S(1,1) - S(0,1)*S(1,0);
    double s2 = s * s;
    double Sdet = S.determinant();
    Vector3d SZ = S * Z;
    Vector3d SSZ = S * SZ;
    double a = s2 - kappa;;
    double b = s2 + Z.dot(Z);
    double c = Sdet + Z.dot(SZ);
    double d = Z.dot(SSZ);
    // Newton's method: lambda_{n+1} = lambda_n - f(lambda) / f'(lambda)
    // f(lambda) = lambda^4 - (a + b) lambda^2 - c lambda + (a b + c s - d)
    // f'(lambda) = 4 lambda^3 - 2 (a + b) lambda + c
 	double lambda = 1.0;
	double old_lambda = 0.0;
	double lamnda2, lamnda3;
	while (fabs(old_lambda - lambda) > 1e-5) {
		old_lambda = lambda;
		lamnda2 = lambda * lambda;
		lamnda3 = lamnda2 * lambda;
		lambda = lambda - ((lamnda3*lambda - (a + b)*lamnda2 - c*lambda + (a*b + c*s - d)) / (4.0 * lamnda3 - 2.0 * (a + b)*lambda - c));
	}

    double alpha = lambda*lambda - a;
    double beta = lambda - s;
    double gamma = alpha*(lambda + s) - Sdet;
    Vector3d x = beta * SZ + SSZ + alpha * Z;
    Quaterniond q(gamma, x.x(), x.y(), x.z());
    q.normalize();
	return q;
}
