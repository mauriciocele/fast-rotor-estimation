#include "Quest.h"

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

inline Matrix3d Adjugate(const Matrix3d& m)
{
    /*
        Adjugate is the transpose of its cofactor matrix:
    */

    Matrix3d Ct;
    // Eigen::Matrix2d d;

    // // (iii, jjj) => 3D matrix index
    // // (ii, jj) => 2D matrix index
    // // (i, j) => 3D Cofactor matrix index, also the row and column to skip
    // for(size_t i = 0 ; i < 3 ; ++i) {
    //     for(size_t j = 0 ; j < 3 ; ++j) {
    //         for(size_t ii = 0, iii = 0 ; ii < 2 ; ++iii) {
    //             if(iii == i) continue;
    //             for(size_t jj = 0, jjj = 0 ; jj < 2 ; ++jjj) {
    //                 if(jjj == j) continue;
    //                 d(ii,jj) = m(iii,jjj);
    //                 jj++;
    //             }
    //             ii++;
    //         }
    //         Ct(j,i) = (i+j) % 2 == 0? d.determinant() : -d.determinant();
    //     }
    // }
    // 00 01 02
    // 10 11 12
    // 20 21 22
    Ct(0,0) = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    Ct(1,0) = -m(1,0)*m(2,2) + m(1,2)*m(2,0);
    Ct(2,0) = m(1,0)*m(2,1) - m(1,1)*m(2,0);
    Ct(0,1) = -m(0,1)*m(2,2) + m(0,2)*m(2,1);
    Ct(1,1) = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    Ct(2,1) = -m(0,0)*m(2,1) + m(0,1)*m(2,0);
    Ct(0,2) = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    Ct(1,2) = -m(0,0)*m(1,2) + m(0,2)*m(1,0);
    Ct(2,2) = m(0,0)*m(1,1) - m(0,1)*m(1,0);
    return Ct;
}

inline Matrix3d AdjugateTransposed(const Matrix3d& m)
{
    /*
        Adjugate transposed is the cofactor matrix:
    */

    Matrix3d C;
    // Eigen::Matrix2d d;

    // // (iii, jjj) => 3D matrix index
    // // (ii, jj) => 2D matrix index
    // // (i, j) => 3D Cofactor matrix index, also the row and column to skip
    // for(size_t i = 0 ; i < 3 ; ++i) {
    //     for(size_t j = 0 ; j < 3 ; ++j) {
    //         for(size_t ii = 0, iii = 0 ; ii < 2 ; ++iii) {
    //             if(iii == i) continue;
    //             for(size_t jj = 0, jjj = 0 ; jj < 2 ; ++jjj) {
    //                 if(jjj == j) continue;
    //                 d(ii,jj) = m(iii,jjj);
    //                 jj++;
    //             }
    //             ii++;
    //         }
    //         C(i,j) = (i+j) % 2 == 0? d.determinant() : -d.determinant();
    //     }
    // }
    // 00 01 02
    // 10 11 12
    // 20 21 22
    C(0,0) = m(1,1)*m(2,2) - m(1,2)*m(2,1);
    C(0,1) = -m(1,0)*m(2,2) + m(1,2)*m(2,0);
    C(0,2) = m(1,0)*m(2,1) - m(1,1)*m(2,0);
    C(1,0) = -m(0,1)*m(2,2) + m(0,2)*m(2,1);
    C(1,1) = m(0,0)*m(2,2) - m(0,2)*m(2,0);
    C(1,2) = -m(0,0)*m(2,1) + m(0,1)*m(2,0);
    C(2,0) = m(0,1)*m(1,2) - m(0,2)*m(1,1);
    C(2,1) = -m(0,0)*m(1,2) + m(0,2)*m(1,0);
    C(2,2) = m(0,0)*m(1,1) - m(0,1)*m(1,0);
    return C;
}

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
    double kappa = S(1,1)*S(2,2) - S(1,2)*S(2,1) + S(0,0)*S(2,2) - S(0,2)*S(2,0) + S(0,0)*S(1,1) - S(0,1)*S(1,0); // Adjugate(S).trace();
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

Matrix3d Foam(const vector<Vector3d>& P, const vector<Vector3d>& Q, const vector<double>& w)
{
	Matrix3d B;
	const size_t N = P.size();
    B.setZero();
	for (size_t j = 0; j < N; ++j) {
		B.noalias() += (w[j] * Q[j]) * P[j].transpose();
	}
    double Bnorm = B.squaredNorm();
    double Bdet = 8 * B.determinant();
    double Badj = 4 * Adjugate(B).squaredNorm();
    // Newton's method: lambda_{n+1} = lambda_n - f(lambda) / f'(lambda)
    // f(lambda) = (lambda^2 - |B|^2_f)^2 - 8 lambda det(B) - 4 |Adjugate(B)|^2_F
    // f'(lambda) = 4 lambda (lambda^2 - |B|^2_f) - 8 det(B)
 	double lambda = 1.0;
	double old_lambda = 0.0;
	double kappa;
	while (fabs(old_lambda - lambda) > 1e-5) {
		old_lambda = lambda;
		kappa = lambda * lambda - Bnorm;
		lambda = lambda - ((kappa * kappa - lambda*Bdet - Badj) / (4.0 * lambda * kappa - Bdet));
	}
    kappa = 0.5 * (lambda * lambda - Bnorm);
    return ((kappa + Bnorm)*B + lambda*AdjugateTransposed(B) - B*B.transpose()*B) / (lambda*kappa - Bdet / 8.0);
}
