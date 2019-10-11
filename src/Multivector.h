
#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

#include "BasisBlade.h"

struct Multivector {
    /** Creates a new instance of Multivector */
    Multivector();
    /** do not modify 'B' for it is not copied */
    Multivector(const std::vector<BasisBlade>& B);
    /** do not modify 'B' for it is not copied */
    Multivector(std::vector<BasisBlade>&& B);
    /** do not modify 'B' for it is not copied */
    Multivector(const BasisBlade& B);
    /** default copy constructor */
    Multivector(const Multivector&) = default;
    Multivector& operator =(const Multivector&) = default;
    /** default copy constructor */
    Multivector(Multivector&&) = default;
    Multivector& operator =(Multivector&&) = default;

    bool equals(Multivector B) const;
    std::string toString() const;
    /**
     * @param bvNames The names of the basis vector (e1, e2, e3) are used when
     * not available
     */
    std::string toString(const std::vector<std::string>& bvNames) const;

    /** @return true if this is really 0.0 */
    bool isNull();
    /** @return true if norm_e2 < epsilon * epsilon*/
    bool isNull(double epsilon);
    /** @return true is this is a scalar (0.0 is also a scalar) */
    bool isScalar();
    /** @return simplification of this multivector (the same Multivector, but blades array can be changed) */
    Multivector simplify();
    /** @return the grade of this if homogeneous, -1 otherwise.
     * 0 is return for null Multivectors.
     */
    int grade() const;
    /** @return bitmap of grades that are in use in 'this'*/
    int gradeUsage() const;
    /** @return index of highest grade in use in 'this'*/
    int topGradeIndex() const;
    /** @return the largest grade part of this */
    Multivector largestGradePart();
    /** @return dimension of space this blade (apparently) lives in */
    int spaceDim() const;
    /**
     * Currently removes all basis blades with |scale| less than epsilon
     *
     * Old version did this:
     * Removes basis blades with whose |scale| is less than <code>epsilon * maxMag</code> where
     * maxMag is the |scale| of the largest basis blade.
     *
     * @return 'Compressed' version of this (the same Multivector, but blades array can be changed)
     */
    Multivector compress(double epsilon);

    /** shortcut to compress(1e-13) */
    Multivector compress();    
    const std::vector<BasisBlade>& getBlades() const;
    /** sorts the blade in 'blades' based on bitmap only */
    void sortBlades();

    // *!*HTML_TAG*!* storage
    /** list of basis blades */
    std::vector<BasisBlade> blades;

    /** when true, the blades have been sorted on bitmap */
    bool bladesSorted;

}; // end of class Multivector

    /** @return basis vector 'idx' range [0 ... dim)*/
Multivector getBasisVector(int idx);
/** @return geometric product of this with a scalar */
Multivector gp(const Multivector&, double a);
/** @return geometric product of this with a 'x' */
Multivector gp(const Multivector& a, const Multivector& x);
/** @return geometric product of this with a 'x' using metric 'm' */
Multivector gp(const Multivector& a, const Multivector& x, const std::vector<double>& m);
/** @return outer product of this with 'x' */
Multivector op(const Multivector& a, const Multivector& x);
/** @return inner product of this with a 'x'
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,
 * RIGHT_CONTRACTION,
 * HESTENES_INNER_PRODUCT or
 * MODIFIED_HESTENES_INNER_PRODUCT.
 */
Multivector ip(const Multivector& a, const Multivector& x, InnerProductTypes type);
/** @return inner product of this with a 'x' using metric 'm'
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,
 * RIGHT_CONTRACTION,
 * HESTENES_INNER_PRODUCT or
 * MODIFIED_HESTENES_INNER_PRODUCT.
 */
Multivector ip(const Multivector& a, const Multivector& x, const std::vector<double>& m, InnerProductTypes type);
// *!*HTML_TAG*!* add
/** @return sum of this with scalar 'a' */
Multivector add(const Multivector& x, double a);
/** @return sum of this with 'x' */
Multivector add(const Multivector& a, const Multivector& x);
/** @return this - scalar 'a' */
Multivector subtract(const Multivector& x, double a);
/** @return this - 'x' */
Multivector subtract(const Multivector& a, const Multivector& x);
/** @return exponential of this */
Multivector exp(const Multivector& b);
/** @return exponential of this under metric 'm' */
Multivector exp(const Multivector& b, const std::vector<double>& m);
/** evaluates exp(this) using special cases if possible, using series otherwise */
Multivector exp(const Multivector& b, const std::vector<double>& M, int order);
/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under Euclidean norm
 */
Multivector unit_e(const Multivector& a);
double norm_e(const Multivector& a);
double norm_e2(const Multivector& a);
/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under 'reverse' norm (this / sqrt(abs(this.reverse(this))))
 */
Multivector unit_r(const Multivector& a);
/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under 'reverse' norm (this / sqrt(abs(this.reverse(this))))
 */
Multivector unit_r(const Multivector& a, const std::vector<double>& m);
/** @return reverse of this */
Multivector reverse(const Multivector& a);
/** @return grade inversion of this */
Multivector gradeInversion(const Multivector& a);
/** @return clifford conjugate of this */
Multivector cliffordConjugate(const Multivector& a);
/**
 * Extracts grade(s) 'G' from this multivector.
 * @return a new multivector of grade(s) 'G'
 */
Multivector extractGrade(const Multivector& a, const std::vector<int>& G);
Multivector dual(const Multivector& a, int dim);
Multivector dual(const Multivector& a, const std::vector<double>& m);
double scalarPart(const Multivector& a);
/**
 * Can throw java.lang.ArithmeticException if versor is not invertible
 * @return inverse of this (assuming it is a versor, no check is made!)
 */
Multivector versorInverse(const Multivector& a);
/**
 * Can throw java.lang.ArithmeticException if versor is not invertible
 * @return inverse of this (assuming it is a versor, no check is made!)
 */
Multivector versorInverse(const Multivector& a, const std::vector<double>& m);
/**
 * Can throw java.lang.ArithmeticException if blade is not invertible
 * @return inverse of arbitrary multivector.
 *
 */
Multivector generalInverse(const Multivector& a, const std::vector<double>& metric);

/** @return scalar product of this with a 'x' */
inline double scalarProduct(const Multivector& a, const Multivector& x) { return scalarPart(ip(a, x, InnerProductTypes::LEFT_CONTRACTION)); }
/** @return scalar product of this with a 'x' using metric 'm' */
inline double scalarProduct(const Multivector& a, const Multivector& x, const std::vector<double>& m) { return scalarPart(ip(a, x, m, InnerProductTypes::LEFT_CONTRACTION)); }
/** shortcut to scalarProduct(...) */
inline double scp(const Multivector& a, const Multivector& x) { return scalarProduct(a, x); }
/** shortcut to scalarProduct(...) */
inline double scp(const Multivector& a, const Multivector& x, const std::vector<double>& m) { return scalarProduct(a, x, m); }
/**
 * Extracts grade 'g' from this multivector.
 * @return a new multivector of grade 'g'
 */
inline Multivector extractGrade(const Multivector& a, int g) { return extractGrade(a, std::vector<int>(1, g)); }

inline Multivector operator + (const Multivector& x, double a) { return add(x, a); }
inline Multivector operator + (double a, const Multivector& x) { return add(x, a); }
inline Multivector operator + (const Multivector& a, const Multivector& x) { return add(a, x); }
inline Multivector operator - (const Multivector& x, double a) { return add(x, -a); }
inline Multivector operator - (double a, const Multivector& x) { return add(gp(x, -1), a); }
inline Multivector operator - (const Multivector& a, const Multivector& x) { return subtract(a, x); }
inline Multivector operator * (const Multivector& x, double a) { return gp(x, a); }
inline Multivector operator * (double a, const Multivector& x) { return gp(x, a); }
inline Multivector operator * (const Multivector& a, const Multivector& x) { return gp(a, x); }
inline Multivector operator ^ (const Multivector& a, const Multivector& x) { return op(a, x); }
