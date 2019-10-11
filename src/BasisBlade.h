#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

enum class InnerProductTypes
{
    LEFT_CONTRACTION = 1,
    RIGHT_CONTRACTION = 2,
    HESTENES_INNER_PRODUCT = 3,
    MODIFIED_HESTENES_INNER_PRODUCT = 4
};

struct BasisBlade {
    /** constructs an instance of BasisBlade */
    BasisBlade(int b, double s);
    /** constructs an instance of a unit BasisBlade */
    BasisBlade(int b);
    /** constructs an instance of a scalar BasisBlade */
    BasisBlade(double s);
    /** constructs an instance of a zero BasisBlade */
    BasisBlade();
    /** default copy constructor */
    BasisBlade(const BasisBlade&) = default;
    BasisBlade& operator =(const BasisBlade&) = default;
    // *!*HTML_TAG*!* reverse
    /**
     * @return reverse of this basis blade (always a newly constructed blade)
     */
    BasisBlade reverse() const;
    // *!*HTML_TAG*!* grade_inversion
    /**
     * @return grade inversion of this basis blade (always a newly constructed blade)
     */
    BasisBlade gradeInversion() const;
    // *!*HTML_TAG*!* clifford_conjugate
    /**
     * @return clifford conjugate of this basis blade (always a newly constructed blade)
     */
    BasisBlade cliffordConjugate() const;
    /** returns the grade of this blade */
    int grade() const;
    /** returns true if equals otherwise false */
    bool equals(const BasisBlade& B) const;
    /**
     * @param bvNames The names of the basis vector (e1, e2, e3) are used when
     * not available
     */
    std::string toString() const;
    /**
     * @param bvNames The names of the basis vector (e1, e2, e3) are used when
     * not available
     */
    std::string toString(const std::vector<std::string>& bvNames) const;
    /**
     * Rounds the scalar part of <code>this</code> to the nearest multiple X of <code>multipleOf</code>,
     * if |X - what| <= epsilon. This is useful when eigenbasis is used to perform products in arbitrary
     * metric, which leads to small roundof errors. You don't want to keep these roundof errors if your
     * are computing a multiplication table.
     *
     * @returns a new basis blade if a change is required.
     */
    BasisBlade round(double multipleOf, double epsilon) const;
    // *!*HTML_TAG*!* storage
    /**
     * This bitmap specifies what basis vectors are
     * present in this basis blade
     */
    int bitmap;

    /**
     * The scale of the basis blade is represented by this double
     */
    double scale;
}; // end of class BasisBlade


// *!*HTML_TAG*!* gp_op
/**
 * @return the geometric product or the outer product
 * of 'a' and 'b'.
 */
BasisBlade gp_op(const BasisBlade& a, const BasisBlade& b, bool outer);
// *!*HTML_TAG*!* gp_restricted_NE_metric
/**
 * Computes the geometric product of two basis blades in limited non-Euclidean metric.
 * @param m is an array of doubles giving the metric for each basis vector.
 */
BasisBlade geometricProduct(const BasisBlade& a, const BasisBlade& b, const std::vector<double>& m);
/**
 * @return the geometric product of two basis blades
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,RIGHT_CONTRACTION, HESTENES_INNER_PRODUCT or MODIFIED_HESTENES_INNER_PRODUCT.
 */
BasisBlade innerProduct(const BasisBlade& a, const BasisBlade& b, InnerProductTypes type);
/**
 * Computes the inner product of two basis blades in limited non-Euclidean metric.
 * @param m is an array of doubles giving the metric for each basis vector.
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,RIGHT_CONTRACTION, HESTENES_INNER_PRODUCT or MODIFIED_HESTENES_INNER_PRODUCT.
 */
BasisBlade innerProduct(const BasisBlade& a, const BasisBlade& b, const std::vector<double>& m, InnerProductTypes type);

/** shortcut to outerProduct() */
inline BasisBlade op(const BasisBlade& a, const BasisBlade& b) { return gp_op(a, b, true); }

/** @return the outer product of two basis blades */
inline BasisBlade outerProduct(const BasisBlade& a, const BasisBlade& b) { return gp_op(a, b, true); }

/** shortcut to geometricProduct() */
inline BasisBlade gp(const BasisBlade& a, const BasisBlade& b) { return gp_op(a, b, false); }

/** return the geometric product of two basis blades */
inline BasisBlade geometricProduct(const BasisBlade& a, const BasisBlade& b) { return gp_op(a, b, false); }

/** shortcut to geometricProduct() */
inline BasisBlade gp(const BasisBlade& a, const BasisBlade& b, const std::vector<double>& m) { return geometricProduct(a, b, m); }

/** shortcut to innerProduct(...) */
inline BasisBlade ip(const BasisBlade& a, const BasisBlade& b, InnerProductTypes type) { return innerProduct(a, b, type); }

/** shortcut to innerProduct(...) */
inline BasisBlade ip(const BasisBlade& a, const BasisBlade& b, const std::vector<double>& m, InnerProductTypes type) { return innerProduct(a, b, m, type); }
