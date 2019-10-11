// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

/*
 * Multivector.java
 *
 * Created on October 10, 2005, 7:37 PM
 *
 * Copyright 2005-2007 Daniel Fontijne, University of Amsterdam
 * fontijne@science.uva.nl
 *
 */

#include "Multivector.h"

using std::vector;
using std::string;

namespace {
    static void addToMatrix(Eigen::MatrixXd& M, const BasisBlade& alpha, const BasisBlade& beta, const BasisBlade& gamma) {
        // add gamma.scale to matrix entry M[gamma.bitmap, beta.bitmap]
        M(gamma.bitmap, beta.bitmap) += gamma.scale;
    }

    static void addToMatrix(Eigen::MatrixXd& M, const BasisBlade& alpha, const BasisBlade& beta, const vector<BasisBlade>& gamma) {
        for (int i = 0; i < gamma.size(); i++)
            addToMatrix(M, alpha, beta, gamma[i]);
    }

    // *!*HTML_TAG*!* bitCount
    /**
     * @return the number of 1 bits in <code>i</code>
     */
    static int bitCount(int i) {
        // Note that unsigned shifting (>>>) is not required.
        i = i - ((i >> 1) & 0x55555555);
        i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
        i = (i + (i >> 4)) & 0x0F0F0F0F;
        i = i + (i >> 8);
        i = i + (i >> 16);
        return i & 0x0000003F;
    }

    /** returns the number of 0 bits before the first 1 bit in <code>i</code>
     * <p>For example if i = 4 (100 binary), then 29 (31 - 2) is returned.
     */
    static int numberOfLeadingZeroBits(int i) {
        // Note that unsigned shifting (>>>) is not required.
        i = i | (i >> 1);
        i = i | (i >> 2);
        i = i | (i >> 4);
        i = i | (i >> 8);
        i = i | (i >>16);
        return bitCount(~i);
    }

    /** returns in the index [-1 31] of the highest bit that is on in <code>i</code> (-1 is returned if no bit is on at all (i == 0)) */
    static int highestOneBit(int i) {
        return 31 - numberOfLeadingZeroBits(i);
    }
    
    /** sorts by bitmap only */
    struct BladesComperator {
        int operator()(const BasisBlade& b1, const BasisBlade& b2) const {
            return b1.bitmap - b2.bitmap;
        }
    };

    /** simplifies list of basis blades; List is modified in the process */
    template<typename Collection>
    static Collection&& simplify(Collection&& L) {
        std::sort(L.begin(), L.end(), BladesComperator()); // sort by bitmap only
        BasisBlade* prevBlade = nullptr;
        bool removeNullBlades = false;
        for (auto I = L.begin(); I != L.end(); ) {
            BasisBlade& curBlade = *I;
            if (curBlade.scale == 0.0) {
                I = L.erase(I);
                prevBlade = nullptr;
            }
            else if ((prevBlade != nullptr) && (prevBlade->bitmap == curBlade.bitmap)) {
                prevBlade->scale += curBlade.scale;
                I = L.erase(I);
            }
            else {
                if ((prevBlade != nullptr) && (prevBlade->scale == 0.0))
                    removeNullBlades = true;
                prevBlade = &curBlade;
                ++I;
            }
        }

        if (removeNullBlades) {
            L.erase(std::remove_if(L.begin(), L.end(), [](BasisBlade& curBlade) { return curBlade.scale == 0.0; }), L.end());
        }

        return std::forward<Collection>(L);
    }

    /** For internal use; M can be null, Metric or double[] */
    Multivector _gp(const Multivector& a, const Multivector& x, const vector<double>& M) {
        if (M.empty()) return gp(a, x);
        else return gp(a, x, M);
    }

    /** For internal use; M can be null, Metric or double[] */
    double _scp(const Multivector& a, const Multivector& x, const vector<double>& M) {
        if (M.empty()) return scp(a, x);
        else return scp(a, x, M);
    }

    /** For internal use; M can be null, Metric or double[] */
    Multivector _versorInverse(const Multivector& a, const vector<double>& M) {
        if (M.empty()) return versorInverse(a);
        else return versorInverse(a, M);
    }
};

/** @return basis vector 'idx' range [0 ... dim)*/
Multivector getBasisVector(int idx) {
    return Multivector(BasisBlade(1 << idx));
}

/** @return geometric product of this with a scalar */
Multivector gp(const Multivector& x, double a) {
    if (a == 0.0) return Multivector();
    else {
        vector<BasisBlade> result = x.blades;
        for (BasisBlade& b : result) { b.scale *= a; }
        return Multivector(std::move(result));
    }
}

// *!*HTML_TAG*!* gp
/** @return geometric product of this with a 'x' */
Multivector gp(const Multivector& a, const Multivector& x) {
    vector<BasisBlade> result(a.blades.size() * x.blades.size());
    // loop over basis blade of 'this'
    int r = 0;
    for (const BasisBlade& B1 : a.blades) {
        // loop over basis blade of 'x'
        for (const BasisBlade& B2 : x.blades) {
            result[r++] = ::gp(B1, B2);
        }
    }
    return Multivector(simplify(std::move(result)));
}

/** @return geometric product of this with a 'x' using metric 'm' */
Multivector gp(const Multivector& a, const Multivector& x, const vector<double>& m) {
    vector<BasisBlade> result(a.blades.size() * x.blades.size());
    int r = 0;
    for (const BasisBlade& B1 : a.blades) {
        for (const BasisBlade& B2 : x.blades) {
            result[r++] = ::gp(B1, B2, m);
        }
    }
    return Multivector(simplify(std::move(result)));
}

// *!*HTML_TAG*!* op
/** @return outer product of this with 'x' */
Multivector op(const Multivector& a, const Multivector& x) {
    vector<BasisBlade> result(a.blades.size() * x.blades.size());
    int r = 0;
    for (const BasisBlade& B1 : a.blades) {
        for (const BasisBlade& B2 : x.blades) {
            result[r++] = ::op(B1, B2);
        }
    }
    return Multivector(simplify(std::move(result)));
}

/** @return inner product of this with a 'x'
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,
 * RIGHT_CONTRACTION,
 * HESTENES_INNER_PRODUCT or
 * MODIFIED_HESTENES_INNER_PRODUCT.
 */
Multivector ip(const Multivector& a, const Multivector& x, InnerProductTypes type) {
    vector<BasisBlade> result(a.blades.size() * x.blades.size());
    int r = 0;
    for (const BasisBlade& B1 : a.blades) {
        for (const BasisBlade& B2 : x.blades) {
            result[r++] = ::ip(B1, B2, type);
        }
    }
    return Multivector(simplify(std::move(result)));
}

/** @return inner product of this with a 'x' using metric 'm'
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,
 * RIGHT_CONTRACTION,
 * HESTENES_INNER_PRODUCT or
 * MODIFIED_HESTENES_INNER_PRODUCT.
 */
Multivector ip(const Multivector& a, const Multivector& x, const vector<double>& m, InnerProductTypes type) {
    vector<BasisBlade> result(a.blades.size() * x.blades.size());
    int r = 0;
    for (const BasisBlade& B1 : a.blades) {
        for (const BasisBlade& B2 : x.blades) {
            result[r++] = ::ip(B1, B2, m, type);
        }
    }
    return Multivector(simplify(std::move(result)));
}

// *!*HTML_TAG*!* add
/** @return sum of this with scalar 'a' */
Multivector add(const Multivector& x, double a) {
    vector<BasisBlade> result = x.blades;
    result.emplace_back(a);
    return Multivector(simplify(std::move(result)));
}

/** @return sum of this with 'x' */
Multivector add(const Multivector& a, const Multivector& x) {
    vector<BasisBlade> result(a.blades.size() + x.blades.size());
    int r = 0;
    for(const BasisBlade& b : a.blades) { result[r++] = b; }
    for(const BasisBlade& b : x.blades) { result[r++] = b; }
    return Multivector(simplify(std::move(result)));
}

/** @return this - scalar 'a' */
Multivector subtract(const Multivector& x, double a) {
    return add(x, -a);
}

// *!*HTML_TAG*!* substract
/** @return this - 'x' */
Multivector subtract(const Multivector& a, const Multivector& x) {
    vector<BasisBlade> result(a.blades.size() + x.blades.size());
    int r = 0;
    for(const BasisBlade& b : a.blades) { result[r++] = b; }
    for(const BasisBlade& b : x.blades) { result[r++] = BasisBlade(b.bitmap, -b.scale); }
    return Multivector(simplify(std::move(result)));
}

/** @return exponential of this */
Multivector exp(const Multivector& a) {
    return exp(a, vector<double>(), 12);
}
/** @return exponential of this under metric 'm' */
Multivector exp(const Multivector& a, const vector<double>& m) {
    return exp(a, m, 12);
}

// *!*HTML_TAG*!* expSeries
/** Evaluates exp using series . . .  (== SLOW & INPRECISE!) */
Multivector expSeries(const Multivector& a, const vector<double>& M, int order) {
    // first scale by power of 2 so that its norm is ~ 1
    long scale=1; {
        double max = norm_e(a);
        if (max > 1.0) scale <<= 1;
        while (max > 1.0) {
            max = max / 2;
            scale <<= 1;
        }
    }

    Multivector scaled = gp(a, 1.0 / scale);

    // taylor approximation
    Multivector result = Multivector(1.0); {
        Multivector tmp = Multivector(1.0);

        for (int i = 1; i < order; i++) {
            tmp = _gp(tmp, gp(scaled, 1.0 / i), M);
            result = add(result, tmp);
        }
    }

    // undo scaling
    while (scale > 1) {
        result = _gp(result, result, M);
        scale >>= 1;
    }

    return result;
}

// *!*HTML_TAG*!* exp
/** evaluates exp(this) using special cases if possible, using series otherwise */
Multivector exp(const Multivector& a, const vector<double>& M, int order) {
    // check out this^2 for special cases
    Multivector A2 = _gp(a, a, M).compress();
    if (A2.isNull(1e-8)) {
        // special case A^2 = 0
        return add(a, 1);
    }
    else if (A2.isScalar()) {
        double a2 = scalarPart(A2);
        // special case A^2 = +-alpha^2
        if (a2 < 0) {
            double alpha = std::sqrt(-a2);
            return add(gp(a, std::sin(alpha) / alpha), std::cos(alpha));
        }
        //hey: todo what if a2 == 0?
        else {
            double alpha = std::sqrt(a2);
            return add(gp(a, std::sinh(alpha) / alpha), std::cosh(alpha));
        }
    }
    else return expSeries(a, M, order);
}

/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under Euclidean norm
 */
Multivector unit_e(const Multivector& a) {
    return unit_r(a);
}

double norm_e(const Multivector& a) {
    double s = scp(a, reverse(a));
    if (s < 0.0) return 0.0; // avoid FP round off causing negative 's'
    else return std::sqrt(s);
}

double norm_e2(const Multivector& a) {
    double s = scp(a, reverse(a));
    if (s < 0.0) return 0.0; // avoid FP round off causing negative 's'
    return s;
}

/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under 'reverse' norm (this / sqrt(abs(this.reverse(this))))
 */
Multivector unit_r(const Multivector& a) {
    double s = scp(a, reverse(a));
    if (s == 0.0) throw std::runtime_error("null multivector");
    else return gp(a, 1 / std::sqrt(std::abs(s)));
}

/**
 * Can throw java.lang.ArithmeticException if multivector is null
 * @return unit under 'reverse' norm (this / sqrt(abs(this.reverse(this))))
 */
Multivector unit_r(const Multivector& a, const vector<double>& m) {
    double s = scp(a, reverse(a), m);
    if (s == 0.0) throw std::runtime_error("null multivector");
    else return gp(a, 1 / std::sqrt(std::abs(s)));
}

// *!*HTML_TAG*!* reverse
/** @return reverse of this */
Multivector reverse(const Multivector& a) {
    vector<BasisBlade> result(a.blades.size());
    // loop over all basis lades, reverse them, add to result
    int i = 0;
    for (const BasisBlade& b : a.blades)
        result[i++] = b.reverse();
    return Multivector(std::move(result));
}

// *!*HTML_TAG*!* grade_inversion
/** @return grade inversion of this */
Multivector gradeInversion(const Multivector& a) {
    vector<BasisBlade> result(a.blades.size());
    int i = 0;
    for (const BasisBlade& b : a.blades)
        result[i++] = b.gradeInversion();
    return Multivector(std::move(result));
}

// *!*HTML_TAG*!* clifford_conjugate
/** @return clifford conjugate of this */
Multivector cliffordConjugate(const Multivector& a) {
    vector<BasisBlade> result(a.blades.size());
    int i = 0;
    for (const BasisBlade& b : a.blades)
        result[i++] = b.cliffordConjugate();
    return Multivector(std::move(result));
}

/**
 * Extracts grade(s) 'G' from this multivector.
 * @return a new multivector of grade(s) 'G'
 */
Multivector extractGrade(const Multivector& a, const vector<int>& G) {
    // what is the maximum grade to be extracted?
    int maxGrade = *std::max_element(G.begin(), G.end());

    // create boolean array of what grade to keep
    vector<bool> keep(maxGrade + 1, false);
    for (int grade : G)
        keep[grade] = true;

    // extract the grade, store in result:
    vector<BasisBlade> result;
    for (const BasisBlade& b : a.blades) {
        int grade = b.grade();
        if (grade > maxGrade) continue;
        else if (keep[grade]) result.push_back(b);
    }

    return Multivector(std::move(result));
}

Multivector dual(const Multivector& a, int dim) {
    Multivector I = Multivector(BasisBlade((1 << dim)-1, 1.0));
    return ip(a, versorInverse(I), InnerProductTypes::LEFT_CONTRACTION);
}

Multivector dual(const Multivector& a, const vector<double>& m) {
    Multivector I = Multivector(BasisBlade((1 << m.size())-1, 1.0));
    return ip(a, versorInverse(I), m, InnerProductTypes::LEFT_CONTRACTION);
}

/** @return scalar part of 'this */
double scalarPart(const Multivector& a) {
    double s = 0.0;
    for (const BasisBlade& b : a.blades) {
        if (b.bitmap == 0) s += b.scale;
    }
    return s;
}

// *!*HTML_TAG*!* versor_inverse
/**
 * Can throw java.lang.ArithmeticException if versor is not invertible
 * @return inverse of this (assuming it is a versor, no check is made!)
 */
Multivector versorInverse(const Multivector& a) {
    Multivector R = reverse(a);
    double s = scp(a, R);
    if (s == 0.0) throw std::runtime_error("non-invertible multivector");
    return gp(R, 1.0 / s);
}

/**
 * Can throw java.lang.ArithmeticException if versor is not invertible
 * @return inverse of this (assuming it is a versor, no check is made!)
 */
Multivector versorInverse(const Multivector& a, const vector<double>& m) {
    Multivector R = reverse(a);
    double s = scp(a, R, m);
    if (s == 0.0) throw std::runtime_error("non-invertible multivector");
    return gp(R, 1.0 / s);
}

// *!*HTML_TAG*!* general_inverse
/**
 * Can throw java.lang.ArithmeticException if blade is not invertible
 * @return inverse of arbitrary multivector.
 *
 */
Multivector generalInverse(const Multivector& a, const vector<double>& metric) {
    int dim = a.spaceDim();

    Eigen::MatrixXd M(1 << dim, 1 << dim);

    // create all unit basis blades for 'dim'
    vector<BasisBlade> B(1 << dim);
    for (int i = 0; i < (1 << dim); i++)
        B[i] = BasisBlade(i);


    // construct a matrix 'M' such that matrix multiplication of 'M' with
    // the coordinates of another multivector 'x' (stored in a vector)
    // would result in the geometric product of 'M' and 'x'
    for (const BasisBlade& b : a.blades) {
        for (int j = 0; j < (1 << dim); j++) {
            if (metric.empty())
                addToMatrix(M, b, B[j], ::gp(b, B[j]));
            else
                addToMatrix(M, b, B[j], ::gp(b, B[j], metric));
        }
    }

    // try to invert matrix (can fail, then we throw an exception)
    Eigen::MatrixXd IM;
    IM = M.inverse();

    // reconstruct multivector from first column of matrix
    vector<BasisBlade> result;
    for (int j = 0; j < (1 << dim); j++) {
        double v = IM(j, 0);
        if (v != 0.0) {
            B[j].scale = v;
            result.push_back(B[j]);
        }
    }
    return Multivector(result);
}

/**
 * This class implements a sample multivector class along with
 * some basic GA operations. Very low performance.
 *
 * <p>mutable :( Should have made it immutable . . .
 * @author  fontijne
 */

/** Creates a new instance of Multivector */
Multivector::Multivector() {
    bladesSorted = false;
}

/** do not modify 'B' for it is not copied */
Multivector::Multivector(const vector<BasisBlade>& B) {
    blades = B;
    bladesSorted = false;
}

/** do not modify 'B' for it is not copied */
Multivector::Multivector(vector<BasisBlade>&& B) {
    blades = std::move(B);
    bladesSorted = false;
}

/** do not modify 'B' for it is not copied */
Multivector::Multivector(const BasisBlade& B) {
    blades.push_back(B);
    bladesSorted = false;
}

bool Multivector::equals(Multivector B) const {
    Multivector zero = subtract(*this, B);
    return (zero.blades.size() == 0);
}

string Multivector::toString() const {
    vector<string> bvNames;
    return toString(bvNames);
}

/**
 * @param bvNames The names of the basis vector (e1, e2, e3) are used when
 * not available
 */
string Multivector::toString(const vector<string>& bvNames) const {
    if (blades.size() == 0) return "0";
    else {
        string result;
        for (int i = 0; i < blades.size(); i++) {
            const BasisBlade& b = blades[i];
            string S = b.toString(bvNames);
            if (i == 0) result.append(S);
            else if (S[0] == '-') {
                result.append(" - ");
                result.append( S.substr(1, S.length() -1));
            }
            else {
                result.append(" + ");
                result.append(S);
            }
        }
        return result;
    }
}

/** @return true if this is really 0.0 */
bool Multivector::isNull() {
    simplify();
    return (blades.size() == 0);
}

/** @return true if norm_e2 < epsilon * epsilon*/
bool Multivector::isNull(double epsilon) {
    double s = norm_e2(*this);
    return (s < epsilon * epsilon);
}

/** @return true is this is a scalar (0.0 is also a scalar) */
bool Multivector::isScalar() {
    if (isNull()) return true;
    else if (blades.size() == 1) {
        return (blades[0].bitmap == 0);
    }
    else return false;
}

/** @return simplification of this multivector (the same Multivector, but blades array can be changed) */
Multivector Multivector::simplify() {
    ::simplify(blades);
    return *this;
}

/** @return the grade of this if homogeneous, -1 otherwise.
 * 0 is return for null Multivectors.
 */
int Multivector::grade() const {
    int g = -1;
    for (const BasisBlade& b : blades) {
        if (g < 0) g = b.grade();
        else if (g != b.grade()) return -1;
    }
    return (g < 0) ? 0 : g;
}


/** @return bitmap of grades that are in use in 'this'*/
int Multivector::gradeUsage() const {
    int gu = 0;
    for (const BasisBlade& b : blades) {
        gu |= 1 << b.grade();
    }
    return gu;
}

/** @return index of highest grade in use in 'this'*/
int Multivector::topGradeIndex() const {
    int maxG = 0;
    for (const BasisBlade& b : blades) {
        maxG = std::max(b.grade(), maxG);
    }
    return maxG;
}

/** @return the largest grade part of this */
Multivector Multivector::largestGradePart() {
    simplify();

    Multivector maxGP;
    double maxNorm = -1.0;
    int gu = gradeUsage();
    int topGradeIdx = topGradeIndex();
    for (int i = 0; i <= topGradeIdx; i++) {
        if ((gu & (1 << i)) == 0) continue;
        Multivector GP = extractGrade(*this, i);
        double n = norm_e(GP);
        if (n > maxNorm) {
            maxGP = GP;
            maxNorm = n;
        }
    }

    return maxGP;
}

/** @return dimension of space this blade (apparently) lives in */
int Multivector::spaceDim() const {
    int maxD = 0;
    for (const BasisBlade& b : blades) {
        maxD = std::max(highestOneBit(b.bitmap), maxD);
    }
    return maxD+1;
}

/**
 * Currently removes all basis blades with |scale| less than epsilon
 *
 * Old version did this:
 * Removes basis blades with whose |scale| is less than <code>epsilon * maxMag</code> where
 * maxMag is the |scale| of the largest basis blade.
 *
 * @return 'Compressed' version of this (the same Multivector, but blades array can be changed)
 */
Multivector Multivector::compress(double epsilon) {
    simplify();

    // find maximum magnitude:
    double maxMag = 0.0;
    for (BasisBlade &b : blades) {
        maxMag = std::max(std::abs(b.scale), maxMag);
    }
    if (maxMag == 0.0) {
        blades.clear();
    }
    else {
        // premultiply maxMag
        maxMag = epsilon; // used to read *=

        blades.erase(std::remove_if(blades.begin(), blades.end(), [=](BasisBlade& b) { return std::abs(b.scale) < maxMag; }), blades.end());
    }
    return *this;
}


/** shortcut to compress(1e-13) */
Multivector Multivector::compress() {
    return compress(1e-13);
}

const vector<BasisBlade>& Multivector::getBlades() const {
    return blades;
}

/** sorts the blade in 'blades' based on bitmap only */
void Multivector::sortBlades() {
    if (bladesSorted) return;
    else {
        std::sort(blades.begin(), blades.end(), BladesComperator());
        bladesSorted = true;
    }
}
