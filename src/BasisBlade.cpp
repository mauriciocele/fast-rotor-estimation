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
 * BasisBlade.java
 *
 * Created on February 1, 2005, 11:41 AM
 *
 * Copyright 2005-2007 Daniel Fontijne, University of Amsterdam
 * fontijne@science.uva.nl
 *
 */

#include "BasisBlade.h"

using std::vector;
using std::string;

namespace utils {
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

    /**
     * Rounds <code>what</code> to the nearest multiple X of <code>multipleOf</code>,
     * if |X - what| <= epsilon
     */
    static double round(double what, double multipleOf, double epsilon) {
        double a = what / multipleOf;
        double b = std::floor(a + 0.5) * multipleOf;
        return (std::abs((what - b)) <= epsilon) ? b : what;
    }
}; // namespace utils

// *!*HTML_TAG*!* minus_one_pow
/** @return pow(-1, i) */
static int minusOnePow(int i) {
  return ((i & 1) == 0) ? 1 : -1;
}

// *!*HTML_TAG*!* canonicalReorderingSign
/**
 * Returns sign change due to putting the blade blades represented
 * by <code>a<code> and <code>b</code> into canonical order.
 */
static double canonicalReorderingSign(int a, int b) {
  // Count the number of basis vector flips required to
  // get a and b into canonical order.
  a >>= 1;
  int sum = 0;
  while (a != 0) {
    sum += utils::bitCount(a & b);
    a >>= 1;
  }

  // even number of flips -> return 1
  // odd number of flips -> return 1
  return ((sum & 1) == 0) ? 1.0 : -1.0;
}

// *!*HTML_TAG*!* inner_product_filter
/**
 * Applies the rules to turn a geometric product into an inner product
 * @param ga Grade of argument 'a'
 * @param gb Grade of argument 'b'
 * @param r the basis blade to be filter
 * @param type the type of inner product required:
 * LEFT_CONTRACTION,RIGHT_CONTRACTION, HESTENES_INNER_PRODUCT or MODIFIED_HESTENES_INNER_PRODUCT
 * @return Either a 0 basis blade, or 'r'
 */
static BasisBlade innerProductFilter(int ga, int gb, const BasisBlade& r, InnerProductTypes type) {
  switch(type) {
    case InnerProductTypes::LEFT_CONTRACTION:
              if ((ga > gb) || (r.grade() != (gb-ga)))
                  return BasisBlade();
              else return r;
    case InnerProductTypes::RIGHT_CONTRACTION:
              if ((ga < gb) || (r.grade() != (ga-gb)))
                  return BasisBlade();
              else return r;
    case InnerProductTypes::HESTENES_INNER_PRODUCT:
              if ((ga == 0) || (gb == 0)) return BasisBlade();
              // drop through to MODIFIED_HESTENES_INNER_PRODUCT
    case InnerProductTypes::MODIFIED_HESTENES_INNER_PRODUCT:
              if (std::abs(ga - gb) == r.grade()) return r;
              else return BasisBlade();
    default:
        return BasisBlade(0);
  }
}

// *!*HTML_TAG*!* gp_op
/**
 * @return the geometric product or the outer product
 * of 'a' and 'b'.
 */
BasisBlade gp_op(const BasisBlade& a, const BasisBlade& b, bool outer) {
  // if outer product: check for independence
  if (outer && ((a.bitmap & b.bitmap) != 0))
    return BasisBlade(0.0);

  // compute the bitmap:
  int bitmap = a.bitmap ^ b.bitmap;

  // compute the sign change due to reordering:
  double sign = canonicalReorderingSign(a.bitmap, b.bitmap);

  // return result:
  return BasisBlade(bitmap, sign * a.scale * b.scale);
}

// *!*HTML_TAG*!* gp_restricted_NE_metric
/**
 * Computes the geometric product of two basis blades in limited non-Euclidean metric.
 * @param m is an array of doubles giving the metric for each basis vector.
 */
BasisBlade geometricProduct(const BasisBlade& a, const BasisBlade& b, const vector<double>& m) {
  // compute the geometric product in Euclidean metric:
  BasisBlade result = geometricProduct(a, b);

  // compute the meet (bitmap of annihilated vectors):
  int bitmap = a.bitmap & b.bitmap;

  // change the scale according to the metric:
  int i = 0;
  while (bitmap != 0) {
    if ((bitmap & 1) != 0) result.scale *= m[i];
    i++;
    bitmap = bitmap >> 1;
  }

  return result;
}

/**
 * @return the geometric product of two basis blades
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,RIGHT_CONTRACTION, HESTENES_INNER_PRODUCT or MODIFIED_HESTENES_INNER_PRODUCT.
 */
BasisBlade innerProduct(const BasisBlade& a, const BasisBlade& b, InnerProductTypes type) {
  return innerProductFilter(a.grade(), b.grade(), geometricProduct(a, b), type);
}

/**
 * Computes the inner product of two basis blades in limited non-Euclidean metric.
 * @param m is an array of doubles giving the metric for each basis vector.
 * @param type gives the type of inner product:
 * LEFT_CONTRACTION,RIGHT_CONTRACTION, HESTENES_INNER_PRODUCT or MODIFIED_HESTENES_INNER_PRODUCT.
 */
BasisBlade innerProduct(const BasisBlade& a, const BasisBlade& b, const vector<double>& m, InnerProductTypes type) {
  return innerProductFilter(a.grade(), b.grade(), geometricProduct(a, b, m), type);
}

/**
 * A simple class to represent a basis blade.
 *
 * <p>mutable :( Should have made it immutable . . .
 *
 * <p>Could use subspace.util.Bits.lowestOneBit() and such to make
 * loops slightly more efficient, but didn't to keep text simple for the book.
 *
 * @author  fontijne
 */
/** constructs an instance of BasisBlade */
BasisBlade::BasisBlade(int b, double s) {
  bitmap = b;
  scale = s;
}

/** constructs an instance of a unit BasisBlade */
BasisBlade::BasisBlade(int b) {
  bitmap = b;
  scale = 1.0;
}

/** constructs an instance of a scalar BasisBlade */
BasisBlade::BasisBlade(double s) {
  bitmap = 0;
  scale = s;
}

/** constructs an instance of a zero BasisBlade */
BasisBlade::BasisBlade() {
  bitmap = 0;
  scale = 0.0;
}

// *!*HTML_TAG*!* reverse
/**
 * @return reverse of this basis blade (always a newly constructed blade)
 */
BasisBlade BasisBlade::reverse() const {
  // multiplier = (-1)^(x(x-1)/2)
  return BasisBlade(bitmap, minusOnePow((grade() * (grade() - 1)) / 2) * scale);
}

// *!*HTML_TAG*!* grade_inversion
/**
 * @return grade inversion of this basis blade (always a newly constructed blade)
 */
BasisBlade BasisBlade::gradeInversion() const {
  // multiplier is (-1)^x
  return BasisBlade(bitmap, minusOnePow(grade()) * scale);
}

// *!*HTML_TAG*!* clifford_conjugate
/**
 * @return clifford conjugate of this basis blade (always a newly constructed blade)
 */
BasisBlade BasisBlade::cliffordConjugate() const {
  // multiplier is ((-1)^(x(x+1)/2)
  return BasisBlade(bitmap, minusOnePow((grade() * (grade() + 1)) / 2) * scale);
}

/** returns the grade of this blade */
int BasisBlade::grade() const {
  return utils::bitCount(bitmap);
}

bool BasisBlade::equals(const BasisBlade& B) const {
  return ((B.bitmap == bitmap) && (B.scale == scale));
}

/**
 * @param bvNames The names of the basis vector (e1, e2, e3) are used when
 * not available
 */
string BasisBlade::toString() const {
  vector<string> bvNames;
  return toString(bvNames);
}

/**
 * @param bvNames The names of the basis vector (e1, e2, e3) are used when
 * not available
 */
string BasisBlade::toString(const vector<string>& bvNames) const {
  string result;
  int i = 1;
  int b = bitmap;
  while (b != 0) {
    if ((b & 1) != 0) {
    if (result.length() > 0) result.append("^");
    if (i > bvNames.size())
      result.append("e" + std::to_string(i));
    else result.append(bvNames[i-1]);
    }
    b >>= 1;
    i++;
  }
  return (result.length() == 0) ? std::to_string(scale) : std::to_string(scale) + "*" + result;
}

/**
 * Rounds the scalar part of <code>this</code> to the nearest multiple X of <code>multipleOf</code>,
 * if |X - what| <= epsilon. This is useful when eigenbasis is used to perform products in arbitrary
 * metric, which leads to small roundof errors. You don't want to keep these roundof errors if your
 * are computing a multiplication table.
 *
 * @returns a new basis blade if a change is required.
 */
BasisBlade BasisBlade::round(double multipleOf, double epsilon) const {
  double a = utils::round(scale, multipleOf, epsilon);
  if (a != scale)
    return BasisBlade(bitmap, a);
  else return *this;
}
