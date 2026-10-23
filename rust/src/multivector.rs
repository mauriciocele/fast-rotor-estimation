//! Minimal port of Daniel Fontijne's reference BasisBlade/Multivector classes
//! (Copyright 2005-2007 Daniel Fontijne, University of Amsterdam), limited to
//! the Euclidean operations the GA rotor estimator uses. Very low performance:
//! it is a reference implementation, not a fast one.

use std::ops::{Add, BitXor, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InnerProductType {
    LeftContraction,
    RightContraction,
    HestenesInnerProduct,
    ModifiedHestenesInnerProduct,
}

/// A scaled basis blade; `bitmap` specifies which basis vectors are present.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BasisBlade {
    pub bitmap: u32,
    pub scale: f64,
}

#[inline]
fn minus_one_pow(i: u32) -> f64 {
    if i & 1 == 0 { 1.0 } else { -1.0 }
}

/// Sign change due to putting the blades represented by `a` and `b` into canonical order.
#[inline]
fn canonical_reordering_sign(mut a: u32, b: u32) -> f64 {
    // Count the number of basis vector flips required to
    // get a and b into canonical order.
    a >>= 1;
    let mut sum = 0;
    while a != 0 {
        sum += (a & b).count_ones();
        a >>= 1;
    }
    // even number of flips -> return 1
    // odd number of flips -> return -1
    if sum & 1 == 0 { 1.0 } else { -1.0 }
}

impl BasisBlade {
    pub fn new(bitmap: u32, scale: f64) -> Self {
        Self { bitmap, scale }
    }

    pub fn scalar(scale: f64) -> Self {
        Self { bitmap: 0, scale }
    }

    pub fn grade(&self) -> u32 {
        self.bitmap.count_ones()
    }

    pub fn reverse(&self) -> Self {
        // multiplier = (-1)^(x(x-1)/2)
        let g = self.grade();
        Self::new(self.bitmap, minus_one_pow(g * g.saturating_sub(1) / 2) * self.scale)
    }

    /// Geometric product (`outer == false`) or outer product (`outer == true`).
    pub fn gp_op(a: &Self, b: &Self, outer: bool) -> Self {
        // if outer product: check for independence
        if outer && (a.bitmap & b.bitmap) != 0 {
            return Self::scalar(0.0);
        }
        let sign = canonical_reordering_sign(a.bitmap, b.bitmap);
        Self::new(a.bitmap ^ b.bitmap, sign * a.scale * b.scale)
    }

    /// Applies the rules to turn a geometric product into an inner product.
    fn inner_product_filter(ga: u32, gb: u32, r: Self, kind: InnerProductType) -> Self {
        let zero = Self::scalar(0.0);
        match kind {
            InnerProductType::LeftContraction => {
                if ga > gb || r.grade() != gb - ga {
                    zero
                } else {
                    r
                }
            }
            InnerProductType::RightContraction => {
                if ga < gb || r.grade() != ga - gb {
                    zero
                } else {
                    r
                }
            }
            InnerProductType::HestenesInnerProduct | InnerProductType::ModifiedHestenesInnerProduct => {
                if kind == InnerProductType::HestenesInnerProduct && (ga == 0 || gb == 0) {
                    zero
                } else if ga.abs_diff(gb) == r.grade() {
                    r
                } else {
                    zero
                }
            }
        }
    }

    pub fn inner_product(a: &Self, b: &Self, kind: InnerProductType) -> Self {
        Self::inner_product_filter(a.grade(), b.grade(), Self::gp_op(a, b, false), kind)
    }

    pub fn to_string_with(&self, bv_names: &[&str]) -> String {
        let mut result = String::new();
        let mut b = self.bitmap;
        let mut i = 1usize;
        while b != 0 {
            if b & 1 != 0 {
                if !result.is_empty() {
                    result.push('^');
                }
                if i > bv_names.len() {
                    result.push_str(&format!("e{i}"));
                } else {
                    result.push_str(bv_names[i - 1]);
                }
            }
            b >>= 1;
            i += 1;
        }
        // std::to_string(double) prints with "%f"
        if result.is_empty() {
            format!("{:.6}", self.scale)
        } else {
            format!("{:.6}*{}", self.scale, result)
        }
    }
}

/// A sum of basis blades.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Multivector {
    pub blades: Vec<BasisBlade>,
}

/// Sorts by bitmap, merges blades with equal bitmaps and drops zero blades.
fn simplify(mut l: Vec<BasisBlade>) -> Vec<BasisBlade> {
    l.sort_by_key(|b| b.bitmap);
    let mut out: Vec<BasisBlade> = Vec::with_capacity(l.len());
    for b in l {
        if b.scale == 0.0 {
            continue;
        }
        match out.last_mut() {
            Some(prev) if prev.bitmap == b.bitmap => prev.scale += b.scale,
            _ => out.push(b),
        }
    }
    out.retain(|b| b.scale != 0.0);
    out
}

impl Multivector {
    pub fn from_blade(b: BasisBlade) -> Self {
        Self { blades: vec![b] }
    }

    fn from_simplified(blades: Vec<BasisBlade>) -> Self {
        Self {
            blades: simplify(blades),
        }
    }

    /// Basis vector `idx` in the range [0 ... dim).
    pub fn basis_vector(idx: u32) -> Self {
        Self::from_blade(BasisBlade::new(1 << idx, 1.0))
    }

    pub fn scalar(s: f64) -> Self {
        Self::from_blade(BasisBlade::scalar(s))
    }

    fn product(&self, x: &Self, f: impl Fn(&BasisBlade, &BasisBlade) -> BasisBlade) -> Self {
        let mut result = Vec::with_capacity(self.blades.len() * x.blades.len());
        for b1 in &self.blades {
            for b2 in &x.blades {
                result.push(f(b1, b2));
            }
        }
        Self::from_simplified(result)
    }

    /// Geometric product with a scalar.
    pub fn gp_scalar(&self, a: f64) -> Self {
        if a == 0.0 {
            Self::default()
        } else {
            Self {
                blades: self
                    .blades
                    .iter()
                    .map(|b| BasisBlade::new(b.bitmap, b.scale * a))
                    .collect(),
            }
        }
    }

    pub fn gp(&self, x: &Self) -> Self {
        self.product(x, |a, b| BasisBlade::gp_op(a, b, false))
    }

    pub fn op(&self, x: &Self) -> Self {
        self.product(x, |a, b| BasisBlade::gp_op(a, b, true))
    }

    pub fn ip(&self, x: &Self, kind: InnerProductType) -> Self {
        self.product(x, |a, b| BasisBlade::inner_product(a, b, kind))
    }

    pub fn add(&self, x: &Self) -> Self {
        Self::from_simplified(self.blades.iter().chain(&x.blades).copied().collect())
    }

    pub fn subtract(&self, x: &Self) -> Self {
        let neg = x.blades.iter().map(|b| BasisBlade::new(b.bitmap, -b.scale));
        Self::from_simplified(self.blades.iter().copied().chain(neg).collect())
    }

    pub fn reverse(&self) -> Self {
        Self {
            blades: self.blades.iter().map(BasisBlade::reverse).collect(),
        }
    }

    pub fn scalar_part(&self) -> f64 {
        self.blades.iter().filter(|b| b.bitmap == 0).map(|b| b.scale).sum()
    }

    /// Inverse of a versor (no check is made that `self` is a versor).
    pub fn versor_inverse(&self) -> Self {
        let r = self.reverse();
        let s = scp(self, &r);
        assert!(s != 0.0, "non-invertible multivector");
        r.gp_scalar(1.0 / s)
    }

    pub fn dual(&self, dim: u32) -> Self {
        let i = Self::from_blade(BasisBlade::new((1 << dim) - 1, 1.0));
        self.ip(&i.versor_inverse(), InnerProductType::LeftContraction)
    }

    pub fn to_string_with(&self, bv_names: &[&str]) -> String {
        if self.blades.is_empty() {
            return "0".to_string();
        }
        let mut result = String::new();
        for (i, b) in self.blades.iter().enumerate() {
            let s = b.to_string_with(bv_names);
            if i == 0 {
                result.push_str(&s);
            } else if let Some(rest) = s.strip_prefix('-') {
                result.push_str(" - ");
                result.push_str(rest);
            } else {
                result.push_str(" + ");
                result.push_str(&s);
            }
        }
        result
    }
}

impl std::fmt::Display for Multivector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_string_with(&[]))
    }
}

/// Scalar product (scalar part of the left contraction).
pub fn scp(a: &Multivector, x: &Multivector) -> f64 {
    a.ip(x, InnerProductType::LeftContraction).scalar_part()
}

// Operators, for owned values and references: `+` add, `-` subtract,
// `*` geometric product (also with f64), `^` outer product.
macro_rules! binop {
    ($tr:ident, $f:ident, $m:ident) => {
        impl $tr<&Multivector> for &Multivector {
            type Output = Multivector;
            fn $f(self, x: &Multivector) -> Multivector {
                Multivector::$m(self, x)
            }
        }
        impl $tr<Multivector> for &Multivector {
            type Output = Multivector;
            fn $f(self, x: Multivector) -> Multivector {
                Multivector::$m(self, &x)
            }
        }
        impl $tr<&Multivector> for Multivector {
            type Output = Multivector;
            fn $f(self, x: &Multivector) -> Multivector {
                Multivector::$m(&self, x)
            }
        }
        impl $tr<Multivector> for Multivector {
            type Output = Multivector;
            fn $f(self, x: Multivector) -> Multivector {
                Multivector::$m(&self, &x)
            }
        }
    };
}
binop!(Add, add, add);
binop!(Sub, sub, subtract);
binop!(Mul, mul, gp);
binop!(BitXor, bitxor, op);

impl Mul<&Multivector> for f64 {
    type Output = Multivector;
    fn mul(self, x: &Multivector) -> Multivector {
        x.gp_scalar(self)
    }
}

impl Mul<Multivector> for f64 {
    type Output = Multivector;
    fn mul(self, x: Multivector) -> Multivector {
        x.gp_scalar(self)
    }
}
