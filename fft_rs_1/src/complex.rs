//! Complex number types for FFT computation.

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg};

// ---------------------------------------------------------------------------
// Complex32
// ---------------------------------------------------------------------------

/// A complex number with `f32` real and imaginary parts.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    /// Create a new `Complex32`.
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Complex32 { re, im }
    }

    /// Real zero, imaginary zero.
    #[inline]
    pub const fn zero() -> Self {
        Complex32 { re: 0.0, im: 0.0 }
    }

    /// Real one, imaginary zero.
    #[inline]
    pub const fn one() -> Self {
        Complex32 { re: 1.0, im: 0.0 }
    }

    /// Complex conjugate.
    #[inline]
    pub fn conjugate(self) -> Self {
        Complex32 { re: self.re, im: -self.im }
    }

    /// Euclidean magnitude (|z|).
    #[inline]
    pub fn magnitude(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Squared Euclidean magnitude (avoids sqrt).
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle in radians ∈ (-π, π].
    #[inline]
    pub fn phase(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Twiddle factor: e^{-2πi·k/N}  (negative exponent for forward FFT)
    ///
    /// Uses `sin_cos()` for improved performance (Proposal 1).
    #[inline]
    pub fn twiddle(n: usize, k: usize) -> Self {
        let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        let (sin, cos) = angle.sin_cos();
        Complex32 { re: cos, im: sin }
    }

    /// Twiddle factor: e^{2πi·k/N}  (positive exponent for IFFT)
    ///
    /// Uses `sin_cos()` for improved performance (Proposal 1).
    #[inline]
    pub fn twiddle_inverse(n: usize, k: usize) -> Self {
        let angle = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
        let (sin, cos) = angle.sin_cos();
        Complex32 { re: cos, im: sin }
    }
}

impl fmt::Debug for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "({} + {}i)", self.re, self.im)
        } else {
            write!(f, "({} - {}i)", self.re, -self.im)
        }
    }
}

impl fmt::Display for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Default for Complex32 {
    fn default() -> Self {
        Self::zero()
    }
}

// -- Arithmetic -------------------------------------------------------------

impl Add for Complex32 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex32 { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for Complex32 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex32 { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Mul for Complex32 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex32 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div<f32> for Complex32 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Complex32 { re: self.re / rhs, im: self.im / rhs }
    }
}

impl Neg for Complex32 {
    type Output = Self;
    fn neg(self) -> Self {
        Complex32 { re: -self.re, im: -self.im }
    }
}

// ---------------------------------------------------------------------------
// Complex64
// ---------------------------------------------------------------------------

/// A complex number with `f64` real and imaginary parts.
#[derive(Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    /// Create a new `Complex64`.
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Complex64 { re, im }
    }

    /// Real zero, imaginary zero.
    #[inline]
    pub const fn zero() -> Self {
        Complex64 { re: 0.0, im: 0.0 }
    }

    /// Real one, imaginary zero.
    #[inline]
    pub const fn one() -> Self {
        Complex64 { re: 1.0, im: 0.0 }
    }

    /// Complex conjugate.
    #[inline]
    pub fn conjugate(self) -> Self {
        Complex64 { re: self.re, im: -self.im }
    }

    /// Euclidean magnitude (|z|).
    #[inline]
    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Squared Euclidean magnitude (avoids sqrt).
    #[inline]
    pub fn magnitude_squared(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle in radians ∈ (-π, π].
    #[inline]
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Twiddle factor: e^{-2πi·k/N}
    ///
    /// Uses `sin_cos()` for improved performance (Proposal 1).
    #[inline]
    pub fn twiddle(n: usize, k: usize) -> Self {
        let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (sin, cos) = angle.sin_cos();
        Complex64 { re: cos, im: sin }
    }

    /// Twiddle factor: e^{2πi·k/N}
    ///
    /// Uses `sin_cos()` for improved performance (Proposal 1).
    #[inline]
    pub fn twiddle_inverse(n: usize, k: usize) -> Self {
        let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let (sin, cos) = angle.sin_cos();
        Complex64 { re: cos, im: sin }
    }
}

impl fmt::Debug for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "({} + {}i)", self.re, self.im)
        } else {
            write!(f, "({} - {}i)", self.re, -self.im)
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl Default for Complex64 {
    fn default() -> Self {
        Self::zero()
    }
}

// -- Arithmetic -------------------------------------------------------------

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex64 { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex64 { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex64 {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Complex64 { re: self.re / rhs, im: self.im / rhs }
    }
}

impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Complex64 { re: -self.re, im: -self.im }
    }
}

// ---------------------------------------------------------------------------
// Approximate equality helpers for testing
// ---------------------------------------------------------------------------

/// Assert two `Complex32` are approximately equal.
#[cfg(test)]
pub fn assert_complex32_eq(a: Complex32, b: Complex32, eps: f32) {
    assert!((a.re - b.re).abs() < eps, "re: {} vs {}", a.re, b.re);
    assert!((a.im - b.im).abs() < eps, "im: {} vs {}", a.im, b.im);
}

/// Assert two `Complex64` are approximately equal.
#[cfg(test)]
pub fn assert_complex64_eq(a: Complex64, b: Complex64, eps: f64) {
    assert!((a.re - b.re).abs() < eps, "re: {} vs {}", a.re, b.re);
    assert!((a.im - b.im).abs() < eps, "im: {} vs {}", a.im, b.im);
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex32_arithmetic() {
        let a = Complex32::new(1.0, 2.0);
        let b = Complex32::new(3.0, -1.0);

        assert_complex32_eq(a + b, Complex32::new(4.0, 1.0), 1e-6);
        assert_complex32_eq(a - b, Complex32::new(-2.0, 3.0), 1e-6);
        // (1+2i)(3-1i) = 3 - 1i + 6i - 2i² = 5 + 5i
        assert_complex32_eq(a * b, Complex32::new(5.0, 5.0), 1e-6);
        assert_complex32_eq(-a, Complex32::new(-1.0, -2.0), 1e-6);
        assert_complex32_eq(a / 2.0, Complex32::new(0.5, 1.0), 1e-6);
    }

    #[test]
    fn test_complex64_arithmetic() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);

        assert_complex64_eq(a + b, Complex64::new(4.0, 1.0), 1e-12);
        assert_complex64_eq(a - b, Complex64::new(-2.0, 3.0), 1e-12);
        assert_complex64_eq(a * b, Complex64::new(5.0, 5.0), 1e-12);
        assert_complex64_eq(-a, Complex64::new(-1.0, -2.0), 1e-12);
        assert_complex64_eq(a / 2.0, Complex64::new(0.5, 1.0), 1e-12);
    }

    #[test]
    fn test_complex32_magnitude_phase() {
        let z = Complex32::new(3.0, 4.0);
        assert!((z.magnitude() - 5.0).abs() < 1e-6);
        assert_eq!(z.magnitude_squared(), 25.0);
    }

    #[test]
    fn test_complex64_magnitude_phase() {
        let z = Complex64::new(3.0, 4.0);
        assert!((z.magnitude() - 5.0).abs() < 1e-12);
        assert_eq!(z.magnitude_squared(), 25.0);
    }

    #[test]
    fn test_complex32_conjugate() {
        let z = Complex32::new(1.0, -2.0);
        assert_complex32_eq(z.conjugate(), Complex32::new(1.0, 2.0), 1e-6);
    }

    #[test]
    fn test_complex64_conjugate() {
        let z = Complex64::new(1.0, -2.0);
        assert_complex64_eq(z.conjugate(), Complex64::new(1.0, 2.0), 1e-12);
    }

    #[test]
    fn test_twiddle_factors() {
        // e^{-2πi·0/N} = 1
        assert_complex64_eq(Complex64::twiddle(8, 0), Complex64::one(), 1e-12);
        // e^{-2πi·N/4 / N} = -i
        assert_complex64_eq(Complex64::twiddle(8, 2), Complex64::new(0.0, -1.0), 1e-12);
        // e^{-2πi·N/2 / N} = -1
        assert_complex64_eq(Complex64::twiddle(8, 4), Complex64::new(-1.0, 0.0), 1e-12);
    }

    #[test]
    fn test_debug_display() {
        let z = Complex32::new(1.0, -2.0);
        let s = format!("{:?}", z);
        assert!(s.contains('+') || s.contains('-'));
    }
}