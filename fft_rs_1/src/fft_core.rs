//! Core FFT / IFFT implementation using the Cooley–Tukey radix-2 DIT algorithm.

use crate::complex::{Complex32, Complex64};
use crate::error::{validate_length, FftResult};

// ---------------------------------------------------------------------------
// Trait: convert a real sample into a complex number
// ---------------------------------------------------------------------------

/// Trait for types that can be converted into a complex sample.
pub trait IntoSample: Copy + Clone {
    /// The complex type produced.
    type Complex: ComplexSample;

    /// Convert a real value to complex (imaginary = 0).
    fn into_complex(self) -> Self::Complex;
}

// Make IntoSample object-safe for external use — the trait and its method are already pub.
// The method `into_complex` is callable because the trait is pub and the method is pub by default.

/// Trait for the complex output types.
pub trait ComplexSample: Clone + Copy + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn twiddle(n: usize, k: usize) -> Self;
    fn twiddle_inverse(n: usize, k: usize) -> Self;
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn neg(self) -> Self;
    fn div_scalar(self, s: Self::Scalar) -> Self;
    fn scalar_from_usize(n: usize) -> Self::Scalar;
    type Scalar: Copy;
}

impl ComplexSample for Complex32 {
    type Scalar = f32; // f32 is Copy, so this is fine
    fn zero() -> Self { Self::zero() }
    fn one() -> Self { Self::one() }
    fn twiddle(n: usize, k: usize) -> Self { Self::twiddle(n, k) }
    fn twiddle_inverse(n: usize, k: usize) -> Self { Self::twiddle_inverse(n, k) }
    fn add(self, rhs: Self) -> Self { self + rhs }
    fn sub(self, rhs: Self) -> Self { self - rhs }
    fn mul(self, rhs: Self) -> Self { self * rhs }
    fn neg(self) -> Self { -self }
    fn div_scalar(self, s: f32) -> Self { self / s }
    fn scalar_from_usize(n: usize) -> f32 { n as f32 }
}

impl ComplexSample for Complex64 {
    type Scalar = f64; // f64 is Copy, so this is fine
    fn zero() -> Self { Self::zero() }
    fn one() -> Self { Self::one() }
    fn twiddle(n: usize, k: usize) -> Self { Self::twiddle(n, k) }
    fn twiddle_inverse(n: usize, k: usize) -> Self { Self::twiddle_inverse(n, k) }
    fn add(self, rhs: Self) -> Self { self + rhs }
    fn sub(self, rhs: Self) -> Self { self - rhs }
    fn mul(self, rhs: Self) -> Self { self * rhs }
    fn neg(self) -> Self { -self }
    fn div_scalar(self, s: f64) -> Self { self / s }
    fn scalar_from_usize(n: usize) -> f64 { n as f64 }
}

impl IntoSample for i32 {
    type Complex = Complex32;
    fn into_complex(self) -> Complex32 {
        Complex32::new(self as f32, 0.0)
    }
}

impl IntoSample for i64 {
    type Complex = Complex64;
    fn into_complex(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
}

impl IntoSample for f32 {
    type Complex = Complex32;
    fn into_complex(self) -> Complex32 {
        Complex32::new(self, 0.0)
    }
}

impl IntoSample for f64 {
    type Complex = Complex64;
    fn into_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

// ---------------------------------------------------------------------------
// Bit-reversal utility
// ---------------------------------------------------------------------------

/// Bit-reverse `x` using `usize::reverse_bits()` (Proposal 2).
///
/// This reduces the overall permutation from O(N log N) to O(N) by leveraging
/// a single CPU instruction (`RBIT` on ARM, `BSWAP`-based on x86).
#[inline]
fn bit_reverse(x: usize, log2n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - log2n)
}

fn bit_reverse_permute<C: ComplexSample>(data: &mut [C], n: usize, log2n: usize) {
    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if i < j {
            data.swap(i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Iterative Cooley-Tukey FFT (forward)
// ---------------------------------------------------------------------------

/// Forward FFT with twiddle factors computed on-the-fly (Proposal 4, Approach A).
///
/// By computing twiddle factors directly in the innermost loop instead of
/// pre-allocating a `Vec`, we eliminate O(log N) heap allocations per FFT call.
fn fft_forward<C: ComplexSample>(data: &mut [C], n: usize, log2n: usize) {
    bit_reverse_permute(data, n, log2n);

    let mut len = 2;
    for _ in 0..log2n {
        let half = len >> 1;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let even_idx = start + k;
                let odd_idx = even_idx + half;
                let t = C::mul(C::twiddle(len, k), data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Iterative Cooley-Tukey IFFT
// ---------------------------------------------------------------------------

/// Inverse FFT with twiddle factors computed on-the-fly (Proposal 4, Approach A).
fn fft_inverse<C: ComplexSample>(data: &mut [C], n: usize, log2n: usize) {
    bit_reverse_permute(data, n, log2n);

    let mut len = 2;
    for _ in 0..log2n {
        let half = len >> 1;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let even_idx = start + k;
                let odd_idx = even_idx + half;
                let t = C::mul(C::twiddle_inverse(len, k), data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }

    let norm = C::scalar_from_usize(n);
    for i in 0..n {
        data[i] = C::div_scalar(data[i], norm); // f32/f64 are Copy so this is fine
    }
}

// ---------------------------------------------------------------------------
// Public API: FFT<T>
// ---------------------------------------------------------------------------

/// FFT (and IFFT) computation handle.
///
/// Accepts `i32`, `i64`, `f32`, or `f64` input. The `compute()` method returns
/// the forward DFT; `compute_inverse()` returns the inverse DFT.
///
/// # Example
///
/// ```
/// use fft_rs::FFT;
///
/// let input = vec![1.0f64, 2.0, 3.0, 4.0];
/// let fft = FFT::new(input).unwrap();
/// let spectrum = fft.compute();
/// ```
pub struct FFT<T: IntoSample> {
    data: Vec<T>,
}

impl<T: IntoSample> FFT<T>
where
    T::Complex: ComplexSample,
{
    /// Create a new `FFT` from a `Vec<T>`.
    ///
    /// Returns `Err` if the length is not a positive power of 2 or exceeds 2^24.
    pub fn new(data: Vec<T>) -> FftResult<Self> {
        validate_length(data.len())?;
        Ok(FFT { data })
    }

    /// Create a new `FFT` by cloning a slice.
    pub fn from_slice(slice: &[T]) -> FftResult<Self> {
        Self::new(slice.to_vec())
    }

    /// Return the number of samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if there are no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the input data.
    pub fn input(&self) -> &[T] {
        &self.data
    }

    /// Compute the forward FFT, returning a `Vec<T::Complex>`.
    pub fn compute(&self) -> Vec<T::Complex> {
        let n = self.data.len();
        if n == 1 {
            return vec![self.data[0].into_complex()];
        }
        let log2n = n.trailing_zeros() as usize;
        let mut buf: Vec<T::Complex> = self.data.iter().copied().map(|s| s.into_complex()).collect();
        fft_forward(&mut buf, n, log2n);
        buf
    }

    /// Compute the forward FFT (same as `compute()`).
    pub fn compute_inplace(&mut self) -> Vec<T::Complex> {
        self.compute()
    }

    /// Compute the inverse FFT of the given complex data.
    pub fn ifft(data: Vec<T::Complex>) -> Vec<T::Complex> {
        let n = data.len();
        validate_length(n).expect("IFFT length must be a positive power of 2");
        if n == 1 {
            return data;
        }
        let log2n = n.trailing_zeros() as usize;
        let mut buf = data;
        fft_inverse(&mut buf, n, log2n);
        buf
    }

    /// Compute the inverse FFT of the previously computed spectrum.
    pub fn compute_inverse(&self, spectrum: &[T::Complex]) -> Vec<T::Complex> {
        Self::ifft(spectrum.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::{assert_complex32_eq, assert_complex64_eq};
    use crate::error::FftError;

    #[test]
    fn test_fft4_f64() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        assert_complex64_eq(out[0], Complex64::new(10.0, 0.0), 1e-10);
        assert_complex64_eq(out[1], Complex64::new(-2.0, 2.0), 1e-10);
        assert_complex64_eq(out[2], Complex64::new(-2.0, 0.0), 1e-10);
        assert_complex64_eq(out[3], Complex64::new(-2.0, -2.0), 1e-10);
    }

    #[test]
    fn test_fft4_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        assert_complex32_eq(out[0], Complex32::new(10.0, 0.0), 1e-5);
        assert_complex32_eq(out[1], Complex32::new(-2.0, 2.0), 1e-5);
        assert_complex32_eq(out[2], Complex32::new(-2.0, 0.0), 1e-5);
        assert_complex32_eq(out[3], Complex32::new(-2.0, -2.0), 1e-5);
    }

    #[test]
    fn test_fft4_i32() {
        let input = vec![1i32, 2, 3, 4];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        assert_complex32_eq(out[0], Complex32::new(10.0, 0.0), 1e-5);
        assert_complex32_eq(out[1], Complex32::new(-2.0, 2.0), 1e-5);
        assert_complex32_eq(out[2], Complex32::new(-2.0, 0.0), 1e-5);
    }

    #[test]
    fn test_fft4_i64() {
        let input = vec![1i64, 2, 3, 4];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        assert_complex64_eq(out[0], Complex64::new(10.0, 0.0), 1e-10);
        assert_complex64_eq(out[1], Complex64::new(-2.0, 2.0), 1e-10);
        assert_complex64_eq(out[2], Complex64::new(-2.0, 0.0), 1e-10);
    }

    #[test]
    fn test_fft_length_1() {
        let fft = FFT::new(vec![42.0f64]).unwrap();
        let out = fft.compute();
        assert_eq!(out.len(), 1);
        assert_complex64_eq(out[0], Complex64::new(42.0, 0.0), 1e-12);
    }

    #[test]
    fn test_roundtrip_f64() {
        let input: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<f64>::ifft(spectrum);
        for i in 0..input.len() {
            assert_complex64_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-8);
        }
    }

    #[test]
    fn test_roundtrip_f32() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).cos()).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<f32>::ifft(spectrum);
        for i in 0..input.len() {
            assert_complex32_eq(recovered[i], Complex32::new(input[i], 0.0), 1e-4);
        }
    }

    #[test]
    fn test_roundtrip_i32() {
        let input: Vec<i32> = vec![1, 0, -1, 0, 1, 0, -1, 0];
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<i32>::ifft(spectrum);
        for i in 0..input.len() {
            assert_complex32_eq(recovered[i], Complex32::new(input[i] as f32, 0.0), 1e-4);
        }
    }

    #[test]
    fn test_parseval_f64() {
        let input: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()).collect();
        let n = input.len();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let time_energy: f64 = input.iter().map(|x| x * x).sum();
        let freq_energy: f64 = spectrum.iter().map(|x| x.magnitude_squared()).sum();
        assert!((time_energy - freq_energy / n as f64).abs() < 1e-6 * time_energy);
    }

    #[test]
    fn test_parseval_f32() {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let n = input.len();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let time_energy: f32 = input.iter().map(|x| x * x).sum();
        let freq_energy: f32 = spectrum.iter().map(|x| x.magnitude_squared()).sum();
        assert!((time_energy - freq_energy / n as f32).abs() < 1e-3 * time_energy);
    }

    #[test]
    fn test_delta_f64() {
        let input = vec![1.0f64, 0.0, 0.0, 0.0];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        for v in &out {
            assert_complex64_eq(*v, Complex64::one(), 1e-10);
        }
    }

    #[test]
    fn test_constant_signal_f64() {
        let input = vec![5.0f64; 8];
        let fft = FFT::new(input).unwrap();
        let out = fft.compute();
        assert_complex64_eq(out[0], Complex64::new(40.0, 0.0), 1e-10);
        for i in 1..8 {
            assert_complex64_eq(out[i], Complex64::zero(), 1e-10);
        }
    }

    #[test]
    fn test_new_rejects_zero_length() {
        assert!(FFT::<f64>::new(vec![]).is_err());
        assert!(matches!(FFT::<f64>::new(vec![]), Err(FftError::ZeroLength)));
    }

    #[test]
    fn test_new_rejects_non_power_of_two() {
        assert!(FFT::<f64>::new(vec![1.0, 2.0, 3.0]).is_err());
        assert!(matches!(FFT::<f64>::new(vec![1.0, 2.0, 3.0]), Err(FftError::NotPowerOfTwo(3))));
    }

    #[test]
    fn test_new_rejects_too_large() {
        assert!(matches!(FFT::<f64>::new(vec![0.0f64; 1usize << 25]), Err(FftError::TooLarge(_))));
    }

    #[test]
    fn test_from_slice() {
        let slice: &[f64] = &[1.0, 2.0, 3.0, 4.0];
        let fft = FFT::from_slice(slice).unwrap();
        assert_eq!(fft.len(), 4);
    }

    fn naive_dft(input: &[f64]) -> Vec<Complex64> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = Complex64::zero();
            for m in 0..n {
                let tw = Complex64::twiddle(n, k * m);
                sum = sum + tw * Complex64::new(input[m], 0.0);
            }
            out.push(sum);
        }
        out
    }

    #[test]
    fn test_fft_vs_naive_dft_n8() {
        let input: Vec<f64> = vec![0.0, 1.0, 0.5, -0.5, 0.25, 0.75, -1.0, 0.1];
        let fft = FFT::new(input.clone()).unwrap();
        let fft_out = fft.compute();
        let naive_out = naive_dft(&input);
        for i in 0..input.len() {
            assert_complex64_eq(fft_out[i], naive_out[i], 1e-10);
        }
    }

    #[test]
    fn test_fft_vs_naive_dft_n16() {
        let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).sin() * (i as f64 * 0.7).cos()).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let fft_out = fft.compute();
        let naive_out = naive_dft(&input);
        for i in 0..input.len() {
            assert_complex64_eq(fft_out[i], naive_out[i], 1e-10);
        }
    }

    #[test]
    fn test_fft_vs_naive_dft_n32() {
        let input: Vec<f64> = (0..32).map(|i| ((i * 3 + 7) % 17) as f64 / 16.0).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let fft_out = fft.compute();
        let naive_out = naive_dft(&input);
        for i in 0..input.len() {
            assert_complex64_eq(fft_out[i], naive_out[i], 1e-10);
        }
    }

    #[test]
    fn test_is_power_of_two_from_error() {
        assert!(crate::error::is_power_of_two(1));
        assert!(crate::error::is_power_of_two(1024));
        assert!(!crate::error::is_power_of_two(0));
        assert!(!crate::error::is_power_of_two(6));
    }

    // -----------------------------------------------------------------------
    // New test: verify bit_reverse at boundary values (Proposal 2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bit_reverse_equivalence() {
        // For N=8 (log2n=3): reverse 3 bits
        assert_eq!(bit_reverse(0, 3), 0);       // 000 -> 000
        assert_eq!(bit_reverse(1, 3), 4);        // 001 -> 100
        assert_eq!(bit_reverse(4, 3), 1);        // 100 -> 001
        assert_eq!(bit_reverse(7, 3), 7);       // 111 -> 111
        // For N=4 (log2n=2):
        assert_eq!(bit_reverse(0, 2), 0);       // 00 -> 00
        assert_eq!(bit_reverse(1, 2), 2);        // 01 -> 10
        assert_eq!(bit_reverse(2, 2), 1);        // 10 -> 01
        assert_eq!(bit_reverse(3, 2), 3);        // 11 -> 11
    }
}