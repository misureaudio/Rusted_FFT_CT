//! Reusable FFT plan for zero-allocation repeated transforms.

use crate::error::{validate_length, FftResult};
use crate::fft_core::{ComplexSample, IntoSample};

/// Pre-allocated twiddle factors for a given FFT size.
struct TwiddleTable<C: ComplexSample> {
    forward: Vec<Vec<C>>,
    inverse: Vec<Vec<C>>,
}

impl<C: ComplexSample> TwiddleTable<C> {
    fn new(_n: usize, log2n: usize) -> Self {
        let mut forward = Vec::with_capacity(log2n);
        let mut inverse = Vec::with_capacity(log2n);

        let mut len = 2;
        for _ in 0..log2n {
            let half = len >> 1;
            forward.push((0..half).map(|k| C::twiddle(len, k)).collect());
            inverse.push((0..half).map(|k| C::twiddle_inverse(len, k)).collect());
            len <<= 1;
        }

        TwiddleTable { forward, inverse }
    }
}

/// A reusable FFT plan for a fixed size `N`.
///
/// The plan pre-computes twiddle factors so that repeated FFT / IFFT calls
/// at the same size avoid re-allocation.
pub struct FFTRun<T: IntoSample>
where
    T::Complex: ComplexSample,
{
    n: usize,
    log2n: usize,
    twiddles: TwiddleTable<T::Complex>,
}

impl<T: IntoSample> FFTRun<T>
where
    T::Complex: ComplexSample,
{
    /// Create a new `FFTRun` for `n` samples.
    pub fn new(n: usize) -> FftResult<Self> {
        validate_length(n)?;
        let log2n = n.trailing_zeros() as usize;
        let twiddles = TwiddleTable::<T::Complex>::new(n, log2n);
        Ok(FFTRun { n, log2n, twiddles })
    }

    /// The number of samples this plan handles.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Compute the forward FFT of a slice of length `n`.
    ///
    /// Panics if `input.len() != self.n`.
    pub fn fft(&self, input: &[T]) -> Vec<T::Complex> {
        assert_eq!(
            input.len(),
            self.n,
            "input length {} does not match plan size {}",
            input.len(),
            self.n
        );

        if self.n == 1 {
            return vec![input[0].into_complex()];
        }

        let mut buf: Vec<T::Complex> = input.iter().copied().map(|s| s.into_complex()).collect();
        fft_forward_with_table(&mut buf, self.n, self.log2n, &self.twiddles.forward);
        buf
    }

    /// Compute the inverse FFT of a complex vector of length `n`.
    pub fn ifft(&self, input: Vec<T::Complex>) -> Vec<T::Complex> {
        assert_eq!(
            input.len(),
            self.n,
            "input length {} does not match plan size {}",
            input.len(),
            self.n
        );

        if self.n == 1 {
            return input;
        }

        let mut buf = input;
        fft_inverse_with_table(&mut buf, self.n, self.log2n, &self.twiddles.inverse);
        buf
    }

    // =========================================================================
    // Proposal 3: Zero-allocation in-place methods
    // =========================================================================

    /// Zero-allocation in-place forward FFT.
    ///
    /// The `buffer` must have length equal to `self.n`. The data is transformed
    /// in-place using the pre-computed twiddle table.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() != self.n`.
    pub fn process_inplace_fft(&self, buffer: &mut [T::Complex]) {
        assert_eq!(buffer.len(), self.n,
            "buffer length {} does not match plan size {}", buffer.len(), self.n);
        if self.n <= 1 {
            return;
        }
        fft_forward_with_table(buffer, self.n, self.log2n, &self.twiddles.forward);
    }

    /// Zero-allocation in-place inverse FFT.
    ///
    /// The `buffer` must have length equal to `self.n`. The data is transformed
    /// in-place using the pre-computed twiddle table.
    ///
    /// # Panics
    ///
    /// Panics if `buffer.len() != self.n`.
    pub fn process_inplace_ifft(&self, buffer: &mut [T::Complex]) {
        assert_eq!(buffer.len(), self.n,
            "buffer length {} does not match plan size {}", buffer.len(), self.n);
        if self.n <= 1 {
            return;
        }
        fft_inverse_with_table(buffer, self.n, self.log2n, &self.twiddles.inverse);
    }
}

// ---------------------------------------------------------------------------
// Bit-reversal
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
// FFT / IFFT using pre-computed twiddle tables
// ---------------------------------------------------------------------------

/// Forward FFT using a pre-computed twiddle table.
///
/// This function operates in-place on `data` and is `pub(crate)` so that
/// `FFTRun::process_inplace_fft()` can call it directly (Proposal 3).
pub(crate) fn fft_forward_with_table<C: ComplexSample>(
    data: &mut [C],
    n: usize,
    log2n: usize,
    twiddles: &[Vec<C>],
) {
    bit_reverse_permute(data, n, log2n);

    let mut len = 2;
    for stage in 0..log2n {
        let half = len >> 1;
        let table = &twiddles[stage];
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let even_idx = start + k;
                let odd_idx = even_idx + half;
                let t = C::mul(table[k], data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }
}

/// Inverse FFT using a pre-computed twiddle table.
///
/// This function operates in-place on `data` and is `pub(crate)` so that
/// `FFTRun::process_inplace_ifft()` can call it directly (Proposal 3).
pub(crate) fn fft_inverse_with_table<C: ComplexSample>(
    data: &mut [C],
    n: usize,
    log2n: usize,
    twiddles: &[Vec<C>],
) {
    bit_reverse_permute(data, n, log2n);

    let mut len = 2;
    for stage in 0..log2n {
        let half = len >> 1;
        let table = &twiddles[stage];
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let even_idx = start + k;
                let odd_idx = even_idx + half;
                let t = C::mul(table[k], data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }

    let norm = C::scalar_from_usize(n);
    for i in 0..n {
        data[i] = C::div_scalar(data[i], norm);
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::{assert_complex32_eq, assert_complex64_eq, Complex32, Complex64};

    #[test]
    fn test_plan_fft4_f64() {
        let plan = FFTRun::<f64>::new(4).unwrap();
        let input = vec![1.0f64, 2.0, 3.0, 4.0];
        let out = plan.fft(&input);
        assert_complex64_eq(out[0], Complex64::new(10.0, 0.0), 1e-10);
        assert_complex64_eq(out[1], Complex64::new(-2.0, 2.0), 1e-10);
        assert_complex64_eq(out[2], Complex64::new(-2.0, 0.0), 1e-10);
        assert_complex64_eq(out[3], Complex64::new(-2.0, -2.0), 1e-10);
    }

    #[test]
    fn test_plan_roundtrip_f64() {
        let plan = FFTRun::<f64>::new(256).unwrap();
        let input: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
        let spectrum = plan.fft(&input);
        let recovered = plan.ifft(spectrum);
        for i in 0..input.len() {
            assert_complex64_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-10);
        }
    }

    #[test]
    fn test_plan_roundtrip_f32() {
        let plan = FFTRun::<f32>::new(128).unwrap();
        let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let spectrum = plan.fft(&input);
        let recovered = plan.ifft(spectrum);
        for i in 0..input.len() {
            assert_complex32_eq(recovered[i], Complex32::new(input[i], 0.0), 1e-4);
        }
    }

    #[test]
    fn test_plan_reuse() {
        let plan = FFTRun::<f64>::new(8).unwrap();
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sa = plan.fft(&a);
        let sb = plan.fft(&b);
        // Delta at position 0: FFT = all ones
        for i in 0..8 {
            assert_complex64_eq(sa[i], Complex64::one(), 1e-10);
        }
        // Delta at position 1: FFT[k] = W_8^k = exp(-2πik/8)
        for k in 0..8 {
            let expected = Complex64::twiddle(8, k);
            assert_complex64_eq(sb[k], expected, 1e-10);
        }
    }

    #[test]
    fn test_plan_rejects_invalid_size() {
        assert!(FFTRun::<f64>::new(0).is_err());
        assert!(FFTRun::<f64>::new(6).is_err());
        assert!(FFTRun::<f64>::new(1 << 25).is_err());
    }

    #[test]
    fn test_plan_n() {
        let plan = FFTRun::<f64>::new(1024).unwrap();
        assert_eq!(plan.n(), 1024);
    }

    #[test]
    #[should_panic]
    fn test_plan_panic_on_wrong_length_panics() {
        let plan = FFTRun::<f64>::new(4).unwrap();
        let input = vec![1.0, 2.0, 3.0];
        plan.fft(&input);
    }

    // =========================================================================
    // New tests for Proposal 3: process_inplace methods
    // =========================================================================

    #[test]
    fn test_process_inplace_fft_matches_allocating() {
        let plan = FFTRun::<f64>::new(8).unwrap();
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Allocating version
        let allocated_out = plan.fft(&input);

        // In-place version
        let mut buf: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
        plan.process_inplace_fft(&mut buf);

        for i in 0..8 {
            assert_complex64_eq(buf[i], allocated_out[i], 1e-12);
        }
    }

    #[test]
    fn test_process_inplace_ifft_roundtrip() {
        let plan = FFTRun::<f64>::new(16).unwrap();
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];

        // Forward in-place
        let mut buf: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
        plan.process_inplace_fft(&mut buf);

        // Inverse in-place
        plan.process_inplace_ifft(&mut buf);

        for i in 0..16 {
            assert_complex64_eq(buf[i], Complex64::new(input[i], 0.0), 1e-10);
        }
    }

    #[test]
    fn test_process_inplace_repeated_calls_stable() {
        // Verify that repeated in-place FFT + IFFT cycles are stable:
        // starting from the same input, two separate FFT->IFFT cycles produce identical results.
        let plan = FFTRun::<f64>::new(8).unwrap();
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // First cycle
        let mut buf1: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
        plan.process_inplace_fft(&mut buf1);
        plan.process_inplace_ifft(&mut buf1);

        // Second cycle (same input)
        let mut buf2: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
        plan.process_inplace_fft(&mut buf2);
        plan.process_inplace_ifft(&mut buf2);

        // Both should recover the original input
        for i in 0..8 {
            assert_complex64_eq(buf1[i], buf2[i], 1e-12);
            assert_complex64_eq(buf1[i], Complex64::new(input[i], 0.0), 1e-10);
        }
    }

    #[test]
    #[should_panic]
    fn test_process_inplace_wrong_length_panics() {
        let plan = FFTRun::<f64>::new(8).unwrap();
        let mut buf = vec![Complex64::zero(); 4];
        plan.process_inplace_fft(&mut buf);
    }

    #[test]
    #[should_panic]
    fn test_process_inplace_ifft_wrong_length_panics() {
        let plan = FFTRun::<f64>::new(8).unwrap();
        let mut buf = vec![Complex64::zero(); 4];
        plan.process_inplace_ifft(&mut buf);
    }
}