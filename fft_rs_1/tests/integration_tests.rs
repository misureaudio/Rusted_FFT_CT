//! Integration tests for fft_rs.

use fft_rs::{FFT, FFTRun, Complex64, FftError, is_power_of_two, IntoSample};

// ============================================================================
// 1. API Misuse Prevention
// ============================================================================

#[test]
fn reject_empty_input() {
    assert!(matches!(FFT::<f64>::new(vec![]), Err(FftError::ZeroLength)));
}

#[test]
fn reject_non_power_of_two_lengths() {
    for &len in &[3, 5, 6, 7, 9, 10, 15, 100, 1023] {
        let data = vec![0.0f64; len];
        let result = FFT::<f64>::new(data);
        assert!(
            matches!(result, Err(FftError::NotPowerOfTwo(_))),
            "expected NotPowerOfTwo for length {}",
            len
        );
    }
}

#[test]
fn reject_too_large_input() {
    assert!(matches!(FFT::<f64>::new(vec![0.0f64; 1usize << 25]), Err(FftError::TooLarge(_))));
}

#[test]
fn accept_maximum_valid_size() {
    let data = vec![0.0f64; 1 << 24];
    let fft = FFT::<f64>::new(data).unwrap();
    assert_eq!(fft.len(), 1 << 24);
}

#[test]
fn accept_all_power_of_two_sizes() {
    for exp in 0..=24 {
        let n = 1usize << exp;
        let data = vec![1.0f64; n];
        let fft = FFT::<f64>::new(data).unwrap();
        assert_eq!(fft.len(), n, "failed at 2^{}", exp);
    }
}

// ============================================================================
// 2. Mathematical Correctness
// ============================================================================

#[test]
fn roundtrip_all_types() {
    // f64
    {
        let input: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin() * (i as f64 * 0.03).cos()).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<f64>::ifft(spectrum);
        for i in 0..input.len() {
            assert!((recovered[i].re - input[i]).abs() < 1e-10);
            assert!(recovered[i].im.abs() < 1e-10);
        }
    }

    // f32
    {
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin() * (i as f32 * 0.03).cos()).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<f32>::ifft(spectrum);
        for i in 0..input.len() {
            assert!((recovered[i].re - input[i]).abs() < 1e-4);
            assert!(recovered[i].im.abs() < 1e-4);
        }
    }

    // i32
    {
        let input: Vec<i32> = (0..256).map(|i| ((i * 7 + 3) % 100 - 50) as i32).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<i32>::ifft(spectrum);
        for i in 0..input.len() {
            assert!((recovered[i].re - input[i] as f32).abs() < 1e-2);
            assert!(recovered[i].im.abs() < 1e-2);
        }
    }

    // i64
    {
        let input: Vec<i64> = (0..256).map(|i| ((i * 7 + 3) % 100 - 50) as i64).collect();
        let fft = FFT::new(input.clone()).unwrap();
        let spectrum = fft.compute();
        let recovered = FFT::<i64>::ifft(spectrum);
        for i in 0..input.len() {
            assert!((recovered[i].re - input[i] as f64).abs() < 1e-6);
            assert!(recovered[i].im.abs() < 1e-6);
        }
    }
}

#[test]
fn frequency_peak_detection() {
    let n = 1024;
    let freq_bin = 5usize;
    let input: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * freq_bin as f64 * i as f64 / n as f64)
        .map(f64::sin)
        .collect();

    let fft = FFT::new(input).unwrap();
    let spectrum = fft.compute();

    let peak_mag = spectrum[freq_bin].magnitude();

    for i in 0..n {
        if i != freq_bin && i != n - freq_bin {
            assert!(
                spectrum[i].magnitude() < peak_mag * 0.01,
                "unexpected large value at bin {}: {} (peak: {})",
                i,
                spectrum[i].magnitude(),
                peak_mag
            );
        }
    }

    assert!((peak_mag - n as f64 / 2.0).abs() < 1.0);
}

#[test]
fn convolution_theorem() {
    let n = 64;

    let a: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64).exp()).collect();
    let b: Vec<f64> = (0..n).map(|i| (-1.0 * i as f64 / n as f64).exp()).collect();

    let fft_a = FFT::new(a.clone()).unwrap().compute();
    let fft_b = FFT::new(b.clone()).unwrap().compute();

    let mut product = Vec::with_capacity(n);
    for i in 0..n {
        product.push(fft_a[i] * fft_b[i]);
    }

    let conv = FFT::<f64>::ifft(product);

    let mut expected = vec![Complex64::zero(); n];
    for i in 0..n {
        for j in 0..n {
            let idx = (i + j) % n;
            expected[idx] = expected[idx] + Complex64::new(a[j] * b[i], 0.0);
        }
    }

    for i in 0..n {
        let diff_re = (conv[i].re - expected[i].re).abs();
        let diff_im = conv[i].im.abs();
        assert!(diff_re < 1e-8, "re diff at {}: {} vs {}", i, conv[i].re, expected[i].re);
        assert!(diff_im < 1e-8);
    }
}

#[test]
fn hermitian_symmetry_for_real_input() {
    let n = 128;
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.3).cos()).collect();

    let fft = FFT::new(input).unwrap();
    let spectrum = fft.compute();

    for k in 1..n / 2 {
        let xk = spectrum[k];
        let xk_mirror = spectrum[n - k];
        assert!((xk.re - xk_mirror.re).abs() < 1e-10, "real part mismatch at k={}", k);
        assert!((xk.im + xk_mirror.im).abs() < 1e-10, "imag part mismatch at k={}", k);
    }

    assert!(spectrum[0].im.abs() < 1e-10);
    assert!(spectrum[n / 2].im.abs() < 1e-10);
}

#[test]
fn plan_vs_nonplan_consistency() {
    let n = 256;
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();

    let fft = FFT::new(input.clone()).unwrap();
    let spectrum_direct = fft.compute();

    let plan = FFTRun::<f64>::new(n).unwrap();
    let spectrum_plan = plan.fft(&input);

    for i in 0..n {
        assert!((spectrum_direct[i].re - spectrum_plan[i].re).abs() < 1e-12);
        assert!((spectrum_direct[i].im - spectrum_plan[i].im).abs() < 1e-12);
    }
}

// ============================================================================
// 3. Edge Cases
// ============================================================================

#[test]
fn length_1_fft() {
    let fft = FFT::new(vec![42.0f64]).unwrap();
    let out = fft.compute();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0], Complex64::new(42.0, 0.0));
}

#[test]
fn length_2_fft() {
    let fft = FFT::new(vec![3.0f64, 5.0]).unwrap();
    let out = fft.compute();
    assert!((out[0].re - 8.0).abs() < 1e-10);
    assert!((out[1].re - (-2.0)).abs() < 1e-10);
}

#[test]
fn all_zeros_input() {
    let input = vec![0.0f64; 64];
    let fft = FFT::new(input).unwrap();
    let out = fft.compute();
    for v in &out {
        assert!(v.magnitude() < 1e-10);
    }
}

#[test]
fn all_ones_input() {
    let input = vec![1.0f64; 64];
    let fft = FFT::new(input).unwrap();
    let out = fft.compute();
    assert!((out[0].re - 64.0).abs() < 1e-10);
    for i in 1..64 {
        assert!(out[i].magnitude() < 1e-10);
    }
}

#[test]
fn alternating_signs() {
    let n = 64;
    let input: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let fft = FFT::new(input).unwrap();
    let out = fft.compute();

    assert!(out[0].magnitude() < 1e-10);
    assert!((out[n / 2].re - 64.0).abs() < 1e-10);
    for i in 1..n / 2 {
        assert!(out[i].magnitude() < 1e-10);
        assert!(out[n - i].magnitude() < 1e-10);
    }
}

#[test]
fn large_negative_and_positive_values() {
    let input = vec![1e15f64, -1e15, 1e15, -1e15];
    let fft = FFT::new(input.clone()).unwrap();
    let out = fft.compute();

    let recovered = FFT::<f64>::ifft(out);
    for i in 0..4 {
        assert!((recovered[i].re - input[i]).abs() < 1e3);
        assert!(recovered[i].im.abs() < 1e3);
    }
}

// ============================================================================
// 4. FFTRun (Plan) Specific Tests
// ============================================================================

#[test]
fn plan_rejects_invalid_sizes() {
    assert!(FFTRun::<f64>::new(0).is_err());
    assert!(FFTRun::<f64>::new(6).is_err());
    assert!(FFTRun::<f64>::new(1 << 25).is_err());
}

#[test]
fn plan_reuses_successfully() {
    let plan = FFTRun::<f64>::new(8).unwrap();

    for _ in 0..100 {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = plan.fft(&input);
        let recovered = plan.ifft(spectrum);
        for i in 0..8 {
            assert!((recovered[i].re - input[i]).abs() < 1e-10);
        }
    }
}

#[test]
#[should_panic]
fn plan_panics_on_wrong_input_length() {
    let plan = FFTRun::<f64>::new(8).unwrap();
    let input = vec![1.0; 4];
    plan.fft(&input);
}

// ============================================================================
// 5. is_power_of_two utility
// ============================================================================

#[test]
fn is_power_of_two_correctness() {
    assert!(is_power_of_two(1));
    assert!(is_power_of_two(2));
    assert!(is_power_of_two(4));
    assert!(is_power_of_two(8));
    assert!(is_power_of_two(1024));
    assert!(is_power_of_two(1 << 24));

    assert!(!is_power_of_two(0));
    assert!(!is_power_of_two(3));
    assert!(!is_power_of_two(5));
    assert!(!is_power_of_two(6));
    assert!(!is_power_of_two(7));
    assert!(!is_power_of_two(15));
    assert!(!is_power_of_two(100));
}

// ============================================================================
// 6. Cross-type consistency
// ============================================================================

#[test]
fn same_signal_different_types_produce_consistent_results() {
    let input_f64: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let input_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_i32: Vec<i32> = vec![1, 2, 3, 4];

    let out_f64 = FFT::new(input_f64).unwrap().compute();
    let out_f32 = FFT::new(input_f32).unwrap().compute();
    let out_i32 = FFT::new(input_i32).unwrap().compute();

    for i in 0..4 {
        assert!((out_f64[i].re - out_f32[i].re as f64).abs() < 1e-5);
        assert!((out_f64[i].im - out_f32[i].im as f64).abs() < 1e-5);
        assert!((out_f64[i].re - out_i32[i].re as f64).abs() < 1e-5);
        assert!((out_f64[i].im - out_i32[i].im as f64).abs() < 1e-5);
    }
}

// ============================================================================
// 7. In-place API tests (Proposal 3)
// ============================================================================

#[test]
fn inplace_fft_matches_allocating_fft() {
    let plan = FFTRun::<f64>::new(8).unwrap();
    let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let allocated_out = plan.fft(&input);

    let mut buf: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
    plan.process_inplace_fft(&mut buf);

    for i in 0..8 {
        assert!((buf[i].re - allocated_out[i].re).abs() < 1e-12);
        assert!((buf[i].im - allocated_out[i].im).abs() < 1e-12);
    }
}

#[test]
fn inplace_roundtrip() {
    let plan = FFTRun::<f64>::new(32).unwrap();
    let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();

    let mut buf: Vec<Complex64> = input.iter().copied().map(|s| s.into_complex()).collect();
    plan.process_inplace_fft(&mut buf);
    plan.process_inplace_ifft(&mut buf);

    for i in 0..32 {
        assert!((buf[i].re - input[i]).abs() < 1e-10);
        assert!(buf[i].im.abs() < 1e-10);
    }
}

#[test]
#[should_panic]
fn inplace_fft_wrong_length_panics() {
    let plan = FFTRun::<f64>::new(8).unwrap();
    let mut buf = vec![Complex64::zero(); 4];
    plan.process_inplace_fft(&mut buf);
}

#[test]
#[should_panic]
fn inplace_ifft_wrong_length_panics() {
    let plan = FFTRun::<f64>::new(8).unwrap();
    let mut buf = vec![Complex64::zero(); 4];
    plan.process_inplace_ifft(&mut buf);
}