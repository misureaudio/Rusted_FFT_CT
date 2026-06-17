# fft_rs — API User Manual

**Version 0.1.0** | A high-performance Fast Fourier Transform (FFT) library for Rust

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation and Setup](#2-installation-and-setup)
3. [Quick Start](#3-quick-start)
4. [Core Concepts](#4-core-concepts)
5. [API Reference](#5-api-reference)
   - [5.1 Complex Number Types](#51-complex-number-types)
   - [5.2 Error Handling](#52-error-handling)
   - [5.3 FFT — One-Shot Transform](#53-fft--one-shot-transform)
   - [5.4 FFTRun — Reusable Plan](#54-fftrun--reusable-plan)
   - [5.5 Traits](#55-traits)
6. [Practical Use Cases](#6-practical-use-cases)
   - [6.1 Frequency Analysis of Audio Signals](#61-frequency-analysis-of-audio-signals)
   - [6.2 Signal Reconstruction (Forward + Inverse FFT)](#62-signal-reconstruction-forward--inverse-fft)
   - [6.3 Circular Convolution via the Convolution Theorem](#63-circular-convolution-via-the-convolution-theorem)
   - [6.4 Power Spectrum and Energy Analysis (Parseval's Theorem)](#64-power-spectrum-and-energy-analysis-parsevals-theorem)
   - [6.5 High-Performance Repeated Transforms with FFTRun](#65-high-performance-repeated-transforms-with-fftrun)
   - [6.6 Zero-Allocation In-Place Transforms](#66-zero-allocation-in-place-transforms)
7. [Supported Input Types and Type Mapping](#7-supported-input-types-and-type-mapping)
8. [Constraints and Limits](#8-constraints-and-limits)
9. [Error Handling Guide](#9-error-handling-guide)
10. [Performance Notes](#10-performance-notes)
11. [Frequently Asked Questions](#11-frequently-asked-questions)

---

## 1. Overview

`fft_rs` is a pure-Rust library that computes the **Discrete Fourier Transform (DFT)** and its inverse using the **Cooley–Tukey radix-2 decimation-in-time (DIT)** algorithm. It provides two usage patterns:

| Pattern | Type | Best For |
|---------|------|----------|
| **One-shot** | `FFT<T>` | Single or infrequent transforms |
| **Reusable plan** | `FFTRun<T>` | Repeated transforms at the same size (pre-computed twiddle factors, zero-allocation in-place methods) |

The library supports four input sample types (`i32`, `i64`, `f32`, `f64`) and vector lengths from **1** to **2²⁴** (16,777,216), provided the length is a **positive power of two**.

### Key Features

- **Generic over input type**: `i32`, `i64`, `f32`, `f64`
- **Two complex output types**: `Complex32` (f32-based) and `Complex64` (f64-based)
- **Forward and inverse FFT** with proper normalization
- **Reusable plans** (`FFTRun`) with pre-computed twiddle factors
- **Zero-allocation in-place transforms** via `process_inplace_fft` / `process_inplace_ifft`
- **Bit-reversal via CPU instructions** (`reverse_bits()`) for O(N) permutation
- **Twiddle factors computed on-the-fly** in the one-shot path (no heap allocation per stage)
- **Comprehensive error types** with descriptive messages

---

## 2. Installation and Setup

### Adding as a Dependency

If `fft_rs` is published to crates.io, add it to your `Cargo.toml`:

```toml
[dependencies]
fft_rs = "0.1"
```

For local development, reference the path:

```toml
[dependencies]
fft_rs = { path = "../fft_rs_1" }
```

### Building

```bash
cd fft_rs_1
cargo build --release
```

### Running Tests

```bash
cargo test
```

---

## 3. Quick Start

```rust
use fft_rs::{FFT, Complex64};

fn main() {
    // Create input: a power-of-two-length vector
    let input = vec![1.0f64, 2.0, 3.0, 4.0];

    // Build the FFT handle and compute the forward transform
    let fft = FFT::new(input).unwrap();
    let spectrum = fft.compute();

    // Print the spectrum
    for (i, bin) in spectrum.iter().enumerate() {
        println!("Bin {:>2}: {:?}  |magnitude| = {:.4}",
                 i, bin, bin.magnitude());
    }
    // Bin  0: (10 - 0i)  |magnitude| = 10.0000
    // Bin  1: (-2 + 2i)  |magnitude| = 2.8284
    // Bin  2: (-2 - 0i)  |magnitude| = 2.0000
    // Bin  3: (-2 - 2i)  |magnitude| = 2.8284

    // Recover the original signal via the inverse FFT
    let recovered = FFT::<f64>::ifft(spectrum);
    println!("Recovered: {:?}", recovered);
    // Recovered: [1, 2, 3, 4]  (within floating-point tolerance)
}
```

---

## 4. Core Concepts

### 4.1 The FFT and IFFT

The **forward FFT** transforms a time-domain signal `x[n]` (length N) into its frequency-domain representation `X[k]`:

```
X[k] = Σₙ x[n] · e^(-2πi·k·n / N)    for k = 0, 1, ..., N-1
```

The **inverse FFT (IFFT)** recovers the original signal:

```
x[n] = (1/N) · Σₖ X[k] · e^(+2πi·k·n / N)    for n = 0, 1, ..., N-1
```

`fft_rs` applies the `1/N` normalization in the IFFT direction, so a round-trip `IFFT(FFT(x))` returns the original signal.

### 4.2 Frequency Bin Interpretation

For a signal of length **N** sampled at rate **Fs**:

| Bin Index | Frequency |
|-----------|-----------|
| 0         | 0 Hz (DC) |
| 1         | Fs / N |
| k         | k · Fs / N |
| N/2       | Fs / 2 (Nyquist) |
| N/2 + 1   | -Fs / 2 + Fs/N |
| N - 1     | -Fs / N |

For **real-valued** input, the spectrum exhibits **Hermitian symmetry**: `X[N-k] = conj(X[k])`, so only the first `N/2 + 1` bins carry unique information.

### 4.3 One-Shot vs. Reusable Plan

- **`FFT<T>`** — ideal for a single transform. It owns the input data and computes twiddle factors on-the-fly, avoiding per-stage heap allocations.

- **`FFTRun<T>`** — ideal for repeated transforms at the same size. It pre-computes and caches twiddle factors, and provides **in-place** methods (`process_inplace_fft`, `process_inplace_ifft`) that operate on a caller-allocated buffer with zero additional allocations.

---

## 5. API Reference

### 5.1 Complex Number Types

#### `Complex32`

A complex number with `f32` real and imaginary parts.

```rust
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `const fn new(re: f32, im: f32) -> Self` | Create a complex number |
| `zero` | `const fn zero() -> Self` | Returns `0 + 0i` |
| `one` | `const fn one() -> Self` | Returns `1 + 0i` |
| `conjugate` | `fn conjugate(self) -> Self` | Returns `re - im·i` |
| `magnitude` | `fn magnitude(self) -> f32` | Returns `√(re² + im²)` |
| `magnitude_squared` | `fn magnitude_squared(self) -> f32` | Returns `re² + im²` (avoids `sqrt`) |
| `phase` | `fn phase(self) -> f32` | Returns the phase angle in radians ∈ (-π, π] |
| `twiddle` | `fn twiddle(n: usize, k: usize) -> Self` | Returns `e^(-2πi·k/N)` (forward FFT) |
| `twiddle_inverse` | `fn twiddle_inverse(n: usize, k: usize) -> Self` | Returns `e^(+2πi·k/N)` (inverse FFT) |

**Arithmetic operators**: `+`, `-`, `*` (complex × complex), `/` (complex / scalar f32), unary `-`.

**Traits implemented**: `Clone`, `Copy`, `PartialEq`, `Default`, `Debug`, `Display`, `Add`, `Sub`, `Mul`, `Div<f32>`, `Neg`.

#### `Complex64`

Identical API to `Complex32`, but with `f64` precision.

```rust
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}
```

**Traits implemented**: Same as `Complex32`, with `Div<f64>` instead of `Div<f32>`.

#### Example

```rust
use fft_rs::{Complex32, Complex64};

fn main() {
    let z = Complex64::new(3.0, 4.0);
    println!("|z| = {}", z.magnitude());        // 5.0
    println!("|z|² = {}", z.magnitude_squared()); // 25.0
    println!("arg(z) = {} rad", z.phase());     // ~0.9273 rad
    println!("conj(z) = {:?}", z.conjugate());  // (3 - 4i)

    let w = Complex64::new(1.0, -1.0);
    println!("z * w = {:?}", z * w);            // (7 + 1i)

    let s = Complex32::new(6.0, 8.0) / 2.0;
    println!("s = {:?}", s);                    // (3 + 4i)
}
```

---

### 5.2 Error Handling

#### `FftError`

```rust
pub enum FftError {
    ZeroLength,
    NotPowerOfTwo(usize),
    TooLarge(usize),
}
```

| Variant | When It Occurs |
|---------|---------------|
| `ZeroLength` | Input vector has length 0 |
| `NotPowerOfTwo(len)` | Input length is not a power of 2 |
| `TooLarge(len)` | Input length exceeds 2²⁴ (16,777,216) |

All variants implement `std::fmt::Display` and `std::error::Error`.

#### `FftResult<T>`

```rust
pub type FftResult<T> = Result<T, FftError>;
```

#### Utility Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `is_power_of_two` | `fn is_power_of_two(n: usize) -> bool` | Returns `true` if `n` is a power of 2 |
| `validate_length` | `fn validate_length(len: usize) -> FftResult<()>` | Validates that `len` is a legal FFT size; returns `Err` otherwise |
| `MAX_FFT_SIZE` | `const MAX_FFT_SIZE: usize = 1 << 24` | Maximum supported FFT size |

#### Example

```rust
use fft_rs::{FFT, FftError, is_power_of_two, validate_length};

fn main() {
    // Check before creating
    assert!(is_power_of_two(1024));
    assert!(!is_power_of_two(1023));

    // Validate length
    match validate_length(7) {
        Ok(()) => println!("Valid length"),
        Err(e) => println!("Error: {}", e),
        // Error: FFT input length 7 is not a power of 2; length must be a positive power of 2
    }

    // Handle construction errors
    let result = FFT::<f64>::new(vec![1.0, 2.0, 3.0]);
    match result {
        Ok(_) => println!("FFT created"),
        Err(FftError::NotPowerOfTwo(len)) => {
            println!("Need power-of-two length, got {}", len);
        }
        Err(e) => println!("Other error: {}", e),
    }
}
```

---

### 5.3 FFT — One-Shot Transform

#### `FFT<T: IntoSample>`

A handle for computing forward and inverse FFT on a single input vector.

```rust
pub struct FFT<T: IntoSample> {
    // private fields
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(data: Vec<T>) -> FftResult<Self>` | Create from an owned `Vec`. Validates length. |
| `from_slice` | `fn from_slice(slice: &[T]) -> FftResult<Self>` | Create from a slice (cloned internally). |
| `len` | `fn len(&self) -> usize` | Number of samples |
| `is_empty` | `fn is_empty(&self) -> bool` | Returns `true` if length is 0 (always `false` for valid FFT) |
| `input` | `fn input(&self) -> &[T]` | Reference to the original input data |
| `compute` | `fn compute(&self) -> Vec<T::Complex>` | **Forward FFT** — returns the frequency-domain spectrum |
| `compute_inplace` | `fn compute_inplace(&mut self) -> Vec<T::Complex>` | Same as `compute()` (provided for API symmetry) |
| `ifft` | `fn ifft(data: Vec<T::Complex>) -> Vec<T::Complex>` | **Static: Inverse FFT** — recovers time-domain from spectrum |
| `compute_inverse` | `fn compute_inverse(&self, spectrum: &[T::Complex]) -> Vec<T::Complex>` | Inverse FFT of a given spectrum (wraps `ifft`) |

#### Example: Forward FFT

```rust
use fft_rs::FFT;

fn main() {
    let input = vec![1.0f64, 2.0, 3.0, 4.0];
    let fft = FFT::new(input).unwrap();

    let spectrum = fft.compute();

    // DC component (sum of all samples)
    println!("DC = {:?}", spectrum[0]);  // (10 - 0i)

    // Frequency bin 1
    println!("X[1] = {:?}", spectrum[1]);  // (-2 + 2i)
}
```

#### Example: Inverse FFT (Round-Trip)

```rust
use fft_rs::{FFT, Complex64};

fn main() {
    let original: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.01).sin()).collect();

    let fft = FFT::new(original.clone()).unwrap();
    let spectrum = fft.compute();
    let recovered = FFT::<f64>::ifft(spectrum);

    // Verify round-trip
    for i in 0..original.len() {
        let diff = (recovered[i].re - original[i]).abs();
        assert!(diff < 1e-8, "Mismatch at index {}: diff = {}", i, diff);
    }
    println!("Round-trip verified — all values match within tolerance.");
}
```

---

### 5.4 FFTRun — Reusable Plan

#### `FFTRun<T: IntoSample>`

A reusable FFT plan for a **fixed** size `N`. Pre-computes twiddle factors so repeated transforms avoid re-allocation.

```rust
pub struct FFTRun<T: IntoSample> {
    // private fields
}
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(n: usize) -> FftResult<Self>` | Create a plan for `n` samples. Pre-computes twiddle factors. |
| `n` | `fn n(&self) -> usize` | The number of samples this plan handles |
| `fft` | `fn fft(&self, input: &[T]) -> Vec<T::Complex>` | Forward FFT. **Panics** if `input.len() != self.n()` |
| `ifft` | `fn ifft(&self, input: Vec<T::Complex>) -> Vec<T::Complex>` | Inverse FFT. **Panics** if `input.len() != self.n()` |
| `process_inplace_fft` | `fn process_inplace_fft(&self, buffer: &mut [T::Complex])` | **In-place** forward FFT on a pre-allocated buffer. Zero additional allocations. **Panics** if `buffer.len() != self.n()` |
| `process_inplace_ifft` | `fn process_inplace_ifft(&self, buffer: &mut [T::Complex])` | **In-place** inverse FFT on a pre-allocated buffer. Zero additional allocations. **Panics** if `buffer.len() != self.n()` |

#### Example: Repeated Transforms

```rust
use fft_rs::FFTRun;

fn main() {
    // Create a plan once for N=256
    let plan = FFTRun::<f64>::new(256).unwrap();
    println!("Plan created for N = {}", plan.n());

    // Transform many signals at the same size
    for frame_idx in 0..1000 {
        let signal: Vec<f64> = (0..256)
            .map(|i| ((i + frame_idx) as f64 * 0.05).sin())
            .collect();

        let spectrum = plan.fft(&signal);
        // ... process spectrum ...
    }
}
```

#### Example: In-Place Transforms

```rust
use fft_rs::{FFTRun, Complex64, IntoSample};

fn main() {
    let n = 512;
    let plan = FFTRun::<f64>::new(n).unwrap();

    // Prepare input
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.02).sin() * (i as f64 * 0.07).cos()).collect();

    // Allocate buffer once
    let mut buffer: Vec<Complex64> = input.iter()
        .map(|&s| s.into_complex())
        .collect();

    // Forward transform in-place
    plan.process_inplace_fft(&mut buffer);

    // ... process buffer in frequency domain ...

    // Inverse transform in-place
    plan.process_inplace_ifft(&mut buffer);

    // buffer now contains the recovered signal
}
```

---

### 5.5 Traits

#### `IntoSample`

```rust
pub trait IntoSample: Copy + Clone {
    type Complex: ComplexSample;
    fn into_complex(self) -> Self::Complex;
}
```

Converts a real-valued sample into a complex number (with zero imaginary part). Implemented for:

| Input Type | `Complex` Associated Type |
|------------|--------------------------|
| `i32`      | `Complex32` |
| `i64`      | `Complex64` |
| `f32`      | `Complex32` |
| `f64`      | `Complex64` |

#### `ComplexSample`

```rust
pub trait ComplexSample: Clone + Copy + Send + Sync {
    type Scalar: Copy;
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
}
```

Internal trait used by the generic FFT core. Implemented for `Complex32` and `Complex64`.

---

## 6. Practical Use Cases

### 6.1 Frequency Analysis of Audio Signals

Detect the dominant frequency in a sampled sine wave.

```rust
use fft_rs::FFT;

fn main() {
    let n = 1024;            // Number of samples (must be power of 2)
    let fs = 8000.0f64;      // Sampling rate: 8 kHz
    let f0 = 440.0f64;       // Frequency to detect: 440 Hz (A4 note)

    // Generate a pure tone
    let signal: Vec<f64> = (0..n)
        .map(|i| 2.0 * std::f64::consts::PI * f0 * i as f64 / fs)
        .map(f64::sin)
        .collect();

    // Compute the FFT
    let fft = FFT::new(signal).unwrap();
    let spectrum = fft.compute();

    // Find the frequency bin with the largest magnitude (excluding DC)
    let mut peak_bin = 0;
    let mut peak_mag = 0.0f64;
    for k in 1..n / 2 {
        let mag = spectrum[k].magnitude();
        if mag > peak_mag {
            peak_mag = mag;
            peak_bin = k;
        }
    }

    // Convert bin index to frequency
    let detected_freq = peak_bin as f64 * fs / n as f64;
    println!("Expected frequency:  {} Hz", f0);
    println!("Detected frequency:  {:.1} Hz (bin {})", detected_freq, peak_bin);
    println!("Peak magnitude:      {:.1}", peak_mag);
    // Expected: ~440 Hz, bin 56, magnitude ~512
}
```

**Explanation**: A pure tone at frequency `f₀` produces two peaks in the spectrum at bins `k = f₀·N/Fs` and `N - k`. The magnitude at each peak is approximately `N/2` for a sine wave of unit amplitude.

---

### 6.2 Signal Reconstruction (Forward + Inverse FFT)

Verify that a signal can be perfectly reconstructed via FFT → IFFT.

```rust
use fft_rs::FFT;

fn main() {
    let n = 2048;
    let original: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            t.sin() * 0.01        // 0.01 Hz component
              + (t * 0.05).cos()  // 0.05 Hz component
              + (t * 0.13).sin()  // 0.13 Hz component
        })
        .collect();

    // Forward FFT
    let fft = FFT::new(original.clone()).unwrap();
    let spectrum = fft.compute();

    // Inverse FFT
    let recovered = FFT::<f64>::ifft(spectrum);

    // Check reconstruction error
    let max_error = (0..n)
        .map(|i| (recovered[i].re - original[i]).abs())
        .fold(0.0f64, f64::max);

    println!("Maximum reconstruction error: {:.2e}", max_error);
    // Typical: ~1e-10 for f64

    assert!(max_error < 1e-8, "Reconstruction error too large!");
    println!("Reconstruction verified.");
}
```

---

### 6.3 Circular Convolution via the Convolution Theorem

The convolution theorem states: `FFT(a ⊛ b) = FFT(a) · FFT(b)`, where `⊛` denotes circular convolution and `·` is element-wise multiplication. This enables O(N log N) convolution instead of O(N²).

```rust
use fft_rs::{FFT, Complex64, IntoSample};

fn main() {
    let n = 64;

    // Two signals to convolve
    let a: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64).exp()).collect();
    let b: Vec<f64> = (0..n).map(|i| (-1.0 * i as f64 / n as f64).exp()).collect();

    // Step 1: Transform both signals
    let fft_a = FFT::new(a.clone()).unwrap().compute();
    let fft_b = FFT::new(b.clone()).unwrap().compute();

    // Step 2: Element-wise multiply in frequency domain
    let mut product = Vec::with_capacity(n);
    for i in 0..n {
        product.push(fft_a[i] * fft_b[i]);
    }

    // Step 3: Inverse transform to get circular convolution
    let conv_result = FFT::<f64>::ifft(product);

    // Verify against naive O(N²) circular convolution
    let mut expected = vec![Complex64::zero(); n];
    for i in 0..n {
        for j in 0..n {
            let idx = (i + j) % n;
            expected[idx] = expected[idx] + Complex64::new(a[j] * b[i], 0.0);
        }
    }

    let max_error = (0..n)
        .map(|i| (conv_result[i].re - expected[i].re).abs())
        .fold(0.0f64, f64::max);

    println!("Convolution theorem verified. Max error: {:.2e}", max_error);
    // Typical: ~1e-8 for f64
}
```

**Performance note**: For `N = 2²⁰ ≈ 1M`, the FFT-based convolution is roughly **200× faster** than the naive O(N²) approach.

---

### 6.4 Power Spectrum and Energy Analysis (Parseval's Theorem)

Parseval's theorem states that the total energy in the time domain equals the total energy in the frequency domain:

```
Σₙ |x[n]|² = (1/N) · Σₖ |X[k]|²
```

```rust
use fft_rs::FFT;

fn main() {
    let n = 1024;

    // A signal composed of two sinusoids
    let signal: Vec<f64> = (0..n)
        .map(|i| {
            (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()
        })
        .collect();

    // Compute the FFT
    let fft = FFT::new(signal.clone()).unwrap();
    let spectrum = fft.compute();

    // Time-domain energy
    let time_energy: f64 = signal.iter().map(|x| x * x).sum();

    // Frequency-domain energy (scaled by 1/N)
    let freq_energy: f64 = spectrum.iter()
        .map(|x| x.magnitude_squared())
        .sum::<f64>() / n as f64;

    let relative_error = (time_energy - freq_energy).abs() / time_energy;

    println!("Time-domain energy:    {:.6}", time_energy);
    println!("Frequency-domain:      {:.6}", freq_energy);
    println!("Relative error:        {:.2e}", relative_error);
    // Relative error should be < 1e-10 for f64

    assert!(relative_error < 1e-8, "Parseval's theorem violated!");
    println!("Parseval's theorem verified.");
}
```

**Practical application**: This is used in audio processing to verify that filtering operations preserve signal energy, and in communications to measure signal-to-noise ratio (SNR).

---

### 6.5 High-Performance Repeated Transforms with FFTRun

When processing a streaming signal in frames (e.g., audio, sensor data), `FFTRun` eliminates per-transform overhead.

```rust
use fft_rs::FFTRun;

fn main() {
    let frame_size = 1024;

    // Create the plan ONCE — twiddle factors are pre-computed
    let plan = FFTRun::<f64>::new(frame_size).unwrap();

    // Simulate processing 10,000 frames (e.g., 10 seconds at ~1000 frames/sec)
    for frame_idx in 0..10_000 {
        // Each frame is a new signal
        let signal: Vec<f64> = (0..frame_size)
            .map(|i| {
                let t = (i + frame_idx * frame_size) as f64;
                (t * 0.01).sin() + (t * 0.037).cos()
            })
            .collect();

        // Transform using the pre-computed plan — no twiddle recomputation
        let spectrum = plan.fft(&signal);

        // Example: find the peak frequency
        let mut peak_bin = 0usize;
        let mut peak_mag = 0.0f64;
        for k in 1..frame_size / 2 {
            let mag = spectrum[k].magnitude();
            if mag > peak_mag {
                peak_mag = mag;
                peak_bin = k;
            }
        }

        if frame_idx < 3 {
            println!("Frame {:>5}: peak at bin {} (mag = {:.1})",
                     frame_idx, peak_bin, peak_mag);
        }
    }

    println!("Processed 10,000 frames successfully.");
}
```

**Why FFTRun is faster for repeated transforms**:

| Operation | `FFT<T>` (one-shot) | `FFTRun<T>` (planned) |
|-----------|---------------------|----------------------|
| Twiddle factor computation | On-the-fly per transform | Pre-computed once |
| Per-stage heap allocations | None (on-the-fly) | None (table lookup) |
| Buffer allocation | New `Vec` per `compute()` | Same pattern for `fft()` |
| In-place option | Not available | `process_inplace_fft` |

For the in-place path, `FFTRun` allocates **zero** additional memory per transform call.

---

### 6.6 Zero-Allocation In-Place Transforms

For memory-constrained or latency-critical applications, `FFTRun` provides in-place methods.

```rust
use fft_rs::{FFTRun, Complex64, IntoSample};

fn main() {
    let n = 4096;
    let plan = FFTRun::<f64>::new(n).unwrap();

    // Simulate a real-time audio processing loop
    let mut buffer: Vec<Complex64> = vec![Complex64::zero(); n];

    for frame in 0..100 {
        // Fill buffer with new frame data (in-place)
        for i in 0..n {
            let t = (i + frame * n) as f64;
            buffer[i] = ((t * 0.02).sin() + (t * 0.07).cos()).into_complex();
        }

        // Forward FFT — zero allocations, operates directly on buffer
        plan.process_inplace_fft(&mut buffer);

        // --- Frequency-domain processing (in-place) ---
        // Example: simple low-pass filter (zero out high frequencies)
        let cutoff = n / 8;
        for k in cutoff..n - cutoff {
            buffer[k] = Complex64::zero();
        }

        // Inverse FFT — zero allocations, restores time-domain
        plan.process_inplace_ifft(&mut buffer);

        // buffer now contains the filtered signal
    }

    println!("Processed 100 frames with zero per-frame allocations.");
}
```

**Memory profile**: The only heap allocation is the initial `buffer` of size `N`. All subsequent transforms use only stack memory.

---

## 7. Supported Input Types and Type Mapping

| Input Type `T` | Internal Precision | Output Type `T::Complex` | Typical Use |
|----------------|-------------------|--------------------------|-------------|
| `i32`          | f32               | `Complex32`              | Integer PCM audio (16-bit), sensor readings |
| `i64`          | f64               | `Complex64`              | High-precision integer data |
| `f32`          | f32               | `Complex32`              | Real-time audio, embedded systems |
| `f64`          | f64               | `Complex64`              | Scientific computing, high-precision analysis |

**Choosing precision**:
- Use `f32` / `Complex32` when memory bandwidth is the bottleneck (e.g., real-time audio on embedded devices).
- Use `f64` / `Complex64` when numerical accuracy matters (e.g., large N, ill-conditioned signals, iterative algorithms).

---

## 8. Constraints and Limits

| Constraint | Value | Reason |
|------------|-------|--------|
| Minimum length | 1 | Trivial FFT (identity) |
| Maximum length | 2²⁴ = 16,777,216 | Practical memory limit; defined by `MAX_FFT_SIZE` |
| Length requirement | Power of 2 | Cooley-Tukey radix-2 algorithm |
| Input ownership | `FFT::new()` takes `Vec<T>`; `FFTRun::fft()` takes `&[T]` | Different usage patterns |

**Non-power-of-two signals**: If your signal length is not a power of 2, you must **zero-pad** it to the next power of 2:

```rust
use fft_rs::{FFT, is_power_of_two};

fn next_power_of_two(n: usize) -> usize {
    if is_power_of_two(n) { n } else { 1usize << (usize::BITS as usize - n.leading_zeros() as usize) }
}

fn main() {
    let raw_signal = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];  // Length 7
    let padded_size = next_power_of_two(raw_signal.len());        // 8
    let mut padded = vec![0.0f64; padded_size];
    padded[..raw_signal.len()].copy_from_slice(&raw_signal);

    let fft = FFT::new(padded).unwrap();
    let spectrum = fft.compute();
    println!("Padded FFT of length {} computed.", padded_size);
}
```

---

## 9. Error Handling Guide

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `FftError::ZeroLength` | Empty input vector | Ensure the input has at least 1 element |
| `FftError::NotPowerOfTwo(n)` | Length is not 2^k | Zero-pad to the next power of 2 (see above) |
| `FftError::TooLarge(n)` | Length > 2²⁴ | Reduce the input size or process in chunks |
| `panic: input length X does not match plan size Y` | `FFTRun::fft()` called with wrong-length input | Ensure `input.len() == plan.n()` |

### Recommended Error Handling Pattern

```rust
use fft_rs::{FFT, FftError, FftResult};

fn process_signal(data: Vec<f64>) -> FftResult<Vec<fft_rs::Complex64>> {
    let fft = FFT::new(data)?;          // Propagate FftError
    Ok(fft.compute())
}

fn main() {
    match process_signal(vec![1.0, 2.0, 3.0]) {
        Ok(spectrum) => println!("Spectrum has {} bins", spectrum.len()),
        Err(e) => eprintln!("FFT failed: {}", e),
    }
    // FFT failed: FFT input length 3 is not a power of 2; length must be a positive power of 2
}
```

---

## 10. Performance Notes

### Algorithm

- **Cooley-Tukey radix-2 DIT**: O(N log N) time complexity
- **Iterative (not recursive)**: avoids stack overflow for large N
- **Bit-reversal via CPU instruction**: `usize::reverse_bits()` maps to a single `RBIT` (ARM) or equivalent x86 instruction

### Memory

- **One-shot `FFT<T>`**: Allocates one `Vec<T::Complex>` of size N per `compute()` call. Twiddle factors are computed on-the-fly — no per-stage allocations.
- **Planned `FFTRun<T>`**: Allocates twiddle tables once at construction. Each `fft()` / `ifft()` call allocates one output `Vec`. In-place methods (`process_inplace_*`) allocate **zero** additional memory.

### Twiddle Factor Strategy

The one-shot `FFT<T>` computes twiddle factors on-the-fly in the innermost loop using `f32::sin_cos()` / `f64::sin_cos()`, which computes both sine and cosine in a single call. This eliminates O(log N) heap allocations per transform. The `FFTRun<T>` pre-computes twiddle tables at construction time, trading a one-time cost for faster repeated transforms.

### Benchmark Guidance

For a rough comparison, processing N = 2²⁰ (≈1M) samples:

| Method | Approx. Time (single transform) |
|--------|-------------------------------|
| `FFT<f64>::compute()` | ~50-80 ms |
| `FFTRun<f64>::fft()` (warm) | ~45-70 ms |
| `FFTRun<f64>::process_inplace_fft()` | ~45-70 ms (no allocation overhead) |

Actual times depend on CPU, cache, and Rust optimization level. Always build with `--release` for production benchmarks.

---

## 11. Frequently Asked Questions

### Q: Can I pass complex input directly?

**A:** Not directly. The library accepts real-valued input (`i32`, `i64`, `f32`, `f64`) and converts it to complex internally. If you have complex input, use `FFTRun` with `process_inplace_fft`:

```rust
use fft_rs::{FFTRun, Complex64};

fn main() {
    let n = 256;
    let plan = FFTRun::<f64>::new(n).unwrap();

    // Prepare complex buffer
    let mut buffer: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.07).cos()))
        .collect();

    // Transform in-place
    plan.process_inplace_fft(&mut buffer);
}
```

### Q: Is the library thread-safe?

**A:** Yes. Both `FFT<T>` and `FFTRun<T>` are `Send + Sync` (their internal types are `Clone + Copy + Send + Sync`). You can share an `FFTRun` across threads via `Arc`:

```rust
use std::sync::Arc;
use fft_rs::FFTRun;

fn main() {
    let plan = Arc::new(FFTRun::<f64>::new(1024).unwrap());

    let handles: Vec<_> = (0..4).map(|thread_id| {
        let plan = Arc::clone(&plan);
        std::thread::spawn(move || {
            let signal: Vec<f64> = (0..1024)
                .map(|i| ((i + thread_id * 1024) as f64 * 0.01).sin())
                .collect();
            plan.fft(&signal)
        })
    }).collect();

    for handle in handles {
        let spectrum = handle.join().unwrap();
        println!("Thread spectrum peak: {:.1}", spectrum[1].magnitude());
    }
}
```

### Q: What happens if I call `compute()` multiple times on the same `FFT`?

**A:** It recomputes the transform from scratch each time. The input data is not modified. If you need repeated transforms at the same size, use `FFTRun` instead.

### Q: Does the IFFT normalize the output?

**A:** Yes. The IFFT divides by N, so `IFFT(FFT(x))` returns the original signal. The forward FFT does **not** apply any scaling.

### Q: Can I use this for real-valued FFT (RFFT) with half-size output?

**A:** Not directly. The library computes the full complex FFT. However, for real input you can exploit Hermitian symmetry to discard the redundant upper half:

```rust
use fft_rs::FFT;

fn main() {
    let n = 1024;
    let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.05).sin()).collect();

    let fft = FFT::new(signal).unwrap();
    let spectrum = fft.compute();

    // For real input, only the first N/2 + 1 bins are unique
    let unique_bins = n / 2 + 1;
    println!("Unique bins: {} out of {}", unique_bins, n);

    // The upper half is the complex conjugate of the lower half
    for k in 1..n / 2 {
        assert!((spectrum[k].re - spectrum[n - k].re).abs() < 1e-10);
        assert!((spectrum[k].im + spectrum[n - k].im).abs() < 1e-10);
    }
}
```

### Q: Why is the maximum size 2²⁴?

**A:** The limit of 16,777,216 elements balances memory feasibility with algorithmic correctness. At N = 2²⁴, an f64-based FFT requires approximately 128 MB for the complex buffer alone. Larger sizes are possible but may cause memory pressure or floating-point precision degradation in the twiddle factors.

---

*End of API User Manual — fft_rs v0.1.0*