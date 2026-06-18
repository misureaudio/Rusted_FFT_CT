//! # fft_rs
//!
//! A high-performance Fast Fourier Transform (FFT) library for Rust.
//!
//! Supports `i32`, `i64`, `f32`, and `f64` input samples with vectors
//! up to 2^24 elements. Length must be a positive power of 2.
//!
//! ## Quick Start
//!
//! ```
//! use fft_rs_ma::{FFT, Complex64};
//!
//! let input = vec![1.0f64, 2.0, 3.0, 4.0];
//! let fft = FFT::new(input).unwrap();
//! let spectrum = fft.compute();
//! ```
//!
//! ## Supported Input Types
//!
//! - `i32` → internal `f32`, output `Complex32`
//! - `i64` → internal `f64`, output `Complex64`
//! - `f32` → internal `f32`, output `Complex32`
//! - `f64` → internal `f64`, output `Complex64`
//!
//! ## Error Handling
//!
//! All public constructors return `Result<T, FftError>`. Errors include:
//! - `ZeroLength` – input length is 0
//! - `NotPowerOfTwo` – input length is not a power of 2
//! - `TooLarge` – input length exceeds 2^24

// we like mathematical loops
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod complex;
pub mod fft_core;
pub mod plan;

pub use error::{FftError, FftResult, is_power_of_two};
pub use complex::{Complex32, Complex64};
pub use fft_core::{FFT, IntoSample};
pub use plan::FFTRun;
