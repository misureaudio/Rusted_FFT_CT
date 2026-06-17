//! Error types for fft_rs.

use std::fmt;

/// Maximum supported FFT size: 2^24
pub const MAX_FFT_SIZE: usize = 1 << 24;

/// Errors that can occur during FFT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FftError {
    /// Input length is zero.
    ZeroLength,
    /// Input length is not a power of two.
    NotPowerOfTwo(usize),
    /// Input length exceeds 2^24.
    TooLarge(usize),
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FftError::ZeroLength => write!(f, "FFT input length is zero; length must be a positive power of 2"),
            FftError::NotPowerOfTwo(len) => write!(
                f,
                "FFT input length {} is not a power of 2; length must be a positive power of 2",
                len
            ),
            FftError::TooLarge(len) => write!(
                f,
                "FFT input length {} exceeds maximum supported size 2^24 ({}); reduce input size",
                len, MAX_FFT_SIZE
            ),
        }
    }
}

impl std::error::Error for FftError {}

/// Convenience type alias for Results using FftError.
pub type FftResult<T> = Result<T, FftError>;

/// Validate that `len` is a legal FFT size.
///
/// Returns `Err` if `len == 0`, `len` is not a power of 2, or `len > 2^24`.
pub fn validate_length(len: usize) -> FftResult<()> {
    if len == 0 {
        return Err(FftError::ZeroLength);
    }
    if len > MAX_FFT_SIZE {
        return Err(FftError::TooLarge(len));
    }
    if !len.is_power_of_two() {
        return Err(FftError::NotPowerOfTwo(len));
    }
    Ok(())
}

/// Check whether `n` is a power of two.
///
/// This is a convenience wrapper around `usize::is_power_of_two()` (Proposal 5).
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n.is_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(4));
        assert!(is_power_of_two(1024));
        assert!(is_power_of_two(1 << 24));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(5));
        assert!(!is_power_of_two(6));
        assert!(!is_power_of_two(7));
        assert!(!is_power_of_two(15));
    }

    #[test]
    fn test_validate_length_zero() {
        assert_eq!(validate_length(0), Err(FftError::ZeroLength));
    }

    #[test]
    fn test_validate_length_not_power_of_two() {
        assert_eq!(validate_length(6), Err(FftError::NotPowerOfTwo(6)));
    }

    #[test]
    fn test_validate_length_too_large() {
        assert_eq!(validate_length(1 << 25), Err(FftError::TooLarge(1 << 25)));
    }

    #[test]
    fn test_validate_length_ok() {
        assert!(validate_length(1).is_ok());
        assert!(validate_length(1024).is_ok());
        assert!(validate_length(1 << 24).is_ok());
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", FftError::ZeroLength),
            "FFT input length is zero; length must be a positive power of 2"
        );
        assert_eq!(
            format!("{}", FftError::NotPowerOfTwo(6)),
            "FFT input length 6 is not a power of 2; length must be a positive power of 2"
        );
        assert!(format!("{}", FftError::TooLarge(1 << 25)).contains("2^24"));
    }
}