//! # Options and Greeks for Algorithmic Trading
//!
//! Compact, well-documented implementations of the core quantitative tools
//! discussed in Chapter 8:
//!
//! - [`option_types`] — vanilla option payoffs and exercise styles
//! - [`black_scholes`] — closed-form European option pricing and implied volatility
//! - [`greeks`] — Delta, Gamma, Theta, Vega, Rho, Vanna, Volga
//! - [`binomial_tree`] — Cox–Ross–Rubinstein tree for European and American options
//! - [`finite_difference`] — Explicit, Implicit and Crank–Nicolson PDE solvers
//! - [`monte_carlo`] — European Monte Carlo pricing and Longstaff–Schwartz for American options
//! - [`heston_model`] — Stochastic volatility characteristic function and Carr–Madan FFT pricing
//! - [`portfolio_risk`] — Portfolio Greeks aggregation, scenarios, P&L attribution and VaR
//!
//! ## Quick example
//!
//! ```rust
//! use options_greeks::black_scholes::BlackScholesModel;
//! use options_greeks::option_types::OptionType;
//!
//! let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
//! let call = bs.price(OptionType::Call);
//! let put = bs.price(OptionType::Put);
//! // Put–call parity: C - P = S - K e^{-rT}
//! let parity = call - put - (bs.spot - bs.strike * (-bs.risk_free_rate * bs.time_to_expiry).exp());
//! assert!(parity.abs() < 1e-9);
//! ```

pub mod binomial_tree;
pub mod black_scholes;
pub mod finite_difference;
pub mod greeks;
pub mod heston_model;
pub mod monte_carlo;
pub mod option_types;
pub mod portfolio_risk;

pub use binomial_tree::BinomialTree;
pub use black_scholes::BlackScholesModel;
pub use finite_difference::{FdScheme, FiniteDifferencePricer};
pub use greeks::{Greeks, GreeksCalculator};
pub use heston_model::{CalibrationResult, HestonModel};
pub use monte_carlo::{LongstaffSchwartz, MonteCarloPricer};
pub use option_types::{ExerciseStyle, OptionType, VanillaOption};
pub use portfolio_risk::{
    OptionPortfolio, PlAttribution, Position, ScenarioShocks,
};

/// Crate-level result type used by all numerical routines.
pub type Result<T> = std::result::Result<T, OptionError>;

/// Errors returned when input is malformed or the computation is not
/// well-defined (e.g. negative volatility, non-finite spot, no convergence).
#[derive(Debug, Clone, PartialEq)]
pub enum OptionError {
    EmptyInput,
    NonFiniteInput(&'static str),
    InvalidParameter(&'static str),
    DimensionMismatch { expected: usize, actual: usize },
    NoConvergence(&'static str),
    OutOfArbitrageBounds,
}

impl std::fmt::Display for OptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input must not be empty"),
            Self::NonFiniteInput(name) => write!(f, "{name} contains NaN or infinite values"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NoConvergence(msg) => write!(f, "no convergence: {msg}"),
            Self::OutOfArbitrageBounds => write!(f, "market price violates arbitrage bounds"),
        }
    }
}

impl std::error::Error for OptionError {}

pub(crate) fn ensure_finite(name: &'static str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OptionError::NonFiniteInput(name))
    }
}

pub(crate) fn ensure_positive(name: &'static str, value: f64) -> Result<()> {
    ensure_finite(name, value)?;
    if value <= 0.0 {
        Err(OptionError::InvalidParameter(name))
    } else {
        Ok(())
    }
}

pub(crate) fn ensure_non_negative(name: &'static str, value: f64) -> Result<()> {
    ensure_finite(name, value)?;
    if value < 0.0 {
        Err(OptionError::InvalidParameter(name))
    } else {
        Ok(())
    }
}

/// Standard normal probability density function `φ(x)`.
#[inline]
pub fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Standard normal cumulative distribution function `Φ(x)` via the
/// Abramowitz–Stegun rational approximation built on `erf`.
///
/// Maximum absolute error of the underlying `erf` approximation is below
/// 1.5e-7, which is more than enough for educational option-pricing work.
#[inline]
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function via Abramowitz–Stegun 7.1.26 rational approximation.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_cdf_known_values() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 1e-9);
        // Φ(1) ≈ 0.8413447
        assert!((norm_cdf(1.0) - 0.841_344_746).abs() < 1e-6);
        // Φ(-1) = 1 - Φ(1)
        assert!((norm_cdf(-1.0) - (1.0 - 0.841_344_746)).abs() < 1e-6);
        // Tail
        assert!(norm_cdf(8.0) > 1.0 - 1e-12);
        assert!(norm_cdf(-8.0) < 1e-12);
    }

    #[test]
    fn norm_pdf_known_values() {
        // φ(0) = 1/sqrt(2π) ≈ 0.398942
        assert!((norm_pdf(0.0) - 0.398_942_280).abs() < 1e-7);
        assert!((norm_pdf(1.0) - 0.241_970_724).abs() < 1e-7);
        assert!((norm_pdf(-1.0) - 0.241_970_724).abs() < 1e-7);
    }
}
