//! Heston (1993) stochastic volatility model with characteristic function
//! and Carr–Madan FFT pricing.
//!
//! ```text
//! dS = μ S dt + √v · S dW₁
//! dv = κ (θ - v) dt + ξ √v dW₂
//! corr(dW₁, dW₂) = ρ.
//! ```
//!
//! Heston's closed-form characteristic function for `ln S_T` admits a
//! Carr–Madan style Fourier inversion that prices an entire strip of
//! strikes at once. Calibration in this file is a small Nelder–Mead style
//! fit over `(v0, κ, θ, ξ, ρ)` minimising squared implied-vol error.

use num_complex::Complex64;

use crate::black_scholes::BlackScholesModel;
use crate::option_types::OptionType;
use crate::{ensure_finite, ensure_non_negative, ensure_positive, OptionError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HestonModel {
    pub spot: f64,
    pub strike: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    /// Initial variance.
    pub v0: f64,
    /// Mean-reversion speed.
    pub kappa: f64,
    /// Long-run variance.
    pub theta: f64,
    /// Vol-of-vol.
    pub xi: f64,
    /// Correlation between spot and variance Brownian motions.
    pub rho: f64,
}

/// Result of a calibration call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibrationResult {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    pub xi: f64,
    pub rho: f64,
    pub final_loss: f64,
    pub iterations: usize,
}

impl HestonModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        dividend_yield: f64,
        v0: f64,
        kappa: f64,
        theta: f64,
        xi: f64,
        rho: f64,
    ) -> Result<Self> {
        ensure_positive("spot", spot)?;
        ensure_positive("strike", strike)?;
        ensure_positive("time_to_expiry", time_to_expiry)?;
        ensure_finite("risk_free_rate", risk_free_rate)?;
        ensure_non_negative("dividend_yield", dividend_yield)?;
        ensure_non_negative("v0", v0)?;
        ensure_positive("kappa", kappa)?;
        ensure_non_negative("theta", theta)?;
        ensure_positive("xi", xi)?;
        if !(-1.0..=1.0).contains(&rho) {
            return Err(OptionError::InvalidParameter("rho must be in [-1, 1]"));
        }
        Ok(Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            dividend_yield,
            v0,
            kappa,
            theta,
            xi,
            rho,
        })
    }

    /// Heston characteristic function of the *log forward log-price*
    /// `x_T = ln(S_T / S_0) - (r - q) T`, evaluated at `u`. This is the
    /// "centered" form most convenient for Carr–Madan / Lewis pricing
    /// because the deterministic drift has been factored out.
    pub fn characteristic_fn(&self, u: Complex64) -> Complex64 {
        let i = Complex64::new(0.0, 1.0);
        let t = self.time_to_expiry;
        let kappa = self.kappa;
        let theta = self.theta;
        let xi = self.xi;
        let rho = self.rho;
        let v0 = self.v0;
        let xi2 = xi * xi;

        // Heston (1993) characteristic function in the "little trap"
        // formulation of Albrecher et al. (2007), which avoids the
        // well-known branch-cut bug at long maturities by using
        // `g = (b − d) / (b + d)` rather than the original `(b + d) /
        // (b − d)`. The centred form prices `x_T = ln(S_T/F_0)`
        // directly, so the deterministic drift `(r − q) T` does not
        // appear here — it is reintroduced explicitly in `price`.
        let b = kappa + rho * xi * i * u;
        let inner = b * b + xi2 * (i * u + u * u);
        let d = inner.sqrt();
        let g = (b - d) / (b + d);
        let exp_dt = (-d * t).exp();

        let c = (kappa * theta / xi2)
            * ((b - d) * t - 2.0 * ((1.0 - g * exp_dt) / (1.0 - g)).ln());
        let d_term = (b - d) / xi2 * ((1.0 - exp_dt) / (1.0 - g * exp_dt));

        (c + d_term * v0).exp()
    }

    /// Price a European option via the Lewis (2001) Fourier integral,
    /// using the centred characteristic function above.
    ///
    /// ```text
    /// C = S e^{-qT}
    ///   - (√(SK) e^{-(r+q)T/2} / π) ∫_0^∞ Re(e^{i u k} φ(u - i/2)) / (u² + 1/4) du,
    /// ```
    ///
    /// with `k = ln(K) - ln(F)` and `F = S e^{(r-q)T}`.
    pub fn price(&self, option_type: OptionType) -> f64 {
        let r = self.risk_free_rate;
        let q = self.dividend_yield;
        let t = self.time_to_expiry;
        let s = self.spot;
        let k = self.strike;
        let i = Complex64::new(0.0, 1.0);

        let f = s * ((r - q) * t).exp();
        let kappa_log = (k / f).ln();

        let u_max = 200.0;
        let num_points = 4096;
        let du = u_max / num_points as f64;
        let mut sum = 0.0;
        for n in 0..num_points {
            let u_lo = n as f64 * du;
            let u_hi = (n as f64 + 1.0) * du;
            let f_lo = lewis_integrand(self, u_lo, kappa_log, i);
            let f_hi = lewis_integrand(self, u_hi, kappa_log, i);
            sum += 0.5 * (f_lo + f_hi) * du;
        }

        let coeff = (s * k).sqrt() * (-(r + q) * t / 2.0).exp() / std::f64::consts::PI;
        let call = s * (-q * t).exp() - coeff * sum;
        match option_type {
            OptionType::Call => call,
            // Put–call parity.
            OptionType::Put => call - s * (-q * t).exp() + k * (-r * t).exp(),
        }
    }

    /// Implied Black–Scholes volatility for the Heston price at this
    /// strike/maturity. Convenience for IV-surface diagnostics.
    pub fn implied_volatility(&self, option_type: OptionType) -> Result<f64> {
        let price = self.price(option_type);
        let bs = BlackScholesModel::new(
            self.spot,
            self.strike,
            self.time_to_expiry,
            self.risk_free_rate,
            0.2,
            self.dividend_yield,
        )?;
        bs.implied_volatility(price, option_type)
    }

    /// Coordinate-descent calibration over `(v0, κ, θ, ξ, ρ)` against an
    /// implied-volatility surface. Each market quote is
    /// `(strike, maturity, market_iv)`. Loss is the sum of squared
    /// model-vs-market IV differences.
    ///
    /// This is intentionally small and educational: 1-D line searches per
    /// parameter, repeated for `max_iter` cycles. For production work
    /// replace with Levenberg–Marquardt on prices, with analytic Jacobians.
    pub fn calibrate(
        &mut self,
        market_quotes: &[(f64, f64, f64)],
        max_iter: usize,
    ) -> Result<CalibrationResult> {
        if market_quotes.is_empty() {
            return Err(OptionError::EmptyInput);
        }
        let loss = |m: &HestonModel| -> f64 {
            let mut s = 0.0;
            for &(k, t, iv) in market_quotes {
                let mut probe = *m;
                probe.strike = k;
                probe.time_to_expiry = t;
                let model_iv = probe.implied_volatility(OptionType::Call).unwrap_or(iv);
                let diff = model_iv - iv;
                s += diff * diff;
            }
            s
        };

        let mut best = *self;
        let mut best_loss = loss(&best);
        let mut iterations = 0;

        for _ in 0..max_iter {
            iterations += 1;
            let mut improved = false;
            // Try multiplicative bumps on each positive parameter and
            // additive bumps on rho.
            for factor in [0.9, 1.1] {
                for &name in &["v0", "kappa", "theta", "xi"] {
                    let mut probe = best;
                    match name {
                        "v0" => probe.v0 = (best.v0 * factor).max(1e-6),
                        "kappa" => probe.kappa = (best.kappa * factor).max(1e-6),
                        "theta" => probe.theta = (best.theta * factor).max(1e-6),
                        "xi" => probe.xi = (best.xi * factor).max(1e-6),
                        _ => unreachable!(),
                    }
                    let l = loss(&probe);
                    if l < best_loss {
                        best_loss = l;
                        best = probe;
                        improved = true;
                    }
                }
            }
            for delta in [-0.05, 0.05] {
                let mut probe = best;
                probe.rho = (best.rho + delta).clamp(-0.99, 0.99);
                let l = loss(&probe);
                if l < best_loss {
                    best_loss = l;
                    best = probe;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        *self = best;
        Ok(CalibrationResult {
            v0: best.v0,
            kappa: best.kappa,
            theta: best.theta,
            xi: best.xi,
            rho: best.rho,
            final_loss: best_loss,
            iterations,
        })
    }
}

fn lewis_integrand(model: &HestonModel, u: f64, k_log: f64, i: Complex64) -> f64 {
    let shifted = Complex64::new(u, -0.5);
    let phi = model.characteristic_fn(shifted);
    let num = (i * u * k_log).exp() * phi;
    num.re / (u * u + 0.25)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As ξ → 0 the variance is deterministic and the Heston model collapses
    /// to Black–Scholes with constant volatility √v0. We pick (v0=θ) so the
    /// variance literally is constant.
    #[test]
    fn heston_collapses_to_black_scholes_when_xi_small() {
        let v0 = 0.04;
        let h = HestonModel::new(
            100.0, 100.0, 1.0, 0.05, 0.0, v0, 2.0, v0, 1e-4, 0.0,
        )
        .unwrap();
        let h_call = h.price(OptionType::Call);
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, v0.sqrt(), 0.0).unwrap();
        let bs_call = bs.price(OptionType::Call);
        assert!((h_call - bs_call).abs() < 0.05, "h={h_call}, bs={bs_call}");
    }

    /// Negative correlation creates a downward-sloping IV skew (out-of-the-
    /// money puts more expensive in vol terms than OTM calls).
    #[test]
    fn negative_rho_produces_skew() {
        let make = |k: f64| {
            HestonModel::new(
                100.0, k, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7,
            )
            .unwrap()
        };
        let iv_low = make(80.0).implied_volatility(OptionType::Put).unwrap();
        let iv_atm = make(100.0).implied_volatility(OptionType::Call).unwrap();
        let iv_high = make(120.0).implied_volatility(OptionType::Call).unwrap();
        assert!(iv_low > iv_atm, "low IV {iv_low} not > ATM {iv_atm}");
        assert!(iv_atm > iv_high, "ATM IV {iv_atm} not > high {iv_high}");
    }

    #[test]
    fn calibration_reduces_loss() {
        // Synthetic surface: generate quotes from a known Heston, perturb
        // initial guess, and check calibration moves us back closer.
        let true_h = HestonModel::new(
            100.0, 100.0, 1.0, 0.03, 0.0, 0.04, 1.5, 0.04, 0.4, -0.5,
        )
        .unwrap();
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let mut quotes = Vec::new();
        for &k in &strikes {
            let mut probe = true_h;
            probe.strike = k;
            let iv = probe.implied_volatility(OptionType::Call).unwrap();
            quotes.push((k, 1.0, iv));
        }
        let mut guess = HestonModel::new(
            100.0, 100.0, 1.0, 0.03, 0.0, 0.06, 1.0, 0.06, 0.6, -0.2,
        )
        .unwrap();
        let initial_loss = {
            let mut sum = 0.0;
            for &(k, t, iv) in &quotes {
                let mut p = guess;
                p.strike = k;
                p.time_to_expiry = t;
                let model_iv = p.implied_volatility(OptionType::Call).unwrap_or(iv);
                sum += (model_iv - iv).powi(2);
            }
            sum
        };
        let result = guess.calibrate(&quotes, 60).unwrap();
        assert!(result.final_loss < initial_loss);
    }
}
