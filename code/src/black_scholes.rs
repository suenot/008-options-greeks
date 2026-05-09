//! Black–Scholes–Merton closed-form European option pricing.
//!
//! The model assumes the underlying follows geometric Brownian motion with
//! constant drift `r - q` and constant volatility `σ`. With dividends it is
//! Merton's extension; for `q = 0` it reduces to the original Black–Scholes
//! formulas of 1973:
//!
//! ```text
//! C = S e^{-qT} Φ(d1) - K e^{-rT} Φ(d2)
//! P = K e^{-rT} Φ(-d2) - S e^{-qT} Φ(-d1)
//!
//! d1 = [ln(S/K) + (r - q + σ²/2) T] / (σ √T)
//! d2 = d1 - σ √T
//! ```
//!
//! Implied volatility is recovered from a market price by Newton–Raphson on
//! Vega with a bisection fallback for robustness.

use crate::greeks::vega_value;
use crate::option_types::OptionType;
use crate::{
    ensure_finite, ensure_non_negative, ensure_positive, norm_cdf, OptionError, Result,
};

/// Inputs for the Black–Scholes formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlackScholesModel {
    pub spot: f64,
    pub strike: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub dividend_yield: f64,
}

impl BlackScholesModel {
    /// Build a validated model. Spot, strike and volatility must be
    /// strictly positive; time, rate and dividend yield must be finite and
    /// non-negative for time.
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
        dividend_yield: f64,
    ) -> Result<Self> {
        ensure_positive("spot", spot)?;
        ensure_positive("strike", strike)?;
        ensure_non_negative("time_to_expiry", time_to_expiry)?;
        ensure_finite("risk_free_rate", risk_free_rate)?;
        ensure_positive("volatility", volatility)?;
        ensure_finite("dividend_yield", dividend_yield)?;
        Ok(Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        })
    }

    /// `d1` from the Black–Scholes formula.
    #[inline]
    pub fn d1(&self) -> f64 {
        let s = self.spot;
        let k = self.strike;
        let r = self.risk_free_rate;
        let q = self.dividend_yield;
        let sigma = self.volatility;
        let t = self.time_to_expiry;
        if t <= 0.0 {
            return f64::INFINITY * (s / k).ln().signum();
        }
        ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
    }

    /// `d2 = d1 - σ √T`.
    #[inline]
    pub fn d2(&self) -> f64 {
        self.d1() - self.volatility * self.time_to_expiry.sqrt()
    }

    /// Black–Scholes(–Merton) price of a European call or put.
    ///
    /// At `T = 0` the formula degenerates to the intrinsic payoff
    /// `max(S - K, 0)` (or `max(K - S, 0)` for puts).
    pub fn price(&self, option_type: OptionType) -> f64 {
        if self.time_to_expiry == 0.0 {
            return match option_type {
                OptionType::Call => (self.spot - self.strike).max(0.0),
                OptionType::Put => (self.strike - self.spot).max(0.0),
            };
        }
        let d1 = self.d1();
        let d2 = self.d2();
        let s_disc = self.spot * (-self.dividend_yield * self.time_to_expiry).exp();
        let k_disc = self.strike * (-self.risk_free_rate * self.time_to_expiry).exp();
        match option_type {
            OptionType::Call => s_disc * norm_cdf(d1) - k_disc * norm_cdf(d2),
            OptionType::Put => k_disc * norm_cdf(-d2) - s_disc * norm_cdf(-d1),
        }
    }

    /// Implied volatility solving `BS(σ) = market_price` via Newton–Raphson
    /// on Vega, with a bisection fallback that always converges as long as
    /// the market price is inside the no-arbitrage bounds
    /// `max(S e^{-qT} - K e^{-rT}, 0) ≤ C ≤ S e^{-qT}` (analogous for puts).
    pub fn implied_volatility(
        &self,
        market_price: f64,
        option_type: OptionType,
    ) -> Result<f64> {
        ensure_finite("market_price", market_price)?;
        if self.time_to_expiry <= 0.0 {
            return Err(OptionError::InvalidParameter("time_to_expiry must be > 0"));
        }
        let s_disc = self.spot * (-self.dividend_yield * self.time_to_expiry).exp();
        let k_disc = self.strike * (-self.risk_free_rate * self.time_to_expiry).exp();
        let (lower, upper) = match option_type {
            OptionType::Call => ((s_disc - k_disc).max(0.0), s_disc),
            OptionType::Put => ((k_disc - s_disc).max(0.0), k_disc),
        };
        if market_price < lower - 1e-10 || market_price > upper + 1e-10 {
            return Err(OptionError::OutOfArbitrageBounds);
        }

        let mut sigma = 0.2_f64.max((2.0 * (market_price / self.spot).abs() / self.time_to_expiry.sqrt()).sqrt());
        sigma = sigma.clamp(1e-4, 5.0);
        for _ in 0..100 {
            let mut candidate = *self;
            candidate.volatility = sigma;
            let price = candidate.price(option_type);
            let diff = price - market_price;
            if diff.abs() < 1e-8 {
                return Ok(sigma);
            }
            let v = vega_value(&candidate);
            if v < 1e-10 {
                break; // fall back to bisection
            }
            let next = sigma - diff / v;
            if !next.is_finite() || next <= 0.0 || next > 5.0 {
                break;
            }
            sigma = next;
        }
        // Bisection fallback.
        let mut lo = 1e-6;
        let mut hi = 5.0;
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let mut candidate = *self;
            candidate.volatility = mid;
            let price = candidate.price(option_type);
            if (price - market_price).abs() < 1e-9 {
                return Ok(mid);
            }
            if price < market_price {
                lo = mid;
            } else {
                hi = mid;
            }
            if (hi - lo) < 1e-10 {
                return Ok(mid);
            }
        }
        Err(OptionError::NoConvergence("implied_volatility"))
    }

    /// Forward price `F = S e^{(r - q) T}`.
    #[inline]
    pub fn forward(&self) -> f64 {
        self.spot * ((self.risk_free_rate - self.dividend_yield) * self.time_to_expiry).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values from Hull "Options, Futures, and Other Derivatives",
    /// 10th ed. Example 15.6: S=42, K=40, r=10%, σ=20%, T=0.5.
    /// Call ≈ 4.7594, Put ≈ 0.8086.
    #[test]
    fn hull_textbook_example() {
        let bs = BlackScholesModel::new(42.0, 40.0, 0.5, 0.10, 0.20, 0.0).unwrap();
        let call = bs.price(OptionType::Call);
        let put = bs.price(OptionType::Put);
        assert!((call - 4.759_422).abs() < 1e-3, "call={call}");
        assert!((put - 0.808_600).abs() < 1e-3, "put={put}");
    }

    #[test]
    fn put_call_parity_with_and_without_dividends() {
        for q in [0.0, 0.02, 0.05] {
            let bs = BlackScholesModel::new(100.0, 105.0, 0.75, 0.04, 0.25, q).unwrap();
            let c = bs.price(OptionType::Call);
            let p = bs.price(OptionType::Put);
            let parity =
                c - p - bs.spot * (-q * bs.time_to_expiry).exp()
                    + bs.strike * (-bs.risk_free_rate * bs.time_to_expiry).exp();
            assert!(parity.abs() < 1e-9, "q={q}, parity={parity}");
        }
    }

    #[test]
    fn intrinsic_at_expiry() {
        let bs = BlackScholesModel::new(120.0, 100.0, 0.0, 0.05, 0.2, 0.0).unwrap();
        assert!((bs.price(OptionType::Call) - 20.0).abs() < 1e-12);
        assert!(bs.price(OptionType::Put).abs() < 1e-12);
    }

    #[test]
    fn implied_volatility_round_trip() {
        for sigma in [0.10, 0.20, 0.35, 0.60] {
            let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, sigma, 0.0).unwrap();
            let price = bs.price(OptionType::Call);
            let bs_ivguess = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.5, 0.0).unwrap();
            let iv = bs_ivguess.implied_volatility(price, OptionType::Call).unwrap();
            assert!((iv - sigma).abs() < 1e-5, "expected={sigma}, got={iv}");
        }
    }

    #[test]
    fn implied_volatility_out_of_bounds() {
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        // Negative price is below lower bound for both calls and puts.
        let err = bs.implied_volatility(-1.0, OptionType::Call).unwrap_err();
        assert_eq!(err, OptionError::OutOfArbitrageBounds);
    }
}
