//! Greeks: closed-form sensitivities under Black–Scholes plus finite
//! differences and portfolio aggregation.
//!
//! All formulas use continuously compounded dividend yield `q`. Theta is
//! returned per *year*; divide by 365 (calendar) or 252 (trading) for the
//! per-day decay traders quote. Vega is per unit of volatility (so 1.0
//! corresponds to a 100-vol-point move, *not* one vol point).

use crate::black_scholes::BlackScholesModel;
use crate::option_types::{OptionType, VanillaOption};
use crate::{norm_cdf, norm_pdf};

/// Bundle of all Greeks for a single instrument or a portfolio.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

impl Greeks {
    pub fn zero() -> Self {
        Self::default()
    }

    /// Element-wise scaling. Useful for `portfolio_greeks` aggregation.
    pub fn scaled(&self, factor: f64) -> Self {
        Self {
            delta: self.delta * factor,
            gamma: self.gamma * factor,
            theta: self.theta * factor,
            vega: self.vega * factor,
            rho: self.rho * factor,
            vanna: self.vanna * factor,
            volga: self.volga * factor,
        }
    }
}

impl std::ops::Add for Greeks {
    type Output = Greeks;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            delta: self.delta + rhs.delta,
            gamma: self.gamma + rhs.gamma,
            theta: self.theta + rhs.theta,
            vega: self.vega + rhs.vega,
            rho: self.rho + rhs.rho,
            vanna: self.vanna + rhs.vanna,
            volga: self.volga + rhs.volga,
        }
    }
}

impl std::ops::AddAssign for Greeks {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

/// Stateless namespace for Greeks computations.
pub struct GreeksCalculator;

impl GreeksCalculator {
    /// Closed-form Greeks under Black–Scholes(–Merton).
    pub fn calculate(bs: &BlackScholesModel, option_type: OptionType) -> Greeks {
        if bs.time_to_expiry <= 0.0 {
            return Greeks::zero();
        }
        let d1 = bs.d1();
        let d2 = bs.d2();
        let n_d1 = norm_cdf(d1);
        let n_d2 = norm_cdf(d2);
        let phi_d1 = norm_pdf(d1);
        let s = bs.spot;
        let k = bs.strike;
        let r = bs.risk_free_rate;
        let q = bs.dividend_yield;
        let sigma = bs.volatility;
        let t = bs.time_to_expiry;
        let sqrt_t = t.sqrt();
        let disc_q = (-q * t).exp();
        let disc_r = (-r * t).exp();

        let delta = match option_type {
            OptionType::Call => disc_q * n_d1,
            OptionType::Put => disc_q * (n_d1 - 1.0),
        };

        let gamma = disc_q * phi_d1 / (s * sigma * sqrt_t);

        let common_theta = -s * disc_q * phi_d1 * sigma / (2.0 * sqrt_t);
        let theta = match option_type {
            OptionType::Call => {
                common_theta - r * k * disc_r * n_d2 + q * s * disc_q * n_d1
            }
            OptionType::Put => {
                common_theta + r * k * disc_r * norm_cdf(-d2)
                    - q * s * disc_q * norm_cdf(-d1)
            }
        };

        let vega = s * disc_q * phi_d1 * sqrt_t;

        let rho = match option_type {
            OptionType::Call => k * t * disc_r * n_d2,
            OptionType::Put => -k * t * disc_r * norm_cdf(-d2),
        };

        // Cross-greeks. Both are sign-independent (do not depend on call vs put).
        let vanna = -disc_q * phi_d1 * d2 / sigma;
        let volga = vega * d1 * d2 / sigma;

        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna,
            volga,
        }
    }

    /// Compute Greeks numerically by central differences in spot, time and
    /// volatility. The pricer closure is called multiple times. Bumps:
    /// `h_spot = h * S`, `h_sigma = h`, `h_time = h`.
    pub fn finite_difference<F: Fn(f64, f64, f64) -> f64>(
        pricer: F,
        spot: f64,
        time: f64,
        sigma: f64,
        h: f64,
    ) -> Greeks {
        let h_s = h * spot.max(1.0);
        let p_up = pricer(spot + h_s, time, sigma);
        let p_dn = pricer(spot - h_s, time, sigma);
        let p_mid = pricer(spot, time, sigma);
        let delta = (p_up - p_dn) / (2.0 * h_s);
        let gamma = (p_up - 2.0 * p_mid + p_dn) / (h_s * h_s);

        let p_t_up = pricer(spot, time + h, sigma);
        let p_t_dn = pricer(spot, time - h, sigma);
        // Theta is dV/dt at the *current* moment; convention here is
        // dV/dt where t is calendar time, so dV/dT_remaining is its negative.
        let theta = -(p_t_up - p_t_dn) / (2.0 * h);

        let p_v_up = pricer(spot, time, sigma + h);
        let p_v_dn = pricer(spot, time, sigma - h);
        let vega = (p_v_up - p_v_dn) / (2.0 * h);

        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
        }
    }

    /// Sum of position-weighted Greeks. Each entry is `(option, quantity)`,
    /// with the option valued under the supplied Black–Scholes parameters
    /// (spot, rate, vol, dividend yield are taken from `template_bs`,
    /// while strike and expiry come from each `VanillaOption`).
    pub fn portfolio_greeks(
        positions: &[(VanillaOption, f64)],
        template_bs: &BlackScholesModel,
    ) -> Greeks {
        let mut acc = Greeks::zero();
        for (opt, qty) in positions {
            let bs = BlackScholesModel {
                strike: opt.strike,
                time_to_expiry: opt.expiry,
                ..*template_bs
            };
            let greeks = Self::calculate(&bs, opt.option_type);
            acc += greeks.scaled(*qty);
        }
        acc
    }
}

/// Internal helper: closed-form Vega without going through `Greeks`. Kept
/// `pub(crate)` because the Black–Scholes implied-volatility solver uses
/// it, and we want to avoid the extra computations the full Greeks bundle
/// performs.
#[inline]
pub(crate) fn vega_value(bs: &BlackScholesModel) -> f64 {
    if bs.time_to_expiry <= 0.0 {
        return 0.0;
    }
    let d1 = bs.d1();
    bs.spot * (-bs.dividend_yield * bs.time_to_expiry).exp() * norm_pdf(d1)
        * bs.time_to_expiry.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_types::{ExerciseStyle, VanillaOption};

    /// Closed-form Greeks must agree with finite differences to a few
    /// basis points for a normal ATM option.
    #[test]
    fn closed_form_matches_finite_difference() {
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
        let g = GreeksCalculator::calculate(&bs, OptionType::Call);

        let pricer = |s: f64, t: f64, sigma: f64| {
            let bsi = BlackScholesModel::new(s, 100.0, t, 0.05, sigma, 0.0).unwrap();
            bsi.price(OptionType::Call)
        };
        let g_fd = GreeksCalculator::finite_difference(pricer, 100.0, 1.0, 0.20, 1e-3);

        assert!((g.delta - g_fd.delta).abs() < 1e-4, "{} vs {}", g.delta, g_fd.delta);
        assert!((g.gamma - g_fd.gamma).abs() < 1e-3);
        assert!((g.vega - g_fd.vega).abs() < 1e-2);
        assert!((g.theta - g_fd.theta).abs() < 1e-2);
    }

    /// Call Δ ∈ (0, e^{-qT}); Put Δ ∈ (-e^{-qT}, 0); Δ_call − Δ_put = e^{-qT}.
    #[test]
    fn delta_bounds_and_parity() {
        let bs = BlackScholesModel::new(100.0, 110.0, 0.5, 0.03, 0.30, 0.02).unwrap();
        let gc = GreeksCalculator::calculate(&bs, OptionType::Call);
        let gp = GreeksCalculator::calculate(&bs, OptionType::Put);
        let disc_q = (-bs.dividend_yield * bs.time_to_expiry).exp();
        assert!(gc.delta > 0.0 && gc.delta < disc_q);
        assert!(gp.delta > -disc_q && gp.delta < 0.0);
        assert!((gc.delta - gp.delta - disc_q).abs() < 1e-9);
    }

    /// Gamma and Vega are equal for calls and puts on the same strike.
    #[test]
    fn gamma_vega_call_put_equal() {
        let bs = BlackScholesModel::new(100.0, 110.0, 0.5, 0.03, 0.30, 0.02).unwrap();
        let gc = GreeksCalculator::calculate(&bs, OptionType::Call);
        let gp = GreeksCalculator::calculate(&bs, OptionType::Put);
        assert!((gc.gamma - gp.gamma).abs() < 1e-12);
        assert!((gc.vega - gp.vega).abs() < 1e-12);
    }

    #[test]
    fn portfolio_greeks_aggregate_linearly() {
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        let call = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            100.0,
            1.0,
            "X",
        )
        .unwrap();
        let put = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::European,
            100.0,
            1.0,
            "X",
        )
        .unwrap();

        let agg = GreeksCalculator::portfolio_greeks(
            &[(call.clone(), 2.0), (put.clone(), -1.0)],
            &bs,
        );

        let g_call = GreeksCalculator::calculate(&bs, OptionType::Call).scaled(2.0);
        let g_put = GreeksCalculator::calculate(&bs, OptionType::Put).scaled(-1.0);
        let expected = g_call + g_put;
        assert!((agg.delta - expected.delta).abs() < 1e-12);
        assert!((agg.gamma - expected.gamma).abs() < 1e-12);
        assert!((agg.vega - expected.vega).abs() < 1e-12);
    }
}
