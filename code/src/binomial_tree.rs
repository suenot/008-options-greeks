//! Cox–Ross–Rubinstein (CRR) binomial tree for European and American
//! options with continuously compounded dividends.
//!
//! Each step the spot moves up by `u = e^{σ √Δt}` or down by `d = 1/u`
//! with risk-neutral probability
//!
//! ```text
//! p = (e^{(r - q) Δt} - d) / (u - d).
//! ```
//!
//! Backward induction gives the European value; for American we additionally
//! check `max(continuation, intrinsic)` at every node.

use crate::greeks::Greeks;
use crate::option_types::OptionType;
use crate::{ensure_non_negative, ensure_positive, OptionError, Result};

/// Configuration for a CRR binomial tree pricer.
#[derive(Debug, Clone, Copy)]
pub struct BinomialTree {
    pub steps: usize,
    pub spot: f64,
    pub strike: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub dividend_yield: f64,
}

impl BinomialTree {
    pub fn new(
        steps: usize,
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
        dividend_yield: f64,
    ) -> Result<Self> {
        if steps < 1 {
            return Err(OptionError::InvalidParameter("steps must be >= 1"));
        }
        ensure_positive("spot", spot)?;
        ensure_positive("strike", strike)?;
        ensure_positive("time_to_expiry", time_to_expiry)?;
        ensure_positive("volatility", volatility)?;
        ensure_non_negative("dividend_yield", dividend_yield)?;
        Ok(Self {
            steps,
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield,
        })
    }

    fn parameters(&self) -> (f64, f64, f64, f64) {
        let dt = self.time_to_expiry / self.steps as f64;
        let u = (self.volatility * dt.sqrt()).exp();
        let d = 1.0 / u;
        let p = (((self.risk_free_rate - self.dividend_yield) * dt).exp() - d) / (u - d);
        (dt, u, d, p)
    }

    /// Backward induction. If `american` is true, intrinsic is checked at
    /// every intermediate node; otherwise pure European.
    fn price_with(&self, option_type: OptionType, american: bool) -> f64 {
        let (dt, u, _d, p) = self.parameters();
        let disc = (-self.risk_free_rate * dt).exp();
        let n = self.steps;
        let sign = option_type.sign();

        // Terminal payoffs.
        let mut values = Vec::with_capacity(n + 1);
        for i in 0..=n {
            // Spot at terminal node `i` ups, `n - i` downs.
            let st = self.spot * u.powi(2 * i as i32 - n as i32);
            let payoff = (sign * (st - self.strike)).max(0.0);
            values.push(payoff);
        }

        for step in (0..n).rev() {
            for i in 0..=step {
                let cont = disc * (p * values[i + 1] + (1.0 - p) * values[i]);
                if american {
                    let st = self.spot * u.powi(2 * i as i32 - step as i32);
                    let intrinsic = (sign * (st - self.strike)).max(0.0);
                    values[i] = cont.max(intrinsic);
                } else {
                    values[i] = cont;
                }
            }
        }
        values[0]
    }

    /// Price a European option on the tree.
    pub fn european_price(&self, option_type: OptionType) -> f64 {
        self.price_with(option_type, false)
    }

    /// Price an American option on the tree.
    pub fn american_price(&self, option_type: OptionType) -> f64 {
        self.price_with(option_type, true)
    }

    /// Greeks via finite differences on the tree itself: bump spot, vol
    /// and time and re-price. Theta uses one fewer step per side.
    pub fn greeks(&self, option_type: OptionType, american: bool) -> Greeks {
        let h_s = 0.01 * self.spot;
        let h_t = (self.time_to_expiry / self.steps as f64).max(1e-4);
        let h_v = 1e-3;
        let pricer = |spot: f64, time: f64, sigma: f64| {
            let mut tree = *self;
            tree.spot = spot;
            tree.time_to_expiry = time;
            tree.volatility = sigma;
            if american {
                tree.american_price(option_type)
            } else {
                tree.european_price(option_type)
            }
        };
        let p_up = pricer(self.spot + h_s, self.time_to_expiry, self.volatility);
        let p_dn = pricer(self.spot - h_s, self.time_to_expiry, self.volatility);
        let p_mid = pricer(self.spot, self.time_to_expiry, self.volatility);
        let delta = (p_up - p_dn) / (2.0 * h_s);
        let gamma = (p_up - 2.0 * p_mid + p_dn) / (h_s * h_s);

        let p_t_up = pricer(self.spot, self.time_to_expiry + h_t, self.volatility);
        let p_t_dn = pricer(self.spot, self.time_to_expiry - h_t, self.volatility);
        let theta = -(p_t_up - p_t_dn) / (2.0 * h_t);

        let p_v_up = pricer(self.spot, self.time_to_expiry, self.volatility + h_v);
        let p_v_dn = pricer(self.spot, self.time_to_expiry, self.volatility - h_v);
        let vega = (p_v_up - p_v_dn) / (2.0 * h_v);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes::BlackScholesModel;

    /// CRR European price converges to Black–Scholes as steps grow.
    #[test]
    fn european_converges_to_black_scholes() {
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
        let bs_price = bs.price(OptionType::Call);
        let tree = BinomialTree::new(800, 100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
        let crr_price = tree.european_price(OptionType::Call);
        assert!(
            (crr_price - bs_price).abs() < 0.05,
            "BS={bs_price}, CRR={crr_price}"
        );
    }

    /// American puts are worth more than European puts (early exercise
    /// premium); American calls without dividends equal European calls.
    #[test]
    fn american_put_premium_and_call_equality_no_dividend() {
        let tree = BinomialTree::new(500, 100.0, 100.0, 1.0, 0.10, 0.30, 0.0).unwrap();
        let am_put = tree.american_price(OptionType::Put);
        let eu_put = tree.european_price(OptionType::Put);
        assert!(am_put >= eu_put - 1e-9, "{am_put} < {eu_put}");
        assert!(am_put > eu_put + 1e-3, "no early-exercise premium");

        let am_call = tree.american_price(OptionType::Call);
        let eu_call = tree.european_price(OptionType::Call);
        assert!((am_call - eu_call).abs() < 1e-3);
    }
}
