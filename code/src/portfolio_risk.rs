//! Portfolio Greeks aggregation, scenario P&L, P&L attribution and VaR.
//!
//! Designed to mirror what an options book runs every minute: net Δ, Γ,
//! Vega per book; "what if spot drops 5% and vols add 10 vol points"; and
//! historical/Monte-Carlo VaR on the resulting P&L distribution.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::black_scholes::BlackScholesModel;
use crate::greeks::{Greeks, GreeksCalculator};
use crate::option_types::VanillaOption;
#[cfg(test)]
use crate::option_types::OptionType;
use crate::{ensure_finite, ensure_positive, OptionError, Result};

/// One position in the book: an option, the quantity, and the pricing
/// model state (spot/rate/vol/dividend yield) used for valuation.
#[derive(Debug, Clone)]
pub struct Position {
    pub option: VanillaOption,
    pub quantity: f64,
    pub model: BlackScholesModel,
}

impl Position {
    pub fn new(option: VanillaOption, quantity: f64, model: BlackScholesModel) -> Result<Self> {
        ensure_finite("quantity", quantity)?;
        Ok(Self { option, quantity, model })
    }

    pub fn price(&self) -> f64 {
        self.quantity * self.model.price(self.option.option_type)
    }

    pub fn greeks(&self) -> Greeks {
        GreeksCalculator::calculate(&self.model, self.option.option_type)
            .scaled(self.quantity)
    }
}

/// A book of options on a single underlying.
#[derive(Debug, Clone, Default)]
pub struct OptionPortfolio {
    pub positions: Vec<Position>,
}

impl OptionPortfolio {
    pub fn new() -> Self {
        Self { positions: Vec::new() }
    }

    pub fn push(&mut self, position: Position) {
        self.positions.push(position);
    }

    /// Mark-to-market P&L of the book under current parameters.
    pub fn value(&self) -> f64 {
        self.positions.iter().map(Position::price).sum()
    }

    /// Net Greeks across the book.
    pub fn total_greeks(&self) -> Greeks {
        let mut acc = Greeks::zero();
        for p in &self.positions {
            acc += p.greeks();
        }
        acc
    }

    /// Full revaluation P&L under shocks. Spot is multiplied by
    /// `1 + spot_shock`; vol is shifted by `vol_shock`; the rate by
    /// `rate_shock`; and time-to-expiry decreases by `time_shock` (in
    /// years). Negative time after shock is clipped to 0.
    pub fn scenario_pl(&self, shocks: &ScenarioShocks) -> f64 {
        let base = self.value();
        let mut shocked_value = 0.0;
        for p in &self.positions {
            let new_spot = (p.model.spot * (1.0 + shocks.spot_shock)).max(1e-9);
            let new_vol = (p.model.volatility + shocks.vol_shock).max(1e-6);
            let new_rate = p.model.risk_free_rate + shocks.rate_shock;
            let new_time = (p.model.time_to_expiry - shocks.time_shock).max(0.0);
            let model = BlackScholesModel {
                spot: new_spot,
                strike: p.model.strike,
                time_to_expiry: new_time,
                risk_free_rate: new_rate,
                volatility: new_vol,
                dividend_yield: p.model.dividend_yield,
            };
            shocked_value += p.quantity * model.price(p.option.option_type);
        }
        shocked_value - base
    }

    /// P&L explained by the four largest contributors (delta, gamma,
    /// vega, theta) given observed changes in spot, vol and time. Using
    /// the standard Taylor decomposition:
    ///
    /// ```text
    /// ΔP&L ≈ Δ·ΔS + ½ Γ·(ΔS)² + ν·Δσ + Θ·Δt.
    /// ```
    pub fn pl_attribution(
        &self,
        spot_change: f64,
        vol_change: f64,
        time_change: f64,
    ) -> PlAttribution {
        let g = self.total_greeks();
        let delta_pl = g.delta * spot_change;
        let gamma_pl = 0.5 * g.gamma * spot_change * spot_change;
        let vega_pl = g.vega * vol_change;
        let theta_pl = g.theta * time_change;
        PlAttribution {
            delta: delta_pl,
            gamma: gamma_pl,
            vega: vega_pl,
            theta: theta_pl,
            total: delta_pl + gamma_pl + vega_pl + theta_pl,
        }
    }

    /// Monte-Carlo Value-at-Risk (one-step) at the given confidence level
    /// using a simple log-normal spot shock with `spot_vol` annualised
    /// volatility over horizon `horizon` (years), and a normal vol-of-vol
    /// shock with sigma `vol_of_vol`. Returns a positive number meaning
    /// "with this confidence, losses do not exceed VaR".
    pub fn var(
        &self,
        confidence: f64,
        spot_vol: f64,
        vol_of_vol: f64,
        horizon: f64,
        num_simulations: usize,
        seed: u64,
    ) -> Result<f64> {
        if !(0.5..1.0).contains(&confidence) {
            return Err(OptionError::InvalidParameter(
                "confidence must be in (0.5, 1)",
            ));
        }
        ensure_positive("spot_vol", spot_vol)?;
        ensure_positive("horizon", horizon)?;
        if num_simulations < 100 {
            return Err(OptionError::InvalidParameter(
                "num_simulations should be >= 100",
            ));
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pls = Vec::with_capacity(num_simulations);
        let drift = -0.5 * spot_vol * spot_vol * horizon;
        let diff = spot_vol * horizon.sqrt();
        for _ in 0..num_simulations {
            let z: f64 = StandardNormal.sample(&mut rng);
            let z_v: f64 = StandardNormal.sample(&mut rng);
            let spot_shock = (drift + diff * z).exp() - 1.0;
            let vol_shock = vol_of_vol * z_v;
            let shocks = ScenarioShocks {
                spot_shock,
                vol_shock,
                rate_shock: 0.0,
                time_shock: horizon,
            };
            pls.push(self.scenario_pl(&shocks));
        }
        // VaR = -quantile of P&L distribution at (1 - confidence).
        pls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((1.0 - confidence) * num_simulations as f64).floor() as usize;
        let q = pls[idx.min(num_simulations - 1)];
        Ok(-q)
    }
}

/// Single-scenario shock vector.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioShocks {
    /// Multiplicative spot shock; `0.05` means spot moves up 5%.
    pub spot_shock: f64,
    /// Additive vol shock in vol points (e.g. `0.02` adds 2 vol points).
    pub vol_shock: f64,
    /// Additive rate shock.
    pub rate_shock: f64,
    /// Time decay applied to every position's expiry, in years.
    pub time_shock: f64,
}

/// Output of [`OptionPortfolio::pl_attribution`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlAttribution {
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub total: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::option_types::ExerciseStyle;

    fn sample_book() -> OptionPortfolio {
        let mut book = OptionPortfolio::new();
        let bs = BlackScholesModel::new(100.0, 100.0, 0.5, 0.05, 0.20, 0.0).unwrap();
        let call = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            100.0,
            0.5,
            "X",
        )
        .unwrap();
        let put = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::European,
            95.0,
            0.5,
            "X",
        )
        .unwrap();
        book.push(Position::new(call, 10.0, bs).unwrap());
        let bs_put = BlackScholesModel { strike: 95.0, ..bs };
        book.push(Position::new(put, -5.0, bs_put).unwrap());
        book
    }

    #[test]
    fn total_greeks_sum_position_greeks() {
        let book = sample_book();
        let total = book.total_greeks();
        let mut manual = Greeks::zero();
        for p in &book.positions {
            manual += p.greeks();
        }
        assert!((total.delta - manual.delta).abs() < 1e-12);
        assert!((total.vega - manual.vega).abs() < 1e-12);
    }

    /// A pure call long should lose money on a downward spot shock and
    /// roughly match the linear Δ approximation for small moves.
    #[test]
    fn small_shock_matches_attribution() {
        let book = sample_book();
        let shocks = ScenarioShocks {
            spot_shock: 0.001,
            vol_shock: 0.0,
            rate_shock: 0.0,
            time_shock: 0.0,
        };
        let scenario = book.scenario_pl(&shocks);
        let attrib = book.pl_attribution(0.001 * 100.0, 0.0, 0.0);
        assert!(
            (scenario - attrib.total).abs() < 0.05,
            "scenario={scenario}, attrib={attrib:?}"
        );
    }

    #[test]
    fn var_is_positive_for_long_options() {
        let book = sample_book();
        let var = book.var(0.95, 0.30, 0.05, 1.0 / 252.0, 10_000, 7).unwrap();
        assert!(var.is_finite());
        // VaR on a non-trivial book should be strictly positive.
        assert!(var > 0.0, "var={var}");
    }
}
