//! Vanilla option payoffs and exercise styles.
//!
//! A *vanilla* option is the simplest contract: the holder of a call has
//! the right to buy the underlying at `strike`, the holder of a put the
//! right to sell. Exercise can be European (expiry only), American (any
//! time), or Bermudan (a discrete set of dates).

use crate::{ensure_non_negative, ensure_positive, OptionError, Result};

/// Whether the option is a call or a put.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    Call,
    Put,
}

impl OptionType {
    /// `+1` for a call, `-1` for a put. Useful as the sign multiplier in
    /// closed-form pricing formulas.
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            OptionType::Call => 1.0,
            OptionType::Put => -1.0,
        }
    }
}

/// When the option may be exercised.
#[derive(Debug, Clone, PartialEq)]
pub enum ExerciseStyle {
    European,
    American,
    /// A finite, sorted list of exercise times in years.
    Bermudan(Vec<f64>),
}

impl Eq for ExerciseStyle {}

/// A vanilla European/American/Bermudan option on a single underlying.
#[derive(Debug, Clone)]
pub struct VanillaOption {
    pub option_type: OptionType,
    pub exercise: ExerciseStyle,
    pub strike: f64,
    pub expiry: f64,
    pub underlying: String,
}

impl VanillaOption {
    /// Build an option, validating the strike/expiry/exercise dates.
    pub fn new(
        option_type: OptionType,
        exercise: ExerciseStyle,
        strike: f64,
        expiry: f64,
        underlying: impl Into<String>,
    ) -> Result<Self> {
        ensure_positive("strike", strike)?;
        ensure_non_negative("expiry", expiry)?;
        if let ExerciseStyle::Bermudan(dates) = &exercise {
            if dates.is_empty() {
                return Err(OptionError::EmptyInput);
            }
            for d in dates {
                ensure_non_negative("bermudan_date", *d)?;
                if *d > expiry {
                    return Err(OptionError::InvalidParameter(
                        "bermudan dates must be <= expiry",
                    ));
                }
            }
            for w in dates.windows(2) {
                if w[0] > w[1] {
                    return Err(OptionError::InvalidParameter(
                        "bermudan dates must be sorted",
                    ));
                }
            }
        }
        Ok(Self {
            option_type,
            exercise,
            strike,
            expiry,
            underlying: underlying.into(),
        })
    }

    /// Terminal payoff `max(S - K, 0)` for a call, `max(K - S, 0)` for a put.
    #[inline]
    pub fn payoff(&self, spot: f64) -> f64 {
        match self.option_type {
            OptionType::Call => (spot - self.strike).max(0.0),
            OptionType::Put => (self.strike - spot).max(0.0),
        }
    }

    /// Intrinsic value: identical to `payoff` for vanillas, but kept as a
    /// distinct method because for path-dependent options it diverges.
    #[inline]
    pub fn intrinsic_value(&self, spot: f64) -> f64 {
        self.payoff(spot)
    }

    /// Time value of the option given the market price: `price - intrinsic`.
    /// May be negative for deep in-the-money American options shortly
    /// before exercise (then early exercise is optimal).
    #[inline]
    pub fn time_value(&self, spot: f64, price: f64) -> f64 {
        price - self.intrinsic_value(spot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payoff_call_and_put() {
        let call = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        assert!((call.payoff(120.0) - 20.0).abs() < 1e-12);
        assert!(call.payoff(80.0).abs() < 1e-12);

        let put = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::European,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        assert!((put.payoff(80.0) - 20.0).abs() < 1e-12);
        assert!(put.payoff(120.0).abs() < 1e-12);
    }

    #[test]
    fn time_value_decomposition() {
        let call = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        // Hypothetical market price.
        let price = 12.5;
        let spot = 105.0;
        let intrinsic = call.intrinsic_value(spot);
        let time = call.time_value(spot, price);
        assert!((intrinsic + time - price).abs() < 1e-12);
    }

    #[test]
    fn invalid_strike_rejected() {
        let err = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            -1.0,
            1.0,
            "TEST",
        )
        .unwrap_err();
        assert_eq!(err, OptionError::InvalidParameter("strike"));
    }

    #[test]
    fn bermudan_dates_must_be_sorted_and_in_range() {
        let err = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::Bermudan(vec![1.5, 0.5]),
            100.0,
            2.0,
            "TEST",
        )
        .unwrap_err();
        assert_eq!(
            err,
            OptionError::InvalidParameter("bermudan dates must be sorted")
        );

        let err = VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::Bermudan(vec![0.5, 3.0]),
            100.0,
            2.0,
            "TEST",
        )
        .unwrap_err();
        assert_eq!(
            err,
            OptionError::InvalidParameter("bermudan dates must be <= expiry")
        );
    }
}
