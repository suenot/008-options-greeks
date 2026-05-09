//! Monte Carlo and Longstaff–Schwartz pricers.
//!
//! Geometric Brownian motion paths are simulated under the risk-neutral
//! measure with the exact log-normal step:
//!
//! ```text
//! S_{t+Δt} = S_t exp((r - q - ½σ²) Δt + σ √Δt · Z),  Z ~ N(0, 1).
//! ```
//!
//! For European options only the terminal price is needed. For American
//! options Longstaff–Schwartz (2001) approximates the continuation value
//! by least-squares regression on simple polynomial basis functions of
//! the spot.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::option_types::VanillaOption;
use crate::{ensure_positive, OptionError, Result};

/// Configuration for the European Monte Carlo pricer.
#[derive(Debug, Clone, Copy)]
pub struct MonteCarloPricer {
    pub num_paths: usize,
    pub seed: u64,
}

impl MonteCarloPricer {
    pub fn new(num_paths: usize, seed: u64) -> Result<Self> {
        if num_paths == 0 {
            return Err(OptionError::InvalidParameter("num_paths must be > 0"));
        }
        Ok(Self { num_paths, seed })
    }

    /// Price a European option by simulating terminal prices and
    /// discounting the average payoff. Uses antithetic variates for
    /// variance reduction.
    pub fn european_price(
        &self,
        option: &VanillaOption,
        spot: f64,
        rate: f64,
        sigma: f64,
        dividend_yield: f64,
    ) -> Result<f64> {
        ensure_positive("spot", spot)?;
        ensure_positive("sigma", sigma)?;
        if option.expiry <= 0.0 {
            return Err(OptionError::InvalidParameter("expiry must be > 0"));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let drift = (rate - dividend_yield - 0.5 * sigma * sigma) * option.expiry;
        let diff = sigma * option.expiry.sqrt();

        let pairs = self.num_paths / 2;
        let mut sum = 0.0;
        for _ in 0..pairs {
            let z: f64 = StandardNormal.sample(&mut rng);
            let s_plus = spot * (drift + diff * z).exp();
            let s_minus = spot * (drift - diff * z).exp();
            sum += option.payoff(s_plus) + option.payoff(s_minus);
        }
        let n = (pairs * 2) as f64;
        let mean = sum / n;
        Ok((-rate * option.expiry).exp() * mean)
    }
}

/// Longstaff–Schwartz Monte Carlo for American (and Bermudan) options.
#[derive(Debug, Clone, Copy)]
pub struct LongstaffSchwartz {
    pub num_paths: usize,
    pub num_time_steps: usize,
    pub basis_functions: usize,
    pub seed: u64,
}

impl LongstaffSchwartz {
    pub fn new(
        num_paths: usize,
        num_time_steps: usize,
        basis_functions: usize,
        seed: u64,
    ) -> Result<Self> {
        if num_paths < 16 {
            return Err(OptionError::InvalidParameter("num_paths must be >= 16"));
        }
        if num_time_steps == 0 {
            return Err(OptionError::InvalidParameter("num_time_steps must be > 0"));
        }
        if !(1..=4).contains(&basis_functions) {
            return Err(OptionError::InvalidParameter(
                "basis_functions must be 1..=4 (1, x, x^2, x^3)",
            ));
        }
        Ok(Self {
            num_paths,
            num_time_steps,
            basis_functions,
            seed,
        })
    }

    /// Simulate `num_paths` GBM paths on a uniform time grid and price
    /// the option by backward induction with regression-based
    /// continuation values.
    pub fn american_price(
        &self,
        option: &VanillaOption,
        spot: f64,
        rate: f64,
        sigma: f64,
        dividend_yield: f64,
    ) -> Result<f64> {
        ensure_positive("spot", spot)?;
        ensure_positive("sigma", sigma)?;
        if option.expiry <= 0.0 {
            return Err(OptionError::InvalidParameter("expiry must be > 0"));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = self.num_paths;
        let m = self.num_time_steps;
        let dt = option.expiry / m as f64;
        let drift = (rate - dividend_yield - 0.5 * sigma * sigma) * dt;
        let diff = sigma * dt.sqrt();

        // Paths matrix: rows = path, cols = time step (0..=m).
        let mut paths = vec![vec![0.0; m + 1]; n];
        // Use antithetic pairing for the terminal layer too.
        let pairs = n / 2;
        for p in 0..pairs {
            paths[2 * p][0] = spot;
            paths[2 * p + 1][0] = spot;
            for t in 1..=m {
                let z: f64 = StandardNormal.sample(&mut rng);
                paths[2 * p][t] = paths[2 * p][t - 1] * (drift + diff * z).exp();
                paths[2 * p + 1][t] = paths[2 * p + 1][t - 1] * (drift - diff * z).exp();
            }
        }
        if n % 2 == 1 {
            paths[n - 1][0] = spot;
            for t in 1..=m {
                let z: f64 = StandardNormal.sample(&mut rng);
                paths[n - 1][t] = paths[n - 1][t - 1] * (drift + diff * z).exp();
            }
        }

        // Cashflows and exercise indicator: time of exercise per path.
        let mut cashflow = vec![0.0_f64; n];
        let mut exercise_t = vec![m; n];
        for i in 0..n {
            cashflow[i] = option.payoff(paths[i][m]);
        }

        let disc = (-rate * dt).exp();
        let k = self.basis_functions;
        // Backward induction over time steps m-1 ... 1 (no early exercise at t=0
        // since intrinsic equals current payoff which equals immediate exercise
        // value; we still compare it at the end).
        for t in (1..m).rev() {
            // Collect in-the-money paths at time t.
            let mut itm: Vec<usize> = Vec::with_capacity(n);
            for (i, p) in paths.iter().enumerate().take(n) {
                if option.payoff(p[t]) > 1e-12 {
                    itm.push(i);
                }
            }
            if itm.len() < (k + 1) {
                continue;
            }
            // Regression: discount cashflow back from exercise time to t.
            let mut x_mat = vec![vec![0.0; k]; itm.len()];
            let mut y = vec![0.0; itm.len()];
            for (row, &i) in itm.iter().enumerate() {
                let s = paths[i][t];
                for j in 0..k {
                    x_mat[row][j] = s.powi(j as i32);
                }
                let dt_diff = (exercise_t[i] - t) as f64;
                y[row] = cashflow[i] * (-rate * dt_diff * dt).exp();
            }
            let coeffs = match normal_equations(&x_mat, &y) {
                Ok(c) => c,
                Err(_) => continue,
            };
            // Decide exercise per ITM path.
            for &i in &itm {
                let s = paths[i][t];
                let mut continuation = 0.0;
                for j in 0..k {
                    continuation += coeffs[j] * s.powi(j as i32);
                }
                let intrinsic = option.payoff(s);
                if intrinsic > continuation {
                    cashflow[i] = intrinsic;
                    exercise_t[i] = t;
                }
            }
            // Discount one step for paths *not* re-cashed at t (handled below
            // by the per-path discount above using exercise_t at the time of
            // the decision).
            let _ = disc;
        }

        // Discount each cashflow from its exercise time to 0.
        let mut sum = 0.0;
        for i in 0..n {
            let t_ex = exercise_t[i] as f64;
            sum += cashflow[i] * (-rate * t_ex * dt).exp();
        }
        let lsm_price = sum / n as f64;

        // Compare against immediate exercise at t=0 — for deep ITM American
        // options the optimal action at inception may be to exercise now.
        let immediate = option.payoff(spot);
        Ok(lsm_price.max(immediate))
    }
}

/// Solve `Aᵀ A x = Aᵀ y` via Gauss elimination on the small `k×k` normal
/// equation matrix. Used by the LSM regression.
fn normal_equations(x_mat: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>> {
    let n = x_mat.len();
    if n == 0 {
        return Err(OptionError::EmptyInput);
    }
    let k = x_mat[0].len();

    let mut ata = vec![vec![0.0; k]; k];
    let mut aty = vec![0.0; k];
    for row in 0..n {
        for i in 0..k {
            aty[i] += x_mat[row][i] * y[row];
            for j in 0..k {
                ata[i][j] += x_mat[row][i] * x_mat[row][j];
            }
        }
    }

    // Gaussian elimination with partial pivoting on the augmented matrix.
    let mut aug = vec![vec![0.0; k + 1]; k];
    for i in 0..k {
        for j in 0..k {
            aug[i][j] = ata[i][j];
        }
        aug[i][k] = aty[i];
    }
    for i in 0..k {
        let mut pivot = i;
        for r in (i + 1)..k {
            if aug[r][i].abs() > aug[pivot][i].abs() {
                pivot = r;
            }
        }
        if aug[pivot][i].abs() < 1e-12 {
            return Err(OptionError::NoConvergence("singular regression"));
        }
        aug.swap(i, pivot);
        for r in (i + 1)..k {
            let factor = aug[r][i] / aug[i][i];
            for c in i..=k {
                aug[r][c] -= factor * aug[i][c];
            }
        }
    }
    let mut sol = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = aug[i][k];
        for j in (i + 1)..k {
            sum -= aug[i][j] * sol[j];
        }
        sol[i] = sum / aug[i][i];
    }
    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes::BlackScholesModel;
    use crate::option_types::{ExerciseStyle, OptionType};

    fn euro_call() -> VanillaOption {
        VanillaOption::new(
            OptionType::Call,
            ExerciseStyle::European,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap()
    }

    #[test]
    fn european_mc_close_to_black_scholes() {
        let mc = MonteCarloPricer::new(200_000, 42).unwrap();
        let v = mc.european_price(&euro_call(), 100.0, 0.05, 0.2, 0.0).unwrap();
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        let target = bs.price(OptionType::Call);
        assert!((v - target).abs() < 0.2, "MC={v}, BS={target}");
    }

    #[test]
    fn lsm_american_put_at_least_european_value() {
        let opt = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::American,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        let lsm = LongstaffSchwartz::new(20_000, 50, 3, 11).unwrap();
        let am = lsm.american_price(&opt, 100.0, 0.10, 0.30, 0.0).unwrap();

        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.10, 0.30, 0.0).unwrap();
        let eu = bs.price(OptionType::Put);
        // LSM is a *lower* bound for the American value; we just want it
        // to be at least roughly the European value, allowing for noise.
        assert!(am >= eu - 0.5, "LSM={am}, EU={eu}");
    }
}
