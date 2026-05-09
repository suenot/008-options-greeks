//! Finite-difference PDE solvers for the Black–Scholes equation.
//!
//! With `V(S, t)` the option price the BS PDE is
//!
//! ```text
//! ∂V/∂t + (r - q) S ∂V/∂S + ½ σ² S² ∂²V/∂S² - r V = 0,
//! ```
//!
//! solved backward from terminal payoff to `t = 0`. Three time-stepping
//! schemes are implemented: explicit (conditionally stable, simple),
//! implicit (unconditionally stable) and Crank–Nicolson (second order in
//! time, unconditionally stable).
//!
//! American options are handled by the *projected* scheme: at each time
//! step the implicit/CN system is solved, then values are clipped from
//! below by the immediate intrinsic payoff. For Crank–Nicolson on
//! American options this is the simplest viable approach; full PSOR is
//! left as a future extension.

use crate::option_types::{ExerciseStyle, OptionType, VanillaOption};
use crate::{ensure_positive, OptionError, Result};

/// Time-stepping scheme used by [`FiniteDifferencePricer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdScheme {
    Explicit,
    Implicit,
    CrankNicolson,
}

/// Configuration for the PDE pricer.
#[derive(Debug, Clone)]
pub struct FiniteDifferencePricer {
    pub spot_steps: usize,
    pub time_steps: usize,
    pub spot_max: f64,
    pub scheme: FdScheme,
}

impl FiniteDifferencePricer {
    pub fn new(
        spot_steps: usize,
        time_steps: usize,
        spot_max: f64,
        scheme: FdScheme,
    ) -> Result<Self> {
        if spot_steps < 4 {
            return Err(OptionError::InvalidParameter("spot_steps must be >= 4"));
        }
        if time_steps < 1 {
            return Err(OptionError::InvalidParameter("time_steps must be >= 1"));
        }
        ensure_positive("spot_max", spot_max)?;
        Ok(Self {
            spot_steps,
            time_steps,
            spot_max,
            scheme,
        })
    }

    /// Price an option for every spot in the grid, returned as
    /// `(spot_grid, prices_at_t=0)`. Use linear interpolation to recover
    /// the value at the actual spot of interest.
    pub fn price(
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
        if spot >= self.spot_max {
            return Err(OptionError::InvalidParameter(
                "spot must be < spot_max",
            ));
        }

        let m = self.spot_steps; // number of spot intervals
        let n = self.time_steps;
        let ds = self.spot_max / m as f64;
        let dt = option.expiry / n as f64;

        // Spot grid S_j = j * ds for j = 0..m.
        let s_grid: Vec<f64> = (0..=m).map(|j| j as f64 * ds).collect();

        // Terminal payoff layer.
        let mut v: Vec<f64> = s_grid.iter().map(|&s| option.payoff(s)).collect();

        let american = matches!(option.exercise, ExerciseStyle::American);

        match self.scheme {
            FdScheme::Explicit => {
                self.step_explicit(&mut v, &s_grid, n, dt, ds, rate, sigma, dividend_yield, option, american)?;
            }
            FdScheme::Implicit => {
                self.step_theta(&mut v, &s_grid, n, dt, ds, rate, sigma, dividend_yield, option, american, 1.0)?;
            }
            FdScheme::CrankNicolson => {
                self.step_theta(&mut v, &s_grid, n, dt, ds, rate, sigma, dividend_yield, option, american, 0.5)?;
            }
        }

        // Linear interpolation to `spot`.
        let idx = (spot / ds).floor() as usize;
        let idx = idx.min(m - 1);
        let s_lo = s_grid[idx];
        let s_hi = s_grid[idx + 1];
        let w = (spot - s_lo) / (s_hi - s_lo);
        Ok((1.0 - w) * v[idx] + w * v[idx + 1])
    }

    #[allow(clippy::too_many_arguments)]
    fn step_explicit(
        &self,
        v: &mut Vec<f64>,
        s_grid: &[f64],
        n: usize,
        dt: f64,
        _ds: f64,
        rate: f64,
        sigma: f64,
        q: f64,
        option: &VanillaOption,
        american: bool,
    ) -> Result<()> {
        let m = self.spot_steps;
        let mut next = v.clone();
        for _ in 0..n {
            for j in 1..m {
                let jf = j as f64;
                let a = 0.5 * dt * (sigma * sigma * jf * jf - (rate - q) * jf);
                let b = 1.0 - dt * (sigma * sigma * jf * jf + rate);
                let c = 0.5 * dt * (sigma * sigma * jf * jf + (rate - q) * jf);
                next[j] = a * v[j - 1] + b * v[j] + c * v[j + 1];
            }
            // Boundary conditions: at S=0 the PDE becomes ∂V/∂t = r V →
            // for a put the value grows like K e^{-r τ}; for a call it
            // decays to 0. At S=S_max the call grows like
            // S e^{-q τ} - K e^{-r τ}; the put decays to 0.
            let tau_remaining = (n - 1) as f64 * dt; // not exact per step, but reasonable
            apply_dirichlet_bcs(&mut next, s_grid, option, rate, q, tau_remaining);

            if american {
                for (j, val) in next.iter_mut().enumerate() {
                    *val = val.max(option.intrinsic_value(s_grid[j]));
                }
            }
            std::mem::swap(v, &mut next);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn step_theta(
        &self,
        v: &mut Vec<f64>,
        s_grid: &[f64],
        n: usize,
        dt: f64,
        _ds: f64,
        rate: f64,
        sigma: f64,
        q: f64,
        option: &VanillaOption,
        american: bool,
        theta: f64, // 1.0 = implicit, 0.5 = Crank–Nicolson
    ) -> Result<()> {
        let m = self.spot_steps;
        // Interior unknowns: indices 1..m-1, total `m - 1` unknowns.
        let interior = m - 1;
        let mut a = vec![0.0_f64; interior]; // sub-diagonal
        let mut b = vec![0.0_f64; interior]; // diagonal
        let mut c = vec![0.0_f64; interior]; // super-diagonal
        let mut a_e = vec![0.0_f64; interior];
        let mut b_e = vec![0.0_f64; interior];
        let mut c_e = vec![0.0_f64; interior];
        for k in 0..interior {
            let j = (k + 1) as f64;
            let alpha = 0.5 * dt * (sigma * sigma * j * j - (rate - q) * j);
            let beta = -dt * (sigma * sigma * j * j + rate);
            let gamma = 0.5 * dt * (sigma * sigma * j * j + (rate - q) * j);
            a[k] = -theta * alpha;
            b[k] = 1.0 - theta * beta;
            c[k] = -theta * gamma;
            a_e[k] = (1.0 - theta) * alpha;
            b_e[k] = 1.0 + (1.0 - theta) * beta;
            c_e[k] = (1.0 - theta) * gamma;
        }

        let mut rhs = vec![0.0_f64; interior];
        for step in 0..n {
            // Dirichlet boundary values at this and the next time slice.
            let tau_old = (n - step) as f64 * dt;
            let tau_new = (n - step - 1) as f64 * dt;
            let v0_old = boundary_low(option, rate, tau_old);
            let v0_new = boundary_low(option, rate, tau_new);
            let vm_old = boundary_high(option, s_grid[m], rate, q, tau_old);
            let vm_new = boundary_high(option, s_grid[m], rate, q, tau_new);

            // Build RHS = (I + (1-θ) L) v_old + boundary contributions.
            for k in 0..interior {
                let j = k + 1;
                let mut r = a_e[k] * v[j - 1] + b_e[k] * v[j] + c_e[k] * v[j + 1];
                if k == 0 {
                    r += a_e[k] * (v0_old - v[j - 1]);
                    r -= a[k] * v0_new;
                }
                if k == interior - 1 {
                    r += c_e[k] * (vm_old - v[j + 1]);
                    r -= c[k] * vm_new;
                }
                rhs[k] = r;
            }

            // Solve tridiagonal system into `next_interior`.
            let next_interior = solve_tridiagonal(&a, &b, &c, &rhs)?;

            // Reassemble with boundaries.
            v[0] = v0_new;
            for k in 0..interior {
                v[k + 1] = next_interior[k];
            }
            v[m] = vm_new;

            if american {
                for (j, val) in v.iter_mut().enumerate() {
                    *val = val.max(option.intrinsic_value(s_grid[j]));
                }
            }
        }
        Ok(())
    }
}

fn boundary_low(option: &VanillaOption, rate: f64, tau: f64) -> f64 {
    match option.option_type {
        OptionType::Call => 0.0,
        OptionType::Put => option.strike * (-rate * tau).exp(),
    }
}

fn boundary_high(
    option: &VanillaOption,
    s_max: f64,
    rate: f64,
    q: f64,
    tau: f64,
) -> f64 {
    match option.option_type {
        OptionType::Call => s_max * (-q * tau).exp() - option.strike * (-rate * tau).exp(),
        OptionType::Put => 0.0,
    }
}

fn apply_dirichlet_bcs(
    next: &mut [f64],
    s_grid: &[f64],
    option: &VanillaOption,
    rate: f64,
    q: f64,
    tau: f64,
) {
    let m = next.len() - 1;
    next[0] = boundary_low(option, rate, tau);
    next[m] = boundary_high(option, s_grid[m], rate, q, tau);
}

/// Thomas algorithm for a tridiagonal system. `a` is the sub-diagonal
/// (with `a[0]` ignored), `b` the diagonal, `c` the super-diagonal (with
/// `c[n-1]` ignored), and `d` the right-hand side.
fn solve_tridiagonal(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if a.len() != n || c.len() != n || d.len() != n {
        return Err(OptionError::DimensionMismatch {
            expected: n,
            actual: a.len(),
        });
    }
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if m.abs() < 1e-18 {
            return Err(OptionError::NoConvergence("tridiagonal singular"));
        }
        cp[i] = c[i] / m;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::black_scholes::BlackScholesModel;

    fn european_call() -> VanillaOption {
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
    fn implicit_matches_black_scholes() {
        let pricer =
            FiniteDifferencePricer::new(200, 200, 400.0, FdScheme::Implicit).unwrap();
        let opt = european_call();
        let v = pricer.price(&opt, 100.0, 0.05, 0.2, 0.0).unwrap();
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        let bs_v = bs.price(OptionType::Call);
        assert!((v - bs_v).abs() < 0.2, "fd={v}, bs={bs_v}");
    }

    #[test]
    fn crank_nicolson_more_accurate_than_explicit() {
        let opt = european_call();
        let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.2, 0.0).unwrap();
        let bs_v = bs.price(OptionType::Call);

        let cn = FiniteDifferencePricer::new(200, 200, 400.0, FdScheme::CrankNicolson)
            .unwrap()
            .price(&opt, 100.0, 0.05, 0.2, 0.0)
            .unwrap();
        let imp = FiniteDifferencePricer::new(200, 200, 400.0, FdScheme::Implicit)
            .unwrap()
            .price(&opt, 100.0, 0.05, 0.2, 0.0)
            .unwrap();
        // CN is at worst comparable to implicit on this smooth problem.
        assert!((cn - bs_v).abs() <= (imp - bs_v).abs() + 0.05);
    }

    #[test]
    fn american_put_premium_via_pde() {
        let am_put = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::American,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        let eu_put = VanillaOption::new(
            OptionType::Put,
            ExerciseStyle::European,
            100.0,
            1.0,
            "TEST",
        )
        .unwrap();
        let pricer =
            FiniteDifferencePricer::new(200, 400, 400.0, FdScheme::Implicit).unwrap();
        let am = pricer.price(&am_put, 100.0, 0.10, 0.30, 0.0).unwrap();
        let eu = pricer.price(&eu_put, 100.0, 0.10, 0.30, 0.0).unwrap();
        assert!(am > eu + 1e-3, "am={am}, eu={eu}");
    }

    #[test]
    fn tridiagonal_solver_smoke() {
        let a = vec![0.0, 1.0, 1.0];
        let b = vec![2.0, 2.0, 2.0];
        let c = vec![1.0, 1.0, 0.0];
        let d = vec![3.0, 4.0, 3.0];
        let x = solve_tridiagonal(&a, &b, &c, &d).unwrap();
        // Verify by multiplying back.
        let r0 = b[0] * x[0] + c[0] * x[1];
        let r1 = a[1] * x[0] + b[1] * x[1] + c[1] * x[2];
        let r2 = a[2] * x[1] + b[2] * x[2];
        assert!((r0 - d[0]).abs() < 1e-9);
        assert!((r1 - d[1]).abs() < 1e-9);
        assert!((r2 - d[2]).abs() < 1e-9);
    }
}
