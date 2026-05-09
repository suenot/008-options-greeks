//! Demonstration CLI for the Chapter 8 implementations.

use options_greeks::{
    binomial_tree::BinomialTree,
    black_scholes::BlackScholesModel,
    finite_difference::{FdScheme, FiniteDifferencePricer},
    greeks::GreeksCalculator,
    heston_model::HestonModel,
    monte_carlo::{LongstaffSchwartz, MonteCarloPricer},
    option_types::{ExerciseStyle, OptionType, VanillaOption},
    portfolio_risk::{OptionPortfolio, Position, ScenarioShocks},
};

fn main() {
    println!("=== Chapter 8: Options and Greeks ===\n");
    black_scholes_demo();
    greeks_demo();
    binomial_tree_demo();
    finite_difference_demo();
    monte_carlo_demo();
    heston_demo();
    portfolio_demo();
}

fn black_scholes_demo() {
    println!("--- Black–Scholes pricing ---");
    let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
    println!("Call: {:.4}", bs.price(OptionType::Call));
    println!("Put:  {:.4}", bs.price(OptionType::Put));
    let market = 11.0;
    let iv = bs.implied_volatility(market, OptionType::Call).unwrap();
    println!("Implied vol for market call price {market}: {iv:.4}\n");
}

fn greeks_demo() {
    println!("--- Greeks ---");
    let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
    let g = GreeksCalculator::calculate(&bs, OptionType::Call);
    println!("Δ={:.4}, Γ={:.4}, Θ={:.4}, ν={:.4}, ρ={:.4}", g.delta, g.gamma, g.theta, g.vega, g.rho);
    println!("Vanna={:.4}, Volga={:.4}\n", g.vanna, g.volga);
}

fn binomial_tree_demo() {
    println!("--- Binomial tree (CRR) ---");
    let tree = BinomialTree::new(500, 100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
    println!("European call: {:.4}", tree.european_price(OptionType::Call));
    println!("American put:  {:.4}\n", tree.american_price(OptionType::Put));
}

fn finite_difference_demo() {
    println!("--- Finite-difference PDE ---");
    let opt = VanillaOption::new(
        OptionType::Put,
        ExerciseStyle::American,
        100.0,
        1.0,
        "X",
    )
    .unwrap();
    let pricer = FiniteDifferencePricer::new(200, 200, 400.0, FdScheme::CrankNicolson).unwrap();
    let price = pricer.price(&opt, 100.0, 0.10, 0.30, 0.0).unwrap();
    println!("American put (Crank–Nicolson): {price:.4}\n");
}

fn monte_carlo_demo() {
    println!("--- Monte Carlo ---");
    let opt_call = VanillaOption::new(
        OptionType::Call,
        ExerciseStyle::European,
        100.0,
        1.0,
        "X",
    )
    .unwrap();
    let mc = MonteCarloPricer::new(100_000, 42).unwrap();
    let v = mc.european_price(&opt_call, 100.0, 0.05, 0.20, 0.0).unwrap();
    println!("European call MC: {v:.4}");

    let opt_put = VanillaOption::new(
        OptionType::Put,
        ExerciseStyle::American,
        100.0,
        1.0,
        "X",
    )
    .unwrap();
    let lsm = LongstaffSchwartz::new(20_000, 50, 3, 7).unwrap();
    let v = lsm.american_price(&opt_put, 100.0, 0.10, 0.30, 0.0).unwrap();
    println!("American put LSM: {v:.4}\n");
}

fn heston_demo() {
    println!("--- Heston FFT pricing ---");
    let h = HestonModel::new(
        100.0, 100.0, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7,
    )
    .unwrap();
    println!("Heston call (ATM): {:.4}", h.price(OptionType::Call));
    println!("Heston implied vol (ATM): {:.4}\n", h.implied_volatility(OptionType::Call).unwrap());
}

fn portfolio_demo() {
    println!("--- Portfolio risk ---");
    let bs = BlackScholesModel::new(100.0, 100.0, 0.5, 0.05, 0.20, 0.0).unwrap();
    let mut book = OptionPortfolio::new();
    book.push(
        Position::new(
            VanillaOption::new(
                OptionType::Call,
                ExerciseStyle::European,
                100.0,
                0.5,
                "X",
            )
            .unwrap(),
            10.0,
            bs,
        )
        .unwrap(),
    );
    let bs_put = BlackScholesModel { strike: 95.0, ..bs };
    book.push(
        Position::new(
            VanillaOption::new(
                OptionType::Put,
                ExerciseStyle::European,
                95.0,
                0.5,
                "X",
            )
            .unwrap(),
            -5.0,
            bs_put,
        )
        .unwrap(),
    );
    let g = book.total_greeks();
    println!("Net Δ={:.3}, Γ={:.3}, ν={:.3}", g.delta, g.gamma, g.vega);
    let pl = book.scenario_pl(&ScenarioShocks {
        spot_shock: -0.05,
        vol_shock: 0.05,
        rate_shock: 0.0,
        time_shock: 1.0 / 252.0,
    });
    println!("Scenario P&L (-5% spot, +5 vol, 1d decay): {pl:.4}");
    let var = book.var(0.99, 0.30, 0.05, 1.0 / 252.0, 20_000, 17).unwrap();
    println!("99% 1-day VaR: {var:.4}\n");
}
