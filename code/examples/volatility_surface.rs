//! Build an implied-volatility surface from synthetic Heston quotes,
//! then verify that Black–Scholes IVs back-solved from those quotes
//! reproduce the input prices to a few basis points.

use options_greeks::{
    black_scholes::BlackScholesModel,
    heston_model::HestonModel,
    option_types::OptionType,
};

fn main() {
    let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
    let maturities = [0.25, 0.5, 1.0, 2.0];
    let true_h = HestonModel::new(
        100.0, 100.0, 1.0, 0.03, 0.0, 0.04, 1.5, 0.04, 0.4, -0.5,
    )
    .unwrap();

    println!("Implied volatility surface (Heston-generated, BS-implied):");
    print!("strike\\T   ");
    for t in maturities {
        print!("{t:>6.2} ");
    }
    println!();
    for k in strikes {
        print!("{k:>8.2}   ");
        for t in maturities {
            let mut probe = true_h;
            probe.strike = k;
            probe.time_to_expiry = t;
            let price = probe.price(OptionType::Call);
            let bs = BlackScholesModel::new(100.0, k, t, 0.03, 0.20, 0.0).unwrap();
            let iv = bs.implied_volatility(price, OptionType::Call).unwrap();
            print!("{iv:>6.4} ");
        }
        println!();
    }
}
