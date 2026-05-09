use criterion::{black_box, criterion_group, criterion_main, Criterion};

use options_greeks::{
    binomial_tree::BinomialTree,
    black_scholes::BlackScholesModel,
    finite_difference::{FdScheme, FiniteDifferencePricer},
    greeks::GreeksCalculator,
    heston_model::HestonModel,
    monte_carlo::MonteCarloPricer,
    option_types::{ExerciseStyle, OptionType, VanillaOption},
};

fn bench_black_scholes(c: &mut Criterion) {
    let bs = BlackScholesModel::new(100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
    c.bench_function("black_scholes_price", |b| {
        b.iter(|| black_box(bs).price(black_box(OptionType::Call)))
    });
    c.bench_function("greeks_calculate", |b| {
        b.iter(|| GreeksCalculator::calculate(black_box(&bs), black_box(OptionType::Call)))
    });
}

fn bench_binomial_tree(c: &mut Criterion) {
    let tree = BinomialTree::new(200, 100.0, 100.0, 1.0, 0.05, 0.20, 0.0).unwrap();
    c.bench_function("binomial_tree_european_200", |b| {
        b.iter(|| tree.european_price(black_box(OptionType::Call)))
    });
    c.bench_function("binomial_tree_american_200", |b| {
        b.iter(|| tree.american_price(black_box(OptionType::Put)))
    });
}

fn bench_finite_difference(c: &mut Criterion) {
    let opt = VanillaOption::new(
        OptionType::Put,
        ExerciseStyle::American,
        100.0,
        1.0,
        "X",
    )
    .unwrap();
    let pricer = FiniteDifferencePricer::new(100, 100, 400.0, FdScheme::CrankNicolson).unwrap();
    c.bench_function("fd_crank_nicolson_100x100", |b| {
        b.iter(|| pricer.price(black_box(&opt), 100.0, 0.10, 0.30, 0.0).unwrap())
    });
}

fn bench_monte_carlo(c: &mut Criterion) {
    let opt = VanillaOption::new(
        OptionType::Call,
        ExerciseStyle::European,
        100.0,
        1.0,
        "X",
    )
    .unwrap();
    let mc = MonteCarloPricer::new(10_000, 42).unwrap();
    c.bench_function("monte_carlo_european_10k", |b| {
        b.iter(|| mc.european_price(black_box(&opt), 100.0, 0.05, 0.20, 0.0).unwrap())
    });
}

fn bench_heston(c: &mut Criterion) {
    let h = HestonModel::new(
        100.0, 100.0, 1.0, 0.05, 0.0, 0.04, 2.0, 0.04, 0.5, -0.7,
    )
    .unwrap();
    c.bench_function("heston_lewis_price", |b| {
        b.iter(|| h.price(black_box(OptionType::Call)))
    });
}

criterion_group!(
    benches,
    bench_black_scholes,
    bench_binomial_tree,
    bench_finite_difference,
    bench_monte_carlo,
    bench_heston,
);
criterion_main!(benches);
