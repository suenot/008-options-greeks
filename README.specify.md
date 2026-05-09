# Chapter 8: Options and Greeks in Algorithmic Trading

## Metadata
- **Difficulty**: Advanced
- **Prerequisites**: Chapters 1–7 (stochastic calculus, microstructure, portfolio
  optimisation, ML, low-latency, information theory, game theory)
- **Implementation languages**: Rust (primary), Python (visualisation), Julia (research)
- **Estimated length**: 100–130 pages

---

## Chapter goals

1. Understand the mathematics of options: from Black–Scholes to stochastic volatility.
2. Master the Greeks (Delta, Gamma, Theta, Vega, Rho) and their interrelations.
3. Learn risk-management strategies for option portfolios.
4. Implement a pricing engine for European and American options.
5. Build a real-time Greeks calculator and risk reporting system.

---

## Scientific foundation

### Foundational works
1. **Black F., Scholes M.** (1973) "The Pricing of Options and Corporate Liabilities" — *Journal of Political Economy*
2. **Merton R.C.** (1973) "Theory of Rational Option Pricing" — *Bell Journal of Economics*
3. **Cox J.C., Ross S.A., Rubinstein M.** (1979) "Option Pricing: A Simplified Approach" — *Journal of Financial Economics*
4. **Heston S.L.** (1993) "A Closed-Form Solution for Options with Stochastic Volatility" — *Review of Financial Studies*

### Recent research (2023–2025)
5. **Gatheral J.** (2024) *The Volatility Surface: A Practitioner's Guide* — updated edition
6. **Bergomi L.** (2025) *Stochastic Volatility Modeling* — 2nd edition, new numerical methods
7. **De Marco S., Jacquier A.** (2024) "Deep Learning for Option Pricing and Hedging" — arXiv
8. **Horvath B., Jacquier A., Tankov P.** (2023) "Learning Rough Volatility: Neural Networks for Pricing and Calibration" — *Quantitative Finance*

---

## Chapter structure

### 8.1 Option fundamentals
- 8.1.1 Types of options (Call/Put, European/American/Bermudan, Vanilla/Exotic)
- 8.1.2 Payoff diagrams
- 8.1.3 Put–call parity, arbitrage bounds, early exercise

### 8.2 Black–Scholes–Merton model
- 8.2.1 Derivation
- 8.2.2 Assumptions
- 8.2.3 Limitations and extensions (smile/skew, local volatility)

### 8.3 Greeks
- Delta, Gamma, Theta, Vega, Rho
- Cross-Greeks: Vanna, Volga
- Delta hedging and Gamma scalping

### 8.4 Numerical methods for American options
- Cox–Ross–Rubinstein binomial tree
- Finite-difference methods (Explicit, Implicit, Crank–Nicolson)
- Monte Carlo and Longstaff–Schwartz

### 8.5 Stochastic volatility
- Heston model
- Characteristic function and FFT/Lewis integral pricing
- Dupire local volatility

### 8.6 Strategies and risk management
- Delta-hedged portfolios, hedging error, transaction costs
- Volatility trading, structured spreads
- Scenario analysis, P&L attribution, VaR

### 8.7 Practical exercises
- Building a volatility surface from real market data
- Calibrating the Heston model
- Delta-hedging simulation with transaction costs
- Comparing tree, PDE and Monte Carlo for American puts

---

## Output format

```
.
├── chapter.en.md              # Full chapter text (English)
├── chapter.ru.md              # Full chapter text (Russian)
├── readme.simple.en.md        # Simple introduction (English)
├── readme.simple.ru.md        # Simple introduction (Russian)
├── README.specify.md          # This specification
└── code/
    ├── Cargo.toml
    ├── benches/
    │   └── pricing_benchmark.rs
    ├── examples/
    │   └── volatility_surface.rs
    └── src/
        ├── lib.rs
        ├── main.rs
        ├── option_types.rs
        ├── black_scholes.rs
        ├── greeks.rs
        ├── binomial_tree.rs
        ├── finite_difference.rs
        ├── monte_carlo.rs
        ├── heston_model.rs
        └── portfolio_risk.rs
```

---

## Acceptance criteria

1. **Text**: 100–130 page equivalent, full coverage of all sections above.
2. **Bilingual**: full versions in Russian and English; simple versions in both.
3. **Code**: production-ready Rust with `Cargo.toml`, benchmarks (Criterion), documentation.
4. **Mathematics**: all formulas in LaTeX, rigorous definitions and proofs.
5. **Analogies**: simple versions contain real-life analogies (as in Chapter 1).
6. **Tests**: unit tests for all mathematical functions, comparison with known benchmark values.
7. **Examples**: working examples with real or synthetic data.
8. **Cross-references**: links to Chapters 1–7 where relevant.
