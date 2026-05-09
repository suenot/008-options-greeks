# Chapter 8 — Options and Greeks in Algorithmic Trading

> *"An option is a bet on volatility wearing the clothes of a bet on direction."* — folklore.

This chapter takes you from the definition of a vanilla option to the
day-to-day machinery of an options trading desk: a pricing engine, a
Greek calculator, an American-option solver and a stochastic-volatility
calibration loop, ending with portfolio-level risk reports. Every
formula is matched to the Rust code in `code/src/` so you can read the
math and the implementation side by side.

**Prerequisites.** Stochastic calculus and Itô's lemma (Chapter 1),
microstructure intuition for hedging cost (Chapter 2), portfolio theory
for risk aggregation (Chapter 3), gradient-based fitting from Chapter 4,
the latency/throughput trade-offs of Chapter 5, the information
arguments of Chapter 6, and the strategic interaction lens of Chapter 7.
This chapter inherits notation from those.

---

## 8.1 Option fundamentals

### 8.1.1 What an option is

A **vanilla option** is a contract that gives its holder the *right* —
but not the *obligation* — to transact a unit of an underlying asset
$S$ at a fixed *strike* $K$ at, or before, an *expiry* date $T$.

* A **call** option lets the holder *buy* at $K$, with payoff
  $(S_T - K)^+$ at expiry.
* A **put** option lets the holder *sell* at $K$, with payoff
  $(K - S_T)^+$ at expiry.

The non-negativity of the payoff is the definition of optionality: you
only exercise when it is profitable. We use $x^+ \equiv \max(x, 0)$
throughout.

In Rust this is the entirety of the contract specification:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionType { Call, Put }

#[derive(Debug, Clone, PartialEq)]
pub enum ExerciseStyle {
    European,            // exercise only at T
    American,            // any time in [0, T]
    Bermudan(Vec<f64>),  // exercise on a fixed set of dates
}
```

(see `code/src/option_types.rs`).

### 8.1.2 Payoff, intrinsic and time value

For a vanilla option of type $\omega \in \{\text{call}, \text{put}\}$
the **payoff** at expiry is

$$
\Pi_\omega(S_T) = \begin{cases} (S_T - K)^+ & \omega = \text{call}, \\ (K - S_T)^+ & \omega = \text{put}. \end{cases}
$$

For an option *before* expiry one decomposes the price $V$ into

$$
V \;=\; \underbrace{\Pi_\omega(S)}_{\text{intrinsic}} \;+\; \underbrace{V - \Pi_\omega(S)}_{\text{time value}}.
$$

Intrinsic value is what you would receive if you exercised *right now*;
time value is everything else and reflects the *option not yet
having to commit*. For a European option the time value can be
negative (because you cannot exercise to capture it); for an American
option it must be non-negative — otherwise immediate exercise
dominates.

### 8.1.3 Put–call parity

For European options on a non-dividend-paying stock,

$$
\boxed{\; C - P \;=\; S - K\,e^{-rT} \;}
$$

with $C, P$ the call and put prices and $r$ the continuously-compounded
risk-free rate. The reason is a static no-arbitrage argument: the
portfolio "long call, short put" pays $S_T - K$ at expiry, regardless
of whether $S_T$ is above or below $K$; this is the same payoff as
"long stock, short $K$ zero-coupon bonds maturing at $T$". With
continuous dividend yield $q$ the parity becomes

$$
C - P \;=\; S\,e^{-qT} - K\,e^{-rT}.
$$

This identity is one of the few things in derivatives pricing that is
*exactly* true: it does not depend on a model. It is the first sanity
check for any pricing engine — the call/put pair priced at the same
parameters must satisfy parity to machine precision. Our test
`black_scholes::tests::parity_holds` enforces this.

### 8.1.4 No-arbitrage bounds

Without any model:

$$
\max\!\big(S\,e^{-qT} - K\,e^{-rT},\, 0\big) \;\le\; C \;\le\; S\,e^{-qT}.
$$

The lower bound is the present value of the forward minus the present
value of the strike (clipped at zero); the upper bound says you cannot
pay more for the right to buy than for the asset itself. Symmetric
bounds hold for puts. Any quote outside these bounds is a free lunch
and trips our `OutOfArbitrageBounds` error during implied-volatility
solving.

### 8.1.5 Why early exercise matters

For a non-dividend-paying stock, a *European call* and an *American
call* have the same value: it is never optimal to exercise the call
early because doing so throws away time value and gives up interest on
the strike. For *puts*, however, it can be optimal to exercise early —
holding a deeply ITM put while the stock heads to zero costs you the
interest you could have earned on the strike. This asymmetry is why
American puts have no closed form and we will need numerical methods
(Section 8.4).

---

## 8.2 The Black–Scholes–Merton model

### 8.2.1 Assumptions

The model assumes a frictionless market with:

1. The underlying follows a geometric Brownian motion under the
   physical measure $\mathbb{P}$:
   $$\,\mathrm{d} S_t = \mu S_t\,\mathrm{d} t + \sigma S_t\,\mathrm{d} W_t.$$
2. Constant continuously-compounded interest rate $r$ and continuous
   dividend yield $q$.
3. Constant volatility $\sigma$.
4. No transaction costs; you can rebalance continuously.
5. No arbitrage; the asset can be shorted freely.

Each of these is wrong in the real market, but as a first-order frame
they buy us a closed-form price and an exact decomposition of risk —
the Greeks.

### 8.2.2 The PDE and its solution

By Itô's lemma, the value of a derivative $V(t, S)$ on $S$ satisfies

$$
\,\mathrm{d} V = \left(\partial_t V + \mu S\,\partial_S V + \tfrac12 \sigma^2 S^2\,\partial_{SS}^2 V\right)\,\mathrm{d} t + \sigma S\,\partial_S V\,\mathrm{d} W_t.
$$

Form the *delta-hedged* portfolio $\Pi = V - \Delta \cdot S$ with
$\Delta = \partial_S V$. The $\,\mathrm{d} W_t$ term cancels and the
portfolio becomes locally riskless. By no-arbitrage it must earn the
risk-free rate, which gives the Black–Scholes PDE,

$$
\boxed{\;\partial_t V + (r - q)\,S\,\partial_S V + \tfrac12 \sigma^2 S^2\,\partial_{SS}^2 V - r V \;=\; 0\;}
$$

with terminal condition $V(T, S) = \Pi_\omega(S)$. For European calls
and puts this PDE has a closed-form solution. Define

$$
d_1 = \frac{\ln(S/K) + (r - q + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}.
$$

Then

$$
\boxed{\begin{aligned} C &= S\,e^{-qT}\,N(d_1) - K\,e^{-rT}\,N(d_2), \\ P &= K\,e^{-rT}\,N(-d_2) - S\,e^{-qT}\,N(-d_1), \end{aligned}}
$$

with $N(\cdot)$ the standard normal CDF.

### 8.2.3 Implementation

```rust
pub fn price(&self, option_type: OptionType) -> f64 {
    if self.time_to_expiry == 0.0 || self.volatility == 0.0 {
        let intrinsic = match option_type {
            OptionType::Call => (self.spot - self.strike).max(0.0),
            OptionType::Put  => (self.strike - self.spot).max(0.0),
        };
        return intrinsic * (-self.risk_free_rate * self.time_to_expiry).exp();
    }
    let d1 = self.d1();
    let d2 = self.d2();
    let disc_r = (-self.risk_free_rate * self.time_to_expiry).exp();
    let disc_q = (-self.dividend_yield * self.time_to_expiry).exp();
    match option_type {
        OptionType::Call =>  self.spot * disc_q * norm_cdf(d1)
                           - self.strike * disc_r * norm_cdf(d2),
        OptionType::Put  =>  self.strike * disc_r * norm_cdf(-d2)
                           - self.spot * disc_q * norm_cdf(-d1),
    }
}
```

(see `code/src/black_scholes.rs`). Boundary cases at $T=0$ or
$\sigma=0$ are handled by collapsing to the discounted intrinsic
value; this avoids `0/0` in $d_1$.

### 8.2.4 The standard normal CDF

We do not pull in `statrs` for one function — we use the
Abramowitz–Stegun rational approximation for `erf` (formula 7.1.26)
inside `code/src/lib.rs`:

```rust
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}
```

Maximum absolute error of the underlying `erf` approximation is
$1.5 \times 10^{-7}$, sufficient for option pricing where the implied
volatility on a million-dollar option moves by at least a basis point.

### 8.2.5 Implied volatility

In practice we *observe* the option price and want the volatility that
produces it under Black–Scholes — the **implied volatility** $\sigma^*$.
The price is monotone in $\sigma$ (Vega $> 0$), so $\sigma^*$ is unique
when it exists. We use Newton–Raphson with the closed-form Vega, with
a bisection fallback for hard cases:

```rust
let mut sigma = sigma0;
for _ in 0..max_iters {
    let m = BlackScholesModel { volatility: sigma, ..*self };
    let price = m.price(option_type);
    let vega  = vega_value(&m);
    let diff  = price - target_price;
    if diff.abs() < tolerance { return Ok(sigma); }
    if vega < 1e-12 { break; }
    sigma -= diff / vega;
    if !sigma.is_finite() || sigma <= 0.0 || sigma > 5.0 {
        break; // bisection fallback
    }
}
```

The arbitrage-bounds check is performed *before* iterating: a quote
that violates $\max(F-K,0) e^{-rT} \le C \le S e^{-qT}$ admits no real
$\sigma$ and would send Newton off to infinity.

---

## 8.3 The Greeks

The price of a derivative is a function of several parameters. The
**Greeks** are the partial derivatives of the price with respect to
each. They are simultaneously the *sensitivities* used for risk
reporting and the *hedge ratios* used to neutralise that risk.

### 8.3.1 Definitions

For a vanilla option with Black–Scholes price $V$:

| Greek | Symbol | Definition |
|------|------|------|
| Delta | $\Delta$ | $\partial V / \partial S$ |
| Gamma | $\Gamma$ | $\partial^2 V / \partial S^2$ |
| Theta | $\Theta$ | $\partial V / \partial t$ |
| Vega  | $\nu$    | $\partial V / \partial \sigma$ |
| Rho   | $\rho$   | $\partial V / \partial r$ |
| Vanna | $\partial^2 V / (\partial S \partial \sigma)$ |
| Volga | $\partial^2 V / \partial \sigma^2$ |

### 8.3.2 Closed-form Black–Scholes Greeks

$$
\Delta_C = e^{-qT} N(d_1), \qquad \Delta_P = -e^{-qT} N(-d_1) = \Delta_C - e^{-qT}.
$$

$$
\Gamma = \frac{e^{-qT} \phi(d_1)}{S \sigma \sqrt{T}},
$$

with $\phi$ the standard normal PDF. Gamma is the same for calls and
puts (a pleasing consequence of put–call parity).

$$
\nu = S\,e^{-qT}\,\phi(d_1)\,\sqrt{T}.
$$

Vega is also the same for calls and puts — and is largest at-the-money,
where the option is most uncertain about which way it will end up.

$$
\Theta_C = -\frac{S\,e^{-qT}\,\phi(d_1)\,\sigma}{2\sqrt{T}} - rK\,e^{-rT} N(d_2) + qS\,e^{-qT} N(d_1).
$$

$$
\rho_C = K\,T\,e^{-rT}\,N(d_2), \qquad \rho_P = -K\,T\,e^{-rT}\,N(-d_2).
$$

The cross-Greeks:

$$
\text{Vanna} = -e^{-qT} \phi(d_1) \frac{d_2}{\sigma}, \qquad \text{Volga} = \nu \cdot \frac{d_1\,d_2}{\sigma}.
$$

Vanna captures how the hedge moves when volatility moves; Volga (also
called "vomma") measures convexity in volatility — the curvature of
the smile we will meet in Section 8.5.

### 8.3.3 Implementation and equivalence with finite differences

The Greeks live in `code/src/greeks.rs`:

```rust
pub fn calculate(model: &BlackScholesModel, option_type: OptionType) -> Greeks {
    let d1 = model.d1();
    let d2 = model.d2();
    let phi = norm_pdf(d1);
    let s_eq  = model.spot * (-model.dividend_yield * model.time_to_expiry).exp();
    let delta = match option_type {
        OptionType::Call =>  s_eq / model.spot * norm_cdf(d1),
        OptionType::Put  => -s_eq / model.spot * norm_cdf(-d1),
    };
    let gamma = s_eq * phi / (model.spot * model.spot * model.volatility * t.sqrt());
    let vega  = s_eq * phi * t.sqrt();
    /* …theta, rho, vanna, volga… */
}
```

A second implementation `GreeksCalculator::finite_difference` re-prices
the option at $S \pm h$, $\sigma \pm h$, etc., and computes the
sensitivities by symmetric differences. The unit test
`closed_form_matches_finite_difference` enforces agreement to better
than $10^{-3}$ for ATM options — a sharp check that the closed-form
formulas are coded correctly.

### 8.3.4 Delta hedging

The Black–Scholes derivation is constructive: if you continuously
rebalance a stock position equal to $-\Delta$ shares, you exactly
replicate the option. In practice you rebalance at discrete
intervals, and the leftover P&L over $[t, t+\,\mathrm{d} t]$ is

$$
\,\mathrm{d} \Pi \;\approx\; \tfrac12 \Gamma\,(\,\mathrm{d} S)^2 - \tfrac12 \Gamma\,\sigma_{\mathrm{impl}}^2 S^2\,\,\mathrm{d} t.
$$

The first term is what you *capture* (realised variance); the second
is what you *pay* (implied variance). Profit per rebalance is

$$
\Delta \Pi \;\approx\; \tfrac12 \Gamma\,S^2\,(\sigma_{\mathrm{real}}^2 - \sigma_{\mathrm{impl}}^2)\,\,\mathrm{d} t.
$$

This is the central identity of vol trading: **Gamma scalping converts
realised vs implied variance into P&L.** A long-gamma book makes money
when realised vol exceeds implied; a short-gamma book makes money when
implied vol exceeds realised. Either way, the Greeks tell you *how
much* and the rebalance schedule from Chapter 5 tells you *how often*
without burning the edge in transaction cost (microstructure from
Chapter 2).

### 8.3.5 Cross-Greeks in practice

Vanna is dominant when you are short an OTM put and the market sells
off — spot drops *and* vol jumps, both moves work against you. Volga
is what makes deep-OTM options expensive in a smile world: their vega
itself rises when the smile steepens. A modern desk targets net
Vanna and Volga, not just Delta and Vega, because the smile means
those second-order risks have a market price.

---

## 8.4 Numerical methods for American options

For an American option the holder can exercise at any $\tau \in [0, T]$.
The price is the supremum over stopping times of the discounted
expected payoff under the risk-neutral measure $\mathbb{Q}$:

$$
V_0 = \sup_{\tau \in \mathcal{T}_{[0,T]}} \mathbb{E}^{\mathbb{Q}}\!\left[ e^{-r\tau}\,\Pi_\omega(S_\tau)\right].
$$

There is no closed form. The three standard solvers all encode the
same backward induction: at each node you compare the *intrinsic
value* with the *continuation value* and take the larger.

### 8.4.1 Cox–Ross–Rubinstein binomial tree

Discretise time into $N$ steps of width $\Delta t = T/N$. From any
node $S$ the spot moves up to $Su$ or down to $Sd$ with $u = e^{\sigma\sqrt{\Delta t}}$,
$d = 1/u$, and risk-neutral probability of an up-move

$$
p = \frac{e^{(r-q)\Delta t} - d}{u - d}.
$$

Terminal payoffs are known. Roll back through the tree by

$$
V_{i,j} = \max\!\Big(\Pi_\omega(S_{i,j}),\;\; e^{-r\Delta t}\big[p V_{i+1,j+1} + (1-p) V_{i+1,j}\big]\Big),
$$

with the `max` operator dropped for European exercise.

```rust
let dt = self.option.time_to_expiry / self.n_steps as f64;
let u  = (self.model.volatility * dt.sqrt()).exp();
let d  = 1.0 / u;
let p  = ((self.model.risk_free_rate - self.model.dividend_yield) * dt).exp();
let p  = (p - d) / (u - d);
let disc = (-self.model.risk_free_rate * dt).exp();

let mut values: Vec<f64> = (0..=self.n_steps).map(|j| {
    let s = self.model.spot * u.powi(j as i32) * d.powi((self.n_steps - j) as i32);
    self.option.payoff(s)
}).collect();

for step in (0..self.n_steps).rev() {
    for j in 0..=step {
        let cont = disc * (p * values[j+1] + (1.0 - p) * values[j]);
        let s = self.model.spot * u.powi(j as i32) * d.powi((step - j) as i32);
        values[j] = if american { cont.max(self.option.payoff(s)) } else { cont };
    }
}
values[0]
```

(see `code/src/binomial_tree.rs`). For European options the tree
converges to the closed-form Black–Scholes price as $N \to \infty$ at
order $O(1/N)$. With Richardson extrapolation between two grid sizes
the convergence becomes $O(1/N^2)$.

The tree also recovers the Greeks, by reading values off the first
few nodes:

$$
\Delta \approx \frac{V_{1,1} - V_{1,0}}{S_{1,1} - S_{1,0}},
$$

and Gamma from the next layer down. These are *almost free* once the
backward sweep is done.

### 8.4.2 Finite-difference solvers

The Black–Scholes PDE is

$$
\partial_t V + (r - q) S\,\partial_S V + \tfrac12\sigma^2 S^2\,\partial_{SS}^2 V - r V = 0.
$$

Discretise $S$ on a uniform grid $\{S_j = j\,h\}_{j=0}^{M}$ and time
on $\{t_n = n\,k\}_{n=0}^{N}$. Approximate

$$
\partial_S V \approx \frac{V_{j+1} - V_{j-1}}{2h}, \qquad \partial_{SS}^2 V \approx \frac{V_{j+1} - 2V_j + V_{j-1}}{h^2}.
$$

We solve backwards from $V(T, S) = \Pi_\omega(S)$.

* **Explicit scheme.** Stable only when $\sigma^2 S_{\max}^2 k / h^2 \le 1$,
  cheap per step.
* **Implicit (fully implicit) scheme.** Unconditionally stable but
  only first-order accurate in time.
* **Crank–Nicolson scheme.** $\theta = 1/2$ blend; second-order
  accurate in both time and space, the workhorse.

Each step is a tridiagonal solve, dispatched to the **Thomas
algorithm**:

```rust
fn solve_tridiagonal(a: &[f64], b: &mut [f64], c: &[f64], d: &mut [f64]) {
    let n = b.len();
    for i in 1..n {
        let m = a[i] / b[i-1];
        b[i] -= m * c[i-1];
        d[i] -= m * d[i-1];
    }
    d[n-1] /= b[n-1];
    for i in (0..n-1).rev() {
        d[i] = (d[i] - c[i] * d[i+1]) / b[i];
    }
}
```

(see `code/src/finite_difference.rs`). For American options the
backward step is followed by a max-with-intrinsic projection, which
matches the early-exercise rule. Crank–Nicolson + projection is the
classic finite-difference American pricer; our test shows it agrees
with the binomial tree to four decimals for an ATM put.

### 8.4.3 Monte Carlo and Longstaff–Schwartz

Monte Carlo simulates many spot paths and averages the discounted
payoff. For European options this is straightforward and embarrassingly
parallel. We use **antithetic variates** to halve variance:

```rust
for _ in 0..self.num_paths/2 {
    let z: f64 = StandardNormal.sample(&mut rng);
    let s_p = self.model.spot * (drift + diff * z).exp();
    let s_m = self.model.spot * (drift - diff * z).exp();
    sum += self.option.payoff(s_p) + self.option.payoff(s_m);
}
let mean = sum / self.num_paths as f64;
let price = (-self.model.risk_free_rate * self.option.time_to_expiry).exp() * mean;
```

The standard error is $\sigma_{\text{payoff}}/\sqrt{N}$; antithetic
sampling exploits the symmetry of the lognormal distribution to
reduce the constant.

For American options Monte Carlo cannot be used naively because there
is no obvious "price tomorrow" for the holder to compare to today's
intrinsic. Longstaff and Schwartz (2001) solved this with a
*regression-based continuation value*:

1. Simulate $N$ paths $\{S_t^{(i)}\}_{t=0}^{T}$ of the spot.
2. Initialise $V^{(i)} = \Pi_\omega(S_T^{(i)})$ at expiry.
3. For $t = T - \,\mathrm{d} t$ down to $\,\mathrm{d} t$:
   * On *in-the-money* paths, regress
     $\,e^{-r\,\mathrm{d} t} V^{(i)}$ on a basis $\{1, S^{(i)}, (S^{(i)})^2\}$
     of $S_t^{(i)}$ to estimate the continuation function $\hat{C}(S)$.
   * On each path, exercise if $\Pi_\omega(S_t^{(i)}) > \hat{C}(S_t^{(i)})$;
     otherwise continue and discount.
4. Average $V^{(i)}$ at $t = 0$.

The regression is closed-form: the normal equations are a small
$3 \times 3$ system, solved in `monte_carlo.rs::normal_equations`. We
include only ITM paths in the regression — this is the original LSM
choice and avoids the regression being dominated by the always-zero
OTM payoffs.

### 8.4.4 Comparing the three solvers

For an at-the-money one-year American put with $S=K=100$, $r=5\%$,
$q=0$, $\sigma=20\%$:

| Method       | Steps / Paths | Price | Error vs CRR(5000) |
|---|---|---|---|
| CRR tree     | 5 000         | 6.0902 | 0       |
| CN finite    | 200 × 200     | 6.0903 | 1e-4   |
| LSM MC       | 50 000 × 100  | 6.092  | 2e-3   |

The tree is the fastest and most accurate for vanilla American
options. Finite-difference wins for grid-natural payoffs like
barriers. Monte Carlo wins as soon as you have multiple underlyings
(the only solver whose cost does not blow up combinatorially with
dimension).

---

## 8.5 Stochastic volatility

### 8.5.1 The implied-volatility surface

Plot Black–Scholes implied volatility against strike for several
expiries and you do *not* see a flat plane. You see a **smile**
(symmetric U for FX) or **smirk** (downward-sloping for equities).
Equivalently, the market disagrees with Black–Scholes' constant-vol
assumption: deep OTM puts are more expensive than the model says, and
deep OTM calls less so. Two empirical facts drive this:

1. The **leverage effect.** When stocks fall, debt-to-equity ratios
   rise and the equity becomes more volatile. Negative correlation
   $\rho < 0$ between price and volatility.
2. **Volatility clustering.** Volatility is mean-reverting: spikes
   relax over weeks, not days.

The Heston (1993) model is the classic two-factor model that captures
both.

### 8.5.2 The Heston model

$$
\,\mathrm{d} S_t = (r - q) S_t\,\,\mathrm{d} t + \sqrt{v_t}\,S_t\,\,\mathrm{d} W^S_t,
$$
$$
\,\mathrm{d} v_t = \kappa(\theta - v_t)\,\,\mathrm{d} t + \xi\sqrt{v_t}\,\,\mathrm{d} W^v_t,
$$
with $\,\mathrm{d} W^S_t \,\mathrm{d} W^v_t = \rho\,\,\mathrm{d} t$. The
five parameters are:

* $\kappa$ — mean-reversion speed of variance,
* $\theta$ — long-run variance,
* $\xi$ — volatility of variance ("vol of vol"),
* $\rho$ — correlation between spot and variance,
* $v_0$ — current variance.

The **Feller condition** $2\kappa\theta \ge \xi^2$ ensures variance
stays strictly positive almost surely.

### 8.5.3 The characteristic function

Heston's celebrated result is that the characteristic function of
$x_T = \ln S_T$ has a closed form. Writing $u$ for the Fourier
variable, the "little Heston trap" parameterisation due to Albrecher
et al. (2007) — which avoids branch-cut errors that plague the
original — is

$$
\varphi(u; T) = \exp\!\big(C(u, T) + D(u, T)\,v_0\big),
$$

where, with $b = \kappa + \rho\xi\,iu$,

$$
d = \sqrt{(\rho\xi\,iu - b)^2 - \xi^2 (-iu - u^2)} = \sqrt{b^2 + \xi^2(u^2 + iu)}\,,
$$

$$
g = \frac{b - d}{b + d},
$$

$$
D = \frac{b - d}{\xi^2}\,\frac{1 - e^{-dT}}{1 - g\,e^{-dT}},
$$

$$
C = \frac{\kappa\theta}{\xi^2}\!\left((b-d)T - 2\ln\frac{1 - g\,e^{-dT}}{1 - g}\right).
$$

```rust
fn characteristic_fn(&self, u: f64, t: f64) -> Complex<f64> {
    let i  = Complex::<f64>::new(0.0, 1.0);
    let xi = self.xi;
    let b  = Complex::new(self.kappa, 0.0) + i * (self.rho * xi * u);
    let d  = (b * b + xi * xi * Complex::new(u * u, u)).sqrt();
    let g  = (b - d) / (b + d);
    let one = Complex::<f64>::new(1.0, 0.0);
    let edt = (-d * t).exp();
    let cap_d = (b - d) / (xi * xi)
              * ((one - edt) / (one - g * edt));
    let cap_c = (self.kappa * self.theta) / (xi * xi)
              * ((b - d) * t - 2.0 * ((one - g * edt) / (one - g)).ln());
    (cap_c + cap_d * self.v0).exp()
}
```

### 8.5.4 Lewis–Lipton pricing formula

Using the centred characteristic function we use the Lewis (2001)
representation,

$$
C(S, K, T) = \frac{\sqrt{S K}\,e^{-(r+q)T/2}}{\pi} \int_0^\infty \mathrm{Re}\!\left[ \frac{e^{i u k}\,\varphi(u - i/2; T)}{u^2 + 1/4} \right] \mathrm{d} u,
$$

with $k = \ln(K/F)$ and $F = S e^{(r-q)T}$. The integrand decays
exponentially so we cap the upper limit at $u_{\max} = 200$ and use
Simpson's rule with $N = 4096$ points. In `code/src/heston_model.rs`:

```rust
let coef = (s*k).sqrt() * (-((r+q)*t)/2.0).exp() / std::f64::consts::PI;
let integrand = |u: f64| {
    let phi = self.characteristic_fn(u - 0.5, t);
    let num = (Complex::<f64>::new(0.0, u * log_moneyness)).exp() * phi;
    (num / Complex::new(u*u + 0.25, 0.0)).re
};
let integral = simpson(integrand, 0.0, u_max, n_steps);
coef * integral
```

For $\xi \to 0$ this reduces to the Black–Scholes price (no
randomness in vol). That is one of our regression tests:
`heston_collapses_to_black_scholes_when_xi_small`.

### 8.5.5 Generating a smile

With $\rho < 0$ the model produces a **downward skew**: low strikes
(OTM puts) have higher IV than high strikes (OTM calls). The example
program `code/examples/volatility_surface.rs` builds a synthetic
surface and recovers the implied vols that Heston implies for it:

```
T = 0.25y |  K=80  K=90  K=100  K=110  K=120
          |  0.265  0.224  0.198   0.187   0.183
T = 1.00y |  0.234  0.214  0.198   0.184   0.174
```

Quantitatively the slope $\partial \sigma_{\text{IV}}/\partial K$
flattens as $T$ increases — another well-documented stylised fact
that Heston reproduces.

### 8.5.6 Calibration

Calibration finds parameters $\{\kappa, \theta, \xi, \rho, v_0\}$
that minimise the squared error between Heston prices and a set of
quoted market prices. We use a simple coordinate-descent loop with
shrinking step sizes; production calibrators use Levenberg–Marquardt
plus a regularisation term that penalises distance from yesterday's
fit. The `calibration_reduces_loss` test checks that one round of our
descent decreases the loss by a finite amount.

For the linear-algebra heavy parts of LM, we lean on `nalgebra` (we
already pull it in for finite-difference grids). On a real desk the
calibrator runs every minute on the front-month and three back months,
producing fresh $\{\kappa, \theta, \xi, \rho, v_0\}$ before the next
quote cycle.

### 8.5.7 Local volatility (Dupire)

Dupire (1994) showed that *any* arbitrage-free smile is consistent
with a *local-volatility* model

$$
\,\mathrm{d} S_t = (r-q) S_t\,\,\mathrm{d} t + \sigma_{\text{loc}}(t, S_t)\,S_t\,\,\mathrm{d} W_t,
$$

with

$$
\sigma_{\text{loc}}^2(T, K) = \frac{\partial_T C + (r-q) K \partial_K C + qC}{\tfrac12 K^2 \partial_{KK}^2 C}.
$$

This is a deterministic function of $(t, S)$ that *exactly* reproduces
today's vanilla prices. It is the cheapest way to be smile-consistent
for path-dependent payoffs (autocallables, barrier options) but gives
the wrong forward smile dynamics. Hybrids — local-stochastic
volatility (LSV) — bridge the two and are what most equity desks
actually run.

---

## 8.6 Strategies and risk management

### 8.6.1 Delta-hedged portfolios

A delta-hedged option book is a *bet on volatility*. If you are long
$N$ calls and short $N\Delta$ shares, the linear move in the
underlying is hedged out and you are exposed to (i) Gamma — convex in
spot, makes you money in either direction; (ii) Theta — paid every
day for holding the option; (iii) Vega — moves with implied vol; (iv)
*hedging error* — which scales with the rebalance frequency, the
realised quadratic variation of the spot, and the bid-ask spread.

The cleanest expression of the daily P&L of a delta-hedged book is

$$
\,\mathrm{d} \Pi \approx \tfrac12 \Gamma S^2 (\sigma_{\text{real}}^2 - \sigma_{\text{impl}}^2)\,\,\mathrm{d} t + \nu\,\,\mathrm{d}\sigma_{\text{impl}} - \text{tcost}.
$$

A market maker is structurally **short Gamma** (they wrote the option
to the customer); they make their living off the bid-ask spread on the
hedge and pray that realised vol underperforms implied. A vol fund is
typically **long Gamma** (they bought the option from the dealer);
they make money when realised vol exceeds implied and pay theta in
the meantime.

### 8.6.2 Common structures

* **Straddle.** Long call + long put at the same $K$. Long
  volatility, no view on direction.
* **Risk reversal.** Long call + short put at different strikes (or
  vice versa). Long delta + long skew.
* **Butterfly.** Long the wings + short the body (e.g.
  $C(K_1) - 2C(K_2) + C(K_3)$). Pays off when spot stays near $K_2$;
  long convexity in volatility (volga).
* **Calendar spread.** Long longer-dated + short shorter-dated at the
  same strike. Long Vega, short Gamma; bets that the term structure
  steepens.

Each of these can be priced and risk-decomposed using the same
`OptionPortfolio` we built in `code/src/portfolio_risk.rs`.

### 8.6.3 Aggregating risk

```rust
pub fn total_greeks(&self) -> Greeks {
    let mut acc = Greeks::zero();
    for p in &self.positions {
        acc += p.greeks();
    }
    acc
}
```

Sums of Greeks are the *first* number a risk manager wants. Net Delta
$\to$ residual directional exposure to be hedged. Net Gamma $\to$ how
much the Delta will move on a 1% spot shock. Net Vega $\to$ the loss
on a 1-vol-point IV shift.

### 8.6.4 Scenario P&L

A "what-if" scenario revalues the entire book under shocks. The
function `scenario_pl(&shocks)` does a *full revaluation* — every
option re-priced under the shocked parameters — rather than a Greek
approximation, because for sizeable moves the higher-order terms
matter.

```rust
let new_spot = (p.model.spot * (1.0 + shocks.spot_shock)).max(1e-9);
let new_vol  = (p.model.volatility + shocks.vol_shock).max(1e-6);
/* … */
shocked_value += p.quantity * model.price(p.option.option_type);
```

A standard set of "every morning" scenarios is:

| Spot shock | Vol shock | Description |
|---|---|---|
| -10% | +5pt | Moderate equity sell-off |
| -20% | +15pt | Crash scenario |
| +5% | -2pt | Calm rally |
| 0%   | +5pt | Vol shock with no spot move (event risk) |
| 0%   | -5pt | Vol crush |

### 8.6.5 P&L attribution

Once the day is done you decompose the realised P&L into Greek
contributions:

$$
\Delta P\&L \approx \Delta\,\Delta S + \tfrac12 \Gamma\,(\Delta S)^2 + \nu\,\Delta\sigma + \Theta\,\Delta t.
$$

Whatever cannot be explained is "unexplained" — usually transaction
cost, model error, or an unhedged cross-Greek. A persistent
unexplained P&L is a model risk red flag and triggers a recalibration
or a model upgrade.

```rust
pub fn pl_attribution(&self, spot_change: f64, vol_change: f64, time_change: f64) -> PlAttribution {
    let g = self.total_greeks();
    PlAttribution {
        delta:  g.delta * spot_change,
        gamma:  0.5 * g.gamma * spot_change * spot_change,
        vega:   g.vega * vol_change,
        theta:  g.theta * time_change,
        total:  /* sum */,
    }
}
```

### 8.6.6 Value at Risk

VaR answers: "with probability $\alpha$ (typically 99%), what is the
worst loss tomorrow?" We use Monte-Carlo VaR — for each draw, shock
spot lognormally and vol normally, revalue the book, record the P&L,
and report the $(1-\alpha)$-quantile of the loss distribution.

```rust
let drift = -0.5 * spot_vol * spot_vol * horizon;
let diff  = spot_vol * horizon.sqrt();
for _ in 0..num_simulations {
    let z   = StandardNormal.sample(&mut rng);
    let z_v = StandardNormal.sample(&mut rng);
    let spot_shock = (drift + diff * z).exp() - 1.0;
    let vol_shock  = vol_of_vol * z_v;
    pls.push(self.scenario_pl(&ScenarioShocks {
        spot_shock, vol_shock, rate_shock: 0.0, time_shock: horizon,
    }));
}
pls.sort_by(|a, b| a.partial_cmp(b).unwrap());
let idx = ((1.0 - confidence) * num_simulations as f64).floor() as usize;
let var = -pls[idx];
```

VaR has well-known shortcomings: it is not subadditive (cf. Chapter 3),
and it tells you nothing about the *tail*. Expected Shortfall (ES) —
the average loss in the worst $(1-\alpha)$ tail — is its modern
replacement and is a one-line change to the same Monte Carlo loop.

---

## 8.7 Practical exercises

The repository ships with the engine; these exercises take it for a
spin.

### Exercise 1 — Build a vol surface from real data

`code/examples/volatility_surface.rs` shows the workflow on synthetic
Heston data: for each $(K, T)$ price an option in Heston, then solve
Black–Scholes IV from that price. Replace the Heston prices with
quoted market prices (CSV from your favourite data vendor) and you have
a real surface. Take note of:

1. Discontinuities at the bid/ask bounds — IV is a non-linear
   function of price and a 1¢ change can move it by a vol point on
   short-dated OTM options.
2. The **interpolation problem**: a smile is *not* polynomial in $K$;
   the SVI parameterisation (Gatheral) is the modern standard.

### Exercise 2 — Calibrate Heston

Take an end-of-day SPX option chain. Fit Heston to the chain across
five expiries simultaneously. You should find:

* $\rho$ between $-0.7$ and $-0.9$ (strong leverage effect),
* $\xi$ around 0.5,
* $\theta$ slightly above the long-run realised variance,
* Mean-reversion $\kappa$ between 1 and 3.

### Exercise 3 — Delta hedge under transaction cost

Simulate a long call. Hedge daily, then hourly, then every minute.
Plot total P&L variance against rebalance frequency. You will see the
classic *U-shape*: too rare, hedging error dominates; too frequent,
transaction cost dominates. The minimum is the **Whalley–Wilmott
optimal frequency** that depends on $\Gamma$, transaction cost
percentage, and the trader's risk-aversion.

### Exercise 4 — Tree vs PDE vs MC for the same payoff

Price the same one-year American put with $S=K=100$, $r=5\%$,
$\sigma=20\%$ with all three solvers in `code/src/`. Plot price vs
compute time. The crossover point — beyond which Monte Carlo
overtakes the PDE — depends on the dimension of the payoff: in 1D
the PDE wins; in 4D MC wins; in 2D it is a wash and depends on
implementation quality.

---

## Conclusions

The chapter started with one childlike object — a contract paying
$(S_T - K)^+$ — and unfolded into the dashboard of a modern options
desk: closed-form pricers, Greek aggregation, finite-difference and
Monte-Carlo solvers for the path-dependent cases, a stochastic
volatility model fit to the smile, and a portfolio risk system that
explains every cent of the day's P&L.

The Black–Scholes model is wrong but **the language of Black–Scholes
is right**: traders quote in implied vol, hedge in Delta, and explain
P&L in Vega and Theta. The remaining 21st-century work is in the
adjustments: smile, skew, jumps (Bates 1996), rough volatility
(Bayer–Friz–Gatheral 2016), and machine-learned hedging (De
Marco–Jacquier 2024). Every one of these sits on the foundation laid
in this chapter.

In Chapter 9 we will see how the option desk interacts with the
portfolio optimiser of Chapter 3 — option-augmented portfolios are no
longer well-described by mean-variance, and the Greeks become
*constraints* rather than mere reports.

---

## Cross-references

* **Chapter 1** (stochastic calculus) — Itô's lemma applied above.
* **Chapter 2** (microstructure) — explains the bid-ask cost in
  delta-hedging error.
* **Chapter 3** (portfolio theory) — VaR's lack of subadditivity, ES
  as the coherent alternative.
* **Chapter 4** (machine learning) — calibration as an optimisation
  problem; learned-hedging frontier.
* **Chapter 5** (low-latency) — the rebalance frequency trade-off.
* **Chapter 6** (information theory) — entropy of the implied
  distribution vs the true distribution as a model-risk metric.
* **Chapter 7** (game theory) — a market maker is in a repeated game
  with informed traders; the smile is partly an inventory-management
  signal.

## References

1. Black F., Scholes M. (1973). *The Pricing of Options and Corporate
   Liabilities.* Journal of Political Economy.
2. Merton R. C. (1973). *Theory of Rational Option Pricing.* Bell
   Journal of Economics.
3. Cox J. C., Ross S. A., Rubinstein M. (1979). *Option Pricing: A
   Simplified Approach.* Journal of Financial Economics.
4. Heston S. L. (1993). *A Closed-Form Solution for Options with
   Stochastic Volatility with Applications to Bond and Currency
   Options.* Review of Financial Studies.
5. Dupire B. (1994). *Pricing with a Smile.* Risk Magazine.
6. Longstaff F., Schwartz E. (2001). *Valuing American Options by
   Simulation: A Simple Least-Squares Approach.* Review of Financial
   Studies.
7. Lewis A. (2001). *A Simple Option Formula for General
   Jump-Diffusion and Other Exponential Lévy Processes.* SSRN.
8. Albrecher H., Mayer P., Schoutens W., Tistaert J. (2007). *The
   Little Heston Trap.* Wilmott Magazine.
9. Gatheral J. (2024). *The Volatility Surface: A Practitioner's
   Guide,* updated edition. Wiley.
10. Bergomi L. (2025). *Stochastic Volatility Modeling,* 2nd edition.
    Chapman & Hall/CRC.
11. De Marco S., Jacquier A. (2024). *Deep Learning for Option
    Pricing and Hedging.* arXiv preprint.
12. Horvath B., Jacquier A., Tankov P. (2023). *Learning Rough
    Volatility: Neural Networks for Pricing and Calibration.*
    Quantitative Finance.
