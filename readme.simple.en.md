# Chapter 8 — Options and Greeks, the Simple Story

## What this chapter is about

Stocks go up or down — that's the whole game. *Options* are different. With
an option you buy the *right*, but not the *obligation*, to buy or sell a
stock at a fixed price by some date. That little asymmetry — "I might use it,
I might not" — turns out to be one of the most powerful tools in modern
finance.

This chapter teaches you how to think about options the way professional
traders do: as an inventory of *risks* you choose to carry, each measured by
a Greek letter, each with its own price and its own way of moving.

## Three real-life analogies

### 1. The flight ticket (a call option)

You buy a refundable airline ticket for €200 to fly to Paris in three months.
If your boss tells you you have to be in Tokyo that week instead, you cancel
and lose nothing — the cost was just the *option* of going. If a flight on
the same day suddenly costs €600 (it's a bank holiday, everybody wants to
go), your ticket is now extremely valuable: you can use it, or sell it, or
re-issue it for someone else.

That refundable ticket is a *call option*. The €200 you paid is the
*premium*. The €600 ticket is the *spot price* of the underlying. The
strike is what you locked in: €200. As the date gets closer and other
flights stay expensive, your ticket gets more valuable. As the date gets
closer and other flights drop to €100, your ticket is worthless — you'd
rather just buy a new one.

### 2. The fire-insurance policy (a put option)

You pay €500 a year to insure your €300,000 house against fire. Most years
the house doesn't burn down and the €500 is "wasted". But that €500 is what
makes you sleep at night, because if the house *does* burn, the insurer
pays you €300,000.

That is a *put option* on your house, with a strike of €300,000. As long
as house prices stay above the strike, the put expires worthless. If
prices crash to €100,000 (or the house burns), the put pays out. The
€500 a year is the premium that buys protection against the downside.

### 3. The lemonade-stand bet (Delta hedging)

You and a friend run a lemonade stand. You bet that on hot days you'll sell
20 cups, on cold days only 5. To make your income smoother, you and a
*friend* with an ice-cream stand strike a deal: every cup of lemonade you
sell, your friend gives you €0.20 of their ice-cream profit; in exchange,
every cup you fail to sell on a cold day, your friend keeps a little extra.

You have just *delta-hedged* your weather exposure. You don't care about
the weather any more — your weather sensitivity (your "Delta") has been
matched by the friend's opposite Delta. You both still make money, but
neither of you is gambling on the temperature.

That is what every option desk in the world does, every minute, against
the underlying stock.

## Topics in plain language

### Black–Scholes is one big "if"

Black, Scholes and Merton (1973) gave a closed-form formula for the price
of an option, *if* you assume:

- The underlying drifts smoothly with constant volatility.
- You can trade as often as you want with no cost.
- Interest rates are constant.

None of those is exactly true. But the formula is so good as a *frame of
reference* that it became the language traders speak even when they
disagree with it. When somebody says "tech is trading at 35 vol" they
mean: *the Black–Scholes model that produces today's market price assumes
35% annual volatility*.

### The Greeks are your dashboard

An option price moves for many reasons. The Greeks decompose the move:

- **Delta**: how much it moves when the *stock* moves €1.
- **Gamma**: how Delta itself changes as the stock moves.
- **Vega**: how much it moves when *volatility* moves 1 vol point.
- **Theta**: how much it loses each day, just from time passing.
- **Rho**: how much it moves with interest rates.

Every overnight P&L on an options desk is decomposed into these buckets.
"We made €40k Delta P&L, lost €30k on Theta, were flat on Vega" tells you
exactly why the desk made or lost money.

### American options are harder than European

A *European* option can only be used on the expiry date. That is what
Black–Scholes prices. An *American* option can be used any day. That extra
flexibility is worth real money — but the price has no closed form. You
have to roll backward through time, asking at every step: "is it better
to use the option now, or hold it one more day?". That backward induction
is what binomial trees, finite-difference grids and Longstaff–Schwartz
Monte Carlo all do, in three different ways.

### Volatility itself is volatile

The Black–Scholes "constant volatility" assumption is comically wrong. In
real markets, volatility is itself a stochastic process — it rises during
crashes (the "leverage effect") and is mean-reverting. The Heston (1993)
model captures this with a CIR-style variance process. The result is a
*volatility smile*: low strikes have higher implied volatility than high
strikes, especially for short maturities. Every modern derivatives desk
prices off a smile model, not Black–Scholes.

### Risk reports are the day job

For a real options book, the math above is a tool. The day job is:

- Net Greeks: how Delta, Gamma, Vega add up across hundreds of positions.
- Scenarios: "if SPX drops 5% and vols add 5 points, what do we lose?"
- VaR: with 99% confidence, our worst loss tomorrow is X.
- Attribution: "today we lost €50k; €30k was from Vega, €20k from Theta".

This chapter implements all of that in code so you can see the dashboard
move as you change the inputs.
