---
status: proposed
---

# A forecast-reading rule holds an expected-returns estimator, and a covariance enters on the constraint that reads it

## Context

[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
fixed the shape of online portfolio selection: one head, `OnlinePortfolioSelection`, the rule on
`alg`, a Rule State beside the rows held once
([ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md)),
and the head's field list — its constraints on `set`, then `fb`, `strict`, `cache` — with **no prior slot**.
[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
named the roster and deferred to this decision the spelling of the variants that change *what is
read*: the exponential-moving-average reversion and the cost-aware rule's two forecasts.
[Issue #1154](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1154) asks where a
Prior enters an online update, what an absent prior means, what a prior that does not fold does,
which blocks of a Prior a rule may read, and which internal matrices are never a Prior's.

The map's ask is that the family *adapt to the priors already in the library*. Six facts from the
two ledgers and the code shaped the decision.

- **Thirteen ledger rows read a forecast** of the next price relative, a vector `x̂_{t+1}` that
  would change if another predictor were plugged in: a simple moving average of *price levels*
  over the last price (Li and Hoi 2012; the cost-aware rule's second form), an exponential moving
  average of levels (Li, Hoi, Sahoo and Liu 2015), a spatial median of levels by Weiszfeld's
  iteration (Huang, Zhou, Li, Hoi and Zhou 2013), the peak of a window of levels (Lai, Dai, Ren and
  Huang 2018; Lai, Yang, Fang and Wu 2018), the lagged price `1 ./ x_t` (Li, Wang, Huang and Hoi
  2018), and a matched conditional sample, which ADR 0156 already made a Sample Selector. Every
  other row reads `x_t` through a loss or a gradient. **No row reads a covariance, scenarios, or
  any moment above the first.**
- **Every price-level statistic reads exactly off the rows the head holds.** The five statistics
  are homogeneous of degree one in the levels, so `x̂ = stat(p_{t−w+1:t}) / p_t` is a function of
  the last `w − 1` price relatives with `p_t = 1`: on three assets with `x₂ = [0.90, 1.05, 1.00]`
  and `x₃ = [1.05, 1.00, 0.98]`, the three-level moving average gives
  `x̂ = (1 + 1/x₃ + 1/(x₂x₃))/3 = [1.0035, 0.9841, 1.0136]` from the rows and the same from the
  levels. As an expected return that is `mu = x̂ − 1`.
- **The library's one-vector idiom is `me::AbstractExpectedReturnsEstimator`.** `EmpiricalPrior`
  holds one and reads `mean(me, X, pnl; dims)`; that seam carries no factor returns. A Prior
  answers `prior(pe, X, F, pnl).mu`, but `LowOrderPrior` refuses an empty `sigma`, so a Prior
  always pays a covariance. No expected-returns estimator over price levels exists
  (`WindowedExpectedReturns`, `ExpWeightedExpectedReturns` and `MedianExpectedReturns` all act on
  returns), and no adapter turns a Prior into an expected-returns estimator, though
  `EquilibriumExpectedReturns(; ce)` turns a covariance estimator into one.
- **The fold-or-refit rule exists.** [ADR 0136](0136-a-prior-folds-and-carries-the-buffer-is-owned-once-and-a-cap-is-either-a-scenario-cap-or-a-window.md):
  a member folds and carries where its fold is exact, and refits from the `Online` buffer
  otherwise; `fold_member(est, x) = supports_partial_fit(est) ? partial_fit!(est, x) : est` is
  that rule in one line, and a batch-only prior with no buffer is refused at `partial_fit!` by
  `assert_sample_buffer`. Under ADR 0157 the head already owns a rows buffer capped by
  `rows_needed`, so the one thing `Online(est; max_history)` does — seed a refit buffer — is done.
- **One consumer on the map reads a covariance**: the constrained update, when a variance,
  tracking-error or uncertainty-set constraint on the projection ships. A held optimiser in a
  follow-the-leader rule reads its own `pe` (ADR 0156). The matrices the rules carry — the
  Newton step's Gram over gradients, the confidence-weighted rule's belief covariance over
  weights, the anti-correlation rule's lagged cross-window correlation, a mixture's expert
  wealths — are none of them a return covariance.
- **The prototype's moving-average and median reversions are one step.** `olmar` and `rmr`
  compute `λ = max(0, (ε − ⟨w, x̂⟩) / ‖x̂ − x̄𝟙‖²)` and `w ← Proj_Δ(w + λ (x̂ − x̄𝟙))` line for
  line; the papers' `α = min(0, (⟨w, x̂⟩ − ε)/…)`, `w − α(…)` is the same expression. They
  differ only in the statistic that forms `x̂`.

## Decision

### The forecast is a slot on the rule, and the head holds no Prior

A rule that reads a forecast holds it on **`me::AbstractExpectedReturnsEstimator`**, and reads
one vector: the **Price Relative Forecast** `x̂ = 1 .+ mu`. The head keeps ADR 0155's field list
and holds no `pe`; the twenty-three rules that read no forecast carry no slot. A rule that needs a
forecast therefore refuses a missing one **at construction, by the field bound**, and ships the
paper's own statistic and default as its default value; an Expert Mixture whose experts differ
only in their window — every paper's `BAH_W` over windows `3:30` — is twenty-eight rules with
twenty-eight forecasters, which a head-level slot could not spell. The Result is
`NaiveOptimisationResult` with **`pr = nothing` for the whole family, in both arms**; a consumer
that scores a result against a carrier is handed one, as it is for `PreviousWeights`.

### A covariance enters on the constraint that reads it

A projection that needs a covariance — a variance bound, a tracking-error bound, an uncertainty
set, whichever the constrained-update decision ships — holds **its own** prior or covariance
estimator, fitted on the rows the head holds at the step. Nothing else on the head reads one: not
the forecast slot, not the rules, not the head. A run under weight bounds alone fits no covariance
ever. `rows_needed(head)` is the maximum over the rule tree *and* the projection, because a
covariance over a reversion's four rows is rank-deficient. When both the forecast slot and a
constraint hold a Prior, the two fit the same rows twice, once for a `mu` whose `sigma` is dropped
and once for a `sigma` whose `mu` is dropped; the exploration of switching off `mu` or `sigma` in a
low-order prior, as `kte = nothing` and `ske = nothing` do in a high-order one, is filed as its own
issue and is not this family's to decide.

### `PriceLevelExpectedReturns(; alg)` is the family's forecaster

One new expected-returns estimator, `PriceLevelExpectedReturns`, whose `alg` slot is bound to an
unexported `AbstractPriceLevelStatistic <: AbstractExpectedReturnsAlgorithm`, as
`ShrunkExpectedReturns(; alg)` is bound. Five statistics ship: `MovingAverage(; window)`,
`ExponentialMovingAverage(; alpha)`, `SpatialMedian(; window, iters, tol)` — the modified
Weiszfeld iteration of Vardi and Zhang (2000) seeded at the coordinatewise median, as the paper
has it and the prototype does not — `WindowPeak(; window)` and `LaggedPrice(; lag)`. Its `mean`
reconstructs the levels from the returns it is handed with the last level at one and answers
`stat / p_t .− 1`, so `EmpiricalPrior(; me = PriceLevelExpectedReturns(...))` is a legal prior and
a search addresses the window as `"alg.me.alg.window" => 3:30`. The window is the statistic's own
field, never `WindowedExpectedReturns`'s, so there is one window knob and it is the paper's.

### `PriorExpectedReturns(; pe)` lets a Prior's mean be a forecast

A Prior enters the slot through a new, library-wide adapter: `PriorExpectedReturns(; pe)` is an
expected-returns estimator whose `mean(me, X, pnl)` is `prior(pe, X, nothing, pnl).mu`. Any `me`
slot in the library may hold it, so a Black–Litterman or shrunk mean may drive a reversion step,
an `ExpectedReturn` risk measure, or an `EmpiricalPrior`. It **refuses at construction a prior
for which `needs_factor_returns(pe) === true`**, because the expected-returns seam carries no
factor returns anywhere in the library; a *take what is given* prior is admitted and fitted
without them. The head's rows buffer is a buffer of returns, which is what every `me` reads
([ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md)).

### A forecaster folds where it can and refits from the head's rows otherwise

Each step the head advances the rule's forecaster by ADR 0136's rule, `supports_partial_fit`
being the question: an estimator with an exact fold — `SimpleExpectedReturns`,
`ExpWeightedExpectedReturns`, a price-level statistic that is a recursion over the relatives
(the exponential moving average, the reweighted relative), a `PriorExpectedReturns` over a prior
the predicate says folds — is folded on the row, and **its state rides on the Rule State's `st`**
as a `ForecasterState`, so a mixture's experts each carry their own; an estimator with no exact
fold is refit on the rows the head holds. `EmpiricalPrior` carries its rows as memory rather
than folding its moments alone, so the library's predicate answers refit for it and the head's
rows are that memory, held once. `rows_needed(me)` is one method per estimator: `0` for one that
folds, `window − 1` for a windowed price-level statistic, `lag` for the lagged price, `window`
for `WindowedExpectedReturns`, and unbounded for a batch-only estimator or Prior, which then
refits on the whole prefix at `O(tN)` a step — the docs state the cost and
`WindowedExpectedReturns(; me, window)` is the user's cap. **`Online(me)` in the slot is refused
by name**: the head's buffer is the buffer, and a second one would hold the rows twice, which
ADR 0157 rejected for the rules. A forecaster carrying a state at the door is refused too: the
head starts cold.

**The cold start truncates the window.** Over the first rows a windowed statistic reads the
levels available — with `window = 5` and two rows folded, three levels — as the reversion
papers' reference implementations do and as the parity test with the prototype requires; a
folding statistic starts from its seed; a composite truncates every window it holds; and the
anti-correlation rule's two windows and a `LastRows` selector truncate the same way. A forecast
is **flat** — one in every asset, which every rule's step holds on — where the forecaster has
fewer rows than its second moment needs (a Prior, a variance or a shrinkage over one row) or
answers a non-finite entry, so a forecaster undefined over the first rows holds until it is
defined instead of raising or stepping on `NaN`.

### `ForecastReversion` is one rule; two paper names are its constructors

The passive-aggressive step toward a Price Relative Forecast — `min ‖w − w_t‖² s.t. ⟨w, x̂⟩ ≥ ε`,
closed form plus projection — is **one struct, `ForecastReversion(; me, eps)`**, named for its
mechanism because two papers share the update and neither names it. `MovingAverageReversion(;
window = 5, eps = 10)` and `RobustMedianReversion(; window = 5, eps = 5, iters, tol)` are
**constructors** of it, filling `me` with the paper's statistic and `eps` with the paper's default,
as `UniversalPortfolio` constructs the mixture. The roster keeps both paper names, marked
constructor. `PassiveAggressiveMeanReversion` stays a distinct rule: its constraint runs the other
way on the realised relative, `⟨w, x_t⟩ ≤ ε` with `ε ≤ 1`.

The variants ADR 0156 deferred here are spelled: the exponential-moving-average reversion is
`ForecastReversion(; me = PriceLevelExpectedReturns(; alg = ExponentialMovingAverage(; alpha = 0.5)), eps = 10)`;
the cost-aware rule's two forecasts are `TransactionCostOptimisation(; me = PriceLevelExpectedReturns(; alg = LaggedPrice(; lag = 1)))`,
its default, and the same under `MovingAverage(; window = 5)`; peak price tracking and the sparse
portfolio read `WindowPeak(; window = 5)`. `LaggedPrice(; lag = 0)` is the current price, the
*hold* branch of a switched statistic. The later papers' forecasts are composites over the same
statistics, on the `alg` slot like any other: `TruncatedExponentialMovingAverage`,
`GaussianWeightedDoubleEstimate`, `TrendSwitch` over a trend test and `CompositeTrend`; the
Gaussian weighting reversion and the local adaptive learning are constructors of
`ForecastReversion`, the adaptive input and composite trend representation and the trend-promote
price tracking constructors of `ForecastTracking`.

### What a rule may read, and what is never a Prior's

The forecast slot serves **`mu` alone**. No rule reads `sigma`, `X`, `ens` or a higher moment from
its forecaster — the ledger has no row that does, so "beyond second order" is empty by
construction and not by refusal. The Newton step's Gram matrix and its `b`, the confidence-weighted
rule's belief covariance, the anti-correlation rule's lagged cross-window correlation and a
mixture's expert wealths are **Rule State carriers on `st`**: a covariance estimator is never
offered for them, because they are statistics over gradients, weights, lagged windows or experts,
not over returns.

## Considered options

1. **A head-level `pe::Option`, default `nothing`**, the rule reading `1 .+ pr.mu`. Rejected: a
   `LowOrderPrior` pays `sigma` every step; a rule that needs a forecast can refuse only at the
   first step; every expert of a mixture reads one forecast; and once the rule has a slot, the
   head's is a second knob for one input.
2. **Both slots** — the rule's for `x̂`, the head's for `sigma` and `X`. Rejected: a dead field
   until a covariance constraint ships, and a `pr` that is sometimes a prior result.
3. **A covariance from the forecast slot's Prior when it is one.** Rejected: a bare
   expected-returns forecaster plus a covariance constraint would be refused, coupling two
   unrelated choices.
4. **A union bound over expected-returns estimators and `PrE_Pr`** with a `forecast(slot, X, pnl)`
   dispatch. Rejected: a field name fitting neither type, a verb only this family calls, and a
   prior *result* in the slot duplicating `CustomValueExpectedReturns(vec)`.
5. **`pe::PrE_Pr` only**, the price-level statistics inside `EmpiricalPrior(; me = …)`.
   Rejected: every paper default fits a covariance it never reads.
6. **One struct per statistic** (`MovingAverageExpectedReturns`, …). Rejected: five docstrings and
   five `rows_needed` methods for one formula family, and no shared slot type for a search to
   address.
7. **`WindowedExpectedReturns` as the window.** Rejected: two window knobs for the exponential
   average and the lag, and a wrapper window over returns rows that is `window − 1` in the
   paper's terms.
8. **Always refit from the rows, forecasters stateless.** Rejected: a full-prefix mean costs
   `O(T²N)`, and the exponential moving average of levels would run as a batch over an unbounded
   buffer instead of its one-line recursion.
9. **Always fold; refuse a forecaster with no exact fold.** Rejected: it refuses
   `MedianExpectedReturns`, `ShrunkExpectedReturns`, the spatial median and every Prior beyond
   `EmpiricalPrior`.
10. **Admit a factor prior by widening `mean(me, X, pnl; F)`** and recording `F` in the head's
    buffer. Rejected: a library-wide seam change for a forecast no ledger row reads.
11. **Keep `MovingAverageReversion` as the struct**, `RobustMedianReversion` a constructor.
    Rejected: the name would be a misnomer the moment the slot holds a median or a Prior mean.
12. **Two structs sharing one update method.** Rejected: the shape ADR 0156 rejected for the
    solved rows.
13. **An own Result type for the family**, because `pr` is always `nothing`. Withdrawn once the
    covariance's home was settled: with no Prior on the head, `pr = nothing` is uniform across the
    family and `NaiveOptimisationResult` already admits it.

## Consequences

- `CONTEXT.md` gains *Price Relative Forecast*; *Expected Returns Estimator* lists
  `PriceLevelExpectedReturns` and `PriorExpectedReturns`; *Online Selection Rule* and the roster
  name `ForecastReversion` with `MovingAverageReversion` and `RobustMedianReversion` as
  constructors.
- ADR 0156 is a draft on `dev` and is rewritten in place: the set-1 table and the naming
  section name `ForecastReversion` and its two constructors, and the forecast-variants sentence
  points here. ADR 0155's example rule list is unchanged — it names the roster as a later
  ticket's.
- The constrained-update decision meets a covariance that lives on its own objects, a
  `rows_needed` that takes the maximum with the rule tree's, and a `w′` that is a fresh vector
  (ADR 0157); it does not reopen where a Prior sits.
- An exploration issue asks whether `EmpiricalPrior`'s `me`/`ce` and `LowOrderPrior`'s
  `mu`/`sigma` can be switched off as a high-order prior's `kte`/`ske` can. It is filed from this
  decision and is not on the map.
- The build tickets graduate: `PriceLevelExpectedReturns` and its five statistics, with the
  Weiszfeld correction and a parity test against the prototype at the moving average;
  `PriorExpectedReturns` and its factor refusal; the `me` slot on `ForecastReversion`,
  `TransactionCostOptimisation`, `ForecastTracking` (whose constructor is `PeakPriceTracking`) and
  `ShortTermSparsePortfolio`; the `scale` slot on `ForecastReversion`, the `ReweightedPriceRelative`
  statistic and the constructors `ReweightedPriceRelativeTracking` and
  `ExponentialMovingAverageReversion` (ADR 0165);
  `rows_needed(me)`; the fold-or-refit of a forecaster on the Rule State; the `Online(me)`
  refusal. The forecast arm is built (#1176): the forecast-reading rules live in their own
  file, the statistics of the later papers in a second, and the kernel-trend pattern tracking —
  a stateful statistic over an elastic-net path — is split into a ticket of its own.
- A factor forecast is recorded as fog on the map, should one ever be wanted; nothing on the
  ledger asks for it.
