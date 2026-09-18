---
status: proposed
---

# Log-wealth regret is a verb over two prediction results, and a hindsight comparator is an estimator fit on the rows it is scored on

## Context

[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
made online portfolio selection one naive head whose batch verb is a Causal Pass, and rejected the
hindsight best constant rebalanced portfolio as that head's batch verb (its option 5) because the
library already solves it: `LogarithmicReturn` is a `JuMPReturnsEst`, so `MeanRisk` under a log
return and a maximum-return objective is Cover's (1991) comparator.
[ADR 0156](0156-every-online-selection-algorithm-ships-as-a-closed-form-rule-an-expert-mixture-or-a-follow-the-leader-over-a-selected-sample.md)
shipped the universal portfolio as `ExpertMixture` over sampled `ConstantRebalancedPortfolio`
experts and made `FollowTheLeader(; sel, opt)` a rule that re-solves an optimiser on a selected
sample, and left to this decision what a caller measures the family with.
[Issue #1157](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1157) asks where
log-wealth regret lives, how the two hindsight benchmarks run, what the walk-forward already
scores, and what the family's docs show.

The literature quotes one number, the **regret** `R_T = log S_T(comparator) − log S_T(strategy)`
over one price sequence, with the comparator read in hindsight: the best constant rebalanced
portfolio (Cover 1991), the best single asset, or the uniform constant rebalanced portfolio (Li
and Hoi 2014, §3.1). The prototype's `regret_against_bcrp(r, X)` fits Cover's (1984) fixed point
inside the verb and returns the total, the per-period form and both terminal wealths. Five facts
measured on `dev` at `dfac7af1f1` shaped the decision.

- **The search already ranks on log wealth.** A candidate's fold score is
  `sgn * expected_risk(r, p)` (`09_Base_SearchCrossValidation.jl:694`), and `MeanReturn(; flag =
  true)` is `mean(log1p.(x))` with `bigger_is_better`. A comparator fitted on the same rows is a
  constant across the candidates, so ranking on regret is ranking on `MeanReturn(; flag = true)`:
  a regret scorer would change no argmax.
- **The performance summary reads one series.** `performance_summary(ret; …)`
  (`18_ExpectedReturns.jl:1398`) has no comparator slot, and every column is a property of that
  series alone.
- **The library's a-versus-b shape exists.** `covariance_forecast_compare(a, b; lags)`
  (`14_CovarianceForecastSummary.jl:401`) reads two evaluation results, refuses unless their
  `dates` agree, and returns a comparison Result holding the per-step difference's mean, its
  Newey–West long-run variance at `lags` lags, a Diebold–Mariano–West `z` and its two-sided `p`.
  `PredictionResult` carries `rd`, the portfolio's return series with its timestamps, so two
  prediction results hold everything the shape needs.
- **Both oracles are reachable today.** `MeanRisk(; obj = MaximumReturn(), opt = JuMPOptimiser(;
  ret = LogarithmicReturn(), wb = WeightBounds(0, 1), bgt = 1))` reproduces the prototype's fixed
  point to `1.8e-12` in weights (the seams ledger, issue #1150); `NearOptimalCentering`,
  `RiskBudgeting`, `RelaxedRiskBudgeting` and `FactorRiskContribution` hold the same
  `JuMPOptimiser` and its `ret` slot. An Asset Selector `ScoreSelector(RankRule(; best = 1),
  MeanReturn(; flag = true))` keeps the asset with the highest log wealth, and `EqualWeighted` on
  one asset is its unit vector. Neither reads anything but the rows it is given, so each is a
  hindsight oracle exactly when it is fit on the rows it is then scored on, and a causal comparator
  when a walk-forward fits it on the training window.
- **The fixed point is the one solver-free route, and it has a second job.** Cover's (1984)
  iteration `w_i ← w_i · (1/T) Σ_t x_{t,i} / ⟨w, x_t⟩` maximises `Σ_t log⟨w, x_t⟩` on the plain
  simplex and nowhere else. `FollowTheLeader` re-solves its `opt` every period, so with `MeanRisk`
  it is `T` solver calls and with the fixed point it is `T` closed-form iterations.

## Decision

### Log-wealth regret is a verb over two prediction results, returning a Result

`log_wealth_regret(a::PredictionResult, b::PredictionResult; lags = 0)` returns a
`LogWealthRegretResult`. `a` is the strategy and `b` the comparator; the verb refuses unless the
two carry the same timestamps, because regret is defined over one sequence. It reads each series
as it was scored, fees included, so the comparator's fee policy is the caller's. The Result holds:

- `regret`, `Σ_t log(1 + r_{b,t}) − Σ_t log(1 + r_{a,t})`, positive when the comparator wins —
  the prototype's sign and the survey's Eq. 2;
- `regret_per_period`, the total over the number of periods;
- `difference`, the per-row series `log(1 + r_{b,t}) − log(1 + r_{a,t})`, whose mean is the
  per-period regret;
- `variance`, `z`, `p`, `lags` and `n_periods`: the Newey–West long-run variance of `difference`
  at `lags` lags, the Diebold–Mariano–West statistic and its two-sided `p` under equal expected
  log growth, computed by the same `newey_west_variance` the covariance comparison uses;
- `wealth_a` and `wealth_b`, the two terminal wealths.

The test is exact only for a comparator that did not read the rows; against a hindsight
comparator it is optimistic by construction, and the docstring says so. The verb accepts the
prediction results `performance_summary` accepts.

No scorer and no summary column ship. The search ranks on `MeanReturn(; flag = true)`, which is
log wealth per period and orders candidates as regret against any fixed comparator would; a
summary column would need a comparator the summary does not hold.

### A hindsight comparator is an estimator fit on the rows it is scored on

The **Hindsight Comparator** is a rule, not a type: fit an estimator on the evaluation rows and
predict it in sample over the same rows, `predict(optimise(est, rd_test), rd_test)`; run the same
estimator through `cross_val_predict` with the online head's `cv` for the causal comparator over
the same rows. No oracle is a member of the online family, because a rule that reads the whole
buffer breaks the Causal Pass.

The docs name the literature's two oracles as recipes of that rule, and state the generalisation:

- the best constant rebalanced portfolio is `MeanRisk` under `LogarithmicReturn` and
  `MaximumReturn` on the simplex, and any JuMP optimiser with a `ret` slot under the same return
  is its own log-return oracle — a risk-budgeted or risk-penalised constant portfolio in hindsight
  is one keyword away;
- the best stock is the top-1 Asset Selector under `MeanReturn(; flag = true)` composed with
  `EqualWeighted`, and `RankRule(; best = k)` is the best `k` stocks equal-weighted.

One naive head ships beside the rule: `BestConstantRebalancedPortfolio <:
NaiveOptimisationEstimator`, Cover's (1984) fixed point with `iters` and `tol` fields at the
prototype's defaults, simplex only, taking the naive family's fields (`wb` through the finaliser,
`cache` as a `ReturnsBufferState` under ADR 0137's batch read-out, `fees` under ADR 0160, `fb`,
`strict`). It is the solver-free comparator for a caller with no JuMP solver, the parity
cross-check the seams ledger ran, and the solver-free `opt` of `FollowTheLeader` on the default
set — under a bounded set it answers the repaired fixed point, and a programme set is refused
([ADR 0164](0164-a-follow-the-leader-rule-solves-its-programme-on-the-allocation-set-and-projects-only-its-damped-mix.md)).
A bounded or constrained comparator stays `MeanRisk`.

### The universal portfolio's bound is stated twice, each statement attributed

The `UniversalPortfolio` constructor's docstring states Cover's (1991) bound `(N − 1) log(T + 1)`
for the exact integral over the simplex; then the mixture's own bound, exact for the shipped
object, `log S_T(best sampled expert) − log S_T(mixture) ≤ log n_experts` for every sequence; then
that the gap between the best sampled expert and the best constant rebalanced portfolio is
sampling error that shrinks with `n_experts` and has no closed form; then the per-row cost
`O(n_experts · N)` and the `n_experts × N` carrier, which is why the method is a reference and not
a workhorse.

### The docs are one chapter, one example, the catalogue rows and the API pages

One docs ticket ships after the last build and blocks the map's verification ticket. It owes:

- a user-guide chapter on the family, on synthetic `StableRNG` fixtures: a mean-reverting and a
  trending market, the reversion and the momentum rules side by side with terminal wealth per
  regime and the prototype's finding stated — the reversion family is a total bet on a market
  property; a regret table against the three comparators (best constant rebalanced portfolio,
  uniform constant rebalanced portfolio, best stock), each built by the Hindsight Comparator rule;
  and one block at `test_size > 1` under `DriftedWeights()` showing the walk-forward's held path
  diverge from the batch Causal Pass, the difference ADR 0160 documents and does not test;
- one example under the optimisers' examples on the S&P 500 data: the roster in a walk-forward
  with a turnover fee, a search over a rule's rate scored on `MeanReturn(; flag = true)`, and the
  regret table;
- the Capability Catalogue rows and the API pages, grouped by Li and Hoi's (2014) five families as
  ADR 0156 ruled.

## Considered options

1. **A verb that takes the comparator estimator and the asset panel**, `regret(pred, est, rd)`,
   the prototype's shape. Rejected: it hides an in-sample fit inside a verb, re-derives the rows
   from `pred.rd.ts`, and ties regret to a hindsight comparator when the causal uniform constant
   rebalanced portfolio is as valid a `b`.
2. **A docs recipe only**, one line of `log1p` sums. Rejected: no timestamp check, no per-period
   form, nothing the example or a table can cite.
3. **A search scorer or a performance-summary column.** Rejected: the scorer changes no ranking;
   the summary has no comparator.
4. **The regret quantities without the test.** Rejected: the comparison-Result precedent carries
   the statistic, its variance function is one reuse, and the test is the one thing the literature
   does not report.
5. **The oracles as members of the online family.** Rejected: a rule that reads the whole buffer
   breaks the Causal Pass and ADR 0155's option 5; the ledger says "not a rule".
6. **The `MeanRisk` configuration and the recipes with no new head.** Rejected: `FollowTheLeader`
   would always need a solver, and Cover's fixed point would live only in a parity test.
7. **Two naive heads, one for each oracle.** Rejected: a `BestStock` head restates a two-step
   composition the library has as a struct.
8. **A `Hindsight(est)` wrapper** whose walk-forward fit reads the test rows, so
   `cross_val_predict` yields the oracle over the same rows in one call. Rejected for now: a new
   seam in the fold loop and a type whose whole purpose is look-ahead, for regret tables alone; it
   sits on top of this decision and can be added if the example's row slicing proves a nuisance.
9. **Cover's bound with a caveat**, or no bound. Rejected: the caveat does not say what the
   approximation costs, and the exact finite-expert bound would go unsaid.
10. **The family's lesson as a section of the online walk-forward chapter.** Rejected: the
    diagnostic and the regret table are the family's lesson, not the loop's.

## Consequences

- `CONTEXT.md` gains *Log-Wealth Regret* and *Hindsight Comparator*; the §4.1 roster names
  `BestConstantRebalancedPortfolio` as the naive head beside the family.
- The evaluation-surface build ticket owes `log_wealth_regret`, `LogWealthRegretResult`,
  `BestConstantRebalancedPortfolio` with a parity test against `MeanRisk` under
  `LogarithmicReturn` on the seams ledger's interior fixture, a test that the best-stock recipe
  composes, and the two bound statements on `UniversalPortfolio`.
- The docs ticket owes the chapter, the example, the catalogue rows and the API pages, and blocks
  the map's verification ticket.
- The map's decisions are complete: every build ticket can now be specified.
