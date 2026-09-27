---
status: proposed
---

# Every online selection algorithm ships as a closed-form rule, an expert mixture, or a follow-the-leader over a selected sample

## Context

[Map #1148](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1148) brings online
portfolio selection into the library, and
[ADR 0155](0155-an-online-portfolio-selection-head-is-a-naive-optimiser-whose-batch-verb-is-a-causal-pass-and-whose-read-out-is-its-own-recursion.md)
fixed the shape: one head, `OnlinePortfolioSelection`, the update rule on `alg`, a member being a
rule struct, the state of its private carriers, and one update.
[Issue #1153](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1153) asks which
algorithms ship, under what names, and which of the literature ledger's rows are first set,
second set or refused.

The literature ledger
([#1149](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1149)) counts 36 rows over
Li and Hoi's (2014) five families and sorts them by mechanism. Twenty-odd are a closed form
followed by the simplex projection. Five are a wealth-weighted or gradient-weighted average over
a set of experts. Thirteen solve a programme on the simplex every period: follow the leader and
its windowed, weighted and exp-concave variants (Gaivoronski and Stella 2000; Hazan and Kale 2012)
play the log-optimal portfolio over the prefix or a window; every pattern-matching row (Györfi and
co-authors 2006, 2008; Li, Hoi and Gopalkrishnan 2011) selects the past rows whose preceding
window resembles the latest one, plays the log-optimal, semi-log-optimal, Markowitz-type or
cost-aware portfolio over that sample, and aggregates its experts by wealth. The prototype ships
seven closed-form rules and no member of the other two mechanisms.

Three facts from the ledger shaped the decision. The prototype's universal portfolio,
`logS .+= log.(B * x_t); w = B' softmax(logS)`, *is* the wealth-weighted mixture over `K` sampled
constant rebalanced portfolios that every reversion and pattern-matching paper uses for its
headline (`BAH_W`). The wealth-weighted update over experts, `p_{t+1} ∝ p_t .* r_t`, *is* the
uniform buy-and-hold benchmark applied to the expert-return vector, and the online gradient and
online Newton updates of Das and Banerjee (2011) are exponentiated gradient and the online Newton
step applied to the same vector. And every programme a solved row plays is one the JuMP head
already expresses: the log-optimal portfolio is `MeanRisk` under `LogarithmicReturn` (map ground
truth 5, reproduced to 1.8e-12), the exp-concave leader adds `L2Regularisation`, the
Markowitz-type row is a utility objective, and fees are a field.

The maintainer ruled that all of the ledger's functionality ships, and that ADR 0155 is rewritten
where the roster needs it.

## Decision

### Nothing is refused; three sets by mechanism

Every row of the ledger ships, as a struct or as a configuration of one. The rows are partitioned
into **three sets by build mechanism**, so each set is one shared build plus its members and no
set waits on a decision it does not need:

| set | mechanism | structs | served as a configuration |
| :--- | :--- | :--- | :--- |
| 1 | closed form, plus the mixture | `BuyAndHold`, `ConstantRebalancedPortfolio`, `MirrorDescent` (ADR 0165), `NewtonStep`, `PassiveAggressiveMeanReversion`, `ForecastReversion`, `ExpertMixture` | the universal portfolio, the Dirichlet(½) universal portfolio, the uniform constant rebalanced portfolio, the uniform buy-and-hold, every paper's `BAH_W`, the exponentiated gradient as the constructor `ExponentiatedGradient` of `MirrorDescent` (ADR 0165), and the moving-average and robust-median reversions as the constructors `MovingAverageReversion` and `RobustMedianReversion` of `ForecastReversion` (ADR 0158) |
| 2 | closed form, beyond the prototype | `ConfidenceWeightedMeanReversion`, `AntiCorrelation`, `TransactionCostOptimisation`, `ForecastTracking`, `ShortTermSparsePortfolio`, `ExpectationMaximisation`, the weightings `AggregatingAlgorithm`, `TopK` and `SwitchingWeighting` | the exponential-moving-average reversion as the constructor `ExponentialMovingAverageReversion`, peak price tracking as the constructor `PeakPriceTracking` of `ForecastTracking`, the switching portfolio as the constructor `SwitchingPortfolio` of `ExpertMixture` under `SwitchingWeighting`, the gradient projection as the constructor `GradientProjection` of `MirrorDescent` (all ADR 0165), fast universalisation, the online gradient and online Newton updates, `CORN-K` |
| 3 | a solve over a selected sample | `FollowTheLeader`, the selectors `Prefix`, `LastRows`, `HistogramMatch`, `KernelMatch`, `NearestNeighbourMatch`, `CorrelationMatch`, `ClusterMatch`, and `FollowTheLeadingHistory` | follow the leader, the successive, windowed and weighted constant rebalanced portfolios, the exp-concave leader, and the histogram, kernel, nearest-neighbour, semi-log-optimal, Markowitz-type, cost-aware and correlation-driven pattern-matching rules |
| 4 | the online-convex-optimisation reading (ADR 0165) | `AdaptiveSubgradient`, `OptimisticStep`, the geometries `TsallisProjection`, `LogBarrierProjection`, `DiagonalProjection`, the Learning-Rate Schedules, `RiskLoss`, `RankOneCovariance` | online gradient descent, exponentiated gradient with uniform mixing, Soft-Bayes, the anytime and self-confident rates, the doubling trick, optimistic and extra-gradient steps, the dynamic-regret mixtures as the constructors `Ader` and `Sword` of `ExpertMixture`, the reweighted price-relative tracking as the constructor `ReweightedPriceRelativeTracking` of `ForecastReversion`, the short-term loss-control portfolio as the constructor `ShortTermLossControlPortfolio` of `FollowTheLeader`, the risk-aversion correlation-driven rule and the combination-weights gradient descent as configurations |

A set is a membership, not a build order; the build tickets graduate from the map once the state,
Prior-slot, constrained-update and fee tickets land, and set 1 waits on nothing those tickets add
beyond what the prototype already computes inside its rules.

### The universal portfolio is the expert mixture over sampled constant rebalanced portfolios

`ExpertMixture(; experts, alg, eset, proj, grad, p0)` (the last two from ADR 0165) is a rule whose state is every expert's state plus the weight
vector `p_t` over the experts, whose update steps every expert and moves `p_t` by `alg`, and whose
answer is `Σ_k p_{t+1,k} h_k(t+1)`. `ConstantRebalancedPortfolio(; w)` is a rule whose update is
`w_{t+1} = w`, uniform by default. `UniversalPortfolio(; n_experts, rng, prior)` is a constructor
that samples `n_experts` experts from `prior` on the simplex and returns the mixture over them;
Cover's uniform prior is the default and Cover and Ordentlich's (1996) Dirichlet(½) is the same
constructor under another `prior`. The universal portfolio has no struct of its own, because the
prototype's update is the mixture's update and a leaf rule would write it a second time over
matrix rows.

### The mixture's weighting is a rule of the family over the expert-return vector

`alg` on `ExpertMixture` is any `AbstractOnlinePortfolioSelectionAlgorithm`, applied to
`r_t = (⟨h_k(t), x_t⟩)_k` in place of `x_t`. `BuyAndHold()` is the wealth-weighted mixture every
paper writes as `BAH_W`, and the default. `ExponentiatedGradient(; eta)` (the entropic
`MirrorDescent`, ADR 0165) is the online gradient update; `SwitchingWeighting(; gamma)` is the
wealth step followed by a fixed share (ADR 0165); `NewtonStep()` is the online Newton update; `AggregatingAlgorithm(; eta)`, the power
weighting `p ∝ p .* r.^eta` of Vovk and Watkins (1998), has buy-and-hold at `eta = 1`;
`TopK(; k)` holds the `k` experts of greatest wealth and is `CORN-K`. A weighting carries its own
state — nothing for buy-and-hold, a `K × K` Gram matrix for the Newton weighting — and the docs
say what a Newton weighting over two thousand sampled experts costs. Follow the leading history
(Hazan and Seshadhri 2009), whose expert set grows and is pruned each period, is its own rule in
set 3 because a mixture over a fixed expert set cannot spell it.

### A solved row is a follow-the-leader over a selected sample

`FollowTheLeader(; sel, opt, gamma)` is a rule whose update selects the past rows `sel` names,
runs `optimise(opt, rows)` on them, and answers `(1 - gamma) w*_t + gamma w_t`; `gamma = 0` by
default. `sel <: AbstractSampleSelector` is `Prefix()` (every row so far; follow the leader),
`LastRows(W)` (the variable rebalanced portfolio), or one of the pattern-matching selectors,
which name the rows whose preceding `window` resembles the latest one by histogram cell
(`HistogramMatch`), kernel radius (`KernelMatch`), nearest neighbours (`NearestNeighbourMatch`) or
Pearson correlation (`CorrelationMatch`). `opt` is an optimisation estimator, `MeanRisk` under
`LogarithmicReturn` and a maximum-return objective by default; `L2Regularisation` on it is the
exp-concave leader, a utility objective the Markowitz-type row, a quadratic objective the
semi-log-optimal row, fees the cost-aware row. An empty selection answers the uniform portfolio,
as the papers do. The pattern-matching papers' aggregation over `(window, l)` experts is the
`ExpertMixture` of set 1 over `FollowTheLeader` rules.

This rewrites two sentences of ADR 0155, which is a draft on `dev`. *A member fits no moment and
solves no programme* becomes *a member fits no moment and solves no programme of its own; a rule
may hold an optimisation estimator and re-solve it on the rows it selects*. *The state holds no
column buffer* becomes *the state holds no shared column buffer; a rule's private carrier may be
the rows it re-solves on, unbounded for a prefix*. The batch–online identity is unchanged: both
arms run the same solves on the same rows.

### Names derive from one rule

A rule struct is named by **the paper's own name for the algorithm, in full words and British
spelling; a leading "Online" is dropped because the family says it; a trailing "Optimisation" or
"System" is dropped because the head says it; "Portfolio" stays where the paper names the
portfolio.** The acronym goes in the docstring and in `CONTEXT.md`, never in a type name. The
rule gives the table above; the one name it costs recognisability is `NewtonStep` for `ONS`,
accepted. Where two papers share one update and neither names the mechanism, the struct is named
for the mechanism and each paper's name is a **constructor** of it, as `UniversalPortfolio`
constructs the mixture: the moving-average and robust-median reversions are one passive-aggressive
step toward a forecast, `ForecastReversion(; me, eps)`, and `MovingAverageReversion(; window, eps)`
and `RobustMedianReversion(; window, eps, iters, tol)` fill its forecaster with the paper's statistic
(ADR 0158).

A paper's numbered variants that change **one formula** inside one rule are a **typed slot** on
the struct, as `Variance(; formulation)` does: `PassiveAggressiveMeanReversion(; slack)` with
`NoSlack()`, `LinearSlack(; C)` and `QuadraticSlack(; C)` for `PAMR-0/1/2`;
`ConfidenceWeightedMeanReversion(; formulation)` with `VarianceUpdate()` and
`StandardDeviationUpdate()` for `CWMR-Var/Stdev`. Variants that change **what is read** —
`OLMAR-2`'s exponential moving average of price levels, `TCO-2`'s moving average — are the same
rule under a different forecast on its `me` slot, spelled by
[ADR 0158](0158-a-forecast-reading-rule-holds-an-expected-returns-estimator-and-a-covariance-enters-on-the-constraint-that-reads-it.md).

### The survey taxonomy is a docs grouping

Benchmark, follow-the-winner, follow-the-loser, pattern-matching and meta-learning are section
headings in the user guide and the Capability Catalogue, and a column of the `CONTEXT.md`
roster. They are not abstract types: nothing dispatches on them, and the mixture and the Newton
weighting each straddle two. The type tree under `AbstractOnlinePortfolioSelectionAlgorithm` is
flat, plus `AbstractSampleSelector` for the `sel` slot and one small abstract type per variant
slot, because a slot needs a bound.

## Considered options

1. **The universal portfolio as its own leaf rule** with `B::Matrix` and `logS`. Rejected: the
   mixture must exist for `BAH_W`, and the leaf would repeat its recursion over matrix rows for
   a better constant on one algorithm.
2. **The universal portfolio as a docs-only benchmark**, dominated by the online Newton step in
   regret per cost. Rejected: the family's founding member absent, and the mixture without its
   first user.
3. **The solved rows refused** as a walk-forward configuration — follow the leader is
   `WalkForward(; test_size = 1)` over `MeanRisk` under `LogarithmicReturn` with an expanding
   window, exactly. Rejected by the maintainer: all of the functionality ships, and a rule can be
   an expert inside a mixture where a walk-forward cannot.
4. **One leaf rule per solved paper.** Rejected: thirteen structs sharing one update.
5. **Pattern-matching selectors as Priors.** A matched sample is a conditional distribution,
   which is what a Prior is here. Rejected for this map: where a forecast enters a rule is the
   forecast-reading arm's ticket (ADR 0158 put it on the rule's `me` slot), and a Prior re-fit each period inside a naive head is the JuMP
   head's walk-forward with more steps. A selector is a small enough object to become a Prior
   later if one is wanted.
6. **The mixture weighted by wealth only**, with the online gradient and Newton updates as
   separate rules. Rejected: two more structs repeating the expert loop.
7. **A closed weighting enum** (`WealthWeighted`, `GradientWeighted`, `NewtonWeighted`).
   Rejected: it rewrites exponentiated gradient and the Newton step under second names.
8. **Two sets, the prototype then the rest.** Rejected: the rest mixes three mechanisms, so it
   is not one build.
9. **Sets by the survey taxonomy.** Rejected: follow-the-winner alone spans all three
   mechanisms.
10. **Keep "Online" where the acronym carries it** (`OnlineNewtonStep`,
    `OnlineMovingAverageReversion`). Rejected: redundant inside `OnlinePortfolioSelection(; alg)`,
    and ADR 0155's `MovingAverageReversion` would move.
11. **Acronyms as struct names.** Rejected: the library spells `HierarchicalRiskParity`.
12. **One struct per numbered variant**, or **an integer `variant` field** as the prototype has.
    Rejected: three copies of one update, or a run-time branch that names nothing.
13. **An abstract-type layer per survey family.** Rejected: five types nothing reads, and the
    straddlers.

## Consequences

- ADR 0155 is rewritten in the two sentences named above; `CONTEXT.md`'s *Online Portfolio
  Selection*, *Online Update* and *Recursion Read-out* entries lose "solves no programme" and
  "no column buffer" in the same sense.
- `CONTEXT.md` §4.1 gains the roster — name, acronym, family, set — and the terms *Expert
  Mixture*, *Sample Selector* and *Online Selection Rule*.
- The state ticket must admit a rule carrier that is a window or a prefix of rows; ADR 0158
  spells `OLMAR-2` and `TCO-2` and merges the two reversion structs into `ForecastReversion`; a rule whose update is
  already a programme takes the head's Allocation Set as its programme's feasible region, and
  projects only its damped mix
  ([ADR 0164](0164-a-follow-the-leader-rule-solves-its-programme-on-the-allocation-set-and-projects-only-its-damped-mix.md)).
- Peak price tracking's update is unverified against its paper (ledger §7), so a task ticket asks
  the maintainer for the paper before set 2 builds that row.
- The map's *Not yet specified* loses "the second algorithm set" and gains the three build sets
  by name.
