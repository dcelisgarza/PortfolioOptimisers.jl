---
status: proposed
---

# A first-order online rule is one mirror-descent step over any geometry, and the evaluation surface reaches dynamic regret

## Context

ADRs 0155 to 0164 specified the online portfolio selection family against a ledger that stops at
the 2014 survey (Li and Hoi 2014) plus five later papers. A second ledger, written for
[issue #1169](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1169), lists what the
field has published since — the price-level tracking school, the exponentiated-gradient
descendants and ensembles, and the online-convex-optimisation reading of the whole family — and
maps each row onto the roster. [Issue #1170](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1170)
asks what of it ships, under the map's standard: parity plus an improvement, never parity alone
where an improvement is one keyword away.

- **The first-order rules are one update.** Zinkevich's (2003) online gradient descent,
  Helmbold, Schapire, Singer and Warmuth's (1998) exponentiated gradient, and the Tsallis-entropy
  and log-barrier steps of the bandit and portfolio literature (Abernethy, Lee and Tewari 2015;
  Zimmert and Seldin 2021; Orseau, Lattimore and Legg 2017 §7) are all
  `w_{t+1} = argmin_w η⟨g_t, w⟩ + D_Ψ(w, w_t)` under a mirror map `Ψ`, and every one of those maps
  is a scalar root on the default Allocation Set. The library already names the map as the
  `proj` slot of [ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md);
  `ExponentiatedGradient` (set 1, built) and `GradientProjection` (set 2) differ only in it. The
  ledger's finding 1: `GradientProjection` under `EuclideanProjection` on the default set *is*
  Zinkevich's fixed-rate step exactly, and more than the 1997 paper's GP, whose non-negativity is
  assumed rather than enforced. Finding 2: `ExpectationMaximisation` with `η ∈ (0, 1)` *is*
  Soft-Bayes (Orseau, Lattimore and Legg 2017), formula for formula, and inherits its
  `O(√(T N log N))` bound with **no lower bound on the price relatives**.
- **What the reading adds is state and slots, not steps.** The uniform-mix variant EG(α, η)
  (Helmbold and co-authors 1998, Theorem 4.2) reads mixed relatives in the update and plays a
  mixed portfolio, so the carried iterate is not the played one; the anytime rates read the
  period count, the self-confident rate reads the run, the doubling trick resets the allocation;
  the optimistic step (Rakhlin and Sridharan 2013) carries a secondary iterate beside the played
  one; AdaGrad (Duchi, Hazan and Singer 2011) carries a per-asset gradient mass that both the
  step and the projection's norm read; the dynamic-regret mixtures (Zhang, Lu and Zhou 2018;
  Zhao, Zhang, Zhang and Zhou 2020) hand every expert one gradient at the blend.
- **One set-2 leaf is a defect as transcribed.** The survey's switching-portfolio update is
  exact only when its `w_t` is the price-adjusted holding (Singer 1997, Eq. 4–6); applied to the
  held `w_t` the leaf never reads `x_t` and glides deterministically to uniform.
- **Regret against a family member is not the literature's comparator.** `FollowTheLeader(;
  sel = Prefix())` through the head plays the leader of rows `1:t−1` on `x_t`; be-the-leader
  plays the leader of rows `1:t` on `x_t` and beats the best constant rebalanced portfolio by the
  FTL–BTL lemma (Kalai and Vempala 2005). Every dynamic notion is a comparator *sequence* and a
  path length `P_T = Σ_t ‖u_t − u_{t−1}‖`; the verb of
  [ADR 0161](0161-log-wealth-regret-is-a-verb-over-two-prediction-results-and-a-hindsight-comparator-is-an-estimator-fit-on-the-rows-it-is-scored-on.md)
  accepts any `b`, and no walk-forward fits on the row it scores.
- **The survey's table has three columns the library lacks**: mean excess return and information
  ratio against a benchmark series, and average turnover of the held path. `calc_turnover`, the
  held path on `PredictionResult` and the tracking measures exist; no summary reads them.
- **The risk-measure library is a loss.** The online-convex-optimisation reading runs the same
  step on any convex loss of the portfolio; a risk measure over a window of rows is one, and its
  gradient exists by finite differences already (`risk_contribution(…; marginal = true)`).

## Decision

### One mirror-descent rule; the paper names are constructors

`MirrorDescent(; eta, proj, alpha = 0, obj = LogWealth())` is the family's first-order rule. Its
update is the mirror step `argmin_w η_t⟨g_t, w⟩ + D_Ψ(w, w_t)` in the geometry `proj` holds, taken
through `project(proj, set, ·, ŵ_t)` as every rule's is. `proj` is bound to every Projection
Geometry that is a scalar root on the default set: `EuclideanProjection`, `EntropicProjection`,
and the new `TsallisProjection(; alpha)` (the power potential; the root is in the budget
multiplier, monotone) and `LogBarrierProjection` (the Burg entropy; `w_i = 1/(1/w_{t,i} + η g_i + λ)`,
the one geometry with a portfolio theorem that needs no lower bound on `x` in mirror-descent form).
On a `ProgrammeAllocationSet` a Tsallis or log-barrier step is a Bregman projection solved by the
set's solver, a power cone and an exponential cone. A barrier potential's unconstrained mirror
step exists only while every base is positive — under the log barrier, while
`η ŵ_{t,i} < 1` for every asset, which a rate below one guarantees — and a step outside that
domain is refused by name, not clipped: the raw step a geometry projects is always a point. `ExponentiatedGradient(; eta)` and `GradientProjection(; eta)` are **constructors**
filling `proj` with the entropic and the Euclidean map, as `UniversalPortfolio` constructs the
mixture; the struct of the first build becomes a constructor in place, and the roster names keep
resolving.

`alpha` is the uniform mix of Helmbold and co-authors (1998): the update reads
`x̃_t = (1 − α/N) x_t + (α/N) 1` and the played allocation is `w̃_t = (1 − α) w_t + (α/N) 1`. The
Rule State carries the **unmixed** iterate in a one-vector carrier, and the verb returns the played
one; at `α = 0` the carrier is the played vector and the rule is the shipped exponentiated
gradient to the last bit. The mix is a convex shift, not a projection, so it is admitted under every
map; the theorem is the entropic one's, and the docstring says so.

### Learning rates are numbers or schedules, and a schedule may read the run and restart it

`eta` on `MirrorDescent` and on `ExpectationMaximisation` is bound to
`Union{Real, <:AbstractLearningRateSchedule}`, read as `learning_rate(sched, t, st)` with `t` the
period count the head's Sample Buffer already carries (`rows.n`; the mixture's own count for a
scheduled weighting) and `st` the rule's carrier, so a schedule may read a running statistic;
`restart(sched, t)` names a stage boundary, at which the rule puts its allocation back at the head's
Start Allocation (uniform by default, as the doubling trick's reset is) and its carrier at the seed
with the period count kept, because the stages are cumulative and the count is what the next stage
is found from. The schedule that sets the share answers it through `mixing_share(sched, t, alpha)`,
which every other schedule and a number pass through.
A schedule's running statistic lives in the rule's carrier, seeded by the schedule, because the
carrier is the rule's (ADR 0157). Three schedules ship: `InverseSquareRootRate(; c)`, `c/√t`, the
anytime rule of Zinkevich (2003, Theorem 1) and Hazan (2016, Theorem 3.1) and, at the paper's
`c`, the anytime Soft-Bayes rate of Orseau, Lattimore and Legg (2017, Theorem 10); the
**doubling trick** of Helmbold and co-authors (1998, Corollary 4.3), stages of `2^i N² log N`
periods with `α` and `η` set from the stage length and a restart at each boundary; and the
**self-confident rate** of Orseau, Lattimore and Legg (2017, Theorems 5 and 6), which reads the
running `C₁ = Σ_t max_i (x_{t,i}/⟨w_t, x_t⟩ − 1)` from the carrier. The fixed-horizon constants of
the theorems need `T`, which no online rule knows, and are documented formulas.

`ExpectationMaximisation`'s update is written in the online form of Orseau, Lattimore and Legg
(2017, Eq. 14): `w′ = (EM step at η_t)·η_{t+1}/η_t + (1 − η_{t+1}/η_t)·w₁`. At a constant `η`
the ratio is one and the pull vanishes, so the shipped step is unchanged; under a decreasing
schedule the fixed-share pull toward the Start Allocation is what makes the bound telescope, which
the plain step with a varying rate does not. The rule cites both papers and carries the
`O(√(T N log N))` guarantee as its own.

The momentum variants of the exponentiated gradient (Li, Zheng, Chen, Wang and Xu 2022) are a
`grad` slot on `MirrorDescent`, bound to `AbstractGradientTransform`: the identity by default, an
exponential moving average of the gradient, a root-mean-square rescaling, or both without bias
correction, each keeping its averages on the Rule State and stacking with any geometry and any
schedule; `EGE`, `EGR` and `EGA` are constructors of the entropic rule under each. At the paper's
own `γ₂ = 0` the root-mean-square rescaling is the sign of the gradient, the same at every asset,
so the rule holds its start; the docstring says so.

`MirrorDescent`, the three schedules and the transforms live in a file of their own after the
family's rules, forecast arm and second set, because the first-set file would cross the size
ceiling with them; the schedule supertype and its five verbs — `learning_rate`, `restart`,
`schedule_state_seed`, `schedule_update!`, `mixing_share` — live in the family's base file,
because two rules in two files bind their `eta` slot to it and a field bound must load first; the
geometries live in the Constrained Update seam's file, their scalar roots beside their
programmes.

### AdaGrad is a leaf rule with a diagonal carrier and a diagonal geometry

`AdaptiveSubgradient(; eta = 1/√2, delta = 0, proj::DiagonalProjection)` carries the per-asset
gradient mass `s_t`, `s_t² = Σ_{s≤t} g_s²`, and steps `w_t + η x_t ./ (⟨w_t, x_t⟩ (δ + s_t))`
followed by the projection onto the set **in the norm `H_t = δI + diag(s_t)`**. The new geometry
`DiagonalProjection` takes the weight vector from the carrier: on the default set the projection is
the weighted scalar root `w_i = max(0, y_i − θ/H_{t,ii})` with `θ` the budget root, which costs
what the Euclidean root costs; on a programme set it is a weighted quadratic programme.
`EuclideanProjection` is the `H = I` case, and the two share one root with a weight argument. The
rule mirrors `NewtonStep` and `GramProjection` one rank down: the full-matrix variant of the paper
is the Newton step's carrier under a square root. On a uniform start the first step is a no-op,
because the per-coordinate normalisation cancels the first gradient exactly; the docstring says so.

### The optimistic step is a wrapper rule carrying the secondary iterate

`OptimisticStep(; alg::MirrorDescent, predictor)` plays Rakhlin and Sridharan's (2013) two
half-steps: `v_{t+1}` = the wrapped rule's step from the secondary point `v_t` on `g_t`, then
`w_{t+1}` = the same step from `v_{t+1}` on the hint `M_{t+1}`. The carrier holds `v_t`; the
played `w_t` is not what the next step reads, so
[ADR 0160](0160-an-online-update-starts-from-its-own-allocation-and-its-trade-is-measured-from-the-price-adjusted-one.md)'s
"own `w_t`" is the rule's own `v_t`, the split `alpha` already introduced. `predictor` is bound to
`AbstractGradientPredictor`: `LastGradient()` (the default; the regret sum becomes the gradient
path length), `MeanGradient()` (the variance hint), and `ForecastGradient(; me)`, which maps an
expected-returns estimator's Price Relative Forecast to `M = −x̂/⟨w, x̂⟩` — the natural portfolio
hint, not in the paper. The extra-gradient step (Chiang and co-authors 2012, the expert of Zhao and
co-authors 2020) is `LastGradient()` taken at the played point, a predictor kind and not a rule.
`eta`, `proj`, `alpha` and `obj` are the wrapped rule's; the regret is
`O(√Σ_t ‖g_t − M_t‖²)` and never worse than the plain step up to a constant.

### The mixture takes a gradient point and a start over experts

`ExpertMixture` gains `grad::Union{OwnPoint, BlendPoint}`, `OwnPoint()` by default — every expert
reads its gradient at its own iterate, today's behaviour — and `BlendPoint()`, under which every
first-order expert reads its gradient at the mixture's played blend while stepping from its own
iterate, the shared gradient of Zhang, Lu and Zhou (2018, Algorithm 4) and Zhao and co-authors
(2020, Algorithm 2) on which their dynamic bounds are stated. It gains `p0`, the mixture's Start
Allocation over its experts, uniform by default and projected once onto the Expert Set in the
weighting's geometry, as `w0` is onto the head's set
([ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md)).
`Ader(; …)` and `Sword(; …)` are constructors: a mixture over `MirrorDescent` experts (and, for
Sword, an `OptimisticStep` expert) at the geometric rate grid `η_i = 2^{i−1} η_min`, under
`ExponentiatedGradient` as the weighting, `BlendPoint()`, and Ader's `p0 ∝ 1/(i(i+1))`. The
second meta layer of `Sword_best`, which learns the hint in parallel, is fog.

### A first-order rule runs on log wealth or on a risk loss over its rows

`obj` on `MirrorDescent` is bound to `Union{LogWealth, <:RiskLoss}`. `LogWealth()` is the default
and every rule above unchanged: `g_t = −x_t/⟨w_t, x_t⟩`, one row. `RiskLoss(; r, window, sc)` is
the loss `ρ_r(w; X)` over the last `window` rows of the head's buffer, with `r` one risk measure or
a vector under the scalariser `sc` — so `MeanReturn` beside `Variance` at a scale is one step on the
mean–variance utility, and `ConditionalValueatRisk` or a drawdown measure is one step toward its
minimiser, re-estimated each period. `rows_needed` is the window. The gradient is a verb,
`risk_gradient(r, w, X)`, whose fallback is the library's finite-difference kernel and which has an
exact method where the library states a closed form — `Variance`, `StandardDeviation`,
`MeanReturn` — with more added as they are wanted, never a differentiation package. `OptimisticStep`
inherits the slot from its wrapped rule; `AdaptiveSubgradient` carries the same slot, because its
step is the same gradient. The docstring states which measures are convex, because the theorems
hold for those only, and that a finite-difference gradient at a kink is a chord.

### The post-survey rows: the tracking step, a scale, a weighting, a covariance, a selector

- **`ForecastTracking(; me, eps)`** is the mechanism struct of the peak-tracking school's step,
  `w_{t+1} = Proj_Δ(w_t + ε x̂_⊥/‖x̂_⊥‖)` with `x̂_⊥` the centred Price Relative Forecast (hold when
  it is zero), the mirror of `ForecastReversion`; `PeakPriceTracking(; window = 5, eps = 100)`
  (Lai, Dai, Ren and Huang 2018) is its constructor filling `me = WindowPeak(window)`. ADR 0156's
  set-2 row is rewritten in place. AICTR and TPPT are constructors filling their composite
  statistics, `CompositeTrend` and a `TrendSwitch` on the pairwise slope sum, now that their
  papers are read (#1185); the Gaussian weighting reversion and the local adaptive learning are
  constructors of `ForecastReversion` over a Gaussian-weighted double estimate and a
  `TrendSwitch` on a regression slope.
- **`ForecastReversion` gains `scale::Option{<:AbstractPriceLevelStatistic}`**, `nothing` by
  default (every existing row): a diagonal preconditioner `D = diag(x̂_scale)` on the reversion
  direction, `w_{t+1} = Proj_Δ(w_t + λ D (x̂ − x̄1))` with `λ` the passive-aggressive multiplier
  unchanged. `ReweightedPriceRelativeTracking(; window, theta, eps)` (Lai, Yang, Fang and Wu 2018,
  as restated with equations by Li, Luo and Xu 2023) is a constructor filling `me` with the new
  statistic `ReweightedPriceRelative(; theta)` — the per-asset recursion
  `φ̂_{t+1} = γ_{t+1} + (1 − γ_{t+1}) φ̂_t ./ x_t`, `γ_{t+1,i} = θ x_{t,i}/(θ x_{t,i} + φ̂_{t,i})` —
  and `scale` with the moving average of the same window.
- **`ExponentialMovingAverageReversion(; alpha = 0.5, eps = 10)`** is a constructor of
  `ForecastReversion` filling `me` with the exponential moving average of price levels (Li, Hoi,
  Sahoo and Liu 2015, Eq. 2). The paper states no `α`; `0.5` is the library's mid-plateau choice
  and the docstring says so.
- **`SwitchingWeighting(; gamma = 1/3)`** is a mixture weighting: the wealth step, then a fixed
  share, keep with probability `1 − γ` and redistribute uniformly to the other `K − 1` experts
  with probability `γ` (Singer 1997, Eq. 4–6; the update resembles Herbster and Warmuth's fixed
  share, as the paper notes). **`SwitchingPortfolio(; N, gamma = 1/3)`** is a constructor of
  `ExpertMixture` over the `N` single-asset `ConstantRebalancedPortfolio`s under it. The set-2
  leaf of ADR 0156 is withdrawn: applied to the held `w_t` it is the affine glide
  `w_{t+1} − 1/N = (1 − γN/(N−1))(w_t − 1/N)` and reads no price. The Krichevsky–Trofimov
  varying-`γ` version, for which the paper states there is no equivalent portfolio update, is
  fog.
- **`RankOneCovariance`** (Lai, Tan, Wu and Fang 2020, Algorithm 1) is a covariance estimator
  on a window: the principal right singular vector of the uncentred window scaled by the centred
  Gram's spectrum. **`ShortTermLossControlPortfolio(; window = 5, gamma = 0.025, slv)`** is a
  constructor of `FollowTheLeader(; sel = LastRows(window), opt = MeanRisk(…))` whose programme
  maximises the worst increasing factor of the window minus `γ` times the rank-one variance —
  `WorstRealisation` beside `Variance` under the rank-one estimator, scalarised.
- **RACORN-K** (Wang, Wang, Wang and Zhang 2018) is `ExpertMixture` over `FollowTheLeader(;
  sel = CorrelationMatch, opt = MeanRisk(…))` with a standard-deviation penalty on the log return,
  under `TopK`; **CW-OGD** (Zhang, Lin, Yang and Long 2021) with the log loss is `MirrorDescent`
  on the Euclidean map over the assets directly. Both are configurations: a docs recipe and a
  test each.
- **`ClusterMatch(; window, clusterer)`** is a Sample Selector: the past windows whose cluster is
  the latest window's, under one of the library's clustering estimators (Khedmati and Azin 2020);
  the paper's cost term is the `fees` slot on `opt`.

### What waits, and what is out

A rule enters a build from its paper, as the peak-tracking task set. The rows whose full text is
closed and whose abstract does not fix the update — AICTR, TPPT, KTPT, GWR, LOAD, PIRA, EGM,
MAEG/AEG, and the weak-aggregating-algorithm weighting with the two papers that run it — wait on
one task ticket: the maintainer supplies the PDF, the ticket records the update, and a build
graduates. Three rows are **out of the map's scope by input**: a rule reading trading volume, a
market-index series or a stock network needs a side panel the head does not carry, a head change
and not a rule. The barrier-regularised Newton step and its adaptive form are fog: a Gram-plus-barrier
geometry that needs a solver on every set, and a restart keyed on a prefix programme whose own
paper calls a faster design likely.

### Dynamic regret is a hindsight splitter and a path length

`HindsightSplit(; prefix = true)` is a cross-validator whose fold `t` trains on rows `1:t`
(`prefix = true`) or on row `t` alone (`prefix = false`) and tests on row `t`, `test_size = 1`. It
is the Hindsight Comparator rule of ADR 0161 per row: `cross_val_predict(est, rd, HindsightSplit())`
with `est = MeanRisk` under `LogarithmicReturn` is **be-the-leader**; with the top-1 Asset Selector
under `MeanReturn(; flag = true)` and `EqualWeighted` and `prefix = false` it is the **per-period
minimiser**, one-hot on the day's best asset, wealth `Π_t max_i x_{t,i}`. A `K`-switch comparator
is any piecewise-constant `b` the caller builds. The splitter's docstring states, first, that its
training set contains its test row by design. `LogWealthRegretResult` gains `path_length`,
`Σ_t ‖u_t − u_{t−1}‖₂` read off `b`'s weight path and `NaN` when `b` carries none, and
`cumulative`, `cumsum(difference)`, the running regret. The verb's signature does not move.

### The survey's metrics are a benchmark keyword and four fields

Every `performance_summary` method takes `benchmark::Option{<:VecNum} = nothing`, and
`PerformanceSummaryResult` gains `excess_ret` (the annualised mean of `ret − benchmark`),
`tracking_error` (its annualised standard deviation), `information_ratio` (their ratio) and
`turnover` (the average per-period turnover of the held path, read off `hw` by the two
prediction-result methods, from the drifted holding under `DriftedWeights` as ADR 0160 measures a
trade); each is `NaN` when its input is absent. The t-test of the survey is `log_wealth_regret`'s
`z` and `p`, with a Newey–West variance the survey's plain t-test lacks.

### Three items owe nothing

No initial wealth: every cumulative path starts at unit wealth, as the survey's tables do, and a
start wealth is a scalar the caller multiplies. Fees: the head's `fees` carries the library's fixed,
proportional and turnover fees, `predict` charges them, and the 2015 reversion paper's cost model —
`γ/2` per side on `|b_t − b̂_{t−1}|` with `b̂` the price-adjusted holding — is the turnover fee under
ADR 0160's base. Parity: every build ticket carries one fixture per set at `test_size = 1`, every
rule's causal path checked against the paper's own numbers where a paper gives any and against the
prototype's corrections where it does not, the numbers committed as literals with their provenance
stated; the verification ticket sweeps them and adds the cross-set identities — `MirrorDescent` on
the entropic map at `α = 0` equals the first build's exponentiated gradient to the last bit,
`ExpectationMaximisation` at a constant `η` equals its Eq. 14 form, `AdaptiveSubgradient` on a
uniform start holds for one period.

## Considered options

1. **Keep `ExponentiatedGradient` as a struct and widen `GradientProjection`'s bound** to the
   non-entropic maps. Rejected: two structs with one update, and `alpha`, the schedule slot and
   `obj` written twice or on one of them.
2. **A new `FirstOrderStep` beside both.** Rejected: a third copy of the step.
3. **`alpha` with the played vector in the state and the update inverting the mix.** Rejected: a
   division per step, and it fails by construction when a bound clips the played vector below
   `α/N`.
4. **`alpha` refused, Soft-Bayes named as the universal member.** Rejected: a parity gap by name.
5. **Schedules that read the period count only, the doubling and self-confident rows as fog.**
   The recommendation; the maintainer chose complete parity, so the schedule reads the carrier
   and may restart the rule.
6. **AdaGrad as a data-driven schedule on `MirrorDescent` under the Euclidean map.** Rejected:
   the projection's norm is the paper's, and a schedule that owns state breaks the carrier
   contract.
7. **A `predictor` slot on `MirrorDescent`, `nothing` by default.** Rejected: the update and the
   carrier's shape branch on the slot's type.
8. **Two optimistic leaf rules, one per map.** Rejected by the first ruling.
9. **`p0` only, the shared gradient as fog.** Rejected: Ader and Sword are not reachable by
   name.
10. **The family on log wealth only.** Rejected: the standing rule that all functionality ships,
    and the item named as the improvement past the ledger.
11. **Exact gradients only, through a differentiation package.** Rejected: a dependency for
    per-type work, and the finite-difference kernel exists.
12. **`PeakPriceTracking` stays the struct, the school's descendants as constructors of it.**
    Rejected: a constructor named for one paper on a struct named for another, the case the
    naming rule of ADR 0156 exists to avoid.
13. **The abstract-only rows built from their abstracts.** Rejected: a rule entered from a
    paraphrase cannot be verified against anything.
14. **The side-panel rows as fog.** Rejected: a side panel is a head change past this map's
    destination; a fresh effort if wanted.
15. **A `comparator` argument on `log_wealth_regret`** that fits internally. Rejected: the verb
    is over two results (ADR 0161); a splitter is the seam the library already has.
16. **A separate `relative_performance_summary`.** Rejected: two verbs for one table.
17. **The survey's metrics as a docs recipe.** Half true — the excess series' Sharpe is the
    information ratio — but turnover stays a hand computation.
18. **An initial-wealth keyword on `predict`.** Rejected: a released verb gains an argument for a
    scalar multiply.
19. **Fixtures on the verification ticket only.** Rejected: a build would land without its
    set's fixture.

## Consequences

- ADR 0156 is rewritten in place: set 1's `ExponentiatedGradient` becomes the constructor of
  `MirrorDescent`; set 2 loses `PeakPriceTracking`, `SwitchingPortfolio`, `GradientProjection` and
  `ExpectationMaximisation` as structs and lists `ForecastTracking`, the constructors and the
  weighting; a fourth set is named. ADR 0158's build line gains `ForecastTracking`, the `scale`
  slot and the two constructors. ADR 0159's first-set bound sentence names `MirrorDescent` and
  the two new geometries. ADR 0161's Result gains `path_length` and `cumulative`, and its
  comparator section names the splitter. ADR 0163's constructor line becomes
  `ExpertMixture(; experts, alg, eset, proj, grad, p0)`.
- `CONTEXT.md` gains *Mirror Descent*, *Learning-Rate Schedule*, *Gradient Predictor*, *Gradient
  Point*, *Risk Loss*, *Hindsight Split*; *Projection Geometry* gains the three maps; *Expert
  Mixture* the two slots; *Online Selection Rule* the fourth set; the roster line moves the names.
- Five build tickets graduate: `MirrorDescent` with its geometries and schedules, blocked by the
  constrained-update build; `AdaptiveSubgradient` and `OptimisticStep`; the risk loss; the
  mixture's slots with `Ader`, `Sword` and the switching weighting; the evaluation surface. One
  task ticket asks the maintainer for the closed papers. The forecast-arm, second-set and third-set
  builds each gain a scope comment; the docs build waits on all five.
- No reference implementation is named here, on the tickets, or in any docstring the builds write;
  papers are cited by author and year.
- Issues [#1170](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1170) and
  [#1169](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1169).
