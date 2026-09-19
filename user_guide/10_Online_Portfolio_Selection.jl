#=
```@meta
Description = "Online portfolio selection in PortfolioOptimisers.jl: solver-free rules that update from each price relative, a two-regime diagnostic and the regret table."
```

# Online portfolio selection

Every optimiser met so far fits a moment and solves a programme. The online portfolio
selection family does neither. A rule of the family is one recursion, `w_{t+1} = f(w_t, x_t)`:
it reads the price relative `x_t = 1 + r_t` the market just realised and moves the allocation
from it, with no covariance, no expected return and no solver. The head that runs a rule is
[`OnlinePortfolioSelection`](@ref), a naive optimiser whose batch verb is the **Causal Pass** —
every row in order, from the start allocation, answering the portfolio for the period *after*
the last row — and whose online verbs are the same recursion one row at a time, so a
walk-forward at `test_size = 1` and the batch pass hold the same path.

A rule carries a guarantee that holds on *every* price sequence, with no statistical
assumption: its wealth cannot fall far behind the best constant portfolio chosen in hindsight.
That guarantee is the family's whole promise, and it is a bound, not a return. This page runs
the two halves of the family on two synthetic markets and shows what each half bets on, then
measures every rule against three hindsight comparators, shows the one place a walk-forward
and the batch pass part, and tunes a rate.

The family is Li and Hoi's (2014) survey, in five groups; the rules this page runs are the
groups' first members, and the full roster by group is at the end of the page.
=#

using PortfolioOptimisers, StableRNGs, DataFrames, PrettyTables, Statistics, Dates

resfmt = (v, i, j) -> isa(v, AbstractFloat) ? "$(round(v; digits = 3))" : v;

#=
## 1. Two markets, side by side

Both fixtures are five assets over five hundred periods from one seed. In the
**mean-reverting** market each log price is pulled back to a level, so a rise is followed by
a fall; in the **trending** market each asset's log *return* is persistent, so a rise is
followed by another rise. Nothing else differs.
=#

function synthetic_market(kind; T = 500, N = 5, seed = 7)
    rng = StableRNG(seed)
    P = ones(T + 1, N)
    lp = zeros(N)
    for t in 1:T
        if kind == :trending
            lp .= 0.90 .* lp .+ 0.008 .* randn(rng, N)   # a persistent log return
            P[t + 1, :] .= P[t, :] .* exp.(lp)
        else
            lp .= 0.90 .* lp .+ 0.04 .* randn(rng, N)    # a log price pulled to zero
            P[t + 1, :] .= exp.(lp)
        end
    end
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2016, 1, 4):Day(1):(Date(2016, 1, 4) + Day(2 * T + 10)))[1:T]
    return ReturnsResult(; nx = ["A$i" for i in 1:N],
                         X = P[2:end, :] ./ P[1:(end - 1), :] .- 1, ts = days)
end

rd_rev = synthetic_market(:reverting)
rd_trend = synthetic_market(:trending)
N = size(rd_rev.X, 2)

#=
The roster below is two benchmark rules, three rules that follow the winner and three that
follow the loser. Every rule runs through the same head and the same walk-forward: twenty
warm-up rows, then one row per fold with the Fold Fit [`OnlineStep`](@ref), so each fold is
one update and one read-out. [`UniversalPortfolio`](@ref) samples its constant portfolios at
construction and needs the universe size, and a seed pins the draw.
=#

rules = ["Buy and hold" => BuyAndHold(),
         "Constant rebalanced" => ConstantRebalancedPortfolio(),
         "Exponentiated gradient" => ExponentiatedGradient(), "Newton step" => NewtonStep(),
         "Universal portfolio" => UniversalPortfolio(; N = N, seed = 1),
         "Passive-aggressive mean reversion" => PassiveAggressiveMeanReversion(),
         "Moving-average reversion" => MovingAverageReversion(),
         "Robust median reversion" => RobustMedianReversion()]

cv = IndexWalkForward(20, 1; ff = OnlineStep())
online(alg, rd) = cross_val_predict(OnlinePortfolioSelection(; alg = alg), rd, cv)
wealth(pred) = prod(1 .+ pred.mrd.X)

preds_rev = Dict(name => online(alg, rd_rev) for (name, alg) in rules)
preds_trend = Dict(name => online(alg, rd_trend) for (name, alg) in rules)

pretty_table(DataFrame("Rule" => first.(rules),
                       "Reverting market" => [wealth(preds_rev[n]) for n in first.(rules)],
                       "Trending market" => [wealth(preds_trend[n]) for n in first.(rules)]);
             formatters = [resfmt],
             title = "Terminal wealth from a unit start, 480 periods")

#=
The two halves of the table are two different objects. The benchmarks and the
follow-the-winner rules land within a factor of each other on both markets — the constant
rebalanced portfolio, the exponentiated gradient, the Newton step and the universal portfolio
sit together near `1.5` on the reverting market and near `2.9` on the trending one — because
each of them moves slowly from a constant portfolio, and a constant portfolio is what their
guarantee is stated against. The follow-the-loser rules are not a strategy in that sense:
**the reversion family is a total bet on one market property**. Where prices revert, the rule
buys every fall and is paid for it on the next row, and the moving-average reversion turns one
unit into twenty-two; where prices trend, the same rule buys every fall into a further fall,
and every reversion rule compounds to a thousandth of its start. Nothing in the rule detects
which market it is in, and the twenty-two and the thousandth are one mechanism. Read the table
before running a reversion rule on money: run a winner rule and a loser rule side by side on
the market at hand, and read the pair as a cheap, estimation-free test of which regime the
market is in.

## 2. The regret table

The family's guarantee is stated against a comparator chosen in hindsight, so the honest
measurement is the **Hindsight Comparator** rule: fit the comparator on the very rows the
strategy was scored on, and predict it in sample over them. Three comparators cover the
literature. The best constant rebalanced portfolio is [`BestConstantRebalancedPortfolio`](@ref),
Cover's (1984) fixed point with no solver — the same portfolio [`MeanRisk`](@ref) reaches under
a [`LogarithmicReturn`](@ref) and a maximum-return objective. The best stock is a top-one
[`ScoreSelector`](@ref) on log wealth followed by [`EqualWeighted`](@ref), through a
[`Pipeline`](@ref). The uniform constant rebalanced portfolio is a causal member of the family,
run through the same walk-forward.
=#

cut(rd, i) = ReturnsResult(; nx = rd.nx, X = rd.X[i, :], ts = rd.ts[i])
test_rows(rd) = cut(rd, 21:size(rd.X, 1))       # the rows every fold was scored on

best_stock = Pipeline(;
                      steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                             rule = RankRule(; best = 1)), EqualWeighted()))
function comparators(rd, preds)
    rdt = test_rows(rd)
    return ["Best CRP" => predict(optimise(BestConstantRebalancedPortfolio(), rdt), rdt),
            "Best stock" => predict(fit(best_stock, rdt), rdt),
            "Uniform CRP" => preds["Constant rebalanced"]]
end

function regret_table(rd, preds)
    df = DataFrame("Rule" => first.(rules))
    for (name, comp) in comparators(rd, preds)
        df[!, name] = [log_wealth_regret(preds[n], comp).regret for n in first.(rules)]
    end
    return df
end

pretty_table(regret_table(rd_rev, preds_rev); formatters = [resfmt],
             title = "Log-wealth regret on the reverting market, comparator minus rule")

#-

pretty_table(regret_table(rd_trend, preds_trend); formatters = [resfmt],
             title = "Log-wealth regret on the trending market, comparator minus rule")

#=
[`log_wealth_regret`](@ref) answers the comparator's log terminal wealth less the strategy's,
so a positive entry is a comparator that won. **Negative regret is expected and is not a
defect**: the bound says a rule cannot fall far *behind* the best constant portfolio, and says
nothing about pulling ahead of it. A causal rule that reads the market's structure does pull
ahead — every reversion rule on the reverting market sits between `−1.4` and `−2.6` against
the best constant portfolio — and the best stock in hindsight is a weaker comparator than the
best constant rebalanced portfolio, so the second column is never above the first; on the
trending market the two coincide, because the best constant portfolio there is one asset. The
Result also carries a per-row difference series and a Newey–West test of it, but against a
comparator that read the rows its `p` is optimistic by construction: the comparator was chosen
to win on exactly this sequence. Read it as a rank, not a probability.

The universal portfolio's row is the one the literature bounds. The [`UniversalPortfolio`](@ref)
docstring states both bounds with their attribution: Cover's (1991) `(N − 1) log(T + 1)` for
the exact integral, which is `24.7` at five assets and 480 periods, and `log n_experts` for the
sampled mixture the library runs, `7.6` at the default two thousand experts, against the best
sampled expert. The row's `0.132` and `0.694` sit far inside both, as every row of the
winner half does.

## 3. The batch pass and the held path

At `test_size = 1` the walk-forward and the Causal Pass hold the same path, and the fold's
target after rows `1:t` is `optimise(opt, rd[1:t])` to the bit. At `test_size = k` a fold is a
**Block Step**, `k` single-row updates and one read-out, and the walk-forward then holds that
one target for `k` periods while the pass would have rebalanced every row; under a Weight
Drift the book the fund holds also drifts with the prices over the block, and
[`DriftedWeights`](@ref) threads that drifted book, not the target, into the next fold's fee.
The identity of the target survives; the wealth does not.
=#

opt = OnlinePortfolioSelection(; alg = PassiveAggressiveMeanReversion())
cv5 = IndexWalkForward(20, 5; ff = OnlineStep())
cv5d = IndexWalkForward(20, 5; ff = OnlineStep(), wd = SelfFinancingDrift(),
                        pws = DriftedWeights())

every_row = preds_rev["Passive-aggressive mean reversion"]
block = cross_val_predict(opt, rd_rev, cv5)
drifted = cross_val_predict(opt, rd_rev, cv5d)

k = 46                                           # one fold, read three ways
t_end = last(split(cv5, rd_rev).train_idx[k])    # the last row the fold folded
(; target_identity = block.pred[k].res.w == optimise(opt, cut(rd_rev, 1:t_end)).w,
 held_book_gap = maximum(abs, drifted.pred[k].hw.w - drifted.pred[k].res.w),
 wealth = (; every_row = wealth(every_row), block = wealth(block),
           drifted = wealth(drifted)))

#=
The target the fold reads out is the pass's target for the same rows, and the book the fund
holds at the end of the block is not: it has drifted four points of weight from the target,
and the rule's next update reads its own target, never the drifted book. The gap in wealth is
the cadence — a reversion rule that rebalances every row is paid on every reversal and turns
one unit into `6.3`, one that rebalances every fifth row is paid on every fifth and reaches
`2.1`, and the drift takes a little more. The difference is documented here and is not a
defect of either path: a rule is designed for the cadence its paper ran it at, and
`test_size` is that cadence.

## 4. Tuning a rate

A rule's rate is a field, and [`GridSearchCrossValidation`](@ref) addresses it by path. The
score is [`MeanReturn`](@ref) with `flag = true`, the mean log return per period, which is log
terminal wealth per period; ranking candidates on it is ranking them on regret against any
one fixed comparator, because the comparator's wealth is the same constant for every
candidate.
=#

grid = GridSearchCrossValidation(["alg.eta" => [0.01, 0.05, 0.2, 0.5]]; cv = cv,
                                 r = MeanReturn(; flag = true))
tuned = search_cross_validation(OnlinePortfolioSelection(; alg = ExponentiatedGradient()),
                                grid, rd_trend)

(; eta = tuned.opt.alg.eta, wealth = wealth(cross_val_predict(tuned.opt, rd_trend, cv)))

#=
The tuned head is the head; run it through the walk-forward like any other. On the trending
market the largest rate offered wins, because the gradient keeps pointing the same way and a
faster step follows it sooner, and it lifts the wealth from `2.9` to `3.0`; on the reverting
market the same search would pick the smallest, for the same reason.

## The roster by group

Every rule is a value of the head's `alg` slot. The groups are Li and Hoi's (2014), and the
[capability catalogue](@ref catalogue-online-portfolio-selection) lists the same roster.

  - **Benchmarks** — [`BuyAndHold`](@ref), [`ConstantRebalancedPortfolio`](@ref), and, in
    hindsight, [`BestConstantRebalancedPortfolio`](@ref).
  - **Follow the winner** — [`MirrorDescent`](@ref) and its constructors
    [`ExponentiatedGradient`](@ref), [`GradientProjection`](@ref), [`EGE`](@ref), [`EGR`](@ref),
    [`EGA`](@ref), [`MAEG`](@ref), [`AEG`](@ref); [`NewtonStep`](@ref);
    [`AdaptiveSubgradient`](@ref); [`OptimisticStep`](@ref); [`ExpectationMaximisation`](@ref);
    [`UniversalPortfolio`](@ref); [`AggregatingExponentialGradient`](@ref). A rate may be a
    schedule: [`InverseSquareRootRate`](@ref), [`DoublingTrickRate`](@ref),
    [`SelfConfidentRate`](@ref), [`HintResidualRate`](@ref), [`WindowedBestRate`](@ref).
  - **Follow the loser** — [`PassiveAggressiveMeanReversion`](@ref);
    [`ForecastReversion`](@ref) and its constructors [`MovingAverageReversion`](@ref),
    [`ExponentialMovingAverageReversion`](@ref), [`RobustMedianReversion`](@ref),
    [`ReweightedPriceRelativeTracking`](@ref), [`GaussianWeightingReversion`](@ref),
    [`LocalAdaptiveLearning`](@ref); [`ForecastTracking`](@ref) with
    [`PeakPriceTracking`](@ref), [`AdaptiveInputCompositeTrend`](@ref),
    [`TrendPromotePriceTracking`](@ref); [`KernelTrendTracking`](@ref);
    [`TransactionCostOptimisation`](@ref); [`ShortTermSparsePortfolio`](@ref);
    [`ConfidenceWeightedMeanReversion`](@ref); [`AntiCorrelation`](@ref).
  - **Pattern matching** — [`FollowTheLeader`](@ref) over a sample selector:
    [`Prefix`](@ref), [`LastRows`](@ref), [`HistogramMatch`](@ref), [`KernelMatch`](@ref),
    [`NearestNeighbourMatch`](@ref), [`CorrelationMatch`](@ref), [`ClusterMatch`](@ref); and
    the solved rules [`ShortTermLossControlPortfolio`](@ref),
    [`LowDimensionEnsemblePortfolio`](@ref).
  - **Meta-learning** — [`ExpertMixture`](@ref) over any experts, with [`Ader`](@ref),
    [`Sword`](@ref), [`SwitchingPortfolio`](@ref) and the weightings
    [`AggregatingAlgorithm`](@ref), [`TopK`](@ref), [`WeakAggregatingAlgorithm`](@ref),
    [`SwitchingWeighting`](@ref); [`FollowTheLeadingHistory`](@ref).

A forecast-reading rule takes its Price Relative Forecast from an expected-returns estimator:
[`PriceLevelExpectedReturns`](@ref) over a price-level statistic, or any prior's expected
return through [`PriorExpectedReturns`](@ref). Every rule's step is projected onto the head's
Allocation Set in the rule's own geometry: the default [`BoundedAllocationSet`](@ref) is the
simplex and closed form, and a [`ProgrammeAllocationSet`](@ref) admits the full constraint
vocabulary through a solver. The evaluation surface is [`log_wealth_regret`](@ref), the
[`HindsightSplit`](@ref) that makes any estimator a per-row Hindsight Comparator, and
[`performance_summary`](@ref) with a benchmark.

## Where to go next

  - [The online portfolio selection example](../examples/3_optimisers/18_Online_Portfolio_Selection.md)
    — the roster on real prices in a walk-forward with a turnover fee, the search, the regret
    table, and the weight path and the discrete allocation of a rule.
  - [The online walk-forward](09_Online_Walk_Forward.md) — the Fold Fit every run on this page
    declares.
  - [Validation and tuning](05_Validation_and_Tuning.md) — the walk-forward and the search.
  - [Optimisers](02_Optimisers.md) — the naive family the head belongs to.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New guide page for #1165 on map #1148. Fixtures follow the prototype's two-regime
#src   construction (research/prototypes/run_novel.jl § 09) at N = 5, T = 500, StableRNG.
#src - Numbers in the prose are re-measured from the run; see the resolution comment on #1165.
