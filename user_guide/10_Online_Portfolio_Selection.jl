#=
```@meta
Description = "Solver-free online portfolio selection rules on two synthetic markets, with static and dynamic regret and a forecast read off a prior."
```

# Online portfolio selection

Every optimiser met so far fits a moment and solves a programme. The online portfolio selection
family does neither. A rule of the family is one recursion, `w_{t+1} = f(w_t, x_t)`. It reads the
price relative `x_t = 1 + r_t` the market just realised and moves the allocation from it, with no
covariance, no expected return and no solver.

[`OnlinePortfolioSelection`](@ref) is the head that runs a rule, a naive optimiser. In batch it
makes one causal pass over the rows. It reads every row in order from the start allocation, and it
answers the portfolio for the period after the last row. Its online updates are the same recursion
one row at a time, so a walk-forward at `test_size = 1` and the batch pass hold the same path.

A rule carries a guarantee that holds on every price sequence, with no statistical assumption. Its
wealth cannot fall far behind the best constant portfolio chosen in hindsight. That guarantee is a
bound, and it is not a return.

This page runs the two halves of the family on two synthetic markets and shows what each half
bets on. It then measures every rule against three hindsight comparators, shows the one place a
walk-forward and the batch pass differ, and tunes a rate.

The family is Li and Hoi's (2014) survey, in five groups. The first four sections run the first
member of each group. The four sections after them run one later member each: a forecast read off
a prior, a second geometry, a leader over a matched sample, and dynamic regret against a
comparator that moves every row. The full roster by group is at the end of the page.
=#

using PortfolioOptimisers, StableRNGs, DataFrames, PrettyTables, Statistics, Dates, Clarabel

resfmt = (v, i, j) -> isa(v, AbstractFloat) ? "$(round(v; digits = 3))" : v;

#=
## 1. Two markets, side by side

Both fixtures are five assets over five hundred periods from one seed. In the mean-reverting
market each log price is pulled back to a level, so a rise is followed by a fall. In the trending
market the log return of each asset is persistent, so a rise is followed by another rise. Nothing
else differs.
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
The roster below is two benchmark rules, three rules that follow the winner and three that follow
the loser. Every rule runs through the same head and the same walk-forward. It takes twenty
warm-up rows, then one row per fold under [`OnlineIndexWalkForward`](@ref), so each fold folds one
row and answers one portfolio. [`UniversalPortfolio`](@ref) samples its constant portfolios when
you construct it, so it needs the universe size, and a seed pins the draw.
=#

rules = ["Buy and hold" => BuyAndHold(),
         "Constant rebalanced" => ConstantRebalancedPortfolio(),
         "Exponentiated gradient" => ExponentiatedGradient(), "Newton step" => NewtonStep(),
         "Universal portfolio" => UniversalPortfolio(; N = N, seed = 1),
         "Passive-aggressive mean reversion" => PassiveAggressiveMeanReversion(),
         "Moving-average reversion" => MovingAverageReversion(),
         "Robust median reversion" => RobustMedianReversion()]

cv = OnlineIndexWalkForward(20, 1)
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
The two halves of the table behave differently. The benchmarks and the follow-the-winner
rules land within a factor of each other on both markets. The constant rebalanced portfolio, the
exponentiated gradient, the Newton step and the universal portfolio sit together near `1.5` on the
reverting market and near `2.9` on the trending one. Each of them moves slowly away from a
constant portfolio, and a constant portfolio is what their guarantee is stated against.

A follow-the-loser rule is a bet on one property of the market. Where prices revert, the rule buys
every fall and is paid for it on the next row, and the moving-average reversion turns one unit
into twenty-two. Where prices trend, the same rule buys every fall into a further fall, and every
reversion rule compounds to a thousandth of its start. Nothing in the rule finds out which market
it is in, and the twenty-two and the thousandth come from the same update.

Read the table before you run a reversion rule on money. Run a winner rule and a loser rule side
by side on the market at hand, and read the pair as a test of whether that market reverts or
trends. The test costs two runs and estimates nothing.

## 2. The regret table

The guarantee is stated against a comparator chosen in hindsight, so the measurement follows one
rule. Fit the comparator on the rows the strategy was scored on, and predict it in sample over
those same rows. Three comparators cover the literature. The best constant rebalanced portfolio is
[`BestConstantRebalancedPortfolio`](@ref), Cover's (1984) fixed point, which needs no solver. It
is the portfolio [`MeanRisk`](@ref) reaches under a [`LogarithmicReturn`](@ref) and a
maximum-return objective. The best stock is a top-one [`ScoreSelector`](@ref) on log wealth
followed by [`EqualWeighted`](@ref), through a [`Pipeline`](@ref). The uniform constant rebalanced
portfolio is a causal member of the family, run through the same walk-forward.
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
[`log_wealth_regret`](@ref) answers the log terminal wealth of the comparator less the log
terminal wealth of the strategy, so a positive entry is a comparator that won. A negative entry is
expected and is not a defect. The bound says that a rule cannot fall far behind the best constant
portfolio, and it says nothing about the rule pulling ahead of it. A causal rule that reads the
structure of the market does pull ahead. Every reversion rule on the reverting market sits between
`-1.4` and `-2.6` against the best constant portfolio.

The best stock in hindsight is a weaker comparator than the best constant rebalanced portfolio, so
the second column never stands above the first. On the trending market the two columns are equal,
because the best constant portfolio there holds one asset.

The Result also carries a per-row difference series and a Newey-West test of it. The comparator
read those rows and was chosen to win on this one sequence, so its `p` is optimistic. Read it as a
rank, and not as a probability.

The literature bounds the row of the universal portfolio. The [`UniversalPortfolio`](@ref)
docstring states both bounds and attributes them. Cover's (1991) `(N − 1) log(T + 1)` bounds the
exact integral, and it is `24.7` at five assets and 480 periods. `log n_experts` bounds the
sampled mixture the library runs against the best sampled expert, and it is `7.6` at the default
two thousand experts. The row's `0.132` and `0.694` sit far inside both, as every row of the
winner half does.

## 3. The batch pass and the held path

At `test_size = 1` the walk-forward and the batch pass hold the same path, and the target of the
fold after rows `1:t` equals `optimise(opt, rd[1:t])`. At `test_size = k` a fold folds `k` rows one
at a time and answers one portfolio. The walk-forward then holds that one portfolio for `k`
periods, where the pass would have rebalanced on every row. If you ask for a weight drift, the book
the fund holds drifts with the prices over the block as well, and [`DriftedWeights`](@ref) hands
that drifted book to the fee of the next fold, in place of the target. The target is still the
target of the pass. The wealth is not the wealth of the pass.
=#

opt = OnlinePortfolioSelection(; alg = PassiveAggressiveMeanReversion())
cv5 = OnlineIndexWalkForward(20, 5)
cv5d = OnlineIndexWalkForward(20, 5; wd = SelfFinancingDrift(), pws = DriftedWeights())

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
The portfolio the fold answers is the portfolio the pass answers over the same rows. The book the
fund holds at the end of the block is not. It has drifted four points of weight away from the
target, and the next update of the rule reads its own target, never the drifted book. The gap in
wealth comes from how often the rule rebalances. A reversion rule that rebalances on every row is
paid on every reversal, and it turns one unit into `6.3`. One that rebalances every fifth row is
paid on every fifth reversal, and it reaches `2.1`. The drift takes a little more. Neither path is
wrong. A rule is designed for the rebalancing frequency its paper ran it at, and `test_size` is
that frequency.

## 4. Tuning a rate

The rate of a rule is a field, and [`GridSearchCrossValidation`](@ref) addresses it by path. The
score is [`MeanReturn`](@ref) with `flag = true`, the mean log return per period, which is the log
terminal wealth per period. Ranking candidates on that score ranks them on regret against any one
fixed comparator, because the wealth of the comparator is the same constant for every candidate.
=#

grid = GridSearchCrossValidation(["alg.eta" => [0.01, 0.05, 0.2, 0.5]]; cv = cv,
                                 r = MeanReturn(; flag = true))
tuned = search_cross_validation(OnlinePortfolioSelection(; alg = ExponentiatedGradient()),
                                grid, rd_trend)

(; eta = tuned.opt.alg.eta, wealth = wealth(cross_val_predict(tuned.opt, rd_trend, cv)))

#=
The search returns a head, so you run it through the walk-forward like any other. On the trending
market the largest rate on offer wins, because the gradient keeps pointing the same way and a
faster step follows it sooner. It lifts the wealth from `2.9` to `3.0`. On the reverting market
the same search picks the smallest rate, for the same reason.

## 5. The forecast is the bet

A follow-the-loser rule has two parts. It forecasts the next price relative, and it updates the
allocation toward that forecast. [`MovingAverageReversion`](@ref) is [`ForecastReversion`](@ref)
over [`PriceLevelExpectedReturns`](@ref) with a moving average. Its forecast is the ratio of the
average price to the last price, which is below one for an asset that just rose. The same update
takes any expected-returns estimator. [`PriorExpectedReturns`](@ref) reads one off a prior, here an
[`EmpiricalPrior`](@ref) whose mean is weighted exponentially toward the recent rows, so the
forecast is the return that just happened, carried forward.
=#

recent = PriorExpectedReturns(;
                              pe = EmpiricalPrior(;
                                                  me = ExpWeightedExpectedReturns(;
                                                                                  decay = 0.9,
                                                                                  min_obs = 1)))
forecasts = ["Moving average" => MovingAverageReversion(),
             "Recent mean" => ForecastReversion(; me = recent)]

pretty_table(DataFrame("Forecast" => first.(forecasts),
                       "Reverting market" =>
                           [wealth(online(a, rd_rev)) for a in last.(forecasts)],
                       "Trending market" =>
                           [wealth(online(a, rd_trend)) for a in last.(forecasts)]);
             formatters = [resfmt], title = "One update, two forecasts")

#=
The two rows invert each other. The update is the same in both, so the forecast is the whole bet.
A moving-average forecast says that a rise reverts. A recent-mean forecast says that a rise
continues. The rule that reads the second forecast turns one unit into more than a thousand on the
trending market, and it loses almost everything on the reverting one. The prior route is the
general one, because the expected return of any prior is a forecast, a factor model's or a
view-tilted one. The rule holds no opinion of its own about what the forecast means.

## 6. A second geometry

Every first-order rule is one [`MirrorDescent`](@ref) step, and `proj` holds the geometry the step
and its projection are taken in. The exponentiated gradient takes the entropic geometry, and
[`GradientProjection`](@ref) takes the Euclidean one. [`TsallisProjection`](@ref) sits between the
two, and [`LogBarrierProjection`](@ref) sits beyond the entropic one.
=#

geometries = ["Entropic" => ExponentiatedGradient(),
              "Tsallis" => MirrorDescent(; proj = TsallisProjection()),
              "Log barrier" => MirrorDescent(; proj = LogBarrierProjection()),
              "Euclidean" => GradientProjection()]

pretty_table(DataFrame("Geometry" => first.(geometries),
                       "Reverting market" =>
                           [wealth(online(a, rd_rev)) for a in last.(geometries)],
                       "Trending market" =>
                           [wealth(online(a, rd_trend)) for a in last.(geometries)]);
             formatters = [resfmt], title = "One rate, four geometries")

#=
The three interior geometries land within two percent of each other on both markets. The Tsallis
step reaches `1.465` on the reverting market, where the entropic one reaches `1.452`. Each of them
is a slow step away from the uniform portfolio, and the geometry only changes how the step is
measured. The Euclidean row sits seven percent away from them, at `1.36` and `2.95`, because one
`eta` is not one step size in two geometries. An entropic step scales the gradient by the weight it
moves, and a Euclidean step does not.

Choose the geometry for the property you need. A multiplicative step never leaves the interior of
the simplex, and a barrier step cannot leave it, so a rule that must hold every asset at all times
takes one of those two. A Euclidean step can reach a corner, and the projection clips it back onto
the boundary of the set in one move.

## 7. A leader over a matched sample

The pattern-matching group is [`FollowTheLeader`](@ref) over a selector of past rows. The rule
selects the past rows whose preceding window resembles the rows just seen, and it plays the best
constant portfolio over the selected rows alone. [`NearestNeighbourMatch`](@ref) selects the rows
whose two-row history is nearest to the last two rows.
=#

matched = FollowTheLeader(; sel = NearestNeighbourMatch(; window = 2))

(; reverting = wealth(online(matched, rd_rev)),
 trending = wealth(online(matched, rd_trend)))

#=
This is the one rule on the page that wins on both markets, and it is the one rule that reads the
market before it bets. On the reverting market the rows that followed a rise were falls. On the
trending market they were further rises. The leader over those rows is the opposite portfolio in
the two cases. The rule reaches `2.85` on the reverting market and several thousand on the
trending one, whose persistence is far stronger than that of any real market.

You pay for it with one programme per row, a fixed point without a solver here, and a solve under a
[`ProgrammeAllocationSet`](@ref). You also wait for history. The selected sample is empty until
enough rows have accrued, and the rule holds the uniform portfolio over those rows.

## 8. Dynamic regret

Section 2 measured every rule against one portfolio chosen in hindsight. A sequence of portfolios
is the stronger comparator. [`HindsightSplit`](@ref) is a walk-forward whose fold `t` trains on the
rows through `t` and tests on row `t`, so an estimator run through it is a comparator refitted at
every row. [`log_wealth_regret`](@ref) against its Result is dynamic regret, and it reports the
path length of the comparator beside it. Under `prefix = true` the best constant portfolio becomes
be-the-leader. Under `prefix = false` the best stock becomes the best stock of each row.
=#

rdt = test_rows(rd_rev)
be_the_leader = cross_val_predict(BestConstantRebalancedPortfolio(), rdt, HindsightSplit())
per_row_stock = cross_val_predict(best_stock, rdt, HindsightSplit(; prefix = false))

dynamic = DataFrame("Rule" => first.(rules))
for (name, comp) in
    ["Be the leader" => be_the_leader, "Best stock per row" => per_row_stock]
    dynamic[!, name] = [log_wealth_regret(preds_rev[n], comp).regret for n in first.(rules)]
end
pretty_table(dynamic; formatters = [resfmt],
             title = "Dynamic regret on the reverting market, comparator minus rule")

#-

(;
 be_the_leader = log_wealth_regret(preds_rev["Constant rebalanced"], be_the_leader).path_length,
 best_stock_per_row = log_wealth_regret(preds_rev["Constant rebalanced"], per_row_stock).path_length)

#=
Be-the-leader reads each row before it bets on it, so it beats every winner rule by about `2.5` of
log wealth, where the fixed leader of section 2 beat them by a tenth. It walks a path of length
`73` over the 480 rows to do so, and the moving-average reversion still sits `0.18` ahead of it.
The best stock of each row holds one asset and jumps on nearly every row. It walks a path of length
`577`, and no causal rule comes within twenty units of log wealth of it.

That is how you read dynamic regret. The bound a rule carries grows with the path length of the
comparator, and no rule tracks a comparator that moves on every row. The `path_length` field
reports the path the comparator walked, so you can state the bound the rule is held to.

Neither of those two comparators is the one the definition of dynamic regret names, which is the
best sequence under a budget on the path length. Be-the-leader is one sequence at its own path
length, and the best stock of each row is the sequence no budget binds.
[`BudgetedHindsightPath`](@ref) solves for the sequence itself, as one programme over the rows. It
maximises the log wealth of a path that sits on the simplex at every row, under a budget on the
summed length of its steps. It follows the same rule as the other comparators. It is fitted on the
rows it is scored on, predicted over them, one fold per row, so [`log_wealth_regret`](@ref) reads
it as it reads any other comparator. At the path length be-the-leader walked, the two are
different paths.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false))
crp = preds_rev["Constant rebalanced"]
L = log_wealth_regret(crp, be_the_leader).path_length
budgeted = predict(optimise(BudgetedHindsightPath(; L = L, slv = slv), rdt), rdt)
budgeted_reg = log_wealth_regret(crp, budgeted)

(; be_the_leader = log_wealth_regret(crp, be_the_leader).regret,
 budgeted = budgeted_reg.regret, path_length = budgeted_reg.path_length)

#=
The budget binds, so the path length is the `73` be-the-leader walked. The regret against the best
sequence of that length is `10.1`, where the regret against be-the-leader was `2.5`. The steps of
the leader are the drift of a running average, and the same length of path, spent where a switch
pays the most, buys another `7.6` of log wealth over 480 rows. The moving-average reversion that
sat `0.18` ahead of be-the-leader sits `7.4` behind this comparator. At the best stock's path
length of `577` the budget is slack, and the two comparators are the same path. At a budget of zero
the path is the fixed leader of section 2. One estimator therefore covers the three comparators of
the literature and every budget between them.

## [The roster by group](@id user-guide-online-selection-roster)

Every rule is a value of the head's `alg` slot. The groups are Li and Hoi's (2014), and the
[capability catalogue](@ref catalogue-online-portfolio-selection) lists the same roster.

  - **Benchmarks.** [`BuyAndHold`](@ref), [`ConstantRebalancedPortfolio`](@ref), and, in
    hindsight, [`BestConstantRebalancedPortfolio`](@ref).
  - **Follow the winner.** [`MirrorDescent`](@ref) and its constructors
    [`ExponentiatedGradient`](@ref), [`GradientProjection`](@ref), [`EGE`](@ref), [`EGR`](@ref),
    [`EGA`](@ref), [`MAEG`](@ref), [`AEG`](@ref); [`NewtonStep`](@ref);
    [`AdaptiveSubgradient`](@ref); [`OptimisticStep`](@ref); [`ExpectationMaximisation`](@ref);
    [`UniversalPortfolio`](@ref); [`AggregatingExponentialGradient`](@ref). A rate may be a
    schedule: [`InverseSquareRootRate`](@ref), [`DoublingTrickRate`](@ref),
    [`SelfConfidentRate`](@ref), [`HintResidualRate`](@ref), [`WindowedBestRate`](@ref).
  - **Follow the loser.** [`PassiveAggressiveMeanReversion`](@ref);
    [`ForecastReversion`](@ref) and its constructors [`MovingAverageReversion`](@ref),
    [`ExponentialMovingAverageReversion`](@ref), [`RobustMedianReversion`](@ref),
    [`ReweightedPriceRelativeTracking`](@ref), [`GaussianWeightingReversion`](@ref),
    [`LocalAdaptiveLearning`](@ref); [`ForecastTracking`](@ref) with
    [`PeakPriceTracking`](@ref), [`AdaptiveInputCompositeTrend`](@ref),
    [`TrendPromotePriceTracking`](@ref); [`KernelTrendTracking`](@ref);
    [`TransactionCostOptimisation`](@ref); [`ShortTermSparsePortfolio`](@ref);
    [`ConfidenceWeightedMeanReversion`](@ref); [`AntiCorrelation`](@ref).
  - **Pattern matching.** [`FollowTheLeader`](@ref) over a sample selector:
    [`Prefix`](@ref), [`LastRows`](@ref), [`HistogramMatch`](@ref), [`KernelMatch`](@ref),
    [`NearestNeighbourMatch`](@ref), [`CorrelationMatch`](@ref), [`ClusterMatch`](@ref); and
    the solved rules [`ShortTermLossControlPortfolio`](@ref),
    [`LowDimensionEnsemblePortfolio`](@ref).
  - **Meta-learning.** [`ExpertMixture`](@ref) over any experts, with [`Ader`](@ref),
    [`Sword`](@ref), [`SwitchingPortfolio`](@ref) and the weightings
    [`AggregatingAlgorithm`](@ref), [`TopK`](@ref), [`WeakAggregatingAlgorithm`](@ref),
    [`SwitchingWeighting`](@ref); [`FollowTheLeadingHistory`](@ref).

A rule that reads a forecast takes it from an expected-returns estimator. That is
[`PriceLevelExpectedReturns`](@ref) over a price-level statistic, or the expected return of any
prior through [`PriorExpectedReturns`](@ref). Every step is projected onto the allocation set of
the head, in the geometry of the rule. The default [`BoundedAllocationSet`](@ref) is the simplex
and has a closed form, and a [`ProgrammeAllocationSet`](@ref) takes the full constraint vocabulary
through a solver. You evaluate a run with [`log_wealth_regret`](@ref), with the
[`HindsightSplit`](@ref) that refits any estimator at every row, with
[`BudgetedHindsightPath`](@ref), the best comparator sequence under a budget on the path length,
and with [`performance_summary`](@ref) and a benchmark.

## Where to go next

  - [The online portfolio selection example](../examples/3_optimisers/18_Online_Portfolio_Selection.md)
    runs the roster on real prices in a walk-forward with a turnover fee, with the search, the
    regret table, the weight path of a rule and its discrete allocation.
  - [The online walk-forward](09_Online_Walk_Forward.md) covers the online scheme every run on
    this page goes through.
  - [Validation and tuning](05_Validation_and_Tuning.md) covers the walk-forward and the search.
  - [Optimisers](02_Optimisers.md) covers the naive family the head belongs to.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New guide page for #1165 on map #1148. Fixtures follow the prototype's two-regime
#src   construction (research/prototypes/run_novel.jl § 09) at N = 5, T = 500, StableRNG.
#src - Numbers in the prose are re-measured from the run; see the resolution comment on #1165.
#src - Section 8 budgeted path (#1216): measured 2026-09-22 at be-the-leader's length 73.02 the
#src   budgeted regret vs the uniform CRP is 10.13 (be-the-leader 2.54), path length 73.02; at the
#src   best stock's 577.0 the two coincide at 22.65. The 480 × 5 solve takes about 20 s.
