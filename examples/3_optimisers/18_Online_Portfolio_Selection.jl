#=
```@meta
Description = "Online portfolio selection on real prices in PortfolioOptimisers.jl: the roster in a fee-charging walk-forward, a rate search and the regret table."
```

# Online portfolio selection

The [user-guide chapter](../../user_guide/10_Online_Portfolio_Selection.md) shows what the two
halves of the family bet on, on two synthetic markets. This example runs the family on real
prices: the benchmark, follow-the-winner and follow-the-loser rules of the roster through a
walk-forward that charges a turnover fee against the book the fund held, a search over a
rule's rate, the regret table against the three hindsight comparators, a step projected onto
a constrained Allocation Set through a solver, and the weight path and discrete allocation of
a rule's Result, which are the same post-processing every optimiser's Result takes.

!!! tip "When to reach for this"
    Reach for the family when you want a solver-free portfolio that reacts to every price
    and carries a worst-case guarantee against the best constant portfolio in hindsight, and
    when you want an assumption-free read on whether a market is trending or reverting. Do
    not reach for a follow-the-loser rule on money before running a follow-the-winner rule
    beside it: the reversion family is a total bet on one market property.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics

pctfmt = (v, i, j) -> isa(v, AbstractFloat) && j > 3 ? "$(round(v * 100; digits = 2)) %" : v;
resfmt = (v, i, j) -> isa(v, AbstractFloat) ? "$(round(v; digits = 3))" : v;

#=
## 1. ReturnsResult data and shared ingredients

The family reads price relatives, `1 + r`, one row at a time, so it wants a long panel more
than a wide one. We take the last thousand rows of the S&P 500 slice the other optimiser
examples load — about four trading years over twenty assets — and the one solver the
hindsight comparator and the constrained update need.
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 1000):end]
rd = prices_to_returns(X)
N = size(rd.X, 2)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 2. The head and the Causal Pass

[`OnlinePortfolioSelection`](@ref) is a naive optimiser: its `alg` is the rule, and its batch
verb runs the rule over every row in order from the start allocation and answers the
portfolio for the period after the last row. Nothing is estimated and nothing is solved.
=#

eg = OnlinePortfolioSelection(; alg = ExponentiatedGradient())
res = optimise(eg, rd)

pretty_table(DataFrame("Asset" => rd.nx, "Next-period allocation" => res.w);
             formatters = [(v, i, j) -> if isa(v, AbstractFloat)
                               "$(round(v * 100; digits = 2)) %"
                           else
                               v
                           end],
             title = "Exponentiated gradient after $(size(rd.X, 1)) rows")

#=
The exponentiated gradient at its default rate moves slowly from the uniform start, so
after a thousand rows every asset still carries weight and the tilt is toward the assets
that compounded best. A reversion rule at its default threshold would be at a corner.

## 3. The roster in a walk-forward with a turnover fee

The walk-forward warms the head up on the first sixty rows and then folds one row per fold
under the Online Scheme [`OnlineIndexWalkForward`](@ref), so every fold is one update and one read-out. A
turnover fee on this family must be measured against the book the fund actually held, and
the scheme's `pws = DriftedWeights()` is that pairing: the loop threads the drifted book into
the next fold's fee, and the fold loop refuses a `tn` fee on this family without it. The fee
is ten basis points per unit of turnover.
=#

fees = Fees(; tn = Turnover(; w = fill(1 / N, N), val = 0.001))
cv = OnlineIndexWalkForward(60, 1; pws = DriftedWeights())

rules = ["Buy and hold" => BuyAndHold(),
         "Constant rebalanced" => ConstantRebalancedPortfolio(),
         "Exponentiated gradient" => ExponentiatedGradient(), "Newton step" => NewtonStep(),
         "Universal portfolio" => UniversalPortfolio(; N = N, seed = 1),
         "Passive-aggressive mean reversion" => PassiveAggressiveMeanReversion(),
         "Moving-average reversion" => MovingAverageReversion(),
         "Robust median reversion" => RobustMedianReversion()]

head(alg) = OnlinePortfolioSelection(; alg = alg, fees = fees)
preds = Dict(name => cross_val_predict(head(alg), rd, cv) for (name, alg) in rules)
wealth(pred) = prod(1 .+ pred.mrd.X)

#=
The same roster without the fee, through the same folds, is the family's own accounting —
rebalanced to the target every row and charged nothing — and the gap between the two
columns is the fee's.
=#

cv0 = OnlineIndexWalkForward(60, 1)
free = Dict(name => cross_val_predict(OnlinePortfolioSelection(; alg = alg), rd, cv0)
            for (name, alg) in rules)

function summary_table(preds, free)
    ps = [performance_summary(preds[n]) for n in first.(rules)]
    return DataFrame("Rule" => first.(rules),
                     "Fee-free wealth" => [wealth(free[n]) for n in first.(rules)],
                     "Net wealth" => [wealth(preds[n]) for n in first.(rules)],
                     "Annualised return" => getproperty.(ps, :ann_return),
                     "Max drawdown" => getproperty.(ps, :max_drawdown),
                     "Turnover per period" => getproperty.(ps, :turnover))
end

pretty_table(summary_table(preds, free); formatters = [pctfmt, resfmt],
             title = "The roster fee-free and net of a 10 bp turnover fee, $(length(preds["Buy and hold"].pred)) periods")

#=
The turnover column is what the fee is charged on: buy and hold trades nothing after the
first fold, so its two columns agree; the constant rebalanced portfolio and the winner rules
trade only what the prices moved, a point of wealth over four years; and a reversion rule at
its default threshold rebalances toward a corner nearly every row and pays two thirds of its
wealth for it. On these years the market trended, so the reversion rules lost before the fee
and the fee took most of the rest. The synthetic chapter shows the same rules turning one
unit into twenty on a reverting market; nothing in the rule knows which market it is on.

## 4. Tuning a rate

A rule's parameters are fields, and a search addresses them by path. The score is
[`MeanReturn`](@ref) with `flag = true`, the mean log return per period, which ranks the
candidates as regret against any fixed comparator would.
=#

grid = GridSearchCrossValidation(["alg.eta" => [0.01, 0.05, 0.2, 0.5, 1.0]]; cv = cv,
                                 r = MeanReturn(; flag = true))
tuned = search_cross_validation(head(ExponentiatedGradient()), grid, rd)
tuned_pred = cross_val_predict(tuned.opt, rd, cv)

(; eta = tuned.opt.alg.eta, wealth = wealth(tuned_pred),
 default = wealth(preds["Exponentiated gradient"]))

#=
The search picks the smallest rate offered: on a panel where no asset keeps winning, the
step that moves least from the uniform portfolio pays the least turnover and gives up the
least to the noise, and the synthetic chapter's trending market picks the largest for the
opposite reason.

## 5. The regret table

The three hindsight comparators are built by the Hindsight Comparator rule, fit on the rows
the strategies were scored on and predicted in sample over them. The best constant rebalanced
portfolio is reached two ways: [`MeanRisk`](@ref) under a [`LogarithmicReturn`](@ref) and a
maximum-return objective through Clarabel, and [`BestConstantRebalancedPortfolio`](@ref),
Cover's fixed point with no solver. On a real panel the best constant portfolio sits at a
corner — three of twenty assets here — and the fixed point's multiplicative iteration
approaches a corner slowly, so at its default iteration cap it reports that it did not
converge and sits a few points of weight and a few thousandths of log wealth short of the
solver's answer. The solver is the comparator on a real panel; the fixed point is the
solver-free cross-check, and its Result says whether to trust it.
=#

rows = 61:size(rd.X, 1)
rdt = ReturnsResult(; nx = rd.nx, X = rd.X[rows, :], ts = rd.ts[rows])

bcrp_jump = optimise(MeanRisk(; obj = MaximumReturn(),
                              opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = slv,
                                                  ret = LogarithmicReturn())), rdt)
bcrp = optimise(BestConstantRebalancedPortfolio(), rdt)
log_wealth(res) = sum(log1p, rdt.X * res.w)

(; converged = bcrp.retcode.res.converged, weight_gap = maximum(abs, bcrp_jump.w - bcrp.w),
 log_wealth_gap = log_wealth(bcrp_jump) - log_wealth(bcrp))

#-

best_stock = Pipeline(;
                      steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                             rule = RankRule(; best = 1)), EqualWeighted()))
comparators = ["Best CRP" => predict(bcrp_jump, rdt),
               "Best stock" => predict(fit(best_stock, rdt), rdt),
               "Uniform CRP" => cross_val_predict(OnlinePortfolioSelection(;
                                                                           alg = ConstantRebalancedPortfolio()),
                                                  rd, cv)]

regret = DataFrame("Rule" => first.(rules))
for (name, comp) in comparators
    regret[!, name] = [log_wealth_regret(preds[n], comp).regret for n in first.(rules)]
end
pretty_table(regret; formatters = [resfmt],
             title = "Log-wealth regret net of fees, comparator minus rule")

#=
Positive is a comparator that won. The comparators are fee-free and the rules are net of
the fee, so every entry is a little above the family's own accounting, and the reversion
rules, which paid the most, sit furthest from every comparator; the winner rules sit within
a few hundredths of the uniform portfolio they started from, which is what a slow step from
it buys. Negative regret is expected and is not a defect, and the `p` the Result carries
against a hindsight comparator is optimistic by construction.

## 6. A constrained update

Every rule's step is projected onto the head's Allocation Set. The default set is the simplex
and the projection is closed form. A [`ProgrammeAllocationSet`](@ref) admits weight bounds,
linear constraints, a turnover ceiling, a risk ceiling and the rest of the constraint
vocabulary, and its projection is a programme the solver runs at every row. Here the
moving-average reversion rule is capped at twenty percent per asset and at ten points of
turnover per asset and period, which turns a corner-seeking rule into a diversified one:
the volatility falls by almost half, and the net wealth more than doubles, most of it
turnover the fee no longer takes. The ceiling is the optimiser's own [`Turnover`](@ref), and
its reference `w` is a placeholder: the head replaces it with the allocation the step trades
from at every row.
=#

capped = OnlinePortfolioSelection(; alg = MovingAverageReversion(), fees = fees,
                                  set = ProgrammeAllocationSet(; slv = slv,
                                                               wb = WeightBounds(0, 0.2),
                                                               tn = Turnover(; w = zeros(N),
                                                                             val = 0.1)))
capped_pred = cross_val_predict(capped, rd, cv)
ps_free = performance_summary(preds["Moving-average reversion"])
ps_capped = performance_summary(capped_pred)

pretty_table(DataFrame("Set" => ["Simplex", "Capped and turnover-limited"],
                       "Terminal wealth" =>
                           [wealth(preds["Moving-average reversion"]), wealth(capped_pred)],
                       "Annualised volatility" =>
                           [ps_free.ann_volatility, ps_capped.ann_volatility],
                       "Max drawdown" => [ps_free.max_drawdown, ps_capped.max_drawdown],
                       "Turnover per period" => [ps_free.turnover, ps_capped.turnover]);
             formatters = [(v, i, j) -> if isa(v, AbstractFloat) && j > 2
                               "$(round(v * 100; digits = 2)) %"
                           else
                               v
                           end, resfmt], title = "Moving-average reversion on two sets")

#=
## 7. The weight path and the discrete allocation

A walk-forward's Result is the same [`MultiPeriodPredictionResult`](@ref) every optimiser
returns, so the area plot renders the path the rule walked and the cumulative-returns
plot renders its wealth; a fold's Result reaches the finite allocation unchanged.
=#

using StatsPlots, GraphRecipes
plot_stacked_area_composition(capped_pred; N = 8)

#-

plot_portfolio_cumulative_returns(tuned_pred; compound = true,
                                  label = "Exponentiated gradient, tuned")

#-

prices = vec(values(X)[end, :])
alloc = optimise(GreedyAllocation(),
                 FiniteAllocationInput(; w = capped_pred.pred[end].res.w, prices = prices,
                                       cash = 100_000.0))
pretty_table(DataFrame("Asset" => rd.nx, "Target weight" => capped_pred.pred[end].res.w,
                       "Shares" => round.(Int, alloc.shares), "Realised weight" => alloc.w);
             formatters = [(v, i, j) -> if isa(v, AbstractFloat)
                               "$(round(v * 100; digits = 2)) %"
                           else
                               v
                           end],
             title = "Discrete allocation of the last fold's target, cash left \$$(round(alloc.cash; digits = 2))")

#=
## Where to go next

  - [Online portfolio selection](../../user_guide/10_Online_Portfolio_Selection.md) — the
    two-regime diagnostic, the regret table on synthetic markets, and the roster by group.
  - [The online walk-forward](../5_validation_tuning/09_Online_Walk_Forward.md) — the Fold
    Fit every run here declares.
  - [Finite allocation](../6_post_processing/01_Finite_Allocation.md) — the discrete
    allocation the last section takes.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New example for #1165 on map #1148. Prose numbers re-measured from the run; see the
#src   resolution comment on #1165.
