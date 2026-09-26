#=
```@meta
Description = "Online portfolio selection on real prices in PortfolioOptimisers.jl: the roster under a turnover fee, a schedule, a mixture, a risk loss and a risk ceiling."
```

# [Online portfolio selection](@id example-online-portfolio-selection)

The [user-guide chapter](@ref user-guide-online-portfolio-selection) compares the
follow-the-winner and the follow-the-loser rules on two synthetic markets. This page runs the
same family on real prices, over four years of twenty S&P 500 names. Sections 3 to 10 each add
one piece.

 3. The benchmark, follow-the-winner and follow-the-loser rules, through a walk-forward that
    charges a turnover fee against the weights you hold after the prices move them.
 4. A search over the rate of one rule.
 5. The regret of each rule against four comparators, three of which saw the whole period.
 6. A step projected onto a set that carries weight bounds and a turnover limit.
 7. A rate that sets itself from the run, in place of one you search for.
 8. A weighted average of the whole roster.
 9. A step that follows a risk measure, and a risk ceiling on the set.
10. The weight path and the share count, which every optimiser's result takes.

!!! tip "When to reach for this"
    Reach for the family when you want a portfolio that reacts to every price and, under most
    rules, needs no solver. Several follow-the-winner rules carry a bound, under the rate and
    the projection their paper states, on how far they can fall behind the best constant
    portfolio chosen with hindsight. The follow-the-loser rules carry none. The family also
    tells you whether a market trended or reverted, and you assume neither beforehand. Run a
    follow-the-winner rule beside a follow-the-loser rule before you put money on the second,
    because a reversion rule bets on one property of the market and on nothing else.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics

pctfmt = (v, i, j) -> isa(v, AbstractFloat) && j > 3 ? "$(round(v * 100; digits = 2)) %" : v;
resfmt = (v, i, j) -> isa(v, AbstractFloat) ? "$(round(v; digits = 3))" : v;

#=
## 1. Data and shared ingredients

A rule of this family reads one row of price relatives, `1 + r`, at a time, so it needs a
long panel more than a wide one. We take the last thousand rows of the S&P 500 slice the other
optimiser examples load, which is about four trading years over twenty assets. We also build
the one solver that the hindsight comparator of section 5 and the constrained sets of sections
6 and 9 need.
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 1000):end]
rd = prices_to_returns(X)
N = size(rd.X, 2)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 2. The optimiser and one pass over the rows

[`OnlinePortfolioSelection`](@ref) is a naive optimiser, and its `alg` field holds the rule.
[`optimise`](@ref) runs that rule over every row in order, from the allocation the optimiser
starts at, and returns the portfolio to hold for the period after the last row. It estimates nothing
and it solves nothing.
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
The exponentiated gradient moves slowly from the uniform start at its default rate. After a
thousand rows every asset still holds weight, and the largest weights sit on the assets that
grew most. A reversion rule at its default threshold would hold one or two assets.

## 3. The roster in a walk-forward with a turnover fee

[`OnlineIndexWalkForward`](@ref) runs the optimiser over the first sixty rows and then gives it
one row per fold, so each fold is one update and one allocation to hold. The prices move the
weights you hold between two rebalances, and we call the moved weights the book. A turnover fee
has to be charged against the book, not against the target you aimed at.
`pws = DriftedWeights()` on the scheme does that. It carries the book of each fold, drifted by
that fold's prices, into the fee of the next one, and the walk-forward refuses a `tn` fee on
this family without it. The fee is ten basis points per
unit traded.
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
We run the same roster over the same folds and charge no fee. That run rebalances to the
target on every row and pays nothing, which is how this family reports its own results. The
difference between the two wealth columns is what the fee took.
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
             title = "Each rule with no fee and net of a turnover fee of ten basis points, over $(length(preds["Buy and hold"].pred)) periods")

#=
The fee applies to the turnover column. That column is the mean, over the rebalances, of the
sum of the absolute weight changes, so a turnover of 100 % moves half the book from some assets
to others. Buy and hold trades nothing after the first fold, so both of its wealth columns hold
the same number. The constant rebalanced portfolio and the
winner rules trade back only what the prices moved, which costs them about a point of wealth
over four years. A reversion rule at its default threshold rebalances toward one or two assets
on nearly every row, and pays two thirds of its wealth for the trading.

On these four years the winner rules almost doubled the money before the fee. The reversion
rules gained less, the passive-aggressive rule lost most of its money before any fee, and the
fee then took most of what the three had. The user-guide chapter runs the same rules on a
synthetic reverting market, where the reversion rules end far above every other rule of its
table. Nothing inside a rule tells it which market it is running on.

## 4. Tuning a rate

The parameters of a rule are fields, and a search names one by its path. The score here is
[`MeanReturn`](@ref) with `flag = true`, the mean log return per period. The user-guide
chapter gives the reason it ranks the candidates in the same order as the regret against any
fixed comparator.
=#

grid = GridSearchCrossValidation(["alg.eta" => [0.01, 0.05, 0.2, 0.5, 1.0]]; cv = cv,
                                 r = MeanReturn(; flag = true))
tuned = search_cross_validation(head(ExponentiatedGradient()), grid, rd)
tuned_pred = cross_val_predict(tuned.opt, rd, cv)

(; eta = tuned.opt.alg.eta, wealth = wealth(tuned_pred),
 default = wealth(preds["Exponentiated gradient"]))

#=
The search picks the smallest rate on the grid. No asset on this panel keeps winning, so the
step that moves least from the uniform portfolio pays the least turnover and loses the least
to noise. The trending market of the user-guide chapter picks the largest rate, for the
opposite reason.

## 5. The regret table

The table measures each rule against four comparators. They are the best constant rebalanced
portfolio, the single stock with the highest mean log return, the uniform constant rebalanced
portfolio, and the best path under a budget on how far it moves. Three of them see the whole
period. Each of the three is fitted on the rows the rules were scored on and then predicted over
those same rows, and that is what hindsight means here. The uniform portfolio is the roster's
own rule, run through the same walk-forward with no fee.

We reach the best constant rebalanced portfolio two ways. The first is [`MeanRisk`](@ref) with
a maximum-return objective under a [`LogarithmicReturn`](@ref), solved by Clarabel. The second
is [`BestConstantRebalancedPortfolio`](@ref), which is Cover's fixed point and needs no solver.

On a real panel the best constant portfolio holds few assets, three of the twenty here. The
multiplicative iteration of the fixed point approaches such a point slowly, so at its default
cap on iterations it reports that it did not converge. It stops a few points of weight and a
few thousandths of log wealth short of the answer the solver gives. Use the solver for the
comparator on a real panel. The fixed point is the check you can run without one, and its
result tells you whether it arrived.
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

#=
The fourth comparator moves between allocations. [`BudgetedHindsightPath`](@ref) is the best
sequence of allocations whose step lengths sum to at most a budget `L`, and dynamic regret
measures a rule against it. It is an estimator like the others, fitted on the rows and
predicted over them, one fold per row.

Moving the whole book from one asset to another costs the square root of two in path length,
so the budget below allows five such moves over the 940 rows. The fit is one programme with 940
exponential cones and 939 step cones. At a budget this tight Clarabel, an interior-point
solver, stalls short of its tolerance. So we pass a vector of solvers, and the fit falls through
to SCS, a first-order solver.
=#

using SCS

slv_path = [slv,
            Solver(; name = :scs, solver = SCS.Optimizer,
                   settings = Dict("verbose" => 0, "eps_abs" => 1e-5, "eps_rel" => 1e-5,
                                   "max_iters" => 200_000),
                   check_sol = (; allow_local = true, allow_almost = true))]
budgeted = predict(optimise(BudgetedHindsightPath(; L = 5 * sqrt(2), slv = slv_path), rdt),
                   rdt)

#-

best_stock = Pipeline(;
                      steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                             rule = RankRule(; best = 1)), EqualWeighted()))
comparators = ["Best CRP" => predict(bcrp_jump, rdt),
               "Best stock" => predict(fit(best_stock, rdt), rdt),
               "Uniform CRP" => cross_val_predict(OnlinePortfolioSelection(;
                                                                           alg = ConstantRebalancedPortfolio()),
                                                  rd, cv), "Budgeted path" => budgeted]

regret = DataFrame("Rule" => first.(rules))
for (name, comp) in comparators
    regret[!, name] = [log_wealth_regret(preds[n], comp).regret for n in first.(rules)]
end
pretty_table(regret; formatters = [resfmt],
             title = "Log-wealth regret net of fees, comparator minus rule")

#-

(; path_length = log_wealth_regret(preds["Buy and hold"], budgeted).path_length,
 wealth = wealth(budgeted), best_crp = prod(1 .+ comparators[1][2].rd.X))

#=
A positive entry is a comparator that beat the rule. The comparators pay no fee and the
rules are net of one, so every entry is a little larger than the family's own accounting would
give. The reversion rules paid the most and sit furthest from every comparator. The winner
rules end within a few hundredths of the uniform portfolio they started from, which is what a
slow step away from it gives you. A negative entry is a rule that beat the comparator, which
happens and is not a defect. The result also carries `p`, the p value of a test that the rule
and the comparator grow at the same expected log rate. Against a comparator fitted on the same
rows, the test overstates the evidence that the comparator is better, so a small `p` there
proves nothing.

Five moves are worth four units of log wealth over the best constant portfolio. The budgeted
path spends its budget where a move pays most. It grows one unit to about 200, where the best
constant portfolio grows it to 3.6, so the regret of every rule against it is its regret
against the static comparator plus four. Read the path length the cell prints against the
budget, `L = 5√2`, about 7.07. SCS is a first-order solver that stops at a tolerance, so the
two need not be equal. A larger budget allows
a longer path and a larger regret, and a budget of zero gives the best constant portfolio
again.

## 6. A constrained update

The optimiser projects the step of every rule onto the set of allocations the rule may hold. The
default set is the simplex, and that projection has a closed form. A
[`ProgrammeAllocationSet`](@ref) takes weight bounds, linear constraints, a turnover limit, a
risk ceiling and the other constraints the library defines, and its projection is a programme
the solver runs on every row.

The cell below caps the moving-average reversion rule at twenty percent per asset, and at ten
points of turnover per asset per period. The rule can no longer move the whole book onto one
asset. Its volatility falls by almost half and its net wealth more than doubles, mostly
because the fee no longer takes the turnover. The limit is the optimiser's own
[`Turnover`](@ref), and the `w` we give it here is a placeholder. The optimiser replaces it on
every row with the allocation the step trades from.
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
## 7. A schedule instead of a search

Section 4 searched for a rate. A rate can instead set itself from the run, and you pass one
of these as the rule's `eta`. [`InverseSquareRootRate`](@ref) and [`DoublingTrickRate`](@ref)
read the number of rows seen so far. [`SelfConfidentRate`](@ref) reads the losses so far.
[`WindowedBestRate`](@ref) scores a grid of rates over a trailing window and plays the best of
them, which is [`MAEG`](@ref) on the exponentiated gradient. None of them needs to know how
many rows are coming, and none needs a second pass over the data.
=#

schedules = ["Fixed rate, tuned" => tuned.opt.alg, "Windowed best rate" => MAEG(),
             "Self-confident rate" => ExponentiatedGradient(; eta = SelfConfidentRate())]
scheduled = Dict(name => cross_val_predict(head(alg), rd, cv) for (name, alg) in schedules)

pretty_table(DataFrame("Rate" => first.(schedules),
                       "Net wealth" => [wealth(scheduled[n]) for n in first.(schedules)],
                       "Turnover per period" => [performance_summary(scheduled[n]).turnover
                                                 for n in first.(schedules)]);
             formatters = [resfmt], title = "The exponentiated gradient under three rates")

#=
The windowed schedule plays the rate that would have won over the last thirty rows, and it
comes out four percent below the tuned fixed rate without a search. Section 4 found that the
smallest rate wins here and wins by little, so the rate that won the last thirty rows is
mostly noise, and the schedule pays a little to follow it. The self-confident rate falls as
the losses add up, and it halves the turnover. Reach for a schedule when the run is the only
pass you get. Reach for a search when you have history to tune on.

## 8. A mixture over the roster

[`ExpertMixture`](@ref) runs a list of rules as experts and plays a weighted average of the
allocations they return. A second rule sets those weights, and it reads the returns the
experts earned. [`WeakAggregatingAlgorithm`](@ref) and [`AggregatingAlgorithm`](@ref) weight
each expert by the wealth it has compounded, and [`TopK`](@ref) follows the `k` wealthiest. Any
first-order rule of the family works there too. The bound a mixture holds is against its best
expert. The experts here are the seven rules of the roster that carry no randomness.
=#

experts = [alg for (name, alg) in rules if name != "Universal portfolio"]
weightings = ["Weak aggregating" => WeakAggregatingAlgorithm(),
              "Aggregating" => AggregatingAlgorithm(), "Top two" => TopK(; k = 2)]
mixtures = Dict(name =>
                    cross_val_predict(head(ExpertMixture(; experts = experts, alg = alg)),
                                      rd, cv) for (name, alg) in weightings)

pretty_table(DataFrame("Weighting" => first.(weightings),
                       "Net wealth" => [wealth(mixtures[n]) for n in first.(weightings)],
                       "Turnover per period" => [performance_summary(mixtures[n]).turnover
                                                 for n in first.(weightings)]);
             formatters = [resfmt],
             title = "A mixture of the seven rules under three weightings")

#=
At its default rate of one, the aggregating weighting is the wealth weighting, and before fees
its log wealth falls at most the logarithm of the number of experts, about 1.95, behind its
best expert. That bound covers the aggregating weighting alone.

A mixture plays the weighted average of its experts' allocations, so it trades whenever an
expert that carries weight trades, and the three reversion experts trade on nearly every row.
The weak aggregating mixture lowers their weight slowly, so it keeps trading with them. It
turns over half its book every period, and the fee brings it to 0.88. The top-two weighting
drops them and reaches 1.53. A mixture spreads your bet over several rules, and the fee it
pays is the turnover of the average it plays, not the turnover of its best expert.

[`Ader`](@ref) and [`Sword`](@ref) are mixtures of this kind over `K` gradient-projection
experts whose rates double from `eta_min`, and the exponentiated gradient weights the experts.
You choose the grid and the rate of the weighting in place of a single step rate.

## 9. Risk in the loss, and risk on the set

A first-order rule steps on the log-wealth loss. [`RiskLoss`](@ref) puts a risk measure
there instead, evaluated over a trailing window. The rule then takes one step per row toward
the portfolio that minimises that measure, and the window moves with each row. The step needs
no solver.

The gradient of a variance is $2 \Sigma w$, which is about the size of a daily variance and
so small. You state the rate against that gradient, so it is far larger than a rate on the
log-wealth loss.
=#

risk_step = MirrorDescent(; obj = RiskLoss(; r = Variance(), window = 60), eta = 50)
risk_pred = cross_val_predict(head(risk_step), rd, cv)
ps_eg = performance_summary(preds["Exponentiated gradient"])
ps_risk = performance_summary(risk_pred)

pretty_table(DataFrame("Loss" => ["Log wealth", "Variance over sixty rows"],
                       "Net wealth" =>
                           [wealth(preds["Exponentiated gradient"]), wealth(risk_pred)],
                       "Annualised volatility" =>
                           [ps_eg.ann_volatility, ps_risk.ann_volatility],
                       "Max drawdown" => [ps_eg.max_drawdown, ps_risk.max_drawdown]);
             formatters = [(v, i, j) -> if isa(v, AbstractFloat) && j > 2
                               "$(round(v * 100; digits = 2)) %"
                           else
                               v
                           end, resfmt], title = "The entropic step on two losses")

#=
The variance step lowers the volatility and the drawdown, and it gives up wealth for that,
which is the trade a minimum-variance portfolio makes.

The other place to put risk is the set. A [`ProgrammeAllocationSet`](@ref) takes any risk
measure whose `settings.ub` holds a ceiling. It resolves that ceiling on every row against a
prior fitted on the rows seen so far, and it projects the step of every rule onto the result.

Choose a ceiling the set can meet. A variance ceiling below the variance of the least volatile
long-only portfolio leaves the programme with no solution. The step is then held, and the rule
trades nothing on that row. On this panel that floor rises above fifteen percent a year on
2020-03-12 and stays above it to the last row, because from then on the prior of the rows seen
so far includes the crash of March 2020. At a ceiling of ten percent a year, the programme has
no solution on about four folds in five.

We run the moving-average reversion rule of section 3, which lost most of its wealth, under a
ceiling on the daily variance equal to a volatility of twenty-five percent a year. No row of the
panel has a floor that high. The last column counts the held steps from the retcode of each
fold, which carries a record for a held row and `nothing` otherwise.
=#

ceiling = Variance(; settings = RiskMeasureSettings(; ub = (0.25 / sqrt(252))^2))
ceiled = OnlinePortfolioSelection(; alg = MovingAverageReversion(), fees = fees,
                                  set = ProgrammeAllocationSet(; slv = slv, r = ceiling))
ceiled_pred = cross_val_predict(ceiled, rd, cv)
ps_ceiled = performance_summary(ceiled_pred)
held_steps(pred) = count(p -> !isnothing(p.res.retcode.res), pred.pred)

pretty_table(DataFrame("Set" => ["Simplex", "Volatility ceiling of 25 %"],
                       "Terminal wealth" =>
                           [wealth(preds["Moving-average reversion"]), wealth(ceiled_pred)],
                       "Annualised volatility" =>
                           [ps_free.ann_volatility, ps_ceiled.ann_volatility],
                       "Turnover per period" => [ps_free.turnover, ps_ceiled.turnover],
                       "Held steps" => [held_steps(preds["Moving-average reversion"]),
                                        held_steps(ceiled_pred)]);
             formatters = [(v, i, j) -> if isa(v, AbstractFloat) && j > 2
                               "$(round(v * 100; digits = 2)) %"
                           else
                               v
                           end, resfmt],
             title = "Moving-average reversion under a risk ceiling")

#=
The realised volatility is above the ceiling, because the ceiling bounds the variance of the
prior fitted on past rows and not the variance of the rows
that follow. The rule still moves about half of its book to other assets every period, and it
ends with less wealth than on the simplex. A risk ceiling bounds the variance the prior gives
each target allocation. It does not change what the rule bets on.

A risk ceiling on the set and a risk loss in the step are two different things. The ceiling is
a constraint every rule meets. The loss is what one rule steps on. Both take any risk measure
the library defines.

## 10. The weight path and the discrete allocation

A walk-forward returns the same [`MultiPeriodPredictionResult`](@ref) that every optimiser
returns. The area plot draws the weight of each asset over the folds, and the
cumulative-returns plot draws the wealth. The result of one fold goes into the finite
allocation as it stands.
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

  - [Online portfolio selection](@ref user-guide-online-portfolio-selection) covers
    the test for a trending or a reverting market, the regret table on synthetic markets, and
    the roster group by group.
  - [The online walk-forward](@ref example-the-online-walk-forward-one-estimator-updated-fold-by-fold) covers the
    walk-forward that every run on this page uses.
  - [Finite allocation](@ref example-finite-allocation) covers the share count
    the last section takes.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New example for #1165 on map #1148. Prose numbers re-measured from the run; see the
#src   resolution comment on #1165.
#src - Section 5 budgeted path (#1216): measured 2026-09-22 at `L = 5√2` on the 940 × 20 panel,
#src   wealth 198.4 (best CRP 3.60), path length 7.094 (SCS at 1e-5), regret 4.62 on the winner
#src   rules and 6.1–7.8 on the reversion rules. Clarabel stalls (INSUFFICIENT_PROGRESS) at this
#src   budget, and the vector falls through to SCS in about 16 s in total.
#src - Section 9 risk ceiling (#1289): measured 2026-09-23. The long-only minimum volatility on
#src   the expanding sample prior exceeds 10 % on 753 of 999 rows, 15 % from 2020-03-12 to the
#src   end, 20 % from 2020-03-20 to 2021-01-11, and peaks at 22.6 % on 2020-04-22; every Held
#src   Step at a ceiling was that infeasibility, not a solver failure. At 10 % the cell held 752
#src   of 940 folds (the count of `held_steps`; the issue counted 719) and printed wealth 1.46
#src   from a book that mostly drifted. At 25 %: no held fold, ex-ante volatility at the ceiling
#src   (within 1e-3) on 813 of 940 folds, median largest weight 0.53 against 1.0 on the simplex,
#src   wealth 0.367, volatility 28.3 %, turnover 0.994 (`sum(abs, dw)`, so half the book).
