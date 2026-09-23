#=
```@meta
Description = "Constraints and costs as keywords of JuMPOptimiser, with one minimal call each for weight bounds, groups, factor exposures, turnover and fees."
```

# Constraints and costs

A real mandate has constraints. You cap the weight of one asset, hold a sector inside a band, limit
how much you trade at each rebalance, and pay fees. In `PortfolioOptimisers.jl` these are keywords
of the [`JuMPOptimiser`](@ref). The `JuMPOptimiser` holds the constraints and the costs, and the
optimiser that takes it, such as `MeanRisk`, holds the objective. This page shows the common ones
with one minimal call each. For the others, see the
[constraints and costs examples](../examples/4_constraints_costs/01_Budget_Constraints.md).

We compute one empirical prior. Every call but the factor call uses it, and every call but the
last two minimises the risk. You can then compare the portfolio under each constraint with the
same base portfolio.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 1. Weight bounds

[`WeightBounds`](@ref) sets a lower and an upper bound on the weight of each asset, through the
`wb` keyword. The default is `lb = 0` and `ub = 1`, a long-only portfolio. A lower `ub` spreads
the weight over more assets, because no asset can hold more than the bound.
=#

res_cap = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                wb = WeightBounds(; lb = 0.0, ub = 0.10))))

#=
The budget, the sum of the weights, is the `bgt` keyword, `1.0` by default. With
[`BudgetRange`](@ref) and a separate short budget `sbgt`, you can build a long-short portfolio or a
leveraged portfolio. See
[Budget Constraints](../examples/4_constraints_costs/01_Budget_Constraints.md).

## 2. Linear and group constraints

You write a group constraint or a linear constraint as a string over a [`UniverseSets`](@ref),
and you pass it to `lcse` in a [`LinearConstraintEstimator`](@ref). Views use the same syntax.
Name a group, then bound it. We require the tech group to hold at least 15% of the weight. The
minimum-risk portfolio gives it almost none.
=#

sets = UniverseSets(; dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"]))
res_grp = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv, sets = sets,
                                                lcse = LinearConstraintEstimator(;
                                                                                 val = ["tech >= 0.15"]))))

#=
The same `lcse` takes a bound on one asset, such as `"AAPL <= 0.1"`, and a bound between two
assets, such as `"MSFT >= AMD"`. For constraints built from the hierarchy of the assets, see
[Phylogeny and Centrality](../examples/4_constraints_costs/04_Phylogeny_Centrality.md).

## 3. Factor exposure constraints

A mandate often names factors and not assets, as in "at most 10% momentum" or "no net exposure
to value". Such a constraint bounds the factor exposures of the portfolio, `w_f = Mᵀw`, where `M`
is the matrix of loadings that a factor model computes.

Put a [`LinearConstraintEstimator`](@ref) in an [`ExposureConstraintEstimator`](@ref), and set its
`space` keyword to [`FactorSpace`](@ref). The keyword has no default. The names in the
constraint must be factor names of the loadings. For a time-series regression, these are the
names that a [`UniverseSets`](@ref) holds under `tfkey`, `"nf"` by default. When the estimator
builds the constraint, it multiplies each row by the loadings. The optimiser then receives an
ordinary constraint on the asset weights, which you can combine with every other constraint on
this page.

The constraint needs loadings, and the loadings need factor data. [`prices_to_returns`](@ref)
takes no factor prices, so we give them to [`price_ingestion`](@ref) as `F`. By default the
loadings come from the prior, so the prior must be a [`FactorPrior`](@ref) and not an
[`EmpiricalPrior`](@ref). If the prior has no loadings, the constraint throws an error, and it
does not drop the row. `FactorSpace(; re = <a fitted Regression>)` fixes the loadings.
`FactorSpace(; re = StepwiseRegression())` fits them when the prior has none, so the constraint
works on any prior. The
[factor exposure example](../examples/4_constraints_costs/10_Factor_Exposure_Constraints.md) gives
the order in which these sources apply.
=#

Fac = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/Factors.csv.gz"));
                timestamp = :Date)[(end - 252):end]
rd_f = prices_to_returns(price_ingestion(PriceIngestion(), X; F = Fac))
sets_f = UniverseSets(; dict = Dict("nx" => rd_f.nx, "nf" => rd_f.nf))

res_fac = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = FactorPrior(), slv = slv,
                                                sets = sets_f,
                                                lcse = ExposureConstraintEstimator(;
                                                                                   lce = LinearConstraintEstimator(;
                                                                                                                   val = "MTUM >= 0.2"),
                                                                                   space = FactorSpace()))),
                   rd_f)

#=
We compute the factor exposures of the result with the loadings on its prior, which are the
loadings that the optimiser used. The `MTUM` row shows the floor of the constraint.
=#

pretty_table(DataFrame("Factor" => rd_f.nf,
                       "Exposure" => transpose(res_fac.pa.pr.rr.M) * res_fac.w);
             formatters = [resfmt], title = "Realised factor exposures")

#=
Pass the prior estimator, not a computed prior. Then the optimiser builds the constraint again
inside each cross-validation fold, with the loadings that the fold fitted. A row that you compute by
hand once does not match the loadings of the later folds, and the factor exposure example measures
how far it moves. That page also covers groups of factors, factor rows mixed with asset rows, and
the constraints that have no factor form.

A constraint has a factor form only if it is a linear row in `w`. Cardinality, thresholds,
turnover, tracking error and fees have none. Weight bounds have none either, because `wb` takes a
box on each asset and no linear rows. To bound a factor exposure from both sides, write the two
rows through `lcse`. To track a factor, you need no factor form. [`ReturnsTracking`](@ref) takes
the return series of a benchmark, and the return series of a factor is a column of the factor
matrix.

## 4. Turnover

Trading limits and costs are keywords too. [`Turnover`](@ref) (`tn`) bounds how far the weight of
each asset can move from a reference portfolio `w`, which is usually the portfolio that you hold.
`val` is the largest change for each asset. We use the equal-weighted portfolio as the portfolio
that you hold, and we bound the change of each asset at 0.02.
=#

w_held = fill(inv(length(rd.nx)), length(rd.nx))
res_tn = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv,
                                               tn = Turnover(; w = w_held, val = 0.02))))

#=
The reference holds 5% in each of the 20 assets, so each weight of the result stays between 3% and
7%. The base portfolio holds about 37% in one asset, and the bound keeps that asset at 7% or less.
The `Turnover` column of the table at the end shows the result. A larger `val` lets the result move
further from the reference, towards the base portfolio. If the reference is the optimum of the same
problem, the bound changes almost nothing, because the optimum is already inside it.

## 5. Fees

[`Fees`](@ref) (`fees`) charges a fee in each period on the positions that you hold, at one rate
on the long weights and at another on the short weights, with optional fixed fees. It can also
charge a fee on the traded weights. The optimiser subtracts the fees from the expected return,
which a return objective and a return floor use, and from the return of each period, which a risk
measure such as [`ConditionalValueatRisk`](@ref) uses. The variance, the default risk measure of
`MeanRisk`, uses neither, and a fee does not change the `MinimumRisk` portfolios of this page. The
minimal form sets `l`, the rate on the long weights.
We maximise the ratio of return to risk with [`MaximumRatio`](@ref) two times, first with no fee and
then with a rate of 0.1% on the long weights. The two portfolios differ only by the fee.
=#

res_ratio = optimise(MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_fee = optimise(MeanRisk(; obj = MaximumRatio(; rf = 4.2 / 100 / 252),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                fees = Fees(; l = 0.001))))

#=
The portfolio is long only and fully invested, so a fee on the long weights costs 0.1% in each
period whatever the weights are. It has the effect of a higher risk-free rate. Both portfolios hold
MRK and XOM, and the fee moves about 21% of the weight from MRK to XOM.

The `l1` and `l2` keywords add an L1 or an L2 penalty on the weights, which you can use in place
of a hard limit on the turnover or on the positions. The `l2c` keyword is a hard constraint, a
ceiling on the 2-norm of the weights, which sets a lower limit on the number of effective assets.
See [Regularisation](../examples/4_constraints_costs/07_Regularisation.md). The `tr` keyword takes
a [`TrackingError`](@ref) to a benchmark. See
[Turnover and Tracking](../examples/4_constraints_costs/05_Turnover_and_Tracking.md).

## 6. Custom objectives and constraints

A mandate can need something that no keyword covers, such as a preference for each asset from a
factor score, or a relation between weights that is not a linear bound. Two keywords of
[`JuMPOptimiser`](@ref) let you write it directly into the JuMP model:

  - `cobj` takes a [`CustomJuMPObjective`](@ref). You implement
    [`add_custom_objective_term!`](@ref) to add a term for a preference, and you add the term with
    [`add_to_objective_penalty!`](@ref).
  - `ccnt` takes a [`CustomJuMPConstraint`](@ref). You implement [`add_custom_constraint!`](@ref)
    to add a constraint to the model.

Each keyword takes one estimator or a vector of estimators, and each function dispatches on the
type of the estimator. The constraint function takes `(model, ccnt, optimiser, attrs)`. The
objective function also takes the [`ObjectiveFunction`](@ref), before the estimator,
`(model, obj, cobj, optimiser, attrs)`. If you define a subtype and no method for it, the
optimiser throws an error when it builds the model.

The library adds a custom objective term to the objective penalty, which enters the objective with
the sign that makes the objective worse, for a minimisation and for a maximisation. You write a
cost as a positive contribution and a reward as a negative contribution. One definition is correct
under every objective, [`MaximumRatio`](@ref) included. The
[custom objectives and constraints example](../examples/4_constraints_costs/09_Custom_Objectives_and_Constraints.md)
builds a momentum tilt and a momentum floor. It also shows the two values of the model that a
constraint written by hand needs. [`get_constraint_scale`](@ref) returns the scale of the
constraints, and [`get_k`](@ref) returns the variable `k`, which rescales the weights under a ratio
objective.
=#

#=
## 7. Comparing the effect

The table and the plot compare the six portfolios. The first four minimise the risk with the
same prior, so a difference between them comes from the constraint. The last two maximise the
ratio of return to risk, and they differ only by the fee.
=#

results = [res_base, res_cap, res_grp, res_tn, res_ratio, res_fee]
labels = ["Base", "Cap 10%", "Tech ≥ 15%", "Turnover", "Max ratio", "Max ratio, fees"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights of the six portfolios")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow guide page: constraints & costs are JuMPOptimiser keywords (wb/bgt/lcse/tn/fees),
#src   one minimal call each, depth deferred to 4_constraints_costs examples. Verified on kaimon
#src   (session f102cae9).
#src - Verified bindings: wb ub=0.10 caps max weight 37%→10%; group FLOOR "tech >= 0.15" forces
#src   tech 0%→15% (a tech CAP never binds on this slice — min-risk AND max-ratio both put ~0%
#src   in AAPL/AMD/MSFT, so I used a floor for a visible effect). Turnover tightening pulls the
#src   solution toward the reference (directional; `val` is NOT literally sum|Δw|, so prose stays
#src   qualitative). Fees solves on a MaximumRatio objective.
#src - Cross-links to 4_constraints_costs/02 (linear/group), 03 (phylogeny/centrality), 04
#src   (turnover/tracking) point at example pages NOT YET AUTHORED — resolve on Documenter
#src   linkcheck once that group lands. → guide/examples cross-link audit.
#src - §5 Custom objectives/constraints: the worked MomentumTilt/MomentumFloor example was
#src   MOVED to examples/4_constraints_costs/09_Custom_Objectives_and_Constraints.jl (a deep
#src   dive: λ sweep, sense/sign, floor sweep, the k idiom, and vector composition). This page
#src   now keeps only a shallow pointer, matching the guide=shallow / examples=deep split.
#src   §6 comparison drops the two momentum columns (res_tilt/res_floor no longer defined here).
