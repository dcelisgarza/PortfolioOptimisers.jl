The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/01_Basic_Optimisation.jl"
```

# Basic optimisation

This is a collection of quickfire tutorials to help you get started with `PortfolioOptimisers.jl` without delving into the examples and/or documentation.

## 1. Computing returns

Usually, price data is obtained using an API, and the returns have to be computed. In `PortfolioOptimisers.jl`, we have [`prices_to_returns`](@ref), which handles asset, factor, and benchmark price data, as well as implied volatilities, and volatility premiums. It performs appropriate data validation checks to ensure the data is consistent. It can also preprocess missing price data, fill gaps using [`Impute.jl`](https://github.com/invenia/Impute.jl), and collapse it to lower frequencies using [`TimeSeries.jl`](https://github.com/JuliaStats/TimeSeries.jl).

Here we show a quick example of a heterogeneous dataset that only returns data with matching timestamps.

````@example 01_Basic_Optimisation
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, LinearAlgebra,
      StableRNGs
resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz"));
                                 timestamp = :Date)[(end - 7 * 252):(end - 252 * 2)],
                       TimeArray(CSV.File(joinpath(@__DIR__, "../examples/Factors.csv.gz"));
                                 timestamp = :Date)[(end - 6 * 252):(end - 252)];
                       B = TimeArray(CSV.File(joinpath(@__DIR__,
                                                       "../examples/SP500_idx.csv.gz"));
                                     timestamp = :Date)[(end - 5 * 252):end])
````

## 2. Basic optimisations

There are many optimisers available in `PortfolioOptimisers.jl`. Here we will showcase their basic usage.

### 2.1 Naive optimisers

Naive optimisers use very basic algorithms that offer robustness and diversification by virtue of being unsophisticated.

#### 2.1.1 Inverse volatility

[`InverseVolatility`]-(@ref) uses the diagonal of the prior covariance to set the asset weights. If the property `sq` is set to `true`, the weights are the inverse of each entry in the diagonal, else it is the inverse of the square root of each entry in the diagonal.

````@example 01_Basic_Optimisation
# Get the variance from the prior covariance.
variance = diag(prior(EmpiricalPrior(), rd).sigma)

# Optimisers
iv1 = InverseVolatility()
iv2 = InverseVolatility(; sq = true)

# Broadcast the optimisers
ress = optimise.([iv1, iv2], rd)

# Calculate the inverse volatility and variance weights
inv_vol = 1 ./ sqrt.(variance)
inv_vol /= sum(inv_vol)
inv_var = 1 ./ variance
inv_var /= sum(inv_var)

# Display results
pretty_table(DataFrame([rd.nx ress[1].w inv_vol ress[2].w inv_var],
                       ["assets", "Opt Vol", "Inv Vol", "Opt Var", "Inv Var"]);
             formatters = [resfmt], title = "Composition")
````

#### 2.1.2 Equal weighted

[`EqualWeighted`]-(@ref) assigns equal weights to all assets.

````@example 01_Basic_Optimisation
res = optimise(EqualWeighted(), rd)
pretty_table(DataFrame([rd.nx res.w], ["assets", "Weights"]); formatters = [resfmt],
             title = "Composition")
````

#### 2.1.3 Random weighted

[`RandomWeighted`]-(@ref) assigns weights according to a [`Dirichlet`](https://juliastats.org/Distributions.jl/latest/multivariate/#Distributions.Dirichlet) distribution. The `alpha` keyword argument is forwarded to the Dirichlet distribution to control its shape. It's also possible to control the random number generator using the `rng` and `seed` keyword arguments.

````@example 01_Basic_Optimisation
res = optimise(RandomWeighted(; alpha = 1, rng = StableRNG(696), seed = 66420), rd)
pretty_table(DataFrame([rd.nx res.w], ["assets", "Weights"]); formatters = [resfmt],
             title = "Composition")
````

### 2.2 JuMP optimisers

JuMP-based optimisers implement traditional mathematical optimisation algorithms using `JuMP`. As such, they are the most flexible when it comes to constraints, and for those which accept them, objective functions. Most risk measures are also compatible with these. There are a few risk measures exclusively compatible with clustering optimisations, and a few others which are incompatible for use in optimisations.

All JuMP-based optimisers require an instance of [`JuMPOptimiser`]-(@ref) with a JuMP-compatible [`Solver`](@ref), or vector of solvers. Other than optimisation-specific constraints, general constraints are applied at the level of the [`JuMPOptimiser`]-(@ref). Problem feasibility depends on the specific constraints and the provided solver's support for constraint types and ability to solve the problem.

If using open-source solvers, we recommend [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) when not using MIP constraints. When using MIP constraints, [`Pajarito`](https://github.com/jump-dev/Pajarito.jl) with [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) as the continuous solver and [`HiGHS`](https://github.com/jump-dev/HiGHS.jl) as the MIP one works very well. This makes it possible to solve problems with exotic constraint combinations.

````@example 01_Basic_Optimisation
using Clarabel
slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = "verbose" => false),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9)),
       Solver(; name = :clarabel5, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.80)),
       Solver(; name = :clarabel7, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.7)),
       Solver(; name = :clarabel8, solver = Clarabel.Optimizer,
              check_sol = (; allow_local = true, allow_almost = true),
              settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                              "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                              "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                              "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                              "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4))];
nothing #hide
````

#### 2.2.1 Mean risk

[`MeanRisk`]-(@ref) is the traditional portfolio optimisation problem. It seeks to minimise the risk with respect to a target return, or maximise the return with respect to a target risk. It supports four objective functions via the `obj` keyword which defaults to [`MinimumRisk`]-(@ref), the risk measure(s) are specified with the `r` keyword which defaults to [`Variance`](@ref).

````@example 01_Basic_Optimisation
# We'll use the same optimiser for all mean risk objectives
opt = JuMPOptimiser(; slv = slv)

# Minimum risk (default)
mr1 = MeanRisk(; obj = MinimumRisk(), opt = opt)

# Maximum utility
mr2 = MeanRisk(; obj = MaximumUtility(), opt = opt)

# Maximum risk adjusted return ratio
mr3 = MeanRisk(; obj = MaximumRatio(), opt = opt)

# Maximum return
mr4 = MeanRisk(; obj = MaximumReturn(), opt = opt)

# Optimise all objective functions at once using broadcasting
ress = optimise.([mr1, mr2, mr3, mr4], rd);
nothing #hide
````

`PortfolioOptimisers.jl` provides users with the ability to use multiple risk measures per optimisation, which means that some risk measure have to keep certain internal statistics to be able to compute the risk.

The [`MeanRisk`]-(@ref) optimiser defaults to the variance so we will use that to compute the risk statistics. We've used the same prior statistics [`EmpiricalPrior`](@ref) estimator, and portfolio returns [`ArithmeticReturn`]-(@ref) estimator for all objectives. So we only need to get the ones in the first result.

Since the package's structs are all immutable, we provide factory functions that create risk measures with the appropriate internal statistics. This enables programmatic construction of risk measures, manual construction is also possible by directly using the risk measure's constructor.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# This generates the standard deviation risk measure with the right covariance matrix.
# Alternatively, we could do `StandardDeviation(; sigma = pr.sigma)`, but factory
# functions let you do this programmatically.
r = factory(StandardDeviation(), pr)

# There are `ArithmeticReturns` (default) and `LogarithmicReturns`.
ret = mr1.opt.ret

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ret, res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt], title = "Statistics")
````

#### 2.2.2 Factor risk contribution

The [`FactorRiskContribution`]-(@ref) is a more complex estimator that requires more setup. It accepts objective functions, but can also define risk contributions per factor for the variance risk measure. The minimum risk optimisation will follow the risk contribution constraints the closest. With enough data and assets, it can be quite exact up to the user-provided convergence settings for the solvers used.

It is compatible with other risk measures, but only the variance can take risk contribution constraints. Without them, or when using other risk measures, it is largely the same as the [`MeanRisk`]-(@ref) estimator.

We need to provide an instance of [`AssetSets`](@ref) via the `sets` keyword defining the sets of factors and their relationships. This way, the estimator can generate the linear constraints by parsing user-provided equations.

[`AssetSets`](@ref) is be used throughout the package for similar purposes, though mostly in the context of defining sets of assets. Other factor-based uses include combining [`FactorPrior`](@ref) with [`EntropyPoolingPrior`](@ref) and/or [`BlackLittermanPrior`](@ref) type prior estimators.

The [`AssetSets`](@ref) struct has a `key` property which defines the default search key in `dict`, `dict` must contain a key matching `key` whose value is taken to be the names of the assets/factors around which the sets are defined.

In this case, we use it to define the set of factors with key "nf" (default "nx"). We use these to define the factor risk contribution constraints, and we have to assign the linear constraint estimator for the risk contribution to the [`Variance`](@ref). We'll define the constraints such that each factor contributes between 30% and 12% of the total variance risk.

````@example 01_Basic_Optimisation
# Define the factor sets using the "nf" key
sets = AssetSets(; key = "nf", dict = Dict("nf" => rd.nf))

# Each factor contributes `12% <= x <= 30%` of the total variance risk
lcs = LinearConstraintEstimator(;
                                val = [["0.12 <= $f" for f in rd.nf];
                                       ["$f <= 0.3" for f in rd.nf]])

# Add the linear constraint estimator to the variance risk measure
r = Variance(; rc = lcs)
````

We can optimise for the different objective functions.

````@example 01_Basic_Optimisation
# We'll use the same optimiser for all factor risk contribution objectives
opt = JuMPOptimiser(; slv = slv)

# Minimum risk (default)
frc1 = FactorRiskContribution(; r = r, obj = MinimumRisk(), sets = sets, opt = opt)

# Maximum utility, `l` controls the risk aversion.
frc2 = FactorRiskContribution(; r = r, obj = MaximumUtility(; l = 8), sets = sets,
                              opt = opt)

# Maximum risk adjusted return ratio, rf is the risk free rate.
frc3 = FactorRiskContribution(; r = r, obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                              sets = sets, opt = opt)

# Maximum return
frc4 = FactorRiskContribution(; r = r, obj = MaximumReturn(), sets = sets, opt = opt)

# Optimise all objective functions at once using broadcasting
ress = optimise.([frc1, frc2, frc3, frc4], rd);
nothing #hide
````

We can display the results and factor risk contributions [`factor_risk_contribution`]-(@ref), but we have to normalise them using their sum. Again we use the factory function to set the appropriate internal parameters. The last entry in the risk contribution is the regression intercept.

````@example 01_Basic_Optimisation
# Generate the risk measure with the right covariance matrix
r = factory(r, pr)

# Compute and normalise the factor risk contributions
rkcs = [factor_risk_contribution(r, res.w, pr.X; rd = rd) for res in ress]
rkcs = rkcs ./ sum.(rkcs)

# Display asset weights
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt], title = "Composition")

# Display factor risk contributions
pretty_table(hcat(DataFrame(:factors => [rd.nf; "Intercept"]),
                  DataFrame(reduce(hcat, rkcs),
                            ["RC MinRisk", "RC Max Util", "RC Max Ratio", "RC Max Ret"]));
             formatters = [resfmt], title = "Factor Risk Contributions")
````

#### 2.2.3 Near optimal centering

[`NearOptimalCentering`]-(@ref) is a way to smear an optimal portfolio within a region around the point of optimality. The size of this region can be tuned by the user via the `bins` keyword, or automatically decided based on the number of observations and assets (default). This makes the portfolio more robust to estimation error and more diversified. It is not compatible with risk measures which produce quadratic risk expressions, so the risk measure keyword `r` defaults to [`StandardDeviation`](@ref).

There are two variants, defined by the `alg` keyword, one which applies all constraints to the inner [`MeanRisk`]-(@ref) optimisation and leaves the near optimal portfolio to fall where it may [`UnconstrainedNearOptimalCentering`]-(@ref), meaning the constraints will not be satisfied by the near optimal portfolio. The second variant applies the constraints to the near optimal portfolio as well [`ConstrainedNearOptimalCentering`]-(@ref).

````@example 01_Basic_Optimisation
# We'll use the standard deviation risk measure, risk measures
# that generate quadratic expressions do not work with `NearOptimalCentering`
r = StandardDeviation()

# We'll use the same optimiser for all optimisations
opt = JuMPOptimiser(; slv = slv)

# Minimum risk (default)
noc1 = NearOptimalCentering(; obj = MinimumRisk(), opt = opt)

# Maximum utility
noc2 = NearOptimalCentering(; obj = MaximumUtility(; l = 0.5), opt = opt)

# Maximum risk adjusted return ratio
noc3 = NearOptimalCentering(; obj = MaximumRatio(), opt = opt)

# Maximum return
noc4 = NearOptimalCentering(; obj = MaximumReturn(), opt = opt)

# Optimise all objective functions at once using broadcasting
ress = optimise.([noc1, noc2, noc3, noc4], rd);
nothing #hide
````

View and compute the results.

````@example 01_Basic_Optimisation
# Prior statistics result, we will use the covariance matrix, `sigma`
pr = ress[1].pr

# This generates the standard deviation risk measure with the right covariance matrix.
# Alternatively, we could do `StandardDeviation(; sigma = pr.sigma)`, but factory
# functions let you do this programmatically.
r = factory(StandardDeviation(), pr)

# There are `ArithmeticReturns` (default) and `LogarithmicReturns`.
ret = mr1.opt.ret

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ret, res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt], title = "Statistics")
````

#### 2.2.4 Risk budgeting

[`RiskBudgeting`]-(@ref) provides a way to allocate risk across assets or factors via the `rba` keyword according to a user-defined risk budgeting vector provided via the `rkb` keyword of [`AssetRiskBudgeting`]-(@ref) and [`FactorRiskBudgeting`]-(@ref) risk budgeting algorithms. The risk budget vectors do not have to be normalised. The risk being budgeted depends on the risk measures used. This does not support objective functions, the optimisation is solely focused on achieving the risk budgeting as closely as possible. It is compatible with the same risk measures as [`MeanRisk`]-(@ref).

##### 2.2.4.1 Asset risk budgeting

This version allocates risk across assets.

````@example 01_Basic_Optimisation
# We'll use the variance risk measure.
r = Variance()

# We'll use the same optimiser for all optimisations
opt = JuMPOptimiser(; slv = slv)

# Equal risk contribution per asset (default)
rba1 = RiskBudgeting(; r = r,
                     rba = AssetRiskBudgeting(;
                                              rkb = RiskBudget(;
                                                               val = fill(1.0,
                                                                          length(rd.nx)))),
                     opt = opt)

# Increasing risk contribution per asset
rba2 = RiskBudgeting(; r = r,
                     rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nx))),
                     opt = opt)

# Optimise all risk budgeting estimators at once using broadcasting
ress = optimise.([rba1, rba2], rd);
nothing #hide
````

View and compute the results.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(r, pr)

# Compute and normalise the asset risk contributions
rkcs = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcs = rkcs ./ sum.(rkcs)

# Display the results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcs)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt])
````

##### 2.2.4.1 Factor risk budgeting

This version allocates risk across factors.

````@example 01_Basic_Optimisation
# We'll use the variance risk measure.
r = Variance()

# We'll use the same optimiser for all optimisations
opt = JuMPOptimiser(; slv = slv)

# Equal risk contribution per factor (default)
rba1 = RiskBudgeting(; r = r,
                     rba = FactorRiskBudgeting(;
                                               rkb = RiskBudget(;
                                                                val = range(; start = 1,
                                                                            stop = 1,
                                                                            length = length(rd.nf)))),
                     opt = opt)

# Increasing risk contribution per factor
rba2 = RiskBudgeting(; r = r,
                     rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nf))),
                     opt = opt)

# Optimise all risk budgeting estimators at once using broadcasting
ress = optimise.([rba1, rba2], rd);
nothing #hide
````

View and compute the results.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(r, pr)

# Compute and normalise the asset risk contributions
rkcas = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcas = rkcas ./ sum.(rkcas)

# Compute and normalise the factor risk contributions
rkcfs = [factor_risk_contribution(r, res.w, pr.X; rd = rd) for res in ress]
rkcfs = rkcfs ./ sum.(rkcfs)

# Display the asset risk contributions
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcas)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt],
             title = "Asset risk contribution")

# Display the factor risk contributions
pretty_table(hcat(DataFrame(:factors => [rd.nf; "Intercept"]),
                  DataFrame(reduce(hcat,
                                   [[[(res.prb.b1 \ res.w); NaN] rkc]
                                    for (res, rkc) in zip(ress, rkcfs)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt],
             title = "Factor risk contribution")
````

#### 2.2.5 Relaxed risk budgeting

[`RelaxedRiskBudgeting`]-(@ref) provides a way to allocate risk across assets or factors according to a user-defined risk budgeting vector, which does not have to be normalised. They are provided in the same way as for [`RiskBudgeting`]-(@ref), it does not accept risk measures as it's only available for the variance, and it will not follow the risk budget as closely.

There are three variants, basic, regularised, and regularised and penalised. Since the asset and factor versions are the same as before we will only show the different relaxed risk budgeting algorithms.

````@example 01_Basic_Optimisation
# We'll use the same optimiser for all optimisations
opt = JuMPOptimiser(; slv = slv)

# Basic
rrb1 = RelaxedRiskBudgeting(;
                            rba = AssetRiskBudgeting(;
                                                     rkb = RiskBudget(;
                                                                      val = range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = length(rd.nx)))),
                            alg = BasicRelaxedRiskBudgeting(), opt = opt)

# Regularised
rrb2 = RelaxedRiskBudgeting(;
                            rba = AssetRiskBudgeting(;
                                                     rkb = RiskBudget(;
                                                                      val = range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = length(rd.nx)))),
                            alg = RegularisedRelaxedRiskBudgeting(), opt = opt)

# Regularised and penalised, `p` is the penalty factor (default 1)
rrrb3 = RelaxedRiskBudgeting(;
                             rba = AssetRiskBudgeting(;
                                                      rkb = RiskBudget(;
                                                                       val = range(;
                                                                                   start = 1,
                                                                                   stop = 1,
                                                                                   length = length(rd.nx)))),
                             alg = RegularisedPenalisedRelaxedRiskBudgeting(; p = 1),
                             opt = opt)

# Optimise all relaxed risk budgeting estimators at once using broadcasting
ress = optimise.([rrb1, rrb2, rrrb3], rd);
nothing #hide
````

View and compute the results.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(StandardDeviation(), pr)

# Compute and normalise the risk contributions for each estimator
rkcs = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcs = rkcs ./ sum.(rkcs)

# Display the results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcs)]),
                            ["B Weights", "B Budget", "Reg Weights", "Reg Budget",
                             "RegPen Weights", "RegPen Budget"])); formatters = [resfmt])
````

### 2.3 Clustering optimisers

Clustering based optimisers use the relationship structure between assets. The weights are a function of the risks associated with those structures. Aside from the [`NestedClusters`]-(@ref) estimator, they all use an instance of [`HierarchicalOptimiser`]-(@ref) which defines common parameters for all clustering optimisers. [`HierarchicalOptimiser`]-(@ref) does not require a solver to be specified via the `slv` keyword unless the clustering optimisation estimator uses a risk measure that requires one.

#### 2.3.1 Hierarchical risk parity

The [`HierarchicalRiskParity`]-(@ref) estimator uses the hierarchical structure of the assets to iteratively partition the assets into smaller and smaller clusters until reaching a leaf node. The weights are computed as a function of the risk between left and right cluster at each partition level. This accepts the same risk measures as JuMP based optimisers, plus some extra ones for which there are no traditional optimisation formulations.

The [`HierarchicalOptimiser`]-(@ref) can be specified via the `opt` keyword, and the risk measure can be specified via the `r` keyword which defaults to the [`Variance`](@ref).

````@example 01_Basic_Optimisation
# Hierarchical risk parity
hrp = HierarchicalRiskParity()

# Optimise the hierarchical risk parity model
res = optimise(hrp, rd);
nothing #hide
````

View and compute the results.

````@example 01_Basic_Optimisation
# Construct the variance with the appropriate covariance matrix
pr = res.pr
r = factory(StandardDeviation(), pr)

# Compute the risk, return and risk adjusted return
rk, rt, ratio = expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr)

# Display hierarchical risk parity weights
pretty_table(DataFrame(:assets => rd.nx, :Weights => res.w); formatters = [resfmt],
             title = "Composition")

# Display risk, return and risk adjusted return
pretty_table(DataFrame(:Stat => ["Std", "Return", "Return/Std"],
                       :Measure => [rk, rt, ratio]); formatters = [resfmt],
             title = "Statistics")
````

#### 2.3.2 Schur complement hierarchical risk parity

The [`SchurComplementHierarchicalRiskParity`]-(@ref) estimator works similarly to [`HierarchicalRiskParity`]-(@ref), but uses the Schur complement to decide whether or not to include more information into the risk calculation. It is only available for the variance and standard deviation because it relies on the Schur complement.

It serves almost as an interpolation between the classic mean variance optimisation and hierarchical risk parity. It has a `params` keyword which allows users to specify an instance, or vector of instances of [`SchurComplementParams`]-(@ref) which specify the risk measure, the "interpolation" parameter `gamma ∈ [0, 1]`, and whether the algorithm should be kept monotonic in risk as `gamma` increases (default).

When `gamma` is `0`, it reduces to the [`HierarchicalRiskParity`]-(@ref) estimator, the closer it is to `1` the closer it is to the classic mean variance optimisation. It's worth noting that there are values of `gamma` for which the Schur augmented matrix may not be positive definite, as it cannot add any more risk information beyond a certain point. So if one wants `gamma` to be large, one should use [`NonMonotonicSchurComplement`]-(@ref) and make sure to disable the positive definite projection in the `pdm` keyword of [`SchurComplementParams`]-(@ref).

````@example 01_Basic_Optimisation
# We can also use the standard deviation
r = Variance()

# Hierarchical risk parity
hrp = HierarchicalRiskParity()

# Schur complement hierarchical risk parity converging to the hierarchical risk parity
sch1 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 0,
                                                                            r = r,
                                                                            alg = MonotonicSchurComplement()))

# Mean variance optimisation
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))

# Schur complement hierarchical risk parity nearing the mean variance optimisation, no positive definite projection, non-monotonic
sch2 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 1,
                                                                            r = r,
                                                                            pdm = nothing,
                                                                            alg = NonMonotonicSchurComplement()))

# Schur complement hierarchical risk parity nearing the mean variance optimisation, monotonic
sch3 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 1,
                                                                            r = r,
                                                                            alg = MonotonicSchurComplement()))

# Optimise all optimisers at once using broadcasting
ress = optimise.([hrp, sch1, mr, sch2, sch3], rd);
nothing #hide
````

We can compute the statistics and visualise the results of each estimator.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(StandardDeviation(), pr)

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            ["HRP", "gamma = 0", "MVO", "gamma = 1", "gamma = :max"]));
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            ["HRP", "gamma = 0", "MVO", "gamma = 1", "gamma = :max"]));
             formatters = [resfmt], title = "Statistics")
````

#### 2.3.3 Hierarchical equal risk contribution

The [`HierarchicalEqualRiskContribution`]-(@ref) estimator uses the hierarchical structure of the assets as well as a score of clustering quality to iteratively break up the asset universe into left and right clusters up until the optimal number of clusters according to the score. Each cluster is treated as a synthetic asset for which the risk is computed. The weight of each cluster is computed based on the risk it represents with respect to the cluster on the other side. The risks of each asset in the cluster are also computed, and the weights computed off of these. The per-asset weights are multiplied by the weight associated with the cluster as a whole. This is repeated for every clustering level until the optimal number of clusters is reached.

The [`HierarchicalOptimiser`]-(@ref) is specified via the `opt` keyword. Since this optimiser breaks up the assets into intra- and inter-cluster optimisations, it's possible to provide inner and outer risk measures via the `ri` and `ro` keywords, both default to the [`Variance`](@ref).

This optimiser got its name from the fact that the original formulation assigned equal weights to the assets within each cluster, while the outer optimisation used the variance. This is a generalization of that formulation.

````@example 01_Basic_Optimisation
using Clustering

# Original Hierarchical equal risk contribution, equal weights as the risk inner risk measure, variance for the outer one
herc1 = HierarchicalEqualRiskContribution(; ri = EqualRiskMeasure(), ro = Variance())

# Flip the original Hierarchical equal risk contribution, equal weights as the risk outer risk measure, variance for the inner one
herc2 = HierarchicalEqualRiskContribution(; ro = EqualRiskMeasure(), ri = Variance())

# Optimise all optimisers at once using broadcasting
ress = optimise.([herc1, herc2], rd);
nothing #hide
````

We can view the results and verify that for the originsl formulation, all assets within a single cluster have the same weight.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(StandardDeviation(), pr)

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights and clustering assignments
pretty_table(DataFrame(:assets => rd.nx, :cluster => assignments(res.clr),
                       :Original => ress[1].w, :Flipped => ress[2].w);
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'), [:Original, :Flipped]));
             formatters = [resfmt], title = "Statistics")
````

#### 2.3.4 Nested clustered

The [`NestedClustered`]-(@ref) optimiser uses the same idea as the [`HierarchicalEqualRiskContribution`]-(@ref), where the optimisation process is split into inner and outer optimisations using the same scoring system for finding the optimal number of clusters. However, unlike [`HierarchicalEqualRiskContribution`]-(@ref), the intra- and inter-cluster optimisations are completely independent. It is possible to provide any non-finite allocation optimisation estimator for the inner and outer estimators independently via the keywords `opti` and `opto` respectively. This means it inherits the requirements for the inner and outer estimators respectively.

It is also possible to optimise the outer estimator by using cross-validation via the `cv` keyword. If provided, a cross-validation prediction is applied to the inner estimators, yielding a predicted returns series for each cluster. The returns vector for each cluster is then taken as the returns vector for a synthetic asset. These are placed into a matrix which is used to optimise the outer estimator. When not using a cross-validation approach, the returns of the synthetic assets are computed directly by multiplying the original returns matrix by an `N×C` matrix, where `N` is the number of assets and `C` is the number of clusters. This weights matrix contains the inner weights of the assets in each cluster, if an asset is not in a cluster, its weight in the corresponding column is 0.

The outer optimisation is performed using the synthetic asset returns matrix using the `opto` estimator. The final weights are computed by multiplying the `N×C` inner weights matrix by the `C×1` outer weights vector, where `N` is the number of assets and `C` is the number of clusters.

[`NestedClustered`]-(@ref) can take any non finite allocation optimiser as the inner or outer optimiser.

````@example 01_Basic_Optimisation
# We'll use the same optimiser for all optimisations
opt = JuMPOptimiser(; slv = slv)

# Emulating the original `HierarchicalEqualRiskContribution`
nco1 = NestedClustered(; opti = EqualWeighted(), opto = RiskBudgeting(; opt = opt))

# Mean risk for both optimisations
nco2 = NestedClustered(; opti = MeanRisk(; opt = opt), opto = MeanRisk(; opt = opt))

# It's even possible to nest them
nco3 = NestedClustered(;
                       opti = NestedClustered(; opti = HierarchicalEqualRiskContribution(;),
                                              opto = RiskBudgeting(; opt = opt)),
                       opto = NestedClustered(; opti = RiskBudgeting(; opt = opt),
                                              opto = MeanRisk(; opt = opt)))

# Optimise all optimisers at once using broadcasting
ress = optimise.([nco1, nco2, nco3], rd);
nothing #hide
````

We can compute some risk characteristics and visualise the results. We can see how the analogous optimisation to the original version of [`HierarchicalEqualRiskContribution`]-(@ref) has a similar behaviour, where all assets within a cluster have the same weight as each other.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Generate the risk measure with the appropriate covariance matrix
r = factory(StandardDeviation(), pr)

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights and clustering assignments
pretty_table(hcat(DataFrame(:assets => rd.nx, :clusters => assignments(ress[1].clr)),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            ["EW-RB", "MR-MR", "NC-HERC-RB_NC-RB-MR"]));
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            ["EW-RB", "MR-MR", "NC-HERC-RB_NC-RB-MR"]));
             formatters = [resfmt])
````

### 2.4 Stacking

The [`Stacking`]-(@ref) optimiser uses a similar approach to [`NestedClustered`]-(@ref), but instead of using a single inner estimator, it uses a vector of estimators, inheriting the requirements of each estimator being used.

The inner weights matrix is constructed by horizontally concatenating the weights returned by optimising each inner estimator. The returns series used in the outer optimisation can be computed using the same approaches as in [`NestedClustered`]-(@ref). The final weights are also computed the same way. It is also possible to provide a vector of weights by which to scale the weights of each inner estimator in order to bias the computation of the outer returns series in favour or against a particular estimator.

[`Stacking`]-(@ref) can take any non finite allocation optimiser as the inner or outer optimiser.

The keywords for the inner and outer optimisers are the same as [`NestedClustered`]-(@ref).

````@example 01_Basic_Optimisation
# We'll use the same optimiser
opt = JuMPOptimiser(; slv = slv)

# Use a few different optimisation estimators
st = Stacking(;
              opti = [MeanRisk(; opt = opt), RiskBudgeting(; opt = opt),
                      InverseVolatility(), HierarchicalEqualRiskContribution()],
              opto = NearOptimalCentering(; opt = opt))

# Optimise estimator
res = optimise(st, rd);
nothing #hide
````

Compute and view the results.

````@example 01_Basic_Optimisation
# Get the prior
pr = res.pr

# Construct the risk measure with the right covariance matrix
r = factory(StandardDeviation(), pr)

# Compute the risk, return, and risk adjusted return ratio
rk, rt, ratio = expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr)

# Display asset weights
pretty_table(DataFrame(:assets => rd.nx, :Weights => res.w); formatters = [resfmt],
             title = "Composition")

# Display statistics
pretty_table(DataFrame(:Stat => ["Std", "Return", "Return/Std"],
                       :Measure => [rk, rt, ratio]); formatters = [resfmt],
             title = "Statistics")
````

### 2.5 Subset resampling

The [`SubsetResampling`]-(@ref) optimiser takes a given number of random asset subsets and optimises those subsets using the given optimiser. The final asset weights are the average weight per asset across all samples, if an asset does not appear in a sample, it is taken to be zero. It is possible to provide a `subset_size` keyword can be a float in `(0, 1)`, in which case it specifies a proportion of the data to use in each subset, or an integer which directly specifies subset size. It is also possible to provide a `n_subsets` keyword to specify the number of subsets to use.

In essence, this is almost an interpolation between the optimiser provided, and the [`EqualWeighted`]-(@ref) optimiser. If `subset_size` is `1`, and `n_subsets` is equal to the number of assets, the optimiser is equivalent to the [`EqualWeighted`]-(@ref) optimiser.

The samples are unique and drawn without replacement.

````@example 01_Basic_Optimisation
# We'll use the same optimiser
opt = JuMPOptimiser(; slv = slv)

# Mean risk for comparison
mr = MeanRisk(; opt = opt)

# Use 80% of the number of assets and take 10 samples
sr1 = SubsetResampling(; rng = StableRNG(666), subset_size = 0.8,
                       opt = MeanRisk(; opt = opt), n_subsets = 10)

# All weights are equal, the rng does not matter
sr2 = SubsetResampling(; subset_size = 1, opt = MeanRisk(; opt = opt),
                       n_subsets = size(rd.X, 2))

# Optimise all estimators at once using broadcasting
ress = optimise.([mr, sr1, sr2], rd);
nothing #hide
````

Compute and view the results.

````@example 01_Basic_Optimisation
# All priors are the same so we can use the first one
pr = ress[1].pr

# Construct the risk measure with the right covariance matrix
r = factory(StandardDeviation(), pr)

# Compute the risk, return and risk adjusted return of all results
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

# Display asset weights
pretty_table(DataFrame(:assets => rd.nx, :MeanRisk => ress[1].w,
                       :SubsetResampling => ress[2].w, :EqualWeighted => ress[3].w);
             formatters = [resfmt], title = "Composition")

# Display statistics
pretty_table(hcat(DataFrame(:Stat => ["Std", "Return", "Return/Std"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MeanRisk, :SubsetResampling, :EqualWeighted]));
             formatters = [resfmt])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
