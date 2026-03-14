#=
# Cheat Sheet

This is a collection of quickfire tutorials to help you get started with `PortfolioOptimisers.jl` without delving into the examples and/or documentation.
=#

#=
## 1. Downloading data

There are both open and close source providers, in Julia we have [`YFinance.jl`](https://github.com/eohne/YFinance.jl) and [`MarketData.jl`](https://github.com/JuliaQuant/MarketData.jl).

## 2. Computing returns

Usually data is obtained from a provider and the returns have to be computed. `PortfolioOptimisers.jl` has a [`prices_to_returns`](@ref) to do so from price data. It can handle asset, factor, and benchmark returns, as well as implied volatilities, and volatility premiums. It performs appropriate data validation checks to ensure the timestamps match and the data is clean. It can also preprocess missing data and collapse to lower frequencies.

Here we show a quick example of a heterogenous dataset which will only return the data with matching timestamps.
=#
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, LinearAlgebra,
      StableRNGs
resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

rd = prices_to_returns(TimeArray(CSV.File(joinpath(@__DIR__, "SP500.csv.gz"));
                                 timestamp = :Date)[(end - 7 * 252):(end - 252 * 2)],
                       TimeArray(CSV.File(joinpath(@__DIR__, "Factors.csv.gz"));
                                 timestamp = :Date)[(end - 6 * 252):(end - 252)];
                       B = TimeArray(CSV.File(joinpath(@__DIR__, "SP500_idx.csv.gz"));
                                     timestamp = :Date)[(end - 5 * 252):end])

#=
## 3. Basic Optimisations

There are many optimisers available in `PortfolioOptimisers.jl`. Here we will showcase their basic usage.

### 3.1 Naive Optimisers

Naive optimisers do not use sophisticated optimisation algorithms, but rather very basic ones that offer robustness and diversificaiton by virtue of being unsophisticated.

#### 3.1.1 Inverse volatility

This uses the diagonal of the covariance to set the weights, if the flag `sq` is true, the weights are just the inverse of each entry in the diagonal, else it is the inverse of the square root of each entry in the diagonal.
=#
variance = diag(prior(EmpiricalPrior(), rd).sigma)

res1 = optimise(InverseVolatility(), rd)
res2 = optimise(InverseVolatility(; sq = true), rd)
inv_vol = 1 ./ sqrt.(variance)
inv_vol /= sum(inv_vol)
inv_var = 1 ./ variance
inv_var /= sum(inv_var)
pretty_table(DataFrame([rd.nx res1.w inv_vol res2.w inv_var],
                       ["assets", "Opt Vol", "Inv Vol", "Opt Var", "Inv Var"]);
             formatters = [resfmt])
#=
#### 3.1.2 Equal weighted

This assigns equal weights to all assets.
=#

res = optimise(EqualWeighted(), rd)
pretty_table(DataFrame([rd.nx res.w], ["assets", "Weights"]); formatters = [resfmt])

#=
#### 3.1.3 Random weighted

This randomly assigns weights according to a [`Dirichlet`](https://juliastats.org/Distributions.jl/latest/multivariate/#Distributions.Dirichlet) distribution. It's possible to provide a custom alpha parameter as a vector or number, random number generator, and seed.
=#

res = optimise(RandomWeighted(; alpha = 1, rng = StableRNG(696), seed = 66420), rd)
pretty_table(DataFrame([rd.nx res.w], ["assets", "Weights"]); formatters = [resfmt])

#=
### 3.2 JuMP optimisers

The JuMP-based optimisers use traditional mathematical optimisation. As such, they are the most flexible when it comes to constraints, and for those which accept them, objective functions. Most risk measures are also compatible with these, aside from a few exclusively compatible with clustering optimisations, as well as other risk measures which are incompatible with any optimisation. All JuMP-based optimisers require the user to provide a JuMP-compatible solver, which supports the type of constraints being used.

If using open-source solvers we recommend [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) when not using MIP constraints. When using MIP constraints, [`Pajarito`](https://github.com/jump-dev/Pajarito.jl) with [`Clarabel`](https://github.com/oxfordcontrol/Clarabel.jl) as the continuous solver, and [`HiGHS`](https://github.com/jump-dev/HiGHS.jl) as the MIP solver.

Users can provide a vector of solvers which will be iterated over until one solves the problem satisfactorily, or all fail. Other than optimisation-specific constraints, general constraints are applied at the level of the [`JuMPOptimiser`]-(@ref), whether the problem is feasable or not depends on the specific constraints and the provided solver's support for constraint types/ability to solve the problem.
=#
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

#=
#### 3.2.1 Mean Risk

[`MeanRisk`]-(@ref) is the traditional portfolio optimisation problem. It seeks to minimise the risk with respect to a target return, or maximise the return with respect to a target risk. It supports four objective functions via the `obj` keyword which defaults to [`MinimumRisk`]-(@ref) which use the relationship between risk and returns in different ways, and the risk measure(s) are specified with the `r` keyword which defaults to [`Variance`](@ref).
=#

## Minimum risk (default)
mr1 = MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))
## Maximum utility
mr2 = MeanRisk(; obj = MaximumUtility(), opt = JuMPOptimiser(; slv = slv))
## Maximum risk adjusted return ratio
mr3 = MeanRisk(; obj = MaximumRatio(), opt = JuMPOptimiser(; slv = slv))
## Maximum return
mr4 = MeanRisk(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv = slv))
## Optimise all objective functions at once
ress = optimise.([mr1, mr2, mr3, mr4], rd);
#=
`PortfolioOptimisers.jl` provides users with the ability to use multiple risk measures per optimisation, which means that some risk measure have to keep certain internal statistics to be able to compute the risk. We provide factory functions that create the risk measures with the appropraite internal statistics.

The [`MeanRisk`]-(@ref) optimiser defaults to the variance so we will use that to compute the risk statistics. All optimisations use the same prior estimator, as well as portfolio return estimator so we will use only the first
=#
pr = ress[1].pr
r = factory(Variance(), pr)
ret = mr1.opt.ret
rk_rt_ratio = [expected_risk_ret_ratio(r, ret, res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)

## Display results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
#=
#### 3.2.2 Factor Risk Contribution

This is a more advanced estimator, it requires some more set up. It allows users to provide objective functions, but also define risk contributions per factor to the variance risk measure. The minimum risk optimisaion will follow the risk contribution constraints the closest, and with enough data and assets can be quite exact up to the user provided convergence settings for the provided solvers.

It is compatible with other risk measures, but only the variance can take risk contribution constraints, without them or when using other risk measures it is largely the same as the [`MeanRisk`]-(@ref) estimator.

First we need to provide a instance of [`AssetSets`](@ref) which defines sets of assets or factors and their relationships, which lets `PortfolioOptimisers.jl` create linear constraints according to how users define them via [`LinearConstraintEstimator`](@ref). This way it's possible to define groups of assets/factors and how they relate to each other. We will showcase them later. For now, we need these to define the relationship between factors for their risk contribution.

The [`AssetSets`](@ref) has a `key` property which defines the default search key in `dict`, `dict` must contain a key matching `key` whose value is taken to tbe the names of the assets/factors around which the sets are defined.

In this case we use this to define the set of factors with key "nf" (default "nx"). We use these to define the factor risk contribution constraints, and we have to assign the linear constraint estimator for the risk contribution to the [`Variance`](@ref). We'll define the constraints such that each factor contributes between 30% and 12% of the total variance risk.
=#
sets = AssetSets(; key = "nf", dict = Dict("nf" => rd.nf))
lcs = LinearConstraintEstimator(;
                                val = [["$f <= 0.3" for f in rd.nf];
                                       ["$f >= 0.12" for f in rd.nf]])
r = Variance(; rc = lcs)

#=
We can optimise for the different objective functions.
=#

## Minimum risk (default)
frc1 = FactorRiskContribution(; r = r, obj = MinimumRisk(), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
## Maximum utility, `l` controls the risk aversion.
frc2 = FactorRiskContribution(; r = r, obj = MaximumUtility(; l = 8), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
## Maximum risk adjusted return ratio, rf is the risk free rate.
frc3 = FactorRiskContribution(; r = r, obj = MaximumRatio(; rf = 4.2 / 252 / 100),
                              sets = sets, opt = JuMPOptimiser(; slv = slv))
## Maximum return
frc4 = FactorRiskContribution(; r = r, obj = MaximumReturn(), sets = sets,
                              opt = JuMPOptimiser(; slv = slv))
## Optimise all objective functions at once
ress = optimise.([frc1, frc2, frc3, frc4], rd);

#=
We can display the results and factor risk contributions [`factor_risk_contribution`]-(@ref), but we have to normalise them using their sum. Again we use the factory function to set the appropriate internal parameters. The last entry in the risk contribution is the regression intercept.
=#
r = factory(r, pr)
rkcs = [factor_risk_contribution(r, res.w, pr.X; rd = rd) for res in ress]
rkcs = rkcs ./ sum.(rkcs)

pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:factors => [rd.nf; "Intercept"]),
                  DataFrame(reduce(hcat, rkcs),
                            ["RC MinRisk", "RC Max Util", "RC Max Ratio", "RC Max Ret"]));
             formatters = [resfmt])
#=
#### 3.2.3 Near Optimal Centering

[`NearOptimalCentering`]-(@ref) is a way to smear an optimal portfolio accross a region around the point of optimality. The size of this region can be tuend by the user via the `bins` keyword, or automatically decided based on the number of observations and assets (default). This makes the portfolio more robust to estimation error and more diversified. It is not compatible with risk measures which produce quadratic risk expressions, so the risk measure keyword `r` defaults to [`StandardDeviation`](@ref).

There are two variants, defined by the `alg` keyword, one which applies all constraints to the inner [`MeanRisk`]-(@ref) optimisation and leave the near optimal portfolio to fall where it may [`UnconstrainedNearOptimalCentering`]-(@ref), meaning the constraints will not be satisfied by the near optimal portfolio. And another which does apply the constraints to the near optimal portfolio [`ConstrainedNearOptimalCentering`]-(@ref).
=#

## Minimum risk (default)
noc1 = NearOptimalCentering(; obj = MinimumRisk(), opt = JuMPOptimiser(; slv = slv))
## Maximum utility
noc2 = NearOptimalCentering(; obj = MaximumUtility(; l = 0.5),
                            opt = JuMPOptimiser(; slv = slv))
## Maximum risk adjusted return ratio
noc3 = NearOptimalCentering(; obj = MaximumRatio(), opt = JuMPOptimiser(; slv = slv))
## Maximum return
noc4 = NearOptimalCentering(; obj = MaximumReturn(), opt = JuMPOptimiser(; slv = slv))
## Optimise all objective functions at once
ress = optimise.([noc1, noc2, noc3, noc4], rd);

#=
View and compute the results.
=#
pr = ress[1].pr
r = factory(StandardDeviation(), pr)
ret = mr1.opt.ret
rk_rt_ratio = [expected_risk_ret_ratio(r, ret, res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)
## Display results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            [:MinimumRisk, :MaximumUtility, :MaximumRatio, :MaximumReturn]));
             formatters = [resfmt])

#=
#### 3.2.4 Risk Budgeting

[`RiskBudgeting`]-(@ref) provides a way to allocate risk accross assets or factors via the `rba` keyword according to a user-defined risk budgeting vector provided via the `rkb` keyword of [`AssetRiskBudgeting`]-(@ref) and [`FactorRiskBudgeting`]-(@ref) risk budgeting algorithms. The risk budget vectors do not have to be normalised. The risk being budgeted depends on the risk measures used. This does not support objective functions, the optimisation is solely focused on achieving the risk budgeting as closely as possible. It is compatible with the same risk measures as [`MeanRisk`]-(@ref).

##### 3.2.4.1 Asset Risk Budgeting

This version allocated risk accross assets.
=#
r = Variance()
## Equal risk contribution per asset (default)
rba1 = RiskBudgeting(; r = r,
                     rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nx))),
                     opt = JuMPOptimiser(; slv = slv))
## Increasing risk contribution per asset
rba2 = RiskBudgeting(; r = r,
                     rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nx))),
                     opt = JuMPOptimiser(; slv = slv))
## Optimise all risk budgeting estimators at once
ress = optimise.([rba1, rba2], rd);

#=
View and compute the results.
=#
r = factory(r, pr)
rkcs = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcs = rkcs ./ sum.(rkcs)
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcs)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt])

#=
##### 3.2.4.1 Factor Risk Budgeting

This version allocated risk accross factors.
=#
r = Variance()
## Equal risk contribution per factor (default)
rba1 = RiskBudgeting(; r = r,
                     rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nf))),
                     opt = JuMPOptimiser(; slv = slv))
## Increasing risk contribution per factor
rba2 = RiskBudgeting(; r = r,
                     rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = 1:length(rd.nf))),
                     opt = JuMPOptimiser(; slv = slv))
## Optimise all risk budgeting estimators at once
ress = optimise.([rba1, rba2], rd);

#=
View and compute the results.
=#
r = factory(r, pr)
rkcas = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcas = rkcas ./ sum.(rkcas)
rkcfs = [factor_risk_contribution(r, res.w, pr.X; rd = rd) for res in ress]
rkcfs = rkcfs ./ sum.(rkcfs)
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcas)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt],
             title = "Asset risk contribution")
pretty_table(hcat(DataFrame(:factors => [rd.nf; "Intercept"]),
                  DataFrame(reduce(hcat,
                                   [[[(res.prb.b1 \ res.w); NaN] rkc]
                                    for (res, rkc) in zip(ress, rkcfs)]),
                            ["Eq Risk Weights", "Eq Risk Budget", "Incr Risk Weights",
                             "Incr Risk Budget"])); formatters = [resfmt],
             title = "Factor risk contribution")

#=
#### 3.2.5 Relaxed Risk Budgeting

[`RelaxedRiskBudgeting`]-(@ref) provides a way to allocate risk accross assets or factors according to a user-defined risk budgeting vector, which does not have to be normalised. They are provided in the same way as for [`RiskBudgeting`]-(@ref), it does not accept risk measures as it's only available for the variance, and it will not follow the risk budget as closely.

There are three variants, basic, regularised, and regularised and penalised. Since the asset and factor versions are the same as before we will only show the different relaxed risk budgeting algorithms.
=#

## Basic
rrb1 = RelaxedRiskBudgeting(;
                            rba = AssetRiskBudgeting(;
                                                     rkb = RiskBudget(;
                                                                      val = range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = length(rd.nx)))),
                            alg = BasicRelaxedRiskBudgeting(),
                            opt = JuMPOptimiser(; slv = slv))
## Regularised
rrb2 = RelaxedRiskBudgeting(;
                            rba = AssetRiskBudgeting(;
                                                     rkb = RiskBudget(;
                                                                      val = range(;
                                                                                  start = 1,
                                                                                  stop = 1,
                                                                                  length = length(rd.nx)))),
                            alg = RegularisedRelaxedRiskBudgeting(),
                            opt = JuMPOptimiser(; slv = slv))
## Regularised and penalised, `p` is the penalty factor (default 1)
rrrb3 = RelaxedRiskBudgeting(;
                             rba = AssetRiskBudgeting(;
                                                      rkb = RiskBudget(;
                                                                       val = range(;
                                                                                   start = 1,
                                                                                   stop = 1,
                                                                                   length = length(rd.nx)))),
                             alg = RegularisedPenalisedRelaxedRiskBudgeting(; p = 1),
                             opt = JuMPOptimiser(; slv = slv))
ress = optimise.([rrb1, rrb2, rrrb3], rd);

#=
View and compute the results.
=#
r = Variance()
r = factory(r, pr)
rkcs = [risk_contribution(r, res.w, pr.X) for res in ress]
rkcs = rkcs ./ sum.(rkcs)
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [[res.w rkc] for (res, rkc) in zip(ress, rkcs)]),
                            ["B Weights", "B Budget", "Reg Weights", "Reg Budget",
                             "RegPen Weights", "RegPen Budget"])); formatters = [resfmt])

#=
### 3.3 Clustering optimisers

Clustering based optimisers use the relatedness of the assets and the risks associated with these structures to assign weights based on those risks. Aside from the [`NestedClusters`]-(@ref) estimator, they all use a [`HierarchicalOptimiser`]-(@ref) in much the same way that JuMP-based optimisers use [`JuMPOptimiser`]-(@ref). In this case, the [`HierarchicalOptimiser`]-(@ref) does not require a solver to be specified via the `slv` keyword unless it uses a risk measure that requires one.

#### 3.3.1 Hierachical risk parity

The [`HierarchicalRiskParity`]-(@ref) estimator uses the hierarchical structure of the assets to iteratively partition the assets into smaller and smaller clusters computing the weights as a function of the risk between left and right cluster at each partition level. This accepts the same risk measures as JuMP based optimisers as well as some extra ones for which there are no traditional optimisation formulations.

The [`HierarchicalOptimiser`]-(@ref) is specified via the `opt` keyword and the risk measure can be specified via the `r` keyword which defaults to the [`Variance`](@ref).
=#

## Hierarchical risk parity
hrp = HierarchicalRiskParity()
res = optimise(hrp, rd)

#=
View and compute the results.
=#
pr = res.pr
r = factory(Variance(), pr)
rk, rt, ratio = expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr)
## Display results
pretty_table(DataFrame(:assets => rd.nx, :Weights => res.w); formatters = [resfmt])
pretty_table(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"],
                       :Measure => [rk, rt, ratio]); formatters = [resfmt])

#=
#### 3.3.2 Schur complement hierarchical risk parity

The [`SchurComplementHierarchicalRiskParity`]-(@ref) estimator works similarly to [`HierarchicalRiskParity`]-(@ref), but uses the Schur complement to compute decide whether to include more information into the risk calculation. It is only available for the variance and standard deviation because it relies on the Schur complement. It serves almost as an interpolation between the classic mean variance optimisation and hierarchical risk parity. It has a `params` keyword which allows users to specify an instance or vector of instances of [`SchurComplementParams`]-(@ref) which specify the risk measure, the "interpolation" parameter `gamma ∈ [0, 1]`, and whether the algorithm should be kept monotonic in risk as `gamma` increases (default).

When `gamma` is zero, it reduces to the [`HierarchicalRiskParity`]-(@ref) estimator, the closer it is to one the closer it is to the classic mean variance optimisation. There is no need to specify all these parameters, as they all have default values, this is for demonstration purposes. It's worth noting that there are values of `gamma` for which the Schur augmented matrix is not positive definite as it cannot add any more risk information beyond a certain point, so when wanting `gamma` to be large, one should use [`NonMonotonicSchurComplement`]-(@ref) and make sure to disable the positive definite projection in the `pdm` keyword of [`SchurComplementParams`]-(@ref.
=#
r = Variance()
## Hierachical risk parity
hrp = HierarchicalRiskParity()
## Schur complement hierarchical risk parity converging to the hierarchical risk parity
sch1 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 0,
                                                                            r = r,
                                                                            alg = MonotonicSchurComplement()))
## Mean variance optimisation
mr = MeanRisk(; opt = JuMPOptimiser(; slv = slv))
## Schur complement hierarchical risk parity nearing the mean variance optimisation, no positive definite projection, non-monotonic
sch2 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 1,
                                                                            r = r,
                                                                            pdm = nothing,
                                                                            alg = NonMonotonicSchurComplement()))
## Schur complement hierarchical risk parity nearing the mean variance optimisation
sch3 = SchurComplementHierarchicalRiskParity(;
                                             params = SchurComplementParams(; gamma = 1,
                                                                            r = r,
                                                                            alg = MonotonicSchurComplement()))
ress = optimise.([hrp, sch1, mr, sch2, sch3], rd)

#=
We can compute the statistics and visualise the results of each estimator.
=#
pr = ress[1].pr
r = factory(Variance(), pr)
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)
## Display results
pretty_table(hcat(DataFrame(:assets => rd.nx),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            ["HRP", "gamma = 0", "MVO", "gamma = 1", "gamma = :max"]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            ["HRP", "gamma = 0", "MVO", "gamma = 1", "gamma = :max"]));
             formatters = [resfmt])

#=
#### 3.3.3 Hierachical equal risk contribution

The [`HierarchicalEqualRiskContribution`]-(@ref) estimator uses the hierarchical structure of the assets as well as a clustering quality score to iteratively break up the asset universe into left and right clusters up until the optimal number of clusters according to the score. Each cluster is treated as a synthetic asset for which their risk is computed. The weight of each cluster is computed based on the risk it represents with respect to the cluster on the other side, and this weight is then multiplied by the weights represented by its member assets which were computed based on the risk they represented in porportion to the other assets in the cluster.

The [`HierarchicalOptimiser`]-(@ref) is specified via the `opt` keyword. Since this optimiser breaks up the assets into intra- and inter-cluster optimisations, it's possible to provide inner and outer risk measures via the `ri` and `ro` keywords, which both default to the [`Variance`](@ref). The original formulation used equal weights for the inner risk measure.
=#
using Clustering
## Hierarchical equal risk contribution, equal weights risk inner risk measure, variance outer risk measure
herc = HierarchicalEqualRiskContribution(; ri = EqualRiskMeasure(), ro = Variance())
res = optimise(herc, rd)

#=
We can view the results and verify that indeed all assets within a single cluster have the same weight.
=#
pr = res.pr
r = factory(Variance(), pr)
rk, rt, ratio = expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr)
## Display results
pretty_table(DataFrame(:assets => rd.nx, :cluster => get_clustering_indices(res.clr),
                       :Weights => res.w); formatters = [resfmt])
pretty_table(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"],
                       :Measure => [rk, rt, ratio]); formatters = [resfmt])

#=
#### 3.3.4 Nested clustered optimisation

The [`NestedClustered`]-(@ref) optimiser uses the same idea as the [`HierarchicalEqualRiskContribution`]-(@ref), where the optimisation process is split into inner and outer optimisations using the same scoring system for finding the optimal number of clusters. However, unlike [`HierarchicalEqualRiskContribution`]-(@ref), the intra- and inter-cluster optimisations are completely independent. It is possible to provide any non finite allocation optimisation estimator for the inner and outer estimators independently via the keywods `opti` and `opto` respectively. This means it inherits the requirements for the inner and outer estimators respectively.

It is also possible to optimise the outer estimator by using cross validation via the `cv` keyword. A cross validation prediction is applied to the inner estimators, yielding a predicted returns series for each cluster. The returns vector for each cluster is then taken as the returns vector for a synthetic asset, and the resulting returns matrix is used to optimise the outer estimator. Otherwise the returns are computed directly by multiplying the inner weights by the original returns matrix. The final weights are computed in a similar way to the [`HierarchicalEqualRiskContribution`]-(@ref), multiplying weights of each cluster by their corresponding inner weights.
=#

## Emulating the original `HierarchicalEqualRiskContribution`
nco1 = NestedClustered(; opti = EqualWeighted(),
                       opto = RiskBudgeting(; opt = JuMPOptimiser(; slv = slv)))
## Mean risk for both optimisations
nco2 = NestedClustered(; opti = MeanRisk(; opt = JuMPOptimiser(; slv = slv)),
                       opto = MeanRisk(; opt = JuMPOptimiser(; slv = slv)))
## It's even possible to nest them
nco3 = NestedClustered(;
                       opti = NestedClustered(; opti = HierarchicalEqualRiskContribution(;),
                                              opto = RiskBudgeting(;
                                                                   opt = JuMPOptimiser(;
                                                                                       slv = slv))),
                       opto = NestedClustered(;
                                              opti = RiskBudgeting(;
                                                                   opt = JuMPOptimiser(;
                                                                                       slv = slv)),
                                              opto = MeanRisk(;
                                                              opt = JuMPOptimiser(;
                                                                                  slv = slv))))
## Optimise them all in one go
ress = optimise.([nco1, nco2, nco3], rd)

#=
We can compute some risk characteristics and visualise the results. We can see how the analogous optimisation to the original version of [`HierarchicalEqualRiskContribution`]-(@ref) has a similar behaviour, where all assets within a cluster have the same weight as each other.
=#
pr = ress[1].pr
r = factory(Variance(), pr)
rk_rt_ratio = [expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr) for res in ress]
rk = map(rr -> rr[1], rk_rt_ratio)
rt = map(rr -> rr[2], rk_rt_ratio)
ratio = map(rr -> rr[3], rk_rt_ratio)
## Display results
pretty_table(hcat(DataFrame(:assets => rd.nx,
                            :clusters => get_clustering_indices(ress[1].clr)),
                  DataFrame(reduce(hcat, [res.w for res in ress]),
                            ["EW-RB", "MR-MR", "NC-HERC-RB_NC-RB-MR"]));
             formatters = [resfmt])
pretty_table(hcat(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"]),
                  DataFrame(vcat(rk', rt', ratio'),
                            ["EW-RB", "MR-MR", "NC-HERC-RB_NC-RB-MR"]));
             formatters = [resfmt])

#=
#### 3.3.5 Stacking optimisation

The [`Stacking`]-(@ref) optimiser uses a similar approach to [`NestedClustered`]-(@ref), but instead of using a single inner estimator, it uses a vector of estimators, inhering the requirements of each estimator being used. The inner weights are optimised by each estimator, and outer estimator uses each inner optimisation as a synthetic asset. The returns series used in the outer optimisation can be computed in the same way as [`NestedClustered`]-(@ref), either directly or by using cross validation predictions. The final weights are computed the same way as well.

[`Stacking`]-(@ref) can be used in [`NestedClustered`]-(@ref) and vice-versa.

The keywords for the inner and outer optimisers are the same as [`NestedClustered`]-(@ref).
=#

## Use a few different optimisers
st = Stacking(;
              opti = [MeanRisk(; opt = JuMPOptimiser(; slv = slv)),
                      RiskBudgeting(; opt = JuMPOptimiser(; slv = slv)),
                      InverseVolatility(), HierarchicalEqualRiskContribution()],
              opto = NearOptimalCentering(; opt = JuMPOptimiser(; slv = slv)))
## Optimise
res = optimise(st, rd)

#=
Compute and view the results.
=#
pr = res.pr
r = factory(Variance(), pr)
rk, rt, ratio = expected_risk_ret_ratio(r, ArithmeticReturn(), res.w, pr)
## Display results
pretty_table(DataFrame(:assets => rd.nx, :Weights => res.w); formatters = [resfmt])
pretty_table(DataFrame(:Stat => ["Variance", "Return", "Return/Variance"],
                       :Measure => [rk, rt, ratio]); formatters = [resfmt])
