#=
# Specialist risk measures: BrownianDistanceVariance and VarianceSkewKurtosis

Some risk measures capture dependence structure or higher-order moment interactions that
variance and CVaR miss. Two such specialist measures are:

  - [`BrownianDistanceVariance`](@ref) — measures non-linear dependence between portfolio
    returns and a reference via the Brownian (distance) covariance framework. It is zero if
    and only if the returns are *statistically independent* of the reference; variance can be
    zero while Brownian Distance Variance is not. It builds a T×T pairwise distance matrix
    so it scales quadratically in observations, not in assets.
  - [`VarianceSkewKurtosis`](@ref) — a composite that combines variance (penalises
    dispersion), negative skewness (penalises asymmetry), and kurtosis (penalises heavy
    tails) into a single objective. It uses large PSD cones so it's best to use a solver that
    supports first-order algorithms such as SCS.

!!! tip "When to reach for this"
    `BrownianDistanceVariance` is useful when you suspect the return distribution has
    non-linear dependence with a factor or benchmark and want that captured in the
    objective. Reach for `VarianceSkewKurtosis` when the third and fourth moments of the
    portfolio return matter — e.g. when you are allocating into assets with fat tails or
    skewed payoffs and standard mean-variance is blind to the shape of the distribution.

!!! warning "Solver compatibility and dataset sizing"
    `BrownianDistanceVariance` builds an O(T²) distance matrix inside the model — the
    number of auxiliary variables grows quadratically with observations. This example uses a
    50-observation slice to keep the model small.

    `VarianceSkewKurtosis` requires **SCS** (or another solver that handles polynomial PSD
    cones). It will fail silently or produce wrong results with a continuous-only solver
    like Clarabel. The high-order prior (`HighOrderPriorEstimator`) must also be used, as
    it pre-computes the coskewness and cokurtosis tensors the risk measure needs. This
    example also uses a 50-observation slice for the same reason.
=#

using PortfolioOptimisers, PrettyTables, DataFrames

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

using CSV, TimeSeries, Clarabel, SCS

#=
## 1. Shared data — 50-observation slice

Both measures in this example use the same short 50-observation slice.
`BrownianDistanceVariance` builds an O(T²) distance matrix, so T is the binding
constraint on model size. `VarianceSkewKurtosis` needs a `HighOrderPriorEstimator`
whose coskewness/cokurtosis computation also scales with T.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 50):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel4, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true))]

opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 2. Brownian Distance Variance
=#

### 2.1 Default formulation

#=
[`BrownianDistanceVariance`](@ref) uses a Norm-1 cone constraint by default
(`NormOneConeBrownianDistanceVariance`). The T×T distance matrix is linearised via
auxiliary variables so Clarabel can handle it as a conic programme.
=#

res_bdvar = optimise(MeanRisk(; r = BrownianDistanceVariance(), opt = opt))

#=
An alternative formulation (`IneqBrownianDistanceVariance`) uses inequality constraints
rather than a cone. On small problems the results are equivalent; on larger problems the
inequality form may be faster because it avoids a large cone.
=#

res_bdvar_ineq = optimise(MeanRisk(;
                                   r = BrownianDistanceVariance(;
                                                                alg2 = IneqBrownianDistanceVariance()),
                                   opt = opt))

#=
### 2.2 Alternative formulations

A second formulation switch controls the **rank constraint** used for the distance matrix:
`QuadRiskExpr` (default) or `RSOCRiskExpr`. The latter uses a rotated second-order cone
and can be faster for very dense problems.
=#

res_bdvar_rsoc = optimise(MeanRisk(; r = BrownianDistanceVariance(; alg1 = RSOCRiskExpr()),
                                   opt = opt))

pretty_table(DataFrame(; :assets => rd.nx, :BDVar_default => res_bdvar.w,
                       :BDVar_ineq => res_bdvar_ineq.w, :BDVar_rsoc => res_bdvar_rsoc.w);
             formatters = [resfmt])

#=
The three formulations produce the same portfolio — the differences are only in how the
conic model is assembled, not in what it optimises.

The composition plot shows that Brownian Distance Variance concentrates into a different
set of names than plain variance minimisation, reflecting its sensitivity to non-linear
dependence rather than just squared-deviation spread.
=#

res_var = optimise(MeanRisk(; r = Variance(), opt = opt))

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_var, res_bdvar], rd)

#=
## 3. VarianceSkewKurtosis

### 3.1 High-order prior and SCS solver

`VarianceSkewKurtosis` requires SCS and a `HighOrderPriorEstimator` prior. We reuse the
same 50-observation slice already loaded in section 1.
=#

pr_ho = prior(HighOrderPriorEstimator(), rd)

scs_slv = Solver(; name = :scs, solver = SCS.Optimizer, settings = "verbose" => false,
                 check_sol = (; allow_local = true, allow_almost = true))
opt_ho = JuMPOptimiser(; pe = pr_ho, slv = scs_slv)

#=
### 3.2 Default composite

[`VarianceSkewKurtosis`](@ref) combines three sub-measures:

  - [`Variance`](@ref) — penalises dispersion.
  - [`Skewness`](@ref) (negative skewness convention) — penalises left-skewed returns.
  - [`Kurtosis`](@ref) — penalises fat tails.

The default scales are 1:1:1. Scaling one component higher makes the objective more
sensitive to that moment.
=#

res_vsk = optimise(MeanRisk(; r = VarianceSkewKurtosis(), opt = opt_ho), rd)
println("VarianceSkewKurtosis retcode: $(res_vsk.retcode)")

#=
### 3.3 Custom component scales

Passing custom [`Skewness`](@ref) and [`Kurtosis`](@ref) with scaled
settings (`MaxRiskMeasureSettings` for skewness and [`RiskMeasureSettings`](@ref) for
kurtosis) lets you control how much each higher-moment penalty
contributes relative to variance.
=#

r_vsk_heavy = VarianceSkewKurtosis(;
                                   sk = Skewness(;
                                                 settings = MaxRiskMeasureSettings(;
                                                                                   scale = 2.0)),
                                   kt = Kurtosis(;
                                                 settings = RiskMeasureSettings(;
                                                                                scale = 2.0)))
res_vsk_heavy = optimise(MeanRisk(; r = r_vsk_heavy, opt = opt_ho), rd)

#=
Compare the two allocations. Heavier skewness/kurtosis penalties push the optimiser further
away from fat-tailed or left-skewed names.
=#

pretty_table(DataFrame(; :assets => rd.nx, :VarianceSkewKurtosis => res_vsk.w,
                       :VSK_heavy_tail => res_vsk_heavy.w); formatters = [resfmt])

#=
### 3.4 Comparison with plain variance (SCS)

We compare the VSK portfolio against a plain minimum-variance portfolio solved with SCS on
the same 50-observation prior, so the only difference is the risk measure.
=#

res_var_scs = optimise(MeanRisk(; r = Variance(), opt = opt_ho))

plot_stacked_bar_composition([res_var_scs, res_vsk, res_vsk_heavy], rd)

#=
## Summary

  - [`BrownianDistanceVariance`](@ref) uses a 50-observation slice with Clarabel.
    It captures non-linear dependence via a quadratic T×T distance matrix; keep T small
    (the O(T²) model size is the limiting factor, not the number of assets).
  - [`VarianceSkewKurtosis`](@ref) must use **SCS** (polynomial PSD cones). Pair it with
    [`HighOrderPriorEstimator`](@ref) and a short observation window to keep the
    higher-moment tensor computation feasible.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env) on the 50-obs slice. All three
#src   `BrownianDistanceVariance` formulations (NormOneCone default, `IneqBrownianDistanceVariance`,
#src   `RSOCRiskExpr`) return the same portfolio, confirming the section-2 prose. Both
#src   `VarianceSkewKurtosis` solves (default scales and heavy skew/kurtosis) return
#src   `OptimisationSuccess` with SCS.
#src - The 50-obs slice and the SCS requirement are load-bearing, not stylistic: VSK needs large PSD
#src   cones (Clarabel cannot solve them directly), and BDVar's O(T²) distance matrix makes T the binding size. Both
#src   are flagged in the opening `!!! warning` so a reader who swaps in Clarabel or a 252-obs
#src   slice knows why it breaks. No solver warnings or plotting deprecations observed.
#src - COSMETIC: the `RSOCRiskExpr` column prints `-0.0 %` for zero weights (signed-zero from the
#src   rotated-cone formulation). Harmless in the table; not worth a fix.
