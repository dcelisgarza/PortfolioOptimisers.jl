#=
```@meta
Description = "Ordered weighted average (OWA) risk measures in PortfolioOptimisers.jl: weight sorted returns to build tail and dispersion measures."
```

# OWA risk measures

An ordered weighted average (OWA) risk measure is a weighted sum of the portfolio returns sorted
from worst to best. The weights of that sum, the OWA weights, decide which part of the
distribution counts most. Weights on the few worst returns give a tail measure like CVaR, and
weights spread over the whole distribution give a measure of dispersion.

This page covers three points about OWA measures.

  1. CVaR, the Gini mean difference, the tail Gini, the worst realisation and the range each
     have a closed-form weight vector. You can also pass your own `w` to
     [`OrderedWeightsArray`](@ref).
  2. One family of OWA weights gives the *L-moments* of the return distribution. They describe
     the spread and the tails of the distribution without a model of its shape.
  3. The exact formulation of an OWA measure is a linear programme in the portfolio weights,
     and no OWA measure needs a covariance matrix. `MeanRisk`, `RiskBudgeting` and the other
     optimisers that take a risk measure accept an OWA measure.

!!! tip "When to reach for this"
    Reach for an OWA risk measure when you want a tail or dispersion measure that needs no
    covariance matrix and covers more cases than CVaR. It suits returns that are far from
    normal, where the variance or one quantile misses part of the distribution and you want
    the measure to depend on its whole shape.

!!! note "Exact and approximate formulations"
    [`OrderedWeightsArray`](@ref) has two formulations. The exact one,
    `ExactOrderedWeightsArray`, is a linear programme that adds `T × T` constraints for `T`
    observations, so it grows fast with the length of the sample. The approximate one,
    `ApproxOrderedWeightsArray`, is the default. It replaces those constraints with one
    p-norm constraint for each entry of `p`, and it gives an upper bound on the exact risk
    that is close when the weights change almost linearly with rank. Every optimisation on
    this page uses the approximate formulation.
=#

using PortfolioOptimisers, PrettyTables, DataFrames

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and shared setup
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
T = size(rd.X, 1)

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
              settings = Dict("verbose" => false, "max_step_fraction" => 0.6,
                              "max_iter" => 1500, "tol_gap_abs" => 1e-4,
                              "tol_gap_rel" => 1e-4, "tol_ktratio" => 1e-3,
                              "tol_feas" => 1e-4, "tol_infeas_abs" => 1e-4,
                              "tol_infeas_rel" => 1e-4, "reduced_tol_gap_abs" => 1e-4,
                              "reduced_tol_gap_rel" => 1e-4, "reduced_tol_ktratio" => 1e-3,
                              "reduced_tol_feas" => 1e-4, "reduced_tol_infeas_abs" => 1e-4,
                              "reduced_tol_infeas_rel" => 1e-4),
              check_sol = (; allow_local = true, allow_almost = true))]

pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 2. Closed-form OWA weight vectors

The library has one function for the OWA weights of each classical case. Each function takes
`T`, the number of observations, and returns a vector of length `T`. We pass each vector as `w`
to [`OrderedWeightsArray`](@ref) to build a risk measure.

| Function | Measure | What it measures |
| -------- | ------- | ---------------- |
| `owa_gmd(T)` | Gini mean difference | The mean absolute difference over all pairs of returns, a dispersion measure |
| `owa_cvar(T)` | CVaR | The CVaR at 5 %, written as an OWA measure |
| `owa_tg(T)` | Tail Gini | The Gini spread within the worst tail, which depends on the shape of the tail as well as its level |
| `owa_tgrg(T)` | Tail Gini range | The tail Gini of the worst returns and of the best returns together, a measure of both tails |
| `owa_wr(T)` | Worst realisation | The negative of the worst return, so only the worst day counts |
| `owa_rg(T)` | Range | The best return minus the worst return |
| `owa_cvarrg(T)` | CVaR range | The mean of the best 5 % of returns minus the mean of the worst 5 % |
| `owa_l_moment_crm(T)` | L-moment CRM | A convex risk measure built from the L-moments of order 2 to `k`, with `k = 2` by default |
=#

r_gmd = OrderedWeightsArray(; w = owa_gmd(T))
r_cvar = OrderedWeightsArray(; w = owa_cvar(T))
r_tg = OrderedWeightsArray(; w = owa_tg(T))
r_tgrg = OrderedWeightsArray(; w = owa_tgrg(T))
r_wr = OrderedWeightsArray(; w = owa_wr(T))
r_rg = OrderedWeightsArray(; w = owa_rg(T))
r_cvarrg = OrderedWeightsArray(; w = owa_cvarrg(T))
r_lcrm = OrderedWeightsArray(; w = owa_l_moment_crm(T; k = 5))

#=
## 3. Minimising each OWA risk measure

We build a minimum-risk `MeanRisk` portfolio for each OWA measure and print the weights. Every
measure uses the default `ApproxOrderedWeightsArray`, so Clarabel solves each one as a conic
problem.
=#

rs = [r_gmd, r_cvar, r_tg, r_tgrg, r_wr, r_rg, r_cvarrg, r_lcrm]
names_r = ["GMD", "CVaR", "TailGini", "TailGiniRange", "WorstReal", "Range", "CVaRRange",
           "L-moment"]

results = [optimise(MeanRisk(; r = r, opt = opt)) for r in rs]

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in results]...),
                       [:assets; Symbol.(names_r)...]); formatters = [resfmt])

#=
Every portfolio minimises a risk, but each measure counts a different part of the distribution,
so the portfolios differ.

  - `GMD` weights every pair of days, so every day counts.
  - `CVaR` uses only the worst 5 % of days and ignores smaller losses.
  - `TailGini` also uses the worst days, and it responds to the Gini spread within the tail
    as well as to its level.
  - `WorstReal` depends on one day and `Range` on two days, so the solver chooses the weights that
    improve those days alone.
  - `L-moment` uses the L-moments of order 2 to 5, `k = 5`. The second L-moment alone has half
    the GMD weights, and the higher orders add weight on the tails. On this sample the
    portfolio holds the same assets as `GMD`, with different weights. Section 7 changes
    the mix of the orders.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition(results, rd)

#=
## 4. Default approximate formulation

[`OrderedWeightsArray`](@ref) with no arguments uses the Gini mean difference weights,
`owa_gmd`, and the formulation `ApproxOrderedWeightsArray` with `p = [2, 3, 4, 10, 50]`. This is
the GMD measure of section 3. We print the largest absolute difference between its weights and
the weights of the `GMD` column. A difference of zero means that the two portfolios are the
same.

Use the approximate formulation for a long sample. Use `ExactOrderedWeightsArray` when the weights
are far from linear in rank, where the approximation is less close.
=#

r_approx = OrderedWeightsArray()
res_approx = optimise(MeanRisk(; r = r_approx, opt = opt))
println("Largest weight difference from the GMD column: ",
        maximum(abs, res_approx.w .- results[1].w))

#=
## 5. OWA range measures for both tails

[`OrderedWeightsArrayRange`](@ref) builds a *range* measure from two OWA weight vectors, `w1`
for the worst returns and `w2` for the best returns. You write `w2` in the same orientation as
`w1`, and the constructor reverses it. The measure then weights the sorted returns by
`w1 - w2`, which covers the worst returns through `w1` and the best through `w2`. Pass `rev = true` if
your `w2` is already reversed. This is the OWA form of a risk measure for both tails.

The default range takes `owa_tg` for both tails, which gives the same weights as `owa_tgrg`.
Its portfolio still differs from the `TailGiniRange` column of section 3. The approximate
formulation of a range approximates each tail on its own and adds the two, and the
`TailGiniRange` measure of section 3 approximates the combined weights in one block. You can
pass your own `w1` and `w2` to choose what each side measures. We also build a range
from `owa_cvar` and `owa_wr`.
=#

## Default range: tail Gini losses vs tail Gini gains.
r_range_default = OrderedWeightsArrayRange()

## Custom range: CVaR of the losses plus the largest return.
T_obs = T
r_range_custom = OrderedWeightsArrayRange(; w1 = owa_cvar(T_obs), w2 = owa_wr(T_obs))

res_range_d = optimise(MeanRisk(; r = r_range_default, opt = opt))
res_range_c = optimise(MeanRisk(; r = r_range_custom, opt = opt))

pretty_table(DataFrame(; :assets => rd.nx, :TailGiniRange => res_range_d.w,
                       :CVaR_plus_MaxReturn => res_range_c.w); formatters = [resfmt])

#=
## 6. Maximum risk-adjusted ratio with an OWA measure

An OWA measure can also be the risk in the ratio objective. We maximise the return, net of the
risk-free rate, per unit of Gini mean difference.
=#

rf = 4.2 / 100 / 252
res_ratio = optimise(MeanRisk(; r = r_gmd, obj = MaximumRatio(; rf = rf), opt = opt))
println("GMD ratio portfolio, max weight: $(round(maximum(res_ratio.w)*100; digits=2)) %")
pretty_table(DataFrame(; :assets => rd.nx, :weight => res_ratio.w); formatters = [resfmt])

#=
## 7. The risk aversion `g` of the L-moment CRM

[`NormalisedConstantRelativeRiskAversion`](@ref) builds the weights of the L-moment CRM from a
risk aversion `g`, with `0 < g < 1`. We use the L-moments of order 2 to 5, `k = 5`. A larger
`g` gives the higher-order L-moments more weight, and the OWA weights then put more of their
mass on the worst returns. A smaller `g` brings the measure closer to the second L-moment,
which is half the Gini mean difference.

We minimise the measure for `g` equal to 0.25, 0.5 and 0.75. The default is `g = 0.5`, which
is the `L-moment` measure of section 3, so the `g=0.5` column repeats that column.
=#

gs = [0.25, 0.5, 0.75]
lcrm_results = map(gs) do g
    w_lcrm = owa_l_moment_crm(T, NormalisedConstantRelativeRiskAversion(; g = g); k = 5)
    r_lcrm_g = OrderedWeightsArray(; w = w_lcrm)
    return optimise(MeanRisk(; r = r_lcrm_g, opt = opt))
end

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in lcrm_results]...),
                       [:assets, Symbol.("g=" .* string.(gs))...]); formatters = [resfmt])

#=
As `g` rises, the portfolio moves weight from PEP to JNJ, and every other weight changes by
less than one percentage point.
=#

## The risk-aversion sweep, side by side, from the smallest `g` (left) to the largest (right).
plot_stacked_bar_composition(lcrm_results, rd)

#=
## Summary

  - The closed-form functions, such as `owa_gmd`, `owa_tg` and `owa_cvar`, return a weight
    vector that you pass to [`OrderedWeightsArray`](@ref). Building the weights needs no
    solver.
  - `owa_l_moment_crm` with `NormalisedConstantRelativeRiskAversion` sets the weight on the
    worst returns through `g`. With its default `k = 2` it gives the Gini mean difference
    portfolio.
  - `OrderedWeightsArrayRange` measures both tails at once.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): all eight closed-form OWA measures, the
#src   default `ApproxOrderedWeightsArray`, both range variants, the `MaximumRatio` objective,
#src   and the L-moment `g`-sweep solve with Clarabel on the 252-obs / 20-asset slice. No
#src   warnings or deprecations from the solve or from `plot_stacked_bar_composition`.
#src - FIXED (this session): the page previously ended on the L-moment table (sections 4–7 were
#src   table-only). Added a closing `plot_stacked_bar_composition(lcrm_results, rd)` so the page
#src   ends on a visualisation per the ADR-0014 authoring standard.
#src - OBSERVED (not a bug): `MaximumRatio` with `owa_gmd` collapses to ~67% in a single name
#src   (MRK). A dispersion measure in the ratio denominator on this small slice concentrates
#src   hard; worth a sentence in future if it surprises readers, but it is the expected
#src   risk-adjusted-return solution, not a solver artefact.
