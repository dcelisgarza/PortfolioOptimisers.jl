#=
```@meta
Description = "Risk contribution in PortfolioOptimisers.jl: bound each asset's or each factor's share of the risk, and compare the result across objectives."
```

# Risk contribution

This page is about how the risk of a portfolio divides between its assets or its factors,
not about how much risk it takes. It covers two jobs:

  - asset risk contribution under the variance measure, where the `rc` field caps what each
    asset contributes and three objective functions run against the same caps;
  - factor risk contribution, where the same kind of cap names factors instead of assets.

!!! tip "When to reach for this"
    Reach for a risk contribution workflow when the weights are not what you want to control.
    It shows you how concentrated the realised risk is, it caps what one asset or one factor
    contributes, and it lets you hold those caps fixed while you change the objective.
=#

using PortfolioOptimisers, PrettyTables

## Format for pretty tables.
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data

We use one year of S&P 500 prices and the factor returns over the same dates. Section 4
needs the factor returns.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))

#=
## 2. Shared optimiser setup

We pass a vector of solver settings. If the first Clarabel setting stalls on this data, the
optimiser tries the next one.
=#

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))]

pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)
sets_asset = UniverseSets(; dict = Dict("nx" => rd.nx))
opt_asset = JuMPOptimiser(; pe = pr, slv = slv, sets = sets_asset)

#=
## 3. Asset risk contribution with `rc` constraints

[`risk_contribution`](@ref) splits the total risk of a portfolio into one contribution per
asset. Risk budgeting asks for a whole budget. The `rc` field of [`Variance`](@ref) instead
caps a contribution directly, and the cell below caps every asset at 20% of the total risk.

The loop then solves the same constraints under three objectives, so you can read how the
objective moves the weights while the caps stay put.
=#

lcs_asset = LinearConstraintEstimator(; val = ["$a <= 0.2" for a in rd.nx])
r_asset = Variance(; rc = lcs_asset)
rf_asset = factory(r_asset, pr)

obj_specs = [(:min_risk, MinimumRisk()), (:max_utility, MaximumUtility()),
             (:max_ratio, MaximumRatio())]

asset_df = DataFrame(; assets = rd.nx)
asset_res = Dict{Symbol, Any}()

for (name, obj) in obj_specs
    res = optimise(MeanRisk(; r = r_asset, obj = obj, opt = opt_asset))
    asset_res[name] = res
    rcs = risk_contribution(rf_asset, res.w, pr.X)
    rcs ./= sum(rcs)
    asset_df[!, Symbol("$(name)_weight")] = res.w
    asset_df[!, Symbol("$(name)_risk")] = rcs
end

pretty_table(asset_df; formatters = [resfmt])

#=
Read the risk columns of the table against the 20% cap. The weights differ from one
objective to the next. The plots below draw one risk profile per objective.
=#

using StatsPlots, GraphRecipes

for (name, _) in obj_specs
    display(plot_risk_contribution(rf_asset, asset_res[name], rd;
                                   title = "Asset RC - $(name)"))
end

#=
## 4. Factor risk contribution optimisation

[`FactorRiskContribution`](@ref) caps a named factor rather than a named asset. It regresses
the asset returns onto the factor returns and computes the contribution of each factor from that
regression. The regression estimator needs the factor returns when the optimiser builds the
problem, so we pass `rd` to [`optimise`](@ref).

The loop below runs the same three objectives against one set of factor constraints. The
constraints cap the share of `VLUE` in the portfolio variance at 74%, keep the share of `QUAL`
at or above -7%, and fix the share of `MTUM` at 9%. A factor can contribute a negative share.
=#

sets = UniverseSets(; dict = Dict("nx" => rd.nf))
lcs = LinearConstraintEstimator(; val = ["VLUE <= 0.74", "QUAL >= -0.07", "MTUM==0.09"])
r_fac = Variance(; rc = lcs)

rf_fac = factory(r_fac, pr)
factor_df = DataFrame(; factor = [rd.nf; "Intercept"])
factor_res = Dict{Symbol, Any}()

for (name, obj) in obj_specs
    res = optimise(FactorRiskContribution(; r = r_fac, obj = obj, opt = opt, sets = sets),
                   rd)
    factor_res[name] = res
    frc_risk = factor_risk_contribution(rf_fac, res.w, pr.X; rd = rd)
    frc_risk ./= sum(frc_risk)
    factor_df[!, Symbol("$(name)_risk")] = frc_risk
end

pretty_table(factor_df; formatters = [resfmt])

#=
The table prints each factor's share of the total, so every column sums to one. The `rc`
constraints bound each factor's share of the portfolio variance. The optimiser states them on a
semidefinite relaxation, in which a matrix variable takes the place of the product of the factor
weights with themselves. When the relaxation is not tight, the shares of the portfolio it returns
miss the targets. On this data only the maximum-ratio column meets all three constraints.
=#

for (name, _) in obj_specs
    display(plot_factor_risk_contribution(rf_fac, factor_res[name], rd;
                                          title = "Factor RC - $(name)"))
end

#=
## Summary

The `rc` field bounds the share of the variance that each asset or factor carries, whatever
the objective.

  - [`Variance`](@ref) with `rc` constraints caps the realised risk contribution of an asset
    or of a factor.
  - [`MeanRisk`](@ref) and [`FactorRiskContribution`](@ref) take the same constraints under
    different objectives, so you can compare the weights and the concentration they give.
  - [`risk_contribution`](@ref) and [`factor_risk_contribution`](@ref) compute the realised
    profile, which you read against the caps you set.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): asset RC (three objectives with per-asset
#src   `rc <= 0.2` caps) and factor RC (three objectives with VLUE/QUAL/MTUM constraints) all
#src   solve. The asset caps clearly bind — under `MinimumRisk`, JNJ and MRK both sit at exactly
#src   20.0% risk contribution while their weights differ (21.9% vs 20.5%).
#src - DOC/ERGO (record-only → risk-contribution rollup): the factor table displays *normalised*
#src   contributions (`frc_risk ./= sum(frc_risk)`), but the `rc` targets are expressed on the
#src   raw contributions, so the equality target is not visible in the output — `MTUM==0.09`
#src   prints as 5.3% under `MinimumRisk`. A reader cannot confirm the constraint from the table.
#src   Either show raw contributions alongside the normalised ones, or add a sentence noting that
#src   the printed figures are renormalised and the constraint lives in raw space.
#src - No solver warnings or plotting deprecations observed.
