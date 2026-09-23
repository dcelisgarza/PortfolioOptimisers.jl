#=
```@meta
Description = "Risk budgeting in PortfolioOptimisers.jl: match per-asset or per-factor risk contributions to a budget, with equal risk contribution as the special case."
```

# Risk budgeting

[`RiskBudgeting`](@ref) takes a different stance from [`MeanRisk`](@ref). It does not trade
expected return against risk through an objective function. It divides the risk itself, and
finds the portfolio whose per-asset or per-factor risk contributions come as close to a budget
you supply as they can.

The best-known special case is the equal risk contribution portfolio, or ERC, where every
asset carries the same share of total risk. Risk budgeting extends it to any budget vector,
and to risk measured by any of the risk measures [`MeanRisk`](@ref) accepts.

!!! tip "When to reach for this"
    Reach for risk budgeting when you care about how the risk is divided rather than about the
    trade-off between return and risk. It spreads risk rather than capital, so it avoids the
    concentration a minimum-variance portfolio often shows, and it lets you state a view such as
    "this group of assets carries 30% of the risk". If you want the best return for a given
    amount of risk instead, use [`MeanRisk`](@ref).
=#

using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. Data

We use one year of S&P 500 constituents. Section 4 adds the factor returns.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
Every optimisation below reads the same data, so we compute the prior statistics once with
[`EmpiricalPrior`](@ref) and hand them to each optimiser.
=#

using Clarabel
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
We budget variance throughout this example. Any risk measure [`MeanRisk`](@ref) accepts works
here, and the budget divides whichever measure you pass.
=#

r = Variance()
N = length(rd.nx)

#=
## 2. Asset risk budgeting

[`AssetRiskBudgeting`](@ref) divides risk across assets. You pass the budget through the
`rkb` keyword as a [`RiskBudget`](@ref), and the vector does not have to sum to one. The
`alg` keyword picks the formulation. [`LogRiskBudgeting`](@ref) is a log-barrier and the
default, and [`MixedIntegerRiskBudgeting`](@ref) needs a mixed-integer solver.

We build two budgets. The equal budget gives the ERC portfolio. The linearly
increasing budget asks each asset to carry more of the risk than the asset before it.
=#

## Equal risk contribution across assets.
rb_eq = RiskBudgeting(; r = r, opt = opt,
                      rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, N)),
                                               alg = LogRiskBudgeting()))
## Linearly increasing risk budget across assets.
rb_inc = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:N),
                                                alg = LogRiskBudgeting()))

## Optimise both. The prior is precomputed, so no data is needed.
res_eq, res_inc = optimise(rb_eq), optimise(rb_inc)

#=
We compute the realised risk contributions, which you read against the budgets we asked for.
[`factory`](@ref) first gives the risk measure the covariance of the prior.
=#

rf = factory(r, pr)
rc_eq = risk_contribution(rf, res_eq.w, pr.X);
rc_eq ./= sum(rc_eq);
rc_inc = risk_contribution(rf, res_inc.w, pr.X);
rc_inc ./= sum(rc_inc);

pretty_table(DataFrame(; :assets => rd.nx, Symbol("Eq weight") => res_eq.w,
                       Symbol("Eq risk") => rc_eq, Symbol("Incr weight") => res_inc.w,
                       Symbol("Incr risk") => rc_inc); formatters = [resfmt])

#=
Read the two risk columns of the table. The equal-budget portfolio carries the same $1/N$
share of variance on every asset. The risk contributions of the increasing-budget portfolio
rise from the first asset to the last.

The bar plot below draws the risk contributions of the equal-budget portfolio.
=#

using StatsPlots, GraphRecipes

plot_risk_contribution(rf, res_eq, rd)

#=
We draw the same plot for the increasing-budget portfolio.
=#

plot_risk_contribution(rf, res_inc, rd)

#=
## 3. Relaxed risk budgeting

[`RelaxedRiskBudgeting`](@ref) (RRB) replaces the non-convex risk-parity constraint with a
second-order cone relaxation. It uses no logarithm and no integer variable, so a solver
handles it faster, which matters on a large universe. It builds the cone on the Cholesky
factor of the covariance, so it works on variance alone and takes no `r`. Three algorithms trade
exactness for regularisation: [`BasicRelaxedRiskBudgeting`](@ref),
[`RegularisedRelaxedRiskBudgeting`](@ref), and
[`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).

A relaxation does not hold the target budget as tightly as the exact log-barrier and
mixed-integer formulations of section 2. Where the covariance is ill-conditioned or the budget
is extreme, the realised contributions can sit well away from the target. In exchange the
problem stays convex, so further constraints add to it directly. That suits a risk budget that
is one goal among several. Use [`RiskBudgeting`](@ref) when the budget has to hold.
=#

rba_eq = AssetRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, N)))
rrb_basic = RelaxedRiskBudgeting(; opt = opt, rba = rba_eq,
                                 alg = BasicRelaxedRiskBudgeting())
rrb_reg = RelaxedRiskBudgeting(; opt = opt, rba = rba_eq,
                               alg = RegularisedRelaxedRiskBudgeting())
res_b, res_r = optimise(rrb_basic), optimise(rrb_reg)

#=
We print the realised contributions of the two relaxed portfolios next to the exact
log-barrier ERC of section 2. On this data the risk contributions of the relaxed portfolios move
away from the flat $1/N$ target, and two assets, JNJ and MRK, carry the largest shares. Check the
contributions whenever you use a relaxed form.
=#

rc_b = risk_contribution(rf, res_b.w, pr.X);
rc_b ./= sum(rc_b);
rc_r = risk_contribution(rf, res_r.w, pr.X);
rc_r ./= sum(rc_r);

pretty_table(DataFrame(; :assets => rd.nx, Symbol("Log ERC risk") => rc_eq,
                       Symbol("Basic RRB risk") => rc_b,
                       Symbol("Regularised RRB risk") => rc_r); formatters = [resfmt])

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - RESOLVED (this session): the RRB concentration vs log RB is NOT a bug. The SOC
#src   formulation is a *relaxation*, so it does not reproduce exact risk parity; in
#src   pathological cases (ill-conditioned covariance, extreme budgets) the realised
#src   contributions deviate noticeably from the flat target. This is now documented in
#src   the section 3 prose and in the `RelaxedRiskBudgeting` docstring (# Notes). The
#src   trade-off is intentional: RRB stays convex and composes with extra constraints,
#src   whereas log-barrier/MIP RB adhere strictly. No formulation fix needed.
#src - ERGO: `AssetRiskBudgeting` takes the formulation via `alg` (Log/MixedInteger) but
#src   `FactorRiskBudgeting` has no `alg` field (fields are `re, rkb, sets, flag`). The
#src   asymmetry is surprising; a user who learns asset RB cannot transfer the `alg` habit.
#src - RESOLVED (this session): factor risk budgeting/contribution with a regression
#src   *estimator* requires `rd` at `optimise` (it fits the regression from `rd.X`/`rd.F`),
#src   whereas a precomputed `Regression` *result* needs none. This used to throw a cryptic
#src   `IsNothingError` deep inside `regression`; added a contextual `@argcheck` in
#src   `set_factor_risk_contribution_constraints!` (src/17_Optimisation/05_JuMP/05_FactorRiskContribution.jl)
#src   that explains the estimator-vs-result contract and points to both fixes. Still worth a
#src   regression test asserting the friendly error fires.
#src - PLOT GAP: there is no single plot that overlays target-vs-realised risk contributions
#src   for several portfolios side by side — would make the RRB deviation above obvious.

#=
## 4. Factor risk budgeting

[`FactorRiskBudgeting`](@ref) divides risk across factors rather than assets. It regresses the
asset returns onto the factor returns. It needs the factor returns, and it does not need a
factor prior, so an [`EmpiricalPrior`](@ref) is enough.
=#

F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rdf = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))
prf = prior(EmpiricalPrior(), rdf)
optf = JuMPOptimiser(; pe = prf, slv = slv)
Nf = length(rdf.nf)

## Equal risk contribution across factors.
frb = RiskBudgeting(; r = Variance(), opt = optf,
                    rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, Nf))))

#=
`re` here is a regression estimator, [`StepwiseRegression`](@ref) by default. An estimator
fits the factor model as the optimiser builds the problem, so you pass the returns data to
[`optimise`](@ref) even though the prior is already computed. Pass a [`Regression`](@ref)
result as `re` instead and the data is not needed. If you leave `rd` out where it is needed,
the error names what is missing.
=#

res_frb = optimise(frb, rdf)

#=
We print the factor risk contributions. The last row is the off-factor contribution, the risk
that comes from the part of the weights with no exposure to any factor. Read the five factor
rows against the equal $1/N_f$ target.
=#

rfk = factory(Variance(), prf)
frc = factor_risk_contribution(rfk, res_frb.w, prf.X; rd = rdf)
frc ./= sum(frc)

pretty_table(DataFrame(; :factor => [rdf.nf; "Off-factor"], :risk => frc);
             formatters = [resfmt])

plot_factor_risk_contribution(rfk, res_frb, rdf)

#=
## Summary

Risk budgeting sets how the risk is divided, not the trade-off between return and risk.

  - [`AssetRiskBudgeting`](@ref) divides risk across assets. An equal budget gives the ERC
    portfolio, and any other budget states how you want the risk divided.
  - [`RelaxedRiskBudgeting`](@ref) is convex and cheaper to solve. Read the realised
    contributions, because a relaxation need not reach exact risk parity.
  - [`FactorRiskBudgeting`](@ref) divides risk across factors instead of assets, and it needs
    the returns data at optimise time.
=#
