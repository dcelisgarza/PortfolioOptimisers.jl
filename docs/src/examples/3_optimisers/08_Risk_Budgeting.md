The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/3_optimisers/05_Risk_Budgeting.jl"
```

# Risk budgeting

[`RiskBudgeting`](@ref) takes a different stance from [`MeanRisk`](@ref). Instead of trading
expected return against risk through an objective function, it allocates *risk itself*: it
finds the portfolio whose per-asset (or per-factor) risk contributions match a user-supplied
budget as closely as possible. There is no objective to maximise — the budget *is* the goal.

The classic special case is the **equal risk contribution (ERC)** portfolio, where every
asset contributes the same share of total risk. Risk budgeting generalises it to any budget
vector, and to risk measured by any of the risk measures [`MeanRisk`](@ref) supports.

!!! tip "When to reach for this"
    Reach for risk budgeting when you care about *how risk is distributed* rather than about
    a return/risk trade-off — diversifying risk rather than capital, avoiding the
    concentration that minimum-variance portfolios are prone to, or expressing a conviction
    as "this sleeve should carry 30% of the risk". If you instead want the best return for a
    given risk budget, use [`MeanRisk`](@ref).

````@example 05_Risk_Budgeting
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

We use one year of S&P 500 constituents, and (for the factor section) the factor returns.

````@example 05_Risk_Budgeting
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
````

Since every optimisation below shares the same data, we precompute the prior statistics once
with [`EmpiricalPrior`](@ref) and reuse them, rather than recomputing them on every call.

````@example 05_Risk_Budgeting
using Clarabel
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
pr = prior(EmpiricalPrior(), rd)
opt = JuMPOptimiser(; pe = pr, slv = slv)
````

We budget *variance* risk throughout this example, but any [`MeanRisk`](@ref)-compatible risk
measure works — the risk being budgeted is whatever risk measure you pass.

````@example 05_Risk_Budgeting
r = Variance()
N = length(rd.nx)
````

## 2. Asset risk budgeting

[`AssetRiskBudgeting`](@ref) allocates risk across assets. The budget is supplied through the
`rkb` keyword as a [`RiskBudget`](@ref); the vector does not need to be normalised. The
`alg` keyword selects the formulation — [`LogRiskBudgeting`](@ref) (a log-barrier, the
default) or [`MixedIntegerRiskBudgeting`](@ref) (which needs a mixed-integer solver).

Two budgets: an **equal** budget (the ERC portfolio) and a **linearly increasing** budget
that asks later assets to carry progressively more of the risk.

````@example 05_Risk_Budgeting
# Equal risk contribution across assets.
rb_eq = RiskBudgeting(; r = r, opt = opt,
                      rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, N)),
                                               alg = LogRiskBudgeting()))
# Linearly increasing risk budget across assets.
rb_inc = RiskBudgeting(; r = r, opt = opt,
                       rba = AssetRiskBudgeting(; rkb = RiskBudget(; val = 1:N),
                                                alg = LogRiskBudgeting()))

# Optimise both at once with broadcasting (the prior is precomputed, so no data is needed).
res_eq, res_inc = optimise(rb_eq), optimise(rb_inc)
````

To verify the budgets were met, we compute the realised risk contributions. The risk measure
must be parametrised with the prior covariance via [`factory`](@ref) before evaluating it.

````@example 05_Risk_Budgeting
rf = factory(r, pr)
rc_eq = risk_contribution(rf, res_eq.w, pr.X);
rc_eq ./= sum(rc_eq);
rc_inc = risk_contribution(rf, res_inc.w, pr.X);
rc_inc ./= sum(rc_inc);

pretty_table(DataFrame(; :assets => rd.nx, Symbol("Eq weight") => res_eq.w,
                       Symbol("Eq risk") => rc_eq, Symbol("Incr weight") => res_inc.w,
                       Symbol("Incr risk") => rc_inc); formatters = [resfmt])
````

The equal-budget portfolio puts an identical $1/N$ share of variance on every asset, while
the increasing-budget portfolio"s risk contributions rise monotonically across the assets —
exactly the budgets we asked for.

The risk-contribution bar plot makes the contrast immediate.

````@example 05_Risk_Budgeting
using StatsPlots, GraphRecipes

plot_risk_contribution(rf, res_eq, rd)
````

And the increasing-budget portfolio:

````@example 05_Risk_Budgeting
plot_risk_contribution(rf, res_inc, rd)
````

## 3. Relaxed risk budgeting

[`RelaxedRiskBudgeting`](@ref) (RRB) replaces the non-convex risk-parity constraint with a
second-order-cone relaxation. It needs neither a logarithm nor integer variables, so it is
cheaper to solve — useful at scale. It is variance-specific (the SOC is built on the Cholesky
factor of the covariance), so it takes no `r`. Three variants trade exactness for
regularisation: [`BasicRelaxedRiskBudgeting`](@ref),
[`RegularisedRelaxedRiskBudgeting`](@ref), and
[`RegularisedPenalisedRelaxedRiskBudgeting`](@ref).

Being a *relaxation*, RRB does not adhere to the target risk budget as tightly as the exact
log-barrier or mixed-integer formulations of section 2 — in pathological cases (ill-conditioned
covariance, extreme budgets) the realised contributions can deviate noticeably. In exchange,
the convex SOC formulation composes cleanly with additional constraints, making it the
friendlier choice when the risk budget is one objective among several rather than a hard
requirement. Reach for [`RiskBudgeting`](@ref) when strict adherence is essential.

````@example 05_Risk_Budgeting
rba_eq = AssetRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, N)))
rrb_basic = RelaxedRiskBudgeting(; opt = opt, rba = rba_eq,
                                 alg = BasicRelaxedRiskBudgeting())
rrb_reg = RelaxedRiskBudgeting(; opt = opt, rba = rba_eq,
                               alg = RegularisedRelaxedRiskBudgeting())
res_b, res_r = optimise(rrb_basic), optimise(rrb_reg)
````

Comparing the relaxed solutions against the exact log-barrier ERC of section 2 shows the
price of the relaxation: on this dataset the relaxed portfolios are noticeably more
concentrated than the exact ERC, so the realised risk contributions spread away from the
flat $1/N$ target. The relaxation buys tractability, not an exact risk-parity solution —
check the realised contributions when you use it.

````@example 05_Risk_Budgeting
rc_b = risk_contribution(rf, res_b.w, pr.X);
rc_b ./= sum(rc_b);
rc_r = risk_contribution(rf, res_r.w, pr.X);
rc_r ./= sum(rc_r);

pretty_table(DataFrame(; :assets => rd.nx, Symbol("Log ERC risk") => rc_eq,
                       Symbol("Basic RRB risk") => rc_b,
                       Symbol("Regularised RRB risk") => rc_r); formatters = [resfmt])
````

## 4. Factor risk budgeting

[`FactorRiskBudgeting`](@ref) allocates risk across *factors* rather than assets, via a
regression of asset returns onto factor returns. It needs the factor returns, so we rebuild
the data with a factor matrix and use a [`FactorPrior`](@ref).

````@example 05_Risk_Budgeting
F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rdf = prices_to_returns(X, F)
prf = prior(FactorPrior(), rdf)
optf = JuMPOptimiser(; pe = prf, slv = slv)
Nf = length(rdf.nf)

# Equal risk contribution across factors.
frb = RiskBudgeting(; r = Variance(), opt = optf,
                    rba = FactorRiskBudgeting(; rkb = RiskBudget(; val = fill(1.0, Nf))))
````

Because `re` here is a regression *estimator* ([`StepwiseRegression`](@ref) by default), the
factor model has to be fit while building the model, so the returns data must be passed to
[`optimise`](@ref) even though the prior is precomputed. (If you instead pass a precomputed
[`Regression`](@ref) *result* as `re`, no data is needed — and a clear error tells you if
you missed passing `rd` when it's required.)

````@example 05_Risk_Budgeting
res_frb = optimise(frb, rdf)
````

The factor risk contributions (the trailing entry is the intercept/idiosyncratic term)
cluster near the equal $1/N_f$ target across the five factors.

````@example 05_Risk_Budgeting
rfk = factory(Variance(), prf)
frc = factor_risk_contribution(rfk, res_frb.w, prf.X; rd = rdf)
frc ./= sum(frc)

pretty_table(DataFrame(; :factor => [rdf.nf; "Intercept"], :risk => frc);
             formatters = [resfmt])

plot_factor_risk_contribution(rfk, res_frb, rdf)
````

## Summary

Risk budgeting targets a *distribution of risk* rather than a return/risk trade-off:

- [`AssetRiskBudgeting`](@ref) spreads risk across assets — equal budgets give the ERC
    portfolio, arbitrary budgets express convictions about where risk should sit.
- [`RelaxedRiskBudgeting`](@ref) is the cheaper convex alternative; verify the realised
    contributions, as the relaxation need not reproduce exact risk parity.
- [`FactorRiskBudgeting`](@ref) budgets risk across factors instead of assets, at the cost
    of needing the returns data at optimise time.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
