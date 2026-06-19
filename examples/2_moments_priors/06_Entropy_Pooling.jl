#=
# Entropy pooling

[Black–Litterman](05_Black_Litterman.md) blends views into the *mean* through a Gaussian
update. **Entropy pooling** is more general in two ways. First, it expresses views as
constraints on *any* moment — mean, variance, CVaR, skewness, kurtosis, even individual
covariances and correlations. Second, it does not assume normality: it reweights the empirical
scenarios so that the new distribution satisfies your views while staying as close as possible
(in relative entropy / Kullback–Leibler divergence) to the original. The output is a fully
reweighted prior, not just a shifted mean.

This is the second page of the view-prior arc — [Black–Litterman](05_Black_Litterman.md) came
first, and [Opinion Pooling](07_Opinion_Pooling.md) follows, combining several entropy-pooling
views into one.

In `PortfolioOptimisers`, [`EntropyPoolingPrior`](@ref) accepts a separate
[`LinearConstraintEstimator`](@ref) per quantity. Mind the naming: `mu_views` is the mean,
`sigma_views` is the **variance**, `var_views` is the **Value at Risk** and `cvar_views` the
**Conditional VaR** (tail-risk views), `sk_views`/`kt_views` are skewness/kurtosis, and
`cov_views`/`rho_views` target covariances/correlations. Each is a list of string constraints
over the [`AssetSets`](@ref) names.

!!! tip "When to reach for this"
    Reach for entropy pooling when your views are richer than "the mean will be x": views on
    volatility, tail risk (CVaR), skewness, or the correlation between two assets, possibly
    several at once. It is also the right tool when you distrust the normality assumption baked
    into Black–Litterman, since it reweights the empirical scenarios directly. For a simple
    mean-only view, Black–Litterman is lighter; to *combine* several entropy-pooling opinions,
    see Opinion Pooling.
=#

using PortfolioOptimisers, PrettyTables

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=4)) %" : v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. Naming assets and groups

As with Black–Litterman, views reference assets and groups by name through an
[`AssetSets`](@ref).
=#

sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                             "energy" => ["CVX"]))

#=
## 3. Views on several moments

Entropy-pooling views are also plain strings, but they can target different quantities. Here we
state a **mean** view (Apple returns 8 bps) via `mu_views`, a **relative mean** view (tech
outperforms energy), and a **variance** view (pin Apple's variance) via `sigma_views`. The
comparison operators a view accepts depend on the moment: `mu_views`, `sigma_views`,
`sk_views`, `kt_views`, `cov_views` and `rho_views` take `==`, `>=` and `<=`; `var_views` (VaR)
takes only `==` and `>=`; and `cvar_views` (CVaR) takes only `==`. An unsupported operator
raises a `ParseError` listing the ones allowed for that view.
=#

mu_views = LinearConstraintEstimator(; val = ["AAPL == 0.0008", "tech >= energy"])
sigma_views = LinearConstraintEstimator(; val = ["AAPL == 0.0003"])

ep = EntropyPoolingPrior(; sets = sets, mu_views = mu_views, sigma_views = sigma_views)

#=
## 4. Prior vs reweighted posterior

We compute the entropy-pooling posterior and compare both the mean **and** the variance of
Apple against the plain empirical prior — the mean view lifts the expected return while the
variance view tightens the dispersion, exactly as instructed.
=#

pr_ep = prior(ep, rd)
pr_emp = prior(EmpiricalPrior(), rd)

i_aapl = findfirst(==("AAPL"), rd.nx)
pretty_table(DataFrame(["moment" => ["mean (AAPL)", "variance (AAPL)"],
                        "Empirical" => [pr_emp.mu[i_aapl], pr_emp.sigma[i_aapl, i_aapl]],
                        "Entropy pooling" =>
                            [pr_ep.mu[i_aapl], pr_ep.sigma[i_aapl, i_aapl]]]);
             formatters = [mmtfmt],
             title = "Apple moments: empirical vs entropy-pooling view")

#=
The full expected-returns vectors, side by side.
=#

pretty_table(DataFrame(["Assets" => rd.nx, "Empirical" => pr_emp.mu,
                        "Entropy pooling" => pr_ep.mu]); formatters = [mmtfmt],
             title = "Expected returns: empirical vs entropy-pooling posterior")

# Entropy-pooling posterior expected returns.
using StatsPlots, GraphRecipes
plot_mu(pr_ep, rd.nx)

#=
## 5. Why it matters: views change the portfolio

Feeding the reweighted prior to a return-seeking optimiser tilts the portfolio toward the
view-favoured assets, just as Black–Litterman did — but here the *whole distribution*, not only
the mean, has been updated.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

res_emp = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr_emp, slv = slv)))
res_ep = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                           opt = JuMPOptimiser(; pe = pr_ep, slv = slv)))

pretty_table(DataFrame(["Assets" => rd.nx, "Empirical" => res_emp.w,
                        "Entropy pooling" => res_ep.w]); formatters = [resfmt],
             title = "Maximum-ratio weights: empirical vs entropy pooling")

#=
The composition plot makes the tilt visible.
=#

plot_stacked_bar_composition([res_emp, res_ep], rd;
                             xticks = (1:2, ["Empirical", "Entropy pooling"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end. Mean (mu_views) and variance (sigma_views) views both take effect
#src   on the named asset, and the reweighted prior tilts the MaximumRatio portfolio.
#src - NAMING GOTCHA (doc, → #126): the EntropyPoolingPrior keyword `var_views` means *Value at
#src   Risk* views, NOT variance; variance views are `sigma_views` (and `cvar_views` = CVaR).
#src   Easy to invert (I did, first pass). The docstrings should call this out explicitly since
#src   "var" overwhelmingly reads as "variance".
#src - RESOLVED (was a misread, → #126): operator support is PER VIEW, not global. parse_equation
#src   defaults to ops1 = ("==", "<=", ">=") so mu/sigma/sk/kt/cov/rho all accept `<=`. Only
#src   var_views passes ops1 = ("==", ">=") and cvar_views ops1 = ("==",). My first-pass claim
#src   that the parser globally rejects `<=` came from testing a var/cvar view. Now documented
#src   per-view in the EntropyPoolingPrior docstring and in section 3 above.
