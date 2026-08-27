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
first, and [Opinion Pooling](08_Opinion_Pooling.md) follows, combining several entropy-pooling
views into one.

In `PortfolioOptimisers`, [`EntropyPoolingPrior`](@ref) accepts a separate
[`LinearConstraintEstimator`](@ref) per quantity. Mind the naming: `mu_views` is the mean,
`sigma_views` is the **variance**, `var_views` is the **Value at Risk**, `cvar_views` the
**Conditional VaR**, `evar_views` the **Entropic VaR** and `rlvar_views` the **Relativistic
VaR** (tail-risk views), `sk_views`/`kt_views` are skewness/kurtosis, and
`cov_views`/`rho_views` target covariances/correlations. Each is a list of string constraints
over the [`UniverseSets`](@ref) names.

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
[`UniverseSets`](@ref).
=#

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX"]))

#=
## 3. Views on several moments

Entropy-pooling views are also plain strings, but they can target different quantities. Here we
state a **mean** view (Apple returns 8 bps) via `mu_views`, a **relative mean** view (tech
outperforms energy), and a **variance** view (pin Apple's variance) via `sigma_views`. The
comparison operators a view accepts depend on the moment: `mu_views`, `sigma_views`,
`sk_views`, `kt_views`, `cov_views`, `rho_views`, `cvar_views`, `evar_views` and
`rlvar_views` take `==`, `>=` and `<=`; `var_views` (VaR) takes only `==` and `>=`. An
unsupported operator raises a `ParseError` listing the ones allowed for that view.

A significance level belongs to the view rather than to the estimator: the CVaR at 1% and at
10% are different statistics of the same series. So `var_views`, `cvar_views`, `evar_views`
and `rlvar_views` each take a [`ValueatRiskView`](@ref), a
[`ConditionalValueatRiskView`](@ref), an [`EntropicValueatRiskView`](@ref) or a
[`RelativisticValueatRiskView`](@ref) — each pairing a group of view equations with the
`alpha` it is read under — or a vector of them for views stated at several levels. A
[`RelativisticValueatRiskView`](@ref) carries a second parameter of the same kind, `kappa`:
RLVaR reduces to the EVaR as `kappa` approaches zero and rises towards the worst loss of the
sample as it approaches one, so the group states both. A `prior(...)` reference inside a group
resolves at that group's level, and at its `kappa` where the family has one.

A tail view is not a linear function of the posterior probabilities, so it needs auxiliary
variables and therefore a [`JuMPEntropyPooling`](@ref) in `opt`. The `alg` field of a tail view
group picks how each view is written; left at `nothing` each takes the cheapest formulation
that expresses it exactly — [`LinearConditionalValueatRiskView`](@ref),
[`ConicEntropicValueatRiskView`](@ref) and [`ConicRelativisticValueatRiskView`](@ref) for a
lower bound or an equality at or above the prior value, and
[`IntegerConditionalValueatRiskView`](@ref), [`GridEntropicValueatRiskView`](@ref) or
[`GridRelativisticValueatRiskView`](@ref) otherwise, which need a mixed-integer conic solver.
For an [`EntropicValueatRiskView`](@ref) or a [`RelativisticValueatRiskView`](@ref) the `alg`
field is also where the grid of dual variables and the big-M constant live, so one group can
take its own [`GridEntropicValueatRiskView`](@ref) or
[`GridRelativisticValueatRiskView`](@ref). [`ValueatRiskView`](@ref) has no `alg`: a VaR view
is linear in the posterior probabilities, so there is no formulation to choose.
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
## 5. A tail view: relativistic value at risk

`rlvar_views` states a view on the **relativistic VaR**, the ``\kappa``-deformed
generalisation of the entropic VaR. Two numbers name the statistic. `alpha` is the significance
level, and `kappa` is the deformation: RLVaR reduces to the EVaR as `kappa` approaches zero,
and rises towards the worst loss of the sample as it approaches one. Both belong to the
[`RelativisticValueatRiskView`](@ref) group rather than to the estimator.

That second number sets the first trap. No reweighting of the sample can push a tail measure
past the worst loss the sample holds, so the room a lower-bound view has is whatever lies
between the prior value and that loss. At `kappa = 0.3` the RLVaR already sits near it, where
the CVaR of the same asset does not.

Every field that takes a `Solver` also takes a vector of them, tried in order until one
answers. Near the worst loss that spare matters. One configuration alone stops short when it
reads the RLVaR of the posterior the `<=` view below produces, and says so. A second one with
a shorter step answers it.
=#

using Clarabel, HiGHS, Pajarito, JuMP

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.85),
              check_sol = (; allow_local = true, allow_almost = true))]

x_aapl = rd.X[:, i_aapl]
rlvar_of = w -> RelativisticValueatRisk(; alpha = 0.05, kappa = 0.3, slv = slv, w = w)(x_aapl)
prior_cvar = ConditionalValueatRisk(; alpha = 0.05)(x_aapl)
prior_evar = EntropicValueatRisk(; alpha = 0.05, slv = slv)(x_aapl)
prior_rlvar = RelativisticValueatRisk(; alpha = 0.05, kappa = 0.3, slv = slv)(x_aapl)
worst_loss = maximum(-x_aapl)

pretty_table(DataFrame(["Statistic" => ["CVaR", "EVaR", "RLVaR, kappa = 0.3", "worst loss"],
                        "AAPL, prior" => [prior_cvar, prior_evar, prior_rlvar, worst_loss]]);
             formatters = [mmtfmt], title = "How much room a lower-bound tail view has")

#=
So `"AAPL >= 1.25*prior(AAPL)"` — a routine ask on `cvar_views` — is refused here with a
`DomainError` naming that worst loss. A multiple near 1.05 is what this statistic affords.

A target below the prior is the other half, and it needs the other formulation.
[`ConicRelativisticValueatRiskView`](@ref) bounds the RLVaR from below only, so a `<=` view
takes [`GridRelativisticValueatRiskView`](@ref) instead. The grid picks one of its points with
a binary vector, so it needs a solver for mixed-integer conic programs. Pajarito supplies one,
driving HiGHS on the outer approximation and Clarabel on the cones.
=#

mip_slv = Solver(; name = :pajarito1,
                 solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                    "oa_solver" =>
                                                        optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  JuMP.MOI.Silent() =>
                                                                                      true),
                                                    "conic_solver" =>
                                                        optimizer_with_attributes(Clarabel.Optimizer,
                                                                                  "verbose" =>
                                                                                      false)),
                 check_sol = (; allow_local = true, allow_almost = true))

lo_view = RelativisticValueatRiskView(;
                                      views = LinearConstraintEstimator(;
                                                                        val = "AAPL >= 1.05*prior(AAPL)"))
hi_view = RelativisticValueatRiskView(;
                                      views = LinearConstraintEstimator(;
                                                                        val = "AAPL <= 0.95*prior(AAPL)"))

pr_lo = prior(EntropyPoolingPrior(; sets = sets, opt = JuMPEntropyPooling(; slv = slv),
                                  rlvar_views = lo_view), rd)
pr_hi = prior(EntropyPoolingPrior(; sets = sets, opt = JuMPEntropyPooling(; slv = mip_slv),
                                  rlvar_views = hi_view), rd)

#=
Each view lands on its target. The divergence column is the price it pays. The `<=` view is
the cheaper of the two: to take 5% off a statistic that already sits near the worst loss is a
smaller ask than to add 5% to it.
=#

kldfmt = (v, i, j) -> begin
    if j == 1
        return v
    elseif j == 2
        return "$(round(v * 100, digits = 4)) %"
    else
        return isa(v, Number) ? string(round(v; sigdigits = 4)) : v
    end
end;

pretty_table(DataFrame(["Prior" => ["empirical", "RLVaR >= 1.05 prior, conic",
                                    "RLVaR <= 0.95 prior, grid"],
                        "AAPL RLVaR" => [prior_rlvar, rlvar_of(pr_lo.w), rlvar_of(pr_hi.w)],
                        "Divergence" => ["", pr_lo.kld, pr_hi.kld]]); formatters = [kldfmt],
             title = "Each RLVaR view lands on its target")

#=
!!! warning "The conic formulation is a demanding solve"
    [`ConicRelativisticValueatRiskView`](@ref) writes ``2T`` power cones. A longer sample, a
    smaller `alpha`, a smaller `kappa`, or two such views in one model can make a conic solver
    stop short of a solution. Give `opt` a vector of solver configurations, shorten the
    sample, or state the view under [`GridRelativisticValueatRiskView`](@ref), whose
    lower-bound rows are linear in the posterior probabilities and write no cone at all.
=#

#=
## 6. Why it matters: views change the portfolio

Feeding the reweighted prior to a return-seeking optimiser tilts the portfolio toward the
view-favoured assets, just as Black–Litterman did — but here the *whole distribution*, not only
the mean, has been updated.
=#

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
