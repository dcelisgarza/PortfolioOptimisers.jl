#=
```@meta
Description = "Entropy pooling in PortfolioOptimisers.jl: views as constraints on any moment, imposed by reweighting scenarios without assuming normality."
```

# Entropy pooling

[Black-Litterman](05_Black_Litterman.md) adds views to the mean through a Gaussian update.
Entropy pooling is more general in two ways. First, a view can constrain any moment, such as
the mean, the variance, the CVaR, the skewness or the kurtosis, and even a single covariance or
correlation. Second, it assumes no normal distribution. It reweights the empirical scenarios so
that the new distribution satisfies your views and stays as close as it can to the original
one, in relative entropy, which is the Kullback-Leibler divergence. The result gives every
scenario a new weight, so every moment of the prior can change.

This page is the second of three on priors built from views.
[Black-Litterman](05_Black_Litterman.md) comes before it, and
[opinion pooling](08_Opinion_Pooling.md) follows and combines several entropy pooling priors
into one.

[`EntropyPoolingPrior`](@ref) takes one [`LinearConstraintEstimator`](@ref) for each quantity,
and the names of the fields need care. `mu_views` holds views on the mean, and `sigma_views`
views on the variance. The tail risk views are `var_views` for the value at risk, which is not
the variance, `cvar_views` for the conditional value at risk, `evar_views` for the entropic
value at risk and `rlvar_views` for the relativistic value at risk. `sk_views` and `kt_views`
hold views on the skewness and the kurtosis, and `cov_views` and `rho_views` views on
covariances and correlations. Each is a list of string constraints over the names of a
[`UniverseSets`](@ref).

!!! tip "When to reach for this"
    Reach for entropy pooling when your views say more than "the mean will be x". Examples are
    views on the volatility, on a tail risk such as the CVaR, on the skewness, or on the
    correlation between two assets, one or several at once. It also suits you when you do not
    trust the normal distribution that Black-Litterman assumes, because it reweights the
    empirical scenarios directly. For a view on the mean alone, Black-Litterman is simpler. To
    combine several entropy pooling priors, see [opinion pooling](08_Opinion_Pooling.md).
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

As with Black-Litterman, a view names assets and groups through a [`UniverseSets`](@ref).
=#

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX"]))

#=
## 3. Views on several moments

Entropy pooling views are strings too, but each field reads a different quantity. We state
three views. In `mu_views`, Apple returns 8 bps a day, and tech returns at least as much as
energy. In `sigma_views`, Apple's variance takes a fixed value. The comparison operators a view
accepts depend on the moment. `mu_views`, `sigma_views`, `sk_views`, `kt_views`, `cov_views`,
`rho_views`, `cvar_views`, `evar_views` and `rlvar_views` take `==`, `>=` and `<=`, and
`var_views` takes only `==` and `>=`. An operator that a field does not accept raises a
`ParseError`, which lists the operators that the field allows.

A significance level belongs to the view and not to the estimator, because the CVaR at 1% and
the CVaR at 10% are different statistics of the same series. So `var_views`, `cvar_views`,
`evar_views` and `rlvar_views` take a [`ValueatRiskView`](@ref), a
[`ConditionalValueatRiskView`](@ref), an [`EntropicValueatRiskView`](@ref) and a
[`RelativisticValueatRiskView`](@ref) respectively. Each pairs a group of view equations with
the `alpha` at which to read them, and a field also takes a vector of them for views at several
levels. A [`RelativisticValueatRiskView`](@ref) holds a second parameter of the same kind,
`kappa`. The RLVaR tends to the EVaR as `kappa` approaches zero, and rises toward the worst loss
of the sample as `kappa` approaches one, so the group states both. A `prior(...)` term inside a
group reads the prior's statistic at the `alpha` of that group, and at its `kappa` where the
measure has one.

A tail view is not a linear function of the posterior probabilities. It needs auxiliary
variables, and therefore a [`JuMPEntropyPooling`](@ref) in `opt`. The `alg` field of a tail
view group sets how the model writes each view. Left at `nothing`, each view takes the
cheapest formulation that states it exactly. A lower bound, or an equality at or above the
prior value, takes [`LinearConditionalValueatRiskView`](@ref),
[`ConicEntropicValueatRiskView`](@ref) or [`ConicRelativisticValueatRiskView`](@ref). Any
other view takes [`IntegerConditionalValueatRiskView`](@ref),
[`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref), which need
a mixed-integer conic solver.

A view over several assets whose coefficients all have one sign, such as a group, is a
positive combination of the measures, and it takes the same dual formulations. A relative view
has coefficients of both signs. For the CVaR it takes the integer formulation. For the other two
measures it takes [`SequentialEntropicValueatRiskView`](@ref) or
[`SequentialRelativisticValueatRiskView`](@ref), which solve a convex program a few times and
need no integer variable. [`SequentialConditionalValueatRiskView`](@ref) gives the CVaR the same
option. For an [`EntropicValueatRiskView`](@ref) or a [`RelativisticValueatRiskView`](@ref), the
`alg` field also holds the grid of dual variables and the big-M constant, so one group can take
its own [`GridEntropicValueatRiskView`](@ref) or [`GridRelativisticValueatRiskView`](@ref).
[`ValueatRiskView`](@ref) has no `alg`, because a VaR view is linear in the posterior
probabilities and there is no formulation to choose.
=#

mu_views = LinearConstraintEstimator(; val = ["AAPL == 0.0008", "tech >= energy"])
sigma_views = LinearConstraintEstimator(; val = ["AAPL == 0.0003"])

ep = EntropyPoolingPrior(; sets = sets, mu_views = mu_views, sigma_views = sigma_views)

#=
## 4. The prior and the reweighted posterior

We compute the entropy pooling posterior, and compare Apple's mean and variance under it with
those of the plain empirical prior. Entropy pooling imposes each view as a constraint, so you
can check each posterior value in the table against the value its view states.
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
The views on Apple and on the tech group move the other assets too, because the reweighting
changes the probability of every scenario. The second table compares the expected returns of
every asset.
=#

pretty_table(DataFrame(["Assets" => rd.nx, "Empirical" => pr_emp.mu,
                        "Entropy pooling" => pr_ep.mu]); formatters = [mmtfmt],
             title = "Expected returns: empirical vs entropy-pooling posterior")

# The expected returns of the entropy pooling posterior.
using StatsPlots, GraphRecipes
plot_mu(pr_ep, rd.nx)

#=
## 5. A view on the relativistic value at risk

`rlvar_views` states a view on the relativistic value at risk, RLVaR, which generalises the
entropic value at risk with a deformation parameter ``\kappa``. Two numbers define the
statistic. `alpha` is the significance level, and `kappa` is the deformation. The RLVaR tends
to the EVaR as `kappa` approaches zero, and rises toward the worst loss of the sample as `kappa`
approaches one. Both belong to the [`RelativisticValueatRiskView`](@ref) group, not to the
estimator.

The second number limits what a view can ask for. No reweighting of the sample can push a tail
measure past the worst loss in the sample, so a lower-bound view has room only between the
prior value and that loss. At `kappa = 0.3` the RLVaR is already close to the worst loss, and
the CVaR of the same asset is not.

Every field that takes a `Solver` also takes a vector of them, and the library tries them in
order until one solves the problem. Near the worst loss the second solver matters. With one
configuration alone, the solver stops short when it computes the RLVaR of the posterior that
the `<=` view below gives, and it reports the failure. A second configuration with a shorter
step solves it.
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
So the view `"AAPL >= 1.25*prior(AAPL)"`, an ordinary request on `cvar_views`, throws a
`DomainError` here, and the message gives the worst loss. On this statistic a view has room
for a multiple near 1.05.

A target below the prior needs the other formulation.
[`ConicRelativisticValueatRiskView`](@ref) bounds the RLVaR from below only, so a `<=` view
takes [`GridRelativisticValueatRiskView`](@ref) instead. The grid picks one of its points with
a binary vector, so it needs a solver for mixed-integer conic programs. Pajarito is one such
solver, and here it uses HiGHS for the outer approximation and Clarabel for the cones.
[`SequentialRelativisticValueatRiskView`](@ref) is the other way to state a `<=` view. It
bounds the RLVaR from above with a constraint that is linear in the posterior probabilities.
It computes that constraint again at each new posterior and solves again until the bound is
tight, so it needs no integer variable.
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
The table prints Apple's RLVaR under the empirical prior and under each posterior, so you can
compare each value with 1.05 and 0.95 times the prior value. The divergence column measures how
far each view moved the distribution away from the prior. The `<=` view moves it less, because
it is easier to take 5% off a statistic that is close to the worst loss than to add 5% to it.
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
!!! warning "The conic formulation is hard to solve"
    [`ConicRelativisticValueatRiskView`](@ref) adds ``2T`` power cones to the model, where
    ``T`` is the number of observations. A longer sample, a smaller `alpha`, a smaller `kappa`,
    or two such views in one model can make a conic solver stop short of a solution. Give the
    `slv` field of `opt` a vector of solver configurations, shorten the sample, or state the
    view with [`GridRelativisticValueatRiskView`](@ref). Its lower-bound constraints are linear
    in the posterior probabilities, and it adds no cone.
=#

#=
## 6. Views change the portfolio

We give the reweighted prior to a maximum-ratio optimiser. As with Black-Litterman, the
portfolio moves weight toward the assets that the views favour. The difference is that here the
views changed the whole distribution.
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
In the composition plot, compare the weights of Apple and of the tech stocks between the two
bars.
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
