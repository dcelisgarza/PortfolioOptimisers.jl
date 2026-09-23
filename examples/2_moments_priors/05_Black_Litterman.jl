#=
```@meta
Description = "The Black-Litterman model in PortfolioOptimisers.jl: tilt an equilibrium prior toward your views on assets and asset groups."
```

# Black-Litterman

The estimators on the earlier pages take the data as it is. But you often hold a view, such as
"Apple will return 8 bps a day", "Microsoft will beat AMD" or "tech as a group will do well",
and you want to add it to the prior without losing what the market already tells you. The
Black-Litterman model does this. It starts from an equilibrium prior, the returns that reverse
optimisation implies for a given portfolio, and moves it toward your views. The confidence of
each view sets how far it moves the prior. The result is a prior with a posterior mean and
covariance, and you give it to an optimiser as you give any other prior.

This page is the first of three on priors built from views.
[Entropy pooling](07_Entropy_Pooling.md) and [opinion pooling](08_Opinion_Pooling.md) follow,
and each builds on the page before it. You can also read each page on its own.

[`BlackLittermanPrior`](@ref) takes a base estimator in `pe`, whose default mean is
[`EquilibriumExpectedReturns`](@ref), a [`UniverseSets`](@ref) that names the assets and the
groups, and the views in `views`. You write the views as strings in a
[`LinearConstraintEstimator`](@ref). `views_conf` sets the confidence of each view, and `tau`
scales the uncertainty of the prior for all the views at once.

!!! tip "When to reach for this"
    Reach for Black-Litterman when you hold forecasts of your own and want to blend them with
    a market equilibrium, rather than replace the mean outright. A view can be absolute,
    relative or about a group, as section 4 shows. It is the simplest of the three view
    priors, because a view enters as a Gaussian update of the mean. If your views are about other quantities, such as the variance, the tail risk or
    the skew, or you want them to hold as constraints on the distribution, see
    [entropy pooling](07_Entropy_Pooling.md).
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
## 1. The data

We load the daily returns of 20 assets over one year. The views below name some of these
assets.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

#=
## 2. The equilibrium prior

Black-Litterman does not start from the sample mean. It starts from the
[`EquilibriumExpectedReturns`](@ref) vector, the returns that reverse optimisation implies for a
portfolio ``\boldsymbol{w}_{mkt}``,
``\boldsymbol{\pi} = \lambda \mathbf{\Sigma} \boldsymbol{w}_{mkt}``. The page passes no weights,
so ``\boldsymbol{w}_{mkt}`` is the equal-weight portfolio. Pass market capitalisations in `w` to
use the market portfolio. The sample mean of a single year is noisy and often negative. The
equilibrium prior is smoother and has an economic reason behind it, and the views move it from
there.

We build both and compare them.
=#

pr_sample = prior(EmpiricalPrior(), rd)
pr_eq = prior(EmpiricalPrior(; me = EquilibriumExpectedReturns()), rd)

pretty_table(DataFrame(["Assets" => rd.nx, "Sample mean" => pr_sample.mu,
                        "Equilibrium" => pr_eq.mu]); formatters = [mmtfmt],
             title = "Sample mean vs equilibrium prior")

using StatsPlots, GraphRecipes
# The sample mean of each asset.
plot_mu(pr_sample, rd.nx)
# The equilibrium expected return of each asset.
plot_mu(pr_eq, rd.nx)

#=
## 3. Naming assets and groups

A view names assets and groups, so we declare a [`UniverseSets`](@ref). The `nx` key holds
every asset, and we add two example groups so that a view can name a group.
=#

sets = UniverseSets(;
                    dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                                "energy" => ["CVX"]))

#=
## 4. The three kinds of views

A view is a string, and Black-Litterman reads three kinds.

  - An absolute view, `"AAPL == 0.0008"`, says that Apple returns 8 bps a day.
  - A relative view, `"MSFT - AMD == 0.0005"`, says that Microsoft beats AMD by 5 bps a day.
  - A group view, `"tech == 0.0006"`, says that the tech group returns 6 bps a day on average.

We build one posterior per kind of view and compare its expected returns with the equilibrium
prior. A Black-Litterman view is soft, so the posterior does not reproduce it exactly. The
model blends the view with the equilibrium prior, and the posterior moves only part of the way.
In the table, the gap between Microsoft and AMD stays well short of the 5 bps the view states.
=#

tau = 1 / size(rd.X, 1)

pr_abs = prior(BlackLittermanPrior(; sets = sets, tau = tau,
                                   views = LinearConstraintEstimator(;
                                                                     val = ["AAPL == 0.0008"])),
               rd)
pr_rel = prior(BlackLittermanPrior(; sets = sets, tau = tau,
                                   views = LinearConstraintEstimator(;
                                                                     val = ["MSFT - AMD == 0.0005"])),
               rd)
pr_grp = prior(BlackLittermanPrior(; sets = sets, tau = tau,
                                   views = LinearConstraintEstimator(;
                                                                     val = ["tech == 0.0006"])),
               rd)

pretty_table(DataFrame(["Assets" => rd.nx, "Equilibrium" => pr_eq.mu,
                        "Absolute" => pr_abs.mu, "Relative" => pr_rel.mu,
                        "Group" => pr_grp.mu]); formatters = [mmtfmt],
             title = "Posterior expected returns by view type")

#=
## 5. The confidence of a view, `views_conf`

`views_conf` sets how far the posterior moves toward each view, as a confidence strictly between
0 and 1. At a low confidence the posterior stays near the equilibrium prior, and at a high
confidence it moves most of the way to the view. We sweep the confidence of one absolute view.
The table prints Apple's posterior expected return at each confidence, and its title gives the
equilibrium value and the 8 bps of the view for comparison.
=#

abs_view = LinearConstraintEstimator(; val = ["AAPL == 0.0008"])
confs = [0.1, 0.3, 0.5, 0.7, 0.9]
pr_confs = [prior(BlackLittermanPrior(; sets = sets, tau = tau, views = abs_view,
                                      views_conf = [c]), rd) for c in confs]

i_aapl = findfirst(==("AAPL"), rd.nx)
pretty_table(DataFrame(; confidence = confs,
                       Symbol("AAPL posterior") => [p.mu[i_aapl] for p in pr_confs]);
             formatters = [mmtfmt],
             title = "AAPL posterior vs view confidence (equilibrium ≈ $(round(pr_eq.mu[i_aapl]*100; digits=4))%, view = 0.08%)")

#=
## 6. The posterior covariance

Black-Litterman updates the covariance as well as the mean. The posterior covariance adds `tau`
times the covariance, for the uncertainty of the mean, and subtracts a smaller term for what the
views tell. The posterior variance is therefore never below the prior one. With a small `tau`
both terms are small. The table compares Apple's posterior variance with the empirical one.
=#

pretty_table(DataFrame(["quantity" => ["AAPL variance"],
                        "Empirical" => [pr_sample.sigma[i_aapl, i_aapl]],
                        "BL posterior" => [pr_abs.sigma[i_aapl, i_aapl]]]);
             formatters = [mmtfmt], title = "Posterior covariance adjustment")

#=
## 7. From views to portfolios

The views change the portfolio, and we show it in two ways. First we compare the
maximum-ratio portfolio under the equilibrium prior with the one under the posterior that is
bullish on Apple. Then we compute an efficient frontier under each, so that you can see the
change over the whole range of risk and return.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

res_eq = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                           opt = JuMPOptimiser(; pe = pr_eq, slv = slv)))
res_bl = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                           opt = JuMPOptimiser(; pe = pr_abs, slv = slv)))

pretty_table(DataFrame(["Assets" => rd.nx, "Equilibrium" => res_eq.w,
                        "Black-Litterman" => res_bl.w]); formatters = [resfmt],
             title = "Maximum-ratio weights: equilibrium vs Black–Litterman")

#=
In the composition plot, compare the weight of Apple in the two bars.
=#

plot_stacked_bar_composition([res_eq, res_bl], rd;
                             xticks = (1:2, ["Equilibrium", "Black-Litterman"]))

#=
Next we compute the two efficient frontiers. Each is a set of minimum-risk portfolios over a
range of return targets, one under the equilibrium prior and one under the Black-Litterman
posterior. The first plot is the equilibrium frontier. The view moves the whole frontier.
=#

fr_eq = optimise(MeanRisk(; obj = MinimumRisk(),
                          opt = JuMPOptimiser(; pe = pr_eq, slv = slv,
                                              ret = ArithmeticReturn(;
                                                                     settings = JuMPReturnsSettings(;
                                                                                                    lb = Frontier(;
                                                                                                                  N = 20))))))
fr_bl = optimise(MeanRisk(; obj = MinimumRisk(),
                          opt = JuMPOptimiser(; pe = pr_abs, slv = slv,
                                              ret = ArithmeticReturn(;
                                                                     settings = JuMPReturnsSettings(;
                                                                                                    lb = Frontier(;
                                                                                                                  N = 20))))))

plot_measures(fr_eq.w, pr_eq; x = Variance(), y = ExpectedReturn(; rt = fr_eq.ret),
              title = "Efficient frontier: equilibrium prior", xlabel = "Variance",
              ylabel = "Expected return")

# The frontier under the Black-Litterman posterior.
plot_measures(fr_bl.w, pr_abs; x = Variance(), y = ExpectedReturn(; rt = fr_bl.ret),
              title = "Efficient frontier: Black–Litterman posterior", xlabel = "Variance",
              ylabel = "Expected return")

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Deep-dive pass (per "examples are deep dives" feedback): added the equilibrium-prior
#src   contrast, all three view types side by side, a views_conf conviction sweep, the
#src   posterior-covariance adjustment, and equilibrium-vs-BL efficient frontiers.
#src - Teaching point confirmed empirically: BL views are *soft* — a relative view
#src   "MSFT - AMD == 0.0005" only moves the posterior gap to ~0.00018, and views_conf
#src   interpolates AAPL's posterior from the equilibrium ≈ 0.00023 (conf 0.1) toward the view
#src   0.0008 (conf 0.9). `tau` showed no visible effect at these values, so it is mentioned
#src   but not swept.
#src - AUTHORING GOTCHA (→ #126): `DataFrame(; "Col" => v)` with *string* keys in keyword
#src   position throws a TypeError; the vector-of-pairs positional form `DataFrame(["Col" =>
#src   v, ...])` works. Symbol keys work in keyword position.
