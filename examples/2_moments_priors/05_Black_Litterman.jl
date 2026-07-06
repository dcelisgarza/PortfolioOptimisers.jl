#=
# Black–Litterman

The estimators so far take the data at face value. But you often *have a view* — "Apple will
return 8 bps a day", "Microsoft will beat AMD", "tech as a group will do well" — and want to
fold that conviction into the prior without throwing away what the market already tells you.
That is exactly what the **Black–Litterman** model does: it starts from a neutral
*equilibrium* prior (the returns implied by holding the market), then tilts it toward your
views, weighting each by its confidence. The result is a posterior mean and covariance you can
feed to any optimiser.

This is the first of a short, sequenced arc on **view-based priors** — Black–Litterman here,
then [Entropy Pooling](07_Entropy_Pooling.md), then [Opinion Pooling](08_Opinion_Pooling.md).
Each builds on the last, but each page also stands alone.

In `PortfolioOptimisers`, [`BlackLittermanPrior`](@ref) takes a base estimator `pe` (whose
default mean is [`EquilibriumExpectedReturns`](@ref)), an [`AssetSets`](@ref) that names assets
and groups, and a `views` estimator. Views are written as plain string constraints through a
[`LinearConstraintEstimator`](@ref), and their conviction is controlled by `views_conf` and the
global scaling parameter `tau`.

!!! tip "When to reach for this"
    Reach for Black–Litterman when you hold subjective forecasts — absolute ("this asset will
    return x"), relative ("A will beat B"), or group-level ("tech beats energy") — and want a
    principled blend of those views with a market-equilibrium baseline rather than overwriting
    the mean wholesale. It is the gentlest of the view priors: views enter as a Gaussian update
    on the mean. If your views are about quantities other than the mean (variance, tail risk,
    skew), or you want them as hard distributional constraints, see Entropy Pooling.
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
## 2. The equilibrium prior

Black–Litterman does not start from the sample mean. Its baseline is the
[`EquilibriumExpectedReturns`](@ref) vector — the returns *implied* by the market via reverse
optimisation (``\\boldsymbol{\\pi} = \\lambda \\mathbf{\\Sigma} \\boldsymbol{w}_{mkt}``). This
matters because the raw sample mean over a single year is noisy and often negative, whereas the
equilibrium prior is a smoother, economically-motivated anchor that the views then nudge.

We build both and compare them.
=#

pr_sample = prior(EmpiricalPrior(), rd)
pr_eq = prior(EmpiricalPrior(; me = EquilibriumExpectedReturns()), rd)

pretty_table(DataFrame(["Assets" => rd.nx, "Sample mean" => pr_sample.mu,
                        "Equilibrium" => pr_eq.mu]); formatters = [mmtfmt],
             title = "Sample mean vs equilibrium prior")

# Noisy sample-mean expected returns.
using StatsPlots, GraphRecipes
# Smoother market-equilibrium prior.
plot_mu(pr_sample, rd.nx)
plot_mu(pr_eq, rd.nx)

#=
## 3. Naming assets and groups

Views refer to assets and groups by name, so we declare an [`AssetSets`](@ref): the `nx` key
holds every asset, and we add two illustrative groups so we can express a group-level view.
=#

sets = AssetSets(;
                 dict = Dict("nx" => rd.nx, "tech" => ["AAPL", "AMD", "MSFT"],
                             "energy" => ["CVX"]))

#=
## 4. The three kinds of views

Views are plain strings, and Black–Litterman understands three shapes:

  - **Absolute** — `"AAPL == 0.0008"`: Apple returns 8 bps a day.
  - **Relative** — `"MSFT - AMD == 0.0005"`: Microsoft beats AMD by 5 bps.
  - **Group** — `"tech == 0.0006"`: the tech group averages 6 bps.

We build one posterior per view type and compare the resulting expected returns against the
equilibrium prior. A key property to notice: BL views are *soft*. The posterior does not
reproduce the view exactly — it Bayesian-blends the view with the equilibrium anchor, so the
realised tilt is partial (the relative gap below lands well short of the stated 5 bps).
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
## 5. Controlling conviction with `views_conf`

How hard the posterior leans on a view is set by `views_conf` — a confidence in ``[0, 1]`` per
view. At low confidence the posterior stays near the equilibrium prior; at high confidence it
moves most of the way to the view. We sweep the confidence on a single absolute view and watch
Apple's posterior expected return climb from the equilibrium baseline toward the 8 bps target.
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

Black–Litterman updates the *covariance* too, not only the mean: the posterior reflects the
extra information the views carry. With a small `tau` the adjustment is modest, but it is there
— compare Apple's posterior variance against the empirical one.
=#

pretty_table(DataFrame(["quantity" => ["AAPL variance"],
                        "Empirical" => [pr_sample.sigma[i_aapl, i_aapl]],
                        "BL posterior" => [pr_abs.sigma[i_aapl, i_aapl]]]);
             formatters = [mmtfmt], title = "Posterior covariance adjustment")

#=
## 7. From views to portfolios

Finally, the payoff: the views reshape the portfolio. We do this two ways. First a single
maximum-ratio portfolio under the equilibrium prior vs the Apple-bullish posterior, then a full
efficient frontier under each so the tilt is visible across the whole risk/return range.
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
The composition plot makes the maximum-ratio tilt visible.
=#

plot_stacked_bar_composition([res_eq, res_bl], rd;
                             xticks = (1:2, ["Equilibrium", "Black-Litterman"]))

#=
And the efficient frontiers: minimum-risk portfolios across a sweep of return targets, under
the equilibrium prior and the Black–Litterman posterior. The view shifts the whole frontier.
=#

fr_eq = optimise(MeanRisk(; obj = MinimumRisk(),
                          opt = JuMPOptimiser(; pe = pr_eq, slv = slv,
                                              ret = ArithmeticReturn(;
                                                                     lb = Frontier(;
                                                                                   N = 20)))))
fr_bl = optimise(MeanRisk(; obj = MinimumRisk(),
                          opt = JuMPOptimiser(; pe = pr_abs, slv = slv,
                                              ret = ArithmeticReturn(;
                                                                     lb = Frontier(;
                                                                                   N = 20)))))

plot_measures(fr_eq.w, pr_eq; x = Variance(), y = ExpectedReturn(; rt = fr_eq.ret),
              title = "Efficient frontier: equilibrium prior", xlabel = "Variance",
              ylabel = "Expected return")

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
