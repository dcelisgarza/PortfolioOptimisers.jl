#=
```@meta
Description = "Advanced Black-Litterman variants in PortfolioOptimisers.jl: factor views, asset and factor views at once, and the Bayesian form."
```

# Advanced Black-Litterman variants

The base [`BlackLittermanPrior`](@ref) of the [previous page](05_Black_Litterman.md) works on
the assets alone, and each view is about one asset or a group of assets. But a view is often
easier to state about a factor, such as "momentum will earn a premium" or "quality minus low
volatility will be negative". Sometimes you also hold views on assets and on factors at once.
The library has three variants of the model for these cases.

  - [`BayesianBlackLittermanPrior`](@ref) is the Bayesian form of the model, and it builds on
    a [`FactorPrior`](@ref). You state the views on the factors, and the factor model carries
    the posterior back to the assets.
  - [`FactorBlackLittermanPrior`](@ref) takes views on the factor premia and passes them to
    the assets through a factor regression. The `rsd` flag sets whether the posterior keeps
    the residual variance of each asset, and `l` is the risk aversion of the implied factor
    equilibrium.
  - [`AugmentedBlackLittermanPrior`](@ref) takes views on assets in `a_views` and views on
    factors in `f_views` in one model, each with its own confidence.

Each variant returns a posterior mean and covariance over the assets, `mu` and `sigma`, and any
optimiser accepts it, as with the base model.

!!! tip "When to reach for this"
    Reach for these when your views are about factors, alone or next to views on single
    assets. Examples are a macro view on momentum or value, a spread between quality and low
    volatility, or a view on a stock that you want to combine with a view on a factor in one
    posterior. If every view is about assets, the base [`BlackLittermanPrior`](@ref) is
    simpler, and it is enough.

!!! note "Factor data required"
    These variants need factor returns, so you must build the [`ReturnsResult`](@ref) with
    the factor data, which the price ingestion step adds:
    `prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))`. Factor views and
    factor sets use the factor names in `rd.nf`.
=#

using PortfolioOptimisers, PrettyTables, DataFrames

mmtfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 4)) %" : v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data, sets and the equilibrium prior

We load the S&P 500 data with its factor returns. Then we declare one `UniverseSets` that all
three variants read. It lists the assets under `"nx"` and the factors under `"nf"`, which are
the default values of `xkey` and `tfkey`, each in the column order of `rd.X` and `rd.F`. It
also holds two groups of assets for the asset views of the augmented model. Every estimator here
looks the names of its factor views up in the declared list of factors. So one object serves a
model with factor views only, a model with asset views only, and the augmented model that has
both. We also compute the [`EquilibriumExpectedReturns`](@ref) prior, which every
Black-Litterman posterior moves away from.
=#

using CSV, TimeSeries

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))

universe_sets = UniverseSets(;
                             dict = Dict("nx" => rd.nx, "nf" => rd.nf,
                                         "tech" => ["AAPL", "AMD", "MSFT"],
                                         "energy" => ["CVX"]))
tau = 1 / size(rd.X, 1)

pr_eq = prior(EmpiricalPrior(; me = EquilibriumExpectedReturns()), rd)

#=
`rd.nf` holds the names of the factors in this data set.
=#

pretty_table(DataFrame(; factor = rd.nf); title = "Factor names (rd.nf)")

#=
## 2. Bayesian Black-Litterman with factor views

[`BayesianBlackLittermanPrior`](@ref) takes a [`FactorPrior`](@ref) as its base estimator, and
views on the factors, whose names it looks up in the declared factors, `sets.dict[sets.tfkey]`.
In this Bayesian form the factor prior gives the structure and the views update the means of
the factors. The model then maps the result back to a posterior over the assets.

We state two factor views. Momentum earns 5 bps a day, and quality returns 3 bps a day less than
low volatility.
=#

factor_views = LinearConstraintEstimator(;
                                         val = ["MTUM == 0.0005", "QUAL - USMV == -0.0003"])

pr_bayes = prior(BayesianBlackLittermanPrior(; pe = FactorPrior(; pe = EmpiricalPrior()),
                                             sets = universe_sets, tau = tau,
                                             views = factor_views), rd)

#=
## 3. Factor Black-Litterman with views on factor premia

[`FactorBlackLittermanPrior`](@ref) also takes factor views, but it passes them to the assets
through the factor regression rather than through a factor prior. It looks the names of its
views up in the declared factors, `sets.dict[sets.tfkey]`. Two parameters change the posterior.

  - `rsd = true` keeps the residual variance of each asset, and `rsd = false` drops it. So
    `rsd` sets whether the posterior covariance is the full covariance of the assets or only
    the part the factors explain.
  - `l` is the risk aversion of the implied factor equilibrium.

We build one prior with each `rsd` setting, and a third with a higher `l`. The table compares
Apple's posterior mean and variance under the two `rsd` settings. The page prints nothing for
the third prior, and you can compare `pr_fbl_l.mu` with `pr_fbl_rsd.mu` to see what `l`
changes.
=#

pr_fbl_rsd = prior(FactorBlackLittermanPrior(; pe = EmpiricalPrior(), rsd = true,
                                             sets = universe_sets, tau = tau,
                                             views = factor_views), rd)
pr_fbl_nors = prior(FactorBlackLittermanPrior(; pe = EmpiricalPrior(), rsd = false,
                                              sets = universe_sets, tau = tau,
                                              views = factor_views), rd)
pr_fbl_l = prior(FactorBlackLittermanPrior(; pe = EmpiricalPrior(), rsd = true, l = 5.0,
                                           sets = universe_sets, tau = tau,
                                           views = factor_views), rd)

i_aapl = findfirst(==("AAPL"), rd.nx)
pretty_table(DataFrame(; quantity = ["AAPL posterior mean", "AAPL posterior variance"],
                       rsd_true = [pr_fbl_rsd.mu[i_aapl], pr_fbl_rsd.sigma[i_aapl, i_aapl]],
                       rsd_false = [pr_fbl_nors.mu[i_aapl],
                                    pr_fbl_nors.sigma[i_aapl, i_aapl]]);
             formatters = [mmtfmt], title = "Factor BL: residual variance on vs off")

#=
With `rsd = false` the covariance keeps only the part the factors explain, so the posterior
variance is smaller. The model then assumes that the factors explain all of the risk.

## 4. Augmented Black-Litterman with asset and factor views

[`AugmentedBlackLittermanPrior`](@ref) is the most general variant. It takes views on assets in
`a_views` and views on factors in `f_views`, in the same posterior. Use it when you hold a view
on a stock and a view on a factor, and do not want to choose between them.

It is the only estimator of the three that reads both declared lists, and it reads them from
the same `universe_sets` as the variants above. It looks the names of `a_views` up in the
assets, `sets.dict[sets.xkey]`, so a view can name the `tech` group. It looks the names of
`f_views` up in the factors, `sets.dict[sets.tfkey]`.
=#

asset_views = LinearConstraintEstimator(; val = ["AAPL == 0.0008", "tech == 0.0006"])

pr_aug = prior(AugmentedBlackLittermanPrior(; sets = universe_sets, tau = tau,
                                            a_views = asset_views, f_views = factor_views),
               rd)

#=
## 5. Comparing the posteriors

Each variant gives a different posterior mean. The table lines them up with the equilibrium
prior for the first few assets. The factor views move every asset, and the augmented model
also moves the assets that its asset views name.
=#

cmp = DataFrame(; Assets = rd.nx, Equilibrium = pr_eq.mu, Bayesian = pr_bayes.mu,
                Factor = pr_fbl_rsd.mu, Augmented = pr_aug.mu)
pretty_table(first(cmp, 8); formatters = [mmtfmt],
             title = "Posterior expected returns (first 8 assets)")

# The posterior expected returns of the augmented model, for every asset.
using StatsPlots, GraphRecipes
plot_mu(pr_aug, rd.nx)

#=
## 6. From views to portfolios

As with the base model, each posterior changes the optimal portfolio. We solve a maximum-ratio
portfolio under the equilibrium prior and under each variant, then compare the compositions.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

priors = ["Equilibrium" => pr_eq, "Bayesian" => pr_bayes, "Factor" => pr_fbl_rsd,
          "Augmented" => pr_aug]
res = [optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                         opt = JuMPOptimiser(; pe = p, slv = slv))) for (_, p) in priors]

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in res]...),
                       [:assets; Symbol.(first.(priors))...]); formatters = [resfmt],
             title = "Maximum-ratio weights by prior")

plot_stacked_bar_composition(res, rd; xticks = (1:length(priors), first.(priors)))

#=
## Summary

The three variants take views that the base model cannot.

  - [`BayesianBlackLittermanPrior`](@ref) puts factor views on a [`FactorPrior`](@ref).
  - [`FactorBlackLittermanPrior`](@ref) passes views on the factor premia through a
    regression. `rsd` sets whether it keeps the residual variance, and `l` is the risk
    aversion of the implied equilibrium.
  - [`AugmentedBlackLittermanPrior`](@ref) combines asset views and factor views in one
    posterior.

Each returns a posterior over the assets, so any optimiser accepts it as it accepts the base
model. The variants differ in what the views are about.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): Bayesian / Factor (rsd on+off, l sweep) /
#src   Augmented posteriors all build, and all four maximum-ratio optimisations solve with
#src   Clarabel on the 252-obs / 20-asset + 5-factor slice (MTUM/QUAL/SIZE/USMV/VLUE).
#src - VERIFIED the variants are genuinely distinct: on the first three assets the posterior
#src   means are equilibrium [0.0231,0.0362,0.0191]%, Bayesian [0.0036,-0.1036,0.0005]%,
#src   Factor [-0.044,-0.1656,-0.0392]%, Augmented [0.0217,-0.1488,-0.0373]% — the factor views
#src   move the whole cross-section, and the augmented asset views add a further tilt.
#src - ERGO (RESOLVED by #232): the variants used to spread the view/sets wiring across different
#src   field names — base/Bayesian/Factor took `sets`+`views` while Augmented split into
#src   `a_sets`/`a_views` + `f_sets`/`f_views`, so a reader moving between them relearned the
#src   keyword names each time. All four now take one `sets`, and this page went from three sets
#src   objects to one. Only the *views* fields still differ, which is the real distinction.
#src - `BayesianBlackLittermanPrior` requires a factor-capable `pe` (a `FactorPrior`); passing a
#src   plain `EmpiricalPrior` is a different (asset-space) model. Worth a docstring note that the
#src   Bayesian variant is factor-prior-based by construction.
