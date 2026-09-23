#=
```@meta
Description = "Uncertainty sets in PortfolioOptimisers.jl: optimise against the worst case over a box or an ellipsoid around the estimated moments."
```

# Uncertainty sets

The view priors of the earlier pages, [Black-Litterman](05_Black_Litterman.md),
[entropy pooling](07_Entropy_Pooling.md) and [opinion pooling](08_Opinion_Pooling.md), change
the moments themselves. Robust optimisation keeps the estimated moments and assumes that they
are wrong by some amount. You choose a region around the estimate that the true covariance or
mean is likely to lie in, the uncertainty set. The optimiser then minimises the worst-case risk,
or maximises the worst-case return, over that region. The risk it reports is an upper bound for
every covariance inside the set, the estimate included.

An uncertainty set comes from an estimator, such as [`NormalUncertaintySet`](@ref), and an
algorithm. [`BoxUncertaintySetAlgorithm`](@ref) gives an interval for each entry of the moment,
and [`EllipsoidalUncertaintySetAlgorithm`](@ref) gives one ellipsoid over all the entries.
[`sigma_ucs`](@ref) builds a covariance set and [`mu_ucs`](@ref) builds a mean set. A robust
risk measure such as [`UncertaintySetVariance`](@ref) takes the covariance set, and
[`ArithmeticReturn`](@ref) takes the mean set to give a worst-case expected return. This page
covers both sets, and the confidence level that sets their size.

!!! tip "When to reach for this"
    Reach for uncertainty sets when the estimated moments are noisy, for example over a short
    window, and you want weights that allow for that noise more than you want to act on a
    forecast. A box set bounds each entry on its own. An ellipsoidal set bounds all the entries
    together. Which of the two spreads the weights depends on what you make robust, as sections
    4 and 5 show. If you hold views about where the moments are going, reach for a view prior
    instead. You can also use both, because a robust risk measure works with any prior.
=#

using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. The returns data

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

#=
## 2. Building covariance uncertainty sets

We build two covariance uncertainty sets with a [`NormalUncertaintySet`](@ref), a box and an
ellipsoid. The estimator resamples the returns, so we fix the random number generator to make
the run repeatable.
=#

ucs_box = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = BoxUncertaintySetAlgorithm()), rd.X)
ucs_ell = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = EllipsoidalUncertaintySetAlgorithm()), rd.X)

#=
## 3. The confidence level `q` sizes the set

`NormalUncertaintySet` takes a confidence level `q`, whose default is `0.05`, and `q` sets the
size of the set. A smaller `q` covers a more extreme worst case, so it gives a larger set. For a
box set every interval widens, and for an ellipsoidal set the radius grows. We build a box set
at four values of `q` and print the total width of each box, which is the sum of the lengths of
its intervals.
=#

qs = [0.01, 0.05, 0.10, 0.20]
box_widths = [let u = sigma_ucs(NormalUncertaintySet(; rng = StableRNG(1), q = q,
                                                     alg = BoxUncertaintySetAlgorithm()),
                                rd.X)
                  sum(abs, u.ub .- u.lb)
              end
              for q in qs]

pretty_table(DataFrame(; q = qs, Symbol("box total width") => box_widths);
             title = "Smaller q → wider (more conservative) uncertainty set")

#=
## 4. Robust and nominal minimum variance

[`UncertaintySetVariance`](@ref) is the robust form of [`Variance`](@ref). It minimises the
worst-case variance over the uncertainty set instead of the variance at the point estimate. We
solve the nominal minimum-variance portfolio, then the robust one with the box set and with the
ellipsoidal set.
=#

using Clarabel

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_nom = optimise(MeanRisk(; r = Variance(), obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_box = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = ucs_box),
                            obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_ell = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = ucs_ell),
                            obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
The table puts the weights of the three portfolios side by side. The box set bounds the error of
each covariance entry on its own, and the ellipsoidal set bounds the errors of all the entries
together. On this data the ellipsoidal set gives the less concentrated weights.
=#

pretty_table(DataFrame(["Assets" => rd.nx, "Nominal" => res_nom.w,
                        "Box-robust" => res_box.w, "Ellipsoid-robust" => res_ell.w]);
             formatters = [resfmt], title = "Minimum-variance weights: nominal vs robust")

# The composition of the nominal, box-robust and ellipsoid-robust portfolios.
using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_nom, res_box, res_ell], rd;
                             xticks = (1:3, ["Nominal", "Box", "Ellipsoid"]))

#=
## 5. Robust expected returns with a worst-case mean

You can make the mean robust too. Build a mean uncertainty set with [`mu_ucs`](@ref) and pass it
to [`ArithmeticReturn`](@ref) through its `ucs` keyword. The optimiser then maximises the
worst-case expected return over the set instead of the return at the point estimate. A sample
mean over one year of daily returns carries a large error, and an objective that seeks return
acts on that error unless the mean is robust.

Pass `ArithmeticReturn` the mean set that `mu_ucs` returns, as `UncertaintySetVariance` takes
the set that `sigma_ucs` returns. If you pass the estimator instead, the optimiser builds the set
when it solves, and it then needs the returns data as well.
=#

rf = 4.2 / 100 / 252

mu_box = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                     alg = BoxUncertaintySetAlgorithm()), rd)
mu_ell = mu_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                     alg = EllipsoidalUncertaintySetAlgorithm()), rd)

ret_nom = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv)))
ret_box = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ret = ArithmeticReturn(; ucs = mu_box))))
ret_ell = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                ret = ArithmeticReturn(; ucs = mu_ell))))

#=
The weights here are the reverse of section 4. With a box mean set, the mean of each asset falls
to the lower bound of its own interval. The worst-case maximum-ratio portfolio then puts almost
all its weight in the asset with the best worst-case Sharpe ratio. The ellipsoidal mean set
bounds the means together, and its worst-case penalty grows as the weights concentrate. So the
worst-case portfolio spreads its weight almost equally over the assets. Whether a box or an
ellipsoid concentrates the weights thus depends on whether you make the mean or the covariance
robust.

!!! note "Both worst-case mean portfolios are at the floor of the ratio's scale"
    With sets of this size, no portfolio has a worst-case return above `rf`. The ratio is then
    zero or negative for every portfolio, and no tangency portfolio exists.
    [`MaximumRatio`](@ref) solves the ratio with a scale variable `k`, and here it returns the
    portfolio at the floor `kmin` of that variable, so the weights still meet their constraints.
    Read them as the best worst-case return at that scale, not as a worst-case tangency
    portfolio. The nominal portfolio has a tangency portfolio, and this does not apply to it.
=#

pretty_table(DataFrame(["Assets" => rd.nx, "Nominal" => ret_nom.w,
                        "Box worst-case mean" => ret_box.w,
                        "Ellipsoid worst-case mean" => ret_ell.w]); formatters = [resfmt],
             title = "Maximum-ratio weights: nominal vs worst-case mean")

plot_stacked_bar_composition([ret_nom, ret_box, ret_ell], rd;
                             xticks = (1:3, ["Nominal", "Box μ", "Ellipsoid μ"]))

#=
## 6. Other set estimators: Delta and the ARCH bootstrap

[`NormalUncertaintySet`](@ref) resamples the returns as if they were Gaussian, and two other
estimators exist. The simpler one is [`DeltaUncertaintySet`](@ref), a box whose intervals are a
fixed fraction of the point estimate on each side, with no resampling. It gives the same set on
every run and costs almost nothing to build. You choose the fraction, and the data does not
change it. We build a covariance box with it and print its total width next to that of the
Normal box of section 2.
=#

ucs_delta = sigma_ucs(DeltaUncertaintySet(), rd.X)

set_width(u) = sum(abs, u.ub .- u.lb)
pretty_table(DataFrame(; estimator = ["Delta (fixed)", "Normal (q=0.05)"],
                       Symbol("box total width") =>
                           [set_width(ucs_delta), set_width(ucs_box)]);
             title = "Delta is a tight, deterministic box")

#=
On this data the Delta box is narrower than the Normal box. [`UncertaintySetVariance`](@ref)
takes it as it takes the Normal set, so we solve a robust minimum-variance portfolio with it and
print its weights next to those of the Normal box of section 4.
=#

res_delta = optimise(MeanRisk(; r = UncertaintySetVariance(; ucs = ucs_delta),
                              obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))

pretty_table(DataFrame(["Assets" => rd.nx, "Delta-robust" => res_delta.w,
                        "Normal-box-robust" => res_box.w]); formatters = [resfmt],
             title = "Minimum-variance weights: Delta vs Normal box")

plot_stacked_bar_composition([res_nom, res_delta, res_box], rd;
                             xticks = (1:3, ["Nominal", "Delta", "Normal box"]))

#=
!!! note "A set from a block bootstrap of the returns"
    [`ARCHUncertaintySet`](@ref) builds a set from the tails and the serial dependence of the
    returns themselves, not from a Gaussian or a fixed fraction. It resamples the returns in
    blocks, with [`StationaryBootstrap`](@ref), [`MovingBootstrap`](@ref) or
    [`CircularBootstrap`](@ref), and you build it in the same way.
    ```julia
    ucs_arch = sigma_ucs(ARCHUncertaintySet(; alg = BoxUncertaintySetAlgorithm(),
                                            bootstrap = StationaryBootstrap(), n_sim = 100,
                                            seed = 1), rd.X)
    ```
    The bootstrap keeps the fat tails and the autocorrelation that the Normal estimator ignores,
    and on this slice it gives a wider box than the Normal one. `UncertaintySetVariance` takes it
    as it takes the other two sets. This page does not run it, because it resamples the returns
    `n_sim` times and is the slowest of the three estimators.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Deep-dive pass (per "examples are deep dives"): added the `q` confidence-level sweep
#src   (set size) and a worst-case-MEAN section (mu_ucs → ArithmeticReturn ucs) alongside the
#src   existing covariance-robust UncertaintySetVariance. All verified end-to-end on kaimon
#src   (real SP500 slice, StableRNG(1)).
#src - Verified contrast (MaximumRatio, rf=4.2%/252): nominal max w ≈ 66% (2 names); box
#src   worst-case mean ≈ 97.7% max (concentrates into best worst-case Sharpe); ellipsoid
#src   worst-case mean ≈ 9.8% max (near-equal-weight, 20 nz). Covariance case is the opposite
#src   (ellipsoid diversifies). Documented this box/ellipsoid ≠ concentrated/diversified nuance
#src   in section 5 because it is genuinely surprising.
#src - q sweep verified monotone: box total width 0.0577 (q=0.01) > 0.0370 (q=0.10). Smaller q =
#src   wider/more conservative. The `q` field docstring only said "Quantile parameter" — added a
#src   set-size note there (→ #126).
#src - WIRING (→ #126): ArithmeticReturn(; ucs=…) needs a PRE-BUILT mu set (mu_ucs result).
#src   Passing the estimator throws `isnothing(rd.X)` at solve unless rd is threaded through.
#src   `sigma_ucs(NormalUncertaintySet(...; alg), X)` then `UncertaintySetVariance(; ucs=…)` is
#src   the symmetric pattern. Both are discoverable mainly from tests — added an ArithmeticReturn
#src   docstring note. Closes the 2_moments_priors group.
#src - Delta/Bootstrap pass: added §6 covering the non-Normal set estimators. `DeltaUncertaintySet()`
#src   builds a deterministic fractional-perturbation box (total width 0.0132 here, tighter than the
#src   Normal box) and feeds `UncertaintySetVariance` like any other set — kept runnable.
#src - `ARCHUncertaintySet` (block bootstrap: Stationary/Moving/Circular) VERIFIED to work in the
#src   docs env — it is native, not a Python `arch` shim despite the name — producing a wider box
#src   (width 0.0474 with StationaryBootstrap, n_sim=100). But n_sim=100 took ~50s on this slice, so
#src   per the build-cost trade-off it is DESCRIBED with a non-executed snippet rather than run in
#src   the rendered page. If the bootstrap is sped up or cached, promote it to executed code.
