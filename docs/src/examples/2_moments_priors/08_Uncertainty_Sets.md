The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/2_moments_priors/08_Uncertainty_Sets.jl"
```

# Uncertainty sets

The view priors so far ([Black–Litterman](05_Black_Litterman.md),
[Entropy Pooling](06_Entropy_Pooling.md), [Opinion Pooling](07_Opinion_Pooling.md)) change
*what* the moments are. **Robust optimisation** takes the opposite stance: it accepts that the
estimated moments are *wrong by some amount* and optimises against the worst case within an
**uncertainty set** around them. Rather than trusting a single point estimate of the
covariance (or mean), you bound a region the true value plausibly lies in and minimise the
worst-case risk (or maximise the worst-case return) over that region. The result is an
allocation that is stable to estimation error by construction.

`PortfolioOptimisers` builds uncertainty sets with an estimator such as
[`NormalUncertaintySet`](@ref) and an algorithm — [`BoxUncertaintySetAlgorithm`](@ref) (a
per-entry interval box) or [`EllipsoidalUncertaintySetAlgorithm`](@ref) (a joint ellipsoid).
The helper [`sigma_ucs`](@ref) produces a covariance set and [`mu_ucs`](@ref) a mean set; the
covariance set parametrises a robust risk measure like [`UncertaintySetVariance`](@ref), while
the mean set plugs into [`ArithmeticReturn`](@ref) to give a worst-case expected return. This
page is a deep dive across both: covariance robustness, the confidence level that sizes the
set, and worst-case-mean robust returns.

!!! tip "When to reach for this"
    Reach for uncertainty sets when you care less about a forecast and more about *robustness*
    to the noise in the moments — short windows, unstable covariances, regime risk. A box set
    is simple and conservative (it bounds each entry independently); an ellipsoidal set captures
    the joint geometry. Which one diversifies depends on *what* you make robust: see sections 4
    and 5 below. If you have actual views about where the moments are headed, reach for the view
    priors instead — or combine both, since a robust risk measure composes with any prior.

````@example 08_Uncertainty_Sets
using PortfolioOptimisers, PrettyTables, StableRNGs

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

We use the same S&P 500 slice as the other examples.

````@example 08_Uncertainty_Sets
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
````

## 2. Building covariance uncertainty sets

We construct two covariance uncertainty sets from a [`NormalUncertaintySet`](@ref) — one box,
one ellipsoidal. The set is estimated by resampling, so we fix an RNG for reproducibility.

````@example 08_Uncertainty_Sets
ucs_box = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = BoxUncertaintySetAlgorithm()), rd.X)
ucs_ell = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = EllipsoidalUncertaintySetAlgorithm()), rd.X)
````

## 3. The confidence level `q` sizes the set

`NormalUncertaintySet` takes a confidence level `q` (default `0.05`). It controls *how big* the
uncertainty set is: a **smaller `q` is more demanding** and yields a **larger, more
conservative** set, because you are insuring against a more extreme worst case. For a box set
this widens every per-entry interval; for an ellipsoidal set it inflates the radius. We sweep
`q` and measure the total width of the box (the sum of interval lengths) to watch the set grow
as `q` shrinks.

````@example 08_Uncertainty_Sets
qs = [0.01, 0.05, 0.10, 0.20]
box_widths = [let u = sigma_ucs(NormalUncertaintySet(; rng = StableRNG(1), q = q,
                                                     alg = BoxUncertaintySetAlgorithm()),
                                rd.X)
                  sum(abs, u.ub .- u.lb)
              end
              for q in qs]

pretty_table(DataFrame(; q = qs, Symbol("box total width") => box_widths);
             title = "Smaller q → wider (more conservative) uncertainty set")
````

## 4. Robust vs nominal minimum-variance

[`UncertaintySetVariance`](@ref) is the robust counterpart of [`Variance`](@ref): it minimises
the worst-case variance over the uncertainty set rather than the point estimate. We compare a
nominal minimum-variance portfolio against the box- and ellipsoid-robust versions.

````@example 08_Uncertainty_Sets
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
````

The robust portfolios hedge against covariance estimation error. For *covariance* robustness the
ellipsoidal set — which captures the joint geometry of the estimation error rather than bounding
each entry on its own — typically produces the more diversified, less concentrated allocation.

````@example 08_Uncertainty_Sets
pretty_table(DataFrame(["Assets" => rd.nx, "Nominal" => res_nom.w,
                        "Box-robust" => res_box.w, "Ellipsoid-robust" => res_ell.w]);
             formatters = [resfmt], title = "Minimum-variance weights: nominal vs robust")

using StatsPlots, GraphRecipes #= Nominal vs box- vs ellipsoid-robust minimum variance. =#

plot_stacked_bar_composition([res_nom, res_box, res_ell], rd;
                             xticks = (1:3, ["Nominal", "Box", "Ellipsoid"]))
````

## 5. Worst-case mean: robust expected returns

Robustness is not only about the covariance. A mean uncertainty set, built with [`mu_ucs`](@ref),
plugs into [`ArithmeticReturn`](@ref) via its `ucs` keyword and makes the optimiser maximise the
**worst-case** expected return over the set instead of the point estimate. This guards a
return-seeking objective against the fact that sample means are extremely noisy over a single
year.

A wiring note worth knowing: pass `ArithmeticReturn` a **pre-built** mean set (the result of
`mu_ucs`), exactly as `UncertaintySetVariance` takes a pre-built `sigma_ucs` result. (Handing it
the *estimator* instead defers construction to solve time and requires the returns data to be
threaded through the optimiser.)

````@example 08_Uncertainty_Sets
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
````

The geometry flips relative to the covariance case. With a **box** mean set every asset's mean is
pushed to its own lower bound independently, so the worst-case maximum-ratio portfolio piles into
whichever single name still has the best worst-case Sharpe — it *concentrates*. The **ellipsoidal**
mean set couples the assets through the joint estimation geometry, so no single name can be cheap
in isolation and the worst-case allocation spreads out toward a near-equal-weight portfolio. The
lesson: "box vs ellipsoid" does not map to "concentrated vs diversified" in the abstract — it
depends on whether you are making the *mean* or the *covariance* robust.

````@example 08_Uncertainty_Sets
pretty_table(DataFrame(["Assets" => rd.nx, "Nominal" => ret_nom.w,
                        "Box worst-case mean" => ret_box.w,
                        "Ellipsoid worst-case mean" => ret_ell.w]); formatters = [resfmt],
             title = "Maximum-ratio weights: nominal vs worst-case mean")

plot_stacked_bar_composition([ret_nom, ret_box, ret_ell], rd;
                             xticks = (1:3, ["Nominal", "Box μ", "Ellipsoid μ"]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
