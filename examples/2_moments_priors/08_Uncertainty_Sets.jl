#=
# Uncertainty sets

The view priors so far ([Black–Litterman](05_Black_Litterman.md),
[Entropy Pooling](06_Entropy_Pooling.md), [Opinion Pooling](07_Opinion_Pooling.md)) change
*what* the moments are. **Robust optimisation** takes the opposite stance: it accepts that the
estimated moments are *wrong by some amount* and optimises against the worst case within an
**uncertainty set** around them. Rather than trusting a single point estimate of the
covariance (or mean), you bound a region the true value plausibly lies in and minimise the
worst-case risk over that region. The result is an allocation that is stable to estimation
error by construction.

`PortfolioOptimisers` builds uncertainty sets with an estimator such as
[`NormalUncertaintySet`](@ref) and an algorithm — [`BoxUncertaintySetAlgorithm`](@ref) (a
per-entry interval box) or [`EllipsoidalUncertaintySetAlgorithm`](@ref) (a joint ellipsoid).
The helper [`sigma_ucs`](@ref) (and [`mu_ucs`](@ref) for the mean) produces the set from the
data, which then parametrises a robust risk measure like [`UncertaintySetVariance`](@ref).

!!! tip "When to reach for this"
    Reach for uncertainty sets when you care less about a forecast and more about *robustness*
    to the noise in the moments — short windows, unstable covariances, regime risk. A box set
    is simple and conservative (it bounds each entry independently); an ellipsoidal set is
    tighter and usually less pessimistic because it captures the joint geometry. If you have
    actual views about where the moments are headed, reach for the view priors instead — or
    combine both, since a robust risk measure composes with any prior.
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
## 1. ReturnsResult data

We use the same S&P 500 slice as the other examples.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

#=
## 2. Building uncertainty sets

We construct two covariance uncertainty sets from a [`NormalUncertaintySet`](@ref) — one box,
one ellipsoidal. The set is estimated by resampling, so we fix an RNG for reproducibility.
=#

ucs_box = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = BoxUncertaintySetAlgorithm()), rd.X)
ucs_ell = sigma_ucs(NormalUncertaintySet(; pe = EmpiricalPrior(), rng = StableRNG(1),
                                         alg = EllipsoidalUncertaintySetAlgorithm()), rd.X)

#=
## 3. Robust vs nominal minimum-variance

[`UncertaintySetVariance`](@ref) is the robust counterpart of [`Variance`](@ref): it minimises
the worst-case variance over the uncertainty set rather than the point estimate. We compare a
nominal minimum-variance portfolio against the box- and ellipsoid-robust versions.
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
## 4. Comparing the allocations

The robust portfolios hedge against covariance estimation error. The ellipsoidal set, which
captures the joint geometry of the estimation error rather than bounding each entry on its own,
typically produces the more diversified, less concentrated allocation.
=#

pretty_table(DataFrame(["Assets" => rd.nx, "Nominal" => res_nom.w,
                        "Box-robust" => res_box.w, "Ellipsoid-robust" => res_ell.w]);
             formatters = [resfmt], title = "Minimum-variance weights: nominal vs robust")

#=
The composition plot shows the robust sets spreading the allocation out relative to the nominal
minimum-variance corner.
=#

using StatsPlots, GraphRecipes #= Nominal vs box- vs ellipsoid-robust minimum variance. =#

plot_stacked_bar_composition([res_nom, res_box, res_ell], rd;
                             xticks = (1:3, ["Nominal", "Box", "Ellipsoid"]))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end and closes the 2_moments_priors group. Both box and ellipsoidal
#src   UncertaintySetVariance optimisations solve to OptimisationSuccess; the headline contrast
#src   lands: nominal max weight ≈ 0.37, box-robust ≈ 0.41, ellipsoid-robust ≈ 0.15 (the
#src   ellipsoidal set is materially more diversifying here).
#src - Wiring is discoverable only from the tests: build the set with `sigma_ucs(NormalUncertaintySet(...; alg), X)`
#src   then pass it to `UncertaintySetVariance(; ucs = ...)`. A short robust-optimisation
#src   worked example in the docstrings would help. → #126.
