#=
```@meta
Description = "Subset resampling under cross-validation in PortfolioOptimisers.jl: how stable a meta-optimiser's out-of-sample weights and frontier are."
```

# Subset resampling and cross-validation

The meta-optimiser page builds the three meta-optimisers. This page asks two further
questions:

  - how far do the out-of-sample predictions of a plain optimiser and of a meta-optimiser move
    from one fold to the next?
  - what does the efficient frontier look like when the optimiser draws random subsets of the
    universe and averages the weights?

[`MeanRisk`](@ref) is the benchmark here and [`SubsetResampling`](@ref) is the meta-optimiser.
The prior and the clustering are the ones the meta-optimiser page builds, so you can read the
allocations of the two pages against each other.

!!! tip "When to reach for this"
    Reach for subset resampling, and for a meta-optimiser in general, when one fit over every
    asset moves too far for you to act on. A small change to the estimation window can move
    the whole allocation, and an average over many drawn universes moves less than one point
    estimate. Cross-validation on this page measures that movement. It does not build the
    portfolio you hold.
=#

using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. ReturnsResult data and shared ingredients

We use the same S&P 500 slice as the other optimiser examples. We compute the prior and the
clustering once, and every cell below reads them.
=#

using CSV, TimeSeries, DataFrames, Clarabel, Statistics

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))]

pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)
jopti = JuMPOptimiser(; pe = pr, slv = slv)
jopto = JuMPOptimiser(; slv = slv)

#=
## 2. Reference allocations

We solve the plain minimum-variance portfolio and the three meta-optimisers. The
meta-optimiser page builds the same four. Sections 3 and 4 measure them.
=#

res_bench = optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv)))

res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(), opt = jopti),
                                   opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)

res_stk = optimise(Stacking(; pe = pr,
                            opti = [MeanRisk(; opt = jopti),
                                    HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       pe = pr)),
                                    InverseVolatility(; pe = pr)],
                            opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)

res_ssr = optimise(SubsetResampling(; pe = pr,
                                    opt = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv)),
                                    subset_size = 0.7, n_subsets = 10, rng = StableRNG(123),
                                    seed = 42), rd)

pretty_table(DataFrame(; :assets => rd.nx, :MinVar => res_bench.w, :NCO => res_nco.w,
                       :Stacking => res_stk.w, :SubsetResampling => res_ssr.w);
             formatters = [resfmt])

#=
Read the three meta-optimiser columns against the plain fit, and read the largest weight in
each. SubsetResampling averages over many smaller universes, so its weights sit closer
together than those of the other three. The plot below stacks the four allocations.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_bench, res_nco, res_stk, res_ssr], rd)

#=
## 3. Cross-validation prediction

We score the benchmark and the resampled optimiser under cross-validation.
[`cross_val_predict`](@ref) takes an estimator and returns one out-of-sample prediction per
fold, so the two sets of predictions are comparable.

The two optimisers we pass it carry no computed prior. Their `JuMPOptimiser` holds a solver
alone, so the prior is an estimator, the default [`EmpiricalPrior`](@ref), and it is fitted
again on each training fold. Cross-validation needs it that way. A prior computed once was
fitted on every observation, including the ones each test fold holds, so cross-validation
refuses it and asks for the estimator. The
[`MeanRisk` objectives](01_MeanRisk_Objectives.md) page covers that distinction.
=#

kfold = KFold(; n = 5)
cv_bench = cross_val_predict(MeanRisk(; opt = JuMPOptimiser(; slv = slv)), rd, kfold)
cv_ssr = cross_val_predict(SubsetResampling(;
                                            opt = MeanRisk(;
                                                           opt = JuMPOptimiser(; slv = slv)),
                                            subset_size = 0.7, n_subsets = 8,
                                            rng = StableRNG(123), seed = 42), rd, kfold)

scorer = NearestQuantilePrediction(; r = LowOrderMoment(; alg = SecondMoment()))
pp_bench = PopulationPredictionResult(; pred = [cv_bench])
pp_ssr = PopulationPredictionResult(; pred = [cv_ssr])
median_bench = scorer(pp_bench)
median_ssr = scorer(pp_ssr)

println("MeanRisk cross-val variance = $(expected_risk(LowOrderMoment(; alg = SecondMoment()), cv_bench))")
println("SubsetResampling cross-val variance = $(expected_risk(LowOrderMoment(; alg = SecondMoment()), cv_ssr))")

plot_cv_scores(LowOrderMoment(; alg = SecondMoment()), cv_bench)
plot_cv_scores(LowOrderMoment(; alg = SecondMoment()), cv_ssr)

#=
The scorer returns the path closest to the median of the population, so you do not choose a
path by hand. The two cells below print the `id` of the path it returned. A population
numbers its paths by position, and each population here holds one
[`cross_val_predict`](@ref) stream, so both print `1`. A population of many paths, such as
the one [`CombinatorialCrossValidation`](@ref) returns, gives the `id` of the path the scorer
selected.
=#

println("Median benchmark path id = $(median_bench.id)")
println("Median SSR path id = $(median_ssr.id)")

#=
## 4. Efficient frontier of a meta-optimiser

The frontier page runs one [`MeanRisk`](@ref) problem per point. We run the same sweep over
the resampled optimiser, so each point of the frontier is an average over drawn universes
rather than one fit over every asset. The table prints the largest weight at each point of
both frontiers, and the plot draws the two frontiers together.
=#

frontier_ret = ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = Frontier(; N = 15)))
mr_front = MeanRisk(; opt = JuMPOptimiser(; pe = pr, slv = slv, ret = frontier_ret))
ssr_front = SubsetResampling(; pe = pr,
                             opt = MeanRisk(;
                                            opt = JuMPOptimiser(; slv = slv,
                                                                ret = frontier_ret)),
                             subset_size = 0.7, n_subsets = 8, rng = StableRNG(123),
                             seed = 42)

res_mf = optimise(mr_front)
res_sf = optimise(ssr_front, rd)

rf = factory(Variance(), pr)
xs_m = [expected_risk(rf, w, pr.X) for w in res_mf.w]
ys_m = [expected_return(ArithmeticReturn(), w, pr) for w in res_mf.w]
xs_s = [expected_risk(rf, w, pr.X) for w in res_sf.w]
ys_s = [expected_return(ArithmeticReturn(), w, pr) for w in res_sf.w]

pretty_table(DataFrame(; :point => 1:length(res_mf.w),
                       :MeanRisk_max_w => [maximum(w) for w in res_mf.w],
                       :SubsetResampling_max_w => [maximum(w) for w in res_sf.w]);
             formatters = [resfmt])

plot(xs_m, ys_m; seriestype = :scatter, marker = (:circle, 5), label = "MeanRisk",
     xlabel = "Variance", ylabel = "Arithmetic return",
     title = "Frontier: plain optimiser vs bagged meta-optimiser")
plot!(xs_s, ys_s; seriestype = :scatter, marker = (:diamond, 6), label = "SubsetResampling")

#=
## 5. A risk-measure slot that follows the refit

Section 3 showed that cross-validation refuses a prior computed once. It cannot refuse a
matrix pasted into a risk measure the same way. `Variance(; sigma = S)` is a valid setting, and
a matrix the caller measured somewhere else looks the same as one fitted on the very sample the
portfolio is about to be scored on.

A slot that a prior fills therefore takes a second form, the estimator that computes the value
rather than the value. That is a [`DeferredQuantity`](@ref). It resolves against the prior the
optimisation runs on, once per subset and once per fold. What reaches the solver is still a
plain matrix, and only the moment the matrix is computed has moved.

The two forms give the same answer unless the estimator reads the universe or the window.
Denoising reads both. It cuts the eigenvalues below a threshold it derives from the ratio of
observations to assets, `T / N`, so the 14-asset block of a 20-asset fit is not a 14-asset
fit.
=#

ce_dn = PortfolioOptimisersCovariance(;
                                      mp = MatrixProcessing(;
                                                            dn = Denoise(;
                                                                         alg = FixedDenoise())))
sigma_full = cov(ce_dn, pr.X)

idx = 1:14
sigma_refit = cov(ce_dn, view(pr.X, :, idx))
sigma_slice = view(sigma_full, idx, idx)

println("Largest entry of the full-universe covariance   = $(maximum(abs, sigma_full))")
println("Refit vs sliced, on a 14-asset subset           = $(maximum(abs, sigma_refit .- sigma_slice))")

#=
### Inside a resample

[`SubsetResampling`](@ref) takes its view of the problem before it computes the prior. It
slices a matrix you state down to the assets it drew. A deferred quantity is not yet a matrix,
so it fits on the assets of the subset.
=#

ssr_rm = r -> SubsetResampling(; pe = pr,
                               opt = MeanRisk(; obj = MinimumRisk(), r = r,
                                              opt = JuMPOptimiser(; slv = slv)),
                               subset_size = 0.7, n_subsets = 10, rng = StableRNG(123),
                               seed = 42)

res_pasted = optimise(ssr_rm(Variance(; sigma = sigma_full)), rd)
res_deferred = optimise(ssr_rm(Variance(; sigma = ce_dn)), rd)

pretty_table(DataFrame(; :assets => rd.nx, :pasted_matrix => res_pasted.w,
                       :deferred_estimator => res_deferred.w); formatters = [resfmt])

#=
### Inside a fold

The same slot under cross-validation. The pasted matrix was fitted on all 252 observations,
so every test fold sits inside it. The estimator fits on the training fold alone.
=#

cv_pasted = cross_val_predict(MeanRisk(; obj = MinimumRisk(),
                                       r = Variance(; sigma = sigma_full),
                                       opt = JuMPOptimiser(; slv = slv)), rd, kfold)
cv_deferred = cross_val_predict(MeanRisk(; obj = MinimumRisk(),
                                         r = Variance(; sigma = ce_dn),
                                         opt = JuMPOptimiser(; slv = slv)), rd, kfold)

sm = LowOrderMoment(; alg = SecondMoment())
println("Pasted-matrix cross-val variance      = $(expected_risk(sm, cv_pasted))")
println("Deferred-estimator cross-val variance = $(expected_risk(sm, cv_deferred))")

#=
The pasted matrix gives the smaller variance. That is the warning, not the result. It was
fitted on the observations each test fold holds, so the score it earns is one no portfolio
could have earned at the time. Read the larger number the deferred estimator gives.

Every slot a prior fills works this way, which is `mu`, `sigma`, `kt` and `sk`. A measure with
two or more such slots takes a prior estimator in `pe` instead, and one fit fills every slot
you left unstated.

```julia
Kurtosis(; pe = EmpiricalPrior())                  # mu and kt from one fit
DistributionValueatRisk(; pe = EmpiricalPrior())   # mu, sigma and chol from one fit
```

A slot you state by hand keeps the value you gave it, and nothing checks it against the others.
The docstring of each measure says so. The library warns you rather than refuse the setting,
because a matrix measured elsewhere is a valid thing to pass.
=#

#=
## Summary

A meta-optimiser helps when one fit over every asset moves too far from window to window.

  - [`cross_val_predict`](@ref) scores the benchmark and the resampled optimiser out of
    sample, one prediction per fold.
  - [`SubsetResampling`](@ref) averages many subset solves, and its weights sit closer
    together than those of one fit.
  - A frontier sweep runs over a meta-optimiser, so you can read its whole curve against the
    curve of the plain optimiser rather than one portfolio against another.
  - A slot a prior fills takes the estimator rather than the value, so it fits again on every
    subset and every fold instead of holding one answer from the whole sample.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): MinVar benchmark plus NCO/Stacking/
#src   SubsetResampling, KFold `cross_val_predict` for the benchmark and the bagged optimiser,
#src   and a 15-point frontier of the meta-optimiser all solve with Clarabel.
#src - Narrative holds: SubsetResampling spreads weight the most (JNJ 20.5% vs 37% for MinVar),
#src   and at every frontier point the SSR max weight sits well below the plain MeanRisk max
#src   (75% vs 100% at the most aggressive point) — the "bagging smooths the frontier" point lands.
#src - RESOLVED (#1250): section 3 printed `nothing` for both ids, because a population built
#src   by hand from `cross_val_predict` streams carried no `id`. `PopulationPredictionResult`
#src   now gives a member without an `id` its position, so both cells print `1`.
#src - No solver warnings or plotting deprecations observed.
#src - Section 5 (Deferred Quantity, added for #286) checked in the test env with `julia -t 1`,
#src   BLAS 1, one Clarabel solver: refit-vs-sliced covariance difference 9.87e-5 against a
#src   largest entry of 1.57e-3; SubsetResampling weights differ by at most 1.35 pp between the
#src   pasted matrix and the deferred estimator; cross-val variance 8.79e-5 (pasted, leaked)
#src   vs 9.46e-5 (deferred, honest). The leak flatters the pasted score, which is the point.
#src   Re-check the printed numbers against the docs env, which uses the three-solver `slv`.
