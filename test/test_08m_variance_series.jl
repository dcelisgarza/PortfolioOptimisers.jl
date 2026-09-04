#=
A point-in-time variance series reads no observation after its own.

The two-pass cross-sectional regression weights an asset at observation `t` by the inverse of the
idiosyncratic variance estimated from residuals up to `t - 1`. A variance that already carries the
date-`t` squared residual embeds the return being regressed, so an asset with a large date-`t`
shock is down-weighted at date `t` itself, and the weights correlate with the residuals.

`variance_series` is that quantity. Its default refits on an expanding window, so every member of
the covariance surface answers it. `RegimeAdjustedExpWeightedVariance` overrides the default with a
single forward pass, because its estimate is a recursion over one observation. The two must agree
exactly: the fast route reads the cache after observation `t`, and the slow route rebuilds that
same cache from observations `1` to `t`.

The file also holds this estimator's first tests. Two defects it found are fixed in the same
change, and the last two testsets are their regressions.
=#
using Test, PortfolioOptimisers, Statistics, StableRNGs, LinearAlgebra

const PO = PortfolioOptimisers

# The expanding-window fallback, reached past the estimator's own override.
function slow_series(ce, X; dims::Int = 1, kwargs...)
    return invoke(PO.variance_series, Tuple{PO.AbstractCovarianceEstimator, PO.MatNum}, ce,
                  X; dims = dims, kwargs...)
end

# The expanding window written out by hand, slicing every per-observation keyword with the data.
# This is what the fallback cannot do, because it cannot know which keyword carries observations.
function masked_slow_series(ce, X; estimation_mask = nothing, active_mask = nothing)
    return permutedims(reduce(hcat,
                              [var(ce, view(X, 1:t, :);
                                   estimation_mask = if isnothing(estimation_mask)
                                       nothing
                                   else
                                       view(estimation_mask, 1:t, :)
                                   end, active_mask = if isnothing(active_mask)
                                       nothing
                                   else
                                       view(active_mask, 1:t, :)
                                   end) for t in axes(X, 1)]))
end

@testset "the fallback refits on an expanding window" begin
    rng = StableRNG(4321)
    X = randn(rng, 12, 3) .* 0.03

    for ce in (SimpleVariance(), Covariance(), SimpleVariance(; corrected = false))
        val = PO.variance_series(ce, X)
        @test size(val) == size(X)
        for t in axes(X, 1)
            @test isequal(val[t, :], vec(var(ce, view(X, 1:t, :))))
        end
        # `dims = 2` names the same series with the observations on the columns.
        @test isequal(PO.variance_series(ce, permutedims(X); dims = 2), permutedims(val))
    end

    # Row 1 is a fit on one observation, so a corrected estimator has nothing to divide by.
    @test all(isnan, PO.variance_series(SimpleVariance(), X)[1, :])
    @test all(iszero, PO.variance_series(SimpleVariance(; corrected = false), X)[1, :])

    # The guard is the family's, not this verb's.
    @test_throws DomainError PO.variance_series(SimpleVariance(), X; dims = 3)
    @test_throws DomainError PO.variance_series(SimpleVariance(), X; dims = 0)
end

@testset "the recursive override is one forward pass over the same cache" begin
    rng = StableRNG(4322)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 4.0        # a loud regime, so the regime multiplier is not one

    for ce in
        (RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3),
         RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 4, regime_min_obs = 2,
                                           centred = true),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           hac_lags = 2),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.LogRegimeAdjusted()),
         RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           regime_method = PO.RootMeanSquaredAdjusted()))
        fast = PO.variance_series(ce, X)
        @test size(fast) == size(X)
        # The recursion and the refit run the same operations in the same order, so the two
        # agree bit for bit rather than to a tolerance.
        @test isequal(fast, slow_series(ce, X))
        # The last row is the whole-sample estimate the family's own verb returns.
        @test isequal(fast[end, :], var(ce, X))
        @test isequal(PO.variance_series(ce, permutedims(X); dims = 2), permutedims(fast))
    end

    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3)
    @test_throws DomainError PO.variance_series(ce, X; dims = 3)
    @test_throws DomainError PO.variance_series(ce, X; dims = 0)
end

@testset "row t reads no observation after t" begin
    rng = StableRNG(4323)
    X = randn(rng, 60, 3) .* 0.02
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3)

    base = PO.variance_series(ce, X)
    for t in (10, 31, 60)
        Xp = copy(X)
        Xp[t, :] .+= 5.0
        pert = PO.variance_series(ce, Xp)
        # Everything before the change is untouched, and the changed observation is felt at once.
        @test isequal(base[1:(t - 1), :], pert[1:(t - 1), :])
        @test !isequal(base[t, :], pert[t, :])
    end

    # The fallback carries the same promise, on an estimator with no recursion.
    sbase = PO.variance_series(SimpleVariance(), X)
    Xp = copy(X)
    Xp[31, :] .+= 5.0
    @test isequal(sbase[1:30, :], PO.variance_series(SimpleVariance(), Xp)[1:30, :])
end

@testset "the masks ride the observation axis" begin
    rng = StableRNG(4324)
    X = randn(rng, 50, 3) .* 0.02
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    emsk = trues(size(X))
    emsk[:, 3] .= false
    amsk = trues(size(X))
    amsk[20:29, 2] .= false

    for kw in ((; estimation_mask = emsk), (; active_mask = amsk),
               (; estimation_mask = emsk, active_mask = amsk))
        fast = PO.variance_series(ce, X; kw...)
        @test isequal(fast, masked_slow_series(ce, X; kw...))
        @test isequal(fast[end, :], var(ce, X; kw...))
    end

    # A mask that does not match the data is refused, as it is for the whole-sample verb.
    @test_throws DimensionMismatch PO.variance_series(ce, X; estimation_mask = trues(4, 4))
    @test_throws DimensionMismatch PO.variance_series(ce, X; active_mask = trues(4, 4))

    # With the observations on the columns, a mask is sliced by column too.
    @test isequal(PO.variance_series(ce, permutedims(X); dims = 2,
                                     estimation_mask = permutedims(emsk),
                                     active_mask = permutedims(amsk)),
                  permutedims(PO.variance_series(ce, X; estimation_mask = emsk,
                                                 active_mask = amsk)))
end

@testset "an observation the pass cannot use leaves the cache where it was" begin
    rng = StableRNG(4327)
    X = randn(rng, 30, 3) .* 0.02
    X[7, :] .= NaN          # no asset is valid, so the observation updates nothing
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    series = PO.variance_series(ce, X)
    @test isequal(series[7, :], series[6, :])
    @test isequal(series, slow_series(ce, X))

    # A floor above every variance the sample reaches leaves every standardised innovation
    # `NaN`, so the regime state never has a reading to smooth.
    quiet = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4,
                                              regime_min_obs = 2, min_val = 1.0)
    qseries = PO.variance_series(quiet, X)
    @test isequal(qseries, slow_series(quiet, X))
    # The multiplier stays at one, so the series is the unadjusted variance.
    @test isequal(qseries[end, :], var(quiet, X))
end

#=
Defect 1. `regime_adjusted_variance` corrected the exponentially weighted bias by indexing the
correction with a mask and assigning a full-length right side, so `var` threw a `DimensionMismatch`
whenever any asset held no observation at all. An asset with no finite return, and an asset that a
whole-sample `active_mask` switches off, are both that case. The expanding-window fallback meets it
at the first observation of any such asset, so the verb this file adds cannot exist without the fix.
=#
@testset "an asset with no observation is NaN, and does not throw" begin
    rng = StableRNG(4325)
    X = randn(rng, 40, 3) .* 0.02
    X[:, 3] .= NaN
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 4, regime_min_obs = 2)

    val = var(ce, X)
    @test isnan(val[3])
    @test all(isfinite, view(val, 1:2))

    series = PO.variance_series(ce, X)
    @test all(isnan, view(series, :, 3))
    @test isequal(series, slow_series(ce, X))

    # The same case reached through a mask that is false for the whole sample.
    amsk = trues(size(X))
    amsk[:, 2] .= false
    @test isnan(var(ce, randn(StableRNG(1), 40, 3) .* 0.02; active_mask = amsk)[2])
end

#=
Defect 2. `process_observation!` advances `regime_state` and `n_regime_obs` with
`Accessors.@reset`, which rebinds the function's own local because the cache is immutable. The
caller kept its original cache, so `n_regime_obs` never left zero, the multiplier was never
reached, and every regime field on the estimator was inert. The function now returns the cache and
the pass rebinds it.
=#
@testset "the regime state accumulates across observations" begin
    rng = StableRNG(4326)
    X = randn(rng, 120, 3) .* 0.02
    X[90:end, :] .*= 5.0

    for rm in (PO.FirstMomentRegimeAdjusted(), PO.LogRegimeAdjusted(),
               PO.RootMeanSquaredAdjusted())
        live = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                 regime_min_obs = 3, regime_method = rm)
        # A regime threshold the sample cannot reach leaves the multiplier at one, which is the
        # answer the whole family used to give.
        inert = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                  regime_min_obs = 10_000,
                                                  regime_method = rm)
        @test !isequal(var(live, X), var(inert, X))
        @test !isequal(PO.variance_series(live, X), PO.variance_series(inert, X))
    end
end
#=
`regime_method = nothing` is the off switch, added by issue #718 for `EWVolatility`. Before it, the
only way to read the plain exponentially weighted recursion was a `regime_min_obs` the sample
cannot reach, which is a property of the data and not of the estimator. The switch advances no
regime state, so `n_regime_obs` stays at zero and the multiplier stays at one whatever the sample
holds.
=#
@testset "no regime method is the plain exponentially weighted recursion" begin
    rng = StableRNG(718)
    X = randn(rng, 90, 4) .* 0.02
    X[70:end, :] .*= 6.0
    decay, min_obs = exp2(-inv(5.0)), 5

    # The recursion written out by hand: uncentred, bias corrected, no regime factor.
    function plain_series(X, decay, min_obs)
        T, N = size(X)
        V = fill(NaN, T, N)
        for i in 1:N
            s, n = 0.0, 0
            for t in 1:T
                if isfinite(X[t, i])
                    s = decay * s + (1 - decay) * X[t, i]^2
                    n += 1
                end
                if n >= min_obs
                    V[t, i] = s / (1 - decay^n)
                end
            end
        end
        return V
    end

    off = RegimeAdjustedExpWeightedVariance(; decay = decay, min_obs = min_obs,
                                            centred = true, regime_method = nothing)
    expected = plain_series(X, decay, min_obs)
    @test isapprox(PO.variance_series(off, X), expected; rtol = 1e-12, nans = true)
    @test isapprox(vec(var(off, X)), expected[end, :]; rtol = 1e-12, nans = true)

    # The default still applies the multiplier, so the switch changes an answer.
    on = RegimeAdjustedExpWeightedVariance(; decay = decay, min_obs = min_obs,
                                           centred = true)
    @test !isapprox(PO.variance_series(on, X), expected; rtol = 1e-12, nans = true)

    # The slow route agrees with the fast one, as it does for every other setting.
    @test isapprox(slow_series(off, X), PO.variance_series(off, X); rtol = 1e-12,
                   nans = true)

    # An asset that turns inactive restarts, and the switch does not change that.
    amsk = trues(size(X))
    amsk[40:45, 2] .= false
    Xm = ifelse.(amsk, X, NaN)
    got = PO.variance_series(off, Xm; active_mask = amsk)
    @test all(isnan, view(got, 40:45, 2))
    @test isapprox(got[50, 2], plain_series(view(Xm, 46:50, 2:2), decay, min_obs)[end, 1];
                   rtol = 1e-12)
end
