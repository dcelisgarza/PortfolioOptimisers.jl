#=
The checks of this subject are split over the files `test_08p_regime_partial_fit*.jl`, so that each file of the
suite runs in about the same time. This file holds one part of them.

The state of the regime-adjusted recursion now survives a call.

`RegimeAdjustedExpWeightedVariance` already held the whole recursion: a state struct, a mutating
verb over one observation, a read-out, and a forward pass that ties the three together. What it did
not hold was the state itself. The pass built the cache and dropped it, so no estimator was
incremental between calls. Issue #308 decided where the state lives, #699 built the root, and this
file covers the lift: the cache is a field on the estimator, `partial_fit!` folds observations into
it, and `var(ce)` reads it.

The promise the file tests is exactness. An incremental fit that does not reproduce the batch
answer to the last bit is a second estimator wearing the first one's name, so every comparison here
is `isequal` rather than a tolerance.

The last testset is the one negative result. This family does **not** merge two states, and the
testset measures why rather than asserting the refusal alone.
=#
using Test, PortfolioOptimisers, Statistics, StableRNGs, LinearAlgebra

const PO = PortfolioOptimisers

# The five configurations that reach every branch of the recursion: the default, a centred fit, a
# HAC fit, and the two regime methods that are not the default.
const RA_CONFIGS = (RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                      regime_min_obs = 3),
                    RegimeAdjustedExpWeightedVariance(; decay = 0.9, min_obs = 4,
                                                      regime_min_obs = 2,
                                                      centring = PreCentred()),
                    RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                      regime_min_obs = 3, hac_lags = 2),
                    RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                      regime_min_obs = 3,
                                                      regime_method = PO.LogRegimeAdjusted()),
                    RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5,
                                                      regime_min_obs = 3,
                                                      regime_method = PO.RootMeanSquaredAdjusted()))

function ra_sample(seed::Integer = 6101)
    X = randn(StableRNG(seed), 60, 4) .* 0.02
    X[40:end, :] .*= 3.0        # a loud regime, so the regime multiplier is not one
    return X
end

@testset "one observation at a time is the fit over the whole sample" begin
    X = ra_sample()
    for ce in RA_CONFIGS
        rows = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce)
        # The recursion runs the same operations in the same order either way, so the two agree
        # bit for bit rather than to a tolerance.
        @test isequal(var(rows), var(ce, X))
        # One call over the matrix is the same fit as one call per row.
        @test isequal(var(partial_fit!(ce, X)), var(ce, X))
        # With the observations on the columns, the matrix method reads the other axis.
        @test isequal(var(partial_fit!(ce, permutedims(X); dims = 2)), var(ce, X))
    end
end

@testset "the state survives a call" begin
    X = ra_sample()
    for ce in RA_CONFIGS
        halves = partial_fit!(partial_fit!(ce, X[1:30, :]), X[31:60, :])
        @test isequal(var(halves), var(ce, X))

        # Three blocks of unequal length, so nothing rides on the split being the midpoint.
        thirds = partial_fit!(partial_fit!(partial_fit!(ce, X[1:7, :]), X[8:41, :]),
                              X[42:60, :])
        @test isequal(var(thirds), var(ce, X))

        # A row folded on its own continues a block folded as a matrix.
        mixed = partial_fit!(partial_fit!(ce, X[1:59, :]), view(X, 60, :))
        @test isequal(var(mixed), var(ce, X))
    end
end
