#=
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
                                                      regime_min_obs = 2, centred = true),
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

@testset "the masks behave the same through partial_fit! as through the pass" begin
    X = ra_sample(6102)
    emsk = trues(size(X))
    emsk[:, 4] .= false
    amsk = trues(size(X))
    amsk[20:29, 2] .= false

    for ce in RA_CONFIGS
        for kw in ((; estimation_mask = emsk), (; active_mask = amsk),
                   (; estimation_mask = emsk, active_mask = amsk))
            @test isequal(var(partial_fit!(ce, X; kw...)), var(ce, X; kw...))
        end

        # Sliced with the data, a mask reaches the incremental route one row at a time.
        rows = foldl(axes(X, 1); init = ce) do c, i
            return partial_fit!(c, view(X, i, :); estimation_mask = view(emsk, i, :),
                                active_mask = view(amsk, i, :))
        end
        @test isequal(var(rows), var(ce, X; estimation_mask = emsk, active_mask = amsk))
    end

    ce = first(RA_CONFIGS)
    @test_throws DimensionMismatch partial_fit!(ce, X; estimation_mask = trues(4, 4))
    @test_throws DimensionMismatch partial_fit!(ce, X; active_mask = trues(4, 4))
    @test_throws DomainError partial_fit!(ce, X; dims = 3)
    @test_throws DomainError partial_fit!(ce, X; dims = 0)
end

@testset "the state is a field on the estimator, and a fit over a sample ignores it" begin
    X = ra_sample(6103)
    ce = first(RA_CONFIGS)

    # The field is `nothing` until the first call, and the call returns a new estimator rather
    # than mutating the one it was given.
    @test isnothing(ce.cache)
    fitted = partial_fit!(ce, X)
    @test isnothing(ce.cache)
    @test isa(fitted.cache, PO.AbstractPartialFitState)
    @test isa(fitted.cache, PO.RegimeAdjustedVarianceCache)

    # An estimator carrying a state still answers any input it is given.
    @test isequal(var(fitted, X), var(ce, X))
    @test isequal(PO.variance_series(fitted, X), PO.variance_series(ce, X))

    # The two read-out forms are the same answer.
    @test isequal(var(fitted), var(fitted, fitted.cache))

    # An estimator with no state has nothing to read.
    @test_throws ArgumentError var(ce)

    # A state over a different number of assets is refused rather than folded.
    @test_throws DimensionMismatch partial_fit!(fitted, view(X, :, 1:3))

    # The keyword constructor carries the field, and the bound refuses an ordinary Result.
    @test isnothing(RegimeAdjustedExpWeightedVariance(; cache = nothing).cache)
    @test_throws TypeError RegimeAdjustedExpWeightedVariance(; cache = PO.ReturnsResult())
end

#=
This family does not merge two states, and the reason is measured here rather than asserted.

The regime state reads each observation's standardised squared innovation, and the reading is
gated by the running observation count. A block fitted from a cold start therefore skips its own
first `min_obs` observations, where the same block fitted after another one does not. The count of
regime observations is a field of the state, so the shortfall is visible in the state itself: no
function of the two block states can put back what neither of them recorded.
=#
@testset "the family refuses a merge, and the shortfall says why" begin
    X = ra_sample(6104)
    ce = RegimeAdjustedExpWeightedVariance(; decay = 0.94, min_obs = 5, regime_min_obs = 3,
                                           centred = true)

    whole = partial_fit!(ce, X).cache
    a = partial_fit!(ce, X[1:30, :]).cache
    b = partial_fit!(ce, X[31:60, :]).cache

    # The exponentially weighted accumulator does fold, as `decay^n_b * v_a + v_b`.
    @test isapprox(whole.variance, ce.decay .^ b.obs_count .* a.variance .+ b.variance;
                   rtol = 1e-14)
    # The regime state does not. The join loses exactly `min_obs` regime observations.
    @test whole.n_regime_obs == a.n_regime_obs + b.n_regime_obs + ce.min_obs

    # So the verb refuses, and it refuses a well-formed pair rather than only a mismatched one.
    @test_throws ArgumentError PO.merge_states(a, b)
    @test_throws ArgumentError PO.merge_states(a, a)
    # The generic refusals still run first.
    @test_throws DimensionMismatch PO.merge_states(a, partial_fit!(ce, X[:, 1:3]).cache)
end

@testset "the seam adds one name to the public surface" begin
    @test Base.isexported(PO, :partial_fit!)
    for name in (:AbstractPartialFitState, :RegimeAdjustedVarianceCache, :merge_states)
        @test !Base.isexported(PO, name)
    end
end
