#=
Strictness of observation weight resolution.

`get_observation_weights` returns `nothing` to mean *no weights were requested*, never
*weights were unavailable*. Every `isnothing` branch downstream reads it the first way and
computes an unweighted result, so a `DynamicAbstractWeights` that resolved to `nothing`
would silently produce a plausible-looking but unweighted answer. It raises instead.

See docs/adr/0043-nothing-observation-weights-means-unweighted-not-unavailable.md and issue
#177.

The weight types live at top level because `@testset` expands to a function body, which
cannot hold a `struct`.
=#

# Implements both documented arities.
struct CompleteObsWeights{T} <: PortfolioOptimisers.DynamicAbstractWeights
    half_life::T
end
function PortfolioOptimisers.get_observation_weights(w::CompleteObsWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(collect(eweights(1:length(X), 2^(-inv(w.half_life)); scale = true)))
end
function PortfolioOptimisers.get_observation_weights(w::CompleteObsWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(collect(eweights(1:size(X, dims), 2^(-inv(w.half_life)); scale = true)))
end

# Implements the vector arity ONLY. Handing this a matrix is the trap under test: before
# the split it resolved to `nothing` and every call site quietly went unweighted.
struct VectorOnlyObsWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::VectorOnlyObsWeights,
                                                     X::PortfolioOptimisers.VecNum;
                                                     kwargs...)
    return aweights(fill(inv(length(X)), length(X)))
end

const PO = PortfolioOptimisers

@testset "Observation weight resolution" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, Statistics

    rng = StableRNG(123456789)
    X = randn(rng, 60, 5)
    x = view(X, :, 1)

    @testset "Resolver contract" begin
        # `nothing` in, `nothing` out: no weights were requested.
        @test isnothing(PO.get_observation_weights(nothing, X; dims = 1))
        @test isnothing(PO.get_observation_weights(nothing, x))

        # Pre-computed weights pass straight through, untouched.
        ew = eweights(1:60, inv(60); scale = true)
        @test PO.get_observation_weights(ew, X; dims = 1) === ew

        # A complete dynamic type resolves on both arities, honouring `dims`.
        cw = CompleteObsWeights(10)
        @test length(PO.get_observation_weights(cw, X; dims = 1)) == 60
        @test length(PO.get_observation_weights(cw, X; dims = 2)) == 5
        @test length(PO.get_observation_weights(cw, x)) == 60

        # The vector arity still works for a partially-implemented type...
        @test length(PO.get_observation_weights(VectorOnlyObsWeights(), x)) == 60
        # ...but the shape it never implemented raises rather than silently unweighting.
        @test_throws PO.ObservationWeightsError PO.get_observation_weights(VectorOnlyObsWeights(),
                                                                           X; dims = 1)
    end

    @testset "ObservationWeightsError" begin
        @test PO.ObservationWeightsError <: PO.PortfolioOptimisersError

        err = try
            PO.get_observation_weights(VectorOnlyObsWeights(), X; dims = 1)
        catch e
            e
        end
        msg = sprint(showerror, err)
        # The message must name the offending type, the shape it was handed, and both
        # signatures to implement - it is the only guidance a user gets.
        @test occursin("ObservationWeightsError", msg)
        @test occursin("VectorOnlyObsWeights", msg)
        @test occursin("2-dimensional input of size (60, 5)", msg)
        @test occursin("X::VecNum", msg)
        @test occursin("X::MatNum", msg)
        @test occursin("DynamicAbstractWeights", msg)
    end

    @testset "moment_window_and_weights" begin
        idx = collect(21:30)

        # All four methods resolve a complete type and reject a partial one. The two
        # windowed methods subset first, so they must still raise on the sliced view.
        cw = CompleteObsWeights(10)
        Xw, w = PO.moment_window_and_weights(X, cw; dims = 1)
        @test size(Xw) == (60, 5)
        @test length(w) == 60
        Xw, w = PO.moment_window_and_weights(X, cw, idx; dims = 1)
        @test size(Xw) == (10, 5)
        @test length(w) == 10
        xw, w = PO.moment_window_and_weights(x, cw)
        @test length(w) == 60
        xw, w = PO.moment_window_and_weights(x, cw, idx)
        @test length(w) == 10

        vw = VectorOnlyObsWeights()
        @test_throws PO.ObservationWeightsError PO.moment_window_and_weights(X, vw;
                                                                             dims = 1)
        @test_throws PO.ObservationWeightsError PO.moment_window_and_weights(X, vw, idx;
                                                                             dims = 1)

        # `nothing` weights remain permissive through the same helper: the strictness must
        # not leak into the legitimately-unweighted path.
        Xw, w = PO.moment_window_and_weights(X, nothing; dims = 1)
        @test isnothing(w)
        Xw, w = PO.moment_window_and_weights(X, nothing, idx; dims = 1)
        @test isnothing(w)
    end

    @testset "08_Moments estimators" begin
        vw = VectorOnlyObsWeights()
        cw = CompleteObsWeights(10)

        # The weights genuinely reach the estimator: weighted != unweighted.
        @test !isapprox(cor(GeneralCovariance(), X), cor(GeneralCovariance(; w = cw), X))
        @test !isapprox(std(SimpleVariance(), X), std(SimpleVariance(; w = cw), X))
        @test !isapprox(mean(SimpleExpectedReturns(), X),
                        mean(SimpleExpectedReturns(; w = cw), X))

        # `cov(::GeneralCovariance, ...)` used to pass `ce.w` to `robust_cov` raw, never
        # resolving it, so ANY dynamic weights type - even a complete one - died with a
        # MethodError while its sibling `cor` worked. It now resolves like `cor` does, and
        # agrees with passing the equivalent static weights directly.
        @test isapprox(cov(GeneralCovariance(; w = cw), X),
                       cov(GeneralCovariance(;
                                             w = PO.get_observation_weights(cw, X;
                                                                            dims = 1)), X))

        # An unresolvable type raises instead of returning an unweighted moment.
        @test_throws PO.ObservationWeightsError cor(GeneralCovariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError cov(GeneralCovariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError std(SimpleVariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError var(SimpleVariance(; w = vw), X)
        @test_throws PO.ObservationWeightsError mean(SimpleExpectedReturns(; w = vw), X)

        # Unweighted estimators are untouched.
        @test isa(cor(GeneralCovariance(), X), Matrix)
        @test isa(mean(SimpleExpectedReturns(), X), AbstractArray)
    end

    @testset "19_RiskMeasures resolve-then-rebuild" begin
        # These measures dispatch on the weights type, resolve, then rebuild themselves and
        # re-dispatch. A `nothing` resolution used to rebuild as an unweighted measure and
        # silently re-dispatch to the unweighted method - the same defect, one layer deeper.
        w = fill(inv(5), 5)
        cw = CompleteObsWeights(10)
        vw = VectorOnlyObsWeights()

        @test isa(LowOrderMoment(; w = cw)(w, X), Number)
        @test isa(HighOrderMoment(; w = cw)(w, X), Number)
        @test_throws PO.ObservationWeightsError LowOrderMoment(; w = vw)(w, X)
        @test_throws PO.ObservationWeightsError HighOrderMoment(; w = vw)(w, X)

        # Unweighted and pre-weighted measures behave exactly as before.
        @test isa(LowOrderMoment()(w, X), Number)
        @test isa(LowOrderMoment(; w = eweights(1:60, inv(60); scale = true))(w, X), Number)
    end
end
