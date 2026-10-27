#=
Regression tests for `robust_cov`/`robust_cor` keyword dispatch: unsupported keyword
arguments are dropped via `hasmethod` dispatch rather than error-swallowing, so genuine
errors thrown by the estimator propagate to the caller.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, Clarabel

struct KwargsCov <: StatsBase.CovarianceEstimator end
function Statistics.cov(::KwargsCov, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                        scale::Real = 1.0)
    return scale * Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims)
end
struct PlainCov <: StatsBase.CovarianceEstimator end
function Statistics.cov(::PlainCov, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims)
end
struct ThrowingCov <: StatsBase.CovarianceEstimator end
function Statistics.cov(::ThrowingCov, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return throw(DomainError(dims, "genuine numerical failure"))
end

@testset "robust_cov/robust_cor keyword dispatch" begin
    rng = StableRNG(987654321)
    X = randn(rng, 40, 5)
    sigma = Statistics.cov(StatsBase.SimpleCovariance(), X)
    # Supported keyword arguments are forwarded.
    @test PortfolioOptimisers.robust_cov(KwargsCov(), X; scale = 2.0) ≈ 2 * sigma
    # Unsupported keyword arguments are dropped by dispatch, not by retry-on-error.
    @test PortfolioOptimisers.robust_cov(PlainCov(), X; scale = 2.0) ≈ sigma
    # Genuine errors propagate instead of being masked by a kwarg-less retry.
    @test_throws DomainError PortfolioOptimisers.robust_cov(ThrowingCov(), X)
    # `StatsBase` gives every `CovarianceEstimator` a slurping
    # `cor(ce, X::AbstractMatrix; kwargs...)`, so this estimator answers `cor` through that
    # generic and the extra keyword rides to its own `cov`. A correlation is scale free, so
    # the scaled covariance still gives the plain correlation. This is NOT the
    # covariance-to-correlation fallback, which `test_08b_robust_moments.jl` drives.
    @test PortfolioOptimisers.robust_cor(KwargsCov(), X; scale = 2.0) ≈ Statistics.cor(X)
    @test_throws DomainError PortfolioOptimisers.robust_cor(ThrowingCov(), X)
    # StatsBase's generic `cor(ce, X, w; kwargs...)` wrapper slurps kwargs (so it passes
    # the `hasmethod` check) but its inner `cov` rejects unknown ones with a MethodError:
    # the call must be retried without them, not fail.
    w = StatsBase.eweights(1:size(X, 1), 0.3)
    @test PortfolioOptimisers.robust_cov(StatsBase.SimpleCovariance(), X, w; iv = nothing) ≈
          Statistics.cov(StatsBase.SimpleCovariance(), X, w)
    @test PortfolioOptimisers.robust_cor(StatsBase.SimpleCovariance(), X, w; iv = nothing) ≈
          Statistics.cor(StatsBase.SimpleCovariance(), X, w)
end

#=
Regression tests for the lazy-observation-matrix hazard (ADR 0065). `StatsBase`'s weighted
moment heads are typed on `DenseMatrix`. With a `mean`, a non-dense `X` makes
`covm(X, mean, w, dims)` resolve to `Statistics.covm(x, xmean, y, ymean, vardim)` — the
cross-covariance of `X` against the weight vector — which returns `N × 1` and raises nothing,
so the `MethodError` retry could not see it. The weighted seam densifies instead.
=#
@testset "robust_cov/robust_cor densify lazy observations" begin
    rng = StableRNG(987654321)
    X = randn(rng, 40, 5)
    N = size(X, 2)
    sc = StatsBase.SimpleCovariance()
    w = StatsBase.eweights(1:size(X, 1), 0.3)
    mu = StatsBase.mean(X, w; dims = 1)
    # `densify` is the identity on a dense matrix and materialises everything else.
    @test PortfolioOptimisers.densify(X) === X
    @test PortfolioOptimisers.densify(transpose(permutedims(X))) isa Matrix
    @test PortfolioOptimisers.densify(view(X, :, :)) isa Matrix
    # A transposed, a viewed and a dense `X` give the same weighted moments, with a `mean`
    # and without one. Before ADR 0065 the first two returned an `N × 1` cross-covariance.
    for Xl in (transpose(permutedims(X)), view(X, 1:size(X, 1), :))
        @test size(PortfolioOptimisers.robust_cov(sc, Xl, w; mean = mu)) == (N, N)
        @test PortfolioOptimisers.robust_cov(sc, Xl, w; mean = mu) ≈
              Statistics.cov(sc, X, w; mean = mu)
        @test PortfolioOptimisers.robust_cov(sc, Xl, w) ≈ Statistics.cov(sc, X, w)
        @test size(PortfolioOptimisers.robust_cor(sc, Xl, w; mean = mu)) == (N, N)
        @test PortfolioOptimisers.robust_cor(sc, Xl, w) ≈ Statistics.cor(sc, X, w)
    end
    # A weighted windowed covariance passes a `SubArray`, and every prior orients `dims = 2`
    # through a `Transpose`. Both now agree with the dense answer.
    ce = PortfolioOptimisers.Covariance(; me = SimpleExpectedReturns(; w = w),
                                        ce = PortfolioOptimisers.GeneralCovariance(;
                                                                                   ce = sc,
                                                                                   w = w))
    @test Statistics.cov(ce, transpose(permutedims(X)); dims = 1) ≈
          Statistics.cov(ce, X; dims = 1)
    @test Statistics.cov(ce, view(X, 1:size(X, 1), :); dims = 1) ≈
          Statistics.cov(ce, X; dims = 1)
    pe = EntropyPoolingPrior(; alg = H1_EntropyPooling())
    @test prior(pe, permutedims(X); dims = 2).sigma ≈ prior(pe, X).sigma
end

# #1262: a field bound to `StatsBase.CovarianceEstimator` admits a bare estimator that the
# library does not own. The library calls a nested estimator with its own keywords and with
# an Asset Panel argument, which a `StatsBase` method refuses. Every bare spelling must
# answer what its `GeneralCovariance` spelling answers.
@testset "a bare StatsBase estimator nested in a library estimator (#1262)" begin
    rng = StableRNG(1262)
    X = 0.02 .* randn(rng, 30, 4)
    sc = StatsBase.SimpleCovariance()
    gc = GeneralCovariance(; ce = sc)
    kw = (; iv = nothing, ivpa = nothing)
    @test PortfolioOptimisers.library_covariance_estimator(gc) === gc
    @test PortfolioOptimisers.library_covariance_estimator(sc) isa GeneralCovariance
    pairs = ((Covariance(; ce = sc), Covariance(; ce = gc)),
             (Covariance(; ce = sc, alg = SemiMoment()),
              Covariance(; ce = gc, alg = SemiMoment())),
             (CorrelationCovariance(; ce = sc), CorrelationCovariance(; ce = gc)),
             (PortfolioOptimisersCovariance(; ce = sc),
              PortfolioOptimisersCovariance(; ce = gc)),
             (DenoiseCovariance(; ce = sc), DenoiseCovariance(; ce = gc)))
    for (bare, wrapped) in pairs, f in (Statistics.cov, Statistics.cor)
        @test f(bare, X; kw...) == f(wrapped, X; kw...)
        @test f(bare, X, nothing; kw...) == f(wrapped, X, nothing; kw...)
    end
    # `ce.w` weights a library estimator and passes a bare one through unchanged, so the
    # weighted bare spelling is its own estimator. The keywords must not move its answer.
    bw = Covariance(; ce = sc, w = eweights(30, 0.1))
    for f in (Statistics.cov, Statistics.cor)
        @test f(bw, X; kw...) == f(bw, X)
    end
    @test prior(EmpiricalPrior(; ce = sc), X; kw...).sigma ==
          prior(EmpiricalPrior(; ce = gc), X; kw...).sigma
    for me in (ShrunkExpectedReturns, EquilibriumExpectedReturns,
               StandardDeviationExpectedReturns, VarianceExpectedReturns)
        @test mean(me(; ce = sc), X; kw...) == mean(me(; ce = gc), X; kw...)
        @test mean(me(; ce = sc), X, nothing; kw...) ==
              mean(me(; ce = gc), X, nothing; kw...)
    end
    # #1263: the two diagonal estimators called `std` and `var` on the bare estimator, which
    # `StatsBase` does not define for a `CovarianceEstimator`.
    d = LinearAlgebra.diag(Statistics.cov(sc, X))
    @test vec(mean(VarianceExpectedReturns(; ce = sc), X)) ≈ d
    @test vec(mean(StandardDeviationExpectedReturns(; ce = sc), X)) ≈ sqrt.(d)
    @test prior(EmpiricalPrior(; me = StandardDeviationExpectedReturns(; ce = sc)), X).mu ≈
          sqrt.(d)
    rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false))
    w = map((Covariance(; ce = sc), Covariance(; ce = gc))) do ce
        pe = EmpiricalPrior(; ce = PortfolioOptimisersCovariance(; ce = ce))
        return optimise(MeanRisk(; opt = JuMPOptimiser(; pe = pe, slv = slv)), rd).w
    end
    @test w[1] == w[2]
end

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

# Implements the matrix arity ONLY, as a type that reads the returns matrix does. A moment
# measure that handed it to a nested variance estimator met the vector arity and raised.
struct MatrixOnlyObsWeights <: PortfolioOptimisers.DynamicAbstractWeights end
function PortfolioOptimisers.get_observation_weights(::MatrixOnlyObsWeights,
                                                     X::PortfolioOptimisers.MatNum;
                                                     dims::Int = 1, kwargs...)
    return aweights(collect(range(1.0, 2.0; length = size(X, dims))))
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

    @testset "kwargs.weights constructor guards" begin
        # Three constructors screen a `weights` entry that they forward to a third-party
        # kernel. The check itself always worked. Its failure path did not: ArgCheck builds
        # the exception with `T(msg::String)`, and `Core.TypeError` has no such
        # constructor, so a wrong type raised `MethodError: no method matching
        # TypeError(::String)` and named neither the keyword nor the type received.
        bad = (; weights = 1.0)
        bad_tgt = LinearModel(; kwargs = bad)
        # `Clustering.kmeans` weights its points, which are the assets, so the k-means
        # message names a point weight. The two regression kernels weight observations.
        for (sym, quantity, f) in
            (("kwargs.weights", "point weights", () -> KMeansAlgorithm(; kwargs = bad)),
             ("tgt.kwargs.weights", "observation weights",
              () -> StepwiseRegression(; tgt = bad_tgt)),
             ("retgt.kwargs.weights", "observation weights",
              () -> DimensionReductionRegression(; retgt = bad_tgt)))
            @test_throws ArgumentError f()
            msg = sprint(showerror, try
                             f()
                         catch e
                             e
                         end)
            # The message must name the offending keyword and the type received.
            @test occursin(sym, msg)
            @test occursin(quantity, msg)
            @test occursin("Float64", msg)
        end

        # The emptiness guard beside it is unaffected: its exception takes a message.
        empty_tgt = LinearModel(; kwargs = (; weights = aweights(Float64[])))
        @test_throws PO.IsEmptyError KMeansAlgorithm(; kwargs = (; weights = Float64[]))
        @test_throws PO.IsEmptyError StepwiseRegression(; tgt = empty_tgt)
        @test_throws PO.IsEmptyError DimensionReductionRegression(; retgt = empty_tgt)

        # A well-formed `weights` entry still constructs.
        good = (; weights = aweights(fill(inv(60), 60)))
        good_tgt = LinearModel(; kwargs = good)
        @test KMeansAlgorithm(; kwargs = good).kwargs === good
        @test StepwiseRegression(; tgt = good_tgt).tgt.kwargs === good
        @test DimensionReductionRegression(; retgt = good_tgt).retgt.kwargs === good
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

    @testset "05_Moments estimators" begin
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

    @testset "16_RiskMeasures resolve-then-rebuild" begin
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

        # All three central-moment measures that read the returns rebuild through one
        # generic, so the rule is asserted on all three, and on both forms of the third
        # moment. `Kurtosis` and `Skewness` read a comoment matrix since #1846, so they
        # hold no observation weights.
        rw = PO.get_observation_weights(cw, X; dims = 1)
        for (dyn, sta) in ((LowOrderMoment(; w = cw), LowOrderMoment(; w = rw)),
                           (HighOrderMoment(; w = cw), HighOrderMoment(; w = rw)),
                           (HighOrderMoment(; w = cw, alg = PO.FourthMoment(; alg = PO.SemiMoment())),
                            HighOrderMoment(; w = rw, alg = PO.FourthMoment(; alg = PO.SemiMoment()))),
                           (PO.ThirdCentralMoment(; w = cw), PO.ThirdCentralMoment(; w = rw)),
                           (PO.ThirdCentralMoment(; w = cw, ve = SimpleVariance()),
                            PO.ThirdCentralMoment(; w = rw, ve = SimpleVariance())))
            # Resolving the weights is all the rebuild does: the answer is the one the
            # measure gives when it is handed the same weights directly.
            @test dyn(w, X) == sta(w, X)
            @test dyn(X * w) == sta(X * w)
            # An unresolvable type still raises rather than going quietly unweighted.
            dyn_vw = PO.Accessors.@set dyn.w = vw
            @test_throws PO.ObservationWeightsError dyn_vw(w, X)
        end

        # The rebuild replaces `w` and the nested estimator, and copies every other field,
        # so a non-default setting survives it. An earlier set of hand-written rebuilds
        # dropped fields: `Skewness` reset its `settings`.
        for r in (LowOrderMoment(; w = cw, settings = RiskMeasureSettings(; scale = 3),
                                 mu = 0.001, alg = SecondMoment()),
                  HighOrderMoment(; w = cw, settings = RiskMeasureSettings(; scale = 3),
                                  mu = 0.001, alg = PO.StandardisedHighOrderMoment()),
                  PO.ThirdCentralMoment(; w = cw,
                                        settings = PO.HierarchicalRiskMeasureSettings(; scale = 3),
                                        mu = 0.001),
                  PO.ThirdCentralMoment(; w = cw,
                                        settings = PO.HierarchicalRiskMeasureSettings(; scale = 3),
                                        mu = 0.001, ve = SimpleVariance()))
            rebuilt = PO.resolve_observation_weights(r, X)
            @test rebuilt.w == rw
            @test all(f -> f in (:w, :alg, :ve) || getfield(rebuilt, f) === getfield(r, f),
                      fieldnames(typeof(r)))
        end
        # A measure that is not a moment measure returns unchanged.
        mad = MedianAbsoluteDeviation(; w = cw)
        @test PO.resolve_observation_weights(mad, X) === mad
    end

    @testset "resolved weights reach the nested variance estimator (#1319)" begin
        # `factory` against a prior writes the weights into `w` and into the variance
        # estimator `ve`. The functor used to resolve `w` alone. So without `factory` the
        # standard deviation went unweighted, and after `factory` with dynamic weights `ve`
        # met the vector arity and raised.
        w = fill(inv(5), 5)
        mw = MatrixOnlyObsWeights()
        rw = PO.get_observation_weights(mw, X)
        pr = prior(HighOrderPriorEstimator(), X)
        mu = pr.mu
        sha = PO.StandardisedHighOrderMoment
        for f in (ow -> LowOrderMoment(; w = ow, mu = mu, alg = SecondMoment()),
                  ow -> LowOrderMoment(; w = ow, mu = mu,
                                       alg = SecondMoment(; alg1 = PO.SemiMoment())),
                  ow -> HighOrderMoment(; w = ow, mu = mu, alg = sha()),
                  ow -> HighOrderMoment(; w = ow, mu = mu, alg = sha(; alg = PO.FourthMoment())),
                  ow -> PO.ThirdCentralMoment(; w = ow, mu = mu, ve = SimpleVariance()))
            dyn, sta = f(mw), f(rw)
            val = factory(sta, pr)(w, X)
            @test isapprox(sta(w, X), val; rtol = 1e-12)
            @test isapprox(dyn(w, X), val; rtol = 1e-12)
            @test isapprox(factory(dyn, pr)(w, X), val; rtol = 1e-12)
            # The series functor resolves against the series, which this type refuses.
            @test_throws PO.ObservationWeightsError dyn(X * w)
        end
    end
end
#=
The conic tail measures resolve their observation weights on both sides of the
returns/drawdown twin.

A tail measure and its drawdown twin are one programme under the substitution
`net_X -> -dd[2:T+1]` (`risk_series`). They used to disagree about what they handed
`get_observation_weights`: the returns tails passed `net_X`, a vector of JuMP expressions
that matches neither documented arity, so a `DynamicAbstractWeights` ALWAYS raised there;
the drawdown twins passed `pr.X` and resolved. One substitution, two answers. Both sides now
pass `pr.X`, which is the documented `MatNum` arity.
=#
@testset "Observation weights resolve on both sides of the returns/drawdown twin" begin
    using Test, PortfolioOptimisers, StableRNGs, StatsBase, Clarabel

    rng = StableRNG(987654321)
    X = randn(rng, 40, 5) ./ 100
    rd = ReturnsResult(; nx = string.('A':'E'), X = X)
    cw = CompleteObsWeights(8)
    sw = PO.get_observation_weights(cw, X; dims = 1)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = "verbose" => false)

    # Build the model without solving it: the resolved weights are visible in the
    # constraints, and a solve would only add tolerance to the comparison.
    function build(r)
        mr = MeanRisk(; r = r, opt = JuMPOptimiser(; slv = slv))
        attrs = PO.processed_jump_optimiser_attributes(mr.opt, rd; dims = 1)
        model = PO.JuMP.Model()
        PO.JuMP.set_string_names_on_creation(model, false)
        PO.set_model_scales!(model, mr.opt.sc, mr.opt.so)
        PO.set_model_observations!(model, size(attrs.pr.X, 1))
        PO.set_maximum_ratio_factor_variables!(model, mr.obj)
        PO.set_w!(model, attrs.pr.X, mr.wi)
        PO.set_weight_constraints!(model, attrs.wb, mr.opt)
        PO.assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
        return sprint(print, model)
    end

    # Resolving the weights is all the dynamic type buys: the model is the one the measure
    # builds when it is handed the same weights directly.
    for (dyn, sta) in ((ValueatRisk(; w = cw, alg = MIPValueatRisk()),
                        ValueatRisk(; w = sw, alg = MIPValueatRisk())),
                       (DrawdownatRisk(; w = cw), DrawdownatRisk(; w = sw)),
                       (ConditionalValueatRisk(; w = cw), ConditionalValueatRisk(; w = sw)),
                       (ConditionalDrawdownatRisk(; w = cw), ConditionalDrawdownatRisk(; w = sw)),
                       (DistributionallyRobustConditionalValueatRisk(; w = cw),
                        DistributionallyRobustConditionalValueatRisk(; w = sw)),
                       (DistributionallyRobustConditionalDrawdownatRisk(; w = cw),
                        DistributionallyRobustConditionalDrawdownatRisk(; w = sw)),
                       (EntropicValueatRisk(; w = cw), EntropicValueatRisk(; w = sw)),
                       (EntropicDrawdownatRisk(; w = cw), EntropicDrawdownatRisk(; w = sw)),
                       (RelativisticValueatRisk(; w = cw), RelativisticValueatRisk(; w = sw)),
                       (RelativisticDrawdownatRisk(; w = cw), RelativisticDrawdownatRisk(; w = sw)),
                       (PowerNormValueatRisk(; w = cw), PowerNormValueatRisk(; w = sw)),
                       (PowerNormDrawdownatRisk(; w = cw), PowerNormDrawdownatRisk(; w = sw)))
        @test build(dyn) == build(sta)
    end

    # A type that cannot resolve the matrix arity still raises rather than going quietly
    # unweighted, and it now raises on both sides rather than only one.
    vw = VectorOnlyObsWeights()
    @test_throws PO.ObservationWeightsError build(ConditionalValueatRisk(; w = vw))
    @test_throws PO.ObservationWeightsError build(ConditionalDrawdownatRisk(; w = vw))

    @testset "DimensionReductionRegression standardises and recovers with one scale" begin
        # `prep_dim_red_reg` used to standardise with `StatsBase.ZScoreTransform` -- the
        # plain mean and the corrected standard deviation -- while `regression` divided the
        # recovered coefficients by `re.ve` and centred the intercept with `re.ve.me`. The
        # two agreed for the default `ve` and for nothing else, so every weighted path
        # recovered a coefficient in no unit at all. See issue #398.
        #
        # The invariant is the one Equations 4.13, 4.15 and 4.20 of the source impose: the
        # prediction in the original factor space must equal the prediction in the reduced
        # space. It holds only when the divisor is the scale that standardised.
        rng2 = StableRNG(987654321)
        F = randn(rng2, 250, 6) ./ 100
        Y = F * randn(rng2, 6, 3) .+ randn(rng2, 250, 3) ./ 500
        pw = pweights(range(; start = 1, stop = 5, length = 250))

        function prediction_gap(re, Y, F)
            reg = regression(re, Y, F)
            f1, Vp, mu, sigma = PO.prep_dim_red_reg(re, F)
            # The two statistics are the ones `regression` recovers with.
            @test isapprox(mu, vec(PO.Statistics.mean(re.ve.me, F; dims = 1)))
            @test isapprox(sigma, vec(PO.Statistics.std(re.ve, F; dims = 1)))
            gap = zero(eltype(F))
            for i in axes(Y, 2)
                y = view(Y, :, i)
                coefs = PO.StatsAPI.coef(PO.StatsAPI.fit(re.retgt, f1, y))
                pred = reg.b[i] .+ F * view(reg.M, i, :)
                gap = max(gap, maximum(abs, f1 * coefs .- pred))
            end
            return gap
        end

        # `ve` is `@fprop`-tagged, so `factory` puts the incoming weights into it.
        re_w = factory(DimensionReductionRegression(), pw)
        @test !isnothing(re_w.ve.w)
        @test prediction_gap(DimensionReductionRegression(), Y, F) < 1e-12
        @test prediction_gap(re_w, Y, F) < 1e-12
        # `corrected = false` is the other way the two scales used to part.
        re_c = DimensionReductionRegression(; ve = SimpleVariance(; corrected = false))
        @test prediction_gap(re_c, Y, F) < 1e-12

        # A constant factor has zero standard deviation. The floor at `eps` keeps the
        # division finite rather than answering `NaN`.
        Fc = copy(F)
        Fc[:, 3] .= 0.01
        reg_c = regression(DimensionReductionRegression(), Y, Fc)
        @test all(isfinite, reg_c.M)
        @test all(isfinite, reg_c.b)
    end
end

#=
`ImpliedVolatility` had no behavioural test, and all three of its defects survived because
of that. `test_08d_dims_guard.jl` reaches the type, but its guard runs before the estimator
computes anything, so a `cov` that throws still passed there.

The three defects, each of which a testset below now pins:

  1. `realised_vol` reshaped its window into a `ws × chunk × N` array and asked
     `Statistics.std(::AbstractVarianceEstimator, ...)` for it. No method takes a 3-D array,
     so `cov(ImpliedVolatility(), X; iv = IV)` -- the type's own default algorithm -- raised
     a `MethodError` for every input.

  2. `cov` and `cor` called `predict_realised_vols(ce.alg, X, iv, ivpa)`, where the method
     declares `(alg, iv, X, ivpa)`. The two matrices were swapped, so the premium algorithm
     returned the last row of the *returns* as a volatility, sign and all.

  3. `cov` called `cov2cor!` on a matrix that was already a correlation, where the
     documented `Sigma = diag(sigma) rho diag(sigma)` needs `cor2cov!`. The returned
     "covariance" carried a unit diagonal whatever the implied volatilities were.

A FOURTH repair gave `realised_vol` a per-block estimator through `obs_weights_view`, so a
weighted variance estimator no longer meets a window with the whole sample's weights.

Sweeping the file (#465) found a fifth hole and closed it. `cov` and `cor` compared `iv`
with nothing, and `implied_vol` takes its row count from `X`, so an `iv` with MORE rows
than `X` sampled windows counted forward from the start of `iv` and answered a covariance
built from the wrong rows, silently. `size(X) == size(iv)` is now checked in both methods,
right after `dims_oriented`.

A SIXTH repair (#504) took the round trip out of `cor`. It scaled the base correlation to a
covariance with the predicted volatilities and scaled it straight back, which a correlation
cannot feel -- except in the last place, and except when a predicted volatility is zero,
where the rescaling divides zero by zero and answers `NaN`. `cor` normalises the base
correlation instead. It still runs the volatility model, and still discards it, because
`cor` must refuse every configuration `cov` refuses.

Three kinds of check run here, and the first two are what catch a drift no invariant sees:

  - REFERENCE VALUES, as the rest of the covariance family does it: nine configurations
    against `assets/ImpliedVolatility.csv.gz`, one flattened covariance per column.

  - AN INDEPENDENT FIT. The log-linear regression is re-derived here from the normal
    equations through `\`, not through `GLM`, and the two must agree. This is the assertion
    the oracle makes of itself -- rebuild the prediction from the
    coefficients, the last implied volatility and the last window's realised volatility --
    so one solver's bug cannot hide behind the other's.

  - INVARIANTS: the documented `Sigma = diag(sigma) rho diag(sigma)`, a correlation that
    does not move when the volatilities do, and the window alignment of `realised_vol`
    against `implied_vol`.

THE IMPLIED VOLATILITY MATRIX IS SYNTHETIC AND DETERMINISTIC. The library ships no option
data, so `make_implied_vol` below derives one from the returns themselves: a 20-day trailing
realised volatility, annualised, lifted by a fixed 1.15 volatility risk premium and
perturbed by a seeded log-normal shock. The result runs from about 9 % to about 120 %
annualised, and the predicted realised volatilities it produces sit between 0.0093 and
0.049 daily, which is the range the oracle's own fixture reaches. THE
RECIPE MUST NOT CHANGE without regenerating `assets/ImpliedVolatility.csv.gz`.
=#
using PortfolioOptimisers, Statistics, StatsBase

# Two custom variance estimators, to pin both sides of the `obs_weights_view` fallback. They
# live at top level because a `@testset` body becomes a function and cannot host a `struct`.
# `OptedOut` implements no method, so its full-sample weights meet a window and raise, which
# is what every weighted estimator did before the verb existed. `OptedIn` implements one.
struct OptedOutVariance{T} <: PortfolioOptimisers.AbstractVarianceEstimator
    w::T
end
struct OptedInVariance{T} <: PortfolioOptimisers.AbstractVarianceEstimator
    w::T
end
function Statistics.std(ve::Union{OptedOutVariance, OptedInVariance},
                        X::PortfolioOptimisers.MatNum; dims::Int = 1, kwargs...)
    return Statistics.std(X, ve.w, dims; corrected = false)
end
function PortfolioOptimisers.obs_weights_view(ve::OptedInVariance, i)
    return OptedInVariance(PortfolioOptimisers.nothing_scalar_array_getindex(ve.w, i))
end

@testset "Implied volatility" begin
    using Test, PortfolioOptimisers, DataFrames, TimeSeries, CSV, StableRNGs, StatsBase,
          Statistics, LinearAlgebra

    rd = prices_to_returns(price_ingestion(PriceIngestion(),
                                           TimeArray(CSV.File(joinpath(@__DIR__,
                                                                       "./assets/SP500.csv.gz"));
                                                     timestamp = :Date)[(end - 252):end];
                                           F = TimeArray(CSV.File(joinpath(@__DIR__,
                                                                           "./assets/Factors.csv.gz"));
                                                         timestamp = :Date)[(end - 252):end]))
    X = rd.X
    T, N = size(X)

    # A 20-day trailing realised volatility, annualised, times a 1.15 volatility risk
    # premium, times a seeded log-normal shock. The leading rows that have no full window
    # take the first full window's value, so no row is built from a single observation --
    # the `log` of such a value is an outlier that would dominate the regression.
    function make_implied_vol(X; w = 20, prem = 1.15, noise = 0.10, seed = 987654321)
        rng = StableRNG(seed)
        T, N = size(X)
        roll = similar(X)
        for t in w:T
            roll[t, :] = vec(Statistics.std(view(X, (t - w + 1):t, :); dims = 1))
        end
        for t in 1:(w - 1)
            roll[t, :] = roll[w, :]
        end
        return (roll * sqrt(252) * prem) .* exp.(noise * randn(rng, T, N))
    end

    iv = make_implied_vol(X)
    ew = eweights(1:T, inv(T); scale = true)
    # `Weights` refuses bias correction, and `SimpleVariance` corrects by default, so a
    # weighted case needs `AnalyticWeights` unless it also passes `corrected = false`. That
    # rule is `StatsBase`'s and it holds over the whole sample too, not only over a window.
    aw = AnalyticWeights(collect(ew))
    ivpav = collect(range(1.05, 1.35; length = N))

    @testset "Reference values" begin
        # One entry per column of `assets/ImpliedVolatility.csv.gz`, in order.
        ces = [(ImpliedVolatility(; ppy = 252), (;)),
               (ImpliedVolatility(; ppy = 252,
                                  alg = ImpliedVolatilityRegression(; ws = 30)), (;)),
               (ImpliedVolatility(; alg = ImpliedVolatilityRegression(; ws = 25), ppy = 63),
                (;)),
               (ImpliedVolatility(; ppy = 252,
                                  alg = ImpliedVolatilityRegression(;
                                                                    ve = SimpleVariance(;
                                                                                        corrected = false))),
                (;)), (ImpliedVolatility(; ppy = 252, ce = SpearmanCovariance()), (;)),
               (ImpliedVolatility(; ppy = 252,
                                  mp = MatrixProcessing(;
                                                        dn = Denoise(;
                                                                     alg = SpectralDenoise()))),
                (;)),
               (ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium()),
                (; ivpa = 1.2)),
               (ImpliedVolatility(; alg = ImpliedVolatilityPremium(), ppy = 63),
                (; ivpa = ivpav)),
               (ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium(),
                                  ce = KendallCovariance()), (; ivpa = ivpav))]
        df = CSV.read(joinpath(@__DIR__, "./assets/ImpliedVolatility.csv.gz"), DataFrame)
        @test length(ces) == size(df, 2)
        for (i, (ce, kwargs)) in pairs(ces)
            sigma = cov(ce, X; iv = iv, kwargs...)
            rho = cor(ce, X; iv = iv, kwargs...)
            # A covariance whose base correlation is recoverable from it.
            @test isapprox(StatsBase.cov2cor(copy(sigma)), rho)
            # Every configuration returns a usable covariance, never a near-singular one.
            @test isposdef(sigma)
            success = isapprox(vec(sigma), df[!, i])
            if !success
                println("Counter: $i")
                find_tol(vec(sigma), df[!, i])
            end
            @test success
        end
    end

    @testset "The regression reproduces an independent least-squares fit" begin
        # The estimator fits through `GLM`. This rebuilds the same model from the normal
        # equations and predicts from the coefficients by hand.
        function hand_predict(ve, X, iv, ws, ppy)
            T, N = size(X)
            chunk = div(T, ws)
            offset = T - chunk * ws
            rv = reduce(vcat,
                        [Statistics.std(ve,
                                        view(X,
                                             (offset + (c - 1) * ws + 1):(offset + c * ws),
                                             :); dims = 1) for c in 1:chunk])
            ivw = (iv / sqrt(ppy))[(T - (chunk - 1) * ws):ws:T, :]
            lrv, liv = log.(rv), log.(ivw)
            out = Vector{Float64}(undef, N)
            for i in 1:N
                D = [ones(chunk - 1) liv[1:(chunk - 1), i] lrv[1:(chunk - 1), i]]
                b = D \ lrv[2:chunk, i]
                out[i] = exp(b[1] + b[2] * liv[chunk, i] + b[3] * lrv[chunk, i])
            end
            return out
        end

        for (ve, ws, ppy) in
            [(SimpleVariance(), 20, 252), (SimpleVariance(; corrected = false), 30, 252),
             (SimpleVariance(), 25, 63)]
            alg = ImpliedVolatilityRegression(; ve = ve, ws = ws)
            got = PortfolioOptimisers.predict_realised_vols(alg, iv / sqrt(ppy), X, nothing)
            @test isapprox(got, hand_predict(ve, X, iv, ws, ppy))
            # The prediction is the diagonal of the covariance the estimator returns.
            sigma = cov(ImpliedVolatility(; alg = alg, ppy = ppy), X; iv = iv)
            @test isapprox(sqrt.(diag(sigma)), got)
        end
    end

    @testset "ImpliedVolatility reproduces its documented covariance" begin
        rho = cor(Covariance(), X)
        for (alg, kwargs) in [(ImpliedVolatilityRegression(), (;)),
                              (ImpliedVolatilityPremium(), (; ivpa = 1.2))]
            ce = ImpliedVolatility(; ppy = 252, alg = alg)
            sigma = cov(ce, X; iv = iv, kwargs...)
            sd = PortfolioOptimisers.predict_realised_vols(alg, iv / sqrt(ce.ppy), X,
                                                           get(kwargs, :ivpa, nothing))
            # The diagonal is the predicted realised volatility, never a unit.
            @test isapprox(sqrt.(diag(sigma)), sd)
            @test all(>(0), sd)
            # Sigma = diag(sigma) rho diag(sigma).
            @test isapprox(sigma, Diagonal(sd) * rho * Diagonal(sd))
            # A correlation does not depend on the volatilities that scale it.
            @test cor(ce, X; iv = iv, kwargs...) == rho
        end
    end

    @testset "ImpliedVolatilityPremium divides the latest implied volatility" begin
        ce = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        for ivpa in (1.2, ivpav)
            sd = sqrt.(diag(cov(ce, X; iv = iv, ivpa = ivpa)))
            @test isapprox(sd, vec(iv[end, :]) ./ sqrt(ce.ppy) ./ ivpa)
        end
        # `ppy` rescales the whole diagonal by sqrt(ppy).
        ce2 = ImpliedVolatility(; alg = ImpliedVolatilityPremium(), ppy = 63)
        @test isapprox(sqrt.(diag(cov(ce2, X; iv = iv, ivpa = 1.2))) * sqrt(63),
                       sqrt.(diag(cov(ce, X; iv = iv, ivpa = 1.2))) * sqrt(252))
    end

    @testset "realised_vol and implied_vol read the same windows" begin
        for ws in (20, 30, 25)
            chunk = div(T, ws)
            offset = T - chunk * ws
            rv = PortfolioOptimisers.realised_vol(SimpleVariance(), X, ws)
            ivw = PortfolioOptimisers.implied_vol(iv, ws)
            @test size(rv) == (chunk, N) == size(ivw)
            for c in 1:chunk
                rows = (offset + (c - 1) * ws + 1):(offset + c * ws)
                @test isapprox(rv[c, :], vec(std(view(X, rows, :); dims = 1)))
                # Window `c` ends at the row `implied_vol` samples for the same window.
                @test ivw[c, :] == iv[last(rows), :]
            end
            # A window count that does not divide `T` drops the OLDEST rows, never the
            # newest: the last window always ends at the last observation.
            @test ivw[chunk, :] == iv[T, :]
        end
    end

    @testset "ImpliedVolatility rejects what it cannot compute" begin
        # A window of one row has no dispersion: its corrected standard deviation is `NaN`
        # and its uncorrected one is zero, whose logarithm is `-Inf`. A window of two rows
        # has a corrected standard deviation with one degree of freedom (#1796).
        @test_throws DomainError ImpliedVolatilityRegression(; ws = 1)
        @test ImpliedVolatilityRegression(; ws = 2).ws == 2
        #=
        The fit has one row fewer than the windows, for three coefficients (#1796). Three
        windows give two rows, so the coefficients have no unique value: the oracle accepts
        that floor, and the second asset of the issue's reproduction got a per-period
        volatility of 15. Four windows give three rows and no residual. So `T = 252` refuses
        `ws = 63` (four windows) and `ws = 84` (three), and accepts `ws = 50` (five).
        =#
        for ws in (126, 84, 63)
            @test_throws DomainError cov(ImpliedVolatility(; ppy = 252,
                                                           alg = ImpliedVolatilityRegression(;
                                                                                             ws = ws)),
                                         X; iv = iv)
        end
        @test all(isfinite,
                  diag(cov(ImpliedVolatility(; ppy = 252,
                                             alg = ImpliedVolatilityRegression(; ws = 50)),
                           X; iv = iv)))
        # The premium algorithm has no default factor.
        @test_throws ArgumentError cov(ImpliedVolatility(; ppy = 252,
                                                         alg = ImpliedVolatilityPremium()),
                                       X; iv = iv)
        #=
        A factor that is not finite and strictly positive divides the last implied
        volatility into something that is not a volatility, and `cor2cov!` hides it. A
        NEGATIVE SCALAR cancels: `cor2cov!` writes `C[i,j] *= s[i] * s[j]` and
        `C[j,j] = s[j]^2`, so `-1.2` returned exactly the matrix `1.2` returns. A NEGATIVE
        ENTRY of a vector flips the sign of every covariance between that asset and the
        others, keeps its variance, and leaves the matrix positive definite, so
        `matrix_processing!` found nothing to repair and the optimiser was handed an asset
        whose correlations all pointed the wrong way. A `NaN` was refused, but only by
        accident, and by `NearestCorrelationMatrix`, which named neither `ivpa` nor the
        estimator. The oracle refuses all three at the fit.
        =#
        cep = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        ivpam = copy(ivpav)
        ivpam[1] = -ivpam[1]
        for bad in (-1.2, zero(eltype(ivpav)), NaN, Inf, ivpam)
            @test_throws DomainError cov(cep, X; iv = iv, ivpa = bad)
            # `cor` refuses what `cov` refuses, though its answer cannot move.
            @test_throws DomainError cor(cep, X; iv = iv, ivpa = bad)
        end
        # The positive factor the bad ones were derived from still answers.
        @test all(isfinite, cov(cep, X; iv = iv, ivpa = ivpav))
        #=
        An implied volatility that is negative is not a volatility either. On the premium
        path a negative last entry divided into a negative volatility, and `cor2cov!` hid its
        sign exactly as it hides a negative factor, so the fit answered. A gap is refused by
        the same check, with a `DomainError` that names `iv`.
        =#
        ivneg = copy(iv)
        ivneg[end, 2] = -ivneg[end, 2]
        ivgap = copy(iv)
        ivgap[3, 2] = NaN
        cer = ImpliedVolatility(; ppy = 252)
        for bad in (ivneg, ivgap), f in (cov, cor)
            @test_throws DomainError f(cep, X; iv = bad, ivpa = ivpav)
            @test_throws DomainError f(cer, X; iv = bad)
            @test_throws "iv" f(cep, X; iv = bad, ivpa = ivpav)
        end
        # A zero implied volatility is a volatility, and the premium path still answers.
        ivzero = copy(iv)
        ivzero[end, 2] = 0
        @test all(isfinite, cov(cep, X; iv = ivzero, ivpa = ivpav))
        #=
        The regression takes the logarithm of the implied volatility at the end of each
        window and of the realised volatility of each window, so a zero there has no value in
        the fit. A zero at the end of an earlier window raised a LAPACK error that named
        neither input, a zero on the last row made a predicted volatility of zero or infinity,
        and a window of constant returns raised the same LAPACK error (R5 of #1817). A zero
        on a row that ends no window is not read, and the fit still answers.
        =#
        sampled = (T - (div(T, 20) - 1) * 20):20:T
        for row in (sampled[5], T)
            ivz = copy(iv)
            ivz[row, 2] = 0
            for f in (cov, cor)
                @test_throws DomainError f(cer, X; iv = ivz)
                @test_throws "`iv`" f(cer, X; iv = ivz)
            end
        end
        ivz = copy(iv)
        ivz[sampled[5] - 1, 2] = 0
        @test cov(cer, X; iv = ivz) == cov(cer, X; iv = iv)
        #=
        The round-off of the mean can leave a constant window a standard deviation of the
        order of `eps`, and whether it does depends on the value and on the host. On the CI
        host, the first value gave about `1e-19`, and the fit answered. The last three give a
        positive value on the host that wrote the test, so the comparison of the returns
        refuses them there too.
        =#
        for v in (X[sampled[5], 2], 0.015453143833033613, -0.009848612891382565,
                  -0.0048880224182851865)
            Xc = copy(X)
            Xc[(sampled[5] - 19):sampled[5], 2] .= v
            for f in (cov, cor)
                @test_throws DomainError f(cer, Xc; iv = iv)
                @test_throws "no dispersion" f(cer, Xc; iv = iv)
            end
        end
        # The implied volatility series itself is mandatory. Every method refuses a missing
        # `iv` and an `iv = nothing` by name, and so does a prior fitted on returns that
        # carry none, since `prior(pe, rd)` passes `iv = rd.iv`.
        cei = ImpliedVolatility(; ppy = 252)
        for f in (cov, cor)
            @test_throws IsNothingError f(cei, X)
            @test_throws IsNothingError f(cei, X; iv = nothing)
            @test_throws IsNothingError f(cei, X, nothing)
            @test_throws IsNothingError f(cei, X, nothing; iv = nothing)
        end
        @test_throws "Pass `iv` to `cov` or `cor`" cov(cei, X)
        @test_throws IsNothingError prior(EmpiricalPrior(;
                                                         ce = PortfolioOptimisersCovariance(;
                                                                                            ce = cei)),
                                          ReturnsResult(; nx = string.(1:size(X, 2)),
                                                        X = X))
        # The periods per year are mandatory too. The implied volatility is annual, and
        # `iv / sqrt(ppy)` is the identity only on annual data, so no default fits every
        # frequency (#1517).
        @test_throws UndefKeywordError ImpliedVolatility()
        @test_throws UndefKeywordError ImpliedVolatility(; alg = ImpliedVolatilityPremium())
        for bad in (0, -252)
            @test_throws DomainError ImpliedVolatility(; ppy = bad)
        end
        # A custom variance estimator that carries observation weights and implements no
        # `obs_weights_view` method keeps its full-sample weights against a window. This is
        # the documented fallback, and it is what every weighted estimator did before the
        # verb existed.
        @test_throws DimensionMismatch PortfolioOptimisers.realised_vol(OptedOutVariance(aw),
                                                                        X, 20)
    end

    #=
    `realised_vol` used to hand each block the estimator untouched. An observation weight is
    one value per row of the WHOLE sample, so a weighted estimator met a block of `ws` rows
    with a vector of `T` weights and raised -- `ArgumentError` when only the dispersion was
    weighted, `DimensionMismatch` when the mean was too. The field is typed
    `AbstractVarianceEstimator` and nothing rejected such a value, so an estimator that can
    never answer was constructible and only failed at the call.

    `obs_weights_view` slices every weights field to the block. It is deliberately NOT
    `factory`: `factory` writes one incoming value into every `@wprop`-tagged field, which
    would give a weighted dispersion to an estimator that only asked for a weighted mean.
    =#
    @testset "obs_weights_view slices each weights field on its own" begin
        rows = 21:40

        # A weighted dispersion stays a weighted dispersion, and the mean stays unweighted.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(; w = aw), rows)
        @test length(v.w) == length(rows)
        @test v.w == aw[rows]
        @test isnothing(v.me.w)

        # A weighted mean stays a weighted mean, and the dispersion stays unweighted. This
        # is the case `factory` cannot express, so the next two lines are the reason the
        # verb exists.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(;
                                                                me = SimpleExpectedReturns(;
                                                                                           w = aw)),
                                                 rows)
        @test isnothing(v.w)
        @test length(v.me.w) == length(rows)
        @test !isnothing(PortfolioOptimisers.factory(SimpleVariance(;
                                                                    me = SimpleExpectedReturns(;
                                                                                               w = aw)),
                                                     aw).w)

        # Both weighted, and `corrected` carried through.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(;
                                                                me = SimpleExpectedReturns(;
                                                                                           w = aw),
                                                                corrected = false, w = aw),
                                                 rows)
        @test length(v.w) == length(v.me.w) == length(rows)
        @test v.corrected == false

        # Neither weighted: nothing to slice, and nothing gained.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(), rows)
        @test isnothing(v.w)
        @test isnothing(v.me.w)

        # A `nothing` mean estimator survives the recursion.
        v = PortfolioOptimisers.obs_weights_view(SimpleVariance(; me = nothing, w = aw),
                                                 rows)
        @test isnothing(v.me)
        @test length(v.w) == length(rows)

        # The wrapper recurses, and `window` is not an observation index, so it is carried
        # through untouched.
        v = PortfolioOptimisers.obs_weights_view(WindowedVariance(;
                                                                  ve = SimpleVariance(;
                                                                                      w = aw),
                                                                  w = aw, window = 15),
                                                 rows)
        @test length(v.w) == length(v.ve.w) == length(rows)
        @test v.window == 15

        # The fallback returns the estimator unchanged, so an estimator that carries no
        # weights and one that implements nothing both behave as they did before.
        c = Covariance()
        @test PortfolioOptimisers.obs_weights_view(c, rows) === c
        o = OptedOutVariance(aw)
        @test PortfolioOptimisers.obs_weights_view(o, rows) === o
    end

    @testset "realised_vol weighs each block by that block's weights" begin
        ws = 20
        chunk = div(T, ws)
        offset = T - chunk * ws

        # Every weighted shape now answers, and each block matches a hand-built weighted
        # standard deviation over that block's own rows and that block's own weights.
        for ve in (SimpleVariance(; w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = ew), corrected = false, w = ew), OptedInVariance(aw))
            rv = PortfolioOptimisers.realised_vol(ve, X, ws)
            @test size(rv) == (chunk, N)
            for c in 1:chunk
                rows = (offset + (c - 1) * ws + 1):(offset + c * ws)
                sliced = PortfolioOptimisers.obs_weights_view(ve, rows)
                @test isapprox(rv[c, :], vec(std(sliced, view(X, rows, :); dims = 1)))
                # The block's weights are that block's slice, never the whole sample's.
                @test length(PortfolioOptimisers.obs_weights_view(ve, rows).w) == ws
            end
        end

        # The unweighted path does not move: it is the same estimator, returned unchanged.
        @test isapprox(PortfolioOptimisers.realised_vol(SimpleVariance(), X, ws),
                       PortfolioOptimisers.realised_vol(PortfolioOptimisers.obs_weights_view(SimpleVariance(),
                                                                                             1:T),
                                                        X, ws))
    end

    @testset "ImpliedVolatility accepts a weighted variance estimator" begin
        base = cov(ImpliedVolatility(; ppy = 252), X; iv = iv)
        for ve in (SimpleVariance(; w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = aw), w = aw),
                   SimpleVariance(; me = SimpleExpectedReturns(; w = ew), corrected = false, w = ew), WindowedVariance(; w = aw, window = 15))
            ce = ImpliedVolatility(; ppy = 252,
                                   alg = ImpliedVolatilityRegression(; ve = ve))
            sigma = cov(ce, X; iv = iv)
            @test all(isfinite, sigma)
            @test isposdef(sigma)
            @test all(>(0), sqrt.(diag(sigma)))
            # The weights change the answer, so they are not being quietly dropped.
            @test !isapprox(sigma, base)
            # The correlation is untouched by the volatility model either way.
            @test cor(ce, X; iv = iv) == cor(Covariance(), X)
        end
    end

    #=
    ------------------------------------------------------------------ sweep #465

    The four testsets below were added when this file was swept. Each pins a claim that
    a docstring in `src/05_Moments/22_ImpliedVolatility.jl` now makes, so a reader who
    doubts the prose can run the number instead.
    =#

    @testset "The documented identities hold to the tolerance they are stated at" begin
        rho = cor(Covariance(), X)
        for (alg, kwargs) in [(ImpliedVolatilityRegression(), (;)),
                              (ImpliedVolatilityPremium(), (; ivpa = 1.2))]
            ce = ImpliedVolatility(; ppy = 252, alg = alg)
            sigma = cov(ce, X; iv = iv, kwargs...)
            sd = PortfolioOptimisers.predict_realised_vols(alg, iv / sqrt(ce.ppy), X,
                                                           get(kwargs, :ivpa, nothing))
            # `cov`'s `# Mathematical definition`. Measured at 1.1e-19 for the regression
            # route and 5.4e-20 for the premium route, against entries of order 1e-4.
            @test maximum(abs, sigma - Diagonal(sd) * rho * Diagonal(sd)) <= 1e-18
            #=
            `cor`'s step 5. The base correlation is normalised and nothing else, so the
            answer is BIT-EXACT. It was not while `cor` scaled `rho` by the predicted
            volatilities and rescaled it back: that round trip moved 38 of the 400 entries
            by half an eps under the regression route, and by a quarter of one under the
            premium route (#504).
            =#
            rhoc = cor(ce, X; iv = iv, kwargs...)
            @test diag(rhoc) == ones(N)
            @test rhoc == rho
        end
    end

    @testset "cov and cor refuse an iv that is not shaped like X" begin
        ce = ImpliedVolatility(; ppy = 252)
        #=
        A LONGER `iv` used to pass silently and answer a covariance built from the wrong
        rows. `implied_vol` is handed the row count of `X`, so with more rows in `iv` it
        sampled windows offset from the START of `iv` rather than counted back from its
        end, and `size(rv) == size(iv)` still held. A shorter `iv` raised a raw
        `BoundsError` from inside `implied_vol`. Neither reaches the estimator now.
        =#
        iv_long = vcat(iv, iv[1:48, :])
        @test size(iv_long, 1) != T
        @test_throws DimensionMismatch cov(ce, X; iv = iv_long)
        @test_throws DimensionMismatch cor(ce, X; iv = iv_long)
        @test_throws DimensionMismatch cov(ce, X; iv = iv[1:200, :])
        @test_throws DimensionMismatch cor(ce, X; iv = iv[1:200, :])
        # A narrower `iv` reached the argcheck inside the regression branch. It is refused
        # before the base correlation is computed now.
        @test_throws DimensionMismatch cov(ce, X; iv = iv[:, 1:10])
        @test_throws DimensionMismatch cor(ce, X; iv = iv[:, 1:10])
        # The premium branch reads no window at all, so nothing there ever raised.
        cep = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        @test_throws DimensionMismatch cov(cep, X; iv = iv_long, ivpa = 1.2)
        @test_throws DimensionMismatch cor(cep, X; iv = iv_long, ivpa = 1.2)
        # The guard runs AFTER `dims_oriented`, so a bad `dims` still answers `DomainError`.
        @test_throws DomainError cov(ce, X; dims = 3, iv = iv_long)
        # Both matrices are oriented before they are compared, so a transposed pair passes
        # and gives the same answer.
        @test isapprox(cov(ce, transpose(X); dims = 2, iv = transpose(iv)),
                       cov(ce, X; iv = iv))
    end

    @testset "factory reaches the algorithm's variance estimator" begin
        #=
        Issue #505. `alg` carried no propagation tag and `ImpliedVolatilityRegression` was
        not `@propagatable`, so `factory` weighted the base correlation estimator and left
        `alg.ve` unweighted. One estimator answered a weighted correlation and an
        unweighted realised volatility, silently.

        Both tags are in place now, so one `factory` call reaches every estimator the type
        holds.
        =#
        f = PortfolioOptimisers.factory(ImpliedVolatility(; ppy = 252), aw)
        @test !isnothing(f.ce.ce.w)
        @test f.alg.ve.w === aw
        @test f.alg.ve.me.w === aw
        # The tag names one field, so everything else the algorithm holds is rebuilt as it
        # stood.
        @test f.alg.ws == ImpliedVolatility(; ppy = 252).alg.ws
        @test f.alg.re isa LinearModel
        # The algorithm answers the same weighted estimator on its own.
        @test PortfolioOptimisers.factory(ImpliedVolatilityRegression(), aw).ve.w === aw
        # `factory` writes the incoming value over a hand-set one, as it does on every
        # `@wprop` field.
        uw = AnalyticWeights(fill(inv(length(aw)), length(aw)))
        g = PortfolioOptimisers.factory(ImpliedVolatility(; ppy = 252,
                                                          alg = ImpliedVolatilityRegression(;
                                                                                            ve = SimpleVariance(;
                                                                                                                w = uw))),
                                        aw)
        @test g.alg.ve.w === aw
        # `ImpliedVolatilityPremium` holds no estimator, so it comes back unchanged.
        p = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        @test PortfolioOptimisers.factory(p, aw).alg === p.alg
        # A `factory` call that carries no weights leaves every estimator alone.
        @test PortfolioOptimisers.factory(ImpliedVolatility(; ppy = 252), nothing).alg.ve.w ===
              nothing
    end

    @testset "the base estimator receives the oriented iv" begin
        #=
        `cov` and `cor` forward `iv` to `ce.ce`. Every shipped estimator absorbs it into
        its own `kwargs...` and ignores it; a nested `ImpliedVolatility` reads it, which is
        the reason the keyword is forwarded rather than dropped.
        =#
        nested = ImpliedVolatility(; ppy = 252,
                                   ce = ImpliedVolatility(; ppy = 252,
                                                          alg = ImpliedVolatilityRegression(;
                                                                                            ws = 30)))
        sigma = cov(nested, X; iv = iv)
        @test size(sigma) == (N, N)
        @test isposdef(sigma)
        # The inner estimator answers a correlation, so only the outer scaling reaches the
        # diagonal.
        @test isapprox(sqrt.(diag(sigma)),
                       PortfolioOptimisers.predict_realised_vols(nested.alg,
                                                                 iv / sqrt(nested.ppy), X,
                                                                 nothing))
        # `implied_vol` answers a view, as its `# Algorithm` step 2 says.
        v = PortfolioOptimisers.implied_vol(iv, 20)
        @test v isa SubArray
        @test parent(v) === iv
    end

    #=
    -------------------------------------------------------------------- fix #504
    =#

    @testset "cor discards the predicted volatilities, and does not cancel them" begin
        #=
        `cor` runs the volatility model and throws the answer away. A correlation is scale
        free, so the prediction cannot move the answer, and `cor` used to spend it on a
        `cor2cov!`/`cov2cor!` round trip that bought nothing.

        The round trip was not free. It moved 38 of the 400 entries by half an eps, and it
        was not safe: a predicted volatility of ZERO makes `cov2cor!` divide zero by zero,
        so one asset whose last implied volatility is zero turned a whole row and column
        into `NaN` and `matrix_processing!` raised on it. `cor` now normalises the base
        correlation and nothing else, so no volatility reaches the answer at all.
        =#
        base = cor(Covariance(), X)
        cep = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        # The answer does not move with the volatilities, at ANY scale, and is bit-exact.
        for ivpa in (1.2, 1.7, ivpav, 1e-6, 1e6)
            @test cor(cep, X; iv = iv, ivpa = ivpa) == base
        end
        for ws in (20, 25, 30)
            @test cor(ImpliedVolatility(; ppy = 252,
                                        alg = ImpliedVolatilityRegression(; ws = ws)), X;
                      iv = iv) == base
        end
        #=
        A zero implied volatility. The covariance that `cov` builds is singular, and a zero
        variance has a zero row and column in a PSD matrix, so the posdef step keeps that row
        and repairs the others (#1429). `cor` never builds that covariance.
        =#
        iv0 = copy(iv)
        iv0[end, 3] = 0
        s0 = cov(cep, X; iv = iv0, ivpa = 1.2)
        @test iszero(s0[3, :]) && iszero(s0[:, 3]) && all(isfinite, s0)
        @test cor(cep, X; iv = iv0, ivpa = 1.2) == base
        #=
        The model still runs, which is why the discarded call is kept rather than deleted.
        `cor` refuses every configuration `cov` refuses, and raises the same type.
        =#
        @test_throws ArgumentError cor(cep, X; iv = iv)
        @test_throws ArgumentError cov(cep, X; iv = iv)
        few = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityRegression(; ws = 100))
        @test_throws DomainError cor(few, X; iv = iv)
        @test_throws DomainError cov(few, X; iv = iv)
    end

    @testset "the Coverage Universe of the fit reads the implied volatilities too" begin
        #=
        Issue #995, under ADR 0135. The layer carries an absent implied volatility as a
        padded `NaN`, exactly as it carries an absent return, and `PricesResult`'s guard on
        `iv` relaxed to non-negative WHERE A VALUE IS PRESENT. So the carrier is no longer
        the thing that keeps a non-finite implied volatility out of the arithmetic, and the
        estimator that reads one becomes it.

        What a `NaN` did if it got through, before this narrowing existed. Every one of
        these is a wrong number rather than an error, which is why the assertions below are
        on the EXPANDED moment and not only on the mask:

          - `cor2cov!(sigma, iv)` writes a `NaN` row AND column into the covariance, and
            `matrix_processing!` then sees a matrix it cannot repair.
          - `predict_realised_vols(::ImpliedVolatilityPremium, ...)` reads ONLY the last
            row, so a `NaN` there poisons that asset's volatility and a `NaN` anywhere else
            is ignored entirely.
          - `predict_realised_vols(::ImpliedVolatilityRegression, ...)` takes `log.(iv)` on
            a strided subsample; `fit` returns `NaN` coefficients rather than throwing.

        The two algorithms therefore read DIFFERENT parts of `iv`, and the loop below plants
        the absence at four rows -- the first, two the regression's stride lands on either
        side of, and the last -- so that a mask derived from ANY absent cell excludes the
        column under both.
        =#
        cep = ImpliedVolatility(; ppy = 252, alg = ImpliedVolatilityPremium())
        creg = ImpliedVolatility(; ppy = 252)
        keep = [i != 3 for i in 1:N]

        # A complete surface narrows nothing, and takes the `nothing` sentinel.
        @test isnothing(PortfolioOptimisers.coverage_mask(X, iv, nothing))

        for row in (1, 7, 8, T), bad in (NaN, Inf, -Inf)
            ivg = copy(iv)
            ivg[row, 3] = bad
            @test PortfolioOptimisers.coverage_mask(X, ivg, nothing) == BitVector(keep)
            for (ce, kw) in ((cep, (; ivpa = 1.2)), (cep, (; ivpa = ivpav)), (creg, (;)))
                for f in (cov, cor)
                    m = f(ce, X, nothing; iv = ivg, kw...)
                    # The absent column takes a `NaN` row and column, as an absent return
                    # does, and every other entry is finite.
                    @test all(isnan, view(m, 3, :))
                    @test all(isnan, view(m, :, 3))
                    @test all(isfinite, view(m, keep, keep))
                    # The surviving block is the fit run on those columns alone.
                    kwc = if haskey(kw, :ivpa) && isa(kw.ivpa, AbstractVector)
                        (; ivpa = kw.ivpa[keep])
                    else
                        kw
                    end
                    @test view(m, keep, keep) ==
                          f(ce, X[:, keep]; iv = ivg[:, keep], kwc...)
                end
            end
        end

        #=
        The narrowing INTERSECTS the mask it is handed rather than replacing it. A gapped
        return, an inactive row of the Asset Panel and an absent implied volatility each
        remove their own column, and the estimator fits on what all three leave.
        =#
        Xg = copy(X)
        Xg[4, 1] = NaN
        ivg = copy(iv)
        ivg[9, 2] = NaN
        amsk = trues(T, N)
        amsk[11, 3] = false
        pnl = AssetPanel(; pf = [NumericPanelField(; name = "mcap", vals = ones(T, N))],
                         amsk = amsk, emsk = amsk)
        keep3 = [i > 3 for i in 1:N]
        @test PortfolioOptimisers.coverage_mask(Xg, ivg, pnl) == BitVector(keep3)
        # The returns alone would have kept assets 2 and 3.
        @test PortfolioOptimisers.coverage_mask(Xg, pnl) ==
              BitVector([i != 1 && i != 3 for i in 1:N])

        sigma = cov(cep, Xg, pnl; iv = ivg, ivpa = 1.2)
        @test all(isnan, view(sigma, 1:3, :))
        @test all(isnan, view(sigma, :, 1:3))
        @test view(sigma, keep3, keep3) ==
              cov(cep, Xg[:, keep3]; iv = ivg[:, keep3], ivpa = 1.2)

        #=
        `dims = 2` takes the same path through `dims_oriented`, so it must answer the same
        mask and the same moment. A moment is assets x assets whichever way the sample was
        handed in, so the two answers are compared entry for entry.
        =#
        @test PortfolioOptimisers.coverage_mask(transpose(Xg), transpose(ivg), pnl;
                                                dims = 2) == BitVector(keep3)
        for f in (cov, cor)
            @test isequal(f(cep, transpose(Xg), pnl; dims = 2, iv = transpose(ivg),
                            ivpa = 1.2), f(cep, Xg, pnl; iv = ivg, ivpa = 1.2))
        end

        #=
        An empty Coverage Universe is refused by name. The message must reach the implied
        volatilities: the tail `coverage_sentinel` writes by default names the returns and
        the Asset Panel alone, and would send the caller looking in the wrong place.
        =#
        err = try
            PortfolioOptimisers.coverage_mask(X, fill(NaN, T, N), nothing)
        catch e
            e
        end
        @test isa(err, IsEmptyError)
        @test occursin("implied volatility", err.msg)
        @test_throws IsEmptyError cov(cep, X, nothing; iv = fill(NaN, T, N), ivpa = 1.2)
    end
end

#=
Check and cover `src/05_Moments/01_Base_Moments.jl` and the five `Windowed*` files, issue
#452 of child map 3 (#417) under the map of maps (#404).

The file's own tests already drive the `@windowed_estimator` machinery -- the parsers, the
suggestion scores, the generated docstrings and every expansion-time rejection all live in
`test_08_moments.jl` -- and `test_08b_robust_moments.jl` drives
`moment_window_and_weights`. What no test reached was the FALLBACK half of the four
`compat_*`/`robust_*` kernels: the retry on a `MethodError`, the rethrow of anything else,
and the covariance-to-correlation tail that answers an estimator with no usable `cor`.

Each probe below states the condition that selects one fallback and then meets it. A matrix
that never triggers the fallback proves nothing, so every probe is asserted against the
number the ordinary path returns for the same data.

ONE FACT SHAPES EVERY PROBE. `StatsBase` gives `CovarianceEstimator` four CATCH-ALL `cov`
methods (`cov.jl`), each of which raises a plain `ErrorException` reading "cov is not defined
for ...". So a `cov` call on an estimator and a matrix ALWAYS resolves, and an estimator that
implements nothing raises an `ErrorException`, never a `MethodError`. The `MethodError` the
two retries answer therefore comes from INSIDE the estimator's own body -- which is the real
hazard, because `StatsBase`'s weighted kernels are typed on `DenseMatrix` and reject a
`Transpose`, an `Adjoint` or a `SubArray` (ADR 0065).

`StatsBase` also gives `cor(ce::CovarianceEstimator, X::AbstractMatrix; kwargs...)`, which
slurps. So `compat_cor`'s outer `hasmethod` guard is true for every estimator UNLESS the
estimator declares a narrower `cor` of its own that names no `mean`. Two probes do exactly
that, which is the only way the covariance-to-correlation tail is entered.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, JuMP

# `cov` slurps the extra keyword arguments, so `hasmethod` accepts `scale` and the call is
# made -- then the body rejects it with an error that is NOT a `MethodError`. This is the
# case `compat_cov`'s `rethrow()` exists for: a genuine failure must not be masked by the
# keyword-less retry.
struct KwargThrowCov <: StatsBase.CovarianceEstimator end
function Statistics.cov(::KwargThrowCov, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                        kwargs...)
    return if isempty(kwargs)
        Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims)
    else
        throw(DomainError(:scale, "genuine numerical failure"))
    end
end

# The same shape on the correlation side, for `compat_cor`'s inner `rethrow()`.
struct KwargThrowCor <: StatsBase.CovarianceEstimator end
function Statistics.cor(::KwargThrowCor, X::AbstractMatrix; dims::Int = 1, mean = nothing,
                        kwargs...)
    return if isempty(kwargs)
        Statistics.cor(X; dims = dims)
    else
        throw(DomainError(:scale, "genuine numerical failure"))
    end
end

# A `cov` whose BODY is typed on a dense matrix. A `Transpose` reaches the estimator, the
# body raises a `MethodError`, and `robust_cov` retries once on `Matrix(X)`. This is the
# ADR 0065 hazard in miniature.
function dense_only_cov(X::DenseMatrix, dims::Int)
    return Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims)
end
struct DenseBodyCov <: StatsBase.CovarianceEstimator end
function Statistics.cov(::DenseBodyCov, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return dense_only_cov(X, dims)
end

# A `cor` that names no `mean`. It is narrower than `StatsBase`'s slurping generic, so
# `hasmethod(Statistics.cor, ..., (:dims, :mean))` is FALSE and `compat_cor` goes straight
# to its covariance-to-correlation tail. `cov` returns a `Matrix`, which is mutable, so the
# tail converts in place with `StatsBase.cov2cor!`.
struct MutableCovOnly <: StatsBase.CovarianceEstimator end
function Statistics.cov(::MutableCovOnly, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims)
end
function Statistics.cor(::MutableCovOnly, X::AbstractMatrix; dims::Int = 1)
    return Statistics.cor(X; dims = dims)
end

# The same, but `cov` returns a `Symmetric`, which is an immutable struct. The tail takes
# its other arm and converts a dense copy with `StatsBase.cov2cor`.
struct ImmutableCovOnly <: StatsBase.CovarianceEstimator end
function Statistics.cov(::ImmutableCovOnly, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing)
    return Symmetric(Statistics.cov(StatsBase.SimpleCovariance(), X; dims = dims))
end
function Statistics.cor(::ImmutableCovOnly, X::AbstractMatrix; dims::Int = 1)
    return Statistics.cor(X; dims = dims)
end

# A `cor` typed on a dense matrix, over a `cov` whose body raises a `MethodError` for ANY
# matrix. A `Transpose` therefore fails on both routes -- the `cor` generic falls through to
# the tail, and the tail's `robust_cov` fails on the lazy matrix AND on its dense copy -- so
# the `MethodError` leaves `compat_cor` and `robust_cor` retries once on `Matrix(X)`, where
# the dense `cor` matches.
vector_only_cov(x::AbstractVector) = Statistics.var(x)
struct DenseCorOnly <: StatsBase.CovarianceEstimator end
function Statistics.cov(::DenseCorOnly, X::AbstractMatrix; dims::Int = 1, mean = nothing)
    return vector_only_cov(X)
end
function Statistics.cor(::DenseCorOnly, X::DenseMatrix; dims::Int = 1, mean = nothing)
    return Statistics.cor(X; dims = dims)
end

@testset "compat_cov/compat_cor and robust_cov/robust_cor fallbacks" begin
    rng = StableRNG(20260824)
    X = randn(rng, 48, 4)
    Xt = transpose(permutedims(X))
    sigma = Statistics.cov(StatsBase.SimpleCovariance(), X)
    rho = Statistics.cor(X)

    @testset "compat_cov and robust_cov agree on the same matrix" begin
        # Without a fallback the two kernels are the same call, so any difference between
        # them is the retry, not the estimator.
        sc = StatsBase.SimpleCovariance()
        @test PortfolioOptimisers.compat_cov(sc, X) ≈ sigma
        @test PortfolioOptimisers.robust_cov(sc, X) ≈ sigma
        @test PortfolioOptimisers.compat_cov(sc, X) ≈ PortfolioOptimisers.robust_cov(sc, X)
        @test PortfolioOptimisers.compat_cor(sc, X) ≈ rho
        @test PortfolioOptimisers.robust_cor(sc, X) ≈ rho
        @test PortfolioOptimisers.compat_cor(sc, X) ≈ PortfolioOptimisers.robust_cor(sc, X)
    end

    @testset "an error that is not a MethodError propagates" begin
        # No extra keyword argument: the `hasmethod` guard is not even reached and the
        # plain call answers.
        @test PortfolioOptimisers.compat_cov(KwargThrowCov(), X) ≈ sigma
        # With one, `hasmethod` accepts the slurp, the body raises a `DomainError`, and
        # `compat_cov` rethrows it rather than retrying without the keyword.
        @test_throws DomainError PortfolioOptimisers.compat_cov(KwargThrowCov(), X;
                                                                scale = 2.0)
        # `robust_cov` catches the same error and rethrows it too, so the caller sees the
        # estimator's own failure and not a `MethodError`.
        @test_throws DomainError PortfolioOptimisers.robust_cov(KwargThrowCov(), X;
                                                                scale = 2.0)
        # The correlation kernel has the same guard in its keyword branch.
        @test PortfolioOptimisers.compat_cor(KwargThrowCor(), X) ≈ rho
        @test_throws DomainError PortfolioOptimisers.compat_cor(KwargThrowCor(), X;
                                                                scale = 2.0)
        @test_throws DomainError PortfolioOptimisers.robust_cor(KwargThrowCor(), X;
                                                                scale = 2.0)
    end

    @testset "robust_cov retries once on a dense copy" begin
        # The estimator answers a dense matrix and raises a `MethodError` on a lazy one.
        @test PortfolioOptimisers.compat_cov(DenseBodyCov(), X) ≈ sigma
        @test_throws MethodError PortfolioOptimisers.compat_cov(DenseBodyCov(), Xt)
        # `robust_cov` sees that `MethodError` and retries on `Matrix(X)`, which answers.
        @test PortfolioOptimisers.robust_cov(DenseBodyCov(), Xt) ≈ sigma
        # The retry is a `Matrix` of the same numbers, so the two agree exactly.
        @test PortfolioOptimisers.robust_cov(DenseBodyCov(), Xt) ==
              PortfolioOptimisers.robust_cov(DenseBodyCov(), X)
    end

    @testset "compat_cor falls back to the covariance matrix" begin
        # The narrower `cor` names no `mean`, so the `hasmethod` guard is false and the
        # tail runs. `cov` returns a `Matrix`, so `StatsBase.cov2cor!` converts in place.
        @test !hasmethod(Statistics.cor, Tuple{MutableCovOnly, Matrix{Float64}},
                         (:dims, :mean))
        @test PortfolioOptimisers.compat_cor(MutableCovOnly(), X) ≈ rho
        @test PortfolioOptimisers.robust_cor(MutableCovOnly(), X) ≈ rho
        # The tail is what answered, not the estimator's own `cor`: the two differ in the
        # value they return for `dims = 2`, and the tail's answer is the covariance one.
        @test PortfolioOptimisers.compat_cor(MutableCovOnly(), permutedims(X); dims = 2) ≈
              rho
        # `cov` returns a `Symmetric`, which is immutable, so the other arm converts a
        # dense copy with `StatsBase.cov2cor`.
        @test !ismutable(Statistics.cov(ImmutableCovOnly(), X))
        @test PortfolioOptimisers.compat_cor(ImmutableCovOnly(), X) ≈ rho
        @test PortfolioOptimisers.robust_cor(ImmutableCovOnly(), X) ≈ rho
    end

    @testset "robust_cor retries once on a dense copy" begin
        # Both routes fail on the lazy matrix: the `cor` generic falls through to the tail,
        # and the tail's `robust_cov` fails on the lazy matrix and on its dense copy alike.
        @test_throws MethodError PortfolioOptimisers.compat_cor(DenseCorOnly(), Xt)
        # So the `MethodError` reaches `robust_cor`, which retries on `Matrix(X)`. There the
        # dense `cor` matches and answers.
        @test PortfolioOptimisers.robust_cor(DenseCorOnly(), Xt) ≈ rho
        @test PortfolioOptimisers.compat_cor(DenseCorOnly(), X) ≈ rho
    end

    @testset "assert_dims guards both weighted seams" begin
        w = StatsBase.eweights(1:size(X, 1), 0.3)
        sc = StatsBase.SimpleCovariance()
        # `assert_dims` builds its rejection with `@argcheck`, which raises a `DomainError`.
        @test_throws DomainError PortfolioOptimisers.robust_cov(sc, X; dims = 3)
        @test_throws DomainError PortfolioOptimisers.robust_cov(sc, X, w; dims = 3)
        @test_throws DomainError PortfolioOptimisers.robust_cor(sc, X; dims = 3)
        @test_throws DomainError PortfolioOptimisers.robust_cor(sc, X, w; dims = 3)
    end
end

@testset "densify accepts both element types of its bound" begin
    X = randn(StableRNG(20260824), 6, 3)
    # A dense matrix of numbers is returned unchanged; anything lazy is materialised.
    @test PortfolioOptimisers.densify(X) === X
    @test PortfolioOptimisers.densify(transpose(permutedims(X))) isa Matrix
    # The bound also names `JuMP.AbstractJuMPScalar`, so a dense matrix of variable
    # references is returned unchanged rather than copied.
    model = JuMP.Model()
    JuMP.@variable(model, x[1:3, 1:2])
    @test x isa Matrix{JuMP.VariableRef}
    @test PortfolioOptimisers.densify(x) === x
    @test PortfolioOptimisers.densify(transpose(x)) isa Matrix{JuMP.VariableRef}
end

@testset "weighted_centre is the one place ADR 0088 lives" begin
    rng = StableRNG(20260901)
    X = randn(rng, 40, 5)
    me = SimpleExpectedReturns()
    w = StatsBase.aweights(range(0.5, 1.5; length = size(X, 1)))
    w2 = StatsBase.aweights(reverse(collect(w)))
    # No weights: the centre is the one the estimator computes on its own.
    @test PortfolioOptimisers.weighted_centre(X, me, nothing; dims = 1) ≈
          Statistics.mean(me, X; dims = 1)
    # `dims = 2` names the other axis, so the estimator shapes `mu` along the first one.
    @test size(PortfolioOptimisers.weighted_centre(X, me, nothing; dims = 2)) == (40, 1)
    # Weights reach the centre through `factory`, so the centre is the WEIGHTED mean.
    muw = Statistics.mean(SimpleExpectedReturns(; w = w), X; dims = 1)
    @test PortfolioOptimisers.weighted_centre(X, me, w; dims = 1) ≈ muw
    @test !isapprox(muw, Statistics.mean(me, X; dims = 1))
    # The incoming `w` wins over the weights that `me` carries, which is what `factory`
    # does on every other path.
    @test PortfolioOptimisers.weighted_centre(X, SimpleExpectedReturns(; w = w2), w;
                                              dims = 1) ≈ muw
    # `mean` is the escape hatch. It is returned unchanged, weights set or not.
    given = fill(0.5, 1, 5)
    @test PortfolioOptimisers.weighted_centre(X, me, nothing; dims = 1, mean = given) ===
          given
    @test PortfolioOptimisers.weighted_centre(X, me, w; dims = 1, mean = given) === given
    # `covariance_centre_and_estimator` reads the same verb, so `Covariance` centres where
    # its three siblings do. It also answers the inner estimator, which is the half no
    # sibling has: `ce.ce` passes through untouched when `ce.w` is `nothing`, and takes the
    # weights through `factory_child` when it is not.
    ce0 = Covariance()
    mu0, cel0 = PortfolioOptimisers.covariance_centre_and_estimator(ce0, X; dims = 1)
    @test mu0 ≈ Statistics.mean(me, X; dims = 1)
    @test cel0 === ce0.ce
    cew = Covariance(; w = w)
    muw2, celw = PortfolioOptimisers.covariance_centre_and_estimator(cew, X; dims = 1)
    @test muw2 ≈ muw
    @test celw.w === w
    @test !(celw === cew.ce)
end

@testset "demean_returns subtracts the mean the caller asked for" begin
    rng = StableRNG(20260824)
    X = randn(rng, 40, 5)
    me = SimpleExpectedReturns()
    # With no `mean`, the estimator computes it. `dims = 1` gives one value per column.
    mu1 = Statistics.mean(me, X; dims = 1)
    @test PortfolioOptimisers.demean_returns(X, me; dims = 1) ≈ X .- mu1
    @test all(x -> isapprox(x, 0; atol = 1e-12),
              Statistics.mean(PortfolioOptimisers.demean_returns(X, me; dims = 1);
                              dims = 1))
    # `dims = 2` names the other axis, so the estimator shapes `mu` along the first one.
    mu2 = Statistics.mean(me, X; dims = 2)
    @test PortfolioOptimisers.demean_returns(X, me; dims = 2) ≈ X .- mu2
    @test size(mu1) == (1, 5)
    @test size(mu2) == (40, 1)
    # A supplied `mean` is used as given, and the estimator is not called.
    given = fill(0.5, 1, 5)
    @test PortfolioOptimisers.demean_returns(X, me; dims = 1, mean = given) ≈ X .- given
    # An estimator that carries observation weights subtracts the WEIGHTED mean, so a
    # recency-weighted estimator and an unweighted one give different answers.
    w = StatsBase.eweights(1:size(X, 1), 0.3)
    mew = SimpleExpectedReturns(; w = w)
    muw = Statistics.mean(mew, X; dims = 1)
    @test PortfolioOptimisers.demean_returns(X, mew; dims = 1) ≈ X .- muw
    @test !isapprox(muw, mu1)
    # `demean_returns` allocates: it never writes into the matrix it is given.
    X0 = copy(X)
    PortfolioOptimisers.demean_returns(X, mew; dims = 1)
    @test X == X0
end

@testset "windowed_preamble leaves the estimator it is given alone" begin
    rng = StableRNG(20260824)
    X = randn(rng, 60, 4)
    w = StatsBase.eweights(1:size(X, 1), 0.3)
    est = SimpleVariance(; me = SimpleExpectedReturns())
    # The preamble builds a NEW estimator through `factory`. The one the caller holds must
    # keep its own `w`, which is `nothing` here: an in-place broadcast on a struct field
    # has mutated an estimator in this library before.
    inner, Xw = PortfolioOptimisers.windowed_preamble(est, w, 1:20, X[:, 1])
    @test isnothing(est.w)
    @test isnothing(est.me.w)
    @test inner !== est
    @test length(Xw) == 20
    inner, Xw, iv = PortfolioOptimisers.windowed_preamble(est, w, 1:20, X)
    @test isnothing(est.w)
    @test isnothing(iv)
    @test size(Xw) == (20, 4)
    # The weights the preamble resolved cover the window, not the whole sample.
    @test length(inner.w) == 20
end

@testset "each windowed estimator equals its inner estimator on the slice" begin
    rng = StableRNG(20260824)
    X = randn(rng, 80, 5)
    win = 21:60
    Xw = X[win, :]
    xw = X[win, 1]
    # One estimator per file. The windowed answer must equal the inner estimator run by
    # hand on the same slice -- that is the whole claim of the family.
    @test Statistics.mean(WindowedExpectedReturns(; window = win), X) ≈
          Statistics.mean(SimpleExpectedReturns(), Xw)
    @test Statistics.cov(WindowedCovariance(; window = win), X) ≈
          Statistics.cov(PortfolioOptimisersCovariance(), Xw)
    @test Statistics.cor(WindowedCovariance(; window = win), X) ≈
          Statistics.cor(PortfolioOptimisersCovariance(), Xw)
    # `WindowedVariance` forwards four generics: `var` and `std`, over a matrix and over a
    # vector. All four must land on the same slice.
    @test Statistics.var(WindowedVariance(; window = win), X) ≈
          Statistics.var(SimpleVariance(), Xw)
    @test Statistics.std(WindowedVariance(; window = win), X) ≈
          Statistics.std(SimpleVariance(), Xw)
    @test Statistics.var(WindowedVariance(; window = win), X[:, 1]) ≈
          Statistics.var(SimpleVariance(), xw)
    @test Statistics.std(WindowedVariance(; window = win), X[:, 1]) ≈
          Statistics.std(SimpleVariance(), xw)
    # `WindowedCoskewness` forwards a pair, so both values are checked.
    sk0, V0 = coskewness(WindowedCoskewness(; window = win), X)
    sk1, V1 = coskewness(Coskewness(), Xw)
    @test sk0 ≈ sk1
    @test V0 ≈ V1
    @test cokurtosis(WindowedCokurtosis(; window = win), X) ≈ cokurtosis(Cokurtosis(), Xw)
end

@testset "the no-op seams of the moment layer" begin
    rng = StableRNG(20260824)
    w = StatsBase.eweights(1:40, 0.3)
    # `factory_child` maps over an array of covariance estimators, one call per element.
    ces = [Covariance(), PortfolioOptimisersCovariance()]
    out = PortfolioOptimisers.factory_child(ces, w)
    @test out isa Vector
    @test length(out) == 2
    @test out[1].ce.w === w
    @test out[2].ce.ce.w === w
    # A single estimator takes the other method and answers a single estimator.
    @test PortfolioOptimisers.factory_child(ces[1], w) isa Covariance
    # An expected returns algorithm carries no asset axis, so its view is itself.
    for alg in (GrandMean(), VolatilityWeighted(), MeanSquaredError())
        @test PortfolioOptimisers.port_opt_view(alg, nothing) === alg
        @test PortfolioOptimisers.port_opt_view(alg, 1:3, nothing) === alg
    end
    # The estimator-level fallback answers the same way.
    me = SimpleExpectedReturns()
    @test PortfolioOptimisers.port_opt_view(me, nothing) === me
end

#=
Check the three dependence-based covariances -- distance, lower tail and rank -- against the
mathematics their docstrings state. Issue #458 of child map 3 (#417) under the map of maps
(#404).

Coverage was ALREADY terminal for all three files, so nothing here exists to reach a line.
Every test below pins a CLAIM the documentation makes, computed a second way. #404's
condition 2 is "checked with real numbers. Not read -- run", and a claim no test holds is a
claim the next reader has to re-derive.

THREE FACTS SHAPE THE PROBES.

1. `StatsBase.corspearman` and `StatsBase.corkendall` are TIE-CORRECTED. The textbook closed
   forms -- Spearman's `1 - 6 sum(d^2) / (T(T^2-1))` and Kendall's `(C-D)/binomial(T,2)` --
   are the NO-TIES case only, and the two disagree measurably on a tied sample. Both rank
   docstrings state the tie-corrected form and name the disagreeing pair of numbers, so both
   sides of that pair are asserted here.

2. `DistanceCovariance` carries NO variance estimator field, so its `cov` is not the generic
   `ve` fallback of `01_Base_Moments.jl`. The diagonal is the DISTANCE standard deviation,
   and `cor` is the matrix rescaled by the square roots of that same diagonal. The other two
   estimators DO take the fallback, and their `cov` diagonal is the ordinary sample variance.

3. `lower_tail_dependence`'s `k > 0` guard is NOT reached by a small positive `alpha`. `ceil`
   of any positive real is at least 1, so `alpha = nextfloat(0.0)` on a non-empty matrix
   still selects one observation. The cases that do reach it are `alpha <= 0` on a direct
   call, and a matrix with no observation. The docstring said otherwise before #458.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs, FLoops

@testset "Distance covariance: the doubly-centred statistic" begin
    rng = StableRNG(987654321)
    X = randn(rng, 40, 4)
    ce = DistanceCovariance(; ex = FLoops.SequentialEx())

    # The reference builds the centring by hand from the Euclidean distance on the line.
    function ref_dcov(v1, v2)
        n = length(v1)
        a = [abs(v1[k] - v1[l]) for k in 1:n, l in 1:n]
        b = [abs(v2[k] - v2[l]) for k in 1:n, l in 1:n]
        A = a .- mean(a; dims = 1) .- mean(a; dims = 2) .+ mean(a)
        B = b .- mean(b; dims = 1) .- mean(b; dims = 2) .+ mean(b)
        return (sqrt(sum(A .* B) / n^2), sqrt(sum(A .* A) / n^2), sqrt(sum(B .* B) / n^2))
    end

    v1, v2 = view(X, :, 1), view(X, :, 2)
    dxy, dxx, dyy = ref_dcov(v1, v2)
    @test PortfolioOptimisers.cov_distance(ce, v1, v2) ≈ dxy
    @test PortfolioOptimisers.cor_distance(ce, v1, v2) ≈ dxy / sqrt(dxx * dyy)

    # A series against itself is exactly 1, and an independent pair is far below a
    # non-linearly dependent one that Pearson cannot see at all.
    @test isone(PortfolioOptimisers.cor_distance(ce, v1, v1))
    rng2 = StableRNG(24680)
    u = randn(rng2, 400)
    indep = randn(rng2, 400)
    nonlin = u .^ 2
    c_indep = PortfolioOptimisers.cor_distance(ce, u, indep)
    c_nonlin = PortfolioOptimisers.cor_distance(ce, u, nonlin)
    @test c_indep < 0.1
    @test c_nonlin > 0.5
    @test abs(cor(u, nonlin)) < 0.2

    # `cov` overrides the generic `ve` fallback: the diagonal is the distance standard
    # deviation, and `cor` is `cov` rescaled by the square roots of that diagonal.
    Sigma = cov(ce, X)
    Rho = cor(ce, X)
    @test diag(Sigma) ≈
          [PortfolioOptimisers.cov_distance(ce, view(X, :, i), view(X, :, i)) for i in 1:4]
    d = sqrt.(diag(Sigma))
    @test Rho ≈ Sigma ./ (d * d')
    @test all(isone, diag(Rho))
    @test !hasproperty(ce, :ve)
end

@testset "Distance covariance: pairwise distances and the asset axis" begin
    # Non-square on purpose: 7 observations, 3 assets, so a transposed index cannot hide.
    Xns = randn(StableRNG(1111), 7, 3)
    ce = DistanceCovariance(; ex = FLoops.SequentialEx())

    @test size(cov(ce, Xns)) == (3, 3)
    @test size(cor(ce, Xns)) == (3, 3)
    @test cov(ce, permutedims(Xns); dims = 2) == cov(ce, Xns)
    @test cor(ce, permutedims(Xns); dims = 2) == cor(ce, Xns)

    # `calc_pairwise_dists` returns two T x T matrices of the metric over the observations.
    # It takes NO weights: #851 moved them off the data and onto the statistic.
    v1, v2 = view(Xns, :, 1), view(Xns, :, 2)
    D1, D2 = PortfolioOptimisers.calc_pairwise_dists(ce, v1, v2)
    @test size(D1) == size(D2) == (7, 7)
    @test D1 ≈ [abs(Xns[k, 1] - Xns[l, 1]) for k in 1:7, l in 1:7]
    @test !hasmethod(PortfolioOptimisers.calc_pairwise_dists,
                     Tuple{DistanceCovariance, PortfolioOptimisers.VecNum,
                           PortfolioOptimisers.VecNum, StatsBase.AbstractWeights})

    # `calc_centred_dists` doubly centres, and its three means carry the weights.
    w = pweights(range(0.5, 1.5; length = 7))
    A = PortfolioOptimisers.calc_centred_dists(D1, w)
    mu1 = [sum(w[k] * D1[k, l] for k in 1:7) / sum(w) for l in 1:7]
    mu2 = [sum(w[l] * D1[k, l] for l in 1:7) / sum(w) for k in 1:7]
    mu3 = sum(w[k] * w[l] * D1[k, l] for k in 1:7, l in 1:7) / sum(w)^2
    @test A ≈ [D1[k, l] - mu2[k] - mu1[l] + mu3 for k in 1:7, l in 1:7]
    @test PortfolioOptimisers.calc_centred_dists(D1, nothing) ≈
          D1 .- mean(D1; dims = 1) .- mean(D1; dims = 2) .+ mean(D1)

    # `calc_dcov2` contracts the two centred matrices on both indices.
    B = PortfolioOptimisers.calc_centred_dists(D2, w)
    @test PortfolioOptimisers.calc_dcov2(A, B, w) ≈
          sum(w[k] * w[l] * A[k, l] * B[k, l] for k in 1:7, l in 1:7) / sum(w)^2
    Au = PortfolioOptimisers.calc_centred_dists(D1, nothing)
    Bu = PortfolioOptimisers.calc_centred_dists(D2, nothing)
    @test PortfolioOptimisers.calc_dcov2(Au, Bu, nothing) ≈ sum(Au .* Bu) / 7^2
end

@testset "Distance covariance: a uniform weight is a no-op, #851" begin
    X = randn(StableRNG(987654321), 40, 4)
    wv = pweights(range(0.5, 1.5; length = 40))
    cew = DistanceCovariance(; w = wv, ex = FLoops.SequentialEx())
    ceu = DistanceCovariance(; ex = FLoops.SequentialEx())

    resolved = PortfolioOptimisers.get_observation_weights(cew.w, X)
    @test cov(cew, X)[1, 2] ==
          PortfolioOptimisers.cov_distance(cew, view(X, :, 1), view(X, :, 2), resolved)
    @test cov(cew, X) != cov(ceu, X)
    @test isnothing(PortfolioOptimisers.get_observation_weights(ceu.w, X))

    #=
    #851: the weights used to multiply the DATA, so a uniform `pweights` of `1/T` divided
    every pairwise distance by `T` and rescaled the whole covariance by `1/T`. They now
    weight the STATISTIC, so a constant weight cancels from every quotient and gives the
    unweighted answer back, whatever the constant is. The correlation route always hid the
    defect, because a common scale cancels in the ratio.
    =#
    T = size(X, 1)
    sigma0 = cov(ceu, X)
    for wc in (pweights(fill(inv(T), T)), pweights(ones(T)), pweights(fill(7.5, T)))
        cec = DistanceCovariance(; w = wc, ex = FLoops.SequentialEx())
        @test isapprox(cov(cec, X), sigma0; rtol = 1e-14)
        @test isapprox(cor(cec, X), cor(ceu, X); rtol = 1e-14)
    end

    # A non-uniform weight is a tilt and not a rescale: it moves no single common factor.
    ratio = cov(cew, X) ./ sigma0
    @test !isapprox(extrema(ratio)..., rtol = 1e-6)
end

@testset "Lower tail dependence: the joint count at the quantile" begin
    Xt = randn(StableRNG(555), 100, 3)
    T, N = size(Xt)
    alpha = 0.05
    k = ceil(Int, T * alpha)
    @test k == 5

    # The threshold is the k-th ORDER STATISTIC, and the estimate is the joint count over k.
    qs = [sort(Xt[:, i])[k] for i in 1:N]
    ref = [count((Xt[:, i] .<= qs[i]) .& (Xt[:, j] .<= qs[j])) / k for i in 1:N, j in 1:N]
    mv = sqrt(eps(eltype(Xt)))
    code = PortfolioOptimisers.lower_tail_dependence(Xt, alpha)
    @test code ≈ clamp.(ref, mv, one(eltype(Xt)))
    @test all(isone, diag(code))
    @test issymmetric(code)

    # The floor is NOT zero. A pair whose tails never coincide is reported as sqrt(eps).
    @test iszero(ref[1, 3])
    @test code[1, 3] == mv
end

@testset "Lower tail dependence: the k == 0 guard" begin
    Xt = randn(StableRNG(555), 100, 3)

    # No small POSITIVE alpha reaches the guard: `ceil` of a positive real is at least 1.
    @test ceil(Int, 100 * 1e-9) == 1
    @test ceil(Int, 100 * nextfloat(0.0)) == 1
    tiny = PortfolioOptimisers.lower_tail_dependence(Xt, 1 / 100)
    @test all(isone, diag(tiny))
    @test !all(iszero, tiny)

    # The cases that DO reach it: alpha <= 0, and a matrix with no observation.
    @test all(iszero, PortfolioOptimisers.lower_tail_dependence(Xt, 0.0))
    @test all(iszero, PortfolioOptimisers.lower_tail_dependence(Xt, -0.5))
    empty_res = PortfolioOptimisers.lower_tail_dependence(zeros(0, 3), 0.05)
    @test size(empty_res) == (3, 3)
    @test all(iszero, empty_res)

    # The function does not check `k <= T`: an alpha above one fails in `partialsort!`.
    @test_throws BoundsError PortfolioOptimisers.lower_tail_dependence(Xt, 1.5)

    # The estimator rules the first case out at construction.
    @test_throws DomainError LowerTailDependenceCovariance(; alpha = 0.0)
    @test_throws DomainError LowerTailDependenceCovariance(; alpha = 1.0)
end

@testset "Rank covariances: the tie correction" begin
    # One tied pair in each series, so the tie-corrected and closed forms MUST disagree.
    Xtie = [1.0 1.0; 2.0 1.0; 2.0 3.0; 4.0 4.0; 5.0 2.0]
    T = size(Xtie, 1)

    rx, ry = tiedrank(Xtie[:, 1]), tiedrank(Xtie[:, 2])
    sp_closed = 1 - 6 * sum((rx .- ry) .^ 2) / (T * (T^2 - 1))
    C = D = 0
    for t in 1:T, s in (t + 1):T
        p = (Xtie[t, 1] - Xtie[s, 1]) * (Xtie[t, 2] - Xtie[s, 2])
        p > 0 && (C += 1)
        p < 0 && (D += 1)
    end
    kd_closed = (C - D) / binomial(T, 2)

    sp = cor(SpearmanCovariance(), Xtie)[1, 2]
    kd = cor(KendallCovariance(), Xtie)[1, 2]

    # The exact pair of numbers both docstrings name.
    @test sp == 0.5526315789473685
    @test sp_closed == 0.575
    @test kd == 0.4444444444444444
    @test kd_closed == 0.4
    @test sp != sp_closed
    @test kd != kd_closed

    # Spearman IS the Pearson correlation of the mid-ranks, and Kendall IS tau_b.
    @test sp == cor(rx, ry)
    n0 = binomial(T, 2)
    nx = ny = binomial(2, 2)   # one tied group of size two in each series
    @test kd == (C - D) / sqrt((n0 - nx) * (n0 - ny))
end

@testset "Rank covariances: without ties the closed forms return" begin
    Xun = [1.0 2.0; 2.0 1.0; 3.0 4.0; 4.0 3.0; 5.0 5.0]
    T = size(Xun, 1)
    ru, su = tiedrank(Xun[:, 1]), tiedrank(Xun[:, 2])
    C = D = 0
    for t in 1:T, s in (t + 1):T
        p = (Xun[t, 1] - Xun[s, 1]) * (Xun[t, 2] - Xun[s, 2])
        p > 0 && (C += 1)
        p < 0 && (D += 1)
    end
    @test cor(SpearmanCovariance(), Xun)[1, 2] ≈
          1 - 6 * sum((ru .- su) .^ 2) / (T * (T^2 - 1))
    @test cor(KendallCovariance(), Xun)[1, 2] ≈ (C - D) / binomial(T, 2)
end

@testset "Rank covariances: Example 3 of kendall1945" begin
    # The ten ranks of Examples 1 and 3. The work scores 33 over sqrt(41 * 36) and prints
    # the truncated 0.8589, so the docstring names the unrounded quotient.
    Xk = [1.0 1.0; 2.5 2.0; 2.5 4.5; 4.5 4.5; 4.5 4.5; 6.5 4.5; 6.5 8.0; 8.0 8.0; 9.5 8.0;
          9.5 10.0]
    tau = cor(KendallCovariance(), Xk)[1, 2]
    @test tau == 33 / sqrt(41 * 36)
    @test round(tau; digits = 5) == 0.85896
    @test trunc(tau; digits = 4) == 0.8589
end

@testset "Dependence covariances: a constant series" begin
    # A constant column has a distance variance of zero, so its distance correlations are
    # 0 / 0. The rank correlations keep a unit diagonal and are NaN off it.
    Xc = [1.0 2.0; 1.0 4.0; 1.0 6.0; 1.0 3.0]
    ce = DistanceCovariance(; ex = FLoops.SequentialEx())
    @test all(isnan, cor(ce, Xc)[1, :])
    @test isone(cor(ce, Xc)[2, 2])
    @test iszero(cov(ce, Xc)[1, 1])
    @test iszero(PortfolioOptimisers.cov_distance(ce, Xc[:, 1], Xc[:, 1]))
    @test isnan(PortfolioOptimisers.cor_distance(ce, Xc[:, 1], Xc[:, 2]))
    for est in (KendallCovariance(), SpearmanCovariance())
        R = cor(est, Xc)
        @test all(isone, diag(R))
        @test isnan(R[1, 2]) && isnan(R[2, 1])
    end
end

@testset "The generic ve fallback rescales the three that carry one" begin
    Xt = randn(StableRNG(555), 100, 3)
    for est in (SpearmanCovariance(), KendallCovariance(),
                LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()))
        R = cor(est, Xt)
        S = cov(est, Xt)
        v = vec(std(est.ve, Xt))
        @test S ≈ R .* (v * v')
        @test diag(S) ≈ v .^ 2
        @test all(isone, diag(R))
        @test est isa PortfolioOptimisers.AbstractCovarianceEstimator
    end

    # The three correlation matrices are genuinely different statistics.
    @test cor(SpearmanCovariance(), Xt) != cor(KendallCovariance(), Xt)
    @test cor(SpearmanCovariance(), Xt) !=
          cor(LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()), Xt)
end

@testset "The four estimators honour dims" begin
    #=
    `cor` is EXACT under the transpose for all four, but `cov` is not for the three that take
    the generic `ve` fallback: `std(ve, X)` sums a transposed view in a different order, and
    the three matrices differ by 4.4e-16. `DistanceCovariance`, which overrides the fallback,
    is exact on both. So the equality here is `isapprox` and not `==` -- a tightening to `==`
    would be a knife-edge test, not a stronger one.
    =#
    Xt = randn(StableRNG(555), 30, 3)
    for est in (SpearmanCovariance(), KendallCovariance(),
                LowerTailDependenceCovariance(; ex = FLoops.SequentialEx()),
                DistanceCovariance(; ex = FLoops.SequentialEx()))
        @test cor(est, permutedims(Xt); dims = 2) == cor(est, Xt)
        @test cov(est, permutedims(Xt); dims = 2) ≈ cov(est, Xt)
        @test_throws DomainError cor(est, Xt; dims = 3)
        @test_throws DomainError cov(est, Xt; dims = 3)
    end
    @test cov(DistanceCovariance(; ex = FLoops.SequentialEx()), permutedims(Xt);
              dims = 2) == cov(DistanceCovariance(; ex = FLoops.SequentialEx()), Xt)
end

#=
Check `src/05_Moments/05_Gerber/02_SmythBrobyCovariance.jl` against the mathematics its docstrings
state, and against the source itself. Issue #455 of child map 3 (#417) under the map of maps
(#404).

Coverage was ALREADY terminal for this file -- 95 executable lines, 0 misses -- so nothing
here exists to reach a line. Every test below pins a CLAIM, computed a second way. #404's
condition 2 is "checked with real numbers. Not read -- run".

FOUR FACTS SHAPE THE PROBES.

1. THE SOURCE DEFINES EXACTLY ONE STATISTIC, and it is `SmythBroby1`. Its equation (5) is
   `sum(delta_C - delta_D) / sum(delta_C + delta_D + delta_N)`. The trailing `0` and `2` are
   the library's own reductions, on the `Gerber0` and `Gerber2` shape. The `SmythBrobyGerber*`
   prefix reads one SENTENCE of the source's conclusion -- "maintaining a count in parallel
   with the sum and then scaling the total sum contribution to match the total sum count" --
   which carries no equation. The `SmythBrobyCount*` prefix is nowhere in the source; the
   source's whole argument is that a contribution sum beats a count. So `sb_reference` below
   is written FROM the source's equations (1)-(5) and its three zones, and the count and
   Gerber prefixes are checked against the library's own stated formulas instead.

2. THE TWO ZONES READ DIFFERENT RETURNS. The confusion zone compares the RAW, UNCENTRED
   return against `c1 * sigma` and rejects only when BOTH assets fall inside it. The
   indecision zone and the outer cut-off compare the CENTRED, STANDARDISED return. That mix
   is the source's own, and `The gate seam` testset shows it is not cosmetic.

3. A MARKER NAMES TWO INDEPENDENT CHOICES. The prefix picks the score triple through
   `sb_pair_scores`; the trailing digit picks the denominator through `comovement_ratio`,
   and it means the same thing as it does in the Gerber family. Nine markers is the shape
   where an unreachable branch hides, so all nine are separated pairwise.

4. `n` IS NON-NEGATIVE, AND `Inf` IS LEGAL (ADR 0091, #496). `c1`, `c2` and `c3` pass
   through `assert_nonempty_nonneg_finite_val`, which also rejects `Inf`; `n` passes
   through `assert_nonneg`, which does not. The three thresholds are read on the scale of
   the data, where `Inf` admits no observation; `n` is an exponent whose infinite limit is
   a hard divergence gate. A NEGATIVE `n` INVERTED the severity penalty: `gamma^n` is `Inf`
   at the zero divergence an asset always has against itself, so the DIAGONAL was zero
   while the off-diagonal was not. That is what the guard now rejects.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs

const PO = PortfolioOptimisers

#=
The source's own statistic, written from its equations rather than from the library.

Equations (1)-(4): `kappa = sqrt((1+|r|)(1+|r|))`, `gamma = ||r_i| - |r_j||`,
`delta = kappa / (1 + gamma^n)`, with `n = 2` the source's default.

The three zones, in the source's words: an observation is eliminated when
`|r_i| < c1 s_i AND |r_j| < c1 s_j` (confusion, on the RAW return); a co-movement in the
indecision zone `|rt_i|, |rt_j| < c2` "is not used in the decision-making process"; and data
"outside the square formed by |rt_i|, |rt_j| < 4" are ignored.

`family` and `variant` are the library's two independent choices, so the same reference
serves all nine markers.
=#
function sb_reference(X, mu, sd; c1 = 0.5, c2 = 0.5, c3 = 4, n = 2, family = :delta,
                      variant = 1)
    T, N = size(X)
    H = zeros(N, N)
    for i in 1:N, j in 1:N
        p = q = u = 0.0
        cp = cq = cu = 0
        for t in 1:T
            xi, xj = X[t, i], X[t, j]
            # The confusion zone: raw, uncentred, and AND across both assets.
            if abs(xi) < c1 * sd[i] && abs(xj) < c1 * sd[j]
                continue
            end
            ri = (xi - mu[i]) / sd[i]
            rj = (xj - mu[j]) / sd[j]
            ari, arj = abs(ri), abs(rj)
            # The outer cut-off.
            if ari > c3 || arj > c3
                continue
            end
            # The indecision zone.
            if ari < c2 && arj < c2
                continue
            end
            d = sqrt((1 + ari) * (1 + arj)) / (1 + abs(ari - arj)^n)
            if ari >= c2 && arj >= c2 && ri * rj > 0
                p += d
                cp += 1
            elseif ari >= c2 && arj >= c2 && ri * rj < 0
                q += d
                cq += 1
            else
                u += d
                cu += 1
            end
        end
        P, Q, U = if family === :delta
            (p, q, u)
        elseif family === :gerber
            (p * cp, q * cq, u * cu)
        else
            (float(cp), float(cq), float(cu))
        end
        H[i, j] = if variant == 0
            iszero(P + Q) ? 0.0 : (P - Q) / (P + Q)
        elseif variant == 1
            iszero(P + Q + U) ? 0.0 : (P - Q) / (P + Q + U)
        else
            P - Q
        end
    end
    if variant == 2
        h = max.(sqrt.(diag(H)), sqrt(eps(Float64)))
        H = H ./ (h * h')
    end
    return H
end

@testset "The Smyth-Broby sweep of #455" begin
    @testset "sb_delta reproduces the source's equations (1) to (4)" begin
        # Both arguments are ALREADY absolute, so `gamma` is the difference of the two
        # MAGNITUDES, which is what equation (3) writes as a one-dimensional Manhattan
        # distance over magnitudes.
        for (ri, rj, n) in
            ((1.7, 0.4, 2), (0.5, 0.5, 2), (3.0, 0.25, 1), (2.5, 1.0, 3), (0.0, 0.0, 2))
            kappa = sqrt((1 + ri) * (1 + rj))
            gamma = abs(ri - rj)
            @test PO.sb_delta(ri, rj, n) == kappa / (1 + gamma^n)
        end

        # It is symmetric in its two returns, and at equal magnitudes the penalty vanishes,
        # so the contribution is exactly the amplitude.
        @test PO.sb_delta(1.7, 0.4, 2) == PO.sb_delta(0.4, 1.7, 2)
        @test PO.sb_delta(2.0, 2.0, 2) == 3.0
        @test PO.sb_delta(0.0, 0.0, 2) == 1.0

        # A pair that moves far together beats a pair that moves far apart, which is the
        # source's whole reason for replacing the vote.
        @test PO.sb_delta(3.0, 3.0, 2) > PO.sb_delta(3.0, 0.5, 2)
    end

    @testset "every one of the nine markers gives a different matrix" begin
        algs = (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber0(),
                SmythBrobyGerber1(), SmythBrobyGerber2(), SmythBrobyCount0(),
                SmythBrobyCount1(), SmythBrobyCount2())
        X = randn(StableRNG(20260824), 60, 5)
        rhos = [cor(SmythBrobyCovariance(; alg = a, pdm = nothing), X) for a in algs]
        for i in 1:9, j in (i + 1):9
            @test !isapprox(rhos[i], rhos[j])
        end
        # Nine markers and one default. Every correlation diagonal is unit, whichever way
        # the marker reaches it: by a per-pair denominator, or by the whole-matrix
        # normalisation of the trailing `2`.
        for r in rhos
            @test all(isapprox.(diag(r), 1))
        end
    end

    @testset "all nine markers reproduce the source-derived reference" begin
        X = randn(StableRNG(4242), 80, 6) .* [0.5 1.0 2.0 0.8 1.5 0.3] .+
            [0.1 -0.2 0.0 0.3 -0.1 0.05]
        ce0 = SmythBrobyCovariance(; pdm = nothing)
        mu = vec(mean(ce0.me, X; dims = 1))
        sd = vec(std(ce0.ve, X; dims = 1))
        fams = (:delta => (SmythBroby0(), SmythBroby1(), SmythBroby2()),
                :gerber => (SmythBrobyGerber0(), SmythBrobyGerber1(), SmythBrobyGerber2()),
                :count => (SmythBrobyCount0(), SmythBrobyCount1(), SmythBrobyCount2()))
        for (fam, algs) in fams, (v, a) in enumerate(algs)
            got = cor(SmythBrobyCovariance(; alg = a, pdm = nothing), X)
            want = sb_reference(X, mu, sd; family = fam, variant = v - 1)
            @test got == want
        end
    end

    @testset "the count prefix recovers the classic Gerber statistic" begin
        #=
        The `SmythBrobyCount*` docstrings claim that dropping `sb_delta` "recovers a Gerber
        statistic evaluated on the Smyth-Broby admission rule rather than on the Gerber
        threshold". Switch the confusion zone off, lift the outer cut-off, and centre on
        zero, and the two admission rules become the SAME rule. The two families then agree
        BIT-FOR-BIT, which is what makes the claim a fact rather than a resemblance.
        =#
        X = randn(StableRNG(31415), 120, 5)
        X .-= mean(X; dims = 1)
        me0 = CustomValueExpectedReturns(; val = 0.0)
        t = 0.5
        for (galg, salg) in
            ((Gerber0(), SmythBrobyCount0()), (Gerber1(), SmythBrobyCount1()),
             (Gerber2(), SmythBrobyCount2()))
            g = cor(GerberCovariance(; alg = galg, t = t, pdm = nothing), X)
            s = cor(SmythBrobyCovariance(; alg = salg, c1 = 0.0, c2 = t, c3 = 1e6, me = me0,
                                         pdm = nothing), X)
            @test g == s
        end

        # The delta prefix does NOT reduce to Gerber under the same settings: it sums
        # contributions where Gerber counts votes, which is the source's own point.
        d = cor(SmythBrobyCovariance(; alg = SmythBroby1(), c1 = 0.0, c2 = t, c3 = 1e6,
                                     me = me0, pdm = nothing), X)
        @test !isapprox(d,
                        cor(GerberCovariance(; alg = Gerber1(), t = t, pdm = nothing), X))
    end

    @testset "the gate seam: the confusion zone reads the UNCENTRED return" begin
        #=
        `comovement_step` gates on the raw return and zones on the centred one. The
        docstring says that mix is the source's and that centring the gate as well moves the
        statistic. On a strongly off-centre sample it moves it by more than 5 points of
        correlation, so the seam is load-bearing rather than cosmetic.
        =#
        X = randn(StableRNG(2718), 90, 4) .+ [3.0 -2.0 0.5 0.0]
        ce = SmythBrobyCovariance(; alg = SmythBroby1(), pdm = nothing)
        mu = vec(mean(ce.me, X; dims = 1))
        sd = vec(std(ce.ve, X; dims = 1))
        as_is = PO.smythbroby(ce, X, mu, sd)
        @test as_is == sb_reference(X, mu, sd)

        # The same reference with the gate CENTRED instead: `mu` is subtracted before the
        # `c1` comparison. Everything else is identical.
        centred = sb_reference(X, zeros(length(mu)), sd; c1 = 0.5)
        centred_full = let c1 = 0.5, c2 = 0.5, c3 = 4, n = 2
            T, N = size(X)
            H = zeros(N, N)
            for i in 1:N, j in 1:N
                p = q = u = 0.0
                for t in 1:T
                    ri = (X[t, i] - mu[i]) / sd[i]
                    rj = (X[t, j] - mu[j]) / sd[j]
                    (abs(ri) < c1 && abs(rj) < c1) && continue
                    ari, arj = abs(ri), abs(rj)
                    (ari > c3 || arj > c3) && continue
                    (ari < c2 && arj < c2) && continue
                    d = sqrt((1 + ari) * (1 + arj)) / (1 + abs(ari - arj)^n)
                    if ari >= c2 && arj >= c2 && ri * rj > 0
                        p += d
                    elseif ari >= c2 && arj >= c2 && ri * rj < 0
                        q += d
                    else
                        u += d
                    end
                end
                H[i, j] = iszero(p + q + u) ? 0.0 : (p - q) / (p + q + u)
            end
            H
        end
        @test !isapprox(as_is, centred_full)
        @test maximum(abs, as_is .- centred_full) > 0.05
        @test centred isa Matrix   # the third reading exists and is a third answer again
    end

    @testset "the two mutating kernels write into rho and into nothing else" begin
        X = randn(StableRNG(99), 50, 4)
        ce = SmythBrobyCovariance(; alg = SmythBroby2(), pdm = nothing)
        mu = vec(mean(ce.me, X; dims = 1))
        sd = vec(std(ce.ve, X; dims = 1))
        Xc, muc, sdc = copy(X), copy(mu), copy(sd)
        pol = PO.SmythBrobyKernel(ce.alg, mu, sd, ce.c1, ce.c2, ce.c3, ce.n)

        rho = Matrix{Float64}(undef, 4, 4)
        PO.gerber_comovement!(rho, ce.ex, X, pol)
        @test X == Xc
        @test mu == muc
        @test sd == sdc
        @test pol.alg === ce.alg

        raw = copy(rho)
        PO.standardise_comovement!(ce.alg, rho)
        @test X == Xc
        @test mu == muc
        @test sd == sdc
        @test rho != raw
        # The normalisation is what puts a unit diagonal on a matrix of net scores.
        @test all(isapprox.(diag(rho), 1))
        @test !all(isapprox.(diag(raw), 1))

        # The fall-through method is a no-op for every marker that is not a trailing `2`.
        for a in (SmythBroby0(), SmythBroby1(), SmythBrobyGerber1(), SmythBrobyCount1(),
                  Gerber1())
            r2 = copy(rho)
            PO.standardise_comovement!(a, r2)
            @test r2 == rho
        end
    end

    @testset "comovement_ratio guards each denominator" begin
        for (alg, want) in ((SmythBroby0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (SmythBrobyGerber0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (SmythBrobyCount0(), (6.0 - 2.0) / (6.0 + 2.0)),
                            (Gerber0(), (6.0 - 2.0) / (6.0 + 2.0)))
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) == want
            @test PO.comovement_ratio(alg, 0.0, 0.0, 5.0, Float64) === 0.0
        end
        for alg in (SmythBroby1(), SmythBrobyGerber1(), SmythBrobyCount1(), Gerber1())
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) ==
                  (6.0 - 2.0) / (6.0 + 2.0 + 5.0)
            @test PO.comovement_ratio(alg, 0.0, 0.0, 0.0, Float64) === 0.0
        end
        # The trailing `2` branch applies no denominator and reads `nn` at all.
        for alg in (SmythBroby2(), SmythBrobyGerber2(), SmythBrobyCount2(), Gerber2())
            @test PO.comovement_ratio(alg, 6.0, 2.0, 5.0, Float64) == 4.0
            @test PO.comovement_ratio(alg, 6.0, 2.0, 0.0, Float64) == 4.0
        end
    end

    @testset "sb_pair_scores selects by prefix and comovement_ratio by digit" begin
        acc = (pos = 7.0, neg = 3.0, nn = 2.0, cpos = 5, cneg = 4, cnn = 6)
        @test PO.sb_pair_scores(SmythBroby1(), acc) == (7.0, 3.0, 2.0)
        @test PO.sb_pair_scores(SmythBrobyGerber1(), acc) == (35.0, 12.0, 12.0)
        @test PO.sb_pair_scores(SmythBrobyCount1(), acc) == (5, 4, 6)

        # `sb_add_*` fills exactly the slot its name says, and mutates nothing.
        zero_acc = (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        d = PO.sb_delta(1.5, 0.5, 2)
        @test PO.sb_add_pos(SmythBroby1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = d, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)
        @test PO.sb_add_pos(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = d, neg = 0.0, nn = 0.0, cpos = 1, cneg = 0, cnn = 0)
        @test PO.sb_add_pos(SmythBrobyCount1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 1, cneg = 0, cnn = 0)
        @test PO.sb_add_neg(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = d, nn = 0.0, cpos = 0, cneg = 1, cnn = 0)
        @test PO.sb_add_neutral(SmythBrobyGerber1(), zero_acc, 1.5, 0.5, 2) ==
              (pos = 0.0, neg = 0.0, nn = d, cpos = 0, cneg = 0, cnn = 1)
        @test zero_acc == (pos = 0.0, neg = 0.0, nn = 0.0, cpos = 0, cneg = 0, cnn = 0)

        # Only a trailing `1` marker tracks a neutral observation. Every other marker takes
        # the fall-through, so its neutral score stays at zero and no reduction reads it.
        for a in (SmythBroby0(), SmythBroby2(), SmythBrobyGerber0(), SmythBrobyGerber2(),
                  SmythBrobyCount0(), SmythBrobyCount2())
            @test PO.sb_add_neutral(a, zero_acc, 1.5, 0.5, 2) === zero_acc
        end
    end

    @testset "the thresholds at their boundaries" begin
        X = randn(StableRNG(11), 60, 4)
        sdX = vec(std(SimpleVariance(), X; dims = 1))

        # A `c1` above every raw observation puts every pair in the confusion zone, so no
        # observation is admitted and every marker returns zero rather than a NaN. The
        # diagonal is a definition rather than a measurement, so `comovement_unit_diagonal!`
        # writes one onto it and the answer is the identity. #495, ADR 0093.
        big = maximum(abs, X) / minimum(sdX) + 1
        for a in (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber1(),
                  SmythBrobyCount1())
            r = cor(SmythBrobyCovariance(; c1 = big, alg = a, pdm = nothing), X)
            @test isapprox(r, I(4))
            @test all(isfinite, r)
            @test all(iszero, r - Matrix(1.0I, 4, 4))
        end

        # A `c3` at the bottom of its range rejects on the outer cut-off instead, with the
        # same result and by a different arm.
        r = cor(SmythBrobyCovariance(; c2 = 0.0, c3 = 1e-12, alg = SmythBroby1(),
                                     pdm = nothing), X)
        @test isapprox(r, I(4))

        # At `c2 = 0` no observation is one-sided, so the neutral score is empty and the
        # trailing `0` and `1` markers coincide exactly.
        r0 = cor(SmythBrobyCovariance(; c1 = 0.0, c2 = 0.0, alg = SmythBroby0(),
                                      pdm = nothing), X)
        r1 = cor(SmythBrobyCovariance(; c1 = 0.0, c2 = 0.0, alg = SmythBroby1(),
                                      pdm = nothing), X)
        @test r0 == r1
    end

    @testset "the constructor guards, and the severity exponent (#496, ADR 0091)" begin
        for kw in
            ((; c1 = -1e-9), (; c2 = -1e-9), (; c3 = -1e-9), (; c1 = Inf), (; c2 = NaN),
             (; c3 = Inf))
            @test_throws DomainError SmythBrobyCovariance(; kw...)
        end
        # `c2 < c3` is strict, so equality throws too.
        @test_throws DomainError SmythBrobyCovariance(; c2 = 0.5, c3 = 0.5)
        @test_throws DomainError SmythBrobyCovariance(; c2 = 2.0, c3 = 1.0)
        @test_throws DomainError SmythBrobyCovariance(; c2 = 0.0, c3 = 0.0)

        #=
        `n` IS NON-NEGATIVE. A negative `n` inverted the severity penalty: `gamma^n` is
        `Inf` at the zero divergence of an asset against itself, so every diagonal
        contribution was zero and the result was not a correlation matrix. `assert_nonneg`
        rejects a negative `n` and `NaN`, and admits `Inf`, whose limit is a hard
        divergence gate rather than an empty statistic.
        =#
        X = randn(StableRNG(11), 60, 4)
        for n in (-2, -1e-9, NaN)
            @test_throws DomainError SmythBrobyCovariance(; n = n)
        end
        for n in (0, 0.5, 2, Inf)
            @test SmythBrobyCovariance(; n = n) isa SmythBrobyCovariance
        end
        # The kernel itself takes any `n`, so the inversion the guard rejects is still
        # visible one level down. This is the behaviour the constructor now keeps out.
        @test PO.sb_delta(1.5, 1.5, -2) == 0.0
        @test PO.sb_delta(1.5, 0.5, -2) > 0.9
        # At `n = Inf` an observation whose two magnitudes differ by more than one gives
        # zero, and one whose magnitudes differ by less than one gives `kappa`, as the
        # `## Validation` of `SmythBrobyCovariance` states.
        @test PO.sb_delta(2.5, 1.0, Inf) == 0.0
        @test PO.sb_delta(1.5, 1.0, Inf) == sqrt(2.5 * 2.0)
        # Every `n` the guard admits gives a unit diagonal and a finite matrix.
        for n in (0, 0.5, 2, Inf)
            r = cor(SmythBrobyCovariance(; n = n, alg = SmythBroby1(), pdm = nothing), X)
            @test all(isapprox.(diag(r), 1))
            @test all(isfinite, r)
        end
        # `n = 2` is the default, and the guard moves no shipped number.
        @test isapprox(cor(SmythBrobyCovariance(; alg = SmythBroby1(), pdm = nothing), X),
                       cor(SmythBrobyCovariance(; n = 2, alg = SmythBroby1(),
                                                pdm = nothing), X))
    end

    @testset "cov is cor rescaled by sd, and the diagonals agree" begin
        X = randn(StableRNG(7), 70, 5) .* [0.4 1.2 2.5 0.9 1.1]
        for a in (SmythBroby0(), SmythBroby1(), SmythBroby2(), SmythBrobyGerber1(),
                  SmythBrobyGerber2(), SmythBrobyCount1(), SmythBrobyCount2())
            ce = SmythBrobyCovariance(; alg = a)
            rho = cor(ce, X)
            sigma = cov(ce, X)
            sd = vec(std(ce.ve, X; dims = 1))
            sd .= max.(sd, eps(eltype(sd)))
            @test isapprox(sigma, rho .* (sd * sd'))
            @test isapprox(diag(sigma), sd .^ 2)
            @test isapprox(diag(rho), ones(5))
            @test isapprox(StatsBase.cov2cor!(copy(sigma), sd), rho)
        end

        # A non-square input returns an `N x N` matrix, so no transposed index hides.
        Y = randn(StableRNG(5), 33, 3)
        @test size(cor(SmythBrobyCovariance(), Y)) == (3, 3)
        @test size(cov(SmythBrobyCovariance(), Y)) == (3, 3)
        # `dims = 2` reads the transpose and gives the same answer.
        @test isapprox(cor(SmythBrobyCovariance(), Y),
                       cor(SmythBrobyCovariance(), Y'; dims = 2))
        @test isapprox(cov(SmythBrobyCovariance(), Y),
                       cov(SmythBrobyCovariance(), Y'; dims = 2))
    end

    @testset "a raw Smyth-Broby matrix needs pdm" begin
        # The estimator carries `pdm` because the statistic is a matrix of pairwise scores
        # and is not positive definite in general. `pdm = nothing` returns the raw one.
        X = randn(StableRNG(2024), 40, 12)
        raw = cor(SmythBrobyCovariance(; alg = SmythBroby0(), pdm = nothing), X)
        @test minimum(eigvals(Symmetric(raw))) < 0
        fixed = cor(SmythBrobyCovariance(; alg = SmythBroby0()), X)
        @test minimum(eigvals(Symmetric(fixed))) >= -sqrt(eps(Float64))
        @test !isapprox(raw, fixed)
    end

    @testset "the neutral term does not keep the matrix positive semidefinite" begin
        #=
        Section 2 of the source states, with no proof (its footnote 2), that the neutral
        sum keeps the covariance matrix positive semidefinite. The docstrings of
        `SmythBroby1` and `GerberComovementOne` used to repeat the claim as a fact. On 30
        standard normal observations of 20 assets the two contribution markers of the `1`
        digit are indefinite, while `Gerber1` and `SmythBrobyCount1` are not.
        =#
        X = randn(StableRNG(21), 30, 20)
        lmin(ce) = minimum(eigvals(Symmetric(cor(ce, X))))
        @test lmin(SmythBrobyCovariance(; alg = SmythBroby1(), pdm = nothing)) < -0.05
        @test lmin(SmythBrobyCovariance(; alg = SmythBrobyGerber1(), pdm = nothing)) < -0.3
        @test lmin(SmythBrobyCovariance(; alg = SmythBrobyCount1(), pdm = nothing)) > 0
        @test lmin(GerberCovariance(; alg = Gerber1(), pdm = nothing)) > 0
        # `pdm` repairs both.
        for a in (SmythBroby1(), SmythBrobyGerber1())
            @test lmin(SmythBrobyCovariance(; alg = a)) >= -sqrt(eps(Float64))
        end
    end

    @testset "posdef! keeps a zero diagonal entry, so the unit diagonal is written first" begin
        # `comovement_unit_diagonal!` runs before `posdef!` because the repair keeps a zero
        # variance and its zero row and column. It never writes the one itself.
        A = [0.0 0.0 0.0; 0.0 1.0 0.5; 0.0 0.5 1.0]
        PO.posdef!(Posdef(), A)
        @test A[1, 1] == 0.0
        @test all(iszero, A[1, :])
        @test all(isfinite, A)
    end
end

#=
Check `src/05_Moments/05_Gerber/03_GerberIQCovariance_a.jl` and `_b.jl` against the mathematics its docstrings
state. Issue #457 of child map 3 (#417) under the map of maps (#404).

Coverage was one line short of terminal when this file was written: the `else zro`
fall-through of `gerber_iq_weight(::PartialGerberIQ)`, which only an exactly zero return
reaches. `The weight of an exactly zero return` below reaches it. Every other testset pins a
CLAIM, computed a second way. #404's condition 2 is "checked with real numbers. Not read --
run".

FIVE FACTS SHAPE THE PROBES.

1. THE THREE TEMPLATES FORM A LADDER, and each rung is a strict generalisation of the one
   below. `BasicGerberIQ` reads the MAGNITUDE alone: one boundary `d` and three weights
   `(1, n, n^2)`. `PartialGerberIQ` adds the SIGN: four boundaries, so a concordant pair is
   judged against `dcp`/`dcn` and a discordant one against `ddp`/`ddn`, giving ten weights.
   `FullGerberIQ` adds a SECOND boundary per sign, so each axis carries a small, a moderate
   and a large class, giving twenty-one weights. `The three templates form a ladder` shows
   each reduction on the whole plane, to exactly zero.

2. THE IQ STATISTIC REDUCES TO THE CLASSIC GERBER STATISTIC. Set `n = 1` so every region
   keeps its full weight, `d = c` so the template has no body, a flat decay, and the
   per-asset scaler that `GerberCovariance` itself uses. `Gerber0` and `Gerber1` then agree
   with `GerberCovariance` bit-for-bit, and `Gerber2` agrees to one unit in the last place.
   A degenerate case that reproduces the parent is the strongest single check this file
   admits. The ulp is the associativity of the `Gerber2` denominator: this family divides by
   `sqrt(di * dj)` and the classic one by `sqrt(di) * sqrt(dj)`. The first form is what makes
   the diagonal EXACTLY one, which fact 5 needs.

3. THE SCALER FIXES THE UNITS A CO-MOVEMENT IS JUDGED IN, and the three routes do not agree.
   `AssetVolatilityGerberIQScaler` returns `(sd_i, sd_j)`, so each asset carries its OWN
   volatility. `nothing` returns the pair MEAN on both axes. A `Function` returns whatever it
   likes. Only the first is PAIR-SEPARABLE -- asset `i`'s threshold does not move when its
   partner changes -- and fact 5 turns on that.

4. THE DECAY IS CONFIGURATION AND IS NEVER WRITTEN INTO. `regenerate_decay` returns the SAME
   object when both parameters are already numbers, and a NEW one otherwise. Its argument
   still reads `nothing` afterwards.

5. THE `Gerber2` BOUND IS A GEOMETRIC-MEAN CONDITION ON THE TEMPLATE, and it is the WHOLE
   condition. Under `Gerber2` the pair entry is the net `pos - neg`, divided by the geometric
   mean of the pair's two DIAGONAL PROJECTIONS: asset `i` measured against itself in the
   units of the pair `(i, j)`. A projection is always concordant, so exactly SIX of the
   twenty-one weights sit on the diagonal -- `n11`, `n4`, `n1`, `n2`, `n5`, `n12`, one per
   magnitude class. The other FIFTEEN each join two distinct classes, one per unordered pair,
   discordant channels included. The source's internet appendix proves that
   `n_ab^2 <= n_aa n_bb` over every such channel is NECESSARY AND SUFFICIENT for
   `|rho| <= 1`. `clamp_gerber_iq_n` enforces all fifteen, every shipped default meets them
   by construction, and the projection makes the bound hold for EVERY scaler. #494 and #500
   are the two defects this replaced, and ADR 0094 records the decision.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs

const PO = PortfolioOptimisers

# The weight of one co-movement, on unit scaling, so a threshold is read in raw units.
giqw(ri, rj, kind) = PO.gerber_iq_weight(ri, rj, abs(ri), abs(rj), 1.0, 1.0, kind)

#=
The six magnitude classes of `FullGerberIQ`, and a representative return in each. A pair
`(x, x)` is concordant whatever `x` is, so `giqw(x, x, kind)` names the class's DIAGONAL
weight, and `giqw(x, y, kind)` for two classes names the weight that joins them.
=#
const FULL_CLASSES = (LP = 3.5, MP = 2.5, SP = 1.5, SN = -1.5, MN = -2.5, LN = -3.5)

# `(class a, class b, the field that joins them)`, for the fifteen mixed weights.
const FULL_MIXED = [(:n11, :n4, :n14), (:n11, :n1, :n15), (:n11, :n2, :n18),
                    (:n11, :n5, :n19), (:n11, :n12, :n13), (:n4, :n1, :n7), (:n4, :n2, :n9),
                    (:n4, :n5, :n6), (:n4, :n12, :n20), (:n1, :n2, :n3), (:n1, :n5, :n10),
                    (:n1, :n12, :n21), (:n2, :n5, :n8), (:n2, :n12, :n16),
                    (:n5, :n12, :n17)]

# The template with every one of the fifteen mixed weights lowered onto its flanking bound.
function clamp_every_mixed(k::FullGerberIQ)
    v = Dict(f => getfield(k, f) for f in fieldnames(FullGerberIQ))
    for (a, b, m) in FULL_MIXED
        v[m] = min(v[m], sqrt(v[a] * v[b]))
    end
    return FullGerberIQ(; (f => v[f] for f in fieldnames(FullGerberIQ))...)
end

# The message of the error `f` raises, or `nothing` when it raises none.
function raised_message(f)
    try
        f()
        return nothing
    catch err
        return sprint(showerror, err)
    end
end

# A decay that implements no `regenerate_decay` of its own. It reaches the fallback that the
# `GerberIQDecayEstimator` interface documents for an extension author.
struct UnresolvedDecay <: PO.GerberIQDecayEstimator end

@testset "GerberIQCovariance" begin
    rng = StableRNG(987654321)
    X = randn(rng, 60, 5) .* 0.02
    sd = vec(std(SimpleVariance(), X; dims = 1))

    @testset "The three templates form a ladder" begin
        #=
        Each shipped default carries a unit diagonal. `BasicGerberIQ` is a different
        statistic from the other two, and THE OTHER TWO AGREE. `FullGerberIQ()` sets
        `dp1 == dp2` and `dn1 == dn2`, so its moderate bands are empty and its six classes
        collapse to the four `PartialGerberIQ` carries, and `PartialGerberIQ()` sets
        `ddp == dcp` and `ddn == dcn`, so its discordant boundaries collapse onto its
        concordant ones. Every default weight is now the geometric mean of the two diagonal
        weights of its channel in BOTH templates, so the two collapse onto the same template.
        Before ADR 0094 an inconsistent `n15` was the only thing that separated them.
        =#
        rhos = map(k -> cor(GerberIQCovariance(; kind = k, pdm = nothing), X),
                   (BasicGerberIQ(), PartialGerberIQ(), FullGerberIQ()))
        for r in rhos
            @test isapprox(diag(r), ones(size(X, 2)))
        end
        @test !isapprox(rhos[1], rhos[2])
        @test !isapprox(rhos[1], rhos[3])
        @test rhos[2] == rhos[3]

        # `PartialGerberIQ` reduces to `BasicGerberIQ` when its four boundaries agree and its
        # ten weights collapse onto the magnitude triple. `FullGerberIQ` reduces to the same
        # template when both moderate bands are empty. The grid skips an exactly zero return,
        # which `BasicGerberIQ` alone gives a weight to; `The weight of an exactly zero
        # return` covers that seam.
        n, d = 0.4, 2.5
        basic = BasicGerberIQ(; d = d, n = n)
        partial = PartialGerberIQ(; dcp = d, dcn = d, ddp = d, ddn = d, n1 = n, n2 = n,
                                  n3 = n, n4 = 1.0, n5 = 1.0, n6 = 1.0, n7 = n^2, n8 = n^2,
                                  n9 = n^2, n10 = n^2)
        full = FullGerberIQ(; dp1 = d, dp2 = d, dn1 = d, dn2 = d, n1 = n, n2 = n, n3 = n,
                            n4 = 1.0, n5 = 1.0, n6 = 1.0, n7 = n^2, n8 = n^2, n9 = n^2,
                            n10 = n^2, n11 = 1.0, n12 = 1.0, n13 = 1.0, n14 = 1.0,
                            n15 = n^2, n16 = n^2, n17 = 1.0, n18 = n^2, n19 = 1.0,
                            n20 = 1.0, n21 = n^2)
        grid = range(-4, 4; length = 81)
        for x in grid, y in grid
            if iszero(x) || iszero(y)
                continue
            end
            @test giqw(x, y, partial) == giqw(x, y, basic)
            @test giqw(x, y, full) == giqw(x, y, basic)
        end
    end

    @testset "The IQ statistic reduces to the classic Gerber statistic" begin
        # `n = 1` gives every region weight one, `d = c` empties the body of the template, a
        # zero decay rate and delay give every observation weight one, and
        # `AssetVolatilityGerberIQScaler` scales each threshold by that asset's own
        # volatility. That is the classic statistic, on all three markers.
        for c in (0.5, 1.5), alg in (Gerber0(), Gerber1(), Gerber2())
            ceq = GerberIQCovariance(; c = c, kind = BasicGerberIQ(; d = c, n = 1.0),
                                     sc = AssetVolatilityGerberIQScaler(),
                                     decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                     alg = alg, pdm = nothing)
            cep = GerberCovariance(; t = c, alg = alg, pdm = nothing)
            #=
            `Gerber0` and `Gerber1` divide by a sum this family builds the same way, so they
            agree bit-for-bit. `Gerber2` divides by `sqrt(di * dj)` where the classic family
            divides by `sqrt(di) * sqrt(dj)`, and the two group the same three operations
            differently. The gap is one unit in the last place, and it buys the exactly
            unit diagonal that `posdef!` reads with `isone`.
            =#
            if alg isa Gerber2
                @test maximum(abs, cor(ceq, X) - cor(cep, X)) <= eps(1.0)
            else
                @test cor(ceq, X) == cor(cep, X)
            end
        end
    end

    @testset "The threshold scaler" begin
        # The estimator hands each asset its own volatility, and the equivalent function
        # agrees with it exactly.
        @test PO.gerber_iq_scaling(AssetVolatilityGerberIQScaler(), sd[1], sd[2]) ==
              (sd[1], sd[2])
        @test PO.gerber_iq_scaling((x, y) -> (x, y), sd[1], sd[2]) == (sd[1], sd[2])
        @test cor(GerberIQCovariance(; sc = AssetVolatilityGerberIQScaler(), pdm = nothing),
                  X) == cor(GerberIQCovariance(; sc = (x, y) -> (x, y), pdm = nothing), X)

        # `nothing` is the pair mean on both axes, so it is a different statistic.
        m = (sd[1] + sd[2]) / 2
        @test PO.gerber_iq_scaling(nothing, sd[1], sd[2]) == (m, m)
        @test !isapprox(cor(GerberIQCovariance(; sc = AssetVolatilityGerberIQScaler(),
                                               pdm = nothing), X),
                        cor(GerberIQCovariance(; sc = nothing, pdm = nothing), X))
    end

    @testset "The temporal decay" begin
        T = size(X, 1)
        decay = ExpGerberIQDecay(; e = 3.0, y = 0.1)
        @test all(decay(T, k) == exp(-0.1 * max(0, T - k - 3.0)) for k in 1:T)

        # A fully numeric decay resolves to itself, and the resolver never writes into its
        # argument.
        @test PO.regenerate_decay(decay, X) === decay
        unset = ExpGerberIQDecay()
        resolved = PO.regenerate_decay(unset, X)
        @test isnothing(unset.e) && isnothing(unset.y)
        @test resolved.e == round(Int, T - T / size(X, 2))
        @test resolved.y == log(2) / size(X, 2)

        # A window of one observation divides by nothing: both defaults read the number of
        # ASSETS, not the number of observations.
        X1 = reshape([0.01, -0.02, 0.015], 1, 3)
        d1 = PO.regenerate_decay(ExpGerberIQDecay(), X1)
        @test isfinite(d1.e) && isfinite(d1.y)
        @test d1(1, 1) == 1
        @test isapprox(cor(GerberIQCovariance(; pdm = nothing), X1), I(3))

        # A single asset is the other degenerate window, and it divides by nothing either.
        d1c = PO.regenerate_decay(ExpGerberIQDecay(), reshape([0.01], 1, 1))
        @test isfinite(d1c.e) && isfinite(d1c.y)

        # A subtype that implements no `regenerate_decay` reaches the fallback of the
        # abstract type, which reads both parameters off `X` with the source's own defaults.
        dfb = PO.regenerate_decay(UnresolvedDecay(), X)
        @test dfb isa ExpGerberIQDecay
        @test dfb.e == PO.gerber_iq_eps(nothing, X)
        @test dfb.y == PO.gerber_iq_gamma(nothing, X)
    end

    @testset "gerber_iq_eps and gerber_iq_gamma" begin
        T, N = size(X)
        # A number passes through, a function is called on `X`, and `nothing` takes the
        # source's own default. The three routes reach the same number when they are asked
        # for it.
        @test PO.gerber_iq_eps(3.0, X) == 3.0
        @test PO.gerber_iq_eps(x -> round(Int, size(x, 1) - size(x, 1) / size(x, 2)), X) ==
              PO.gerber_iq_eps(nothing, X) ==
              round(Int, T - T / N)
        @test PO.gerber_iq_gamma(0.1, X) == 0.1
        @test PO.gerber_iq_gamma(x -> log(2) / size(x, 2), X) ==
              PO.gerber_iq_gamma(nothing, X) ==
              log(2) / N
    end

    @testset "The clamp guard" begin
        # A template already on the bound is left alone, and one past it is brought back to
        # the bound exactly.
        bound = sqrt(0.2 * 0.8)
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = bound),
                                   Gerber2()).n7 == bound
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = 1.0),
                                   Gerber2()).n7 == bound
        @test PO.clamp_gerber_iq_n(FullGerberIQ(; n4 = 0.2, n11 = 0.8, n14 = 1.0),
                                   Gerber2()).n14 == bound

        # The other two markers divide by a sum of the same weights, so they cannot leave
        # `[-1, 1]` and the clamp is a no-op for them.
        for alg in (Gerber0(), Gerber1())
            @test PO.clamp_gerber_iq_n(PartialGerberIQ(; n1 = 0.2, n4 = 0.8, n7 = 1.0),
                                       alg).n7 == 1.0
        end
    end

    @testset "The unit-interval guard over the weights" begin
        # Every weight of the three templates is a compression weight in `[0, 1]`, and one
        # guard states that for all thirty-two of them. Both ends belong to the interval, so
        # a weight that switches a region off and one that keeps its full value both pass.
        @test BasicGerberIQ(; n = 0.0).n == 0.0
        @test BasicGerberIQ(; n = 1.0).n == 1.0
        # Past either end the raise carries the value in `.val` and names the field the
        # caller wrote in `.msg`.
        @test raised_message(() -> BasicGerberIQ(; n = -0.1)) ==
              "DomainError with -0.1:\n0 <= n <= 1 must hold. Got\nn => -0.1"
        # The guard reaches every weight of each template. Each derived default is built from
        # weights of a lower index, so a weight set past the upper end raises on its own row.
        for i in 1:10
            @test raised_message(() -> PartialGerberIQ(; Dict(Symbol("n$i") => 2.0)...)) ==
                  "DomainError with 2.0:\n0 <= n$i <= 1 must hold. Got\nn$i => 2.0"
        end
        for i in 1:21
            @test raised_message(() -> FullGerberIQ(; Dict(Symbol("n$i") => 2.0)...)) ==
                  "DomainError with 2.0:\n0 <= n$i <= 1 must hold. Got\nn$i => 2.0"
        end
    end

    @testset "The c against d guard" begin
        # `c` inside the innermost boundary passes.
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, BasicGerberIQ(; d = 0.5)))
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, PartialGerberIQ(; dcp = 0.5)))
        @test isnothing(PO.gerber_iq_assert_c_d(0.5, FullGerberIQ(; dp1 = 0.5)))

        # Past it the raise carries `c` in `.val`, and names the boundary that failed and
        # both values in `.msg`.
        @test raised_message(() -> PO.gerber_iq_assert_c_d(0.6, BasicGerberIQ(; d = 0.5))) ==
              "DomainError with 0.6:\n`c` is 0.6, and `kind.d` is 0.5. The centre band is inside the boundary, so `c <= kind.d` must hold."
        # Each boundary of the two multi-boundary templates is checked on its own, so raise
        # each one in turn by keeping the others clear of `c`.
        for (f, kind) in
            (("dcp", PartialGerberIQ(; dcp = 1.0, dcn = 3.0, ddp = 3.0, ddn = 3.0)),
             ("dcn", PartialGerberIQ(; dcp = 3.0, dcn = 1.0, ddp = 3.0, ddn = 3.0)),
             ("ddp", PartialGerberIQ(; dcp = 3.0, dcn = 3.0, ddp = 1.0, ddn = 3.0)),
             ("ddn", PartialGerberIQ(; dcp = 3.0, dcn = 3.0, ddp = 3.0, ddn = 1.0)))
            @test raised_message(() -> PO.gerber_iq_assert_c_d(2.0, kind)) ==
                  "DomainError with 2.0:\n`c` is 2.0, and `kind.$f` is 1.0. The centre band is inside the boundary, so `c <= kind.$f` must hold."
        end
        # `FullGerberIQ` sorts each pair of boundaries, so `dp2 <= dp1` and `dn2 <= dn1`
        # always hold and only the inner one of each pair can raise.
        for (f, kind) in
            (("dp2", FullGerberIQ(; dp1 = 3.0, dp2 = 1.0, dn1 = 3.0, dn2 = 3.0)),
             ("dn2", FullGerberIQ(; dp1 = 3.0, dp2 = 3.0, dn1 = 3.0, dn2 = 1.0)))
            @test raised_message(() -> PO.gerber_iq_assert_c_d(2.0, kind)) ==
                  "DomainError with 2.0:\n`c` is 2.0, and `kind.$f` is 1.0. The centre band is inside the boundary, so `c <= kind.$f` must hold."
        end
        # The estimator runs the guard, so a wrong pair never reaches the kernel.
        @test_throws DomainError GerberIQCovariance(; c = 3.0,
                                                    kind = BasicGerberIQ(; d = 2.0))
    end

    @testset "gerber_IQ_delta" begin
        # The delta is the region weight times the decay weight, and nothing else.
        decay = ExpGerberIQDecay(; e = 1.0, y = 0.25)
        kind = BasicGerberIQ(; d = 2.0, n = 0.5)
        # One return large, one small, so the template selects `n^2`.
        @test PO.gerber_IQ_delta(3.0, -1.0, 3.0, 1.0, decay, 10, 4, 1.0, 1.0, kind) ==
              0.5^2 * exp(-0.25 * max(0, 10 - 4 - 1.0))
        # Both large, so the template selects one and the delta is the decay weight alone.
        @test PO.gerber_IQ_delta(3.0, -3.0, 3.0, 3.0, decay, 10, 4, 1.0, 1.0, kind) ==
              exp(-0.25 * max(0, 10 - 4 - 1.0))
    end

    @testset "The weight of an exactly zero return" begin
        # `PartialGerberIQ` and `FullGerberIQ` read the SIGN of a return, and zero has none,
        # so a zero return falls through every channel and carries weight zero. This is the
        # `else zro` arm of both methods.
        for kind in (PartialGerberIQ(), FullGerberIQ())
            for other in (-3.0, -1.0, 0.0, 1.0, 3.0)
                @test iszero(giqw(0.0, other, kind))
                @test iszero(giqw(other, 0.0, kind))
            end
        end
        # `BasicGerberIQ` reads the magnitude alone, so a zero return is small and carries a
        # weight. The two families differ here, and the difference is deliberate.
        @test giqw(0.0, 0.0, BasicGerberIQ(; d = 2.0, n = 0.5)) == 0.5
    end

    @testset "cor and cov" begin
        ce = GerberIQCovariance(; pdm = nothing)
        rho = cor(ce, X)
        sigma = cov(ce, X)
        @test isapprox(diag(rho), ones(size(X, 2)))
        # The covariance is the correlation rescaled by the same volatilities that scaled the
        # thresholds, so its diagonal is exactly the variance.
        @test sigma == rho .* (sd * transpose(sd))
        @test isapprox(diag(sigma), sd .^ 2)
        # `dims = 2` reads the transpose and answers the same matrix.
        @test cor(ce, transpose(X); dims = 2) == rho
        @test_throws DomainError cor(ce, X; dims = 3)
    end

    @testset "The Gerber2 flanking map" begin
        # Sentinel weights `n_k = k / 100` name every channel. Six of the twenty-one weights
        # sit on the diagonal, one per magnitude class, and the other fifteen each join two
        # distinct classes -- one per unordered pair, discordant channels included.
        full = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0,
                            Dict(Symbol("n$k") => k / 100 for k in 1:21)...)
        diagonal = (LP = :n11, MP = :n4, SP = :n1, SN = :n2, MN = :n5, LN = :n12)
        for (cls, f) in pairs(diagonal)
            x = FULL_CLASSES[cls]
            @test giqw(x, x, full) == getfield(full, f)
        end
        seen = Symbol[]
        for (a, b, m) in FULL_MIXED
            ca = findfirst(==(a), diagonal)
            cb = findfirst(==(b), diagonal)
            xa, xb = FULL_CLASSES[ca], FULL_CLASSES[cb]
            @test giqw(xa, xb, full) == getfield(full, m)
            @test giqw(xb, xa, full) == getfield(full, m)
            push!(seen, m)
        end
        # The six diagonal weights and the fifteen mixed ones are the whole template.
        @test sort(vcat(collect(values(diagonal)), seen)) ==
              sort([Symbol("n$k") for k in 1:21])
    end

    @testset "The Gerber2 bound holds (#494, #500)" begin
        #=
        The source's internet appendix proves `n_ab^2 <= n_aa n_bb` NECESSARY AND SUFFICIENT
        for `|rho| <= 1` under a geometric-mean denominator, over every channel that joins
        two distinct magnitude classes. `clamp_gerber_iq_n` enforces all fifteen of them for
        `FullGerberIQ` and all six for `PartialGerberIQ`, and every shipped default is
        written as the geometric mean of the two diagonal weights of the classes its channel
        joins. So no default owes the clamp anything, and the clamp returns the template it
        was handed.
        =#
        fd = FullGerberIQ()
        dg = (n11 = fd.n11, n4 = fd.n4, n1 = fd.n1, n2 = fd.n2, n5 = fd.n5, n12 = fd.n12)
        breaks = [m
                  for (a, b, m) in FULL_MIXED
                  if getfield(fd, m) > sqrt(dg[a] * dg[b]) * (1 + 1e-12)]
        @test isempty(breaks)
        @test PO.clamp_gerber_iq_n(fd, Gerber2()) == fd
        @test PO.clamp_gerber_iq_n(PartialGerberIQ(), Gerber2()) == PartialGerberIQ()

        # Two assets, one always large positive and one always small positive, then both
        # negative. Every co-movement is concordant, so the pair entry is `n15 + n16` and the
        # two projections are built from `n11`, `n12` and `n1`, `n2`. Under the earlier
        # defaults this sample returned `1.0299`, which is #494.
        Xb = Float64[4.0 1.5; 4.0 1.5; -4.0 -1.5; -4.0 -1.5]
        kind = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0)
        mk(k) = GerberIQCovariance(; c = 1.0, kind = k, alg = Gerber2(),
                                   sc = (x, y) -> (1.0, 1.0),
                                   decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                   pdm = nothing)
        hand = 2 * (kind.n15 + kind.n16) /
               sqrt((2 * kind.n11 + 2 * kind.n12) * (2 * kind.n1 + 2 * kind.n2))
        @test cor(mk(kind), Xb)[1, 2] == hand
        @test hand == 1
        # The clamp is what holds it there. A template that breaks the bound reaches the
        # estimator through the clamp, and comes out on the bound.
        kraw = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0, n15 = 1.0,
                            n16 = 1.0)
        @test mk(kraw).kind == clamp_every_mixed(kraw)
        @test cor(mk(kraw), Xb)[1, 2] == 1

        # #494's own reproduction, on the discordant channel `n13`. Its flanking classes are
        # large positive and large negative, and BOTH sit on the diagonal, so the
        # geometric-mean rule reads for a discordant channel exactly as it does for a
        # concordant one. It returned `-10` before the clamp reached `n13`.
        Xn = Float64[4.0 -4.0; 4.0 -4.0; -4.0 4.0; -4.0 4.0]
        kbad = FullGerberIQ(; dp1 = 3.0, dp2 = 2.0, dn1 = 3.0, dn2 = 2.0, n11 = 0.1,
                            n12 = 0.1, n13 = 1.0)
        @test mk(kbad).kind.n13 == sqrt(kbad.n11 * kbad.n12)
        @test cor(mk(kbad), Xn)[1, 2] == -1

        #=
        A PAIR-DEPENDENT scaler cannot break the bound, because the denominator is read in
        the pair's own units. `min` hands the pair the SMALLER of the two volatilities, so
        the volatile asset clears a threshold off the diagonal that it never clears against
        itself alone. The projection follows it there, so the class it takes in the numerator
        is the class it takes in the denominator. This sample returned `sqrt(2)` while the
        denominator was read from the assembled diagonal, which is #500.
        =#
        Xm = Float64[10.0 1.0; 10.0 1.0; -10.0 -1.0; -10.0 -1.0]
        mks(sc) = GerberIQCovariance(; kind = PartialGerberIQ(), alg = Gerber2(), sc = sc,
                                     decay = ExpGerberIQDecay(; e = 0.0, y = 0.0),
                                     pdm = nothing)
        sep = cor(mks(AssetVolatilityGerberIQScaler()), Xm)
        dep = cor(mks((x, y) -> (min(x, y), min(x, y))), Xm)
        @test sep == dep
        @test dep[1, 2] == 1

        #=
        The diagonal is EXACTLY one, and not one to within a rounding. The pair `(i, i)`
        makes the numerator and both projections the same sum, so the ratio is `x / sqrt(x
        * x)`. `posdef!` reads the diagonal with an exact `isone` test to choose between a
        correlation matrix and a covariance matrix, so an exact one decides that branch by
        construction. #500's second finding is what this closes.
        =#
        for sc in
            (nothing, AssetVolatilityGerberIQScaler(), (x, y) -> (min(x, y), min(x, y)))
            r = cor(GerberIQCovariance(; kind = FullGerberIQ(), alg = Gerber2(), sc = sc,
                                       pdm = nothing), X)
            @test all(isone, diag(r))
            @test maximum(abs, r) <= 1
        end

        #=
        The two conditions the bound needs are the template's, and nothing else. Random
        diagonal weights, random boundaries, random data with a twenty-fold volatility
        spread, and the three scalers in turn: the clamped template never leaves `[-1, 1]`.
        =#
        rngb = StableRNG(24680)
        worst = 0.0
        for trial in 1:60
            Xr = randn(rngb, 60, 4) .* transpose(1 .+ 19 .* rand(rngb, 4))
            d2 = 0.3 + 1.2 * rand(rngb)
            k = FullGerberIQ(; dp1 = d2 + 1.5 * rand(rngb), dp2 = d2,
                             dn1 = d2 + 1.5 * rand(rngb), dn2 = d2,
                             Dict(Symbol("n$i") => rand(rngb) for i in 1:21)...)
            sc = if trial % 3 == 0
                nothing
            else
                (if trial % 3 == 1
                     (x, y) -> (min(x, y), min(x, y))
                 else
                     (x, y) -> (max(x, y), max(x, y))
                 end)
            end
            ce = GerberIQCovariance(; c = d2 * rand(rngb), kind = k, alg = Gerber2(),
                                    sc = sc,
                                    decay = ExpGerberIQDecay(; e = 0.0,
                                                             y = 0.05 * rand(rngb)),
                                    pdm = nothing)
            worst = max(worst, maximum(abs, cor(ce, Xr)))
        end
        @test worst <= 1
    end

    @testset "A one-sided co-movement reaches Gerber1 and Gerber2 (#456)" begin
        # The diagram bullets of the three templates once said that only `Gerber1` reads a
        # co-movement with one return inside the noise zone. `Gerber2` reads it too: the
        # asset that crosses adds its diagonal projection to the denominator. `Gerber0`
        # ignores it. The last row of `X1` moves asset 1 alone.
        X0 = [1.0 1.0; -1.0 -1.0; 1.0 -1.0; 1.0 1.0; -1.0 -1.0]
        X1 = vcat(X0, [3.0 0.0])
        sd1 = ones(1, 2)
        no_decay = ExpGerberIQDecay(; e = 100, y = 0.0)
        function rho(a, Xs)
            ce1 = GerberIQCovariance(; alg = a, pdm = nothing, decay = no_decay,
                                     sc = AssetVolatilityGerberIQScaler(), c = 0.1)
            return PortfolioOptimisers.gerber_IQ(ce1, Xs, sd1)[1, 2]
        end
        @test rho(Gerber0(), X1) == rho(Gerber0(), X0)
        @test rho(Gerber1(), X1) < rho(Gerber1(), X0)
        @test rho(Gerber2(), X1) < rho(Gerber2(), X0)

        # The `ex` field docstring follows `field_dict[:ex]`, which already ends in a
        # period, and once rendered as "strategy..".
        iq_doc = Docs.meta(PortfolioOptimisers)[Docs.Binding(PortfolioOptimisers,
                                                             :GerberIQCovariance)]
        iq_ex = strip(first(values(iq_doc.docs)).data[:fields][:ex])
        @test endswith(iq_ex, "strategy.")
        @test !endswith(iq_ex, "..")
    end
end
