#=
Check and cover `src/08_Moments/01_Base_Moments.jl` and the five `Windowed*` files, issue
#452 of child map 3 (#417) under the map of maps (#404).

The file's own tests already drive the `@windowed_estimator` machinery -- the parsers, the
suggestion scores, the generated docstrings and every expansion-time rejection all live in
`test_08_moments.jl` -- and `test_08c_observation_weights.jl` drives
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
