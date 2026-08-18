#=
Regression tests for `robust_cov`/`robust_cor` keyword dispatch: unsupported keyword
arguments are dropped via `hasmethod` dispatch rather than error-swallowing, so genuine
errors thrown by the estimator propagate to the caller.
=#
using Test, PortfolioOptimisers, Statistics, StatsBase, LinearAlgebra, StableRNGs

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
    # Estimators without a `cor` method fall back to cov2cor(robust_cov(...)).
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
