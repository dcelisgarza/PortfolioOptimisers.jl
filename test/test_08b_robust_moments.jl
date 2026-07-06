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
