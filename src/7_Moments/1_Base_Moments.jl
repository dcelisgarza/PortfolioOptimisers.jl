abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end
abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end
struct Full <: AbstractMomentAlgorithm end
struct Semi <: AbstractMomentAlgorithm end
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
function robust_cov(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        cov(ce, X; dims = dims, mean = mean, kwargs...)
    catch
        cov(ce, X; dims = dims, mean = mean)
    end
end
function robust_cov(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix,
                    w::AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        cov(ce, X, w; dims = dims, mean = mean, kwargs...)
    catch
        cov(ce, X, w; dims = dims, mean = mean)
    end
end
function robust_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        try
            cor(ce, X; dims = dims, mean = mean, kwargs...)
        catch
            cor(ce, X; dims = dims, mean = mean)
        end
    catch
        sigma = robust_cov(ce, X; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.cov2cor!(sigma, sqrt.(diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
end
function robust_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix,
                    w::AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        try
            cor(ce, X, w; dims = dims, mean = mean, kwargs...)
        catch
            cor(ce, X, w; dims = dims, mean = mean)
        end
    catch
        sigma = robust_cov(ce, X, w; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.cov2cor!(sigma, sqrt.(diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
end

export Full, Semi
