abstract type PortfolioOptimisersCovarianceEstimator <: StatsBase.CovarianceEstimator end

function robust_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    return try
        cor(ce, X; dims = dims, mean = mean)
    catch
        sigma = cov(ce, X; dims = dims, mean = mean)
        if ismutable(sigma)
            StatsBase.cov2cor!(sigma, sqrt.(diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
end

function robust_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix,
                    w::AbstractWeights; dims::Int = 1, mean = nothing)
    return try
        cor(ce, X, w; dims = dims, mean = mean)
    catch
        sigma = cov(ce, X, w; dims = dims, mean = mean)
        if ismutable(sigma)
            StatsBase.cov2cor!(sigma, sqrt.(diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
end

export robust_cor
