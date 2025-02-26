struct GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: Union{Nothing, <:AbstractWeights}} <:
       PearsonCovarianceEstimator
    ce::T1
    w::T2
end
function GeneralWeightedCovariance(;
                                   ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                                  corrected = true),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance{typeof(ce), typeof(w)}(ce, w)
end
function StatsBase.cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing)
    return if isnothing(ce.w)
        cov(ce.ce, X; dims = dims, mean = mean)
    else
        cov(ce.ce, X, ce.w; dims = dims, mean = mean)
    end
end
function StatsBase.cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                       mean = nothing)
    try
        if isnothing(ce.w)
            cor(ce.ce, X; dims = dims, mean = mean)
        else
            cor(ce.ce, X, ce.w; dims = dims, mean = mean)
        end
    catch
        sigma = if isnothing(ce.w)
            cov(ce.ce, X; dims = dims, mean = mean)
        else
            cov(ce.ce, X, ce.w; dims = dims, mean = mean)
        end
        isa(sigma, Matrix) ? StatsBase.cov2cor(sigma) : StatsBase.cov2cor(Matrix(sigma))
    end
end

export GeneralWeightedCovariance
