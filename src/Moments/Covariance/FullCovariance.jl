struct FullCovariance{T1 <: StatsBase.CovarianceEstimator,
                      T2 <: Union{Nothing, <:AbstractWeights}} <: PearsonCovarianceEstimator
    ce::T1
    w::T2
end
function FullCovariance(;
                        ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                       corrected = true),
                        w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCovariance{typeof(ce), typeof(w)}(ce, w)
end
function StatsBase.cov(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1)
    return isnothing(ce.w) ? cov(ce.ce, X; dims = dims) : cov(ce.ce, X, ce.w; dims = dims)
end
function StatsBase.cor(ce::FullCovariance, X::AbstractMatrix; dims::Int = 1)
    return try
        isnothing(ce.w) ? cor(ce.ce, X; dims = dims) : cor(ce.ce, X, ce.w; dims = dims)
    catch
        sigma = if isnothing(ce.w)
            cov(ce.ce, X; dims = dims)
        else
            cov(ce.ce, X, ce.w; dims = dims)
        end
        isa(sigma, Matrix) ? StatsBase.cov2cor(sigma) : StatsBase.cov2cor(Matrix(sigma))
    end
end

export FullCovariance
