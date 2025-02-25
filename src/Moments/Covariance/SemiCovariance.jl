struct SemiCovariance{T1 <: StatsBase.CovarianceEstimator,
                      T2 <: Union{Nothing, <:AbstractWeights}} <: PearsonCovarianceEstimator
    ce::T1
    w::T2
end
function SemiCovariance(;
                        ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                       corrected = true),
                        w::Union{Nothing, <:AbstractWeights} = nothing)
    return SemiCovariance{typeof(ce), typeof(w)}(ce, w)
end
function StatsBase.cov(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1)
    mu = isnothing(ce.w) ? mean(X; dims = dims) : mean(X, ce.w; dims = dims)
    X = min.(X .- mu, zero(eltype(X)))
    return if isnothing(ce.w)
        cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
    else
        cov(ce.ce, X, ce.w; dims = dims, mean = zero(eltype(X)))
    end
end
function StatsBase.cor(ce::SemiCovariance, X::AbstractMatrix; dims::Int = 1)
    mu = isnothing(ce.w) ? mean(X; dims = dims) : mean(X, ce.w; dims = dims)
    X = min.(X .- mu, zero(eltype(X)))
    try
        if isnothing(ce.w)
            cor(ce.ce, X; dims = dims, mean = zero(eltype(X)))
        else
            cor(ce.ce, X, ce.w; dims = dims, mean = zero(eltype(X)))
        end
    catch
        sigma = if isnothing(ce.w)
            cov(ce.ce, X; dims = dims, mean = zero(eltype(X)))
        else
            cov(ce.ce, X, ce.w; dims = dims, mean = zero(eltype(X)))
        end
        isa(sigma, Matrix) ? StatsBase.cov2cor(sigma) : StatsBase.cov2cor(Matrix(sigma))
    end
end

export SemiCovariance
