struct EquilibriumExpectedReturns{T1 <: StatsBase.CovarianceEstimator, T2 <: Real,
                                  T3 <: Union{Nothing, <:AbstractVector}} <:
       ShrunkExpectedReturnsEstimator
    ce::T1
    l::T2
    w::T3
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    l::Real = 1.0,
                                    w::Union{Nothing, <:AbstractVector} = nothing)
    return EquilibriumExpectedReturns{typeof(ce), typeof(l), typeof(w)}(ce, l, w)
end
function StatsBase.mean(me::EquilibriumExpectedReturns, X::AbstractMatrix; dims::Int = 1,
                        kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end

export EquilibriumExpectedReturns
