struct EquilibriumExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                  T2 <: Union{Nothing, <:AbstractVector}, T3 <: Real} <:
       AbstractShrunkExpectedReturnsEstimator
    ce::T1
    w::T2
    l::T3
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    w::Union{Nothing, <:AbstractVector} = nothing,
                                    l::Real = 1.0)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return EquilibriumExpectedReturns{typeof(ce), typeof(w), typeof(l)}(ce, w, l)
end
function Statistics.mean(me::EquilibriumExpectedReturns, X::AbstractArray; dims::Int = 1,
                         kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end
function factory(ce::EquilibriumExpectedReturns, args...)
    return ce
end

export EquilibriumExpectedReturns
