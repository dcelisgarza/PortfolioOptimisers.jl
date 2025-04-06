struct EquilibriumExpectedReturns{T1 <: StatsBase.CovarianceEstimator, T2 <: Real,
                                  T3 <: Union{Nothing, <:AbstractVector}} <:
       AbstractShrunkExpectedReturnsEstimator
    ce::T1
    l::T2
    w::T3
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    l::Real = 1.0,
                                    w::Union{Nothing, <:AbstractVector} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return EquilibriumExpectedReturns{typeof(ce), typeof(l), typeof(w)}(ce, l, w)
end
function StatsBase.mean(me::EquilibriumExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    sigma = cov(me.ce, X; dims = dims)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end
function factory(ce::EquilibriumExpectedReturns, args...)
    return ce
end

export EquilibriumExpectedReturns
