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
function moment_factory_w(ce::EquilibriumExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return EquilibriumExpectedReturns(; ce = moment_factory_w(ce.ce, w), l = ce.l, w = ce.w)
end

export EquilibriumExpectedReturns
