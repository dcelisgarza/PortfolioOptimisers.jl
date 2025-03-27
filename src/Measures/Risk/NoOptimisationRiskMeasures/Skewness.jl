struct Skewness{T1 <: PortfolioOptimisersVarianceEstimator,
                T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                T3 <: Union{Nothing, <:AbstractWeights},
                T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetNoOptimisationRiskMeasure
    ve::T1
    target::T2
    w::T3
    mu::T4
end
function Skewness(; ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                  target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                  w::Union{Nothing, <:AbstractWeights} = nothing,
                  mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return Skewness{typeof(ve), typeof(target), typeof(w), typeof(mu)}(ve, target, w, mu)
end
function (r::Skewness)(w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    sigma = std(r.ve, x)
    return sum(val .^ 3) / length(x) / sigma^3
end

export Skewness
