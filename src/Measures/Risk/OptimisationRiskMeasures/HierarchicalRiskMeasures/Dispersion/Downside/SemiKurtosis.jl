struct SemiKurtosis{T1 <: RiskMeasureSettings, T2 <: PortfolioOptimisersVarianceEstimator,
                    T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                    T4 <: Union{Nothing, <:AbstractWeights},
                    T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetHierarchicalRiskMeasure
    settings::T1
    ve::T2
    target::T3
    w::T4
    mu::T5
end
function SemiKurtosis(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                      ve::PortfolioOptimisersVarianceEstimator = SimpleVariance(),
                      target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      w::Union{Nothing, <:AbstractWeights} = nothing,
                      mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return SemiKurtosis{typeof(settings), typeof(ve), typeof(target), typeof(w),
                        typeof(mu)}(settings, ve, target, w, mu)
end
function (r::SemiKurtosis)(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    val = val[val <= zero(target)]
    sigma = std(r.ve, val; mean = zero(target))
    return sum(val[val <= zero(target)] .^ 4) / length(x) / sigma^4
end

export SemiKurtosis
