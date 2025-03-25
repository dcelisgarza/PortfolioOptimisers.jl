struct SemiStandardDeviation{T1 <: RiskMeasureSettings,
                             T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             T3 <: Union{Nothing, <:AbstractWeights},
                             T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function SemiStandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                               target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return SemiStandardDeviation{typeof(settings), typeof(target), typeof(w), typeof(mu)}(settings,
                                                                                          target,
                                                                                          w,
                                                                                          mu)
end
function (r::SemiStandardDeviation)(w::AbstractVector, X::AbstractMatrix,
                                    fees::Fees = Fees())
    x = calc_net_returns(w, X, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return sqrt(dot(val, val) / (length(x) - 1))
end

export SemiStandardDeviation
