struct FourthLowerPartialMoment{T1 <: RiskMeasureSettings,
                                T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                                T3 <: Union{Nothing, <:AbstractWeights},
                                T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetHierarchicalRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function FourthLowerPartialMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                                  w::Union{Nothing, <:AbstractWeights} = nothing,
                                  mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return FourthLowerPartialMoment{typeof(settings), typeof(target), typeof(w),
                                    typeof(mu)}(settings, target, w, mu)
end
function (r::FourthLowerPartialMoment)(w::AbstractVector, X::AbstractMatrix,
                                       fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val[val .<= zero(eltype(val))] .^ 4) / length(x)
end

export FourthLowerPartialMoment
