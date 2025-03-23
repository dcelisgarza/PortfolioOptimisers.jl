struct SemiVariance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                    T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                    T4 <: Union{Nothing, <:AbstractWeights},
                    T5 <: Union{Nothing, <:AbstractVector{<:Real}}} <: TargetRiskMeasure
    settings::T1
    formulation::T2
    target::T3
    w::T4
    mu::T5
end
function SemiVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                      formulation::VarianceFormulation = SOC(),
                      target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      w::Union{Nothing, <:AbstractWeights} = nothing,
                      mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    return SemiVariance(settings, formulation, target, w, mu)
end
function (r::SemiVariance)(X::AbstractMatrix, w::AbstractVector, fees::Fees = Fees())
    x = calc_net_returns(X, w, fees)
    mu = calc_target_ret_mu(x, w, r)
    val = x .- mu
    val = val[val .<= zero(eltype(val))]
    return dot(val, val) / (length(x) - 1)
end

export SemiVariance
