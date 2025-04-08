struct UncertaintySetVariance{T1 <: RiskMeasureSettings,
                              T2 <: Union{Nothing, <:AbstractUncertaintySet,
                                          <:AbstractUncertaintySetEstimator},
                              T3 <: Union{Nothing, <:AbstractMatrix{<:Real}}} <:
       SigmaRiskMeasure
    settings::T1
    ucs::T2
    sigma::T3
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                ucs::Union{Nothing, <:AbstractUncertaintySet,
                                           <:AbstractUncertaintySetEstimator} = nothing,
                                sigma::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
    end
    return UncertaintySetVariance{typeof(settings), typeof(ucs), typeof(sigma)}(settings,
                                                                                ucs, sigma)
end
function (r::UncertaintySetVariance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::UncertaintySetVariance, prior::AbstractPriorResult, ::Any,
                              ucs::Union{Nothing, <:AbstractUncertaintySet,
                                         <:AbstractUncertaintySetEstimator} = nothing,
                              args...; kwargs...)
    uset = ucs_factory(r.ucs, ucs)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma)
    return UncertaintySetVariance(; settings = r.settings, ucs = uset, sigma = sigma)
end
function cluster_risk_measure_factory(r::UncertaintySetVariance, prior::AbstractPriorResult,
                                      cluster::AbstractVector, ::Any,
                                      ucs::Union{Nothing, <:AbstractUncertaintySet,
                                                 <:AbstractUncertaintySetEstimator} = nothing,
                                      args...; kwargs...)
    uset = ucs_factory(r.ucs, ucs, cluster)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return UncertaintySetVariance(; settings = r.settings, ucs = uset, sigma = sigma)
end

export UncertaintySetVariance
