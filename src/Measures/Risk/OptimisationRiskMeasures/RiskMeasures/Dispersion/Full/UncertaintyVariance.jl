struct UncertaintySetVariance{T1 <: RiskMeasureSettings,
                              T2 <: Union{Nothing, <:UncertaintySet},
                              T3 <: Union{Nothing, <:AbstractMatrix{<:Real}}} <:
       SigmaRiskMeasure
    settings::T1
    uncertainty_set::T2
    sigma::T3
end
function UncertaintySetVariance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                uncertainty_set::Union{Nothing, <:UncertaintySet} = nothing,
                                sigma::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
    end
    return UncertaintySetVariance{typeof(settings), typeof(uncertainty_set), typeof(sigma)}(settings,
                                                                                            uncertainty_set,
                                                                                            sigma)
end
function (r::UncertaintySetVariance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::UncertaintySetVariance, prior::AbstractPriorModel, ::Any,
                              uncertainty_set::Union{Nothing, <:UncertaintySet} = nothing,
                              args...)
    uset = uncertainty_set_factory(r.uncertainty_set, uncertainty_set)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma)
    return UncertaintySetVariance(; settings = r.settings, uncertainty_set = uset,
                                  sigma = sigma)
end
function cluster_risk_measure_factory(r::UncertaintySetVariance, prior::AbstractPriorModel,
                                      cluster::AbstractVector, ::Any,
                                      uncertainty_set::Union{Nothing, <:UncertaintySet} = nothing,
                                      args...)
    uset = uncertainty_set_factory(r.uncertainty_set, uncertainty_set, cluster)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return UncertaintySetVariance(; settings = r.settings, uncertainty_set = uset,
                                  sigma = sigma)
end

export UncertaintySetVariance
