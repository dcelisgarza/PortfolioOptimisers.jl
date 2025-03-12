struct StandardDeviation{T1 <: RiskMeasureSettings,
                         T2 <: Union{Nothing, <:AbstractMatrix}} <: SigmaRiskMeasure
    settings::T1
    sigma::T2
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    issquarepermissive(sigma)
    return StandardDeviation{typeof(settings), typeof(sigma)}(settings, sigma)
end
function (sd::StandardDeviation)(w::AbstractVector)
    return sqrt(dot(w, sd.sigma, w))
end
function cluster_risk_measure_factory(r::StandardDeviation, prior::AbstractPriorModel,
                                      cluster::AbstractVector)
    sigma = risk_measure_nothing_matrix_factory_cluster(r.sigma, prior.sigma, cluster)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end

export StandardDeviation
