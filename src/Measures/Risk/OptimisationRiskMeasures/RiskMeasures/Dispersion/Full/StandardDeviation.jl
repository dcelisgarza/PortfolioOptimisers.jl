struct StandardDeviation{T1 <: RiskMeasureSettings,
                         T2 <: Union{Nothing, <:AbstractMatrix}} <: SigmaRiskMeasure
    settings::T1
    sigma::T2
end
function StandardDeviation(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                           sigma::Union{Nothing, <:AbstractMatrix} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        issquare(sigma)
    end
    return StandardDeviation{typeof(settings), typeof(sigma)}(settings, sigma)
end
function (r::StandardDeviation)(w::AbstractVector)
    return sqrt(dot(w, r.sigma, w))
end
function risk_measure_factory(r::StandardDeviation, prior::AbstractPriorResult, args...;
                              kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end
function cluster_risk_measure_factory(r::StandardDeviation, prior::AbstractPriorResult,
                                      cluster::AbstractVector, args...; kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return StandardDeviation(; settings = r.settings, sigma = sigma)
end

export StandardDeviation
