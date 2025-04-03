abstract type VarianceFormulation end
struct Quad <: VarianceFormulation end
struct SOC <: VarianceFormulation end
struct RSOC <: VarianceFormulation end
struct Variance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                T3 <: Union{Nothing, <:AbstractMatrix},
                T4 <:
                Union{Nothing, <:LinearConstraint, <:AbstractVector{<:LinearConstraint},
                      <:LinearConstraintModel}} <: RiskContributionSigmaRiskMeasure
    settings::T1
    formulation::T2
    sigma::T3
    rc::T4
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  formulation::VarianceFormulation = SOC(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  rc::Union{Nothing, <:LinearConstraint,
                            <:AbstractVector{<:LinearConstraint}, <:LinearConstraintModel} = nothing)
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        issquare(sigma)
    end
    return Variance{typeof(settings), typeof(formulation), typeof(sigma), typeof(rc)}(settings,
                                                                                      formulation,
                                                                                      sigma,
                                                                                      rc)
end
function (r::Variance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::Variance, prior::AbstractPriorModel, args...; kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    rc = r.rc)
end
function risk_measure_cluster_factory(r::Variance, prior::AbstractPriorModel,
                                      cluster::AbstractVector, args...; kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    rc = r.rc)
end

export Variance
