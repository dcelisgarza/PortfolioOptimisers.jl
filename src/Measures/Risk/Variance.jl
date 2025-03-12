abstract type VarianceFormulation end
struct Quad <: VarianceFormulation end
struct SOC <: VarianceFormulation end
struct RSOC <: VarianceFormulation end
struct Variance{T1 <: RiskMeasureSettings, T2 <: VarianceFormulation,
                T3 <: Union{Nothing, <:AbstractMatrix},
                T4 <: Union{Nothing, <:AbstractMatrix},
                T5 <: Union{Nothing, <:AbstractVector}} <: RiskContributionSigmaRiskMeasure
    settings::T1
    formulation::T2
    sigma::T3
    a_rc::T4
    b_rc::T5
end
function Variance(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                  formulation::VarianceFormulation = SOC(),
                  sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                  a_rc::Union{Nothing, <:AbstractMatrix} = nothing,
                  b_rc::Union{Nothing, <:AbstractVector} = nothing)
    issquarepermissive(sigma)
    if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        @smart_assert(size(a_rc, 1) == length(b_rc))
    end
    return Variance{typeof(settings), typeof(formulation), typeof(sigma), typeof(a_rc),
                    typeof(b_rc)}(settings, formulation, sigma, a_rc, b_rc)
end
function (variance::Variance)(w::AbstractVector)
    return dot(w, variance.sigma, w)
end
function risk_measure_factory(r::Variance, prior::AbstractPriorModel,
                              cluster::AbstractVector)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    a_rc = r.a_rc, b_rc = r.b_rc)
end

export Variance