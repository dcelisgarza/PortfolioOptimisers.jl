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
    if isa(sigma, AbstractMatrix)
        @smart_assert(!isempty(sigma))
        issquare(sigma)
    end
    a_flag = isa(a_rc, AbstractMatrix)
    b_flag = isa(b_rc, AbstractVector)
    if a_flag || b_flag
        @smart_assert(a_flag && b_flag)
        @smart_assert(!isempty(a_rc))
        @smart_assert(!isempty(b_rc))
        @smart_assert(size(a_rc, 1) == length(b_rc))
    end
    return Variance{typeof(settings), typeof(formulation), typeof(sigma), typeof(a_rc),
                    typeof(b_rc)}(settings, formulation, sigma, a_rc, b_rc)
end
function (r::Variance)(w::AbstractVector)
    return dot(w, r.sigma, w)
end
function risk_measure_factory(r::Variance, prior::AbstractPriorModel, args...; kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    a_rc = r.a_rc, b_rc = r.b_rc)
end
function risk_measure_cluster_factory(r::Variance, prior::AbstractPriorModel,
                                      cluster::AbstractVector, args...; kwargs...)
    sigma = risk_measure_nothing_matrix_factory(r.sigma, prior.sigma, cluster)
    return Variance(; settings = r.settings, formulation = r.formulation, sigma = sigma,
                    a_rc = r.a_rc, b_rc = r.b_rc)
end

export Variance
