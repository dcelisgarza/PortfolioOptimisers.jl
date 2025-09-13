abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
abstract type AbstractUncertaintySetResult <: AbstractResult end
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
function ucs(::Nothing, args...; kwargs...)
    return nothing
end
function mu_ucs(::Nothing, args...; kwargs...)
    return nothing
end
function sigma_ucs(::Nothing, args...; kwargs...)
    return nothing
end
function ucs(uc::Tuple{<:AbstractUncertaintySetResult, <:AbstractUncertaintySetResult},
             args...; kwargs...)
    return uc
end
function mu_ucs(uc::AbstractUncertaintySetResult, args...; kwargs...)
    return uc
end
function sigma_ucs(uc::AbstractUncertaintySetResult, args...; kwargs...)
    return uc
end
function ucs_factory(::Nothing, ::Nothing)
    return nothing
end
function ucs_factory(risk_ucs::Union{<:AbstractUncertaintySetResult,
                                     <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs_factory(::Nothing,
                     prior_ucs::Union{<:AbstractUncertaintySetResult,
                                      <:AbstractUncertaintySetEstimator})
    return prior_ucs
end
function ucs_view(risk_ucs::Union{Nothing, <:AbstractUncertaintySetEstimator}, ::Any)
    return risk_ucs
end
function ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
function mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return mu_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
function sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
    return sigma_ucs(uc, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
struct BoxUncertaintySet{T1, T2} <: AbstractUncertaintySetResult
    lb::T1
    ub::T2
end
function BoxUncertaintySet(; lb::AbstractArray, ub::AbstractArray)
    @argcheck(!isempty(lb) && !isempty(ub))
    @argcheck(size(lb) == size(ub))
    return BoxUncertaintySet(lb, ub)
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractVector, <:AbstractVector},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i), ub = view(risk_ucs.ub, i))
end
function ucs_view(risk_ucs::BoxUncertaintySet{<:AbstractMatrix, <:AbstractMatrix},
                  i::AbstractVector)
    return BoxUncertaintySet(; lb = view(risk_ucs.lb, i, i), ub = view(risk_ucs.ub, i, i))
end
struct NormalKUncertaintyAlgorithm{T1} <: AbstractUncertaintyKAlgorithm
    kwargs::T1
end
function NormalKUncertaintyAlgorithm(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyAlgorithm(kwargs)
end
struct GeneralKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
struct ChiSqKUncertaintyAlgorithm <: AbstractUncertaintyKAlgorithm end
function k_ucs(km::NormalKUncertaintyAlgorithm, q::Real, X::AbstractMatrix,
               sigma_X::AbstractMatrix)
    k_mus = diag(X * (sigma_X \ I) * transpose(X))
    return sqrt(quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyAlgorithm, q::Real, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyAlgorithm, q::Real, X::AbstractArray, args...)
    return sqrt(cquantile(Chisq(size(X, 1)), q))
end
function k_ucs(type::Real, args...)
    return type
end
struct EllipseUncertaintySetAlgorithm{T1, T2} <: AbstractUncertaintySetAlgorithm
    method::T1
    diagonal::T2
end
function EllipseUncertaintySetAlgorithm(;
                                        method::Union{<:AbstractUncertaintyKAlgorithm,
                                                      <:Real} = ChiSqKUncertaintyAlgorithm(),
                                        diagonal::Bool = true)
    return EllipseUncertaintySetAlgorithm(method, diagonal)
end
abstract type AbstractEllipseUncertaintySetResultClass <: AbstractUncertaintySetResult end
struct MuEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end
struct SigmaEllipseUncertaintySet <: AbstractEllipseUncertaintySetResultClass end
struct EllipseUncertaintySet{T1, T2, T3} <: AbstractUncertaintySetResult
    sigma::T1
    k::T2
    class::T3
end
function EllipseUncertaintySet(; sigma::AbstractMatrix, k::Real,
                               class::AbstractEllipseUncertaintySetResultClass)
    @argcheck(!isempty(sigma))
    assert_matrix_issquare(sigma)
    @argcheck(zero(k) < k)
    return EllipseUncertaintySet(sigma, k, class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:SigmaEllipseUncertaintySet},
                  i::AbstractVector)
    i = fourth_moment_index_factory(floor(Int, sqrt(size(risk_ucs.sigma, 1))), i)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end
function ucs_view(risk_ucs::EllipseUncertaintySet{<:AbstractMatrix, <:Any,
                                                  <:MuEllipseUncertaintySet},
                  i::AbstractVector)
    return EllipseUncertaintySet(; sigma = view(risk_ucs.sigma, i, i), k = risk_ucs.k,
                                 class = risk_ucs.class)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipseUncertaintySetAlgorithm, EllipseUncertaintySet
