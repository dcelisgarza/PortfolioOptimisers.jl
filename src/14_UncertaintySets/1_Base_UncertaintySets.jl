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
struct BoxUncertaintySetResult{T1 <: AbstractArray, T2 <: AbstractArray} <:
       AbstractUncertaintySetResult
    lb::T1
    ub::T2
end
function BoxUncertaintySetResult(; lb::AbstractArray, ub::AbstractArray)
    @smart_assert(!isempty(lb) && !isempty(ub))
    @smart_assert(size(lb) == size(ub))
    return BoxUncertaintySetResult{typeof(lb), typeof(ub)}(lb, ub)
end
struct NormalKUncertaintyAlgorithm{T1 <: NamedTuple} <: AbstractUncertaintyKAlgorithm
    kwargs::T1
end
function NormalKUncertaintyAlgorithm(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyAlgorithm{typeof(kwargs)}(kwargs)
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
struct EllipseUncertaintySetAlgorithm{T1 <: Union{<:AbstractUncertaintyKAlgorithm, <:Real},
                                      T2 <: Bool} <: AbstractUncertaintySetAlgorithm
    method::T1
    diagonal::T2
end
function EllipseUncertaintySetAlgorithm(;
                                        method::Union{<:AbstractUncertaintyKAlgorithm,
                                                      <:Real} = ChiSqKUncertaintyAlgorithm(),
                                        diagonal::Bool = true)
    return EllipseUncertaintySetAlgorithm{typeof(method), typeof(diagonal)}(method,
                                                                            diagonal)
end
abstract type AbstractEllipseUncertaintySetResultClass <: AbstractUncertaintySetResult end
struct MuEllipseUncertaintySetResult <: AbstractEllipseUncertaintySetResultClass end
struct SigmaEllipseUncertaintySetResult <: AbstractEllipseUncertaintySetResultClass end
struct EllipseUncertaintySetResult{T1 <: AbstractMatrix, T2 <: Real,
                                   T3 <: AbstractEllipseUncertaintySetResultClass} <:
       AbstractUncertaintySetResult
    sigma::T1
    k::T2
    class::T3
end
function EllipseUncertaintySetResult(; sigma::AbstractMatrix, k::Real,
                                     class::AbstractEllipseUncertaintySetResultClass)
    @smart_assert(!isempty(sigma))
    assert_matrix_issquare(sigma)
    @smart_assert(zero(k) < k)
    return EllipseUncertaintySetResult{typeof(sigma), typeof(k), typeof(class)}(sigma, k,
                                                                                class)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySetResult,
       NormalKUncertaintyAlgorithm, GeneralKUncertaintyAlgorithm,
       ChiSqKUncertaintyAlgorithm, EllipseUncertaintySetAlgorithm,
       EllipseUncertaintySetResult
