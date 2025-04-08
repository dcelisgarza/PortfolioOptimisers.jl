abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end
abstract type AbstractUncertaintySetAlgorithm <: AbstractAlgorithm end
abstract type AbstractUncertaintySet <: AbstractResult end
abstract type AbstractUncertaintyKAlgorithm <: AbstractAlgorithm end
function ucs end
function mu_ucs end
function sigma_ucs end
ucs(::Nothing, args...; kwargs...) = nothing
mu_ucs(::Nothing, args...; kwargs...) = nothing
sigma_ucs(::Nothing, args...; kwargs...) = nothing
ucs(uc::AbstractUncertaintySet, args...; kwargs...) = uc
mu_ucs(uc::AbstractUncertaintySet, args...; kwargs...) = uc
sigma_ucs(uc::AbstractUncertaintySet, args...; kwargs...) = uc

struct BoxUncertaintySetAlgorithm <: AbstractUncertaintySetAlgorithm end
struct BoxUncertaintySet{T1 <: Union{<:AbstractVector, <:AbstractMatrix},
                         T2 <: Union{<:AbstractVector, <:AbstractMatrix}} <:
       AbstractUncertaintySet
    lb::T1
    ub::T2
end
function BoxUncertaintySet(; lb::Union{<:AbstractVector, <:AbstractMatrix},
                           ub::Union{<:AbstractVector, <:AbstractMatrix})
    @smart_assert(!isempty(lb) && !isempty(ub))
    @smart_assert(size(lb) == size(ub))
    return BoxUncertaintySet{typeof(lb), typeof(ub)}(lb, ub)
end
struct NormalKUncertaintyMethod{T1 <: NamedTuple} <: AbstractUncertaintyKAlgorithm
    kwargs::T1
end
function NormalKUncertaintyMethod(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyMethod{typeof(kwargs)}(kwargs)
end
struct GeneralKUncertaintyMethod <: AbstractUncertaintyKAlgorithm end
struct ChiSqKUncertaintyMethod <: AbstractUncertaintyKAlgorithm end
function k_ucs(km::NormalKUncertaintyMethod, q::Real, X::AbstractMatrix,
               sigma_X::AbstractMatrix)
    k_mus = diag(X * (sigma_X \ I) * transpose(X))
    return sqrt(quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_ucs(::GeneralKUncertaintyMethod, q::Real, args...)
    return sqrt((one(q) - q) / q)
end
function k_ucs(::ChiSqKUncertaintyMethod, q::Real, X::AbstractArray, args...)
    return sqrt(cquantile(Chisq(size(X, 1)), q))
end
function k_ucs(type::Real, args...)
    return type
end
struct EllipseUncertaintySetAlgorithm{T1 <: Bool,
                                      T2 <: Union{<:AbstractUncertaintyKAlgorithm, Real}} <:
       AbstractUncertaintySetAlgorithm
    diagonal::T1
    method::T2
end
function EllipseUncertaintySetAlgorithm(; diagonal::Bool = true,
                                        method::Union{<:AbstractUncertaintyKAlgorithm,
                                                      Real} = ChiSqKUncertaintyMethod())
    return EllipseUncertaintySetAlgorithm{typeof(diagonal), typeof(method)}(diagonal,
                                                                            method)
end
struct EllipseUncertaintySet{T1 <: AbstractMatrix, T2 <: Real} <: AbstractUncertaintySet
    sigma::T1
    k::T2
end
function EllipseUncertaintySet(; sigma::AbstractMatrix, k::Real)
    @smart_assert(!isempty(sigma))
    issquare(sigma)
    @smart_assert(zero(k) < k)
    return EllipseUncertaintySet{typeof(sigma), typeof(k)}(sigma, k)
end

export ucs, mu_ucs, sigma_ucs, BoxUncertaintySetAlgorithm, BoxUncertaintySet,
       NormalKUncertaintyMethod, GeneralKUncertaintyMethod, ChiSqKUncertaintyMethod,
       EllipseUncertaintySetAlgorithm, EllipseUncertaintySet
