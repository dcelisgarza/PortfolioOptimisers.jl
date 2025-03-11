struct NormalKUncertaintyMethod{T1 <: NamedTuple} <: UncertaintyKMethod
    kwargs::T1
end
function NormalKUncertaintyMethod(; kwargs::NamedTuple = (;))
    return NormalKUncertaintyMethod{typeof(kwargs)}(kwargs)
end
struct GeneralKUncertaintyMethod <: UncertaintyKMethod end
struct ChiSqKUncertaintyMethod <: UncertaintyKMethod end
function k_uncertainty_set(km::NormalKUncertaintyMethod, q::Real, X::AbstractMatrix,
                           sigma_X::AbstractMatrix)
    k_mus = diag(X * (sigma_X \ I) * transpose(X))
    return sqrt(quantile(k_mus, one(q) - q; km.kwargs...))
end
function k_uncertainty_set(::GeneralKUncertaintyMethod, q::Real, args...)
    return sqrt((one(q) - q) / q)
end
function k_uncertainty_set(::ChiSqKUncertaintyMethod, q::Real, X::AbstractArray, args...)
    return sqrt(cquantile(Chisq(size(X, 1)), q))
end
function k_uncertainty_set(type::Real, args...)
    return type
end
struct EllipseUncertaintySetClass{T1 <: Bool, T2 <: Union{<:UncertaintyKMethod, Real}} <:
       UncertaintySetClass
    diagonal::T1
    method::T2
end
function EllipseUncertaintySetClass(; diagonal::Bool = true,
                                    method::Union{<:UncertaintyKMethod, Real} = ChiSqKUncertaintyMethod())
    return EllipseUncertaintySetClass{typeof(diagonal), typeof(method)}(diagonal, method)
end
struct EllipseUncertaintySet{T1 <: AbstractMatrix, T2 <: Real} <: UncertaintySet
    sigma::T1
    k::T2
end
function EllipseUncertaintySet(; sigma::AbstractMatrix, k::Real)
    @smart_assert(size(sigma, 1) == size(sigma, 2))
    @smart_assert(zero(k) < k)
    return EllipseUncertaintySet{typeof(sigma), typeof(k)}(sigma, k)
end

export NormalKUncertaintyMethod, GeneralKUncertaintyMethod, ChiSqKUncertaintyMethod,
       EllipseUncertaintySetClass, EllipseUncertaintySet
