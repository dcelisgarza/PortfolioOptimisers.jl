struct Turnover{T1 <: AbstractVector{<:Real},
                T2 <: Union{<:Real, <:AbstractVector{<:Real}}} <: AbstractEstimator
    w::T1
    val::T2
end
function Turnover(; w::AbstractVector{<:Real},
                  val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    if isa(val, AbstractVector)
        @smart_assert(!isempty(val))
        @smart_assert(length(val) == length(w))
        @smart_assert(all(isfinite, val) && all(val .>= zero(eltype(val))))
    else
        @smart_assert(isfinite(val) && val >= zero(eltype(val)))
    end
    @smart_assert(!isempty(w))
    return Turnover{typeof(w), typeof(val)}(w, val)
end
function turnover_view(::Nothing, ::Any)
    return nothing
end
function turnover_view(tn::Turnover, i::AbstractVector)
    w = view(tn.w, i)
    val = nothing_scalar_array_view(tn.val, i)
    return Turnover(; w = w, val = val)
end

export Turnover
