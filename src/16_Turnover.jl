struct Turnover{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                T2 <: AbstractVector{<:Real}} <: AbstractEstimator
    val::T1
    w::T2
end
function Turnover(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                  w::AbstractVector{<:Real})
    if isa(val, AbstractVector)
        @smart_assert(!isempty(val))
        @smart_assert(length(val) == length(w))
        @smart_assert(all(isfinite, val) && all(val .>= zero(eltype(val))))
    else
        @smart_assert(isfinite(val) && val >= zero(eltype(val)))
    end
    @smart_assert(!isempty(w))
    return Turnover{typeof(val), typeof(w)}(val, w)
end
function turnover_view(::Nothing, ::Any)
    return nothing
end
function turnover_view(turnover::Turnover, i::AbstractVector)
    val = nothing_scalar_array_view(turnover.val, i)
    w = view(turnover.w, i)
    return Turnover(; val = val, w = w)
end

export Turnover
