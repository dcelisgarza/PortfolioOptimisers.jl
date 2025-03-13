abstract type AbstractTurnover end
struct NoTurnover <: AbstractTurnover end
struct Turnover{T1 <: Union{<:Real, <:AbstractVector{<:Real}},
                T2 <: AbstractVector{<:Real}} <: AbstractTurnover
    val::T1
    w::T2
end
function Turnover(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                  w::AbstractVector{<:Real})
    if isa(val, AbstractVector)
        @smart_assert(!isempty(val))
        @smart_assert(length(val) == length(w))
    end
    @smart_assert(!isempty(w))
    return Turnover{typeof(val), typeof(w)}(val, w)
end

export NoTurnover, Turnover
