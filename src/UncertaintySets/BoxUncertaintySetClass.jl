struct BoxUncertaintySetClass <: UncertaintySetClass end
struct BoxUncertaintySet{T1 <: Union{<:AbstractVector, <:AbstractMatrix},
                         T2 <: Union{<:AbstractVector, <:AbstractMatrix}} <: UncertaintySet
    lo::T1
    hi::T2
end
function BoxUncertaintySet(; lo::Union{<:AbstractVector, <:AbstractMatrix},
                           hi::Union{<:AbstractVector, <:AbstractMatrix})
    @smart_assert(!isempty(lo) && !isempty(hi))
    @smart_assert(size(lo) == size(hi))
    return BoxUncertaintySet{typeof(lo), typeof(hi)}(lo, hi)
end

export BoxUncertaintySetClass, BoxUncertaintySet
