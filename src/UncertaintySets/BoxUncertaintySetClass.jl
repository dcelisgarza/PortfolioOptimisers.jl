struct BoxUncertaintySetClass <: UncertaintySetClass end
struct BoxUncertaintySet{T1 <: Union{<:AbstractVector, <:AbstractMatrix},
                         T2 <: Union{<:AbstractVector, <:AbstractMatrix}} <: UncertaintySet
    lb::T1
    ub::T2
end
function BoxUncertaintySet(; lb::Union{<:AbstractVector, <:AbstractMatrix},
                           ub::Union{<:AbstractVector, <:AbstractMatrix})
    @smart_assert(!isempty(lb) && !isempty(ub))
    @smart_assert(size(lb) == size(ub))
    return BoxUncertaintySet{typeof(lb), typeof(ub)}(lb, ub)
end

export BoxUncertaintySetClass, BoxUncertaintySet
