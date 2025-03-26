struct Turnover{T1 <: Union{<:Real, <:AbstractVector{<:Real}}, T2 <: AbstractVector{<:Real}}
    val::T1
    w::T2
end
function Turnover(; val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                  w::AbstractVector{<:Real})
    if isa(val, AbstractVector)
        @smart_assert(!isempty(val))
        @smart_assert(length(val) == length(w))
        @smart_assert(all(isfinite.(val)))
    else
        @smart_assert(isfinite(val))
    end
    @smart_assert(!isempty(w))
    return Turnover{typeof(val), typeof(w)}(val, w)
end
function cluster_turnover_factory(::Nothing, ::AbstractVector)
    return nothing
end
function cluster_turnover_factory(turnover::Turnover, cluster::AbstractVector)
    val = cluster_real_or_vector_factory(turnover.val, cluster)
    w = view(turnover.w, cluster)
    return Turnover(; val = val, w = w)
end

export Turnover
