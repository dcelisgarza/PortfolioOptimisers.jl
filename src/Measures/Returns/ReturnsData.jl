struct ReturnsData{T1 <: Union{Nothing, <:AbstractVector},
                   T2 <: Union{Nothing, AbstractMatrix},
                   T3 <: Union{Nothing, <:AbstractVector},
                   T4 <: Union{Nothing, <:AbstractMatrix},
                   T5 <: Union{Nothing, <:AbstractVector}}
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
end
function ReturnsData(; nx::Union{Nothing, <:AbstractVector} = nothing,
                     X::Union{Nothing, AbstractMatrix} = nothing,
                     nf::Union{Nothing, AbstractVector} = nothing,
                     F::Union{Nothing, AbstractMatrix} = nothing,
                     ts::Union{Nothing, AbstractVector} = nothing)
    nxs_flag = !isnothing(nx)
    X_flag = !isnothing(X)
    if nxs_flag || X_flag
        @smart_assert(nxs_flag && X_flag)
        @smart_assert(!isempty(nx) && !isempty(X))
        @smart_assert(length(nx) == size(X, 2))
    end
    nfs_flag = !isnothing(nf)
    F_flag = !isnothing(F)
    if (nfs_flag || F_flag) && nxs_flag
        @smart_assert(nfs_flag && F_flag)
        @smart_assert(!isempty(nf) && !isempty(F))
        @smart_assert(length(nf) == size(F, 2))
        @smart_assert(size(X, 1) == size(F, 1))
    end
    if !isnothing(ts)
        @smart_assert(!isempty(ts))
        @smart_assert(length(ts) == size(X, 1))
    end
    return ReturnsData{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts)}(nx, X, nf,
                                                                                 F, ts)
end

export ReturnsData
