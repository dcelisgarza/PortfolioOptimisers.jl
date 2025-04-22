struct BlackLittermanViewsEstimator{T1 <: LinearConstraintSide, T2 <: Real} <:
       AbstractEstimator
    A::T1
    B::T2
end
function BlackLittermanViewsEstimator(; A::LinearConstraintSide, B::Real = 0.0)
    return BlackLittermanViewsEstimator{typeof(A), typeof(B)}(A, B)
end
function Base.iterate(S::BlackLittermanViewsEstimator, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
struct BlackLittermanViewsResult{T1 <: AbstractMatrix, T2 <: AbstractVector} <:
       AbstractResult
    P::T1
    Q::T2
end
function BlackLittermanViewsResult(; P::AbstractMatrix, Q::AbstractVector)
    @smart_assert(!isempty(P) && !isempty(Q))
    @smart_assert(size(P, 1) == length(Q))
    return BlackLittermanViewsResult{typeof(P), typeof(Q)}(P, Q)
end
function get_black_litterman_views_data(blve::LinearConstraintSide{<:AbstractVector,
                                                                   <:AbstractVector,
                                                                   <:AbstractVector},
                                        sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(blve.coef)}(undef, 0)
    for (group, name, coef) ∈ zip(blve.group, blve.name, blve.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(blve)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(blve)")
                end
                continue
            end
            if count(idx) > one(eltype(idx))
                idx = coef * idx
                sc = sign(coef)
                idx /= sum(idx)
                idx .*= sc
            else
                idx = coef * idx
            end
            append!(A, idx)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(blve)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(blve).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, nrow(sets), :); dims = 2))
    end
    return A
end
function get_black_litterman_views_data(blve::LinearConstraintSide{<:Any, <:Any, <:Real},
                                        sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(blve.coef)}(undef, 0)
    (; group, name, coef) = blve
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero, idx)
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(blve)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(blve)")
            end
            return A
        end
        if count(idx) > one(eltype(idx))
            idx = coef * idx
            sc = sign(coef)
            idx /= sum(idx)
            idx .*= sc
        else
            idx = coef * idx
        end
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(blve)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(blve)")
    end
    return A
end
function black_littterman_views(blves::Union{<:BlackLittermanViewsEstimator,
                                             <:AbstractVector{<:BlackLittermanViewsEstimator}},
                                sets::DataFrame; datatype::Type = Float64,
                                strict::Bool = false)
    if isa(blves, AbstractVector)
        @smart_assert(!isempty(blves))
    end
    @smart_assert(!isempty(sets))
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    for blve ∈ blves
        blve_A = get_black_litterman_views_data(blve.A, sets, strict)
        if isempty(blve_A) || all(iszero, blve_A)
            continue
        end
        append!(P, blve_A)
        append!(Q, blve.B)
    end
    return if !isempty(P)
        P = transpose(reshape(P, nrow(sets), :))
        BlackLittermanViewsResult(; P = P, Q = Q)
    else
        nothing
    end
end
function black_littterman_views(::Nothing, args...; kwargs...)
    return nothing
end
function black_littterman_views(blves::BlackLittermanViewsResult, args...; kwargs...)
    return blves
end
export black_littterman_views, BlackLittermanViewsEstimator, BlackLittermanViewsResult
