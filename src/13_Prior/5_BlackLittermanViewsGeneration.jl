struct BlackLittermanViews{T1, T2} <: AbstractResult
    P::T1
    Q::T2
end
function BlackLittermanViews(; P::AbstractMatrix, Q::AbstractVector)
    @smart_assert(!isempty(P) && !isempty(Q))
    @smart_assert(size(P, 1) == length(Q))
    return BlackLittermanViews(P, Q)
end
function black_litterman_views(::Nothing, args...; kwargs...)
    return nothing
end
function black_litterman_views(blves::BlackLittermanViews, args...; kwargs...)
    return blves
end
function get_black_litterman_views(lcs::Union{<:ParsingResult,
                                              <:AbstractVector{<:ParsingResult}},
                                   sets::AssetSets; datatype::DataType = Float64,
                                   strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    nx = sets.dict[sets.key]
    At = Vector{datatype}(undef, length(nx))
    for lc in lcs
        fill!(At, zero(eltype(At)))
        for (v, c) in zip(lc.vars, lc.coef)
            Ai = (nx .== v)
            if !any(isone, Ai)
                msg = "$(v) is not found in $(nx)."
                strict ? throw(ArgumentError(msg)) : @warn(msg)
                continue
            end
            At += Ai * c
        end
        append!(P, At)
        append!(Q, lc.rhs)
    end
    return if !isempty(P)
        P = transpose(reshape(P, length(nx), :))
        BlackLittermanViews(; P = P, Q = Q)
    else
        nothing
    end
end
function black_litterman_views(eqn::Union{<:AbstractString, Expr,
                                          <:AbstractVector{<:AbstractString},
                                          <:AbstractVector{Expr},
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    lcs = parse_equation(eqn; datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true)
    return get_black_litterman_views(lcs, sets; datatype = datatype, strict = strict)
end
#=
#! Start: to delete
struct BlackLittermanViewsEstimator{T1, T2} <: AbstractEstimator
    A::T1
    B::T2
end
function BlackLittermanViewsEstimator(; A::LinearConstraintSide, B::Real = 0.0)
    return BlackLittermanViewsEstimator(A, B)
end
function Base.iterate(S::BlackLittermanViewsEstimator, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function get_black_litterman_views_data(blve::LinearConstraintSide{<:AbstractVector,
                                                                   <:AbstractVector,
                                                                   <:AbstractVector},
                                        sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(blve.coef)}(undef, 0)
    for (group, name, coef) in zip(blve.group, blve.name, blve.coef)
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
function black_litterman_views(blves::Union{<:BlackLittermanViewsEstimator,
                                            <:AbstractVector{<:BlackLittermanViewsEstimator}},
                               sets::DataFrame; datatype::DataType = Float64,
                               strict::Bool = false)
    if isa(blves, AbstractVector)
        @smart_assert(!isempty(blves))
    end
    @smart_assert(!isempty(sets))
    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)
    for blve in blves
        blve_A = get_black_litterman_views_data(blve.A, sets, strict)
        if isempty(blve_A) || all(iszero, blve_A)
            continue
        end
        append!(P, blve_A)
        append!(Q, blve.B)
    end
    return if !isempty(P)
        P = transpose(reshape(P, nrow(sets), :))
        BlackLittermanViews(; P = P, Q = Q)
    else
        nothing
    end
end
#! End: to delete
=#
export black_litterman_views, BlackLittermanViewsEstimator, BlackLittermanViews
