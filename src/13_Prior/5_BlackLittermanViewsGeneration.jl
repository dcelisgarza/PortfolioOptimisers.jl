struct BlackLittermanViews{T1, T2} <: AbstractResult
    P::T1
    Q::T2
end
function BlackLittermanViews(; P::AbstractMatrix, Q::AbstractVector)
    @argcheck(!isempty(P) && !isempty(Q))
    @argcheck(size(P, 1) == length(Q))
    return BlackLittermanViews(P, Q)
end
function black_litterman_views(::Nothing, args...; kwargs...)
    return nothing
end
function black_litterman_views(blves::LinearConstraintEstimator, args...; kwargs...)
    return blves
end
function get_black_litterman_views(lcs::Union{<:ParsingResult,
                                              <:AbstractVector{<:ParsingResult}},
                                   sets::AssetSets; datatype::DataType = Float64,
                                   strict::Bool = false)
    if isa(lcs, AbstractVector)
        @argcheck(!isempty(lcs))
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
                                          <:AbstractVector{<:Union{<:AbstractString, Expr}}},
                               sets::AssetSets; datatype::DataType = Float64,
                               strict::Bool = false)
    lcs = parse_equation(eqn; ops1 = ("==",), ops2 = (:call, :(==)), datatype = datatype)
    lcs = replace_group_by_assets(lcs, sets, true)
    return get_black_litterman_views(lcs, sets; datatype = datatype, strict = strict)
end
function black_litterman_views(lcs::LinearConstraintEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return black_litterman_views(lcs.val, sets; datatype = datatype, strict = strict)
end
function black_litterman_views(views::BlackLittermanViews, args...; kwargs...)
    return views
end
function assert_bl_views_conf(::Nothing, args...)
    return nothing
end
function assert_bl_views_conf(views_conf::Real,
                              val::Union{<:AbstractString, Expr,
                                         <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    @argcheck(!isa(val, AbstractVector))
    @argcheck(zero(views_conf) < views_conf < one(views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::AbstractVector{<:Real},
                              val::Union{<:AbstractString, Expr,
                                         <:AbstractVector{<:Union{<:AbstractString, Expr}}})
    @argcheck(isa(val, AbstractVector))
    @argcheck(!isempty(views_conf))
    @argcheck(length(val) == length(views_conf))
    @argcheck(all(x -> zero(x) < x < one(x), views_conf))
    return nothing
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::LinearConstraintEstimator)
    return assert_bl_views_conf(views_conf, views.val)
end
function assert_bl_views_conf(views_conf::Union{<:Real, <:AbstractVector{<:Real}},
                              views::BlackLittermanViews)
    return @argcheck(length(views_conf) == length(views.Q))
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
        @argcheck(!isempty(blves))
    end
    @argcheck(!isempty(sets))
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
export black_litterman_views, BlackLittermanViews
