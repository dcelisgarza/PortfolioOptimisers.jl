struct BlackLittermanView{T1 <: A_LinearConstraint, T2 <: Real}
    A::T1
    B::T2
end
function BlackLittermanView(; A::A_LinearConstraint = A_LinearConstraint(), B::Real = 0.0)
    return BlackLittermanView{typeof(A), typeof(B)}(A, B)
end
struct BlackLittermanViewsModel{T1 <: AbstractMatrix, T2 <: AbstractVector}
    P::T1
    Q::T2
end
function BlackLittermanViewsModel(; P::AbstractMatrix, Q::AbstractVector)
    @smart_assert(!isempty(P) && !isempty(Q))
    @smart_assert(size(P, 1) == length(Q))
    return BlackLittermanViewsModel{typeof(P), typeof(Q)}(P, Q)
end
function get_A_black_litterman_views_data(blv::A_LinearConstraint{<:AbstractVector,
                                                                  <:AbstractVector,
                                                                  <:AbstractVector},
                                          sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(blv.coef)}(undef, 0)
    for (group, name, coef) ∈ zip(blv.group, blv.name, blv.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero.(idx))
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(blv)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(blv)")
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
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(blv)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(blv).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, nrow(sets), :); dims = 2))
    end
    return A
end
function get_A_black_litterman_views_data(blv::A_LinearConstraint{<:Any, <:Any, <:Real},
                                          sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(blv.coef)}(undef, 0)
    (; group, name, coef) = blv
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero.(idx))
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(blv)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(blv)")
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
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(blv)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(blv)")
    end
    return A
end
function views_constraints(lcas::Union{<:BlackLittermanView,
                                       <:AbstractVector{<:BlackLittermanView}},
                           sets::DataFrame; datatype::Type = Float64, strict::Bool = false)
    if isa(lcas, AbstractVector)
        @smart_assert(!isempty(lcas))
    end
    @smart_assert(!isempty(sets))

    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)

    for lc ∈ lcas
        lc_A = get_A_black_litterman_views_data(lc.A, sets, strict)

        if isempty(lc_A) || all(iszero.(lc_A))
            continue
        end

        append!(P, lc_A)
        append!(Q, lc.B)
    end

    if !isempty(P)
        P = transpose(reshape(P, nrow(sets), :))
    end

    return BlackLittermanViewsModel(; P = P, Q = Q)
end

export views_constraints, BlackLittermanView
