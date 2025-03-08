function get_black_litterman_views_data(lca::LinearConstraintAtom{<:AbstractVector,
                                                                  <:AbstractVector,
                                                                  <:AbstractVector, <:Real},
                                        sets::DataFrame; strict::Bool = false)
    group_names = names(sets)
    N = nrow(sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, length(lca.group))
    for (group, name, coef) ∈ zip(lca.group, lca.name, lca.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero.(idx))
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group).\n$(lca)"))
                else
                    @warn("$(string(name)) is not in $(group).\n$(lca)")
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
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lca).")
        end
    end
    tcnst = lca.cnst
    return if isempty(A)
        A, tcnst
    else
        vec(sum(reshape(A, N, :); dims = 2)), tcnst
    end
end
function get_black_litterman_views_data(lca::LinearConstraintAtom{<:Any, <:Any, <:Real,
                                                                  <:Real}, sets::DataFrame;
                                        strict::Bool = false)
    group_names = names(sets)
    N = nrow(sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, N)
    (; group, name, coef) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero.(idx))
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group).\n$(lca)"))
            else
                @warn("$(string(name)) is not in $(group).\n$(lca)")
            end
            return A, lca.cnst
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
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
    end
    tcnst = lca.cnst
    return A, tcnst
end
function views_constraints(lcas::Union{<:LinearConstraintAtom,
                                       <:AbstractVector{<:LinearConstraintAtom}},
                           sets::DataFrame; datatype::Type = Float64, strict::Bool = false)
    N = nrow(sets)

    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)

    for lc ∈ lcas
        lc_A, lc_B = get_black_litterman_views_data(lc, sets; strict = strict)

        if isempty(lc_A) || all(iszero.(lc_A))
            continue
        end

        append!(P, lc_A)
        append!(Q, lc_B)
    end

    if !isempty(P)
        P = transpose(reshape(P, N, :))
        P = convert.(typeof(promote(P...)[1]), P)
        Q = convert.(typeof(promote(Q...)[1]), Q)
    end

    return P, Q
end

export views_constraints
