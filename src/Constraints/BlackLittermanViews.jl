function get_black_litterman_views_data(lca::LinearConstraintAtom{<:AbstractVector,
                                                                  <:AbstractVector,
                                                                  <:AbstractVector, <:Real},
                                        asset_sets::DataFrame; normalise::Bool = false,
                                        strict::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, length(lca.group))
    for (group, name, coef) ∈ zip(lca.group, lca.name, lca.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = asset_sets[!, group] .== name
            idx = coef * idx
            sc = sign(coef)
            idx /= sum(idx)
            idx .*= sc
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
                                                                  <:Real},
                                        asset_sets::DataFrame; normalise::Bool = false,
                                        strict::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, N)
    (; group, name, coef) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = asset_sets[!, group] .== name
        idx = coef * idx
        sc = sign(coef)
        idx /= sum(idx)
        idx .*= sc
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
                           asset_sets::DataFrame; datatype::Type = Float64,
                           strict::Bool = false)
    N = nrow(asset_sets)

    P = Vector{datatype}(undef, 0)
    Q = Vector{datatype}(undef, 0)

    for lc ∈ lcas
        lc_A, lc_B = get_black_litterman_views_data(lc, asset_sets; strict = strict)

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
