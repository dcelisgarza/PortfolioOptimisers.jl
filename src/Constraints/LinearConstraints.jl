abstract type LinearConstraintKind end
struct AssetLinearConstraint <: LinearConstraintKind end
struct FactorLinearConstraint <: LinearConstraintKind end
struct LinearConstraint{T1 <: LinearConstraintAtom, T2 <: LinearConstraintAtom,
                        T3 <: ComparisonOperators, T4 <: LinearConstraintKind}
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
end
function LinearConstraint(; lhs::LinearConstraintAtom, rhs::LinearConstraintAtom,
                          comp::ComparisonOperators = LEQ(),
                          kind::LinearConstraintKind = AssetLinearConstraint())
    return LinearConstraint{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind)}(lhs, rhs,
                                                                                  comp,
                                                                                  kind)
end
function relative_factor_constraint_sign(::AssetLinearConstraint)
    return 1
end
function relative_factor_constraint_sign(::FactorLinearConstraint)
    return -1
end
function get_asset_constraint_data(lca::LinearConstraintAtom{<:AbstractVector,
                                                             <:AbstractVector,
                                                             <:AbstractVector, <:Real},
                                   asset_sets::DataFrame; strict::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, length(lca.group))
    for (group, name, coef) ∈ zip(lca.group, lca.name, lca.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = asset_sets[!, group] .== name
            idx = coef * idx
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
function get_asset_constraint_data(lca::LinearConstraintAtom{<:Any, <:Any, <:Real, <:Real},
                                   asset_sets::DataFrame; strict::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(lca.coef), typeof(lca.cnst))}(undef, 0)
    sizehint!(A, N)
    (; group, name, coef) = lca
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = asset_sets[!, group] .== name
        idx = coef * idx
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lca)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lca)")
    end
    tcnst = lca.cnst
    return A, tcnst
end
function linear_constraints(lcs::Union{<:LinearConstraint,
                                       <:AbstractVector{<:LinearConstraint}},
                            asset_sets::DataFrame; datatype::Type = Float64,
                            strict::Bool = false)
    N = nrow(asset_sets)
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    for lc ∈ lcs
        lhs = lc.lhs
        rhs = lc.rhs

        lhs_A, lhs_B = get_asset_constraint_data(lhs, asset_sets; strict = strict)
        rhs_A, rhs_B = get_asset_constraint_data(rhs, asset_sets; strict = strict)

        lhs_flag = isempty(lhs_A) || all(iszero.(lhs_A))
        rhs_flag = isempty(rhs_A) || all(iszero.(rhs_A))

        if lhs_flag && rhs_flag
            continue
        end

        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)
        rlhs_A = if lhs_flag
            -rhs_A * d
        elseif rhs_flag
            lhs_A * d
        else
            sign = relative_factor_constraint_sign(lc.kind)
            sign * (lhs_A - rhs_A) * d
        end
        rlhs_B = (rhs_B - lhs_B) * d

        if flag_ineq
            append!(A_ineq, rlhs_A)
            append!(B_ineq, rlhs_B)
        else
            append!(A_eq, rlhs_A)
            append!(B_eq, rlhs_B)
        end
    end

    if !isempty(A_ineq)
        A_ineq = transpose(reshape(A_ineq, N, :))
        A_ineq = convert.(typeof(promote(A_ineq...)[1]), A_ineq)
        B_ineq = convert.(typeof(promote(B_ineq...)[1]), B_ineq)
    end
    if !isempty(A_eq)
        A_eq = transpose(reshape(A_eq, N, :))
        A_eq = convert.(typeof(promote(A_eq...)[1]), A_eq)
        B_eq = convert.(typeof(promote(B_eq...)[1]), B_eq)
    end

    return A_ineq, B_ineq, A_eq, B_eq
end

export linear_constraints, LinearConstraintAtom, LinearConstraint, AssetLinearConstraint,
       FactorLinearConstraint
