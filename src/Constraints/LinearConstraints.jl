function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end
struct LinearConstraintSide{T1, T2, T3, T4 <: Real}
    group::T1
    name::T2
    coef::T3
    cnst::T4
end
function LinearConstraintSide(; group = nothing, name = nothing,
                              coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                              cnst::Real = 0.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if any((group_flag, name_flag, coef_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag)))
        @smart_assert(length(group) == length(name) == length(coef))
        for (g, n) ∈ zip(group, name)
            if isnothing(g) || isnothing(n)
                @smart_assert(isnothing(g) && isnothing(n))
            end
        end
    else
        if isnothing(group) || isnothing(name)
            @smart_assert(isnothing(group) && isnothing(name))
        end
    end
    return LinearConstraintSide{typeof(group), typeof(name), typeof(coef), typeof(cnst)}(group,
                                                                                         name,
                                                                                         coef,
                                                                                         cnst)
end
function Base.getindex(hs::LinearConstraintSide, i::Int)
    if isa(hs.group, AbstractVector)
        return hs.group[i], hs.name[i], hs.coef[i]
    else
        return hs.group, hs.name, hs.coef
    end
end
abstract type LinearConstraintKind end
struct AssetLinearConstraint <: LinearConstraintKind end
struct FactorLinearConstraint <: LinearConstraintKind end
struct LinearConstraint{T1 <: LinearConstraintSide, T2 <: LinearConstraintSide,
                        T3 <: ComparisonOperators, T4 <: LinearConstraintKind, T5 <: Bool}
    lhs::T1
    rhs::T2
    comp::T3
    kind::T4
    normalise::T5
end
function LinearConstraint(; lhs::LinearConstraintSide, rhs::LinearConstraintSide,
                          comp::ComparisonOperators = LEQ(),
                          kind::LinearConstraintKind = AssetLinearConstraint(),
                          normalise::Bool = false)
    return LinearConstraint{typeof(lhs), typeof(rhs), typeof(comp), typeof(kind),
                            typeof(normalise)}(lhs, rhs, comp, kind, normalise)
end
function get_asset_constraint_data(hs::LinearConstraintSide{<:AbstractVector,
                                                            <:AbstractVector,
                                                            <:AbstractVector, <:Real},
                                   asset_sets::DataFrame)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(hs.coef), typeof(hs.cnst))}(undef, 0)
    sizehint!(A, length(hs.group))
    tcnst = hs.cnst
    for (group, name, coef) ∈ zip(hs.group, hs.name, hs.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = asset_sets[!, group] .== name
            append!(A, coef * idx)
        end
    end
    return if isempty(A)
        A, tcnst
    else
        vec(sum(reshape(A, N, :); dims = 2)), tcnst
    end
end
function get_asset_constraint_data(hs::LinearConstraintSide{<:Any, <:Any, <:Real, <:Real},
                                   asset_sets::DataFrame)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(hs.coef), typeof(hs.cnst))}(undef, 0)
    sizehint!(A, N)
    tcnst = hs.cnst
    (; group, name, coef) = hs
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = asset_sets[!, group] .== name
        append!(A, coef * idx)
    end
    return A, tcnst
end
function relative_factor_constraint_sign(::AssetLinearConstraint)
    return 1
end
function relative_factor_constraint_sign(::FactorLinearConstraint)
    return -1
end
function linear_constraints(lcs::Union{LinearConstraint,
                                       <:AbstractVector{<:LinearConstraint}},
                            asset_sets::DataFrame, datatype::Type = Float64)
    N = nrow(asset_sets)

    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)

    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)

    for lc ∈ lcs
        lhs = lc.lhs
        rhs = lc.rhs

        lhs_A, lhs_B = get_asset_constraint_data(lhs, asset_sets)
        rhs_A, rhs_B = get_asset_constraint_data(rhs, asset_sets)

        empty_lhs_A = isempty(lhs_A)
        zero_lhs_A = all(iszero.(lhs_A))

        empty_rhs_A = isempty(rhs_A)
        zero_rhs_A = all(iszero.(rhs_A))

        if empty_lhs_A && empty_rhs_A || zero_lhs_A && zero_rhs_A
            continue
        end

        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)

        rlhs_A = if empty_lhs_A || zero_lhs_A
            -rhs_A * d
        elseif empty_rhs_A || zero_rhs_A
            lhs_A * d
        else
            sign = relative_factor_constraint_sign(lc.kind)
            sign * (lhs_A - rhs_A) * d
        end

        if isempty(rlhs_A) || all(iszero.(rlhs_A))
            continue
        end

        rlhs_B = (rhs_B - lhs_B) * d

        if lc.normalise
            s = sum(rlhs_A)
            if !iszero(s)
                rlhs_A /= abs(sum(rlhs_A))
            end
        end

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

export linear_constraints, LinearConstraintSide, LinearConstraint, AssetLinearConstraint,
       FactorLinearConstraint
