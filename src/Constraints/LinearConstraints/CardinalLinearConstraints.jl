struct CardinalLinearConstraintSide{T1, T2, T3} <: AbstractLinearConstraintSide
    group::T1
    name::T2
    cnst::T3
end
function CardinalLinearConstraintSide(; group = nothing, name = nothing,
                                      cnst::Union{<:Real, <:AbstractVector{<:Real}} = 0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    cnst_flag = isa(cnst, AbstractVector)
    if any((group_flag, name_flag, cnst_flag))
        @smart_assert(all((group_flag, name_flag, cnst_flag)))
        @smart_assert(length(group) == length(name) == length(cnst))
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
    return CardinalLinearConstraintSide{typeof(group), typeof(name), typeof(cnst)}(group,
                                                                                   name,
                                                                                   cnst)
end
function Base.getindex(hs::CardinalLinearConstraintSide, i::Int)
    if isa(hs.group, AbstractVector)
        return hs.group[i], hs.name[i], hs.cnst[i]
    else
        return hs.group, hs.name, hs.cnst
    end
end
struct CardinalLinearConstraint{T1 <: CardinalLinearConstraintSide,
                                T2 <: CardinalLinearConstraintSide,
                                T3 <: ComparisonOperators} <: AbstractLinearConstraint
    lhs::T1
    rhs::T2
    comp::T3
end
function CardinalLinearConstraint(; lhs::CardinalLinearConstraintSide,
                                  rhs::CardinalLinearConstraintSide,
                                  comp::ComparisonOperators = LEQ())
    return CardinalLinearConstraint{typeof(lhs), typeof(rhs), typeof(comp)}(lhs, rhs, comp)
end
function get_asset_constraint_data(hs::CardinalLinearConstraintSide, asset_sets::DataFrame)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector(undef, 0)
    tcnst = 0.0
    for i ∈ eachindex(hs.cnst)
        group, name, cnst = hs[i]
        group = string(group)
        if isnothing(group) || group ∉ group_names
            continue
        end
        idx = asset_sets[!, group] .== name
        append!(A, idx)
        tcnst += cnst
    end
    if isempty(A)
        A, tcnst
    else
        vec(sum(reshape(A, N, :); dims = 2)), ceil(Int, tcnst)
    end
end
function linear_constraints(lcs::Union{CardinalLinearConstraint,
                                       <:AbstractVector{<:CardinalLinearConstraint}},
                            asset_sets::DataFrame)
    N = nrow(asset_sets)

    A_ineq = Vector(undef, 0)
    B_ineq = Vector(undef, 0)

    A_eq = Vector(undef, 0)
    B_eq = Vector(undef, 0)

    for lc ∈ lcs
        lhs = lc.lhs
        rhs = lc.rhs

        lhs_A, lhs_B = get_asset_constraint_data(lhs, asset_sets)
        rhs_A, rhs_B = get_asset_constraint_data(rhs, asset_sets)

        if isempty(lhs_A) && isempty(lhs_B)
            continue
        end

        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)

        rlhs_A, rlhs_B = if isempty(lhs_A)
            -rhs_A * d, rhs_B * d
        elseif isempty(rhs_A)
            lhs_A * d, -lhs_B * d
        else
            (lhs_A - rhs_A) * d, (rhs_B - lhs_B) * d
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

export CardinalLinearConstraintSide, CardinalLinearConstraint
