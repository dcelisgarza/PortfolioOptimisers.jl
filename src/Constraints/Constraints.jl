struct LinearConstraintSide{T1 <: Union{<:AbstractString, Symbol, Nothing,
                                        <:AbstractVector{<:Union{<:AbstractString, Symbol,
                                                                 Nothing}}},
                            T2 <: Union{<:AbstractString, Symbol, Nothing, <:Real,
                                        <:AbstractVector{<:Union{<:AbstractString, Symbol,
                                                                 Nothing, <:Real}}},
                            T3 <: Union{<:Real, <:AbstractVector{<:Real}},
                            T4 <: Union{<:Real, <:AbstractVector{<:Real}}}
    group::T1
    name::T2
    coef::T3
    cnst::T4
end
function LinearConstraintSide(;
                              group::Union{<:AbstractString, Symbol, Nothing,
                                           <:AbstractVector{<:Union{<:AbstractString,
                                                                    Symbol, Nothing}}} = nothing,
                              name::Union{<:AbstractString, Symbol, Nothing, <:Real,
                                          <:AbstractVector{<:Union{<:AbstractString, Symbol,
                                                                   Nothing, <:Real}}} = nothing,
                              coef::Union{<:Real, <:AbstractVector{<:Real}} = 1,
                              cnst::Union{<:Real, <:AbstractVector{<:Real}} = 0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    cnst_flag = isa(cnst, AbstractVector)
    if any((group_flag, name_flag, coef_flag, cnst_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag, cnst_flag)))
        @smart_assert(length(group) == length(name) == length(coef) == length(cnst))
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
        return hs.group[i], hs.name[i], hs.coef[i], hs.cnst[i]
    else
        return hs.group, hs.name, hs.coef, hs.cnst
    end
end
struct LinearConstraint{T1 <: LinearConstraintSide, T2 <: LinearConstraintSide,
                        T3 <: Union{<:AbstractString, Symbol}, T4 <: Bool}
    lhs::T1
    rhs::T2
    comp::T3
    factor::T4
end
function LinearConstraint(; lhs::LinearConstraintSide, rhs::LinearConstraintSide,
                          comp::Union{<:AbstractString, Symbol}, factor::Bool = false)
    if isa(comp, Symbol)
        @smart_assert(comp in (Symbol("=="), Symbol("<="), Symbol(">=")))
    else
        @smart_assert(comp in ("==", "<=", ">="))
    end
    return LinearConstraint{typeof(lhs), typeof(rhs), typeof(comp), typeof(factor)}(lhs,
                                                                                    rhs,
                                                                                    comp,
                                                                                    factor)
end
function get_constraint_sign_ineq(lc::LinearConstraint)
    if lc.comp == Symbol("==") ||
       lc.comp == "==" ||
       lc.comp == Symbol("<=") ||
       lc.comp == "<="
        val = 1
        ineq_flag = if lc.comp == Symbol("==") || lc.comp == "=="
            false
        else
            true
        end
    else
        val = -1
        ineq_flag = true
    end
    return val, ineq_flag
end
function get_asset_constraint_data(hs::LinearConstraintSide, asset_sets::DataFrame)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector(undef, 0)
    tcnst = 0.0
    for i ∈ eachindex(hs.cnst)
        group, name, coef, cnst = hs[i]
        group = string(group)
        if isnothing(group) || group ∉ group_names
            continue
        end
        idx = asset_sets[!, group] .== name
        append!(A, coef * idx)
        tcnst += cnst
    end
    if isempty(A)
        A, tcnst
    else
        vec(sum(reshape(A, N, :); dims = 2)), tcnst
    end
end
function asset_constraints(lcs::Union{LinearConstraint,
                                      <:AbstractVector{<:LinearConstraint}},
                           asset_sets::DataFrame; sum_to_one::Bool = false,
                           factor::Bool = false)
    N = nrow(asset_sets)

    A_ineq = Vector(undef, 0)
    B_ineq = Vector(undef, 0)

    A_eq = Vector(undef, 0)
    B_eq = Vector(undef, 0)

    for lc ∈ lcs
        lhs = lc.lhs
        rhs = lc.rhs
        d, flag_ineq = get_constraint_sign_ineq(lc)

        lhs_A, lhs_B = get_asset_constraint_data(lhs, asset_sets)
        rhs_A, rhs_B = get_asset_constraint_data(rhs, asset_sets)

        if isempty(lhs_A) && isempty(lhs_B)
            continue
        end

        # a*lhs1 + b*lhs2 + lhs_cnst <= c*rhs1 + d*rhs2 + rhs_cnst
        # a*lhs1 + b*lhs2 - c*rhs1 - d*rhs2 <= (rhs_cnst - lhs_cnst)

        rlhs_A, rlhs_B = if isempty(lhs_A)
            -rhs_A * d, rhs_B * d
        elseif isempty(rhs_A)
            lhs_A * d, -lhs_B * d
        else
            sign = if !lc.factor
                1
            else
                -1
            end
            sign * (lhs_A - rhs_A) * d, (rhs_B - lhs_B) * d
        end

        if sum_to_one
            rlhs_A /= abs(sum(rlhs_A))
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

export LinearConstraintSide, LinearConstraint, asset_constraints
