abstract type ComparisonOperators end
abstract type EqualityComparisonOperators <: ComparisonOperators end
abstract type InequalityComparisonOperators <: ComparisonOperators end
struct EQ <: EqualityComparisonOperators end
struct LEQ <: InequalityComparisonOperators end
struct GEQ <: InequalityComparisonOperators end
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end
struct LinearConstraintAtom{T1, T2, T3, T4 <: Real}
    group::T1
    name::T2
    coef::T3
    cnst::T4
end
function LinearConstraintAtom(; group = nothing, name = nothing,
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
    return LinearConstraintAtom{typeof(group), typeof(name), typeof(coef), typeof(cnst)}(group,
                                                                                         name,
                                                                                         coef,
                                                                                         cnst)
end
export EQ, LEQ, GEQ
