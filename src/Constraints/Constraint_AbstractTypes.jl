abstract type ComparisonOperators end
abstract type EqualityComparisonOperators <: ComparisonOperators end
abstract type InequalityComparisonOperators <: ComparisonOperators end
struct EQ <: EqualityComparisonOperators end
struct LEQ <: InequalityComparisonOperators end
struct GEQ <: InequalityComparisonOperators end
abstract type AbstractConstraintModel end
function comparison_sign_ineq_flag(::EQ)
    return 1, false
end
function comparison_sign_ineq_flag(::LEQ)
    return 1, true
end
function comparison_sign_ineq_flag(::GEQ)
    return -1, true
end
abstract type ConstraintSide end
abstract type A_Constraint <: ConstraintSide end
struct A_LinearConstraint{T1, T2, T3 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       A_Constraint
    group::T1
    name::T2
    coef::T3
end
function A_LinearConstraint(; group, name,
                            coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return A_LinearConstraint{typeof(group), typeof(name), typeof(coef)}(group, name, coef)
end
struct A_CardinalityConstraint{T1, T2} <: A_Constraint
    group::T1
    name::T2
end
function A_CardinalityConstraint(; group, name)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    if group_flag || name_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return A_CardinalityConstraint{typeof(group), typeof(name)}(group, name)
end

export EQ, LEQ, GEQ, A_LinearConstraint, A_CardinalityConstraint
