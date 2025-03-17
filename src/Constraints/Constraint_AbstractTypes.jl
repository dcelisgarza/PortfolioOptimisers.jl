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
#! This can be used like black litterman views to construct risk budget constraints
struct PartialLinearConstraintAtom{T1, T2, T3}
    group::T1
    name::T2
    coef::T3
end
function PartialLinearConstraintAtom(; group = nothing, name = nothing,
                                     coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if any((group_flag, name_flag, coef_flag))
        @smart_assert(all((group_flag, name_flag, coef_flag)))
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
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
    return PartialLinearConstraintAtom{typeof(group), typeof(name), typeof(coef)}(group,
                                                                                  name,
                                                                                  coef)
end
struct LinearConstraintAtom{T1 <: PartialLinearConstraintAtom, T2 <: Real}
    plca::T1
    cnst::T2
end
function LinearConstraintAtom(; group = nothing, name = nothing,
                              coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                              cnst::Real = 0.0)
    plca = PartialLinearConstraintAtom(; group = group, name = name, coef = coef)
    return LinearConstraintAtom{typeof(plca), typeof(cnst)}(plca, cnst)
end
function Base.getproperty(obj::LinearConstraintAtom, sym::Symbol)
    return if sym == :group
        obj.plca.group
    elseif sym == :name
        obj.plca.name
    elseif sym == :coef
        obj.plca.coef
    else
        return getfield(obj, sym)
    end
end

export EQ, LEQ, GEQ, PartialLinearConstraintAtom, LinearConstraintAtom
