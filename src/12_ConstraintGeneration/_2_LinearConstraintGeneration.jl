struct LinearConstraintSide{T1, T2, T3} <: AbstractConstraintSide
    group::T1
    name::T2
    coef::T3
end
function LinearConstraintSide(; group, name,
                              coef::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return LinearConstraintSide(group, name, coef)
end
# function linear_constraint_side_view(lc::LinearConstraintSide{<:Any, <:Any, <:Any}, ::Any)
#     return lc
# end
# function linear_constraint_side_view(lc::LinearConstraintSide{<:AbstractVector,
#                                                               <:AbstractVector,
#                                                               <:AbstractVector},
#                                      i::AbstractVector)
#     group = nothing_scalar_array_view(lc.group, i)
#     name = nothing_scalar_array_view(lc.name, i)
#     coef = nothing_scalar_array_view(lc.coef, i)
#     return LinearConstraintSide(; group = group, name = name, coef = coef)
# end
struct LinearConstraintEstimator{T1, T2, T3} <: AbstractConstraint
    A::T1
    B::T2
    comp::T3
end
function LinearConstraintEstimator(; A::LinearConstraintSide, B::Real = 0.0,
                                   comp::ComparisonOperators = LEQ())
    return LinearConstraintEstimator(A, B, comp)
end
function Base.iterate(S::Union{<:LinearConstraintSide, <:LinearConstraintEstimator,
                               <:PartialLinearConstraint, <:LinearConstraint}, state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
# function linear_constraint_view(::Nothing, ::Any)
#     return nothing
# end
# function linear_constraint_view(lc::LinearConstraintEstimator, i::AbstractVector)
#     A = linear_constraint_side_view(lc.A, i)
#     return LinearConstraintEstimator(; A = A, B = lc.B, comp = lc.comp)
# end
# function linear_constraint_view(lc::AbstractVector{<:LinearConstraintEstimator}, i::AbstractVector)
#     return linear_constraint_view.(lc, Ref(i))
# end
function get_constraint_data(lc::LinearConstraintSide{<:Any, <:Any, <:Any}, sets::DataFrame,
                             strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(lc.coef)}(undef, 0)
    (; group, name, coef) = lc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        if all(iszero, idx)
            if strict
                throw(ArgumentError("$(string(name)) is not in $(group). If you are sure the name is in the group make sure to use the same type in both. Comparing strings to symbols will yield false.\n$(lc)\nsets[!, group] = $(sets[!, group])."))
            else
                @warn("$(string(name)) is not in $(group). If you are sure the name is in the group make sure to use the same typeinr both. Comparing strings to symbols will yield false.\n$(lc)\nsets[!, group] = $(sets[!, group]).")
            end
        end
        idx = coef * idx
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names)."))
    else
        @warn("$(string(group)) is not in $(group_names).")
    end
    return A
end
function get_constraint_data(lc::LinearConstraintSide{<:AbstractVector, <:AbstractVector,
                                                      <:AbstractVector}, sets::DataFrame,
                             strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(lc.coef)}(undef, 0)
    for (group, name, coef) in zip(lc.group, lc.name, lc.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            if all(iszero, idx)
                if strict
                    throw(ArgumentError("$(string(name)) is not in $(group). If you are sure the name is in the group make sure to use the same type for bothinomparing strings to symbols will yield false.\n$(lc)\nsets[!, group] = $(sets[!, group])."))
                else
                    @warn("$(string(name)) is not in $(group). If you are sure the name is in the group make sure to use the same type for inh. Comparing strings to symbols will yield false.\n$(lc)\nsets[!, group] = $(sets[!, group]).")
                end
            end
            idx = coef * idx
            append!(A, idx)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names)."))
        else
            @warn("$(string(group)) is not in $(group_names).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, nrow(sets), :); dims = 2))
    end
    return A
end
function linear_constraints(lcs::Union{<:LinearConstraintEstimator,
                                       <:AbstractVector{<:LinearConstraintEstimator}},
                            sets::DataFrame; datatype::DataType = Float64,
                            strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    for lc in lcs
        A = get_constraint_data(lc.A, sets, strict)
        lhs_flag = isempty(A) || all(iszero, A)
        if lhs_flag
            continue
        end
        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)
        A .*= d
        B = d * lc.B
        if flag_ineq
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    ineq_flag = !isempty(A_ineq)
    eq_flag = !isempty(A_eq)
    if ineq_flag
        A_ineq = transpose(reshape(A_ineq, nrow(sets), :))
    end
    if eq_flag
        A_eq = transpose(reshape(A_eq, nrow(sets), :))
    end
    return if !ineq_flag && !eq_flag
        nothing
    else
        LinearConstraint(; ineq = if ineq_flag
                             PartialLinearConstraint(; A = A_ineq, B = B_ineq)
                         else
                             nothing
                         end, eq = if eq_flag
                             PartialLinearConstraint(; A = A_eq, B = B_eq)
                         else
                             nothing
                         end)
    end
end

export LinearConstraintSide, LinearConstraintEstimator
