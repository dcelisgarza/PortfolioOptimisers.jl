struct CardinalityConstraintSide{T1, T2} <: AbstractConstraintSide
    group::T1
    name::T2
end
function CardinalityConstraintSide(; group, name)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    if group_flag || name_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(coef))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return CardinalityConstraintSide{typeof(group), typeof(name)}(group, name)
end
struct CardinalityConstraint{T1 <: CardinalityConstraintSide, T2 <: Real,
                             T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function CardinalityConstraint(; A::CardinalityConstraintSide, B::Real = 0.0,
                               comp::ComparisonOperators = LEQ())
    return CardinalityConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function get_cardinality_constraint_data(A_lc::CardinalityConstraintSide{<:Any, <:Any},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
    (; group, name) = A_lc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(A_lc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(A_lc)")
    end
    return A
end
function get_cardinality_constraint_data(A_lc::CardinalityConstraintSide{<:AbstractVector,
                                                                         <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
    for (group, name) ∈ zip(A_lc.group, A_lc.name)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, idx)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(A_lc)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(A_lc).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, nrow(sets), :); dims = 2))
    end
    return A
end
function cardinality_constraints(lcs::Union{<:CardinalityConstraint,
                                            <:AbstractVector{<:CardinalityConstraint}},
                                 sets::DataFrame; datatype::Type = Float64,
                                 strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    for lc ∈ lcs
        A = get_cardinality_constraint_data(lc.A, sets, strict)
        B = lc.B
        lhs_flag = isempty(A) || all(iszero.(A))
        if lhs_flag
            continue
        end
        d, flag_ineq = comparison_sign_ineq_flag(lc.comp)
        A = d * A
        B = d * lc.B
        if flag_ineq
            append!(A_ineq, A)
            append!(B_ineq, B)
        else
            append!(A_eq, A)
            append!(B_eq, B)
        end
    end
    if !isempty(A_ineq)
        A_ineq = transpose(reshape(A_ineq, nrow(sets), :))
    else
        A_ineq = nothing
        B_ineq = nothing
    end
    if !isempty(A_eq)
        A_eq = transpose(reshape(A_eq, nrow(sets), :))
    else
        A_eq = nothing
        B_eq = nothing
    end
    return LinearConstraintResult(;
                                  ineq = PartialLinearConstraintResult(; A = A_ineq,
                                                                       B = B_ineq),
                                  eq = PartialLinearConstraintResult(; A = A_eq, B = B_eq))
end
function cardinality_constraints(lcs::LinearConstraintResult, args...; kwargs...)
    return lcs
end
function cardinality_constraints(::Nothing, args...; kwargs...)
    return nothing
end
