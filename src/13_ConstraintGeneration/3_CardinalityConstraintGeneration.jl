struct CardinalityConstraintSide{T1, T2} <: AbstractConstraintSide
    group::T1
    name::T2
end
function CardinalityConstraintSide(; group, name)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    if group_flag || name_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name))
        @smart_assert(length(group) == length(name))
    end
    return CardinalityConstraintSide{typeof(group), typeof(name)}(group, name)
end
struct CardinalityConstraint{T1 <: CardinalityConstraintSide, T2 <: Integer,
                             T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function CardinalityConstraint(; A::CardinalityConstraintSide, B::Integer = 1,
                               comp::ComparisonOperators = LEQ())
    return CardinalityConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function Base.iterate(S::Union{<:CardinalityConstraintSide, <:CardinalityConstraint},
                      state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function get_cardinality_constraint_data(lc::CardinalityConstraintSide{<:Any, <:Any},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
    (; group, name) = lc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lc)")
    end
    return A
end
function get_cardinality_constraint_data(lc::CardinalityConstraintSide{<:AbstractVector,
                                                                       <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
    for (group, name) ∈ zip(lc.group, lc.name)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            append!(A, idx)
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lc)."))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(lc).")
        end
    end
    if !isempty(A)
        A = vec(sum(reshape(A, nrow(sets), :); dims = 2))
    end
    return A
end
function cardinality_constraints(lcs::LinearConstraintResult, args...; kwargs...)
    return lcs
end
function cardinality_constraints(::Nothing, args...; kwargs...)
    return nothing
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
        lhs_flag = isempty(A) || all(iszero.(A))
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
    return LinearConstraintResult(;
                                  ineq = if ineq_flag
                                      PartialLinearConstraintResult(; A = A_ineq,
                                                                    B = B_ineq)
                                  else
                                      nothing
                                  end,
                                  eq = if eq_flag
                                      PartialLinearConstraintResult(; A = A_eq, B = B_eq)
                                  else
                                      nothing
                                  end)
end

export CardinalityConstraintSide, CardinalityConstraint, cardinality_constraints
