struct LinearConstraint{T1 <: A_LinearConstraint, T2 <: Real, T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function LinearConstraint(; A::A_LinearConstraint, B::Real = 0.0,
                          comp::ComparisonOperators = LEQ())
    return LinearConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
struct CardinalityConstraint{T1 <: A_CardinalityConstraint, T2 <: Real,
                             T3 <: ComparisonOperators}
    A::T1
    B::T2
    comp::T3
end
function CardinalityConstraint(; A::A_CardinalityConstraint, B::Real = 0.0,
                               comp::ComparisonOperators = LEQ())
    return CardinalityConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
struct PartialLinearConstraintModel{T1 <: Union{Nothing, <:AbstractMatrix},
                                    T2 <: Union{Nothing, <:AbstractVector}} <:
       AbstractConstraintResult
    A::T1
    B::T2
end
function PartialLinearConstraintModel(; A::Union{Nothing, <:AbstractMatrix},
                                      B::Union{Nothing, <:AbstractVector})
    if isnothing(A) || isnothing(B)
        @smart_assert(isnothing(A) && isnothing(B))
    else
        @smart_assert(!isempty(A) && !isempty(B))
    end
    return PartialLinearConstraintModel{typeof(A), typeof(B)}(A, B)
end
struct LinearConstraintModel{T1 <: PartialLinearConstraintModel,
                             T2 <: PartialLinearConstraintModel} <: AbstractConstraintResult
    ineq::T1
    eq::T2
end
function LinearConstraintModel(; ineq::PartialLinearConstraintModel,
                               eq::PartialLinearConstraintModel)
    return LinearConstraintModel{typeof(ineq), typeof(eq)}(ineq, eq)
end
function Base.getproperty(obj::LinearConstraintModel, sym::Symbol)
    return if sym == :A_ineq
        obj.ineq.A
    elseif sym == :B_ineq
        obj.ineq.B
    elseif sym == :A_eq
        obj.eq.A
    elseif sym == :B_eq
        obj.eq.B
    else
        getfield(obj, sym)
    end
end
function get_A_constraint_data(A_lc::A_LinearConstraint{<:Any, <:Any, <:Any},
                               sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(A_lc.coef)}(undef, 0)
    (; group, name, coef) = A_lc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        idx = coef * idx
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(A_lc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(A_lc)")
    end
    return A
end
function get_A_constraint_data(A_lc::A_LinearConstraint{<:AbstractVector, <:AbstractVector,
                                                        <:AbstractVector}, sets::DataFrame,
                               strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(A_lc.coef)}(undef, 0)
    for (group, name, coef) ∈ zip(A_lc.group, A_lc.name, A_lc.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            idx = coef * idx
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
function linear_constraints(lcs::Union{<:LinearConstraint,
                                       <:AbstractVector{<:LinearConstraint}},
                            sets::DataFrame; datatype::Type = Float64, strict::Bool = false)
    if isa(lcs, AbstractVector)
        @smart_assert(!isempty(lcs))
    end
    @smart_assert(!isempty(sets))
    A_ineq = Vector{datatype}(undef, 0)
    B_ineq = Vector{datatype}(undef, 0)
    A_eq = Vector{datatype}(undef, 0)
    B_eq = Vector{datatype}(undef, 0)
    for lc ∈ lcs
        A = get_A_constraint_data(lc.A, sets, strict)
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
    return LinearConstraintModel(;
                                 ineq = PartialLinearConstraintModel(; A = A_ineq,
                                                                     B = B_ineq),
                                 eq = PartialLinearConstraintModel(; A = A_eq, B = B_eq))
end
function linear_constraints(lcs::LinearConstraintModel, args...; kwargs...)
    return lcs
end
function linear_constraints(::Nothing, args...; kwargs...)
    return nothing
end

function get_A_cardinality_constraint_data(A_lc::A_CardinalityConstraint{<:Any, <:Any},
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
function get_A_cardinality_constraint_data(A_lc::A_CardinalityConstraint{<:AbstractVector,
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
        A = get_A_cardinality_constraint_data(lc.A, sets, strict)
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
    return LinearConstraintModel(;
                                 ineq = PartialLinearConstraintModel(; A = A_ineq,
                                                                     B = B_ineq),
                                 eq = PartialLinearConstraintModel(; A = A_eq, B = B_eq))
end
function cardinality_constraints(lcs::LinearConstraintModel, args...; kwargs...)
    return lcs
end
function cardinality_constraints(::Nothing, args...; kwargs...)
    return nothing
end
export linear_constraints, LinearConstraint, LinearConstraintModel,
       PartialLinearConstraintModel, cardinality_constraints, CardinalityConstraint
