struct LinearConstraintSide{T1, T2, T3 <: Union{<:Real, <:AbstractVector{<:Real}}} <:
       AbstractConstraintSide
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
    return LinearConstraintSide{typeof(group), typeof(name), typeof(coef)}(group, name,
                                                                           coef)
end
struct LinearConstraint{T1 <: LinearConstraintSide, T2 <: Real,
                        T3 <: ComparisonOperators} <: AbstractConstraint
    A::T1
    B::T2
    comp::T3
end
function LinearConstraint(; A::LinearConstraintSide, B::Real = 0.0,
                          comp::ComparisonOperators = LEQ())
    return LinearConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
struct PartialLinearConstraintResult{T1 <: Union{Nothing, <:AbstractMatrix},
                                     T2 <: Union{Nothing, <:AbstractVector}} <:
       AbstractConstraintResult
    A::T1
    B::T2
end
function PartialLinearConstraintResult(; A::Union{Nothing, <:AbstractMatrix},
                                       B::Union{Nothing, <:AbstractVector})
    if isnothing(A) || isnothing(B)
        @smart_assert(isnothing(A) && isnothing(B))
    else
        @smart_assert(!isempty(A) && !isempty(B))
    end
    return PartialLinearConstraintResult{typeof(A), typeof(B)}(A, B)
end
struct LinearConstraintResult{T1 <: PartialLinearConstraintResult,
                              T2 <: PartialLinearConstraintResult} <:
       AbstractConstraintResult
    ineq::T1
    eq::T2
end
function LinearConstraintResult(; ineq::PartialLinearConstraintResult,
                                eq::PartialLinearConstraintResult)
    return LinearConstraintResult{typeof(ineq), typeof(eq)}(ineq, eq)
end
function Base.getproperty(obj::LinearConstraintResult, sym::Symbol)
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
function Base.iterate(S::Union{<:LinearConstraintSide, <:LinearConstraint,
                               <:PartialLinearConstraintResult, <:LinearConstraintResult},
                      state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function get_constraint_data(lc::LinearConstraintSide{<:Any, <:Any, <:Any}, sets::DataFrame,
                             strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(lc.coef)}(undef, 0)
    (; group, name, coef) = lc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        idx = coef * idx
        append!(A, idx)
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(lc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(lc)")
    end
    return A
end
function get_constraint_data(lc::LinearConstraintSide{<:AbstractVector, <:AbstractVector,
                                                      <:AbstractVector}, sets::DataFrame,
                             strict::Bool = false)
    group_names = names(sets)
    A = Vector{eltype(lc.coef)}(undef, 0)
    for (group, name, coef) ∈ zip(lc.group, lc.name, lc.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            idx = coef * idx
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
function linear_constraints(lcs::LinearConstraintResult, args...; kwargs...)
    return lcs
end
function linear_constraints(::Nothing, args...; kwargs...)
    return nothing
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
        A = get_constraint_data(lc.A, sets, strict)
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

export LinearConstraintSide, LinearConstraint, PartialLinearConstraintResult,
       LinearConstraintResult, linear_constraints