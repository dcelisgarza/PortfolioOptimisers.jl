struct CardinalityConstraintSide{T1, T2,
                                 T3 <: Union{<:Integer, <:AbstractVector{<:Integer}}} <:
       AbstractConstraintSide
    group::T1
    name::T2
    coef::T3
end
function CardinalityConstraintSide(; group, name,
                                   coef::Union{<:Integer, <:AbstractVector{<:Integer}} = 1)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name))
        @smart_assert(length(group) == length(name))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return CardinalityConstraintSide{typeof(group), typeof(name), typeof(coef)}(group, name,
                                                                                coef)
end
function cardinality_constraint_side_view(lc::CardinalityConstraintSide{<:Any, <:Any},
                                          ::Any)
    return lc
end
function cardinality_constraint_side_view(lc::CardinalityConstraintSide{<:AbstractVector,
                                                                        <:AbstractVector,
                                                                        <:AbstractVector},
                                          i::AbstractVector)
    group = nothing_scalar_array_view(lc.group, i)
    name = nothing_scalar_array_view(lc.name, i)
    coef = nothing_scalar_array_view(lc.coef, i)
    return LinearConstraintSide(; group = group, name = name, coef = coef)
end
struct CardinalityConstraint{T1 <: CardinalityConstraintSide, T2 <: Integer,
                             T3 <: ComparisonOperators} <: AbstractConstraint
    A::T1
    B::T2
    comp::T3
end
function CardinalityConstraint(; A::CardinalityConstraintSide, B::Integer = 1,
                               comp::ComparisonOperators = LEQ())
    return CardinalityConstraint{typeof(A), typeof(B), typeof(comp)}(A, B, comp)
end
function cardinality_constraint_view(::Nothing, ::Any)
    return nothing
end
function cardinality_constraint_view(cc::CardinalityConstraint, i::AbstractVector)
    A = cardinality_constraint_side_view(cc.A, i)
    return CardinalityConstraint(; A = A, B = cc.B, comp = cc.comp)
end
function cardinality_constraint_view(cc::AbstractVector{<:CardinalityConstraint},
                                     i::AbstractVector)
    return cardinality_constraint_view.(cc, Ref(i))
end
function Base.iterate(S::Union{<:CardinalityConstraintSide, <:CardinalityConstraint},
                      state = 1)
    return state > 1 ? nothing : (S, state + 1)
end
function get_cardinality_constraint_data(lc::CardinalityConstraintSide{<:Any, <:Any, <:Any},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
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
function get_cardinality_constraint_data(lc::CardinalityConstraintSide{<:AbstractVector,
                                                                       <:AbstractVector,
                                                                       <:AbstractVector},
                                         sets::DataFrame, strict::Bool = false)
    group_names = names(sets)
    A = Vector{Int}(undef, 0)
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
        LinearConstraintResult(;
                               ineq = if ineq_flag
                                   PartialLinearConstraintResult(; A = A_ineq, B = B_ineq)
                               else
                                   nothing
                               end,
                               eq = if eq_flag
                                   PartialLinearConstraintResult(; A = A_eq, B = B_eq)
                               else
                                   nothing
                               end)
    end
end
function asset_sets_matrix(smtx::Union{Nothing, Symbol, <:AbstractString}, args...;
                           kwargs...)
    return smtx
end
function asset_sets_matrix(smtx::Union{Symbol, <:AbstractString}, sets::DataFrame)
    @smart_assert(!isempty(sets))
    sets = sets[!, smtx]
    unique_sets = unique(sets)
    A = BitMatrix(undef, length(sets), length(unique_sets))
    for (i, s) ∈ pairs(unique_sets)
        A[:, i] = sets .== s
    end
    return transpose(A)
end
function asset_sets_matrix_view(smtx::Union{Nothing, Symbol, <:AbstractString}, ::Any)
    return smtx
end
function asset_sets_matrix_view(smtx::AbstractMatrix, i::AbstractVector)
    return view(smtx, :, i)
end

export CardinalityConstraintSide, CardinalityConstraint, cardinality_constraints,
       asset_sets_matrix
