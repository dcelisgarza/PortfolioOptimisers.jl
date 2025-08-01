function CardinalityConstraintSide(; group, name,
                                   coef::Union{<:Integer, <:AbstractVector{<:Integer}} = 1)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    coef_flag = isa(coef, AbstractVector)
    if group_flag || name_flag || coef_flag
        @smart_assert(group_flag && name_flag && coef_flag)
        @smart_assert(!isempty(group) && !isempty(name))
        @smart_assert(length(group) == length(name) == length(coef))
    end
    return LinearConstraintSide(group, name, coef)
end
function CardinalityConstraint(; A::LinearConstraintSide, B::Integer = 1,
                               comp::ComparisonOperators = LEQ())
    return LinearConstraintEstimator(A, B, comp)
end
function asset_sets_matrix(smtx::Union{Symbol, <:AbstractString}, sets::DataFrame;
                           kwargs...)
    @smart_assert(!isempty(sets))
    sets = sets[!, smtx]
    unique_sets = unique(sets)
    A = BitMatrix(undef, length(sets), length(unique_sets))
    for (i, s) in pairs(unique_sets)
        A[:, i] = sets .== s
    end
    return transpose(A)
end

export CardinalityConstraintSide, CardinalityConstraint
