struct LinearConstraintSide{T1, T2, T3, T4 <: Real}
    group::T1
    name::T2
    coef::T3
    cnst::T4
end
function LinearConstraintSide(; group = nothing, name = nothing,
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
    return LinearConstraintSide{typeof(group), typeof(name), typeof(coef), typeof(cnst)}(group,
                                                                                         name,
                                                                                         coef,
                                                                                         cnst)
end
function get_asset_constraint_data(hs::LinearConstraintSide{<:AbstractVector,
                                                            <:AbstractVector,
                                                            <:AbstractVector, <:Real},
                                   asset_sets::DataFrame, throw_if_missing::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(hs.coef), typeof(hs.cnst))}(undef, 0)
    sizehint!(A, length(hs.group))
    for (group, name, coef) ∈ zip(hs.group, hs.name, hs.coef)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = asset_sets[!, group] .== name
            append!(A, coef * idx)
        elseif throw_if_missing
            throw(ArgumentError("$(string(group)) is not in $(group_names)."))
        end
    end
    tcnst = hs.cnst
    return if isempty(A)
        A, tcnst
    else
        vec(sum(reshape(A, N, :); dims = 2)), tcnst
    end
end
function get_asset_constraint_data(hs::LinearConstraintSide{<:Any, <:Any, <:Real, <:Real},
                                   asset_sets::DataFrame, throw_if_missing::Bool = false)
    group_names = names(asset_sets)
    N = nrow(asset_sets)
    A = Vector{promote_type(eltype(hs.coef), typeof(hs.cnst))}(undef, 0)
    sizehint!(A, N)
    (; group, name, coef) = hs
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = asset_sets[!, group] .== name
        append!(A, coef * idx)
    elseif throw_if_missing
        throw(ArgumentError("$(string(group)) is not in $(group_names)."))
    end
    tcnst = hs.cnst
    return A, tcnst
end
