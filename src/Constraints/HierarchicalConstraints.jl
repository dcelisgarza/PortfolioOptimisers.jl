struct HierarchicalConstraint{T1, T2, T3, T4}
    group::T1
    name::T2
    lo::T3
    hi::T4
end
function HierarchicalConstraint(; group = nothing, name = nothing,
                                lo::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                                hi::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    lo_flag = isa(lo, AbstractVector)
    hi_flag = isa(hi, AbstractVector)
    if any((group_flag, name_flag, lo_flag, hi_flag))
        @smart_assert(all((group_flag, name_flag, lo_flag, hi_flag)))
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(lo) && !isempty(hi))
        @smart_assert(length(group) == length(name) == length(lo) == length(hi))
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
    @smart_assert(all(lo .<= hi))
    return HierarchicalConstraint{typeof(group), typeof(name), typeof(lo), typeof(hi)}(group,
                                                                                       name,
                                                                                       lo,
                                                                                       hi)
end
function hc_constraints(hcc::HierarchicalConstraint{<:Any, <:Any, <:Real, <:Real},
                        sets::DataFrame; strict::Bool = false)
    group_names = names(sets)
    N = nrow(sets)
    w1 = zeros(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    w2 = ones(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    (; group, name, lo, hi) = hcc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        w1[idx] .= lo
        w2[idx] .= hi
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
    end
    return w1, w2
end
function hc_constraints(hcc::HierarchicalConstraint{<:AbstractVector, <:AbstractVector,
                                                    <:AbstractVector, <:AbstractVector},
                        sets::DataFrame; strict::Bool = false)
    group_names = names(sets)
    N = nrow(sets)
    w1 = zeros(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    w2 = ones(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    for (group, name, lo, hi) ∈ zip(hcc.group, hcc.name, hcc.lo, hcc.hi)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            w1[idx] .= lo
            w2[idx] .= hi
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
        end
    end
    return w1, w2
end

export HierarchicalConstraint, hc_constraints
