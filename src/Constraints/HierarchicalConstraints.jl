struct HierarchicalConstraint{T1, T2, T3, T4}
    group::T1
    name::T2
    lo::T3
    hi::T4
end
struct WeightBoundsModel{T1 <: Union{<:Real, <:AbstractVector},
                         T2 <: Union{<:Real, <:AbstractVector}}
    w_min::T1
    w_max::T2
end
function WeightBoundsModel(; w_min::Union{<:Real, <:AbstractVector} = 0.0,
                           w_max::Union{<:Real, <:AbstractVector} = 1.0)
    if isa(w_min, AbstractVector)
        @smart_assert(!isempty(w_min))
    end
    if isa(w_max, AbstractVector)
        @smart_assert(!isempty(w_max))
    end
    @smart_assert(all(w_min .<= w_max))
    return WeightBoundsModel{typeof(w_min), typeof(w_max)}(w_min, w_max)
end
function HierarchicalConstraint(; group, name,
                                lo::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                                hi::Union{<:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    lo_flag = isa(lo, AbstractVector)
    hi_flag = isa(hi, AbstractVector)
    if group_flag || name_flag || lo_flag || hi_flag
        @smart_assert(group_flag && name_flag && lo_flag && hi_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(lo) && !isempty(hi))
        @smart_assert(length(group) == length(name) == length(lo) == length(hi))
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
    w_min = zeros(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    w_max = ones(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    (; group, name, lo, hi) = hcc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        w_min[idx] .= lo
        w_max[idx] .= hi
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
    end
    return WeightBoundsModel(; w_min = w_min, w_max = w_max)
end
function hc_constraints(hcc::HierarchicalConstraint{<:AbstractVector, <:AbstractVector,
                                                    <:AbstractVector, <:AbstractVector},
                        sets::DataFrame; strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    N = nrow(sets)
    w_min = zeros(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    w_max = ones(promote_type(eltype(hcc.lo), eltype(hcc.hi)), N)
    for (group, name, lo, hi) ∈ zip(hcc.group, hcc.name, hcc.lo, hcc.hi)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            w_min[idx] .= lo
            w_max[idx] .= hi
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
        end
    end
    return WeightBoundsModel(; w_min = w_min, w_max = w_max)
end

export HierarchicalConstraint, hc_constraints
