struct WeightBounds{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                    T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}}
    lb::T1
    ub::T2
end
function WeightBounds(; lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                      ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 1.0)
    lb_flag = !isnothing(lb)
    ub_flag = !isnothing(ub)
    @smart_assert(!lb_flag ⊼ !ub_flag)
    if lb_flag && ub_flag
        lb_flag = isa(lb, Real)
        ub_flag = isa(ub, Real)
        if lb_flag
            @smart_assert(isfinite(lb) && lb >= zero(lb))
        else
            @smart_assert(all(isfinite.(lb)) && all(lb .>= zero(lb)))
        end
        if ub_flag
            @smart_assert(isfinite(ub) && ub >= zero(ub))
        else
            @smart_assert(all(isfinite.(ub)) && all(ub .>= zero(ub)))
        end
        if lb_flag && ub_flag
            @smart_assert(lb <= ub)
        elseif !lb_flag && !ub_flag
            @smart_assert(length(lb) == length(ub))
            @smart_assert(all(lb .<= ub))
        else
            @smart_assert(all(lb .<= ub))
        end
    elseif !lb_flag && ub_flag
        if isa(lb, Real)
            @smart_assert(isfinite(lb) && lb >= zero(lb))
        else
            @smart_assert(all(isfinite.(lb)) && all(lb .>= zero(lb)))
        end
    elseif lb_flag && !ub_flag
        if isa(ub, Real)
            @smart_assert(isfinite(ub) && ub >= zero(ub))
        else
            @smart_assert(all(isfinite.(ub)) && all(ub .>= zero(ub)))
        end
    end
    return WeightBounds{typeof(lb), typeof(ub)}(lb, ub)
end
struct WeightBoundsConstraints{T1, T2, T3 <: Union{<:Real, <:AbstractVector{<:Real}},
                               T4 <: Union{<:Real, <:AbstractVector{<:Real}},
                               T5 <: DataFrame}
    group::T1
    name::T2
    lb::T3
    ub::T4
    sets::T5
end
function WeightBoundsConstraints(; group, name,
                                 lb::Union{<:Real, <:AbstractVector{<:Real}} = 0.0,
                                 ub::Union{<:Real, <:AbstractVector{<:Real}} = 1.0,
                                 sets::DataFrame)
    @smart_assert(!isempty(sets))
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    lo_flag = isa(lb, AbstractVector)
    hi_flag = isa(ub, AbstractVector)
    if group_flag || name_flag || lo_flag || hi_flag
        @smart_assert(group_flag && name_flag && lo_flag && hi_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(lb) && !isempty(ub))
        @smart_assert(length(group) == length(name) == length(lb) == length(ub))
    end
    @smart_assert(all(lb .<= ub))
    return WeightBoundsConstraints{typeof(group), typeof(name), typeof(lb), typeof(ub),
                                   typeof(sets)}(group, name, lb, ub, sets)
end
function weight_bounds_constraints(hcc::WeightBoundsConstraints{<:Any, <:Any, <:Real,
                                                                <:Real, <:Any};
                                   strict::Bool = false)
    sets = hcc.sets
    group_names = names(sets)
    N = nrow(sets)
    LB = zeros(promote_type(eltype(hcc.lb), eltype(hcc.ub)), N)
    UB = ones(promote_type(eltype(hcc.lb), eltype(hcc.ub)), N)
    (; group, name, lb, ub) = hcc
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        LB[idx] .= lb
        UB[idx] .= ub
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
    end
    return WeightBounds(; lb = LB, ub = UB)
end
function weight_bounds_constraints(hcc::WeightBoundsConstraints{<:AbstractVector,
                                                                <:AbstractVector,
                                                                <:AbstractVector,
                                                                <:AbstractVector, <:Any};
                                   strict::Bool = false, kwargs...)
    sets = hcc.sets
    group_names = names(sets)
    N = nrow(sets)
    LB = zeros(promote_type(eltype(hcc.lb), eltype(hcc.ub)), N)
    UB = ones(promote_type(eltype(hcc.lb), eltype(hcc.ub)), N)
    for (group, name, lb, ub) ∈ zip(hcc.group, hcc.name, hcc.lb, hcc.ub)
        if !(isnothing(group) || string(group) ∉ group_names)
            idx = sets[!, group] .== name
            LB[idx] .= lb
            UB[idx] .= ub
        elseif strict
            throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
        else
            @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
        end
    end
    return WeightBounds(; lb = LB, ub = UB)
end
function weight_bounds_constraints(wb::WeightBounds; N::Integer, kwargs...)
    lb_flag = isa(wb.lb, Real)
    ub_flag = isa(wb.ub, Real)
    return if lb_flag || ub_flag
        lb = if lb_flag
            range(; start = wb.lb, stop = wb.lb, length = N)
        else
            wb.lb
        end
        ub = if ub_flag
            range(; start = wb.ub, stop = wb.ub, length = N)
        else
            wb.ub
        end
        WeightBounds(; lb = lb, ub = ub)
    else
        wb
    end
end
function weight_bounds_constraints(wb::Nothing; N::Integer, kwargs...)
    return WeightBounds(; lb = range(; start = 0, stop = 0, length = N),
                        ub = range(; start = 1, stop = 1, length = N))
end

export WeightBoundsConstraints, weight_bounds_constraints
