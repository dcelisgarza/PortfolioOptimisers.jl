struct WeightBoundsResult{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractResult
    lb::T1
    ub::T2
end
function weight_bounds_view(::Nothing, ::Any)
    return nothing
end
function weight_bounds_view(wb::WeightBoundsResult, i::AbstractVector)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBoundsResult(; lb = lb, ub = ub)
end
function validate_bounds(lb::Real, ub::Real)
    @smart_assert(lb <= ub)
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::Real)
    @smart_assert(!isempty(lb) && all(lb .<= ub))
    return nothing
end
function validate_bounds(lb::Real, ub::AbstractVector)
    @smart_assert(!isempty(ub) && all(lb .<= ub))
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::AbstractVector)
    @smart_assert(!isempty(lb) &&
                  !isempty(ub) &&
                  length(lb) == length(ub) &&
                  all(lb .<= ub))
    return nothing
end
function validate_bounds(lb::AbstractVector, ::Any)
    @smart_assert(!isempty(lb))
    return nothing
end
function validate_bounds(::Any, ub::AbstractVector)
    @smart_assert(!isempty(ub))
    return nothing
end
function validate_bounds(args...)
    return nothing
end
function WeightBoundsResult(; lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                            ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 1.0)
    @smart_assert(isnothing(lb) ⊼ isnothing(ub))
    validate_bounds(lb, ub)
    return WeightBoundsResult{typeof(lb), typeof(ub)}(lb, ub)
end
struct WeightBoundsConstraint{T1, T2,
                              T3 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                              T4 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractEstimator
    group::T1
    name::T2
    lb::T3
    ub::T4
end
function WeightBoundsConstraint(; group, name,
                                lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                                ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 1.0)
    group_flag = isa(group, AbstractVector)
    name_flag = isa(name, AbstractVector)
    lb_flag = isa(lb, AbstractVector)
    ub_flag = isa(ub, AbstractVector)
    if group_flag || name_flag || lb_flag || ub_flag
        @smart_assert(group_flag && name_flag)
        @smart_assert(!isempty(group) && !isempty(name) && !isempty(lb) && !isempty(ub))
        @smart_assert(length(group) == length(name) == length(lb) == length(ub))
    end
    validate_bounds(lb, ub)
    return WeightBoundsConstraint{typeof(group), typeof(name), typeof(lb), typeof(ub)}(group,
                                                                                       name,
                                                                                       lb,
                                                                                       ub)
end
function weight_bounds_view(wb::WeightBoundsConstraint{<:Any, <:Any,
                                                       <:Union{Nothing, <:Real},
                                                       <:Union{Nothing, <:Real}}, ::Any)
    return wb
end
function weight_bounds_view(wb::WeightBoundsConstraint{<:AbstractVector, <:AbstractVector,
                                                       <:AbstractVector, <:AbstractVector},
                            i::AbstractVector)
    group = nothing_scalar_array_view(wb.group, i)
    name = nothing_scalar_array_view(wb.group, i)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBoundsConstraint(; group = group, name = name, lb = lb, ub = ub)
end
function weight_bounds_constraints(hcc::WeightBoundsConstraint{<:Any, <:Any,
                                                               <:Union{Nothing, <:Real},
                                                               <:Union{Nothing, <:Real}},
                                   sets::DataFrame; strict::Bool = false)
    @smart_assert(!isempty(sets))
    group_names = names(sets)
    N = nrow(sets)
    lbtype = !isnothing(hcc.lb) ? eltype(hcc.lb) : Float64
    ubtype = !isnothing(hcc.ub) ? eltype(hcc.ub) : Float64
    LB = zeros(promote_type(lbtype, ubtype), N)
    UB = ones(promote_type(lbtype, ubtype), N)
    (; group, name, lb, ub) = hcc
    if isnothing(lb)
        lb = -Inf
    end
    if isnothing(ub)
        ub = Inf
    end
    if !(isnothing(group) || string(group) ∉ group_names)
        idx = sets[!, group] .== name
        LB[idx] .= lb
        UB[idx] .= ub
    elseif strict
        throw(ArgumentError("$(string(group)) is not in $(group_names).\n$(hcc)"))
    else
        @warn("$(string(group)) is not in $(group_names).\n$(hcc)")
    end
    return WeightBoundsResult(; lb = LB, ub = UB)
end
function weight_bounds_constraints(hcc::WeightBoundsConstraint{<:AbstractVector,
                                                               <:AbstractVector,
                                                               <:AbstractVector,
                                                               <:AbstractVector},
                                   sets::DataFrame; strict::Bool = false, kwargs...)
    @smart_assert(!isempty(sets))
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
    return WeightBoundsResult(; lb = LB, ub = UB)
end
function weight_bounds_constraints(wb::WeightBoundsResult{<:Any, <:Any}, args...;
                                   scalar::Bool = false, N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return wb
    end
    lb = wb.lb
    ub = wb.ub
    if isnothing(lb)
        lb = fill(-Inf, N)
    elseif isa(lb, Real)
        lb = range(; start = lb, stop = lb, length = N)
    end
    if isnothing(ub)
        ub = fill(Inf, N)
    elseif isa(ub, Real)
        ub = range(; start = ub, stop = ub, length = N)
    end
    return WeightBoundsResult(; lb = lb, ub = ub)
end
function weight_bounds_constraints(wb::WeightBoundsResult{<:AbstractVector,
                                                          <:AbstractVector}, args...;
                                   kwargs...)
    return wb
end
function weight_bounds_constraints(wb::Nothing, args...; scalar::Bool = false,
                                   N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return WeightBoundsResult(; lb = -Inf, ub = Inf)
    end
    return WeightBoundsResult(; lb = fill(-Inf, N), ub = fill(Inf, N))
end

export WeightBoundsConstraint, WeightBoundsResult, weight_bounds_constraints
