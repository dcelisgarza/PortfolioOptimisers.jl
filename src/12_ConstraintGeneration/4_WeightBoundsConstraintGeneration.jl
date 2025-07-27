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
    @smart_assert(!isempty(lb) && all(x -> x <= ub, lb))
    return nothing
end
function validate_bounds(lb::Real, ub::AbstractVector)
    @smart_assert(!isempty(ub) && all(x -> lb <= x, ub))
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::AbstractVector)
    @smart_assert(!isempty(lb) &&
                  !isempty(ub) &&
                  length(lb) == length(ub) &&
                  all(map((x, y) -> x <= y, lb, ub)))
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
    # @smart_assert(isnothing(lb) ⊼ isnothing(ub))
    validate_bounds(lb, ub)
    return WeightBoundsResult{typeof(lb), typeof(ub)}(lb, ub)
end
struct WeightBoundsConstraint{T1 <: Union{Nothing, <:AbstractDict},
                              T2 <: Union{Nothing, <:AbstractDict}} <: AbstractEstimator
    lb::T1
    ub::T2
end
function WeightBoundsConstraint(; lb::Union{Nothing, <:AbstractDict} = nothing,
                                ub::Union{Nothing, <:AbstractDict} = nothing)
    if isa(lb, AbstractDict)
        @smart_assert(!isempty(lb))
    end
    if isa(ub, AbstractDict)
        @smart_assert(!isempty(ub))
    end
    return WeightBoundsConstraint{typeof(lb), typeof(ub)}(lb, ub)
end
function weight_bounds_view(wb::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:AbstractString},
                                      <:AbstractVector{Expr},
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                      <:WeightBoundsConstraint}, ::Any)
    return wb
end
function get_weight_bounds(sets::AssetSets, ::Nothing, lb::Bool, args...; kwargs...)
    nx = sets.dict[sets.key]
    val = lb ? 0.0 : 1.0
    return range(; start = val, stop = val, length = length(nx))
end
function get_weight_bounds(sets::AssetSets, bounds::AbstractDict, lb::Bool;
                           strict::Bool = false, datatype::DataType = Float64)
    nx = sets.dict[sets.key]
    wb = fill(lb ? zero(datatype) : one(datatype), length(nx))
    for (key, val) in bounds
        if key in nx
            wb[nx[key]] .= val
        else
            assets = get(sets.dict, key, nothing)
            if isnothing(assets)
                if strict
                    throw(ArgumentError("$(key) is not in $(keys(sets.dict)).\n$(bounds)"))
                else
                    @warn("$(key) is not in $(keys(sets.dict)).\n$(bounds)")
                end
            else
                unique!(assets)
                wb[[findfirst(x -> x == asset, nx) for asset in assets]] .= val
            end
        end
    end
    return wb
end
function weight_bounds_constraints(wb::WeightBoundsConstraint, sets::AssetSets;
                                   strict::Bool = false, datatype::DataType = Float64,
                                   kwargs...)
    return WeightBoundsResult(;
                              lb = get_weight_bounds(sets, wb.lb, true; strict = strict,
                                                     datatype = datatype),
                              ub = get_weight_bounds(sets, wb.ub, false; strict = strict,
                                                     datatype = datatype))
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
