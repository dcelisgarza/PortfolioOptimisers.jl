function validate_bounds(lb::Real, ub::Real)
    @argcheck(lb <= ub)
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::Real)
    @argcheck(!isempty(lb) && all(x -> x <= ub, lb))
    return nothing
end
function validate_bounds(lb::Real, ub::AbstractVector)
    @argcheck(!isempty(ub) && all(x -> lb <= x, ub))
    return nothing
end
function validate_bounds(lb::AbstractVector, ub::AbstractVector)
    @argcheck(!isempty(lb) &&
              !isempty(ub) &&
              length(lb) == length(ub) &&
              all(map((x, y) -> x <= y, lb, ub)))
    return nothing
end
function validate_bounds(lb::AbstractVector, ::Any)
    @argcheck(!isempty(lb))
    return nothing
end
function validate_bounds(::Any, ub::AbstractVector)
    @argcheck(!isempty(ub))
    return nothing
end
function validate_bounds(args...)
    return nothing
end
function weight_bounds_view(::Nothing, ::Any)
    return nothing
end
struct WeightBounds{T1, T2} <: AbstractConstraintResult
    lb::T1
    ub::T2
end
function WeightBounds(; lb::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 0.0,
                      ub::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = 1.0)
    validate_bounds(lb, ub)
    return WeightBounds(lb, ub)
end
function weight_bounds_view(wb::WeightBounds, i::AbstractVector)
    lb = nothing_scalar_array_view(wb.lb, i)
    ub = nothing_scalar_array_view(wb.ub, i)
    return WeightBounds(; lb = lb, ub = ub)
end
abstract type CustomWeightBoundsConstraint <: AbstractConstraintEstimator end
struct UniformScaledBounds <: CustomWeightBoundsConstraint end
struct WeightBoundsEstimator{T1, T2} <: AbstractConstraintEstimator
    lb::T1
    ub::T2
end
function WeightBoundsEstimator(;
                               lb::Union{Nothing, <:AbstractDict,
                                         <:Pair{<:AbstractString, <:Real},
                                         <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                         <:CustomWeightBoundsConstraint} = nothing,
                               ub::Union{Nothing, <:AbstractDict,
                                         <:Pair{<:AbstractString, <:Real},
                                         <:AbstractVector{<:Pair{<:AbstractString, <:Real}},
                                         <:CustomWeightBoundsConstraint} = nothing)
    if isa(lb, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(lb), IsEmptyError(non_empty_msg("`lb`") * "."))
    end
    if isa(ub, Union{<:AbstractDict, <:AbstractVector})
        @argcheck(!isempty(ub), IsEmptyError(non_empty_msg("`ub`") * "."))
    end
    return WeightBoundsEstimator(lb, ub)
end
function weight_bounds_view(wb::Union{<:AbstractString, Expr,
                                      <:AbstractVector{<:AbstractString},
                                      <:AbstractVector{Expr},
                                      <:AbstractVector{<:Union{<:AbstractString, Expr}},
                                      <:WeightBoundsEstimator}, ::Any)
    return wb
end
function get_weight_bounds(wb::Union{Nothing, <:Real, <:AbstractVector}, args...; kwargs...)
    return wb
end
function get_weight_bounds(bounds::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                         <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           lub::Bool, sets::AssetSets; strict::Bool = false,
                           datatype::DataType = Float64)
    return estimator_to_val(bounds, sets, ifelse(lub, zero(datatype), one(datatype));
                            strict = strict)
end
function get_weight_bounds(bounds::UniformScaledBounds, lub::Bool, sets::AssetSets;
                           datatype::DataType = Float64, kwargs...)
    return lub ? inv(length(sets.dict[sets.key])) : one(datatype)
end
function weight_bounds_constraints(wb::WeightBoundsEstimator, sets::AssetSets;
                                   strict::Bool = false, datatype::DataType = Float64,
                                   kwargs...)
    return WeightBounds(;
                        lb = get_weight_bounds(wb.lb, true, sets; strict = strict,
                                               datatype = datatype),
                        ub = get_weight_bounds(wb.ub, false, sets; strict = strict,
                                               datatype = datatype))
end
function weight_bounds_constraints_side(::Nothing, N::Integer, val::Real)
    return fill(val, N)
end
function weight_bounds_constraints_side(wb::Real, N::Integer, val::Real)
    return if isinf(wb)
        fill(val, N)
    elseif isa(wb, Real)
        range(; start = wb, stop = wb, length = N)
    end
end
function weight_bounds_constraints_side(wb::AbstractVector, args...)
    return wb
end
function weight_bounds_constraints(wb::WeightBounds{<:Any, <:Any}, args...;
                                   scalar::Bool = false, N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return wb
    end
    return WeightBounds(; lb = weight_bounds_constraints_side(wb.lb, N, -Inf),
                        ub = weight_bounds_constraints_side(wb.ub, N, Inf))
end
function weight_bounds_constraints(wb::WeightBounds{<:AbstractVector, <:AbstractVector},
                                   args...; kwargs...)
    return wb
end
function weight_bounds_constraints(wb::Nothing, args...; scalar::Bool = false,
                                   N::Integer = 0, kwargs...)
    if scalar || iszero(N)
        return WeightBounds(; lb = -Inf, ub = Inf)
    end
    return WeightBounds(; lb = fill(-Inf, N), ub = fill(Inf, N))
end

export WeightBoundsEstimator, WeightBounds, weight_bounds_constraints, UniformScaledBounds
