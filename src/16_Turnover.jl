"""
    struct TurnoverEstimator{T1, T2, T3} <: AbstractEstimator
        w::T1
        val::T2
        default::T3
    end

Estimator for turnover portfolio constraints.

`TurnoverEstimator` specifies turnover constraints for each asset in a portfolio, based on current portfolio weights `w`, asset-specific turnover values `val`, and a default value for assets not explicitly specified. Supports asset-specific turnover via dictionaries, pairs, or vectors of pairs, and validates all inputs for non-emptiness, non-negativity, and finiteness.

# Fields

  - `w`: Vector of current portfolio weights.
  - `val`: Asset-specific turnover values, as a dictionary, pair, or vector of pairs.
  - `default`: Default turnover value for assets not specified in `val`.

# Constructor

    TurnoverEstimator(; w::AbstractVector{<:Real},
                      val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                 <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                      default::Real = 0.0)

## Validation

  - `!isempty(w)`.
  - If `val` is a vector: `!isempty(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
  - If `val` is a scalar: `isfinite(val)` and `val >= 0`.

# Examples

```jldoctest
julia> TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2), default = 0.0)
TurnoverEstimator
        w | Vector{Float64}: [0.2, 0.3, 0.5]
      val | Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  default | Float64: 0.0
```

# Related

  - [`Turnover`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AbstractEstimator`](@ref)
"""
struct TurnoverEstimator{T1, T2, T3} <: AbstractEstimator
    w::T1
    val::T2
    default::T3
    function TurnoverEstimator(w::AbstractVector{<:Real},
                               val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                          <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                               default::Real)
        @argcheck(!isempty(w))
        assert_nonneg_finite_val(val)
        @argcheck(default >= zero(default))
        return new{typeof(w), typeof(val), typeof(default)}(w, val, default)
    end
end
function TurnoverEstimator(; w::AbstractVector{<:Real},
                           val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                      <:AbstractVector{<:Pair{<:AbstractString, <:Real}}},
                           default::Real = 0.0)
    return TurnoverEstimator(w, val, default)
end
"""
    turnover_constraints(tn::TurnoverEstimator, sets::AssetSets; strict::Bool = false)

Generate turnover portfolio constraints from a `TurnoverEstimator` and asset set.

`turnover_constraints` constructs a [`Turnover`](@ref) object representing turnover constraints for the assets in `sets`, using the specifications in `tn`. Supports scalar, vector, dictionary, pair, or custom turnover types for flexible assignment and validation.

# Arguments

  - `tn`: [`TurnoverEstimator`](@ref) specifying current weights, asset-specific turnover values, and default value.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `Turnover`: Object containing portfolio weights and turnover values aligned with `sets`.

# Details

  - Turnover values are extracted and mapped to assets using `estimator_to_val`.
  - If a turnover value is missing for an asset, assigns the default value unless `strict` is `true`.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1, "B" => 0.2), 0.0);

julia> turnover_constraints(tn, sets)
Turnover
    w | Vector{Float64}: [0.2, 0.3, 0.5]
  val | Vector{Float64}: [0.1, 0.2, 0.0]
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function turnover_constraints(tn::TurnoverEstimator, sets::AssetSets; strict::Bool = false)
    return Turnover(; w = tn.w,
                    val = estimator_to_val(tn.val, sets, tn.default; strict = strict))
end
"""
    struct Turnover{T1, T2} <: AbstractResult
        w::T1
        val::T2
    end

Container for turnover portfolio constraints.

`Turnover` stores the portfolio weights and turnover constraint values for each asset. The turnover constraint can be specified as a scalar (applied to all assets) or as a vector of per-asset values. Input validation ensures all weights and turnover values are non-empty, non-negative, and finite.

# Fields

  - `w`: Vector of portfolio weights.
  - `val`: Scalar or vector of turnover constraint values.

# Constructor

    Turnover(; w::AbstractVector{<:Real}, val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)

## Validation

  - `!isempty(w)`.
  - If `val` is a vector: `!isempty(val)`, `length(w) == length(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
  - If `val` is a scalar: `isfinite(val)` and `val >= 0`.

# Examples

```jldoctest
julia> Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0])
Turnover
    w | Vector{Float64}: [0.2, 0.3, 0.5]
  val | Vector{Float64}: [0.1, 0.2, 0.0]

julia> Turnover(; w = [0.2, 0.3, 0.5], val = 0.02)
Turnover
    w | Vector{Float64}: [0.2, 0.3, 0.5]
  val | Float64: 0.02
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref)
  - [`AbstractResult`](@ref)
"""
struct Turnover{T1, T2} <: AbstractResult
    w::T1
    val::T2
    function Turnover(w::AbstractVector{<:Real},
                      val::Union{<:Real, <:AbstractVector{<:Real}})
        @argcheck(!isempty(w), IsEmptyError(non_empty_msg("`w`") * "."))
        if isa(val, AbstractVector)
            @argcheck(!isempty(val) &&
                      length(val) == length(w) &&
                      any(isfinite, val) &&
                      all(x -> x >= zero(x), val),
                      AssertionError("The following conditions must hold:\n`val` must be non-empty => $(!isempty(val))\n`val` must have the same length as `w` => $(length(val) == length(w))\n`val` must be non-negative and finite => $(all(x -> isfinite(x) && x >= zero(x), val))"))
        else
            @argcheck(isfinite(val) && val >= zero(eltype(val)),
                      DomainError("`val` must be non-negative and finite:\nval => $val"))
        end
        return new{typeof(w), typeof(val)}(w, val)
    end
end
function Turnover(; w::AbstractVector{<:Real},
                  val::Union{<:Real, <:AbstractVector{<:Real}} = 0.0)
    return Turnover(w, val)
end
"""
    turnover_constraints(tn::Union{Nothing, <:Turnover}, args...; kwargs...)

Propagate or pass through turnover portfolio constraints.

`turnover_constraints` returns the input [`Turnover`](@ref) object unchanged or `nothing`. This method is used to propagate already constructed turnover constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `tn`: An existing [`Turnover`](@ref) object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `tn::Union{Nothing, Turnover}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0]);

julia> turnover_constraints(tn)
Turnover
    w | Vector{Float64}: [0.2, 0.3, 0.5]
  val | Vector{Float64}: [0.1, 0.2, 0.0]
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`turnover_constraints`](@ref)
"""
function turnover_constraints(tn::Union{Nothing, <:Turnover}, args...; kwargs...)
    return tn
end
"""
    turnover_constraints(tn::Union{<:AbstractVector{<:TurnoverEstimator},
                                   <:AbstractVector{<:Turnover},
                                   <:AbstractVector{<:Union{<:TurnoverEstimator, <:Turnover}}},
                         sets::AssetSets; strict::Bool = false)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.
"""
function turnover_constraints(tn::Union{<:AbstractVector{<:TurnoverEstimator},
                                        <:AbstractVector{<:Turnover},
                                        <:AbstractVector{<:Union{<:TurnoverEstimator,
                                                                 <:Turnover}}},
                              sets::AssetSets; strict::Bool = false)
    return turnover_constraints.(tn, Ref(sets); strict = strict)
end
function turnover_view(::Nothing, ::Any)
    return nothing
end
function turnover_view(tn::TurnoverEstimator, i::AbstractVector)
    w = view(tn.w, i)
    return TurnoverEstimator(; w = w, val = tn.val, default = tn.default)
end
function turnover_view(tn::Turnover, i::AbstractVector)
    w = view(tn.w, i)
    val = nothing_scalar_array_view(tn.val, i)
    return Turnover(; w = w, val = val)
end
function turnover_view(tn::AbstractVector{<:Turnover}, i::AbstractVector)
    return turnover_view.(tn, Ref(i))
end
function factory(tn::Turnover, w::AbstractVector)
    return Turnover(; w = w, val = tn.val)
end

export TurnoverEstimator, Turnover, turnover_constraints
