"""
    struct BuyInThresholdEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Estimator for buy-in threshold portfolio constraints.

`BuyInThresholdEstimator` specifies a minimum allocation threshold for each asset in a portfolio. Only assets with weights above the threshold are allocated nonzero weight. The estimator supports asset-specific thresholds via dictionaries, pairs, or vectors of pairs, and validates input for non-emptiness.

# Fields

  - `val`: Asset-specific threshold values, as a dictionary, pair, or vector of pairs.

# Constructor

    BuyInThresholdEstimator(; val::EstValType)

## Validation

  - If `val` is a `AbstractDict` or `AbstractVector`, `!isempty(val)`.

# Examples

```jldoctest
julia> BuyInThresholdEstimator(Dict("A" => 0.05, "B" => 0.1))
BuyInThresholdEstimator
   val ┼ Dict{String, Float64}: Dict("B" => 0.1, "A" => 0.05)
  dval ┴ nothing

julia> BuyInThresholdEstimator(["A" => 0.05, "B" => 0.1])
BuyInThresholdEstimator
   val ┼ Vector{Pair{String, Float64}}: ["A" => 0.05, "B" => 0.1]
  dval ┴ nothing

julia> BuyInThresholdEstimator("A" => 0.05)
BuyInThresholdEstimator
   val ┼ Pair{String, Float64}: "A" => 0.05
  dval ┴ nothing

julia> BuyInThresholdEstimator(0.05)
BuyInThresholdEstimator
   val ┼ Float64: 0.05
  dval ┴ nothing

julia> BuyInThresholdEstimator(UniformlyDistributedBounds())
BuyInThresholdEstimator
   val ┼ UniformlyDistributedBounds()
  dval ┴ nothing
```

# Related

  - [`BuyInThreshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
struct BuyInThresholdEstimator{T1, T2} <: AbstractConstraintEstimator
    val::T1
    dval::T2
    function BuyInThresholdEstimator(val::Union{<:EstValType,
                                                <:CustomWeightBoundsConstraint},
                                     dval::Option{<:Number} = nothing)
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        return new{typeof(val), typeof(dval)}(val, dval)
    end
end
function BuyInThresholdEstimator(; val::Union{<:EstValType, <:CustomWeightBoundsConstraint},
                                 dval::Option{<:Number} = nothing)
    return BuyInThresholdEstimator(val, dval)
end
"""
    struct BuyInThreshold{T1} <: AbstractConstraintResult
        val::T1
    end

Container for buy-in threshold portfolio constraints.

`BuyInThreshold` stores the minimum allocation threshold(s) for assets in a portfolio. The threshold can be specified as a scalar (applied to all assets) or as a vector of per-asset values. Input validation ensures all thresholds are finite and non-negative.

# Fields

  - `val`: Scalar or vector of threshold values for portfolio weights.

# Constructor

    BuyInThreshold(; val::UNumNumVec)

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> BuyInThreshold(0.05)
BuyInThreshold
  val ┴ Float64: 0.05

julia> BuyInThreshold([0.05, 0.1, 0.0])
BuyInThreshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
struct BuyInThreshold{T1} <: AbstractConstraintResult
    val::T1
    function BuyInThreshold(val::UNumNumVec)
        assert_nonempty_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function BuyInThreshold(; val::UNumNumVec)
    return BuyInThreshold(val)
end
function threshold_view(::Nothing, ::Any)
    return nothing
end
function threshold_view(t::BuyInThresholdEstimator, i)
    return BuyInThresholdEstimator(; val = nothing_scalar_array_view(t.val, i),
                                   dval = t.dval)
end
function threshold_view(t::BuyInThreshold, i)
    return BuyInThreshold(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                  <:BuyInThresholdEstimator}}, i)
    return [threshold_view(ti, i) for ti in t]
end
"""
    threshold_constraints(t::Option{<:BuyInThreshold}, args...; kwargs...)

Propagate or pass through buy-in threshold portfolio constraints.

`threshold_constraints` returns the input [`BuyInThreshold`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed buy-in threshold constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `t`: An existing [`BuyInThreshold`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `bt::Option{<:BuyInThreshold}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> threshold_constraints(BuyInThreshold(0.05))
BuyInThreshold
  val ┴ Float64: 0.05

julia> threshold_constraints(nothing)

```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`BuyInThreshold`](@ref)
  - [`threshold_constraints`](@ref)
"""
function threshold_constraints(t::Option{<:BuyInThreshold}, args...; kwargs...)
    return t
end
"""
    threshold_constraints(t::BuyInThresholdEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)

Generate buy-in threshold portfolio constraints from a `BuyInThresholdEstimator` and asset set.

`threshold_constraints` constructs a [`BuyInThreshold`](@ref) object representing minimum allocation thresholds for the assets in `sets`, using the specifications in `t`. Supports scalar, vector, dictionary, pair, or custom threshold types for flexible assignment and validation.

# Arguments

  - `t`: [`BuyInThresholdEstimator`](@ref) specifying asset-specific threshold values.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for thresholds.
  - `strict`: If `true`, enforces strict matching between assets and thresholds (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `bt::BuyInThreshold`: Object containing threshold values aligned with `sets`.

# Details

  - Thresholds are extracted and mapped to assets using `estimator_to_val`.
  - If a threshold is missing for an asset, assigns zero (no threshold) unless `strict` is `true`.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> t = BuyInThresholdEstimator(Dict("A" => 0.05, "B" => 0.1));

julia> threshold_constraints(t, sets)
BuyInThreshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`BuyInThreshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function threshold_constraints(t::BuyInThresholdEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return BuyInThreshold(;
                          val = estimator_to_val(t.val, sets, t.dval; datatype = datatype,
                                                 strict = strict))
end
"""
    threshold_constraints(t::AbstractVector{<:Union{Nothing, <:BuyInThresholdEstimator,
                                                    <:BuyInThreshold}}, sets::AssetSets;
                          kwargs...)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.
"""
function threshold_constraints(t::AbstractVector{<:Union{Nothing, <:BuyInThresholdEstimator,
                                                         <:BuyInThreshold}},
                               sets::AssetSets; kwargs...)
    return [threshold_constraints(ti, sets; kwargs...) for ti in t]
end

export BuyInThreshold, BuyInThresholdEstimator, threshold_constraints
