"""
    struct BuyInThresholdEstimator{T1} <: AbstractConstraintEstimator
        val::T1
    end

Estimator for buy-in threshold portfolio constraints.

`BuyInThresholdEstimator` specifies a minimum allocation threshold for each asset in a portfolio. Only assets with weights above the threshold are allocated nonzero weight. The estimator supports asset-specific thresholds via dictionaries, pairs, or vectors of pairs, and validates input for non-emptiness.

# Fields

  - `val`: Asset-specific threshold values, as a dictionary, pair, or vector of pairs.

# Constructor

    BuyInThresholdEstimator(;
                            val::Union{<:AbstractDict, <:Pair{<:AbstractString, <:Real},
                                       <:AbstractVector{<:Pair{<:AbstractString, <:Real}}})

## Validation

  - If `val` is a `AbstractDict` or `AbstractVector`, `!isempty(val)`.

# Examples

```jldoctest
julia> BuyInThresholdEstimator(Dict("A" => 0.05, "B" => 0.1))
BuyInThresholdEstimator
  val └ Dict{String, Float64}: Dict("B" => 0.1, "A" => 0.05)

julia> BuyInThresholdEstimator(["A" => 0.05, "B" => 0.1])
BuyInThresholdEstimator
  val └ Vector{Pair{String, Float64}}: ["A" => 0.05, "B" => 0.1]
```

# Related

  - [`BuyInThreshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
struct BuyInThresholdEstimator{T1} <: AbstractConstraintEstimator
    val::T1
    function BuyInThresholdEstimator(val::Union{<:AbstractDict,
                                                <:Pair{<:AbstractString, <:Real},
                                                <:AbstractVector{<:Pair{<:AbstractString,
                                                                        <:Real}}})
        if isa(val, Union{<:AbstractDict, <:AbstractVector})
            @argcheck(!isempty(val))
        end
        return new{typeof(val)}(val)
    end
end
function BuyInThresholdEstimator(;
                                 val::Union{<:AbstractDict,
                                            <:Pair{<:AbstractString, <:Real},
                                            <:AbstractVector{<:Pair{<:AbstractString,
                                                                    <:Real}}})
    return BuyInThresholdEstimator(val)
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

    BuyInThreshold(; val::Union{<:Real, <:AbstractVector{<:Real}})

## Validation

  - `val` is validated with [`assert_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> BuyInThreshold(0.05)
BuyInThreshold
  val └ Float64: 0.05

julia> BuyInThreshold([0.05, 0.1, 0.0])
BuyInThreshold
  val └ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
struct BuyInThreshold{T1} <: AbstractConstraintResult
    val::T1
    function BuyInThreshold(val::Union{<:Real, <:AbstractVector{<:Real}})
        assert_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function BuyInThreshold(; val::Union{<:Real, <:AbstractVector{<:Real}})
    return BuyInThreshold(val)
end
function threshold_view(t::Union{Nothing, <:BuyInThresholdEstimator}, ::Any)
    return t
end
function threshold_view(t::BuyInThreshold, i::AbstractVector)
    return BuyInThreshold(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::AbstractVector{<:Union{Nothing, <:BuyInThreshold,
                                                  <:BuyInThresholdEstimator}},
                        i::AbstractVector)
    return [threshold_view(_t, i) for _t in t]
end
"""
    threshold_constraints(t::Union{Nothing, <:BuyInThreshold}, args...; kwargs...)

Propagate or pass through buy-in threshold portfolio constraints.

`threshold_constraints` returns the input [`BuyInThreshold`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed buy-in threshold constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `t`: An existing [`BuyInThreshold`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `bt::Union{Nothing, <:BuyInThreshold}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> threshold_constraints(BuyInThreshold(0.05))
BuyInThreshold
  val └ Float64: 0.05

julia> threshold_constraints(nothing)

```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`BuyInThreshold`](@ref)
  - [`threshold_constraints`](@ref)
"""
function threshold_constraints(t::Union{Nothing, <:BuyInThreshold}, args...; kwargs...)
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
  val └ Vector{Float64}: [0.05, 0.1, 0.0]
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
                          val = estimator_to_val(t.val, sets, zero(datatype);
                                                 strict = strict))
end
"""
    threshold_constraints(bounds::UniformlyDistributedBounds, sets::AssetSets; kwargs...)

Generate uniform buy-in threshold portfolio constraints for all assets.

`threshold_constraints` constructs a [`BuyInThreshold`](@ref) object with a uniform threshold value for each asset in `sets`, using [`UniformlyDistributedBounds`](@ref). The threshold is set to `1/N`, where `N` is the number of assets, ensuring equal minimum allocation across all assets.

# Arguments

  - `bounds`: [`UniformlyDistributedBounds`](@ref) specifying uniform threshold logic.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `BuyInThreshold`: Object with uniform threshold values for all assets.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> threshold_constraints(UniformlyDistributedBounds(), sets)
BuyInThreshold
  val └ Float64: 0.3333333333333333
```

# Related

  - [`BuyInThresholdEstimator`](@ref)
  - [`BuyInThreshold`](@ref)
  - [`UniformlyDistributedBounds`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function threshold_constraints(bounds::UniformlyDistributedBounds, sets::AssetSets;
                               kwargs...)
    return BuyInThreshold(; val = inv(length(sets.dict[sets.key])))
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
    return [threshold_constraints(_t, sets; kwargs...) for _t in t]
end

export BuyInThreshold, BuyInThresholdEstimator, threshold_constraints
