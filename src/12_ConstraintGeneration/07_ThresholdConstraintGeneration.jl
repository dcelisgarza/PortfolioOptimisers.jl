"""
    struct ThresholdEstimator{T1, T2, T3} <: AbstractConstraintEstimator
        val::T1
        key::T2
        dval::T3
    end

Estimator for buy-in threshold portfolio constraints.

`ThresholdEstimator` specifies a minimum allocation threshold for each asset in a portfolio. Only assets with weights above the threshold are allocated nonzero weight. The estimator supports asset-specific thresholds via dictionaries, pairs, or vectors of pairs, and validates input for non-emptiness.

# Fields

  - `val`: Asset-specific threshold values, as a dictionary, pair, or vector of pairs.
  - `key`: (Optional) Key in the [`AssetSets`](@ref) to specify the asset universe for constraint generation. When provided, takes precedence over `key` field of [`AssetSets`](@ref).
  - `dval`: Default threshold value applied to assets not explicitly specified in `val`.

# Constructor

    ThresholdEstimator(; val::EstValType, dval::Option{<:Number} = nothing,
                             key::Option{<:AbstractString} = nothing)

## Validation

  - If `val` is a `AbstractDict` or `AbstractVector`, `!isempty(val)`.
  - If `dval` is not `nothing`, it is validated with [`assert_nonempty_nonneg_finite_val`](@ref).
  - If `key` is not `nothing`, it is a non-empty string.

# Examples

```jldoctest
julia> ThresholdEstimator(; val = Dict("A" => 0.05, "B" => 0.1))
ThresholdEstimator
   val ┼ Dict{String, Float64}: Dict("B" => 0.1, "A" => 0.05)
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = "A" => 0.05)
ThresholdEstimator
   val ┼ Pair{String, Float64}: "A" => 0.05
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = 0.05)
ThresholdEstimator
   val ┼ Float64: 0.05
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = [0.05])
ThresholdEstimator
   val ┼ Vector{Float64}: [0.05]
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = UniformValues())
ThresholdEstimator
   val ┼ UniformValues()
   key ┼ nothing
  dval ┴ nothing
```

# Related

  - [`Threshold`](@ref)
  - [`EstValType`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintEstimator`](@ref)
"""
struct ThresholdEstimator{T1, T2, T3} <: AbstractConstraintEstimator
    val::T1
    key::T2
    dval::T3
    function ThresholdEstimator(val::EstValType, key::Option{<:AbstractString} = nothing,
                                dval::Option{<:Number} = nothing)
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        if !isnothing(key)
            @argcheck(!isempty(key))
        end
        return new{typeof(val), typeof(key), typeof(dval)}(val, key, dval)
    end
end
function ThresholdEstimator(; val::EstValType, key::Option{<:AbstractString} = nothing,
                            dval::Option{<:Number} = nothing)
    return ThresholdEstimator(val, key, dval)
end
"""
    struct Threshold{T1} <: AbstractConstraintResult
        val::T1
    end

Container for buy-in threshold portfolio constraints.

`Threshold` stores the minimum allocation threshold(s) for assets in a portfolio. The threshold can be specified as a scalar (applied to all assets) or as a vector of per-asset values. Input validation ensures all thresholds are finite and non-negative.

# Fields

  - `val`: Scalar or vector of threshold values for portfolio weights.

# Constructor

    Threshold(; val::Num_VecNum)

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> Threshold(0.05)
Threshold
  val ┴ Float64: 0.05

julia> Threshold([0.05, 0.1, 0.0])
Threshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
"""
struct Threshold{T1} <: AbstractConstraintResult
    val::T1
    function Threshold(val::Num_VecNum)
        assert_nonempty_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function Threshold(; val::Num_VecNum)
    return Threshold(val)
end
const BtE_Bt = Union{<:Threshold, <:ThresholdEstimator}
const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}
const BtE_Bt_VecOptBtE_Bt = Union{<:BtE_Bt, <:VecOptBtE_Bt}
const VecOptBt = AbstractVector{<:Option{<:Threshold}}
const Bt_VecOptBt = Union{<:Threshold, <:VecOptBt}
function threshold_view(::Nothing, ::Any)
    return nothing
end
function threshold_view(t::ThresholdEstimator, i)
    return ThresholdEstimator(; val = nothing_scalar_array_view(t.val, i), dval = t.dval,
                              key = t.key)
end
function threshold_view(t::Threshold, i)
    return Threshold(; val = nothing_scalar_array_view(t.val, i))
end
function threshold_view(t::VecOptBtE_Bt, i)
    return [threshold_view(ti, i) for ti in t]
end
"""
    threshold_constraints(t::Option{<:Threshold}, args...; kwargs...)

Propagate or pass through buy-in threshold portfolio constraints.

`threshold_constraints` returns the input [`Threshold`](@ref) object or `nothing` unchanged. This method is used to propagate already constructed buy-in threshold constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `t`: An existing [`Threshold`](@ref) object or `nothing`.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `bt::Option{<:Threshold}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> threshold_constraints(Threshold(0.05))
Threshold
  val ┴ Float64: 0.05

julia> threshold_constraints(nothing)

```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`Threshold`](@ref)
  - [`threshold_constraints`](@ref)
"""
function threshold_constraints(t::Option{<:Threshold}, args...; kwargs...)
    return t
end
"""
    threshold_constraints(t::ThresholdEstimator, sets::AssetSets;
                          datatype::DataType = Float64, strict::Bool = false)

Generate buy-in threshold portfolio constraints from a `ThresholdEstimator` and asset set.

`threshold_constraints` constructs a [`Threshold`](@ref) object representing minimum allocation thresholds for the assets in `sets`, using the specifications in `t`. Supports scalar, vector, dictionary, pair, or custom threshold types for flexible assignment and validation.

# Arguments

  - `t`: [`ThresholdEstimator`](@ref) specifying asset-specific threshold values.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for thresholds.
  - `strict`: If `true`, enforces strict matching between assets and thresholds (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `bt::Threshold`: Object containing threshold values aligned with `sets`.

# Details

  - Thresholds are extracted and mapped to assets using `estimator_to_val`.
  - If a threshold is missing for an asset, assigns zero (no threshold) unless `strict` is `true`.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> t = ThresholdEstimator(Dict("A" => 0.05, "B" => 0.1));

julia> threshold_constraints(t, sets)
Threshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`Threshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AssetSets`](@ref)
"""
function threshold_constraints(t::ThresholdEstimator, sets::AssetSets;
                               datatype::DataType = Float64, strict::Bool = false)
    return Threshold(;
                     val = estimator_to_val(t.val, sets, t.dval, t.key; datatype = datatype,
                                            strict = strict))
end
"""
    threshold_constraints(t::VecOptBtE_Bt, sets::AssetSets;
                          kwargs...)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.
"""
function threshold_constraints(t::VecOptBtE_Bt, sets::AssetSets; kwargs...)
    return [threshold_constraints(ti, sets; kwargs...) for ti in t]
end

export Threshold, ThresholdEstimator, threshold_constraints
