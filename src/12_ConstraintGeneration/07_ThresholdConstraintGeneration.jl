"""
$(DocStringExtensions.TYPEDEF)

Estimator for buy-in threshold portfolio constraints.

`ThresholdEstimator` specifies a minimum allocation threshold for each asset in a portfolio. Only assets with weights above the threshold are allocated nonzero weight. The estimator supports asset-specific thresholds via dictionaries, pairs, or vectors of pairs, and validates input for non-emptiness.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdEstimator(;
        val::EstValType,
        dval::Option{<:Number} = nothing,
        key::Option{<:AbstractString} = nothing
    ) -> ThresholdEstimator

## Validation

  - If `val` is a `AbstractDict` or `AbstractVector`, `!isempty(val)`.
  - If `dval` is not `nothing`, it is validated with [`assert_nonempty_nonneg_finite_val`](@ref).
  - If `key` is not `nothing`, it is a non-empty string.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

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
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct ThresholdEstimator <: AbstractConstraintEstimator
    """
    $(field_dict[:thr_val])
    """
    @vprop val
    """
    $(field_dict[:ekey])
    """
    key
    """
    $(field_dict[:dval])
    """
    dval
    function ThresholdEstimator(val::EstValType, key::Option{<:AbstractString} = nothing,
                                dval::Option{<:Number} = nothing)::ThresholdEstimator
        assert_nonempty_nonneg_finite_val(val, :val)
        assert_nonempty_nonneg_finite_val(dval, :dval)
        if !isnothing(key)
            @argcheck(!isempty(key))
        end
        return new{typeof(val), typeof(key), typeof(dval)}(val, key, dval)
    end
end
function ThresholdEstimator(; val::EstValType, key::Option{<:AbstractString} = nothing,
                            dval::Option{<:Number} = nothing)::ThresholdEstimator
    return ThresholdEstimator(val, key, dval)
end
"""
$(DocStringExtensions.TYPEDEF)

Container for buy-in threshold portfolio constraints.

`Threshold` stores the minimum allocation threshold(s) for assets in a portfolio. The threshold can be specified as a scalar (applied to all assets) or as a vector of per-asset values. Input validation ensures all thresholds are finite and non-negative.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Threshold(;
        val::Num_VecNum
    ) -> Threshold

## Validation

  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

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

  - [`short_mip_threshold_constraints`](@ref)
  - [`short_smip_threshold_constraints`](@ref)
  - [`mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct Threshold <: AbstractConstraintResult
    """
    $(field_dict[:thr_res_val])
    """
    @vprop val
    function Threshold(val::Num_VecNum)::Threshold
        assert_nonempty_nonneg_finite_val(val)
        return new{typeof(val)}(val)
    end
end
function Threshold(; val::Num_VecNum)::Threshold
    return Threshold(val)
end
"""
    const BtE_Bt = Union{<:Threshold, <:ThresholdEstimator}

Alias for a threshold constraint result or estimator.

Matches either a [`Threshold`](@ref) result or a [`ThresholdEstimator`](@ref). Used internally for dispatch in threshold constraint generation.

# Related

  - [`Threshold`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
"""
const BtE_Bt = Union{<:Threshold, <:ThresholdEstimator}
"""
    const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}

Alias for a vector of optional threshold estimators or results.

Represents a collection of optional [`BtE_Bt`](@ref) elements (threshold estimators or results, or `nothing`).

# Related

  - [`BtE_Bt`](@ref)
  - [`BtE_Bt_VecOptBtE_Bt`](@ref)
"""
const VecOptBtE_Bt = AbstractVector{<:Option{<:BtE_Bt}}
"""
    const BtE_Bt_VecOptBtE_Bt = Union{<:BtE_Bt, <:VecOptBtE_Bt}

Alias for a single or vector of optional threshold estimators or results.

Matches either a single [`BtE_Bt`](@ref) or a vector of optional ones.

# Related

  - [`BtE_Bt`](@ref)
  - [`VecOptBtE_Bt`](@ref)
"""
const BtE_Bt_VecOptBtE_Bt = Union{<:BtE_Bt, <:VecOptBtE_Bt}
"""
    const VecOptBt = AbstractVector{<:Option{<:Threshold}}

Alias for a vector of optional threshold results.

Represents a collection of optional [`Threshold`](@ref) elements.

# Related

  - [`Threshold`](@ref)
  - [`Bt_VecOptBt`](@ref)
"""
const VecOptBt = AbstractVector{<:Option{<:Threshold}}
"""
    const Bt_VecOptBt = Union{<:Threshold, <:VecOptBt}

Alias for a single threshold result or a vector of optional threshold results.

Matches either a single [`Threshold`](@ref) or a vector of optional [`Threshold`](@ref) objects.

# Related

  - [`Threshold`](@ref)
  - [`VecOptBt`](@ref)
"""
const Bt_VecOptBt = Union{<:Threshold, <:VecOptBt}
function port_opt_view(t::VecOptBtE_Bt, i, args...)
    return [port_opt_view(ti, i) for ti in t]
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
function threshold_constraints(t::Option{<:Threshold}, args...;
                               kwargs...)::Option{<:Threshold}
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
                               datatype::DataType = Float64,
                               strict::Bool = false)::Threshold
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
