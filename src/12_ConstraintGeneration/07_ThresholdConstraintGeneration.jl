"""
$(DocStringExtensions.TYPEDEF)

Resolves a minimum-holding threshold written in asset or group names against a universe.

[`threshold_constraints`](@ref) turns it into a [`Threshold`](@ref): every name is mapped to its indices in the universe `key` selects, and an unnamed asset takes `dval`, which defaults to no threshold. A threshold may also be a scalar, a vector, or an algorithmic rule such as [`UniformValues`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ThresholdEstimator(;
        val::EstValType,
        key::Option{<:AbstractString} = nothing,
        dval::Option{<:Number} = nothing
    ) -> ThresholdEstimator

Keywords correspond to the struct's fields.

## Validation

  - `val` and `dval` are both validated with [`assert_nonempty_nonneg_finite_val`](@ref), so a threshold is non-empty, non-negative and finite.
  - If `key` is not `nothing`, it is a non-empty string.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

Only a vector `val` is sliced. A `val` that is a scalar, a `Dict`, a `Pair` or an algorithmic rule is not indexed by asset, so a view passes it through untouched and it resolves against the viewed universe when the estimator runs.

# Examples

```jldoctest
julia> ThresholdEstimator(; val = Dict(\"A\" => 0.05, \"B\" => 0.1))
ThresholdEstimator
   val ┼ Dict{String, Float64}: Dict("B" => 0.1, "A" => 0.05)
   key ┼ nothing
  dval ┴ nothing

julia> ThresholdEstimator(; val = \"A\" => 0.05)
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
  - [`UniverseSets`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.4.
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
            @argcheck(!isempty(key), IsEmptyError("key cannot be empty"))
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

Forces every held position to reach a minimum size, and drives anything smaller to zero.

The threshold is a scalar shared by every asset or a vector of one value per asset. It exists to keep a mixed-integer model from answering with a long tail of positions too small to trade.

# Mathematical definition

The threshold is the lower half of a buy-in constraint, stated against the held binary:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\mathrm{opt}}\\quad & \\phi(\\boldsymbol{w})\\\\
\\textrm{s.t.}\\quad & \\ell_i z_i \\leq w_i \\leq u_i z_i\\,,\\quad \\forall i = 1,\\ldots,N\\,,\\\\
& \\boldsymbol{z} \\in \\{0, 1\\}^{N}\\,,\\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{z}``: Held binary, one entry per asset.
  - ``\\ell_i``: Minimum-holding threshold for asset ``i``, the `val` field.
  - ``u_i``: Upper weight bound for asset ``i``, from [`WeightBounds`](@ref).
  - $(math_dict[:N])
  - ``\\phi``: Objective function of the optimiser.
  - ``\\mathcal{W}``: Rest of the feasible set.

The binary carries both halves. Where ``z_i = 0`` the two bounds collapse to ``w_i = 0``; where ``z_i = 1`` the position must reach ``\\ell_i``. A long and a short threshold are separate objects, each bound to its own side's binary.

The threshold binds the **held** weight, not the trade. The source writes the same constraint over positive and negative trades against a reference portfolio; this library writes it over the position, so no reference portfolio enters it. On a six-asset conditional-value-at-risk model at `Threshold(0.15)` every held weight came back at or above `0.15`, with the smallest exactly `0.15`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Threshold(
        val::Num_VecNum
    ) -> Threshold
    Threshold(;
        val::Num_VecNum
    ) -> Threshold

Keywords correspond to the struct's fields.

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
  - [`mip_constraints`](@ref)
  - [`set_mip_constraints!`](@ref)
  - [`ThresholdEstimator`](@ref)
  - [`threshold_constraints`](@ref)
  - [`AbstractConstraintResult`](@ref)
  - [`WeightBounds`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.4.
"""
@propagatable @concrete struct Threshold <: AbstractConstraintResult
    """
    $(field_dict[:thr_res_val])
    """
    @vprop val
    function Threshold(val::Num_VecNum)::Threshold
        assert_nonempty_nonneg_finite_val(val, :val)
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
    threshold_constraints(t::ThresholdEstimator, sets::UniverseSets;
                          datatype::DataType = Float64, strict::Bool = false)

Generate buy-in threshold portfolio constraints from a `ThresholdEstimator` and asset set.

`threshold_constraints` constructs a [`Threshold`](@ref) object representing minimum allocation thresholds for the assets in `sets`, using the specifications in `t`. Supports scalar, vector, dictionary, pair, or custom threshold types for flexible assignment and validation.

# Arguments

  - `t`: [`ThresholdEstimator`](@ref) specifying asset-specific threshold values.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `datatype`: Output data type for thresholds.
  - `strict`: If `true`, a name in `t.val` that `sets` does not resolve throws; if `false`, it issues a warning and is skipped.

# Returns

  - `bt::Threshold`: Object containing threshold values aligned with `sets`.

# Details

  - Thresholds are extracted and mapped to assets by [`estimator_to_val`](@ref), against the universe `t.key` selects.
  - An asset that `t.val` does not name takes `t.dval`. The default `dval = nothing` gives it `zero(datatype)`, which is no threshold.
  - `strict` governs unresolvable **names**, not unnamed assets. An unnamed asset always takes the default and never throws.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> t = ThresholdEstimator(Dict(\"A\" => 0.05, \"B\" => 0.1));

julia> threshold_constraints(t, sets)
Threshold
  val ┴ Vector{Float64}: [0.05, 0.1, 0.0]
```

# Related

  - [`ThresholdEstimator`](@ref)
  - [`Threshold`](@ref)
  - [`threshold_constraints`](@ref)
  - [`UniverseSets`](@ref)
"""
function threshold_constraints(t::ThresholdEstimator, sets::UniverseSets;
                               datatype::DataType = Float64,
                               strict::Bool = false)::Threshold
    return Threshold(;
                     val = estimator_to_val(t.val, sets, t.dval, t.key; datatype = datatype,
                                            strict = strict))
end
"""
    threshold_constraints(t::VecOptBtE_Bt, sets::UniverseSets;
                          kwargs...)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.
"""
function threshold_constraints(t::VecOptBtE_Bt, sets::UniverseSets; kwargs...)
    return [threshold_constraints(ti, sets; kwargs...) for ti in t]
end

export Threshold, ThresholdEstimator, threshold_constraints
