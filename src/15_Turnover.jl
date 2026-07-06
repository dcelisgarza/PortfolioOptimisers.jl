"""
$(DocStringExtensions.TYPEDEF)

Estimator for turnover portfolio constraints.

`TurnoverEstimator` specifies turnover constraints for each asset in a portfolio, based on current portfolio weights `w`, asset-specific turnover values `val`, and a default value for assets not explicitly specified. Supports asset-specific turnover via dictionaries, pairs, or vectors of pairs.

This estimator can be converted into a concrete [`Turnover`](@ref) constraint using the [`turnover_constraints`](@ref) function, which maps the estimator's specifications to the assets in a given [`AssetSets`](@ref) object.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TurnoverEstimator(;
        w::VecNum,
        val::EstValType,
        dval::Option{<:Number} = nothing,
        fixed::Bool = false
    ) -> TurnoverEstimator

## Validation

  - `w` is validated with [`assert_nonempty_finite_val`](@ref).
  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).
  - `dval`, if not `nothing`, `dval >= 0`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2), dval = 0.0)
TurnoverEstimator
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: false
```

# Related

  - [`Turnover`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`EstValType`](@ref)
  - [`Option`](@ref)
  - [`turnover_constraints`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct TurnoverEstimator <: AbstractEstimator
    """
    $(field_dict[:w_tn])
    """
    @vprop w
    """
    $(field_dict[:val])
    """
    @vprop val
    """
    $(field_dict[:dval])
    """
    dval
    """
    $(field_dict[:fixed])
    """
    fixed
    function TurnoverEstimator(w::VecNum, val::EstValType, dval::Option{<:Number},
                               fixed::Bool)::TurnoverEstimator
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(val, :val)
        if !isnothing(dval)
            @argcheck(zero(dval) <= dval, DomainError)
        end
        return new{typeof(w), typeof(val), typeof(dval), typeof(fixed)}(w, val, dval, fixed)
    end
end
function TurnoverEstimator(; w::VecNum, val::EstValType, dval::Option{<:Number} = nothing,
                           fixed::Bool = false)::TurnoverEstimator
    return TurnoverEstimator(w, val, dval, fixed)
end
"""
    factory(tn::TurnoverEstimator, w::VecNum)

Create a new `TurnoverEstimator` with updated portfolio weights.

Constructs a new [`TurnoverEstimator`](@ref) object using the provided portfolio weights `w` and the turnover values and default value from an existing `TurnoverEstimator` `tn`.

# Arguments

  - `tn`: Existing `TurnoverEstimator` object. Supplies turnover values and default value.
  - `w`: New portfolio weights vector.

# Returns

  - `tn::TurnoverEstimator`: New estimator object with the same values and default but updated weights.

# Validation

  - `w` is validated to be non-empty, finite, and numeric.

# Examples

```jldoctest
julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                              dval = 0.0)
TurnoverEstimator
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: false

julia> factory(tn, [0.1, 0.4, 0.5])
TurnoverEstimator
      w ┼ Vector{Float64}: [0.1, 0.4, 0.5]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: false

julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                              dval = 0.0, fixed = true)
TurnoverEstimator
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: true

julia> factory(tn, [0.1, 0.4, 0.5])
TurnoverEstimator
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: true
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`turnover_constraints`](@ref)
"""
function factory(tn::TurnoverEstimator, w::VecNum)::TurnoverEstimator
    return if tn.fixed
        tn
    else
        TurnoverEstimator(; w = w, val = tn.val, dval = tn.dval, fixed = tn.fixed)
    end
end
"""
    turnover_constraints(tn::TurnoverEstimator, sets::AssetSets; datatype::DataType = Float64,
                         strict::Bool = false)

Generate turnover portfolio constraints from a `TurnoverEstimator` and asset set.

`turnover_constraints` constructs a [`Turnover`](@ref) object representing turnover constraints for the assets in `sets`, using the specifications in `tn`. Supports scalar, vector, dictionary, pair, or custom turnover types for flexible assignment and validation.

# Arguments

  - `tn`: [`TurnoverEstimator`](@ref) specifying current weights, asset-specific turnover values, and default value.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Data type for default turnover values when `dval` is `nothing`.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `tn::Turnover`: Object containing portfolio weights and turnover values aligned with `sets`.

# Details

  - Turnover values are extracted and mapped to assets using [`estimator_to_val`](@ref).

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2));

julia> turnover_constraints(tn, sets)
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`AssetSets`](@ref)
"""
function turnover_constraints(tn::TurnoverEstimator, sets::AssetSets;
                              datatype::DataType = Float64, strict::Bool = false)::Turnover
    return Turnover(; w = tn.w,
                    val = estimator_to_val(tn.val, sets, tn.dval; datatype = datatype,
                                           strict = strict), fixed = tn.fixed)
end
"""
$(DocStringExtensions.TYPEDEF)

Container for turnover portfolio constraints.

`Turnover` stores the portfolio weights and turnover constraint values for each asset. The turnover constraint can be specified as a scalar (applied to all assets) or as a vector of per-asset values.

# Mathematical definition

```math
\\begin{align}
    \\boldsymbol{Tn}(\\boldsymbol{w}) &\\coloneqq \\lvert \\boldsymbol{w} - \\boldsymbol{w}_b \\rvert
\\end{align}
```

Where:

  - ``\\boldsymbol{Tn}(\\boldsymbol{w})``: `N × 1` turnover vector.
  - ``\\boldsymbol{w}``: `N × 1` vector of current portfolio weights.
  - ``\\boldsymbol{w}_b``: `N × 1` vector of benchmark portfolio weights.
  - ``\\lvert \\cdot \\rvert``: Element-wise absolute value.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Turnover(;
        w::VecNum,
        val::Num_VecNum = 0.0,
        fixed::Bool = false
    ) -> Turnover

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.

  - `val`:

      + `AbstractVector`: `!isempty(val)`, `length(w) == length(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
      + `Number`: `isfinite(val)` and `val >= 0`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0])
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false

julia> Turnover(; w = [0.2, 0.3, 0.5], val = 0.02, fixed = true)
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Float64: 0.02
  fixed ┴ Bool: true
```

# Related

  - [`set_turnover_constraints!`](@ref)
  - [`_set_turnover_constraints!`](@ref)
  - [`set_turnover_fees!`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`turnover_constraints`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct Turnover <: AbstractResult
    """
    $(field_dict[:w_tn])
    """
    @vprop w
    """
    $(field_dict[:val])
    """
    @vprop val
    """
    $(field_dict[:fixed])
    """
    fixed
    function Turnover(w::VecNum, val::Num_VecNum, fixed::Bool)::Turnover
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(val, :val)
        if isa(val, VecNum)
            @argcheck(length(val) == length(w), DimensionMismatch)
        end
        return new{typeof(w), typeof(val), typeof(fixed)}(w, val, fixed)
    end
end
function Turnover(; w::VecNum, val::Num_VecNum = 0.0, fixed::Bool = false)::Turnover
    return Turnover(w, val, fixed)
end
"""
    factory(tn::Turnover, w::VecNum)

Create a new `Turnover` constraint with updated portfolio weights.

`factory` constructs a new [`Turnover`](@ref) object using the provided portfolio weights `w` and the turnover values from an existing `Turnover` constraint `tn`.

# Arguments

  - `tn`: Existing `Turnover` constraint object.
  - `w`: New portfolio weights vector.

# Returns

  - `tn::Turnover`: New constraint object with updated weights and original turnover values.

# Examples

```jldoctest
julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0])
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false

julia> factory(tn, [0.0, 0.2, 0.8])
Turnover
      w ┼ Vector{Float64}: [0.0, 0.2, 0.8]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false

julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0], fixed = true)
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: true

julia> factory(tn, [0.0, 0.2, 0.8])
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: true
```

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref)
"""
function factory(tn::Turnover, w::VecNum)::Turnover
    return tn.fixed ? tn : Turnover(; w = w, val = tn.val, fixed = tn.fixed)
end
"""
    turnover_constraints(tn::Option{<:Turnover}, args...; kwargs...)

Propagate or pass through turnover portfolio constraints.

`turnover_constraints` returns the input [`Turnover`](@ref) object unchanged or `nothing`. This method is used to propagate already constructed turnover constraints, enabling composability and uniform interface handling in constraint generation workflows.

# Arguments

  - `tn`: An existing [`Turnover`](@ref) object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `tn::Option{<:Turnover}`: The input constraint object, unchanged.

# Examples

```jldoctest
julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0]);

julia> turnover_constraints(tn)
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`Option`](@ref)
"""
function turnover_constraints(tn::Option{<:Turnover}, args...;
                              kwargs...)::Option{<:Turnover}
    return tn
end
"""
    const TnE_Tn = Union{<:Turnover, <:TurnoverEstimator}

Alias for a turnover constraint or estimator.

Represents either a constructed turnover constraint or a turnover constraint estimator. Used for flexible dispatch in turnover constraint generation and processing.

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const TnE_Tn = Union{<:Turnover, <:TurnoverEstimator}
"""
    const VecTnE_Tn = AbstractVector{<:TnE_Tn}

Alias for a vector of turnover constraints or estimators.

Represents a collection of turnover constraints or estimators, enabling batch processing and broadcasting of turnover constraint generation.

# Related

  - [`TnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const VecTnE_Tn = AbstractVector{<:TnE_Tn}
"""
    const VecTn = AbstractVector{<:Turnover}

Alias for a vector of turnover constraints.

Represents a collection of constructed turnover constraints for multiple portfolios or assets.

# Related

  - [`Turnover`](@ref)
"""
const VecTn = AbstractVector{<:Turnover}
"""
    const Tn_VecTn = Union{<:Turnover, <:VecTn}

Alias for a single turnover constraint or a vector of turnover constraints.

Enables flexible dispatch for functions that accept either a single turnover constraint or multiple constraints.

# Related

  - [`Turnover`](@ref)
  - [`VecTn`](@ref)
"""
const Tn_VecTn = Union{<:Turnover, <:VecTn}
"""
    const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}

Alias for a single turnover constraint/estimator or a vector of them.

Supports flexible dispatch for turnover constraint generation and processing, accepting either a single constraint/estimator or a collection.

# Related

  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}
"""
    turnover_constraints(tn::VecTnE_Tn, sets::AssetSets; datatype::DataType = Float64,
                         strict::Bool = false)

Broadcasts [`turnover_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Data type for default turnover values when `dval` is `nothing`.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `res::VecTn`: Vector of constructed turnover constraints.

# Examples

```jldoctest
julia> sets = AssetSets(; dict = Dict("nx" => ["A", "B", "C"]));

julia> tn1 = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2));

julia> tn2 = TurnoverEstimator(; w = [0.1, 0.4, 0.5], val = Dict("B" => 0.15, "C" => 0.3));

julia> turnover_constraints([tn1, tn2], sets)
2-element Vector{Turnover{Vector{Float64}, Vector{Float64}, Bool}}:
 Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false

 Turnover
      w ┼ Vector{Float64}: [0.1, 0.4, 0.5]
    val ┼ Vector{Float64}: [0.0, 0.15, 0.3]
  fixed ┴ Bool: false
```

# Related

  - [`VecTnE_Tn`](@ref)
  - [`AssetSets`](@ref)
"""
function turnover_constraints(tn::VecTnE_Tn, sets::AssetSets; datatype::DataType = Float64,
                              strict::Bool = false)
    return [turnover_constraints(tni, sets; datatype = datatype, strict = strict)
            for tni in tn]
end
"""
    factory(tn::VecTnE_Tn, w::VecNum)

Create new turnover constraints or estimators with updated portfolio weights.

Applies [`factory`](@ref) to each element in `tn`, constructing a new collection of turnover constraints or estimators with the provided portfolio weights `w`.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `w`: New portfolio weights vector.

# Returns

  - `res::VecTnE_Tn`: Vector of updated turnover constraints or estimators.

# Examples

```jldoctest
julia> tn1 = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0]);

julia> tn2 = Turnover(; w = [0.2, 0.3, 0.5], val = [0.05, 0.1, 0.0]);

julia> factory([tn1, tn2], [0.1, 0.4, 0.5])
2-element Vector{Turnover{Vector{Float64}, Vector{Float64}, Bool}}:
 Turnover
      w ┼ Vector{Float64}: [0.1, 0.4, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false

 Turnover
      w ┼ Vector{Float64}: [0.1, 0.4, 0.5]
    val ┼ Vector{Float64}: [0.05, 0.1, 0.0]
  fixed ┴ Bool: false
```

# Related

  - [`VecTnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`factory(tn::TurnoverEstimator, w::VecNum)`](@ref)
"""
function factory(tn::VecTnE_Tn, w::VecNum)
    val = [factory(tni, w) for tni in tn]
    if isabstracttype(eltype(val))
        val = concrete_typed_array(val)
    end
    return val
end
"""
    port_opt_view(tn::VecTnE_Tn, i)

Create views of multiple turnover constraints or estimators for a subset of assets.

`port_opt_view` returns a vector of turnover constraint or estimator objects, each restricted to the indices or assets specified by `i`.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `res::VecTnE_Tn`: Vector of turnover constraint or estimator objects, each restricted to the specified subset.

# Details

  - Applies `port_opt_view` to each element in `tn`.
  - Supports both `Turnover` and `TurnoverEstimator` types.
  - Enables composable and uniform processing of asset subsets for batch turnover constraints.

# Examples

```jldoctest
julia> tn1 = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0], fixed = true);

julia> tn2 = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                               dval = 0.0, fixed = true);

julia> PortfolioOptimisers.port_opt_view(concrete_typed_array([tn1, tn2]), 1:2)
2-element Vector{Union{Turnover{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Bool}, TurnoverEstimator{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Dict{String, Float64}, Float64, Bool}}}:
 Turnover
      w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
    val ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.1, 0.2]
  fixed ┴ Bool: true

 TurnoverEstimator
      w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
    val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
   dval ┼ Float64: 0.0
  fixed ┴ Bool: true
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`turnover_constraints`](@ref)
  - [`port_opt_view`](@ref)
  - [`concrete_typed_array`](@ref)
"""
function port_opt_view(tn::VecTnE_Tn, i, args...)
    val = [port_opt_view(tni, i) for tni in tn]
    if isabstracttype(eltype(val))
        val = concrete_typed_array(val)
    end
    return val
end
"""
    needs_previous_weights(tn::TnE_Tn) -> Bool
    needs_previous_weights(tn::VecTnE_Tn) -> Bool

Check if a turnover constraint or estimator requires previous portfolio weights.

# Arguments

  - `tn`: Turnover constraint or estimator.

# Returns

  - `Bool`: `true` if previous weights are needed, `false` otherwise.

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
"""
function needs_previous_weights(tn::TnE_Tn)::Bool
    return !tn.fixed
end
function needs_previous_weights(tn::VecTnE_Tn)::Bool
    return any(needs_previous_weights.(tn))
end

export TurnoverEstimator, Turnover, turnover_constraints
