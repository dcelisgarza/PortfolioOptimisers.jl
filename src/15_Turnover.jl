"""
    turnover_view(::Nothing, ::Any)

Returns `nothing` when the turnover constraint or estimator is not provided.

Used as a fallback method for missing turnover constraints or estimators, ensuring composability and uniform interface handling in constraint processing workflows.

# Arguments

  - `::Nothing`: Indicates absence of a turnover constraint or estimator.
  - `::Any`: Index or argument (ignored).

# Returns

  - `nothing`.

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`turnover_constraints`](@ref)
"""
function turnover_view(::Nothing, ::Any)
    return nothing
end
"""
    struct TurnoverEstimator{T1, T2, T3} <: AbstractEstimator
        w::T1
        val::T2
        dval::T3
    end

Estimator for turnover portfolio constraints.

`TurnoverEstimator` specifies turnover constraints for each asset in a portfolio, based on current portfolio weights `w`, asset-specific turnover values `val`, and a default value for assets not explicitly specified. Supports asset-specific turnover via dictionaries, pairs, or vectors of pairs.

This estimator can be converted into a concrete [`Turnover`](@ref) constraint using the [`turnover_constraints`](@ref) function, which maps the estimator's specifications to the assets in a given [`AssetSets`](@ref) object.

# Fields

  - `w`: Vector of current portfolio weights.
  - `val`: Asset-specific turnover values, as a dictionary, pair, or vector of pairs.
  - `dval`: Default turnover value for assets not specified in `val`.

# Constructor

    TurnoverEstimator(; w::VecNum, val::EstValType, dval::Option{<:Number} = nothing)

## Validation

  - `w` is validated with [`assert_nonempty_finite_val`](@ref).
  - `val` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).
  - `dval`, if not `nothing`, `dval >= 0`.

# Examples

```jldoctest
julia> TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2), dval = 0.0)
TurnoverEstimator
     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
   val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  dval ┴ Float64: 0.0
```

# Related

  - [`Turnover`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`EstValType`](@ref)
  - [`Option`](@ref)
  - [`turnover_constraints`](@ref)
"""
struct TurnoverEstimator{T1, T2, T3} <: AbstractEstimator
    w::T1
    val::T2
    dval::T3
    function TurnoverEstimator(w::VecNum, val::EstValType, dval::Option{<:Number} = nothing)
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(val)
        if !isnothing(dval)
            @argcheck(zero(dval) <= dval, DomainError)
        end
        return new{typeof(w), typeof(val), typeof(dval)}(w, val, dval)
    end
end
function TurnoverEstimator(; w::VecNum, val::EstValType, dval::Option{<:Number} = nothing)
    return TurnoverEstimator(w, val, dval)
end
"""
    turnover_view(tn::TurnoverEstimator, i)

Create a view of a `TurnoverEstimator` for a subset of assets.

`turnover_view` returns a new `TurnoverEstimator` with portfolio weights and turnover values restricted to the indices or assets specified by `i`. The default turnover value is propagated unchanged.

# Arguments

  - `tn`: An instance of `TurnoverEstimator`.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `tn::TurnoverEstimator`: A new estimator with fields restricted to the specified subset.

# Details

  - Uses `view` to create a subset of the weights.
  - Uses [`nothing_scalar_array_view`](@ref) to subset turnover values.
  - Propagates the default turnover value.

# Examples

```jldoctest
julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                              dval = 0.0);

julia> PortfolioOptimisers.turnover_view(tn, 1:2)
TurnoverEstimator
     w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
   val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  dval ┴ Float64: 0.0
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`turnover_constraints`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function turnover_view(tn::TurnoverEstimator, i)
    w = view(tn.w, i)
    val = nothing_scalar_array_view(tn.val, i)
    return TurnoverEstimator(; w = w, val = val, dval = tn.dval)
end
"""
    factory(tn::TurnoverEstimator, w::VecNum)

Create a new `TurnoverEstimator` with updated portfolio weights.

Constructs a new [`TurnoverEstimator`](@ref) object using the provided portfolio weights `w` and the turnover values and default value from an existing `TurnoverEstimator` `tn`. This enables composable updating of weights while preserving turnover constraint values.

# Arguments

  - `tn`: Existing `TurnoverEstimator` object. Supplies turnover values and default value.
  - `w`: New portfolio weights vector.

# Returns

  - `tn::TurnoverEstimator`: New estimator object with updated weights and original turnover values and default.

# Validation

  - `w` is validated to be non-empty, finite, and numeric.

# Details

  - Copies turnover values and default value from `tn`.
  - Updates only the weights field.

# Examples

```jldoctest
julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                              dval = 0.0)
TurnoverEstimator
     w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
   val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  dval ┴ Float64: 0.0

julia> factory(tn, [0.1, 0.4, 0.5])
TurnoverEstimator
     w ┼ Vector{Float64}: [0.1, 0.4, 0.5]
   val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  dval ┴ Float64: 0.0
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`turnover_constraints`](@ref)
"""
function factory(tn::TurnoverEstimator, w::VecNum)
    return TurnoverEstimator(; w = w, val = tn.val, dval = tn.dval)
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

julia> tn = TurnoverEstimator([0.2, 0.3, 0.5], Dict("A" => 0.1, "B" => 0.2));

julia> turnover_constraints(tn, sets)
Turnover
    w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
  val ┴ Vector{Float64}: [0.1, 0.2, 0.0]
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`AssetSets`](@ref)
  - [`turnover_constraints`](@ref)
"""
function turnover_constraints(tn::TurnoverEstimator, sets::AssetSets;
                              datatype::DataType = Float64, strict::Bool = false)
    return Turnover(; w = tn.w,
                    val = estimator_to_val(tn.val, sets, tn.dval; datatype = datatype,
                                           strict = strict))
end
"""
    struct Turnover{T1, T2} <: AbstractResult
        w::T1
        val::T2
    end

Container for turnover portfolio constraints.

`Turnover` stores the portfolio weights and turnover constraint values for each asset. The turnover constraint can be specified as a scalar (applied to all assets) or as a vector of per-asset values.

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

  - `w`: Vector of benchmark portfolio weights.

  - `val`: Scalar or vector of turnover constraint values. Scalar values are broadcast to all assets.

      + When used as a constraint, this value is used to constrain the maximum allowed turnover per asset.
      + When used in [`Fees`](@ref), this value represents the turnover fee per asset.

# Constructor

    Turnover(; w::VecNum, val::Num_VecNum = 0.0)

Keyword arguments correspond to the fields above.

## Validation

  - `!isempty(w)`.

  - `val`:

      + `AbstractVector`: `!isempty(val)`, `length(w) == length(val)`, `any(isfinite, val)`, `all(x -> x >= 0, val)`.
      + `Number`: `isfinite(val)` and `val >= 0`.

# Examples

```jldoctest
julia> Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0])
Turnover
    w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
  val ┴ Vector{Float64}: [0.1, 0.2, 0.0]

julia> Turnover(; w = [0.2, 0.3, 0.5], val = 0.02)
Turnover
    w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
  val ┴ Float64: 0.02
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`turnover_constraints`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`turnover_view`](@ref)
"""
struct Turnover{T1, T2} <: AbstractResult
    w::T1
    val::T2
    function Turnover(w::VecNum, val::Num_VecNum)
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(val)
        if isa(val, VecNum)
            @argcheck(length(val) == length(w), DimensionMismatch)
        end
        return new{typeof(w), typeof(val)}(w, val)
    end
end
function Turnover(; w::VecNum, val::Num_VecNum = 0.0)
    return Turnover(w, val)
end
"""
    turnover_view(tn::Turnover, i)

Create a view of a `Turnover` for a subset of assets.

Returns a new `Turnover` object with portfolio weights and turnover values restricted to the indices or assets specified by `i`.

# Arguments

  - `tn`: A `Turnover` object containing portfolio weights and turnover values.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `tn::Turnover`: A new turnover constraint object with fields restricted to the specified subset.

# Details

  - Uses `view` to create a subset of the weights.
  - Uses [`nothing_scalar_array_view`](@ref) to subset turnover values.

# Examples

```jldoctest
julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0]);

julia> PortfolioOptimisers.turnover_view(tn, 1:2)
Turnover
    w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
  val ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.1, 0.2]
```

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function turnover_view(tn::Turnover, i)
    w = view(tn.w, i)
    val = nothing_scalar_array_view(tn.val, i)
    return Turnover(; w = w, val = val)
end
"""
    factory(tn::Turnover, w::VecNum)

Create a new `Turnover` constraint with updated portfolio weights.

`factory` constructs a new [`Turnover`](@ref) object using the provided portfolio weights `w` and the turnover values from an existing `Turnover` constraint `tn`. This enables composable updating of weights while preserving turnover constraint values.

# Arguments

  - `tn`: Existing `Turnover` constraint object.
  - `w`: New portfolio weights vector.

# Returns

  - `tn::Turnover`: New constraint object with updated weights and original turnover values.

# Details

  - Copies turnover values from `tn`.
  - Updates only the weights field.

# Examples

```jldoctest
julia> tn = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0])
Turnover
    w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
  val ┴ Vector{Float64}: [0.1, 0.2, 0.0]

julia> factory(tn, [0.0, 0.2, 0.8])
Turnover
    w ┼ Vector{Float64}: [0.0, 0.2, 0.8]
  val ┴ Vector{Float64}: [0.1, 0.2, 0.0]
```

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref)
"""
function factory(tn::Turnover, w::VecNum)
    return Turnover(; w = w, val = tn.val)
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
  val ┴ Vector{Float64}: [0.1, 0.2, 0.0]
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`Option`](@ref)
"""
function turnover_constraints(tn::Option{<:Turnover}, args...; kwargs...)
    return tn
end
"""
    const TnE_Tn = Union{<:Turnover, <:TurnoverEstimator}

Alias for a turnover constraint or estimator.

Represents either a constructed turnover constraint or a turnover constraint estimator. Used for flexible dispatch in turnover constraint generation and processing.

# Related Types

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const TnE_Tn = Union{<:Turnover, <:TurnoverEstimator}
"""
    const VecTnE_Tn = AbstractVector{<:TnE_Tn}

Alias for a vector of turnover constraints or estimators.

Represents a collection of turnover constraints or estimators, enabling batch processing and broadcasting of turnover constraint generation.

# Related Types

  - [`TnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const VecTnE_Tn = AbstractVector{<:TnE_Tn}
"""
    const VecTn = AbstractVector{<:Turnover}

Alias for a vector of turnover constraints.

Represents a collection of constructed turnover constraints for multiple portfolios or assets.

# Related Types

  - [`Turnover`](@ref)
"""
const VecTn = AbstractVector{<:Turnover}
"""
    const Tn_VecTn = Union{<:Turnover, <:VecTn}

Alias for a single turnover constraint or a vector of turnover constraints.

Enables flexible dispatch for functions that accept either a single turnover constraint or multiple constraints.

# Related Types

  - [`Turnover`](@ref)
  - [`VecTn`](@ref)
"""
const Tn_VecTn = Union{<:Turnover, <:VecTn}
"""
    const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}

Alias for a single turnover constraint/estimator or a vector of them.

Supports flexible dispatch for turnover constraint generation and processing, accepting either a single constraint/estimator or a collection.

# Related Types

  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
"""
const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}
"""
    turnover_constraints(tn::VecTnE_Tn, sets::AssetSets; datatype::DataType = Float64,
                         strict::Bool = false)

Broadcasts [`threshold_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `sets`: [`AssetSets`](@ref) containing asset names or indices.
  - `datatype`: Data type for default turnover values when `dval` is `nothing`.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `res::VecTn`: Vector of constructed turnover constraints.

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
    turnover_view(tn::VecTnE_Tn, i)

Create views of multiple turnover constraints or estimators for a subset of assets.

`turnover_view` returns a vector of turnover constraint or estimator objects, each restricted to the indices or assets specified by `i`. This enables batch processing and composable handling of asset subsets across multiple turnover specifications.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `i`: Index or indices specifying the subset of assets.

# Returns

  - `res::VecTnE_Tn`: Vector of turnover constraint or estimator objects, each restricted to the specified subset.

# Details

  - Applies `turnover_view` to each element in `tn`.
  - Supports both `Turnover` and `TurnoverEstimator` types.
  - Enables composable and uniform processing of asset subsets for batch turnover constraints.

# Examples

```jldoctest
julia> tn1 = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0]);

julia> tn2 = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict("A" => 0.1, "B" => 0.2),
                               dval = 0.0);

julia> PortfolioOptimisers.turnover_view(concrete_typed_array([tn1, tn2]), 1:2)
2-element Vector{Union{Turnover{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}}, TurnoverEstimator{SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}, Dict{String, Float64}, Float64}}}:
 Turnover
    w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
  val ┴ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.1, 0.2]

 TurnoverEstimator
     w ┼ SubArray{Float64, 1, Vector{Float64}, Tuple{UnitRange{Int64}}, true}: [0.2, 0.3]
   val ┼ Dict{String, Float64}: Dict("B" => 0.2, "A" => 0.1)
  dval ┴ Float64: 0.0
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`turnover_constraints`](@ref)
  - [`turnover_view`](@ref)
  - [`concrete_typed_array`](@ref)
"""
function turnover_view(tn::VecTnE_Tn, i)
    val = [turnover_view(tni, i) for tni in tn]
    isconcretetype(eltype(val)) ? nothing : (val = concrete_typed_array(val))
    return val
end

export TurnoverEstimator, Turnover, turnover_constraints
