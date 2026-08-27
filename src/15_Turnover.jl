"""
$(DocStringExtensions.TYPEDEF)

Names the per-asset turnover bounds, for [`turnover_constraints`](@ref) to align to a universe.

`val` accepts a dictionary, a pair, or a vector of pairs keyed by asset or group name, and `dval` fills every asset the keys miss. [`turnover_constraints`](@ref) resolves the names against a [`UniverseSets`](@ref) and returns a [`Turnover`](@ref), whose `val` is a plain per-asset vector.

As on [`Turnover`](@ref), the `w` field holds the **reference** weights, not the candidate weights.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TurnoverEstimator(;
        w::VecNum,
        val::EstValType,
        dval::Option{<:Number} = nothing,
        fixed::Bool = false
    ) -> TurnoverEstimator

Keywords correspond to the struct's fields.

## Validation

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `any(isfinite, w)`.

  - `val`, through [`assert_nonempty_nonneg_finite_val`](@ref):

      + `AbstractDict`: `!isempty(val)`, `any(isfinite, values(val))` and `all(x -> x >= 0, values(val))`.
      + Vector of pairs: `!isempty(val)`, `any(isfinite, getindex.(val, 2))` and `all(x -> x[2] >= 0, val)`.
      + `Pair`: `isfinite(val[2])` and `val[2] >= 0`.

  - `dval`: if not `nothing`, `dval >= 0`.

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `val`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2), dval = 0.0)
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

Replace the reference weights of a [`TurnoverEstimator`](@ref), unless `fixed` holds them.

The `fixed` field decides which weight vector survives. A `fixed` estimator pins the reference weights it was built with, so the incoming `w` is discarded and the argument `tn` is returned itself.

# Algorithm

 1. Read `tn.fixed`. When it is `true`, return `tn` unchanged: the reference weights `tn.w` survive and the argument `w` is discarded.
 2. When it is `false`, build a new [`TurnoverEstimator`](@ref) whose `w` is the argument `w`, and whose `val`, `dval` and `fixed` are those of `tn`. The argument `w` survives.

# Arguments

  - `tn`: Existing `TurnoverEstimator` object. Supplies the turnover values, the default value and the `fixed` flag.
  - `w`: Candidate reference weights vector.

# Validation

  - Step 2 builds a `TurnoverEstimator`, so `w` meets that constructor's rules. Step 1 builds nothing and checks nothing.

# Returns

  - `tn::TurnoverEstimator`: `tn` itself when `tn.fixed` is `true`, otherwise a new estimator carrying `w` as its reference weights.

# Examples

```jldoctest
julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2),
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

julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2),
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
    turnover_constraints(tn::TurnoverEstimator, sets::UniverseSets; datatype::DataType = Float64,
                         strict::Bool = false)

Generate turnover portfolio constraints from a `TurnoverEstimator` and asset set.

`turnover_constraints` constructs a [`Turnover`](@ref) object representing turnover constraints for the assets in `sets`, using the specifications in `tn`. Supports scalar, vector, dictionary, pair, or custom turnover types for flexible assignment and validation.

# Algorithm

 1. Resolve `tn.val` against the universe of `sets` with [`estimator_to_val`](@ref), giving one turnover bound per asset. The bounds follow the order of the universe, not the order of the keys of `tn.val`. Every asset the keys miss takes `tn.dval`, or `zero(datatype)` when `tn.dval` is `nothing`. A key that names neither an asset nor a group raises when `strict` is `true`, and warns otherwise.
 2. Build a [`Turnover`](@ref) from `tn.w`, the bound vector of step 1 and `tn.fixed`.

# Arguments

  - `tn`: [`TurnoverEstimator`](@ref) specifying current weights, asset-specific turnover values, and default value.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `datatype`: Data type for default turnover values when `dval` is `nothing`.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `tn::Turnover`: Object containing portfolio weights and turnover values aligned with `sets`.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> tn = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2));

julia> turnover_constraints(tn, sets)
Turnover
      w ┼ Vector{Float64}: [0.2, 0.3, 0.5]
    val ┼ Vector{Float64}: [0.1, 0.2, 0.0]
  fixed ┴ Bool: false
```

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`UniverseSets`](@ref)
  - [`estimator_to_val`](@ref): resolves the names of `tn.val` against the universe of `sets`.
"""
function turnover_constraints(tn::TurnoverEstimator, sets::UniverseSets;
                              datatype::DataType = Float64, strict::Bool = false)::Turnover
    return Turnover(; w = tn.w,
                    val = estimator_to_val(tn.val, sets, tn.dval; datatype = datatype,
                                           strict = strict), fixed = tn.fixed)
end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the per-asset weight change against a reference portfolio.

!!! warning

    The `w` field holds the **reference** weights, not the candidate weights. The candidate is the optimiser's own weight variable, which no field carries. `factory(tn, w)` replaces the reference, which is how the previous rebalance becomes the next reference.

`val` is the bound. [`Fees`](@ref) reuses the same type with `val` read as a per-asset **fee rate** instead, so read the meaning of `val` off the type that holds the `Turnover`.

# Mathematical definition

```math
\\begin{align}
    \\boldsymbol{Tn}(\\boldsymbol{w}) &\\coloneqq \\lvert \\boldsymbol{w} - \\boldsymbol{w}_{0} \\rvert\\,, \\\\
    \\boldsymbol{Tn}(\\boldsymbol{w}) &\\leq \\boldsymbol{\\delta}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{Tn}(\\boldsymbol{w})``: `N × 1` turnover vector.
  - ``\\boldsymbol{w}``: `N × 1` vector of candidate portfolio weights.
  - ``\\boldsymbol{w}_{0}``: `N × 1` vector of reference portfolio weights, the `w` field.
  - ``\\boldsymbol{\\delta}``: `N × 1` vector of maximum turnover, the `val` field. A scalar `val` broadcasts to every asset.
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

The rules are listed in the order of the raises, so the first rule a value breaks is the one it is told about.

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `any(isfinite, w)`.

  - `val`, through [`assert_nonempty_nonneg_finite_val`](@ref):

      + `AbstractVector`: `!isempty(val)`, `any(isfinite, val)` and `all(x -> x >= 0, val)`.
      + `Number`: `isfinite(val)` and `val >= 0`.

  - `length(w) == length(val)` when `val` is an `AbstractVector`, raising a `DimensionMismatch`. This rule is checked last.

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

  - [`set_turnover_constraints!`](@ref): writes the bound as the two linear rows the source gives, one per side of the absolute value.
  - [`_set_turnover_constraints!`](@ref)
  - [`set_turnover_fees!`](@ref): reads `val` as a per-asset fee rate rather than as a bound.
  - [`TurnoverEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`turnover_constraints`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`Fees`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equations 9.10 and 9.11.
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

Replace the reference weights of a [`Turnover`](@ref), unless `fixed` holds them.

The `fixed` field decides which weight vector survives. A `fixed` constraint pins the reference weights it was built with, so the incoming `w` is discarded and the argument `tn` is returned itself.

# Algorithm

 1. Read `tn.fixed`. When it is `true`, return `tn` unchanged: the reference weights `tn.w` survive and the argument `w` is discarded.
 2. When it is `false`, build a new [`Turnover`](@ref) whose `w` is the argument `w`, and whose `val` and `fixed` are those of `tn`. The argument `w` survives.

# Arguments

  - `tn`: Existing `Turnover` constraint object. Supplies the turnover values and the `fixed` flag.
  - `w`: Candidate reference weights vector.

# Validation

  - Step 2 builds a `Turnover`, so `w` meets that constructor's rules. Step 1 builds nothing and checks nothing.

# Returns

  - `tn::Turnover`: `tn` itself when `tn.fixed` is `true`, otherwise a new constraint carrying `w` as its reference weights.

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

# Algorithm

 1. Return `tn`. A [`Turnover`](@ref) already carries one bound per asset, so no universe is resolved. The method reads none of its other arguments and none of its keywords.

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

Union of the two turnover types, one turnover constraint on either side of name resolution. Both carry `w`, `val` and `fixed`, so [`factory`](@ref), [`port_opt_view`](@ref) and [`needs_previous_weights`](@ref) act on either without a branch; they differ only in whether `val` is already one bound per asset.

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref): the verb that turns the estimator side into the result side.
  - [`FeesEstimator`](@ref): holds one of these in its `tn` field.
"""
const TnE_Tn = Union{<:Turnover, <:TurnoverEstimator}
"""
    const VecTnE_Tn = AbstractVector{<:TnE_Tn}

Vector of [`TnE_Tn`](@ref), the type the broadcast methods of this file dispatch on. Several turnover constraints can hold at once, so the singular alias needs a plural counterpart; the element type may mix both sides of name resolution, which is why the methods that build a new vector keep it concrete.

# Related

  - [`TnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
  - [`needs_previous_weights`](@ref)
"""
const VecTnE_Tn = AbstractVector{<:TnE_Tn}
"""
    const VecTn = AbstractVector{<:Turnover}

Vector of [`Turnover`](@ref) alone, the resolved side of [`VecTnE_Tn`](@ref). It is what the vector method of [`turnover_constraints`](@ref) returns, so a method that dispatches on it is downstream of name resolution and reads `val` as one bound per asset.

# Related

  - [`Turnover`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Tn_VecTn`](@ref)
  - [`turnover_constraints`](@ref)
"""
const VecTn = AbstractVector{<:Turnover}
"""
    const Tn_VecTn = Union{<:Turnover, <:VecTn}

One resolved turnover constraint or a vector of them. This is the shape a `JuMPOptimiser` holds after name resolution and the shape [`set_turnover_constraints!`](@ref) reads, so no method that dispatches on it ever meets a [`TurnoverEstimator`](@ref).

# Related

  - [`Turnover`](@ref)
  - [`VecTn`](@ref)
  - [`TnE_Tn_VecTnE_Tn`](@ref): the unresolved counterpart, which the optimiser accepts from the caller.
  - [`set_turnover_constraints!`](@ref)
"""
const Tn_VecTn = Union{<:Turnover, <:VecTn}
"""
    const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}

Widest turnover alias: one constraint or estimator, or a vector of them. It is the type a `JuMPOptimiser` accepts for its `tn` keyword, because a caller may pass either side of name resolution and either count; [`turnover_constraints`](@ref) narrows every case of it to [`Tn_VecTn`](@ref).

# Related

  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`Tn_VecTn`](@ref): what this narrows to once the names are resolved.
  - [`turnover_constraints`](@ref)
"""
const TnE_Tn_VecTnE_Tn = Union{<:TnE_Tn, <:VecTnE_Tn}
"""
    turnover_constraints(tn::VecTnE_Tn, sets::UniverseSets; datatype::DataType = Float64,
                         strict::Bool = false)

Broadcasts [`turnover_constraints`](@ref) over the vector.

Provides a uniform interface for processing multiple constraint estimators simultaneously.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`turnover_constraints`](@ref) on `tni` with `sets`, `datatype` and `strict`. An entry that is already a [`Turnover`](@ref) passes through.
 2. Collect the results into a vector that preserves the order of `tn`.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `sets`: [`UniverseSets`](@ref) containing asset names or indices.
  - `datatype`: Data type for default turnover values when `dval` is `nothing`.
  - `strict`: If `true`, enforces strict matching between assets and turnover values (throws error on mismatch); if `false`, issues a warning.

# Returns

  - `res::VecTn`: Vector of constructed turnover constraints.

# Examples

```jldoctest
julia> sets = UniverseSets(; dict = Dict(\"nx\" => [\"A\", \"B\", \"C\"]));

julia> tn1 = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2));

julia> tn2 = TurnoverEstimator(; w = [0.1, 0.4, 0.5], val = Dict(\"B\" => 0.15, \"C\" => 0.3));

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
  - [`UniverseSets`](@ref)
"""
function turnover_constraints(tn::VecTnE_Tn, sets::UniverseSets;
                              datatype::DataType = Float64, strict::Bool = false)
    return [turnover_constraints(tni, sets; datatype = datatype, strict = strict)
            for tni in tn]
end
"""
    factory(tn::VecTnE_Tn, w::VecNum)

Create new turnover constraints or estimators with updated portfolio weights.

Applies [`factory`](@ref) to each element in `tn`, constructing a new collection of turnover constraints or estimators with the provided portfolio weights `w`.

This is the generic vector [`factory`](@ref) plus [`concrete_typed_array_if_abstract`](@ref): a mixed vector of [`Turnover`](@ref) and [`TurnoverEstimator`](@ref) keeps a concrete element type.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`factory`](@ref) on `tni` with `w`. A `fixed` entry returns itself and keeps its own reference weights, so a vector may hold both outcomes.
 2. Pass the collected vector through [`concrete_typed_array_if_abstract`](@ref). Step 1 can widen the element type to an abstract one, because the two turnover types have no common concrete type; this step narrows it back to a `Union` the compiler can dispatch on.

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
    return concrete_typed_array_if_abstract([factory(tni, w) for tni in tn])
end
"""
    port_opt_view(tn::VecTnE_Tn, i, args...)

Create views of multiple turnover constraints or estimators for a subset of assets.

`port_opt_view` returns a vector of turnover constraint or estimator objects, each restricted to the indices or assets specified by `i`.

This is the generic vector [`port_opt_view`](@ref) plus [`concrete_typed_array_if_abstract`](@ref): a mixed vector of [`Turnover`](@ref) and [`TurnoverEstimator`](@ref) keeps a concrete element type.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`port_opt_view`](@ref) on `tni` with `i` and `args...`. The `@vprop` tags of each type decide what is sliced: `w` and a vector `val` become views over `i`, and a scalar or dictionary `val` passes through unchanged.
 2. Pass the collected vector through [`concrete_typed_array_if_abstract`](@ref), for the reason [`factory(tn::VecTnE_Tn, w::VecNum)`](@ref) gives.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `i`: Index or indices specifying the subset of assets.
  - `args...`: Further arguments, forwarded unchanged to each element's [`port_opt_view`](@ref).

# Returns

  - `res::VecTnE_Tn`: Vector of turnover constraint or estimator objects, each restricted to the specified subset.

# Examples

```jldoctest
julia> tn1 = Turnover(; w = [0.2, 0.3, 0.5], val = [0.1, 0.2, 0.0], fixed = true);

julia> tn2 = TurnoverEstimator(; w = [0.2, 0.3, 0.5], val = Dict(\"A\" => 0.1, \"B\" => 0.2),
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
  - [`concrete_typed_array_if_abstract`](@ref)
"""
function port_opt_view(tn::VecTnE_Tn, i, args...)
    return concrete_typed_array_if_abstract([port_opt_view(tni, i, args...) for tni in tn])
end
"""
    needs_previous_weights(tn::TnE_Tn) -> Bool
    needs_previous_weights(tn::VecTnE_Tn) -> Bool

Check if a turnover constraint or estimator requires previous portfolio weights.

A `fixed` entry pins its own reference weights, so it needs none: the scalar method answers `!tn.fixed`. The vector method answers `any` and not `all`, so one entry that is not `fixed` makes the whole vector need them.

# Algorithm

 1. On a single [`TnE_Tn`](@ref), return `!tn.fixed`.
 2. On a [`VecTnE_Tn`](@ref), apply step 1 to every entry and reduce with `any`.

# Arguments

  - `tn`: One turnover constraint or estimator, or a vector of them.

# Returns

  - `Bool`: `true` if previous weights are needed, `false` otherwise.

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref): the verb that reads the same `fixed` flag.
"""
function needs_previous_weights(tn::TnE_Tn)::Bool
    return !tn.fixed
end
function needs_previous_weights(tn::VecTnE_Tn)::Bool
    return any(needs_previous_weights.(tn))
end

export TurnoverEstimator, Turnover, turnover_constraints
