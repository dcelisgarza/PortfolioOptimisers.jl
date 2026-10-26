"""
$(DocStringExtensions.TYPEDEF)

Turnover bounds before [`turnover_constraints`](@ref) aligns them to a universe. The bounds can name assets or groups.

`val` can be a number, a vector, a pair, a dictionary, a vector of pairs, or a [`VectorAbstractEstimatorValueAlgorithm`](@ref). A pair, a dictionary and a vector of pairs use asset or group names as keys, and `dval` is the bound of every asset that the keys do not name. [`turnover_constraints`](@ref) resolves the names against a [`UniverseSets`](@ref) and returns a [`Turnover`](@ref) whose `val` holds one bound per asset.

The `w` field holds the reference weights, not the candidate weights, as on [`Turnover`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TurnoverEstimator(;
        w::VecNum,
        val::VectorEstValType,
        dval::Option{<:Number} = nothing,
        fixed::Bool = false
    ) -> TurnoverEstimator

Keywords correspond to the struct's fields.

## Validation

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`.

  - `val`, through [`assert_nonempty_nonneg_finite_val`](@ref):

      + `AbstractDict`: `!isempty(val)`, `all(isfinite, values(val))` and `all(x -> x >= 0, values(val))`.
      + Vector of pairs: `!isempty(val)`, `all(isfinite, getindex.(val, 2))` and `all(x -> x[2] >= 0, val)`.
      + `Pair`: `isfinite(val[2])` and `val[2] >= 0`.
      + `AbstractVector` of numbers: `!isempty(val)`, `all(isfinite, val)` and `all(x -> x >= 0, val)`.
      + `Number`: `isfinite(val)` and `val >= 0`.
      + [`VectorAbstractEstimatorValueAlgorithm`](@ref): no rule.

  - `dval`, through [`assert_nonneg`](@ref): if not `nothing`, `dval >= 0`. This rule admits `+Inf`, which the rules of `val` refuse. `dval` is the bound of every asset that `val` does not name, and `dval = +Inf` leaves these assets with no bound. `dval = nothing` gives them a bound of zero, which freezes their weights.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

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
  - [`VectorEstValType`](@ref)
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
    $(field_dict[:val_tn])
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
    function TurnoverEstimator(w::VecNum, val::VectorEstValType, dval::Option{<:Number},
                               fixed::Bool)::TurnoverEstimator
        assert_nonempty_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(val, :val)
        if !isnothing(dval)
            # `dval = +Inf` is the one way to give the assets that `val` does not name no
            # bound, because `dval = nothing` gives them a bound of zero. So this check
            # admits `+Inf`, which the check of `val` refuses.
            assert_nonneg(dval, :dval)
        end
        return new{typeof(w), typeof(val), typeof(dval), typeof(fixed)}(w, val, dval, fixed)
    end
end
function TurnoverEstimator(; w::VecNum, val::VectorEstValType,
                           dval::Option{<:Number} = nothing,
                           fixed::Bool = false)::TurnoverEstimator
    return TurnoverEstimator(w, val, dval, fixed)
end
"""
    factory(tn::TurnoverEstimator, w::VecNum)

Replace the reference weights of a [`TurnoverEstimator`](@ref), unless its `fixed` field is `true`.

The `fixed` field selects the weight vector that the result holds. A `fixed` estimator keeps the reference weights that it was built with, so the function ignores `w` and returns `tn`.

# Algorithm

 1. If `tn.fixed` is `true`, return `tn` unchanged. The result holds the reference weights `tn.w`, and the argument `w` has no effect.
 2. If `tn.fixed` is `false`, build a new [`TurnoverEstimator`](@ref) whose `w` is the argument `w`, and whose `val`, `dval` and `fixed` are those of `tn`. The result holds the argument `w`.

# Arguments

  - `tn`: The estimator to update. It gives the turnover bounds, the default bound and the `fixed` flag.
  - `w`: New reference weight vector.

# Validation

  - Step 2 builds a `TurnoverEstimator`, so its constructor checks `w`. Step 1 checks nothing.

# Returns

  - `tn::TurnoverEstimator`: `tn` if `tn.fixed` is `true`. Otherwise, a new estimator with `w` as its reference weights.

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

Resolve the names of a [`TurnoverEstimator`](@ref) against a universe, and return a [`Turnover`](@ref).

The result holds the reference weights of `tn` and one turnover bound per asset of `sets`. [`TurnoverEstimator`](@ref) lists the shapes that `tn.val` can take.

# Algorithm

 1. Resolve `tn.val` against the universe of `sets` with [`estimator_to_val`](@ref). The result is one turnover bound per asset, in the order of the universe and not in the order of the keys of `tn.val`. Every asset that the keys do not name takes `tn.dval`, or `zero(datatype)` if `tn.dval` is `nothing`. A key that names neither an asset nor a group raises an error if `strict` is `true`, and logs a warning if `strict` is `false`.
 2. Build a [`Turnover`](@ref) from `tn.w`, the bounds of step 1 and `tn.fixed`.

# Arguments

  - `tn`: The estimator to resolve. It gives the reference weights, the turnover bounds and the default bound.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names of the universe and the groups.
  - `datatype`: Element type of the bounds if `tn.dval` is `nothing`. It has no effect if `tn.dval` is a number.
  - `strict`: If `true`, a key that names no asset and no group raises an error. If `false`, it logs a warning.

# Returns

  - `tn::Turnover`: The reference weights of `tn`, with one turnover bound per asset of `sets`.

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

Bound on the weight change of each asset from a reference portfolio.

!!! warning

    The `w` field holds the **reference** weights, not the candidate weights. The candidate weights are the weight variable of the optimiser, and no field holds them. `factory(tn, w)` replaces the reference weights with `w`. Give it the weights of the last rebalance, and they become the reference of the next rebalance.

`val` is the bound. A [`Fees`](@ref) holds the same type and reads `val` as a fee rate per asset. The type that holds the `Turnover` thus gives the meaning of `val`.

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

The constructor checks the rules in the order of this list, so its error names the first rule that a value breaks.

  - `w`, through [`assert_nonempty_finite_val`](@ref): `!isempty(w)` and `all(isfinite, w)`.

  - `val`, through [`assert_nonempty`](@ref) and [`assert_nonneg`](@ref):

      + `AbstractVector`: `!isempty(val)` and `all(x -> x >= 0, val)`.
      + `Number`: `val >= 0`.

    A bound of `+Inf` leaves an asset with no bound, so the constructor does not check that `val` is finite. It refuses `NaN` and `-Inf`, because neither is `>= 0`.

  - `length(w) == length(val)` if `val` is an `AbstractVector`. The constructor raises a `DimensionMismatch` if this rule breaks.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `val`: Sliced to the selected indices through [`port_opt_view`](@ref).

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

  - [`set_turnover_constraints!`](@ref): writes the absolute value of each asset as one `NormOneCone` row on an auxiliary variable, and bounds that variable by `val`.
  - [`_set_turnover_constraints!`](@ref)
  - [`set_turnover_fees!`](@ref): reads `val` as a fee rate per asset, not as a bound.
  - [`TurnoverEstimator`](@ref)
  - [`AbstractResult`](@ref)
  - [`VecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`turnover_constraints`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref)
  - [`Fees`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 9.1, Equations 9.10 and 9.11. $(ref_dict[:no_original_source])
"""
@propagatable @concrete struct Turnover <: AbstractResult
    """
    $(field_dict[:w_tn])
    """
    @vprop w
    """
    $(field_dict[:val_tn])
    """
    @vprop val
    """
    $(field_dict[:fixed])
    """
    fixed
    function Turnover(w::VecNum, val::Num_VecNum, fixed::Bool)::Turnover
        assert_nonempty_finite_val(w, :w)
        # A bound of `+Inf` leaves an asset with no bound, and the `dval = +Inf` of a
        # `TurnoverEstimator` gives it. So this constructor does not check that `val`
        # is finite.
        assert_nonempty(val, :val)
        assert_nonneg(val, :val)
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

Replace the reference weights of a [`Turnover`](@ref), unless its `fixed` field is `true`.

The `fixed` field selects the weight vector that the result holds. A `fixed` constraint keeps the reference weights that it was built with, so the function ignores `w` and returns `tn`.

# Algorithm

 1. If `tn.fixed` is `true`, return `tn` unchanged. The result holds the reference weights `tn.w`, and the argument `w` has no effect.
 2. If `tn.fixed` is `false`, build a new [`Turnover`](@ref) whose `w` is the argument `w`, and whose `val` and `fixed` are those of `tn`. The result holds the argument `w`.

# Arguments

  - `tn`: The constraint to update. It gives the turnover bounds and the `fixed` flag.
  - `w`: New reference weight vector.

# Validation

  - Step 2 builds a `Turnover`, so its constructor checks `w`. Step 1 checks nothing.

# Returns

  - `tn::Turnover`: `tn` if `tn.fixed` is `true`. Otherwise, a new constraint with `w` as its reference weights.

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
    turnover_constraints(tns::AbstractVector{<:Turnover}, ::Nothing, args...; kwargs...)

Return a resolved turnover constraint, a vector of them, or `nothing`, unchanged.

A [`Turnover`](@ref) holds one bound per asset, so it needs no universe. A vector of them with no universe also returns unchanged, as a vector of [`LinearConstraint`](@ref) does. The constructors of [`JuMPOptimiser`](@ref) and of [`ProgrammeAllocationSet`](@ref) refuse a vector that holds a [`TurnoverEstimator`](@ref) when they have no universe, so such a vector never reaches this method.

# Algorithm

 1. Return the first argument unchanged. The method reads no other argument and no keyword.

# Arguments

  - `tn`: A [`Turnover`](@ref), or `nothing`.
  - `tns`: A vector of [`Turnover`](@ref).
  - `args...`: More positional arguments. The method ignores them.
  - `kwargs...`: Keyword arguments. The method ignores them.

# Returns

  - `tn::Option{<:Turnover}`: The input constraint, unchanged.
  - `tns::AbstractVector{<:Turnover}`: The input vector, unchanged.

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
function turnover_constraints(tns::AbstractVector{<:Turnover}, ::Nothing, args...;
                              kwargs...)::AbstractVector{<:Turnover}
    return tns
end
"""
    const TnE_Tn = Union{Turnover, TurnoverEstimator}

One turnover constraint, before or after the resolution of its names. Both types have the fields `w`, `val` and `fixed`, and [`factory`](@ref), [`port_opt_view`](@ref) and [`needs_previous_weights`](@ref) read these fields on either type. The two types differ only in `val`. A [`Turnover`](@ref) holds one bound per asset, and a [`TurnoverEstimator`](@ref) can hold the bounds by name.

# Related

  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`turnover_constraints`](@ref): resolves a `TurnoverEstimator` to a `Turnover`.
  - [`FeesEstimator`](@ref): holds one of these in its `tn` field.
"""
const TnE_Tn = Union{Turnover, TurnoverEstimator}
"""
    const VecTnE_Tn = AbstractVector{<:TnE_Tn}

Vector of [`TnE_Tn`](@ref). The vector methods of this file dispatch on it, because a model can hold more than one turnover constraint. A vector can hold both types. So the methods that build a new vector pass it through [`concrete_typed_array_if_abstract`](@ref), which narrows its element type from an abstract type to a `Union` of the two types.

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

Vector of [`Turnover`](@ref), with no [`TurnoverEstimator`](@ref). It is the subset of [`VecTnE_Tn`](@ref) after the resolution of the names. The vector method of [`turnover_constraints`](@ref) returns it, so a method that dispatches on it reads `val` as one value per asset.

# Related

  - [`Turnover`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Tn_VecTn`](@ref)
  - [`turnover_constraints`](@ref)
"""
const VecTn = AbstractVector{<:Turnover}
"""
    const Tn_VecTn = Union{Turnover, VecTn}

One resolved turnover constraint, or a vector of them. [`ProcessedJuMPOptimiserAttributes`](@ref) holds this type after the resolution of the names, and [`set_turnover_constraints!`](@ref) reads it. A method that dispatches on it never receives a [`TurnoverEstimator`](@ref).

# Related

  - [`Turnover`](@ref)
  - [`VecTn`](@ref)
  - [`TnE_Tn_VecTnE_Tn`](@ref): the unresolved counterpart, which the optimiser accepts from the caller.
  - [`set_turnover_constraints!`](@ref)
"""
const Tn_VecTn = Union{Turnover, VecTn}
"""
    const TnE_Tn_VecTnE_Tn = Union{TnE_Tn, VecTnE_Tn}

One turnover constraint or estimator, or a vector of them. It is the widest turnover alias. The `tn` keyword of [`JuMPOptimiser`](@ref) and of [`ProgrammeAllocationSet`](@ref) accepts it, because a caller can give either type, and one entry or a vector. [`turnover_constraints`](@ref) resolves each case of it to a [`Tn_VecTn`](@ref).

# Related

  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`Turnover`](@ref)
  - [`TurnoverEstimator`](@ref)
  - [`Tn_VecTn`](@ref): what this narrows to once the names are resolved.
  - [`turnover_constraints`](@ref)
"""
const TnE_Tn_VecTnE_Tn = Union{TnE_Tn, VecTnE_Tn}
"""
    turnover_constraints(tn::VecTnE_Tn, sets::UniverseSets; datatype::DataType = Float64,
                         strict::Bool = false)

Apply [`turnover_constraints`](@ref) to each entry of a vector of turnover constraints or estimators.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`turnover_constraints`](@ref) on `tni` with `sets`, `datatype` and `strict`. An entry that is already a [`Turnover`](@ref) passes through.
 2. Collect the results into a vector that preserves the order of `tn`.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `sets`: [`UniverseSets`](@ref) that holds the asset names of the universe and the groups.
  - `datatype`: Element type of the bounds of an estimator whose `dval` is `nothing`.
  - `strict`: If `true`, a key that names no asset and no group raises an error. If `false`, it logs a warning.

# Returns

  - `res::VecTn`: One resolved turnover constraint per entry of `tn`, in the order of `tn`.

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

Replace the reference weights of each entry of a vector of turnover constraints or estimators.

The method does what the generic vector method of [`factory`](@ref) does, and then calls [`concrete_typed_array_if_abstract`](@ref). The element type of a vector that holds a [`Turnover`](@ref) and a [`TurnoverEstimator`](@ref) is then a `Union` of the two types, not an abstract type.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`factory`](@ref) on `tni` with `w`. A `fixed` entry returns itself with its own reference weights, so the result can hold entries of both kinds.
 2. Pass the vector of step 1 through [`concrete_typed_array_if_abstract`](@ref). The two turnover types have no common concrete type, so step 1 can give the vector an abstract element type. This step narrows it to a `Union` of the element types.

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `w`: New reference weight vector.

# Returns

  - `res::VecTnE_Tn`: One updated entry per entry of `tn`, in the order of `tn`.

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

View each entry of a vector of turnover constraints or estimators at a subset of the assets.

The method does what the generic vector method of [`port_opt_view`](@ref) does, and then calls [`concrete_typed_array_if_abstract`](@ref), for the reason that [`factory(tn::VecTnE_Tn, w::VecNum)`](@ref) gives.

# Algorithm

 1. For each entry `tni` of `tn`, in the order of `tn`, call [`port_opt_view`](@ref) on `tni` with `i` and `args...`. The `@vprop` tags of each type select the fields to slice. `w` and a vector `val` become views at `i`, and a number or a dictionary `val` stays unchanged.
 2. Pass the vector of step 1 through [`concrete_typed_array_if_abstract`](@ref).

# Arguments

  - `tn`: Vector of turnover constraints or estimators.
  - `i`: Index or indices of the subset of assets.
  - `args...`: More arguments, which the method gives unchanged to the [`port_opt_view`](@ref) of each entry.

# Returns

  - `res::VecTnE_Tn`: One view per entry of `tn`, in the order of `tn`.

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

Tell whether a turnover constraint or estimator needs the previous portfolio weights.

A `fixed` entry keeps its own reference weights, so it needs no previous weights, and the method for one entry returns `!tn.fixed`. The vector method reduces with `any`, not with `all`. So one entry that is not `fixed` makes the whole vector need the previous weights.

# Algorithm

 1. On a single [`TnE_Tn`](@ref), return `!tn.fixed`.
 2. On a [`VecTnE_Tn`](@ref), apply step 1 to every entry and reduce with `any`.

# Arguments

  - `tn`: One turnover constraint or estimator, or a vector of them.

# Returns

  - `Bool`: `true` if `tn` needs the previous weights, else `false`.

# Related

  - [`TurnoverEstimator`](@ref)
  - [`Turnover`](@ref)
  - [`TnE_Tn`](@ref)
  - [`VecTnE_Tn`](@ref)
  - [`factory(tn::Turnover, w::VecNum)`](@ref): reads the same `fixed` flag.
"""
function needs_previous_weights(tn::TnE_Tn)::Bool
    return !tn.fixed
end
function needs_previous_weights(tn::VecTnE_Tn)::Bool
    return any(needs_previous_weights, tn)
end

export TurnoverEstimator, Turnover, turnover_constraints
