"""
$(DocStringExtensions.TYPEDEF)

Supertype of the policies that resolve a blank cell of a raw Panel Field.

All concrete types stating what a Panel Field's blank cell becomes should subtype `AbstractPanelFillAlgorithm`.

A blank never reaches a carrier. [`asset_panel`](@ref) resolves every one of them, so every Panel Field comes out finite. The policy says what the resolved value is; the observed mask the Panel Field also carries says which cells the resolution touched.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractPanelFillAlgorithm` and implement the following methods:

## `panel_fill`

  - `panel_fill(alg::AbstractPanelFillAlgorithm, v::AbstractVector, name::AbstractString) -> Vector`: One asset's column of one raw Panel Field, along the observation axis, with every blank resolved.

### Arguments

  - `alg`: The concrete subtype instance.
  - `v`: One asset's raw values along the observation axis, blanks included.
  - `name`: The Panel Field's name, displayed in an error message.

### Returns

  - `filled::Vector`: The same length as `v`, and free of blanks.

# Related

  - [`NoPanelFill`](@ref)
  - [`ConstantPanelFill`](@ref)
  - [`ForwardPanelFill`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
abstract type AbstractPanelFillAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Refuses a blank cell instead of resolving one.

This is the default, and it is the right policy for a Panel Field that is complete by construction. A Panel Field carrying this policy contributes **no** observed-mask column, because it has nothing to record: every cell is observed or the build throws.

# Constructors

    NoPanelFill() -> NoPanelFill

# Examples

```jldoctest
julia> NoPanelFill()
NoPanelFill()
```

# Related

  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`ConstantPanelFill`](@ref)
  - [`ForwardPanelFill`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
struct NoPanelFill <: AbstractPanelFillAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Resolves every blank cell to one constant.

This is the policy for a quantity whose absence *means* a value: a zero dividend before the first payment, or a residual category before a classification exists.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConstantPanelFill(;
        val::Union{<:Number, <:AbstractString} = 0.0
    ) -> ConstantPanelFill

Keywords correspond to the struct's fields.

## Validation

  - If `val` is a `Number`, `isfinite(val)`. A non-finite fill would put an infinity into a Panel Field, which [`assert_panel_finite`](@ref) refuses.

# Examples

```jldoctest
julia> ConstantPanelFill()
ConstantPanelFill
  val ┴ Float64: 0.0
```

# Related

  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`NoPanelFill`](@ref)
  - [`ForwardPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
@concrete struct ConstantPanelFill <: AbstractPanelFillAlgorithm
    """
    The value every blank cell becomes: a number for a numeric or tensor Panel Field, a level label for a categorical one.
    """
    val
    function ConstantPanelFill(val::Union{<:Number, <:AbstractString})
        if isa(val, Number)
            assert_finite(val, :val)
        end
        return new{typeof(val)}(val)
    end
end
function ConstantPanelFill(;
                           val::Union{<:Number, <:AbstractString} = 0.0)::ConstantPanelFill
    return ConstantPanelFill(val)
end
"""
$(DocStringExtensions.TYPEDEF)

Resolves a blank cell to the nearest earlier observed value of the same asset.

This is the safe fill over a cross-validation fold. It looks **backward** along the observation axis, so a fold that starts later reads only rows that fold already holds, and the value a fold computes does not depend on rows outside it.

A leading blank has no earlier value to take, so it falls through to `val`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ForwardPanelFill(;
        val::Union{<:Number, <:AbstractString} = 0.0,
        lim::Option{<:Integer} = nothing
    ) -> ForwardPanelFill

Keywords correspond to the struct's fields.

## Validation

  - If `val` is a `Number`, `isfinite(val)`.
  - If `lim` is not `nothing`, `lim > 0`.

# Examples

```jldoctest
julia> ForwardPanelFill()
ForwardPanelFill
  val ┼ Float64: 0.0
  lim ┴ nothing
```

# Related

  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`ConstantPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
@concrete struct ForwardPanelFill <: AbstractPanelFillAlgorithm
    """
    The value a cell becomes when the fill reaches it and no earlier observed value is available, or the run of blanks is longer than `lim`.
    """
    val
    """
    Longest run of consecutive blanks the fill carries a value across, or `nothing` for no limit.
    """
    lim
    function ForwardPanelFill(val::Union{<:Number, <:AbstractString},
                              lim::Option{<:Integer})
        assert_panel_fill(val, lim)
        return new{typeof(val), typeof(lim)}(val, lim)
    end
end
function ForwardPanelFill(; val::Union{<:Number, <:AbstractString} = 0.0,
                          lim::Option{<:Integer} = nothing)::ForwardPanelFill
    return ForwardPanelFill(val, lim)
end
"""
$(DocStringExtensions.TYPEDEF)

Resolves a blank cell to the nearest later observed value of the same asset.

!!! warning "This policy looks forward, and it leaks across a fold boundary"

    [`asset_panel`](@ref) runs **once**, over the whole history, and has no fold machinery. A backward fill therefore carries a value from an observation into an earlier one, and a fold that ends before the source row still sees the value the source row supplied. A cross-validation score computed over a panel built this way is optimistic, and the size of the leak is the length of the blank run. Use [`ForwardPanelFill`](@ref) for anything a fold will read.

    The policy is offered rather than refused because a Panel Field may be built outside any fold, where nothing looks forward into anything.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BackwardPanelFill(;
        val::Union{<:Number, <:AbstractString} = 0.0,
        lim::Option{<:Integer} = nothing
    ) -> BackwardPanelFill

Keywords correspond to the struct's fields.

## Validation

  - If `val` is a `Number`, `isfinite(val)`.
  - If `lim` is not `nothing`, `lim > 0`.

# Examples

```jldoctest
julia> BackwardPanelFill()
BackwardPanelFill
  val ┼ Float64: 0.0
  lim ┴ nothing
```

# Related

  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`ForwardPanelFill`](@ref)
  - [`ConstantPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
@concrete struct BackwardPanelFill <: AbstractPanelFillAlgorithm
    """
    The value a cell becomes when the fill reaches it and no later observed value is available, or the run of blanks is longer than `lim`.
    """
    val
    """
    Longest run of consecutive blanks the fill carries a value across, or `nothing` for no limit.
    """
    lim
    function BackwardPanelFill(val::Union{<:Number, <:AbstractString},
                               lim::Option{<:Integer})
        assert_panel_fill(val, lim)
        return new{typeof(val), typeof(lim)}(val, lim)
    end
end
function BackwardPanelFill(; val::Union{<:Number, <:AbstractString} = 0.0,
                           lim::Option{<:Integer} = nothing)::BackwardPanelFill
    return BackwardPanelFill(val, lim)
end
"""
    assert_panel_fill(val::Union{<:Number, <:AbstractString}, lim::Option{<:Integer}) -> nothing

Check the terminal value and the run limit shared by the two directional fill policies.

# Algorithm

 1. When `val` is a `Number`, check that it is finite.
 2. When `lim` is not `nothing`, check that it is `> 0`.

# Arguments

  - `val`: The value a cell becomes when the fill has nothing to carry into it.
  - `lim`: Longest run of consecutive blanks the fill carries a value across, or `nothing`.

# Validation

  - `isfinite(val)` when `val` is a `Number`. Raises an [`IsNonFiniteError`](@ref).
  - `lim > 0` when `lim` is not `nothing`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`ForwardPanelFill`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`Option`](@ref)
"""
function assert_panel_fill(val::Union{<:Number, <:AbstractString},
                           lim::Option{<:Integer})::Nothing
    if isa(val, Number)
        assert_finite(val, :val)
    end
    if !isnothing(lim)
        assert_gt0(lim, :lim)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether one raw cell of a Panel Field is blank.

A raw Panel Field carries its blanks in whichever of the two conventions its source used, and both mean the same thing here: `missing`, and a floating-point `NaN`.

# Algorithm

 1. Return `true` when the cell is `missing`.
 2. Return `true` when the cell is a `Number` and `isnan` of it.
 3. Return `false` otherwise.

# Arguments

  - `x`: One raw cell.

# Returns

  - `blank::Bool`: Whether the cell is blank.

# Related

  - [`asset_panel`](@ref)
  - [`AbstractPanelFillAlgorithm`](@ref)
"""
function is_panel_blank(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
"""
    panel_fill(alg::NoPanelFill, v::AbstractVector, name::AbstractString) -> Vector
    panel_fill(alg::ConstantPanelFill, v::AbstractVector, name::AbstractString) -> Vector
    panel_fill(alg::ForwardPanelFill, v::AbstractVector, name::AbstractString) -> Vector
    panel_fill(alg::BackwardPanelFill, v::AbstractVector, name::AbstractString) -> Vector

Resolve the blanks of one asset's column of one raw Panel Field, along the observation axis.

# Algorithm

The method that Julia selects is the algorithm, and the four differ in where the replacement comes from.

 1. [`NoPanelFill`](@ref): throw when any cell is blank, naming the Panel Field and the first offending observation. Otherwise return the column unchanged.
 2. [`ConstantPanelFill`](@ref): replace every blank by `alg.val`.
 3. [`ForwardPanelFill`](@ref): walk the observations in order, carrying the last observed value. Fill a blank with the carried value while the run of blanks is no longer than `alg.lim`, and with `alg.val` otherwise.
 4. [`BackwardPanelFill`](@ref): the same walk, in reverse order. This looks forward in time; the type's docstring states what that costs a fold.

# Arguments

  - `alg`: The fill policy.
  - `v`: One asset's raw values along the observation axis, blanks included.
  - `name`: The Panel Field's name, displayed in the [`NoPanelFill`](@ref) error message.

# Validation

  - Under [`NoPanelFill`](@ref), `v` carries no blank. Raises an `ArgumentError`.

# Returns

  - `filled::Vector`: The same length as `v`, and free of blanks.

# Related

  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`is_panel_blank`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_fill(::NoPanelFill, v::AbstractVector, name::AbstractString)
    i = findfirst(is_panel_blank, v)
    @argcheck(isnothing(i),
              ArgumentError("the Panel Field \"$name\" carries a blank cell at position $i, and its fill policy is NoPanelFill, which refuses one. A blank never reaches a carrier, so give the field a fill policy — ForwardPanelFill is the one that is safe across a cross-validation fold — or remove the blank from the raw input."))
    return collect(v)
end
function panel_fill(alg::ConstantPanelFill, v::AbstractVector, ::AbstractString)
    return [is_panel_blank(x) ? alg.val : x for x in v]
end
function panel_fill(alg::ForwardPanelFill, v::AbstractVector, ::AbstractString)
    return panel_directional_fill(v, alg.val, alg.lim, eachindex(v))
end
function panel_fill(alg::BackwardPanelFill, v::AbstractVector, ::AbstractString)
    return panel_directional_fill(v, alg.val, alg.lim, reverse(eachindex(v)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Carry the last observed value along one traversal of the observation axis.

The one body behind [`ForwardPanelFill`](@ref) and [`BackwardPanelFill`](@ref): the two differ only in the order they walk `v`, so they pass a different `order` and share everything else.

# Algorithm

 1. Start with no carried value and a run length of zero.
 2. Walk `order`. On an observed cell, take it as the carried value and reset the run to zero.
 3. On a blank cell, raise the run by one. Write the carried value when one exists and the run is within `lim`, and write `val` otherwise.

# Arguments

  - `v`: One asset's raw values along the observation axis, blanks included.
  - `val`: The value a cell becomes when nothing can be carried into it.
  - `lim`: Longest run of consecutive blanks a value is carried across, or `nothing` for no limit.
  - `order`: The traversal order of the observation axis.

# Returns

  - `filled::Vector`: The same length as `v`, and free of blanks.

# Related

  - [`ForwardPanelFill`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`panel_fill`](@ref)
"""
function panel_directional_fill(v::AbstractVector, val, lim::Option{<:Integer}, order)
    out = Vector{typeof(val)}(undef, length(v))
    carry = nothing
    run = 0
    for i in order
        x = v[i]
        if !is_panel_blank(x)
            carry = x
            run = 0
            out[i] = x
        else
            run += 1
            ok = !isnothing(carry) && (isnothing(lim) || run <= lim)
            out[i] = ok ? carry : val
        end
    end
    return out
end
"""
$(DocStringExtensions.TYPEDEF)

Supertype of the raw, blank-carrying forms one Panel Field enters [`asset_panel`](@ref) in.

All concrete types holding one Panel Field's raw values, its metadata and its fill policy should subtype `AbstractPanelFieldInput`.

An input is **not** a carrier and never becomes one. It holds the blanks, and [`asset_panel`](@ref) resolves them on the way into `Z`; nothing downstream ever sees an unresolved panel. This is why the blank-carrying form is a plain argument to the builder rather than a preprocessing estimator: an estimator fitted inside a fold would need a carrier for the unfilled panel, and the all-finite rule on `Z` gives it none.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractPanelFieldInput` and implement the following methods:

## `panel_resolve`

  - `panel_resolve(inp::AbstractPanelFieldInput) -> Tuple`: The Panel Field's values with every blank resolved, and the boolean array recording which cells were observed.

### Arguments

  - `inp`: The concrete subtype instance.

### Returns

  - `vals::AbstractArray`: The resolved values, in the raw input's own shape.
  - `obs::AbstractArray{Bool}`: Whether each raw cell was observed. Its trailing axes number `panel_field_observables` of the Panel Field's kind.

## `panel_input_kind`

  - `panel_input_kind(inp::AbstractPanelFieldInput, vals::AbstractArray) -> AbstractPanelFieldKind`: The kind the input builds, read from the resolved values where the input left the metadata to be derived.

### Arguments

  - `inp`: The concrete subtype instance.
  - `vals`: The resolved values, as [`panel_resolve`](@ref) returned them.

### Returns

  - `kind::AbstractPanelFieldKind`: The Panel Field's kind.

## `panel_write!`

  - `panel_write!(Z::AbstractArray, kind::AbstractPanelFieldKind, vals::AbstractArray, cols::VecInt) -> nothing`: Write the resolved values into the value columns of `Z`.

### Arguments

  - `Z`: The feature matrix under construction, `observations × assets × features`.
  - `kind`: The Panel Field's kind.
  - `vals`: The resolved values.
  - `cols`: The columns of `Z` the kind claims, in its own column order.

### Returns

  - `nothing`.

# Related

  - [`NumericPanelInput`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`TensorPanelInput`](@ref)
  - [`asset_panel`](@ref)
"""
abstract type AbstractPanelFieldInput <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Raw form of a Panel Field holding one numeric quantity per observation and asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NumericPanelInput(;
        name::AbstractString,
        vals::AbstractMatrix,
        alg::AbstractPanelFillAlgorithm = NoPanelFill()
    ) -> NumericPanelInput

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(name)`.
  - `!isempty(vals)`.

# Examples

```jldoctest
julia> NumericPanelInput(; name = \"mcap\", vals = [1.0 2.0; 3.0 4.0])
NumericPanelInput
  name ┼ String: "mcap"
  vals ┼ 2×2 Matrix{Float64}
   alg ┴ NoPanelFill()
```

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`TensorPanelInput`](@ref)
  - [`asset_panel`](@ref)
"""
@concrete struct NumericPanelInput <: AbstractPanelFieldInput
    """
    The Panel Field's name, which names its column of a derived Feature Matrix.
    """
    name
    """
    Raw values, blanks included: `assets` when static, `observations × assets` when time-varying. A blank is a `missing` or a `NaN`.
    """
    vals
    """
    The policy that resolves the blanks.
    """
    alg
    function NumericPanelInput(name::AbstractString, vals::AbstractArray,
                               alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals, 1, 2)
        return new{typeof(name), typeof(vals), typeof(alg)}(name, vals, alg)
    end
end
function NumericPanelInput(; name::AbstractString, vals::AbstractArray,
                           alg::AbstractPanelFillAlgorithm = NoPanelFill())::NumericPanelInput
    return NumericPanelInput(name, vals, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Raw form of a Panel Field holding one category label per observation and asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CategoricalPanelInput(;
        name::AbstractString,
        vals::AbstractMatrix,
        levels::Option{<:VecStr} = nothing,
        alg::AbstractPanelFillAlgorithm = NoPanelFill()
    ) -> CategoricalPanelInput

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(name)`.
  - `!isempty(vals)`.
  - If `levels` is not `nothing`, it passes [`assert_panel_labels`](@ref).

# Examples

```jldoctest
julia> CategoricalPanelInput(; name = \"sector\", vals = [\"T\" \"E\"; \"T\" \"E\"])
CategoricalPanelInput
    name ┼ String: "sector"
    vals ┼ 2×2 Matrix{String}
  levels ┼ nothing
     alg ┴ NoPanelFill()
```

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`NumericPanelInput`](@ref)
  - [`TensorPanelInput`](@ref)
  - [`asset_panel`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct CategoricalPanelInput <: AbstractPanelFieldInput
    """
    The Panel Field's name, which prefixes each of its columns of a derived Feature Matrix.
    """
    name
    """
    Raw labels, blanks included: `assets` when static, `observations × assets` when time-varying. A blank is a `missing`.
    """
    vals
    """
    The category levels, in column order, or `nothing` to read them off the resolved labels in sorted order.
    """
    levels
    """
    The policy that resolves the blanks. A `val` it carries must itself be a level.
    """
    alg
    function CategoricalPanelInput(name::AbstractString, vals::AbstractArray,
                                   levels::Option{<:VecStr},
                                   alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals, 1, 2)
        if !isnothing(levels)
            assert_panel_labels(levels, :levels)
        end
        return new{typeof(name), typeof(vals), typeof(levels), typeof(alg)}(name, vals,
                                                                            levels, alg)
    end
end
function CategoricalPanelInput(; name::AbstractString, vals::AbstractArray,
                               levels::Option{<:VecStr} = nothing,
                               alg::AbstractPanelFillAlgorithm = NoPanelFill())::CategoricalPanelInput
    return CategoricalPanelInput(name, vals, levels, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Raw form of a Panel Field whose third axis carries its own labels, and optionally its own groups.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TensorPanelInput(;
        name::AbstractString,
        vals::AbstractArray{<:Any, 3},
        axis::AbstractString,
        labels::VecStr,
        groups::Option{<:VecStr} = nothing,
        alg::AbstractPanelFillAlgorithm = NoPanelFill()
    ) -> TensorPanelInput

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(name)`.
  - `!isempty(vals)`.
  - `size(vals, 3) == length(labels)`.
  - `axis`, `labels` and `groups` are checked by [`TensorPanelField`](@ref), which this input builds.

# Examples

```jldoctest
julia> TensorPanelInput(; name = \"beta\", vals = ones(2, 2, 1), axis = \"factor\", labels = [\"size\"])
TensorPanelInput
    name ┼ String: "beta"
    vals ┼ Array{Float64, 3}: [1.0 1.0; 1.0 1.0;;;]
    axis ┼ String: "factor"
  labels ┼ Vector{String}: ["size"]
  groups ┼ nothing
     alg ┴ NoPanelFill()
```

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`TensorPanelField`](@ref)
  - [`NumericPanelInput`](@ref)
  - [`CategoricalPanelInput`](@ref)
  - [`asset_panel`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct TensorPanelInput <: AbstractPanelFieldInput
    """
    The Panel Field's name, which prefixes each of its columns of a derived Feature Matrix.
    """
    name
    """
    Raw values, blanks included: `assets × labels` when static, `observations × assets × labels` when time-varying. A blank is a `missing` or a `NaN`.
    """
    vals
    """
    Name of what the third axis represents, such as `"factor"`.
    """
    axis
    """
    Labels of the third-axis entries, one per third-axis entry of `vals`.
    """
    labels
    """
    Optional group of each third-axis entry, such as a Factor Family, one per label.
    """
    groups
    """
    The policy that resolves the blanks.
    """
    alg
    function TensorPanelInput(name::AbstractString, vals::AbstractArray,
                              axis::AbstractString, labels::VecStr,
                              groups::Option{<:VecStr}, alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals, 2, 3)
        @argcheck(size(vals, ndims(vals)) == length(labels),
                  DimensionMismatch("the tensor Panel Field \"$name\" needs one label per trailing-axis entry of vals, got $(size(vals, ndims(vals))) trailing entries and length(labels) = $(length(labels))"))
        return new{typeof(name), typeof(vals), typeof(axis), typeof(labels), typeof(groups),
                   typeof(alg)}(name, vals, axis, labels, groups, alg)
    end
end
function TensorPanelInput(; name::AbstractString, vals::AbstractArray, axis::AbstractString,
                          labels::VecStr, groups::Option{<:VecStr} = nothing,
                          alg::AbstractPanelFillAlgorithm = NoPanelFill())::TensorPanelInput
    return TensorPanelInput(name, vals, axis, labels, groups, alg)
end
"""
    assert_panel_input(name::AbstractString, vals::AbstractArray) -> nothing

Check the name and the raw values shared by every [`AbstractPanelFieldInput`](@ref).

# Algorithm

 1. Check that `name` is not empty.
 2. Check that `vals` is not empty.

# Arguments

  - `name`: The Panel Field's name.
  - `vals`: The Panel Field's raw values.

# Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).
  - `!isempty(vals)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`NumericPanelInput`](@ref)
"""
function assert_panel_input(name::AbstractString, vals::AbstractArray, s::Integer,
                            t::Integer)::Nothing
    @argcheck(!isempty(name),
              IsEmptyError("the name of a Panel Field input cannot be empty: it is the key its Panel Field is looked up by, and it names its columns of a derived Feature Matrix"))
    @argcheck(!isempty(vals),
              IsEmptyError("the raw values (vals) of the Panel Field \"$name\" cannot be empty"))
    @argcheck(ndims(vals) == s || ndims(vals) == t,
              DimensionMismatch("the raw values (vals) of the Panel Field \"$name\" are $s-dimensional when static and $t-dimensional when time-varying, got a $(ndims(vals))-dimensional array of size $(size(vals))"))
    return nothing
end
"""
    panel_input_is_static(inp::NumericPanelInput) -> Bool
    panel_input_is_static(inp::CategoricalPanelInput) -> Bool
    panel_input_is_static(inp::TensorPanelInput) -> Bool

Return whether a raw Panel Field carries no observation axis.

The rank of the raw values is what declares the shape: a numeric or a categorical input is `assets` when static and `observations × assets` when time-varying, and a tensor input is `assets × labels` when static and `observations × assets × labels` when time-varying.

# Algorithm

The method that Julia selects is the algorithm, and the three differ only in the rank the static shape takes.

# Arguments

  - `inp`: The raw Panel Field.

# Returns

  - `static::Bool`: `true` when the raw values carry no observation axis.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`asset_panel`](@ref)
  - [`panel_is_static`](@ref)
"""
function panel_input_is_static(inp::NumericPanelInput)::Bool
    return isone(ndims(inp.vals))
end
function panel_input_is_static(inp::CategoricalPanelInput)::Bool
    return isone(ndims(inp.vals))
end
function panel_input_is_static(inp::TensorPanelInput)::Bool
    return ndims(inp.vals) == 2
end
"""
    assert_panel_input_fill(inp::AbstractPanelFieldInput) -> nothing

Check that a static raw Panel Field does not carry a directional fill policy.

[`ForwardPanelFill`](@ref) and [`BackwardPanelFill`](@ref) carry the last observed value along the observation axis, and a static Panel Field has none. Carrying along the asset axis instead would give asset `k` the value of asset `k - 1`, which is not a fill but a fabrication, so the two are refused rather than reinterpreted. [`NoPanelFill`](@ref) and [`ConstantPanelFill`](@ref) are cell-wise and are admitted.

# Algorithm

 1. Return when the raw Panel Field is time-varying.
 2. Throw when its fill policy is directional.

# Arguments

  - `inp`: The raw Panel Field.

# Validation

  - The fill policy of a static raw Panel Field is not directional. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`panel_input_is_static`](@ref)
  - [`ForwardPanelFill`](@ref)
  - [`BackwardPanelFill`](@ref)
  - [`asset_panel`](@ref)
"""
function assert_panel_input_fill(inp::AbstractPanelFieldInput)::Nothing
    @argcheck(!(panel_input_is_static(inp) &&
                isa(inp.alg, Union{<:ForwardPanelFill, <:BackwardPanelFill})),
              ArgumentError("the Panel Field \"$(inp.name)\" is static, so it has no observation axis to carry a value along, and its fill policy $(nameof(typeof(inp.alg))) is directional. Use NoPanelFill or ConstantPanelFill, or give the raw values an observation axis."))
    return nothing
end
"""
    panel_fill_array(vals::AbstractArray, alg::AbstractPanelFillAlgorithm, name::AbstractString, tv::Bool)

Resolve the blanks of one raw Panel Field, and return the filled array.

The fill runs **per asset, along the observation axis**, which is the only axis a point-in-time panel blanks along: an asset lists late or delists, so its history has a head or a tail of blanks, and the cross-section at one observation is not the thing being carried across.

A static raw Panel Field has no observation axis. Its two admitted policies are cell-wise, so the whole array is resolved as one flat run and reshaped back.

# Algorithm

 1. When the raw Panel Field is static, resolve `vec(vals)` with [`panel_fill`](@ref) and reshape the result.
 2. Otherwise walk the trailing axes, and resolve each column of the observation axis with [`panel_fill`](@ref).

# Arguments

  - `vals`: The raw values, blanks included.
  - `alg`: The fill policy.
  - `name`: The Panel Field's name, displayed in the error messages.
  - `tv`: Whether the raw values carry an observation axis.

# Returns

  - `filled::AbstractArray`: The same size as `vals`, and free of blanks.

# Related

  - [`panel_fill`](@ref)
  - [`panel_resolve`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_fill_array(vals::AbstractArray, alg::AbstractPanelFillAlgorithm,
                          name::AbstractString, tv::Bool)
    if !tv
        return reshape(panel_fill(alg, vec(vals), name), size(vals))
    end
    cols = CartesianIndices(size(vals)[2:end])
    first = panel_fill(alg, view(vals, :, cols[1]), name)
    out = Array{eltype(first)}(undef, size(vals))
    out[:, cols[1]] = first
    for k in 2:length(cols)
        out[:, cols[k]] = panel_fill(alg, view(vals, :, cols[k]), name)
    end
    return out
end
"""
    panel_resolve(inp::NumericPanelInput) -> Tuple{Array{Float64}, BitArray}
    panel_resolve(inp::CategoricalPanelInput) -> Tuple{Array{String}, BitArray}
    panel_resolve(inp::TensorPanelInput) -> Tuple{Array{Float64}, BitArray}

Resolve one raw Panel Field's blanks, and record which of its cells were observed.

# Algorithm

The method that Julia selects is the algorithm, and the three differ only in the element type they resolve into.

 1. Fill the blanks with [`panel_fill_array`](@ref).
 2. Walk the raw cells, recording which were observed and copying the filled value into the output.
 3. Check that a numeric or a tensor Panel Field carries no non-finite value, with [`assert_panel_finite`](@ref).

# Arguments

  - `inp`: The raw Panel Field.

# Returns

  - `vals::AbstractArray`: The resolved values, the same size as the raw ones.
  - `obs::BitArray`: The observed mask, the same size as the raw values.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`panel_fill_array`](@ref)
  - [`panel_input_field`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_resolve(inp::NumericPanelInput)
    f = panel_fill_array(inp.vals, inp.alg, inp.name, !panel_input_is_static(inp))
    out = Array{Float64}(undef, size(inp.vals))
    obs = BitArray(undef, size(inp.vals))
    for i in CartesianIndices(inp.vals)
        obs[i] = !is_panel_blank(inp.vals[i])
        out[i] = f[i]
    end
    assert_panel_finite(out, inp.name)
    return out, obs
end
function panel_resolve(inp::CategoricalPanelInput)
    f = panel_fill_array(inp.vals, inp.alg, inp.name, !panel_input_is_static(inp))
    out = Array{String}(undef, size(inp.vals))
    obs = BitArray(undef, size(inp.vals))
    for i in CartesianIndices(inp.vals)
        obs[i] = !is_panel_blank(inp.vals[i])
        out[i] = string(f[i])
    end
    return out, obs
end
function panel_resolve(inp::TensorPanelInput)
    f = panel_fill_array(inp.vals, inp.alg, inp.name, !panel_input_is_static(inp))
    out = Array{Float64}(undef, size(inp.vals))
    obs = BitArray(undef, size(inp.vals))
    for i in CartesianIndices(inp.vals)
        obs[i] = !is_panel_blank(inp.vals[i])
        out[i] = f[i]
    end
    assert_panel_finite(out, inp.name)
    return out, obs
end
"""
    panel_input_field(inp::NumericPanelInput, vals, obs) -> NumericPanelField
    panel_input_field(inp::CategoricalPanelInput, vals, obs) -> CategoricalPanelField
    panel_input_field(inp::TensorPanelInput, vals, obs) -> TensorPanelField

Return the Panel Field a resolved raw Panel Field builds, deriving what the input left to be derived.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelInput`](@ref): a [`NumericPanelField`](@ref) over the resolved values.
 2. [`CategoricalPanelInput`](@ref): a [`CategoricalPanelField`](@ref) over `inp.levels`, or, when that is `nothing`, over the distinct resolved labels in sorted order. The resolved labels are read rather than the raw ones, so a level that only a fill policy introduces still gets a code. Each label is then encoded to its level's position.
 3. [`TensorPanelInput`](@ref): a [`TensorPanelField`](@ref) over the input's own axis, labels and groups.

The observed mask rides only when the fill policy is not [`NoPanelFill`](@ref): a Panel Field that refuses a blank observed every cell, so a mask of it carries no information.

# Arguments

  - `inp`: The raw Panel Field.
  - `vals`: The resolved values, as [`panel_resolve`](@ref) returned them.
  - `obs`: The observed mask, as [`panel_resolve`](@ref) returned it.

# Validation

  - Every resolved label of a categorical Panel Field is one of its levels. Raises an `ArgumentError`.

# Returns

  - `f::AbstractPanelField`: The Panel Field.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`AbstractPanelField`](@ref)
  - [`panel_resolve`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_input_field(inp::NumericPanelInput, vals::AbstractArray{Float64},
                           obs::BitArray)
    return NumericPanelField(; name = inp.name, vals = vals,
                             omsk = isa(inp.alg, NoPanelFill) ? nothing : obs)
end
function panel_input_field(inp::CategoricalPanelInput, vals::AbstractArray{String},
                           obs::BitArray)
    levels = isnothing(inp.levels) ? sort!(unique(vals)) : String.(inp.levels)
    pos = Dict(l => k for (k, l) in pairs(levels))
    codes = Array{Int}(undef, size(vals))
    for i in CartesianIndices(vals)
        k = get(pos, vals[i], 0)
        @argcheck(k > 0,
                  ArgumentError("the categorical Panel Field \"$(inp.name)\" carries the label `$(vals[i])` at $(Tuple(i)), which is not one of its levels$(did_you_mean(vals[i], levels)). Its levels are $(join(levels, ", "))"))
        codes[i] = k
    end
    return CategoricalPanelField(; name = inp.name, levels = levels, codes = codes,
                                 omsk = isa(inp.alg, NoPanelFill) ? nothing : obs)
end
function panel_input_field(inp::TensorPanelInput, vals::AbstractArray{Float64},
                           obs::BitArray)
    return TensorPanelField(; name = inp.name, axis = inp.axis, labels = inp.labels,
                            groups = inp.groups, vals = vals,
                            omsk = isa(inp.alg, NoPanelFill) ? nothing : obs)
end
"""
    asset_panel(
        inputs::AbstractVector{<:AbstractPanelFieldInput};
        amsk::Option{<:AbstractMatrix{Bool}} = nothing,
        emsk::Option{<:AbstractMatrix{Bool}} = nothing
    ) -> AssetPanel

Build the [`AssetPanel`](@ref) a carrier holds, from the raw, blank-carrying form of each Panel Field.

This is the **build seam**. It takes each Panel Field's raw values with its fill policy, and it returns the panel alone: the panel owns the values, so there is nothing else for a carrier to be handed. The blanks stop here, and every Panel Field comes out finite.

The result goes straight into the keyword the carriers have, `ReturnsResult(; nx = nx, X = X, pnl = asset_panel(inputs))`, and the same keyword reaches [`prices_to_returns`](@ref).

The **static entry** is the rank of the raw values. An input whose values carry no observation axis builds a static panel: a fundamentals table or a sector classification with no history is that shape. There [`ForwardPanelFill`](@ref) and [`BackwardPanelFill`](@ref) are refused, because there is no observation axis to carry a value along, and the mask keywords must be `nothing`, because a static panel carries no universe mask.

# Algorithm

 1. Check that `inputs` is not empty and that the Panel Field names are unique.
 2. Check each input's fill policy against its shape, with [`assert_panel_input_fill`](@ref).
 3. Resolve every input with [`panel_resolve`](@ref), which fills its blanks and records the observed cells, and build its Panel Field with [`panel_input_field`](@ref).
 4. Read the shape the Panel Fields agreed on. When it is static, check that no mask was given and return the panel.
 5. Otherwise fill in all-`true` masks for the ones that were not given, and return the panel. The [`AssetPanel`](@ref) constructor checks that every Panel Field shares one shape.

# Arguments

  - `inputs`: The raw Panel Fields, in the order their columns are derived in.
  - `amsk`: The active mask (observations × assets), or `nothing` for all-`true`.
  - `emsk`: The estimation mask (observations × assets), or `nothing` for all-`true`.

# Validation

  - `!isempty(inputs)`. Raises an [`IsEmptyError`](@ref).
  - The Panel Field names are non-empty and unique. See [`assert_panel_labels`](@ref).
  - A static input carries no directional fill policy. See [`assert_panel_input_fill`](@ref).
  - `amsk` and `emsk` are `nothing` when the Panel Fields are static. Raises a `DimensionMismatch`.

# Returns

  - `pnl::AssetPanel`: The Asset Panel.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1.0, 2.0, 3.0]),
                          CategoricalPanelInput(; name = \"sector\", vals = [\"Fin\", \"Tech\", \"Fin\"])]);

julia> panel_feature_matrix(pnl)[1]
3-element Vector{String}:
 "mcap"
 "sector=Fin"
 "sector=Tech"
```

# Related

  - [`AssetPanel`](@ref)
  - [`AbstractPanelFieldInput`](@ref)
  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`panel_input_field`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`ReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
"""
function asset_panel(inputs::AbstractVector{<:AbstractPanelFieldInput};
                     amsk::Option{<:AbstractMatrix{Bool}} = nothing,
                     emsk::Option{<:AbstractMatrix{Bool}} = nothing)
    @argcheck(!isempty(inputs),
              IsEmptyError("an Asset Panel needs at least one Panel Field input: an empty build carries no feature data"))
    assert_panel_labels([inp.name for inp in inputs], "the Panel Field names")
    pf = AbstractPanelField[]
    for inp in inputs
        assert_panel_input_fill(inp)
        v, o = panel_resolve(inp)
        push!(pf, panel_input_field(inp, v, o))
    end
    ax = panel_field_axes(pf[1])
    if isone(length(ax))
        @argcheck(isnothing(amsk) && isnothing(emsk),
                  DimensionMismatch("the Panel Fields of this build carry no observation axis, so the Asset Panel is static and carries no universe mask; drop amsk and emsk, or give the raw values an observation axis"))
        return AssetPanel(; pf = pf)
    end
    T, N = ax
    return AssetPanel(; pf = pf, amsk = isnothing(amsk) ? trues(T, N) : amsk,
                      emsk = isnothing(emsk) ? trues(T, N) : emsk)
end
export asset_panel, NumericPanelInput, CategoricalPanelInput, TensorPanelInput, NoPanelFill,
       ConstantPanelFill, ForwardPanelFill, BackwardPanelFill
