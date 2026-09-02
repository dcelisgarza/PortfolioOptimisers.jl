"""
$(DocStringExtensions.TYPEDEF)

Supertype of the policies that resolve a blank cell of a raw Panel Field.

All concrete types stating what a Panel Field's blank cell becomes should subtype `AbstractPanelFillAlgorithm`.

A blank never reaches a carrier. [`asset_panel`](@ref) resolves every one of them, so the feature matrix `Z` stays finite and `check_feature_matrix` keeps its `assert_all_finite` guarantee. The policy says what the resolved value is; the observed-mask column the Panel Field also contributes says which cells the resolution touched.

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

  - If `val` is a `Number`, `isfinite(val)`. A non-finite fill would put an infinity into `Z`, which `check_feature_matrix` refuses.

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
              ArgumentError("the Panel Field \"$name\" carries a blank cell at observation $i, and its fill policy is NoPanelFill, which refuses one. A blank never reaches a carrier, so give the field a fill policy — ForwardPanelFill is the one that is safe across a cross-validation fold — or remove the blank from the raw input."))
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
    The Panel Field's name, which names its column of `nz`.
    """
    name
    """
    Raw values (observations × assets), blanks included. A blank is a `missing` or a `NaN`.
    """
    vals
    """
    The policy that resolves the blanks.
    """
    alg
    function NumericPanelInput(name::AbstractString, vals::AbstractMatrix,
                               alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals)
        return new{typeof(name), typeof(vals), typeof(alg)}(name, vals, alg)
    end
end
function NumericPanelInput(; name::AbstractString, vals::AbstractMatrix,
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
    The Panel Field's name, which prefixes each of its one-hot columns of `nz`.
    """
    name
    """
    Raw labels (observations × assets), blanks included. A blank is a `missing`.
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
    function CategoricalPanelInput(name::AbstractString, vals::AbstractMatrix,
                                   levels::Option{<:VecStr},
                                   alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals)
        if !isnothing(levels)
            assert_panel_labels(levels, :levels)
        end
        return new{typeof(name), typeof(vals), typeof(levels), typeof(alg)}(name, vals,
                                                                            levels, alg)
    end
end
function CategoricalPanelInput(; name::AbstractString, vals::AbstractMatrix,
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
    The Panel Field's name, which prefixes each of its third-axis columns of `nz`.
    """
    name
    """
    Raw values (observations × assets × third axis), blanks included. A blank is a `missing` or a `NaN`.
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
    function TensorPanelInput(name::AbstractString, vals::AbstractArray{<:Any, 3},
                              axis::AbstractString, labels::VecStr,
                              groups::Option{<:VecStr}, alg::AbstractPanelFillAlgorithm)
        assert_panel_input(name, vals)
        @argcheck(size(vals, 3) == length(labels),
                  DimensionMismatch("the tensor Panel Field \"$name\" needs one third-axis label per third-axis entry of vals, got size(vals, 3) = $(size(vals, 3)) and length(labels) = $(length(labels))"))
        return new{typeof(name), typeof(vals), typeof(axis), typeof(labels), typeof(groups),
                   typeof(alg)}(name, vals, axis, labels, groups, alg)
    end
end
function TensorPanelInput(; name::AbstractString, vals::AbstractArray{<:Any, 3},
                          axis::AbstractString, labels::VecStr,
                          groups::Option{<:VecStr} = nothing,
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
function assert_panel_input(name::AbstractString, vals::AbstractArray)::Nothing
    @argcheck(!isempty(name),
              IsEmptyError("the name of a Panel Field input cannot be empty: it is the key its Panel Field is looked up by, and it names its columns of the feature axis"))
    @argcheck(!isempty(vals),
              IsEmptyError("the raw values (vals) of the Panel Field \"$name\" cannot be empty"))
    return nothing
end
"""
    panel_resolve(inp::NumericPanelInput) -> Tuple{Matrix{Float64}, BitMatrix}
    panel_resolve(inp::CategoricalPanelInput) -> Tuple{Matrix{String}, BitMatrix}
    panel_resolve(inp::TensorPanelInput) -> Tuple{Array{Float64, 3}, BitArray{3}}

Resolve one raw Panel Field's blanks, and record which of its cells were observed.

The fill runs **per asset, along the observation axis**, which is the only axis a point-in-time panel blanks along: an asset lists late or delists, so its history has a head or a tail of blanks and the cross-section at one observation is not the thing being carried across.

# Algorithm

The method that Julia selects is the algorithm, and the three differ only in the axes they walk.

 1. [`NumericPanelInput`](@ref): walk the assets, resolve each asset's column with [`panel_fill`](@ref), and record the observed cells.
 2. [`CategoricalPanelInput`](@ref): the same walk, over labels rather than numbers.
 3. [`TensorPanelInput`](@ref): walk the assets and the third-axis entries, and resolve each `(asset, entry)` column.

# Arguments

  - `inp`: The raw Panel Field.

# Returns

  - `vals::AbstractArray`: The resolved values, in the raw input's own shape.
  - `obs::AbstractArray{Bool}`: Whether each raw cell was observed.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`panel_fill`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_resolve(inp::NumericPanelInput)
    T, N = size(inp.vals)
    out = Matrix{Float64}(undef, T, N)
    obs = BitMatrix(undef, T, N)
    for a in axes(inp.vals, 2)
        col = view(inp.vals, :, a)
        f = panel_fill(inp.alg, col, inp.name)
        for t in axes(inp.vals, 1)
            obs[t, a] = !is_panel_blank(col[t])
            out[t, a] = f[t]
        end
    end
    assert_panel_finite(out, inp.name)
    return out, obs
end
function panel_resolve(inp::CategoricalPanelInput)
    T, N = size(inp.vals)
    out = Matrix{String}(undef, T, N)
    obs = BitMatrix(undef, T, N)
    for a in axes(inp.vals, 2)
        col = view(inp.vals, :, a)
        f = panel_fill(inp.alg, col, inp.name)
        for t in axes(inp.vals, 1)
            obs[t, a] = !is_panel_blank(col[t])
            out[t, a] = string(f[t])
        end
    end
    return out, obs
end
function panel_resolve(inp::TensorPanelInput)
    T, N, L = size(inp.vals)
    out = Array{Float64, 3}(undef, T, N, L)
    obs = BitArray{3}(undef, T, N, L)
    for l in axes(inp.vals, 3), a in axes(inp.vals, 2)
        col = view(inp.vals, :, a, l)
        f = panel_fill(inp.alg, col, inp.name)
        for t in axes(inp.vals, 1)
            obs[t, a, l] = !is_panel_blank(col[t])
            out[t, a, l] = f[t]
        end
    end
    assert_panel_finite(out, inp.name)
    return out, obs
end
"""
    assert_panel_finite(vals::AbstractArray{<:Real}, name::AbstractString) -> nothing

Check that a resolved Panel Field carries no non-finite value.

The fill policies each write a finite value by construction, so this catches a non-finite cell that the *raw input* carried and that no policy touched: an infinity is not a blank, so [`is_panel_blank`](@ref) leaves it where it stands. It would then reach `check_feature_matrix`, which names `Z` and not the Panel Field that spoiled it.

# Algorithm

 1. Find the first non-finite entry of `vals`.
 2. Throw when there is one, naming the Panel Field and the position.

# Arguments

  - `vals`: The resolved values of one Panel Field.
  - `name`: The Panel Field's name, displayed in the error message.

# Validation

  - `all(isfinite, vals)`. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`panel_resolve`](@ref)
  - [`is_panel_blank`](@ref)
  - [`asset_panel`](@ref)
"""
function assert_panel_finite(vals::AbstractArray{<:Real}, name::AbstractString)::Nothing
    i = findfirst(!isfinite, vals)
    @argcheck(isnothing(i),
              IsNonFiniteError("the Panel Field \"$name\" carries a non-finite value at $(isnothing(i) ? "" : string(Tuple(i))). An infinity is not a blank, so no fill policy resolves it; correct the raw input. The feature matrix Z is checked for finiteness by check_feature_matrix, which names Z rather than the Panel Field."))
    return nothing
end
"""
    panel_input_kind(inp::NumericPanelInput, vals::AbstractArray) -> NumericPanelField
    panel_input_kind(inp::CategoricalPanelInput, vals::AbstractArray) -> CategoricalPanelField
    panel_input_kind(inp::TensorPanelInput, vals::AbstractArray) -> TensorPanelField

Return the kind a raw Panel Field builds, deriving what the input left to be derived.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelInput`](@ref): a [`NumericPanelField`](@ref), which carries no metadata.
 2. [`CategoricalPanelInput`](@ref): a [`CategoricalPanelField`](@ref) over `inp.levels`, or, when that is `nothing`, over the distinct resolved labels in sorted order. The resolved labels are read rather than the raw ones, so a level that only a fill policy introduces still gets its column.
 3. [`TensorPanelInput`](@ref): a [`TensorPanelField`](@ref) over the input's own axis, labels and groups.

# Arguments

  - `inp`: The raw Panel Field.
  - `vals`: The resolved values, as [`panel_resolve`](@ref) returned them.

# Returns

  - `kind::AbstractPanelFieldKind`: The Panel Field's kind.

# Related

  - [`AbstractPanelFieldInput`](@ref)
  - [`AbstractPanelFieldKind`](@ref)
  - [`panel_resolve`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_input_kind(::NumericPanelInput, ::AbstractArray)::NumericPanelField
    return NumericPanelField()
end
function panel_input_kind(inp::CategoricalPanelInput,
                          vals::AbstractArray)::CategoricalPanelField
    levels = isnothing(inp.levels) ? sort!(unique(vals)) : inp.levels
    return CategoricalPanelField(; levels = levels)
end
function panel_input_kind(inp::TensorPanelInput, ::AbstractArray)::TensorPanelField
    return TensorPanelField(; axis = inp.axis, labels = inp.labels, groups = inp.groups)
end
"""
    panel_write!(Z::AbstractArray, kind::NumericPanelField, vals, cols::VecInt) -> nothing
    panel_write!(Z::AbstractArray, kind::CategoricalPanelField, vals, cols::VecInt) -> nothing
    panel_write!(Z::AbstractArray, kind::TensorPanelField, vals, cols::VecInt) -> nothing

Write one resolved Panel Field's values into the value columns of the feature matrix.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref): copy the resolved matrix into the one column.
 2. [`CategoricalPanelField`](@ref): write a `1` in the column of each cell's level. `Z` enters the call as zeros, so the other columns of that cell stay `0`, which is what one-hot means. A label that is not a level throws.
 3. [`TensorPanelField`](@ref): copy each third-axis slice into its own column.

# Arguments

  - `Z`: The feature matrix under construction, `observations × assets × features`.
  - `kind`: The Panel Field's kind.
  - `vals`: The resolved values.
  - `cols`: The columns of `Z` the kind claims, in its own column order.

# Validation

  - On the categorical method, every resolved label is one of `kind.levels`. Raises an `ArgumentError` carrying a [`did_you_mean`](@ref) suggestion.

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`panel_resolve`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_write!(Z::AbstractArray, ::NumericPanelField, vals, cols::VecInt)::Nothing
    Z[:, :, cols[1]] = vals
    return nothing
end
function panel_write!(Z::AbstractArray, kind::CategoricalPanelField, vals,
                      cols::VecInt)::Nothing
    pos = Dict(string(l) => k for (k, l) in pairs(kind.levels))
    for i in CartesianIndices(vals)
        k = get(pos, vals[i], nothing)
        @argcheck(!isnothing(k),
                  ArgumentError("the categorical Panel Field carries the label `$(vals[i])` at $(Tuple(i)), which is not one of its levels$(did_you_mean(vals[i], string.(kind.levels))). Its levels are $(join(string.(kind.levels), ", "))"))
        Z[i[1], i[2], cols[k]] = one(eltype(Z))
    end
    return nothing
end
function panel_write!(Z::AbstractArray, kind::TensorPanelField, vals, cols::VecInt)::Nothing
    for l in eachindex(kind.labels)
        Z[:, :, cols[l]] = view(vals, :, :, l)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the names of the observed-mask columns one Panel Field contributes to `nz`.

A Panel Field with a single observable takes `"<name>::observed"`. One with several takes its value column's own name with `"::observed"` appended, so a tensor Panel Field keeps one mask column per third-axis label.

The separator is `"::"` rather than the `"="` the value columns use, so a mask column cannot be mistaken for a level of the same Panel Field. It is a convention and nothing more: [`asset_panel`](@ref) checks that the whole feature axis is unique, and every consumer reaches a column through the field index rather than by reading its name.

# Algorithm

 1. Read the observable count from [`panel_field_observables`](@ref).
 2. When it is one, return the single name `"<name>::observed"`.
 3. Otherwise append `"::observed"` to each name [`panel_field_labels`](@ref) gives.

# Arguments

  - `kind`: The Panel Field's kind.
  - `name`: The Panel Field's own name.

# Returns

  - `labels::Vector{String}`: One name per observed-mask column.

# Related

  - [`panel_field_labels`](@ref)
  - [`panel_field_observables`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_observed_labels(kind::AbstractPanelFieldKind,
                               name::AbstractString)::Vector{String}
    return if isone(panel_field_observables(kind))
        ["$name::observed"]
    else
        ["$l::observed" for l in panel_field_labels(kind, name)]
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Lay the feature axis out, resolve every raw Panel Field, and build the field index.

The first half of [`asset_panel`](@ref), split off because a build does two separable things: it decides *where* each Panel Field's columns go, and it then *writes* them. The layout half is what a reader checks a column convention against.

# Algorithm

 1. Walk the inputs in order. Check that each shares the observation count `T` and the asset count `N` of the first.
 2. Resolve the input with [`panel_resolve`](@ref), and read its kind with [`panel_input_kind`](@ref).
 3. Claim the next `length(panel_field_labels(kind, name))` columns as the Panel Field's value columns, and append their names to the feature axis.
 4. When the fill policy is not [`NoPanelFill`](@ref), claim the next columns as its observed-mask columns, and append their names from [`panel_observed_labels`](@ref). A Panel Field that cannot blank claims none.
 5. Record the kind, the resolved values, the observed mask and the [`PanelField`](@ref).

# Arguments

  - `inputs`: The raw Panel Fields, in the order their columns take on the feature axis.
  - `T`: The observation count every input must share.
  - `N`: The asset count every input must share.

# Validation

  - Every input has the observation count `T` and the asset count `N`. Raises a `DimensionMismatch` naming the Panel Field and both shapes.

# Returns

  - `nz::Vector{String}`: The feature axis.
  - `kinds::Vector{AbstractPanelFieldKind}`: One kind per Panel Field, in the same order.
  - `vals::Vector{Any}`: One resolved value array per Panel Field.
  - `obss::Vector{Any}`: One observed-mask array per Panel Field.
  - `pf::Vector{PanelField}`: The field index.

# Related

  - [`asset_panel`](@ref)
  - [`panel_matrix`](@ref)
  - [`panel_resolve`](@ref)
  - [`panel_input_kind`](@ref)
"""
function panel_layout(inputs::AbstractVector{<:AbstractPanelFieldInput}, T::Integer,
                      N::Integer)
    nz = String[]
    kinds = AbstractPanelFieldKind[]
    vals = Any[]
    obss = Any[]
    pf = PanelField[]
    for inp in inputs
        @argcheck(size(inp.vals, 1) == T && size(inp.vals, 2) == N,
                  DimensionMismatch("every Panel Field of one Asset Panel shares its observation axis and its asset axis, and the Panel Field \"$(inp.name)\" does not: got $(size(inp.vals, 1)) × $(size(inp.vals, 2)) against the $T × $N of \"$(inputs[1].name)\""))
        v, o = panel_resolve(inp)
        kind = panel_input_kind(inp, v)
        cols = panel_claim!(nz, panel_field_labels(kind, inp.name))
        ocols = if isa(inp.alg, NoPanelFill)
            nothing
        else
            panel_claim!(nz, panel_observed_labels(kind, inp.name))
        end
        push!(kinds, kind)
        push!(vals, v)
        push!(obss, o)
        push!(pf, PanelField(; name = inp.name, kind = kind, cols = cols, ocols = ocols))
    end
    return nz, kinds, vals, obss, pf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Append a Panel Field's column names to the feature axis, and return the columns they took.

The one place a column index is minted, so the feature axis and the field index cannot disagree about where a Panel Field's columns are.

# Algorithm

 1. Read the feature axis's current length, which is the last column already claimed.
 2. Append `labels` to it.
 3. Return the range of columns the append occupied, as a vector.

# Arguments

  - `nz`: The feature axis under construction. It is appended to.
  - `labels`: The column names to claim.

# Returns

  - `cols::Vector{Int}`: The columns `labels` took, in order.

# Related

  - [`panel_layout`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_claim!(nz::AbstractVector{String}, labels::AbstractVector{String})
    cols = collect((length(nz) + 1):(length(nz) + length(labels)))
    append!(nz, labels)
    return cols
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Allocate the feature matrix and write every Panel Field's values and observed masks into it.

The second half of [`asset_panel`](@ref). It allocates zeros, which is what makes a one-hot column correct: [`panel_write!`](@ref) writes only the `1`s, and the rest of that cell's level columns are already `0`.

# Algorithm

 1. Allocate `Z` as `zeros(Float64, T, N, nc)`.
 2. For each Panel Field, write its values into its value columns with [`panel_write!`](@ref).
 3. Write its observed mask, one column per observable. A three-dimensional mask contributes one column per third-axis entry; a two-dimensional one contributes its single column.

# Arguments

  - `kinds`: One kind per Panel Field, as [`panel_layout`](@ref) returned them.
  - `vals`: One resolved value array per Panel Field.
  - `obss`: One observed-mask array per Panel Field.
  - `pf`: The field index.
  - `nc`: The length of the feature axis.
  - `T`: The observation count.
  - `N`: The asset count.

# Returns

  - `Z::Array{Float64, 3}`: The feature matrix, `observations × assets × features`.

# Related

  - [`asset_panel`](@ref)
  - [`panel_layout`](@ref)
  - [`panel_write!`](@ref)
"""
function panel_matrix(kinds, vals, obss, pf::AbstractVector{<:PanelField}, nc::Integer,
                      T::Integer, N::Integer)
    Z = zeros(Float64, T, N, nc)
    for (k, f) in pairs(pf)
        panel_write!(Z, kinds[k], vals[k], f.cols)
        panel_write_observed!(Z, obss[k], f.ocols)
    end
    return Z
end
"""
    panel_write_observed!(Z::AbstractArray, obs, ocols::Nothing) -> nothing
    panel_write_observed!(Z::AbstractArray, obs, ocols::VecInt) -> nothing

Write one Panel Field's observed mask into its observed-mask columns of the feature matrix.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ocols` is `nothing`: the Panel Field cannot blank, so it has no mask column and nothing is written.
 2. `ocols` is a column vector: write the mask as `0`/`1`. A three-dimensional mask writes one third-axis entry per column; a two-dimensional one writes its single column.

# Arguments

  - `Z`: The feature matrix under construction.
  - `obs`: The observed mask, as [`panel_resolve`](@ref) returned it.
  - `ocols`: The observed-mask columns, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`panel_matrix`](@ref)
  - [`panel_resolve`](@ref)
"""
function panel_write_observed!(::AbstractArray, ::Any, ::Nothing)::Nothing
    return nothing
end
function panel_write_observed!(Z::AbstractArray, obs::AbstractMatrix,
                               ocols::VecInt)::Nothing
    Z[:, :, ocols[1]] = obs
    return nothing
end
function panel_write_observed!(Z::AbstractArray, obs::AbstractArray{<:Any, 3},
                               ocols::VecInt)::Nothing
    for (l, c) in pairs(ocols)
        Z[:, :, c] = view(obs, :, :, l)
    end
    return nothing
end
"""
    asset_panel(
        inputs::AbstractVector{<:AbstractPanelFieldInput};
        amsk::Option{<:AbstractMatrix{Bool}} = nothing,
        emsk::Option{<:AbstractMatrix{Bool}} = nothing
    ) -> @NamedTuple{nz::Vector{String}, Z::Array{Float64, 3}, pnl::AssetPanel}

Build the three things a point-in-time panel enters a carrier as: its feature names, its feature matrix, and its [`AssetPanel`](@ref).

This is the **build seam**. It takes the raw, blank-carrying form of each Panel Field with its fill policy, and it returns the resolved triple. The blanks stop here: `Z` comes out finite, so `check_feature_matrix` keeps its `assert_all_finite` guarantee and no existing consumer of a feature matrix loses one.

The result splats straight into the keywords the carriers already have, `ReturnsResult(; nx = nx, X = X, asset_panel(inputs)...)`, and the same three keywords reach [`prices_to_returns`](@ref).

# Algorithm

 1. Check that `inputs` is not empty and that the Panel Field names are unique.
 2. Resolve every input with [`panel_resolve`](@ref), which fills its blanks per asset along the observation axis and records the observed cells. Check that every input has the same observation count and asset count.
 3. Read each Panel Field's kind with [`panel_input_kind`](@ref), and lay out the feature axis: the value columns from [`panel_field_labels`](@ref), then, for a Panel Field whose fill policy is not [`NoPanelFill`](@ref), the observed-mask columns from [`panel_observed_labels`](@ref).
 4. Check that the feature axis is unique, so a Panel Field's own name cannot silently collide with another's level.
 5. Allocate `Z` as zeros, write every Panel Field's values with [`panel_write!`](@ref), and write each observed mask as a `0`/`1` column beside the field it belongs to.
 6. Default a missing mask to all-true, and build the [`AssetPanel`](@ref), whose constructor holds the subset invariant.

# Arguments

  - `inputs`: The raw Panel Fields, in the order their columns take on the feature axis.
  - `amsk`: The active mask (observations × assets), or `nothing` for all-true.
  - `emsk`: The estimation mask (observations × assets), or `nothing` for all-true.

# Validation

  - `!isempty(inputs)`. Raises an [`IsEmptyError`](@ref).
  - The Panel Field names are non-empty and unique (see [`assert_panel_labels`](@ref)).
  - Every input has the same observation count and asset count. Raises a `DimensionMismatch`.
  - The feature axis it builds is unique. Raises an `ArgumentError` naming the two Panel Fields that collide.
  - The masks, and the subset invariant, are checked by [`AssetPanel`](@ref).

# Returns

  - `nz::Vector{String}`: The feature axis, one name per column of `Z`.
  - `Z::Array{Float64, 3}`: The feature matrix, `observations × assets × features`.
  - `pnl::AssetPanel`: The field index and the two masks.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"mcap\", vals = [1.0 missing; 3.0 4.0],
                                            alg = ForwardPanelFill(; val = 0.0))]);

julia> res.nz
2-element Vector{String}:
 "mcap"
 "mcap::observed"

julia> res.Z[:, :, 1]
2×2 Matrix{Float64}:
 1.0  0.0
 3.0  4.0

julia> res.Z[:, :, 2]
2×2 Matrix{Float64}:
 1.0  0.0
 1.0  1.0
```

# Related

  - [`AssetPanel`](@ref)
  - [`AbstractPanelFieldInput`](@ref)
  - [`AbstractPanelFillAlgorithm`](@ref)
  - [`ReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
"""
function asset_panel(inputs::AbstractVector{<:AbstractPanelFieldInput};
                     amsk::Option{<:AbstractMatrix{Bool}} = nothing,
                     emsk::Option{<:AbstractMatrix{Bool}} = nothing)
    @argcheck(!isempty(inputs),
              IsEmptyError("an Asset Panel needs at least one Panel Field input: an empty build describes no column of the feature axis"))
    assert_panel_labels([inp.name for inp in inputs], "the Panel Field names")
    T, N = size(inputs[1].vals, 1), size(inputs[1].vals, 2)
    nz, kinds, vals, obss, pf = panel_layout(inputs, T, N)
    assert_panel_feature_axis(nz, pf)
    Z = panel_matrix(kinds, vals, obss, pf, length(nz), T, N)
    return (; nz = nz, Z = Z,
            pnl = AssetPanel(; pf = pf, amsk = isnothing(amsk) ? trues(T, N) : amsk,
                             emsk = isnothing(emsk) ? trues(T, N) : emsk))
end
"""
    assert_panel_feature_axis(nz::VecStr, pf::AbstractVector{<:PanelField}) -> nothing

Check that the feature axis an Asset Panel build produced holds no repeated name.

A repeated name is the one way the naming conventions can bite. `"<field>=<level>"` and `"<field>::observed"` are unique within one Panel Field by construction, so a collision is always between two Panel Fields — a field literally named `"sector=Tech"` beside a `"sector"` field with a `"Tech"` level, say. The index makes the collision harmless to a consumer, which reads columns as integers; it is refused anyway, because `ReturnsResult` requires a unique `nz` and would otherwise refuse it later with a message naming no Panel Field.

# Algorithm

 1. Find the first repeated name of `nz`.
 2. Throw when there is one, naming it and the Panel Fields that claim its two columns.

# Arguments

  - `nz`: The feature axis the build produced.
  - `pf`: The field index the build produced.

# Validation

  - `allunique(nz)`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`asset_panel`](@ref)
  - [`PanelField`](@ref)
  - [`ReturnsResult`](@ref)
"""
function assert_panel_feature_axis(nz::VecStr, pf::AbstractVector{<:PanelField})::Nothing
    j = findfirst(k -> nz[k] in view(nz, 1:(k - 1)), eachindex(nz))
    if !isnothing(j)
        # `j` is the *second* occurrence, so the first one exists. `something` says so to
        # the type system, which otherwise carries a `Nothing` into `panel_column_owner`.
        i = something(findfirst(==(nz[j]), nz), j)
        owners = [panel_column_owner(pf, c) for c in (i, j)]
        throw(ArgumentError("the Asset Panel build produced the feature name \"$(nz[j])\" twice, at columns $i and $j, claimed by the Panel Fields \"$(owners[1])\" and \"$(owners[2])\". A carrier needs a unique feature axis, so rename one of the two Panel Fields."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the name of the Panel Field that claims one column of the feature axis.

The inverse of the field index, and the only reader of it that walks: the index is keyed by Panel Field, and this answers the other question, which an error message asks once and nothing else asks at all.

# Algorithm

 1. Walk the field index, and return the first Panel Field whose value columns or observed-mask columns hold `c`.
 2. Return `"?"` when no Panel Field claims it, which the [`AssetPanel`](@ref) constructor makes unreachable through a built panel.

# Arguments

  - `pf`: The field index.
  - `c`: The column of the feature axis.

# Returns

  - `name::String`: The claiming Panel Field's name, or `"?"`.

# Related

  - [`assert_panel_feature_axis`](@ref)
  - [`PanelField`](@ref)
"""
function panel_column_owner(pf::AbstractVector{<:PanelField}, c::Integer)::String
    for f in pf
        if c in f.cols || (!isnothing(f.ocols) && c in f.ocols)
            return String(f.name)
        end
    end
    return "?"
end
export asset_panel, NumericPanelInput, CategoricalPanelInput, TensorPanelInput, NoPanelFill,
       ConstantPanelFill, ForwardPanelFill, BackwardPanelFill
