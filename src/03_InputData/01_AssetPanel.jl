"""
$(DocStringExtensions.TYPEDEF)

Supertype of the Panel Fields an [`AssetPanel`](@ref) holds.

All concrete types that carry one Panel Field's name, its values and its observed mask should subtype `AbstractPanelField`.

A Panel Field **owns its values**. Nothing else holds them, and nothing needs an index to find them: an [`AssetPanel`](@ref) is a vector of Panel Fields, and [`panel_field`](@ref) looks one up by name. A Panel Field is static or time-varying, and every Panel Field of one panel agrees.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractPanelField` and implement the following methods:

## `panel_field_axes`

  - `panel_field_axes(f::AbstractPanelField) -> Tuple{Vararg{Int}}`: `(N,)` when the Panel Field is static, and `(T, N)` when it is time-varying.

## `panel_field_labels`

  - `panel_field_labels(f::AbstractPanelField) -> Vector{String}`: The column names the Panel Field contributes to a derived Feature Matrix, in column order.

## `panel_field_stack!`

  - `panel_field_stack!(Z::AbstractArray, f::AbstractPanelField, cols::VecInt) -> nothing`: Write the Panel Field's value columns into `cols` of `Z`.

## `panel_field_view`

  - `panel_field_view(f::AbstractPanelField, i, j) -> AbstractPanelField`: The Panel Field over the observations `i` and the assets `j`.

# Related

  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`panel_field`](@ref)
"""
abstract type AbstractPanelField <: AbstractResult end
"""
    assert_panel_labels(labels::VecStr, sym::Sym_Str) -> nothing

Check that a label vector is non-empty, holds no empty entry, and holds no repeat.

Shared by every name vector a panel carries: the levels of a [`CategoricalPanelField`](@ref), the labels of a [`TensorPanelField`](@ref), and the Panel Field names of an [`AssetPanel`](@ref). Each names a column of a derived Feature Matrix, or a Panel Field a consumer looks up, so a repeat makes one name mean two things.

# Algorithm

 1. Check that `labels` is not empty.
 2. Check that no entry is the empty string, naming the first offending position.
 3. Check that no entry repeats, naming the first repeated position.

# Arguments

  - `labels`: The label vector to check.
  - `sym`: Symbolic name of the vector, displayed in the error messages.

# Validation

  - `!isempty(labels)`. Raises an [`IsEmptyError`](@ref).
  - No entry is empty. Raises an `ArgumentError`.
  - `allunique(labels)`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`VecStr`](@ref)
"""
function assert_panel_labels(labels::VecStr, sym::Sym_Str)::Nothing
    @argcheck(!isempty(labels),
              IsEmptyError("$sym cannot be empty: a Panel Field with no labels claims no column of a derived Feature Matrix"))
    i = findfirst(isempty, labels)
    @argcheck(isnothing(i),
              ArgumentError("$sym names columns of a derived Feature Matrix, so no entry may be the empty string; the first empty entry is at position $i"))
    j = findfirst(k -> labels[k] in view(labels, 1:(k - 1)), eachindex(labels))
    @argcheck(isnothing(j),
              ArgumentError("$sym must be unique, because each entry names one column of a derived Feature Matrix; the first repeated entry is at position $j"))
    return nothing
end
"""
    assert_panel_field_name(name::AbstractString) -> nothing

Check that a Panel Field's name is not empty.

The name is the key [`panel_field`](@ref) looks a Panel Field up by, so an empty one is unreachable.

# Arguments

  - `name`: The Panel Field's name.

# Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_field`](@ref)
"""
function assert_panel_field_name(name::AbstractString)::Nothing
    @argcheck(!isempty(name),
              IsEmptyError("the name of a Panel Field cannot be empty: it is the key the panel is looked up by"))
    return nothing
end
"""
    assert_panel_field_shape(vals::AbstractArray, name::AbstractString, s::Integer, t::Integer) -> nothing

Check that a Panel Field's values are non-empty and carry the static or the time-varying rank.

A Panel Field takes one of two ranks, and which pair of ranks it takes depends on its kind: `s` is the static rank and `t` the time-varying one. The time-varying rank is the static one with the observation axis prepended, so the two always differ by one.

# Algorithm

 1. Check that `vals` is not empty.
 2. Check that `ndims(vals)` is `s` or `t`.

# Arguments

  - `vals`: The Panel Field's values.
  - `name`: The Panel Field's name, displayed in the error messages.
  - `s`: The static rank.
  - `t`: The time-varying rank.

# Validation

  - `!isempty(vals)`. Raises an [`IsEmptyError`](@ref).
  - `ndims(vals) in (s, t)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_field_axes`](@ref)
"""
function assert_panel_field_shape(vals::AbstractArray, name::AbstractString, s::Integer,
                                  t::Integer)::Nothing
    @argcheck(!isempty(vals),
              IsEmptyError("the values of the Panel Field \"$name\" cannot be empty"))
    @argcheck(ndims(vals) == s || ndims(vals) == t,
              DimensionMismatch("the Panel Field \"$name\" is $s-dimensional when static and $t-dimensional when time-varying, got a $(ndims(vals))-dimensional array of size $(size(vals))"))
    return nothing
end
"""
    assert_panel_field_mask(vals::AbstractArray, omsk::Nothing, name::AbstractString) -> nothing
    assert_panel_field_mask(vals::AbstractArray, omsk::AbstractArray{Bool}, name::AbstractString) -> nothing

Check that a Panel Field's observed mask covers its values entry for entry.

The mask says which cells the raw source observed, and which a fill policy wrote. It is therefore the same shape as the values, down to a tensor Panel Field's label axis, which keeps the per-entry resolution the raw input carried.

# Algorithm

The method that Julia selects is the algorithm.

 1. `omsk` is `nothing`: the Panel Field cannot blank, so there is nothing to check.
 2. `omsk` is an array: check that its size matches the values.

# Arguments

  - `vals`: The Panel Field's values.
  - `omsk`: The observed mask, or `nothing`.
  - `name`: The Panel Field's name, displayed in the error message.

# Validation

  - `size(omsk) == size(vals)`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
"""
function assert_panel_field_mask(::AbstractArray, ::Nothing, ::AbstractString)::Nothing
    return nothing
end
function assert_panel_field_mask(vals::AbstractArray, omsk::AbstractArray{Bool},
                                 name::AbstractString)::Nothing
    @argcheck(size(omsk) == size(vals),
              DimensionMismatch("the observed mask (omsk) of the Panel Field \"$name\" marks one cell of its values as observed or filled, so the two are the same size, got size(omsk) = $(size(omsk)) and size(vals) = $(size(vals))"))
    return nothing
end
"""
    assert_panel_finite(vals::AbstractArray{<:Real}, name::AbstractString) -> nothing

Check that a resolved Panel Field carries no non-finite value.

The fill policies each write a finite value by construction, so this catches a non-finite cell that the *raw input* carried and that no policy touched: an infinity is not a blank, so [`is_panel_blank`](@ref) leaves it where it stands.

# Algorithm

 1. Find the first non-finite cell.
 2. Throw when there is one, naming the Panel Field and the cell.

# Arguments

  - `vals`: The resolved values.
  - `name`: The Panel Field's name, displayed in the error message.

# Validation

  - Every cell is finite. Raises an [`IsNonFiniteError`](@ref).

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
              IsNonFiniteError("the Panel Field \"$name\" carries a non-finite value at $(isnothing(i) ? "" : string(Tuple(i))). An infinity is not a blank, so no fill policy resolves it; correct the raw input."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

A Panel Field holding one number per asset, and per observation when it is time-varying.

A market capitalisation, a book-to-price ratio or a trailing volume is this kind. It contributes one column to a derived Feature Matrix, named after the Panel Field itself.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    NumericPanelField(name::AbstractString, vals::AbstractArray{<:Real},
                      omsk::Option{<:AbstractArray{Bool}} = nothing)

# Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).
  - `!isempty(vals)`. Raises an [`IsEmptyError`](@ref).
  - `ndims(vals) in (1, 2)`. Raises a `DimensionMismatch`.
  - `size(omsk) == size(vals)` when `omsk` is given. Raises a `DimensionMismatch`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`Option`](@ref)
"""
@concrete struct NumericPanelField <: AbstractPanelField
    """
    The Panel Field's name, which names its column of a derived Feature Matrix.
    """
    name
    """
    Values: `assets` when static, `observations × assets` when time-varying.
    """
    vals
    """
    Observed mask, the same size as the values, or `nothing` when the Panel Field cannot blank.
    """
    omsk
    function NumericPanelField(name::AbstractString, vals::AbstractArray{<:Real},
                               omsk::Option{<:AbstractArray{Bool}})
        assert_panel_field_name(name)
        assert_panel_field_shape(vals, name, 1, 2)
        assert_panel_finite(vals, name)
        assert_panel_field_mask(vals, omsk, name)
        return new{typeof(name), typeof(vals), typeof(omsk)}(name, vals, omsk)
    end
end
function NumericPanelField(; name::AbstractString, vals::AbstractArray{<:Real},
                           omsk::Option{<:AbstractArray{Bool}} = nothing)::NumericPanelField
    return NumericPanelField(name, vals, omsk)
end
"""
$(DocStringExtensions.TYPEDEF)

A Panel Field holding one category label per asset, and per observation when it is time-varying.

A sector, an industry or a country classification is this kind. It stores **integer codes over its levels** rather than the labels themselves, and rather than a one-hot block: a code is what a cross-sectional group label is, and the one-hot form is built where a matrix is needed. It contributes one column per level to a derived Feature Matrix, named `"<field>=<level>"`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    CategoricalPanelField(name::AbstractString, levels::VecStr,
                          codes::AbstractArray{<:Integer},
                          omsk::Option{<:AbstractArray{Bool}} = nothing)

# Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).
  - `levels` is non-empty, holds no empty entry and holds no repeat. See [`assert_panel_labels`](@ref).
  - `!isempty(codes)`. Raises an [`IsEmptyError`](@ref).
  - `ndims(codes) in (1, 2)`. Raises a `DimensionMismatch`.
  - Every code lies in `1:length(levels)`. Raises a `DomainError`.
  - `size(omsk) == size(codes)` when `omsk` is given. Raises a `DimensionMismatch`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`NumericPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`assert_panel_labels`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct CategoricalPanelField <: AbstractPanelField
    """
    The Panel Field's name, which prefixes each of its columns of a derived Feature Matrix.
    """
    name
    """
    The category levels, one per derived column, in column order.
    """
    levels
    """
    Integer codes over `levels`: `assets` when static, `observations × assets` when time-varying.
    """
    codes
    """
    Observed mask, the same size as the codes, or `nothing` when the Panel Field cannot blank.
    """
    omsk
    function CategoricalPanelField(name::AbstractString, levels::VecStr,
                                   codes::AbstractArray{<:Integer},
                                   omsk::Option{<:AbstractArray{Bool}})
        assert_panel_field_name(name)
        assert_panel_labels(levels, :levels)
        assert_panel_field_shape(codes, name, 1, 2)
        nl = length(levels)
        i = findfirst(c -> c < 1 || c > nl, codes)
        @argcheck(isnothing(i),
                  DomainError(nl,
                              "the codes of the categorical Panel Field \"$name\" index its $nl level(s), so every code lies in 1:$nl; the first offending code is at $(isnothing(i) ? "" : string(Tuple(i)))"))
        assert_panel_field_mask(codes, omsk, name)
        return new{typeof(name), typeof(levels), typeof(codes), typeof(omsk)}(name, levels,
                                                                              codes, omsk)
    end
end
function CategoricalPanelField(; name::AbstractString, levels::VecStr,
                               codes::AbstractArray{<:Integer},
                               omsk::Option{<:AbstractArray{Bool}} = nothing)::CategoricalPanelField
    return CategoricalPanelField(name, levels, codes, omsk)
end
"""
$(DocStringExtensions.TYPEDEF)

A Panel Field whose trailing axis carries its own labels, and optionally its own groups.

A factor exposure tensor is this kind: its trailing axis is the factors, and its groups are the Factor Families. It contributes one column per label to a derived Feature Matrix, named `"<field>=<label>"`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    TensorPanelField(name::AbstractString, axis::AbstractString, labels::VecStr,
                     groups::Option{<:VecStr}, vals::AbstractArray{<:Real},
                     omsk::Option{<:AbstractArray{Bool}} = nothing)

# Validation

  - `!isempty(name)`. Raises an [`IsEmptyError`](@ref).
  - `!isempty(axis)`. Raises an [`IsEmptyError`](@ref).
  - `labels` is non-empty, holds no empty entry and holds no repeat. See [`assert_panel_labels`](@ref).
  - `length(groups) == length(labels)` when `groups` is given. Raises a `DimensionMismatch`.
  - `!isempty(vals)`. Raises an [`IsEmptyError`](@ref).
  - `ndims(vals) in (2, 3)`. Raises a `DimensionMismatch`.
  - `size(vals, ndims(vals)) == length(labels)`. Raises a `DimensionMismatch`.
  - `size(omsk) == size(vals)` when `omsk` is given. Raises a `DimensionMismatch`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`assert_panel_labels`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct TensorPanelField <: AbstractPanelField
    """
    The Panel Field's name, which prefixes each of its columns of a derived Feature Matrix.
    """
    name
    """
    Name of what the trailing axis represents, such as `"factor"`.
    """
    axis
    """
    Labels of the trailing-axis entries, one per trailing-axis entry of the values.
    """
    labels
    """
    Optional group of each trailing-axis entry, such as a Factor Family, one per label.
    """
    groups
    """
    Values: `assets × labels` when static, `observations × assets × labels` when time-varying.
    """
    vals
    """
    Observed mask, the same size as the values, or `nothing` when the Panel Field cannot blank.
    """
    omsk
    function TensorPanelField(name::AbstractString, axis::AbstractString, labels::VecStr,
                              groups::Option{<:VecStr}, vals::AbstractArray{<:Real},
                              omsk::Option{<:AbstractArray{Bool}})
        assert_panel_field_name(name)
        @argcheck(!isempty(axis),
                  IsEmptyError("the trailing-axis name (axis) of the tensor Panel Field \"$name\" cannot be empty: it names what the axis represents, such as \"factor\""))
        assert_panel_labels(labels, :labels)
        if !isnothing(groups)
            @argcheck(length(groups) == length(labels),
                      DimensionMismatch("the tensor Panel Field \"$name\" needs one group per label, got length(groups) = $(length(groups)) and length(labels) = $(length(labels))"))
        end
        assert_panel_field_shape(vals, name, 2, 3)
        assert_panel_finite(vals, name)
        @argcheck(size(vals, ndims(vals)) == length(labels),
                  DimensionMismatch("the tensor Panel Field \"$name\" needs one label per trailing-axis entry of vals, got $(size(vals, ndims(vals))) trailing entries and length(labels) = $(length(labels))"))
        assert_panel_field_mask(vals, omsk, name)
        return new{typeof(name), typeof(axis), typeof(labels), typeof(groups), typeof(vals),
                   typeof(omsk)}(name, axis, labels, groups, vals, omsk)
    end
end
function TensorPanelField(; name::AbstractString, axis::AbstractString, labels::VecStr,
                          groups::Option{<:VecStr} = nothing, vals::AbstractArray{<:Real},
                          omsk::Option{<:AbstractArray{Bool}} = nothing)::TensorPanelField
    return TensorPanelField(name, axis, labels, groups, vals, omsk)
end
"""
    panel_field_axes(f::NumericPanelField) -> Tuple{Vararg{Int}}
    panel_field_axes(f::CategoricalPanelField) -> Tuple{Vararg{Int}}
    panel_field_axes(f::TensorPanelField) -> Tuple{Vararg{Int}}

Return a Panel Field's observation and asset axes, without its trailing label axis.

This is the shape every Panel Field of one [`AssetPanel`](@ref) shares, and it is what says whether the panel is static. A static Panel Field returns `(N,)`; a time-varying one returns `(T, N)`.

# Algorithm

The method that Julia selects is the algorithm. A numeric and a categorical Panel Field carry no label axis, so their whole size is returned. A tensor Panel Field drops its trailing label axis.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `ax::Tuple{Vararg{Int}}`: `(N,)` when static, `(T, N)` when time-varying.

# Related

  - [`AbstractPanelField`](@ref)
  - [`AssetPanel`](@ref)
  - [`panel_is_static`](@ref)
"""
function panel_field_axes(f::NumericPanelField)
    return size(f.vals)
end
function panel_field_axes(f::CategoricalPanelField)
    return size(f.codes)
end
function panel_field_axes(f::TensorPanelField)
    return size(f.vals)[1:(ndims(f.vals) - 1)]
end
"""
    panel_field_labels(f::NumericPanelField) -> Vector{String}
    panel_field_labels(f::CategoricalPanelField) -> Vector{String}
    panel_field_labels(f::TensorPanelField) -> Vector{String}

Return the column names one Panel Field contributes to a derived Feature Matrix, in column order.

# Algorithm

The method that Julia selects is the algorithm. Each kind names its columns differently.

 1. [`NumericPanelField`](@ref): one column, named after the Panel Field itself.
 2. [`CategoricalPanelField`](@ref): one column per level, named `"<name>=<level>"`.
 3. [`TensorPanelField`](@ref): one column per label, named `"<name>=<label>"`.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `labels::Vector{String}`: One name per derived column.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`panel_field_observed_labels`](@ref)
"""
function panel_field_labels(f::NumericPanelField)::Vector{String}
    return [String(f.name)]
end
function panel_field_labels(f::CategoricalPanelField)::Vector{String}
    return ["$(f.name)=$level" for level in f.levels]
end
function panel_field_labels(f::TensorPanelField)::Vector{String}
    return ["$(f.name)=$label" for label in f.labels]
end
"""
    panel_field_observed_labels(f::AbstractPanelField) -> Vector{String}

Return the names of the observed-mask columns one Panel Field contributes to a derived Feature Matrix.

A Panel Field with a single observable takes `"<name>::observed"`. One with several takes each value column's own name with `"::observed"` appended, so a tensor Panel Field keeps one mask column per label.

The separator is `"::"` rather than the `"="` the value columns use, so a mask column cannot be mistaken for a level of the same Panel Field.

# Algorithm

 1. Read the Panel Field's value column names from [`panel_field_labels`](@ref).
 2. When there is one, return the single name `"<name>::observed"`.
 3. Otherwise append `"::observed"` to each of them.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `labels::Vector{String}`: One name per observed-mask column.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_field_labels`](@ref)
  - [`panel_feature_matrix`](@ref)
"""
function panel_field_observed_labels(f::AbstractPanelField)::Vector{String}
    labels = panel_field_labels(f)
    return if isone(length(labels))
        ["$(f.name)::observed"]
    else
        ["$l::observed" for l in labels]
    end
end
"""
    panel_field_stack!(Z::AbstractArray, f::NumericPanelField, cols::VecInt) -> nothing
    panel_field_stack!(Z::AbstractArray, f::CategoricalPanelField, cols::VecInt) -> nothing
    panel_field_stack!(Z::AbstractArray, f::TensorPanelField, cols::VecInt) -> nothing

Write one Panel Field's value columns into a derived Feature Matrix.

`Z` is allocated as zeros, which is what makes a one-hot column correct: the categorical method writes only the `1`s.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref): write the values into the single column.
 2. [`CategoricalPanelField`](@ref): write a `1` into the column its code names, for each cell.
 3. [`TensorPanelField`](@ref): write one label slice per column.

# Arguments

  - `Z`: The derived Feature Matrix under construction.
  - `f`: The Panel Field.
  - `cols`: The columns the Panel Field claims, in order.

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`panel_field_labels`](@ref)
  - [`VecInt`](@ref)
"""
function panel_field_stack!(Z::AbstractArray, f::NumericPanelField, cols::VecInt)::Nothing
    selectdim(Z, ndims(Z), cols[1]) .= f.vals
    return nothing
end
function panel_field_stack!(Z::AbstractArray, f::CategoricalPanelField,
                            cols::VecInt)::Nothing
    for i in CartesianIndices(f.codes)
        Z[i, cols[f.codes[i]]] = one(eltype(Z))
    end
    return nothing
end
function panel_field_stack!(Z::AbstractArray, f::TensorPanelField, cols::VecInt)::Nothing
    d = ndims(f.vals)
    for (l, c) in pairs(cols)
        selectdim(Z, ndims(Z), c) .= selectdim(f.vals, d, l)
    end
    return nothing
end
"""
    panel_field_stack_observed!(Z::AbstractArray, f::AbstractPanelField, cols::VecInt) -> nothing

Write one Panel Field's observed mask into a derived Feature Matrix, as `0`/`1` columns.

# Algorithm

 1. Return when the Panel Field carries no mask.
 2. Write the whole mask into the single column when the Panel Field claims one.
 3. Otherwise write one label slice of the mask per column.

# Arguments

  - `Z`: The derived Feature Matrix under construction.
  - `f`: The Panel Field.
  - `cols`: The observed-mask columns the Panel Field claims, in order.

# Returns

  - `nothing`.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_field_observed_labels`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`VecInt`](@ref)
"""
function panel_field_stack_observed!(Z::AbstractArray, f::AbstractPanelField,
                                     cols::VecInt)::Nothing
    omsk = f.omsk
    if isnothing(omsk)
        return nothing
    end
    if isone(length(cols))
        selectdim(Z, ndims(Z), cols[1]) .= omsk
    else
        d = ndims(omsk)
        for (l, c) in pairs(cols)
            selectdim(Z, ndims(Z), c) .= selectdim(omsk, d, l)
        end
    end
    return nothing
end
"""
    panel_array_view(A::Nothing, i, j) -> nothing
    panel_array_view(A::AbstractVector, i, j) -> SubArray
    panel_array_view(A::AbstractMatrix, i, j) -> SubArray

View one label-free Panel Field array over the observations `i` and the assets `j`.

A [`NumericPanelField`](@ref) and a [`CategoricalPanelField`](@ref) carry no label axis, so the rank alone says which axes they have: a vector is static and its one axis is the assets, and a matrix is time-varying and its axes are the observations and the assets. A tensor Panel Field is viewed by [`panel_tensor_view`](@ref) instead, because its asset axis is not the last one.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return `nothing`.
 2. `A` is a vector: return `view(A, j)`.
 3. `A` is a matrix: return `view(A, i, j)`.

# Arguments

  - `A`: The array to view, or `nothing`.
  - `i`: Observation index.
  - `j`: Asset index.

# Returns

  - A view of `A`, or `nothing`.

# Related

  - [`panel_field_view`](@ref)
  - [`panel_tensor_view`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_array_view(::Nothing, ::Any, ::Any)
    return nothing
end
function panel_array_view(A::AbstractVector, ::Any, j)
    return view(A, j)
end
function panel_array_view(A::AbstractMatrix, i, j)
    return view(A, i, j)
end
"""
    panel_tensor_view(A::Nothing, i, j) -> nothing
    panel_tensor_view(A::AbstractMatrix, i, j) -> SubArray
    panel_tensor_view(A::AbstractArray{<:Any, 3}, i, j) -> SubArray

View one [`TensorPanelField`](@ref) array over the observations `i` and the assets `j`.

A tensor Panel Field keeps its labels on its **trailing** axis, so its asset axis is the first one when it is static and the second when it is time-varying. The label axis is never touched: it addresses the features, and an asset view does not reach it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `A` is `nothing`: return `nothing`.
 2. `A` is a matrix, which is `assets × labels`: return `view(A, j, :)`.
 3. `A` is a 3-dimensional array, which is `observations × assets × labels`: return `view(A, i, j, :)`.

# Arguments

  - `A`: The array to view, or `nothing`.
  - `i`: Observation index.
  - `j`: Asset index.

# Returns

  - A view of `A`, or `nothing`.

# Related

  - [`panel_field_view`](@ref)
  - [`panel_array_view`](@ref)
  - [`TensorPanelField`](@ref)
"""
function panel_tensor_view(::Nothing, ::Any, ::Any)
    return nothing
end
function panel_tensor_view(A::AbstractMatrix, ::Any, j)
    return view(A, j, :)
end
function panel_tensor_view(A::AbstractArray{<:Any, 3}, i, j)
    return view(A, i, j, :)
end
"""
    panel_field_view(f::NumericPanelField, i, j) -> NumericPanelField
    panel_field_view(f::CategoricalPanelField, i, j) -> CategoricalPanelField
    panel_field_view(f::TensorPanelField, i, j) -> TensorPanelField

Return a view of one Panel Field over the observations `i` and the assets `j`.

A static Panel Field has no observation axis, so its caller passes a `Colon` for `i`. The trailing label axis of a tensor Panel Field is not touched: it addresses the features, and an asset view does not reach it.

# Algorithm

The method that Julia selects is the algorithm, and each kind views its own value array and its own mask: a numeric and a categorical Panel Field through [`panel_array_view`](@ref), and a tensor Panel Field through [`panel_tensor_view`](@ref), whose asset axis is not the last one.

# Arguments

  - `f`: The Panel Field.
  - `i`: Observation index.
  - `j`: Asset index.

# Returns

  - A Panel Field of the same kind over the selected observations and assets.

# Related

  - [`AbstractPanelField`](@ref)
  - [`panel_array_view`](@ref)
  - [`port_opt_view`](@ref)
  - [`AssetPanel`](@ref)
"""
function panel_field_view(f::NumericPanelField, i, j)
    return NumericPanelField(; name = f.name, vals = panel_array_view(f.vals, i, j),
                             omsk = panel_array_view(f.omsk, i, j))
end
function panel_field_view(f::CategoricalPanelField, i, j)
    return CategoricalPanelField(; name = f.name, levels = f.levels,
                                 codes = panel_array_view(f.codes, i, j),
                                 omsk = panel_array_view(f.omsk, i, j))
end
function panel_field_view(f::TensorPanelField, i, j)
    return TensorPanelField(; name = f.name, axis = f.axis, labels = f.labels,
                            groups = f.groups, vals = panel_tensor_view(f.vals, i, j),
                            omsk = panel_tensor_view(f.omsk, i, j))
end
"""
$(DocStringExtensions.TYPEDEF)

The Asset Panel: the Panel Fields of one universe, and the two point-in-time universe masks.

The panel **is** the feature data. Its Panel Fields own their values, so nothing else on a carrier holds a feature matrix, and the Feature Matrix a distance measures is derived from the panel by [`panel_feature_matrix`](@ref) and stored nowhere.

One panel takes one of two shapes, and its type parameters say which.

  - **Static**: every Panel Field is `assets` or `assets × labels`, and both masks are `nothing`. A fundamentals table or a sector classification with no history is this shape.
  - **Time-varying**: every Panel Field prepends an observation axis, and both masks are `observations × assets`. A point-in-time panel is this shape.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    AssetPanel(pf::AbstractVector{<:AbstractPanelField},
               amsk::Option{<:AbstractMatrix{Bool}} = nothing,
               emsk::Option{<:AbstractMatrix{Bool}} = nothing)

# Validation

  - `!isempty(pf)`. Raises an [`IsEmptyError`](@ref).
  - The Panel Field names are non-empty and unique. See [`assert_panel_labels`](@ref).
  - Every Panel Field shares one [`panel_field_axes`](@ref). Raises a `DimensionMismatch`.
  - The masks are both `nothing` when the Panel Fields are static, and both given when they are time-varying. See [`assert_panel_masks`](@ref).

# Related

  - [`AbstractPanelField`](@ref)
  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`asset_panel`](@ref)
  - [`panel_field`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`assert_panel_masks`](@ref)
  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`Option`](@ref)
"""
@concrete struct AssetPanel <: AbstractResult
    """
    The Panel Fields, each owning its own values and its own observed mask.
    """
    pf
    """
    The active mask (observations × assets): whether the asset is in the universe at that observation. `nothing` when the panel is static.
    """
    amsk
    """
    The estimation mask (observations × assets): whether the asset enters the cross-sectional estimate at that observation. Always a subset of the active mask. `nothing` when the panel is static.
    """
    emsk
    function AssetPanel(pf::AbstractVector{<:AbstractPanelField},
                        amsk::Option{<:AbstractMatrix{Bool}},
                        emsk::Option{<:AbstractMatrix{Bool}})
        @argcheck(!isempty(pf),
                  IsEmptyError("an Asset Panel needs at least one Panel Field: an empty panel carries no feature data"))
        assert_panel_labels([f.name for f in pf], "the Panel Field names")
        ax = panel_field_axes(pf[1])
        k = findfirst(f -> panel_field_axes(f) != ax, pf)
        @argcheck(isnothing(k),
                  DimensionMismatch("every Panel Field of one Asset Panel shares its observation axis and its asset axis, and \"$(isnothing(k) ? "" : pf[k].name)\" does not: got $(isnothing(k) ? "" : string(panel_field_axes(pf[k]))) against the $ax of \"$(pf[1].name)\""))
        assert_panel_masks(ax, amsk, emsk)
        return new{typeof(pf), typeof(amsk), typeof(emsk)}(pf, amsk, emsk)
    end
end
function AssetPanel(; pf::AbstractVector{<:AbstractPanelField},
                    amsk::Option{<:AbstractMatrix{Bool}} = nothing,
                    emsk::Option{<:AbstractMatrix{Bool}} = nothing)::AssetPanel
    return AssetPanel(pf, amsk, emsk)
end
"""
    assert_panel_masks(ax::Tuple, amsk::Nothing, emsk::Nothing) -> nothing
    assert_panel_masks(ax::Tuple, amsk, emsk) -> nothing

Check an Asset Panel's two universe masks against the shape its Panel Fields agreed on.

The masks are the one thing that says whether a panel is static: they are `nothing` if and only if its Panel Fields carry no observation axis. That rule is what makes the static shape a type parameter rather than a runtime branch, so a mask consumer dispatches on `AssetPanel{PF, Nothing, Nothing}` and never tests.

# Algorithm

The method that Julia selects is the algorithm.

 1. Both masks are `nothing`: check that the Panel Fields are static, that is, that `ax` names one axis.
 2. Otherwise: check that both masks are given, that the Panel Fields are time-varying, that both masks are `ax`, and that `emsk` is a subset of `amsk`.

# Arguments

  - `ax`: The observation and asset axes the Panel Fields agreed on.
  - `amsk`: The active mask, or `nothing`.
  - `emsk`: The estimation mask, or `nothing`.

# Validation

  - `length(ax) == 1` when the masks are `nothing`, and `length(ax) == 2` otherwise. Raises a `DimensionMismatch`.
  - The masks are both `nothing` or both given. Raises a `DimensionMismatch`.
  - `size(amsk) == size(emsk) == ax`. Raises a `DimensionMismatch`.
  - `emsk` is a subset of `amsk`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_field_axes`](@ref)
  - [`panel_is_static`](@ref)
  - [`Option`](@ref)
"""
function assert_panel_masks(ax::Tuple, ::Nothing, ::Nothing)::Nothing
    @argcheck(isone(length(ax)),
              DimensionMismatch("an Asset Panel whose Panel Fields carry an observation axis is time-varying, so it needs both universe masks; got Panel Fields of shape $ax and no mask. Pass amsk and emsk, or drop the observation axis from the Panel Fields."))
    return nothing
end
function assert_panel_masks(ax::Tuple, amsk::Option{<:AbstractMatrix{Bool}},
                            emsk::Option{<:AbstractMatrix{Bool}})::Nothing
    @argcheck(!isnothing(amsk) && !isnothing(emsk),
              DimensionMismatch("the two universe masks of an Asset Panel are given together or not at all, because they are what says the panel is time-varying; got amsk = $(isnothing(amsk) ? "nothing" : "a matrix") and emsk = $(isnothing(emsk) ? "nothing" : "a matrix")"))
    @argcheck(length(ax) == 2,
              DimensionMismatch("an Asset Panel with universe masks is time-varying, so its Panel Fields carry an observation axis; got Panel Fields of shape $ax. Drop the masks, or prepend the observation axis."))
    @argcheck(size(amsk) == ax && size(emsk) == ax,
              DimensionMismatch("the universe masks of an Asset Panel are observations × assets, so they match its Panel Fields, got size(amsk) = $(size(amsk)), size(emsk) = $(size(emsk)) and Panel Fields of shape $ax"))
    idx = findfirst(k -> emsk[k] && !amsk[k], eachindex(emsk))
    @argcheck(isnothing(idx),
              ArgumentError("the estimation mask (emsk) must be a subset of the active mask (amsk): an asset that is not in the universe at an observation cannot enter that observation's estimate. Intersect them yourself with `emsk .& amsk` — the rule is checked rather than coerced, because a coercion allocates and port_opt_view returns views. The first offending entry is at $(isnothing(idx) ? "" : string(Tuple(CartesianIndices(emsk)[idx])))"))
    return nothing
end
"""
    panel_is_static(pnl::AssetPanel) -> Bool

Return whether an [`AssetPanel`](@ref) is the static shape.

A static panel's Panel Fields carry no observation axis, and its masks are `nothing`. The two go together by construction, so either one answers.

# Arguments

  - `pnl`: The Asset Panel.

# Returns

  - `static::Bool`: `true` when the panel is static.

# Related

  - [`AssetPanel`](@ref)
  - [`assert_panel_masks`](@ref)
"""
function panel_is_static(pnl::AssetPanel)::Bool
    return isnothing(pnl.amsk)
end
"""
    panel_field(pnl::AssetPanel, name::AbstractString) -> AbstractPanelField

Look one Panel Field up in an [`AssetPanel`](@ref) by name.

This is the only supported route from a Panel Field's name to its values. A consumer that parses a derived column name instead is reading a convention rather than the panel, and the two part company as soon as a Panel Field's own name carries the convention's punctuation.

# Algorithm

 1. Find the first Panel Field whose name matches.
 2. Throw a `KeyError` naming the nearest match and the whole panel when none does.

# Arguments

  - `pnl`: The Asset Panel.
  - `name`: The Panel Field's name.

# Validation

  - The panel holds a Panel Field named `name`. Raises a `KeyError`.

# Returns

  - `f::AbstractPanelField`: The Panel Field.

# Related

  - [`AssetPanel`](@ref)
  - [`AbstractPanelField`](@ref)
  - [`did_you_mean`](@ref)
"""
function panel_field(pnl::AssetPanel, name::AbstractString)
    i = findfirst(f -> f.name == name, pnl.pf)
    @argcheck(!isnothing(i),
              KeyError("the Asset Panel holds no Panel Field named `$name`$(did_you_mean(name, [f.name for f in pnl.pf])). It holds $(length(pnl.pf)): $(join([f.name for f in pnl.pf], ", "))"))
    return pnl.pf[i]
end
"""
    panel_claim!(nz::AbstractVector{String}, labels::AbstractVector{String}) -> Vector{Int}

Append a Panel Field's column names to a derived Feature Matrix's names, and return the columns they took.

The one place a derived column index is minted, so the names and the write cannot disagree about where a Panel Field's columns are.

# Algorithm

 1. Read the current length of `nz`, which is the last column already claimed.
 2. Append `labels` to it.
 3. Return the range of columns the append occupied, as a vector.

# Arguments

  - `nz`: The derived column names under construction. It is appended to.
  - `labels`: The column names to claim.

# Returns

  - `cols::Vector{Int}`: The columns `labels` took, in order.

# Related

  - [`panel_feature_matrix`](@ref)
  - [`panel_field_labels`](@ref)
"""
function panel_claim!(nz::AbstractVector{String}, labels::AbstractVector{String})
    cols = collect((length(nz) + 1):(length(nz) + length(labels)))
    append!(nz, labels)
    return cols
end
"""
    panel_feature_matrix(pnl::Nothing) -> Tuple{Nothing, Nothing}
    panel_feature_matrix(pnl::AssetPanel) -> Tuple{Vector{String}, Array{Float64}}

Derive the Feature Matrix an [`AssetPanel`](@ref)'s Panel Fields stack into, and name its columns.

A carrier that holds no panel derives nothing, so the `nothing` method answers with two of them and no consumer needs a branch of its own.

Nothing stores the result. A Feature Matrix is what a distance measures, so it is built where it is measured and thrown away after: the panel is the data, and the matrix is one view of it.

The column order is the Panel Field order, and within one Panel Field its value columns come first and its observed-mask columns after. A static panel gives an `assets × features` matrix, and a time-varying one an `observations × assets × features` matrix.

# Algorithm

The method that Julia selects decides whether there is anything to derive.

 1. Walk the Panel Fields in order. Claim each one's value columns from [`panel_field_labels`](@ref), then its observed-mask columns from [`panel_field_observed_labels`](@ref) when it carries a mask.
 2. Allocate the matrix as zeros, over the panel's own observation and asset axes and the claimed column count.
 3. Write each Panel Field's values with [`panel_field_stack!`](@ref) and its mask with [`panel_field_stack_observed!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.

# Returns

  - `nz::Vector{String}`: One name per column of the derived Feature Matrix.
  - `Z::Array{Float64}`: The derived Feature Matrix.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_field_labels`](@ref)
  - [`panel_field_observed_labels`](@ref)
  - [`panel_field_stack!`](@ref)
  - [`panel_claim!`](@ref)
  - [`feature_matrix_panel`](@ref)
"""
function panel_feature_matrix(::Nothing)
    return nothing, nothing
end
function panel_feature_matrix(pnl::AssetPanel)
    nz = String[]
    cols = Vector{Int}[]
    ocols = Vector{Int}[]
    for f in pnl.pf
        push!(cols, panel_claim!(nz, panel_field_labels(f)))
        push!(ocols,
              isnothing(f.omsk) ? Int[] : panel_claim!(nz, panel_field_observed_labels(f)))
    end
    Z = zeros(Float64, panel_field_axes(pnl.pf[1])..., length(nz))
    for (k, f) in pairs(pnl.pf)
        panel_field_stack!(Z, f, cols[k])
        if !isempty(ocols[k])
            panel_field_stack_observed!(Z, f, ocols[k])
        end
    end
    return nz, Z
end
"""
    assert_feature_matrix_columns(nz::VecStr, Z::MatNum_Arr3Num) -> nothing

Check that a Feature Matrix has one usable name per column.

# Algorithm

 1. Check the names with [`assert_panel_labels`](@ref).
 2. Check that their count matches the trailing axis of `Z`.

# Arguments

  - `nz`: One name per column of `Z`.
  - `Z`: The Feature Matrix.

# Validation

  - `nz` is non-empty, holds no empty entry and holds no repeat. See [`assert_panel_labels`](@ref).
  - `length(nz) == size(Z, ndims(Z))`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`feature_matrix_panel`](@ref)
  - [`assert_panel_labels`](@ref)
  - [`MatNum_Arr3Num`](@ref)
  - [`VecStr`](@ref)
"""
function assert_feature_matrix_columns(nz::VecStr, Z::MatNum_Arr3Num)::Nothing
    assert_panel_labels(nz, :nz)
    @argcheck(length(nz) == size(Z, ndims(Z)),
              DimensionMismatch("a Feature Matrix needs one name per column, got length(nz) = $(length(nz)) and $(size(Z, ndims(Z))) column(s)"))
    return nothing
end
"""
    feature_matrix_panel(nz::VecStr, Z::MatNum; amsk = nothing, emsk = nothing) -> AssetPanel
    feature_matrix_panel(nz::VecStr, Z::Arr3Num; amsk = nothing, emsk = nothing) -> AssetPanel

Build an [`AssetPanel`](@ref) of one [`NumericPanelField`](@ref) per column of a Feature Matrix.

The inverse of [`panel_feature_matrix`](@ref), and exact: a numeric Panel Field contributes one column under its own name, so the round trip returns the names and the values it was given. It is what a routine that produces a bare matrix — a producer, or a meta-optimiser collapse onto a synthetic universe — puts that matrix on a carrier through.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Z` is a `MatNum`, which is `assets × features`: build a static panel of one numeric Panel Field per column, and check that no mask was given.
 2. `Z` is an `Arr3Num`, which is `observations × assets × features`: build a time-varying panel of one numeric Panel Field per column, and fill in all-`true` masks when none are given.

# Arguments

  - `nz`: One name per column of `Z`.
  - `Z`: The Feature Matrix.
  - `amsk`: The active mask, or `nothing`.
  - `emsk`: The estimation mask, or `nothing`.

# Validation

  - `nz` names the columns of `Z`. See [`assert_feature_matrix_columns`](@ref).
  - `amsk` and `emsk` are `nothing` when `Z` is static. Raises a `DimensionMismatch`.

# Returns

  - `pnl::AssetPanel`: The Asset Panel.

# Related

  - [`AssetPanel`](@ref)
  - [`NumericPanelField`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`assert_feature_matrix_columns`](@ref)
  - [`MatNum`](@ref)
  - [`Arr3Num`](@ref)
  - [`Option`](@ref)
"""
function feature_matrix_panel(nz::VecStr, Z::MatNum;
                              amsk::Option{<:AbstractMatrix{Bool}} = nothing,
                              emsk::Option{<:AbstractMatrix{Bool}} = nothing)
    assert_feature_matrix_columns(nz, Z)
    @argcheck(isnothing(amsk) && isnothing(emsk),
              DimensionMismatch("a static assets × features Feature Matrix builds a static Asset Panel, which carries no universe mask; pass a time-varying observations × assets × features Z instead"))
    return AssetPanel(;
                      pf = [NumericPanelField(; name = String(nz[k]), vals = view(Z, :, k))
                            for k in eachindex(nz)])
end
function feature_matrix_panel(nz::VecStr, Z::Arr3Num;
                              amsk::Option{<:AbstractMatrix{Bool}} = nothing,
                              emsk::Option{<:AbstractMatrix{Bool}} = nothing)
    assert_feature_matrix_columns(nz, Z)
    T, N = size(Z, 1), size(Z, 2)
    return AssetPanel(;
                      pf = [NumericPanelField(; name = String(nz[k]),
                                              vals = view(Z, :, :, k))
                            for k in eachindex(nz)],
                      amsk = isnothing(amsk) ? trues(T, N) : amsk,
                      emsk = isnothing(emsk) ? trues(T, N) : emsk)
end
"""
    panel_mask_view(msk::Nothing, i, j) -> nothing
    panel_mask_view(msk::AbstractMatrix{Bool}, i, j) -> SubArray

View one universe mask of an [`AssetPanel`](@ref) over the observations `i` and the assets `j`.

# Algorithm

The method that Julia selects is the algorithm. A static panel carries no mask, so there is nothing to view.

# Arguments

  - `msk`: The mask, or `nothing`.
  - `i`: Observation index.
  - `j`: Asset index.

# Returns

  - A view of `msk`, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
"""
function panel_mask_view(::Nothing, ::Any, ::Any)
    return nothing
end
function panel_mask_view(msk::AbstractMatrix{Bool}, i, j)
    return view(msk, i, j)
end
"""
    port_opt_view(pnl::AssetPanel, i) -> AssetPanel
    port_opt_view(pnl::AssetPanel, i, j, sq::Bool = false) -> AssetPanel

Return a view of the [`AssetPanel`](@ref) over the observations `i` and the assets `j`.

Every Panel Field owns its values, so an asset view reaches them all: the one-argument arity keeps every observation and selects assets, and the three-argument arity selects both. A static panel has no observation axis and ignores the observation index, which is the same asymmetry the two [`port_opt_view`](@ref) arities have for `ivpa`.

`sq` says that the panel's Panel Fields **are** the assets, one per asset, which is what a square Feature Matrix becomes on a carrier. The Panel Field vector is then selected by the same asset index, so the feature axis follows the universe.

# Algorithm

 1. Select the Panel Fields by `j` when `sq`, and keep them all otherwise.
 2. View each surviving Panel Field with [`panel_field_view`](@ref), passing a `Colon` for the observation index of a static panel.
 3. View both masks with [`panel_mask_view`](@ref), which keeps them `nothing` when the panel is static.

# Arguments

  - `pnl`: The Asset Panel.
  - `i`: Observation index.
  - `j`: Asset index.
  - `sq`: Whether the Panel Fields are the assets.

# Returns

  - `new_pnl::AssetPanel`: An Asset Panel over the selected observations and assets.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_field_view`](@ref)
  - [`panel_mask_view`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
"""
function port_opt_view(pnl::AssetPanel, i)
    return port_opt_view(pnl, :, i, false)
end
function port_opt_view(pnl::AssetPanel, i, j, sq::Bool = false)
    pf = sq ? view(pnl.pf, j) : pnl.pf
    it = panel_is_static(pnl) ? Colon() : i
    return AssetPanel(; pf = [panel_field_view(f, it, j) for f in pf],
                      amsk = panel_mask_view(pnl.amsk, i, j),
                      emsk = panel_mask_view(pnl.emsk, i, j))
end
"""
    check_asset_panel(pnl::Nothing, na, nobs, na_sym) -> nothing
    check_asset_panel(pnl::AssetPanel, na, nobs, na_sym) -> nothing

Check an [`AssetPanel`](@ref) against the asset and observation axes of the carrier that holds it.

The panel owns its own values, so this is the only check a carrier owes it: that the universe it describes is the carrier's universe.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: the carrier has no panel, so there is nothing to check.
 2. `pnl` is an [`AssetPanel`](@ref): read its shape from [`panel_field_axes`](@ref), check the asset axis against `na`, and check the observation axis against `nobs` when the panel is time-varying.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `na`: Asset count of the carrier.
  - `nobs`: Observation count of the carrier.
  - `na_sym`: Symbolic name of the asset axis, displayed in the error messages.

# Validation

  - `na` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - The panel's asset axis is `na`. Raises a `DimensionMismatch`.
  - `nobs` is not `nothing` and matches the panel's observation axis, when the panel is time-varying. Raises an [`IsNothingError`](@ref) or a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`PricesResult`](@ref)
  - [`panel_field_axes`](@ref)
  - [`Option`](@ref)
  - [`Sym_Str`](@ref)
"""
function check_asset_panel(::Nothing, ::Option{<:Integer}, ::Option{<:Integer},
                           ::Sym_Str)::Nothing
    return nothing
end
function check_asset_panel(pnl::AssetPanel, na::Option{<:Integer}, nobs::Option{<:Integer},
                           na_sym::Sym_Str)::Nothing
    ax = panel_field_axes(pnl.pf[1])
    @argcheck(!isnothing(na),
              IsNothingError("an Asset Panel (pnl) describes a universe, so it needs an asset axis to bind to, but $na_sym is nothing"))
    @argcheck(ax[end] == na,
              DimensionMismatch("the Panel Fields of an Asset Panel are indexed by asset, so their asset axis must be the carrier's, got $(ax[end]) and $na_sym = $na"))
    if length(ax) == 2
        @argcheck(!isnothing(nobs),
                  IsNothingError("a time-varying Asset Panel (pnl) has an observation axis to bind to; provide the asset data its observations are parallel to, or pass a static Asset Panel instead"))
        @argcheck(ax[1] == nobs,
                  DimensionMismatch("a time-varying Asset Panel is observations × assets, so its leading axis must be the carrier's observations, got $(ax[1]) and $nobs observations"))
    end
    return nothing
end
export AssetPanel, NumericPanelField, CategoricalPanelField, TensorPanelField, panel_field,
       panel_feature_matrix, feature_matrix_panel
