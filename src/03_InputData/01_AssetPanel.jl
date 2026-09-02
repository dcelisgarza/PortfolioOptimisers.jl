"""
$(DocStringExtensions.TYPEDEF)

Supertype of the kinds a Panel Field can take.

All concrete types describing the shape and the metadata of one Panel Field should subtype `AbstractPanelFieldKind`.

A kind answers two questions about one Panel Field: how many columns of `nz` its values occupy, and what metadata those columns carry. The values themselves never live here — they live in the carrier's `Z`, and [`PanelField`](@ref) records which columns hold them.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractPanelFieldKind` and implement the following methods:

## `panel_field_labels`

  - `panel_field_labels(kind::AbstractPanelFieldKind, name::AbstractString) -> Vector{String}`: The column names the kind contributes to `nz`, one per value column, in column order.

### Arguments

  - `kind`: The concrete subtype instance.
  - `name`: The Panel Field's own name.

### Returns

  - `labels::Vector{String}`: One name per value column.

## `panel_field_observables`

  - `panel_field_observables(kind::AbstractPanelFieldKind) -> Int`: The number of observed-mask columns the kind needs when its Panel Field can blank.

### Arguments

  - `kind`: The concrete subtype instance.

### Returns

  - `n::Int`: The count of observed-mask columns.

# Related

  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`PanelField`](@ref)
  - [`AssetPanel`](@ref)
"""
abstract type AbstractPanelFieldKind <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Marks a Panel Field that holds one numeric quantity per observation and asset.

A market capitalisation, a book equity or a trailing volatility is this kind. It occupies exactly one column of `nz`, named after the Panel Field itself.

# Constructors

    NumericPanelField() -> NumericPanelField

# Examples

```jldoctest
julia> NumericPanelField()
NumericPanelField()
```

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`PanelField`](@ref)
"""
struct NumericPanelField <: AbstractPanelFieldKind end
"""
    assert_panel_labels(labels::VecStr, sym::Sym_Str) -> nothing

Check that a Panel Field's label vector is non-empty, unique, and free of empty entries.

The three checks are shared by [`CategoricalPanelField`](@ref) and [`TensorPanelField`](@ref). Labels name columns of `nz`, and `nz` itself is checked for uniqueness by [`ReturnsResult`](@ref), so a duplicate caught here names the Panel Field that produced it instead of the carrier that received it.

# Algorithm

 1. Check that `labels` is not empty.
 2. Check that no entry of `labels` is the empty string, naming the first offending position.
 3. Check that `labels` holds no duplicate, naming the first repeated entry.

# Arguments

  - `labels`: The label vector to check.
  - `sym`: Symbolic name of the vector, displayed in the error messages.

# Validation

  - `!isempty(labels)`. Raises an [`IsEmptyError`](@ref).
  - `all(!isempty, labels)`. Raises an `ArgumentError` naming the first empty position.
  - `allunique(labels)`. Raises an `ArgumentError` naming the first repeated entry.

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
              IsEmptyError("$sym cannot be empty: a Panel Field with no labels claims no column of the feature axis"))
    i = findfirst(isempty, labels)
    @argcheck(isnothing(i),
              ArgumentError("$sym names columns of the feature axis, so no entry may be the empty string; the first empty entry is at position $i"))
    j = findfirst(k -> labels[k] in view(labels, 1:(k - 1)), eachindex(labels))
    @argcheck(isnothing(j),
              ArgumentError("$sym must be unique, because each entry names one column of the feature axis; the first repeated entry is at position $j"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Marks a Panel Field that holds one category label per observation and asset.

A sector, an industry or a country classification is this kind. It occupies one column of `nz` per level, one-hot encoded, and the columns are named `"<field>=<level>"` — the convention [`asset_sets_features`](@ref) already writes a taxonomy under. No side table of integer codes is needed, because a one-hot column is numeric and finite and therefore rides `Z` unchanged.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CategoricalPanelField(;
        levels::VecStr
    ) -> CategoricalPanelField

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(levels)`, `allunique(levels)`, and no level is the empty string (see [`assert_panel_labels`](@ref)).

# Examples

```jldoctest
julia> CategoricalPanelField(; levels = [\"Tech\", \"Energy\"])
CategoricalPanelField
  levels ┴ Vector{String}: ["Tech", "Energy"]
```

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`NumericPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`PanelField`](@ref)
  - [`assert_panel_labels`](@ref)
  - [`asset_sets_features`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct CategoricalPanelField <: AbstractPanelFieldKind
    """
    The category levels, one per one-hot column, in column order.
    """
    levels
    function CategoricalPanelField(levels::VecStr)
        assert_panel_labels(levels, :levels)
        return new{typeof(levels)}(levels)
    end
end
function CategoricalPanelField(; levels::VecStr)::CategoricalPanelField
    return CategoricalPanelField(levels)
end
"""
$(DocStringExtensions.TYPEDEF)

Marks a Panel Field whose third axis carries its own labels, and optionally its own groups.

A factor exposure tensor is this kind: its third axis is the factors, and its groups are the Factor Families. It occupies one column of `nz` per third-axis label, named `"<field>=<label>"`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TensorPanelField(;
        axis::AbstractString,
        labels::VecStr,
        groups::Option{<:VecStr} = nothing
    ) -> TensorPanelField

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(axis)`.
  - `!isempty(labels)`, `allunique(labels)`, and no label is the empty string (see [`assert_panel_labels`](@ref)).
  - If `groups` is not `nothing`, `length(groups) == length(labels)`. Groups repeat by design — a Factor Family names many factors — so they carry no uniqueness rule.

# Examples

```jldoctest
julia> TensorPanelField(; axis = \"factor\", labels = [\"size\", \"value\"])
TensorPanelField
    axis ┼ String: "factor"
  labels ┼ Vector{String}: ["size", "value"]
  groups ┴ nothing
```

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`PanelField`](@ref)
  - [`assert_panel_labels`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
@concrete struct TensorPanelField <: AbstractPanelFieldKind
    """
    Name of what the third axis represents, such as `"factor"`.
    """
    axis
    """
    Labels of the third-axis entries, one per column, in column order.
    """
    labels
    """
    Optional group of each third-axis entry, such as a Factor Family, one per label.
    """
    groups
    function TensorPanelField(axis::AbstractString, labels::VecStr,
                              groups::Option{<:VecStr})
        @argcheck(!isempty(axis),
                  IsEmptyError("the third-axis name (axis) of a tensor Panel Field cannot be empty: it names what the third axis represents, such as \"factor\""))
        assert_panel_labels(labels, :labels)
        if !isnothing(groups)
            @argcheck(length(groups) == length(labels),
                      DimensionMismatch("a tensor Panel Field needs one group per third-axis label, got length(groups) = $(length(groups)) and length(labels) = $(length(labels))"))
        end
        return new{typeof(axis), typeof(labels), typeof(groups)}(axis, labels, groups)
    end
end
function TensorPanelField(; axis::AbstractString, labels::VecStr,
                          groups::Option{<:VecStr} = nothing)::TensorPanelField
    return TensorPanelField(axis, labels, groups)
end
"""
    panel_field_labels(kind::NumericPanelField, name::AbstractString) -> Vector{String}
    panel_field_labels(kind::CategoricalPanelField, name::AbstractString) -> Vector{String}
    panel_field_labels(kind::TensorPanelField, name::AbstractString) -> Vector{String}

Return the column names one Panel Field kind contributes to `nz`, in column order.

# Algorithm

The method that Julia selects is the algorithm. Each kind names its columns differently.

 1. [`NumericPanelField`](@ref): one column, named after the Panel Field itself.
 2. [`CategoricalPanelField`](@ref): one column per level, named `"<name>=<level>"`.
 3. [`TensorPanelField`](@ref): one column per third-axis label, named `"<name>=<label>"`.

# Arguments

  - `kind`: The Panel Field's kind.
  - `name`: The Panel Field's own name.

# Returns

  - `labels::Vector{String}`: One name per value column.

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`panel_field_observables`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_field_labels(::NumericPanelField, name::AbstractString)::Vector{String}
    return [String(name)]
end
function panel_field_labels(kind::CategoricalPanelField,
                            name::AbstractString)::Vector{String}
    return ["$name=$level" for level in kind.levels]
end
function panel_field_labels(kind::TensorPanelField, name::AbstractString)::Vector{String}
    return ["$name=$label" for label in kind.labels]
end
"""
    panel_field_observables(kind::NumericPanelField) -> Int
    panel_field_observables(kind::CategoricalPanelField) -> Int
    panel_field_observables(kind::TensorPanelField) -> Int

Return the number of observed-mask columns a Panel Field kind needs when it can blank.

A numeric Panel Field observes one quantity, and a categorical one observes one label, so each needs one mask column. A tensor Panel Field observes one entry per third-axis label, so it needs one mask column per label and keeps the per-entry resolution the raw input carried.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref) and [`CategoricalPanelField`](@ref): one column.
 2. [`TensorPanelField`](@ref): `length(kind.labels)` columns.

# Arguments

  - `kind`: The Panel Field's kind.

# Returns

  - `n::Int`: The count of observed-mask columns.

# Related

  - [`AbstractPanelFieldKind`](@ref)
  - [`panel_field_labels`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_field_observables(::NumericPanelField)::Int
    return 1
end
function panel_field_observables(::CategoricalPanelField)::Int
    return 1
end
function panel_field_observables(kind::TensorPanelField)::Int
    return length(kind.labels)
end
"""
    assert_panel_columns(cols::VecInt, sym::Sym_Str, name::AbstractString) -> nothing

Check that a Panel Field's column vector is non-empty, unique, and strictly positive.

Shared by the two column vectors of [`PanelField`](@ref), so a value column and an observed-mask column fail the same way and name the same Panel Field.

# Algorithm

 1. Check that `cols` is not empty.
 2. Check that every entry is `> 0`, naming the first offending position.
 3. Check that `cols` holds no duplicate, naming the first repeated entry.

# Arguments

  - `cols`: The column vector to check.
  - `sym`: Symbolic name of the vector, displayed in the error messages.
  - `name`: The Panel Field's name, displayed in the error messages.

# Validation

  - `!isempty(cols)`. Raises an [`IsEmptyError`](@ref).
  - `all(>(0), cols)`. Raises a `DomainError`.
  - `allunique(cols)`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`PanelField`](@ref)
  - [`VecInt`](@ref)
"""
function assert_panel_columns(cols::VecInt, sym::Sym_Str, name::AbstractString)::Nothing
    @argcheck(!isempty(cols),
              IsEmptyError("$sym of the Panel Field \"$name\" cannot be empty: a Panel Field that claims no column carries no data"))
    i = findfirst(<=(0), cols)
    @argcheck(isnothing(i),
              DomainError(cols,
                          "$sym of the Panel Field \"$name\" indexes the feature axis, so every entry must be > 0; the first offending entry is at position $i"))
    j = findfirst(k -> cols[k] in view(cols, 1:(k - 1)), eachindex(cols))
    @argcheck(isnothing(j),
              ArgumentError("$sym of the Panel Field \"$name\" must be unique, because each entry names one column of the feature axis; the first repeated entry is at position $j"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

One row of an [`AssetPanel`](@ref)'s field index: a Panel Field's name, its kind, and its columns in `nz`.

This is what makes the index an index rather than a naming convention. A consumer looks a Panel Field up and reads its columns as integers, so nothing parses a column name, and a Panel Field whose own name carries the convention's punctuation cannot collide silently with a level of another.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PanelField(;
        name::AbstractString,
        kind::AbstractPanelFieldKind,
        cols::VecInt,
        ocols::Option{<:VecInt} = nothing
    ) -> PanelField

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(name)`.
  - `!isempty(cols)`, `allunique(cols)`, and `all(>(0), cols)` (see [`assert_panel_columns`](@ref)).
  - `length(cols) == length(panel_field_labels(kind, name))`, which is 1 for a [`NumericPanelField`](@ref), the level count for a [`CategoricalPanelField`](@ref), and the label count for a [`TensorPanelField`](@ref).
  - If `ocols` is not `nothing`: it passes [`assert_panel_columns`](@ref), `length(ocols) == panel_field_observables(kind)`, and `ocols` shares no entry with `cols`.

# Examples

```jldoctest
julia> PanelField(; name = \"mcap\", kind = NumericPanelField(), cols = [1])
PanelField
   name ┼ String: "mcap"
   kind ┼ NumericPanelField()
   cols ┼ Vector{Int64}: [1]
  ocols ┴ nothing
```

# Related

  - [`AssetPanel`](@ref)
  - [`AbstractPanelFieldKind`](@ref)
  - [`assert_panel_columns`](@ref)
  - [`panel_field_labels`](@ref)
  - [`panel_field_observables`](@ref)
  - [`Option`](@ref)
  - [`VecInt`](@ref)
"""
@concrete struct PanelField <: AbstractResult
    """
    The Panel Field's name, which is the key the field index is looked up by.
    """
    name
    """
    The Panel Field's kind, which fixes how many columns it claims and what they mean.
    """
    kind
    """
    Columns of `nz` holding the Panel Field's values, in the kind's own column order.
    """
    cols
    """
    Columns of `nz` holding the Panel Field's observed mask, or `nothing` when the Panel Field cannot blank.
    """
    ocols
    function PanelField(name::AbstractString, kind::AbstractPanelFieldKind, cols::VecInt,
                        ocols::Option{<:VecInt})
        @argcheck(!isempty(name),
                  IsEmptyError("the name of a Panel Field cannot be empty: it is the key the field index is looked up by"))
        assert_panel_columns(cols, :cols, name)
        nc = length(panel_field_labels(kind, name))
        @argcheck(length(cols) == nc,
                  DimensionMismatch("the Panel Field \"$name\" is a $(nameof(typeof(kind))), which claims $nc column(s) of the feature axis, got length(cols) = $(length(cols))"))
        if !isnothing(ocols)
            assert_panel_columns(ocols, :ocols, name)
            no = panel_field_observables(kind)
            @argcheck(length(ocols) == no,
                      DimensionMismatch("the Panel Field \"$name\" is a $(nameof(typeof(kind))), which needs $no observed-mask column(s), got length(ocols) = $(length(ocols))"))
            @argcheck(isdisjoint(ocols, cols),
                      ArgumentError("the Panel Field \"$name\" claims a column of the feature axis as both a value column and an observed-mask column, and a column carries one meaning; the shared column(s) are $(sort!(collect(intersect(ocols, cols))))"))
        end
        return new{typeof(name), typeof(kind), typeof(cols), typeof(ocols)}(name, kind,
                                                                            cols, ocols)
    end
end
function PanelField(; name::AbstractString, kind::AbstractPanelFieldKind, cols::VecInt,
                    ocols::Option{<:VecInt} = nothing)::PanelField
    return PanelField(name, kind, cols, ocols)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the structure of an Asset Panel: its field index, and the two point-in-time universe masks.

The panel's *values* are not here. They ride the feature matrix the carrier already holds — `ReturnsResult.Z` for the numbers and `ReturnsResult.nz` for the column names — and this result records what those columns mean. Splitting the panel this way is what lets a point-in-time panel travel through every view and every cross-validation fold that already slices `Z` in step with `X`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetPanel(;
        pf::AbstractVector{<:PanelField},
        amsk::AbstractMatrix{Bool},
        emsk::AbstractMatrix{Bool}
    ) -> AssetPanel

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(pf)`, and the Panel Field names are unique.
  - No two Panel Fields claim the same column of the feature axis, counting value columns and observed-mask columns alike.
  - `!isempty(amsk)` and `size(amsk) == size(emsk)`.
  - `all(emsk .<= amsk)`, the subset invariant: an asset that is not listed cannot be estimated. The rule is **checked, not coerced**. A coercion would allocate a new mask, and [`port_opt_view`](@ref) must return views; a slice of a pair that already satisfies the rule satisfies it again, so a view never has to re-establish it.

## View propagation

[`port_opt_view`](@ref) slices both masks and carries `pf` through unchanged. The field index addresses the *feature* axis, which no asset view and no observation view touches: a panel's column names are not asset names, so [`feature_matrix_view`](@ref) leaves that axis whole.

# Examples

```jldoctest
julia> AssetPanel(; pf = [PanelField(; name = \"mcap\", kind = NumericPanelField(), cols = [1])],
                  amsk = trues(2, 3), emsk = trues(2, 3))
AssetPanel
    pf ┼ 1-element Vector{PanelField}
       │ PanelField ⋯
  amsk ┼ 2×3 BitMatrix
  emsk ┴ 2×3 BitMatrix
```

# Related

  - [`PanelField`](@ref)
  - [`ReturnsResult`](@ref)
  - [`asset_panel`](@ref)
  - [`panel_field`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct AssetPanel <: AbstractResult
    """
    The field index: one [`PanelField`](@ref) per Panel Field, naming its kind and its columns of `nz`.
    """
    pf
    """
    The active mask (observations × assets): whether the asset is in the universe at that observation.
    """
    amsk
    """
    The estimation mask (observations × assets): whether the asset enters the cross-sectional estimate at that observation. Always a subset of the active mask.
    """
    emsk
    function AssetPanel(pf::AbstractVector{<:PanelField}, amsk::AbstractMatrix{Bool},
                        emsk::AbstractMatrix{Bool})
        @argcheck(!isempty(pf),
                  IsEmptyError("an Asset Panel needs at least one Panel Field: an empty field index describes no column of the feature axis"))
        assert_panel_labels([f.name for f in pf], "the Panel Field names")
        assert_panel_field_columns(pf)
        @argcheck(!isempty(amsk),
                  IsEmptyError("the active mask (amsk) of an Asset Panel cannot be empty"))
        @argcheck(size(amsk) == size(emsk),
                  DimensionMismatch("the two Asset Panel masks are both observations × assets, so they must have the same size, got size(amsk) = $(size(amsk)) and size(emsk) = $(size(emsk))"))
        idx = findfirst(k -> emsk[k] && !amsk[k], eachindex(emsk, amsk))
        @argcheck(isnothing(idx),
                  ArgumentError("the estimation mask (emsk) must be a subset of the active mask (amsk): an asset that is not in the universe at an observation cannot enter that observation's estimate. Intersect them yourself with `emsk .& amsk` — the rule is checked rather than coerced, because a coercion allocates and port_opt_view returns views. The first offending entry is at $(isnothing(idx) ? "" : string(Tuple(CartesianIndices(emsk)[idx])))"))
        return new{typeof(pf), typeof(amsk), typeof(emsk)}(pf, amsk, emsk)
    end
end
function AssetPanel(; pf::AbstractVector{<:PanelField}, amsk::AbstractMatrix{Bool},
                    emsk::AbstractMatrix{Bool})::AssetPanel
    return AssetPanel(pf, amsk, emsk)
end
"""
    assert_panel_field_columns(pf::AbstractVector{<:PanelField}) -> nothing

Check that no two Panel Fields of a field index claim the same column of the feature axis.

Value columns and observed-mask columns share one axis, so the check runs over their union. Two Panel Fields that claim one column give that column two meanings, and every consumer that reads it through the index gets one of them at random.

# Algorithm

 1. Walk the Panel Fields in order, and for each walk its value columns and then its observed-mask columns.
 2. Record the Panel Field that first claimed each column.
 3. Throw on the first column that is claimed twice, naming both Panel Fields and the column.

# Arguments

  - `pf`: The field index to check.

# Validation

  - Every column appears in at most one Panel Field's `cols` or `ocols`. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`PanelField`](@ref)
"""
function assert_panel_field_columns(pf::AbstractVector{<:PanelField})::Nothing
    owner = Dict{Int, String}()
    for f in pf
        for c in (isnothing(f.ocols) ? f.cols : vcat(f.cols, f.ocols))
            prev = get(owner, c, nothing)
            @argcheck(isnothing(prev),
                      ArgumentError("the Panel Fields \"$prev\" and \"$(f.name)\" both claim column $c of the feature axis; a column carries one meaning, so the field index must partition the columns it names"))
            owner[c] = f.name
        end
    end
    return nothing
end
"""
    panel_field(pnl::AssetPanel, name::AbstractString) -> PanelField

Look one Panel Field up in an [`AssetPanel`](@ref)'s field index by name.

This is the only supported route from a Panel Field's name to its columns. A consumer that parses a column name of `nz` instead is reading a convention rather than the index, and the two part company as soon as a Panel Field's own name carries the convention's punctuation.

# Algorithm

 1. Find the first Panel Field of `pnl.pf` whose `name` matches.
 2. Return it, or throw a `KeyError` carrying a [`did_you_mean`](@ref) suggestion drawn from the Panel Field names.

# Arguments

  - `pnl`: The Asset Panel to read.
  - `name`: The Panel Field's name.

# Validation

  - `name` names a Panel Field of `pnl.pf`. Raises a `KeyError`.

# Returns

  - `f::PanelField`: The index row for `name`.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [PanelField(; name = \"mcap\", kind = NumericPanelField(), cols = [1])],
                        amsk = trues(2, 3), emsk = trues(2, 3));

julia> PortfolioOptimisers.panel_field(pnl, \"mcap\").cols
1-element Vector{Int64}:
 1
```

# Related

  - [`AssetPanel`](@ref)
  - [`PanelField`](@ref)
  - [`did_you_mean`](@ref)
"""
function panel_field(pnl::AssetPanel, name::AbstractString)::PanelField
    i = findfirst(f -> f.name == name, pnl.pf)
    @argcheck(!isnothing(i),
              KeyError("the Asset Panel holds no Panel Field named `$name`$(did_you_mean(name, [f.name for f in pnl.pf])). It holds $(length(pnl.pf)): $(join([f.name for f in pnl.pf], ", "))"))
    return pnl.pf[i]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the [`AssetPanel`](@ref) for the assets at indices `i`.

Both masks are `observations × assets`, so an asset selection slices their columns and keeps every observation. The field index passes through untouched: it addresses the feature axis, and an asset view does not reach it.

# Algorithm

 1. View both masks as `view(mask, :, i)`.
 2. Rebuild the [`AssetPanel`](@ref) with the field index unchanged.

No step copies data. The subset invariant survives the slice, because a slice of two masks that satisfy it satisfies it again.

# Arguments

  - `pnl`: The Asset Panel to view.
  - `i`: Indices of the assets to view.

# Returns

  - `new_pnl::AssetPanel`: An Asset Panel over the selected assets.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)

* * *

    port_opt_view(
        pnl::AssetPanel,
        i,
        j,
        k = :
    ) -> AssetPanel

Return a view of the [`AssetPanel`](@ref) for observations at indices `i` and assets at indices `j`.

!!! warning

    As on [`port_opt_view(rd::ReturnsResult, i, j, k)`](@ref), the first index of this arity selects **observations**, not assets. The two-argument arity selects assets.

# Algorithm

 1. View both masks as `view(mask, i, j)`, which slices the observation axis and the asset axis together.
 2. Rebuild the [`AssetPanel`](@ref) with the field index unchanged. The factor index `k` reaches no axis a panel carries, so it is ignored.

# Arguments

  - `pnl`: The Asset Panel to view.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view. A panel has no factor axis, so this is ignored.

# Returns

  - `new_pnl::AssetPanel`: An Asset Panel over the selected observations and assets.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
"""
function port_opt_view(pnl::AssetPanel, i)
    return AssetPanel(; pf = pnl.pf, amsk = view(pnl.amsk, :, i),
                      emsk = view(pnl.emsk, :, i))
end
function port_opt_view(pnl::AssetPanel, i, j, ::Any = :)
    return AssetPanel(; pf = pnl.pf, amsk = view(pnl.amsk, i, j),
                      emsk = view(pnl.emsk, i, j))
end
"""
    check_asset_panel(pnl::Nothing, nz, Z) -> nothing
    check_asset_panel(pnl::AssetPanel, nz, Z) -> nothing

Check an [`AssetPanel`](@ref) against the feature axis and the feature matrix that carry its values.

The panel is *structure*; `nz` and `Z` are the values it describes. So a carrier that holds one holds all three, and the three must agree. This is the one check that needs all of them, which is why it is its own function rather than a clause of [`check_names_and_feature_matrix`](@ref): a carrier with a feature matrix and no panel is ordinary, and must not pay for the panel's rules.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: the carrier has no panel, so there is nothing to check.
 2. `pnl` is an [`AssetPanel`](@ref): check that `nz` and `Z` are both given, that `Z` is the time-varying shape, that the masks match `Z`'s observation and asset axes, and that every column the field index names is a column of `nz`.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `nz`: The feature axis of the carrier.
  - `Z`: The feature matrix of the carrier.

# Validation

  - `nz` and `Z` are both given. Raises an [`IsNothingError`](@ref).
  - `Z` is an [`Arr3Num`](@ref). A point-in-time panel varies in time, so a static `assets × features` feature matrix cannot carry one. Raises a `DimensionMismatch`.
  - `size(pnl.amsk) == (size(Z, 1), size(Z, 2))`. Raises a `DimensionMismatch`.
  - Every value column and every observed-mask column of every Panel Field lies in `1:length(nz)`. Raises a `DimensionMismatch` naming the Panel Field.

# Returns

  - `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`check_names_and_feature_matrix`](@ref)
  - [`Arr3Num`](@ref)
"""
function check_asset_panel(::Nothing, ::Any, ::Any)::Nothing
    return nothing
end
function check_asset_panel(pnl::AssetPanel, nz::Option{<:VecStr},
                           Z::Option{<:MatNum_Arr3Num})::Nothing
    @argcheck(!isnothing(nz) && !isnothing(Z),
              IsNothingError("an Asset Panel (pnl) describes the columns of a feature matrix, so it needs the feature names (nz) and the feature matrix (Z) beside it; got nz = $(isnothing(nz) ? "nothing" : "a vector") and Z = $(isnothing(Z) ? "nothing" : "an array")"))
    @argcheck(isa(Z, Arr3Num),
              DimensionMismatch("an Asset Panel is point-in-time, so its feature matrix (Z) is the time-varying shape, observations × assets × features; got a static $(ndims(Z))-dimensional Z of size $(size(Z))"))
    @argcheck(size(pnl.amsk) == (size(Z, 1), size(Z, 2)),
              DimensionMismatch("the Asset Panel masks are observations × assets, so they must match the first two axes of the feature matrix (Z), got size(pnl.amsk) = $(size(pnl.amsk)) and size(Z)[1:2] = $(size(Z)[1:2])"))
    nzl = length(nz)
    for f in pnl.pf
        for c in (isnothing(f.ocols) ? f.cols : vcat(f.cols, f.ocols))
            @argcheck(c <= nzl,
                      DimensionMismatch("the Panel Field \"$(f.name)\" names column $c of the feature axis, which holds only $nzl column(s). The field index and the feature axis come out of one asset_panel call, so a mismatch means the two were built apart."))
        end
    end
    return nothing
end
export AssetPanel, PanelField, NumericPanelField, CategoricalPanelField, TensorPanelField,
       panel_field
