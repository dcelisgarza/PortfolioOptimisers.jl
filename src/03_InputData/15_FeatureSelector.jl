"""
$(DocStringExtensions.TYPEDEF)

Selects the labels of a tensor Panel Field whose group is one name, as the right side of a Feature Selector entry.

The entry `"exposures" => LabelGroup("momentum")` selects every label of the tensor Panel Field `"exposures"` whose group is `"momentum"`, one column per label, in the order of the labels. The groups of a factor exposure tensor are its Factor Families, so this entry selects one family. Two groups take two entries. A string on the right side of an entry already names one label, so a group needs its own type.

A group that the Panel Field does not hold, and a Panel Field that carries no groups, resolve against nothing. The entry is then dropped with a warning, or refused under `strict = true`, as an unknown label is.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LabelGroup(group::AbstractString) -> LabelGroup
    LabelGroup(; group::AbstractString) -> LabelGroup

## Validation

  - `!isempty(group)`. Raises an [`IsEmptyError`](@ref).

# Examples

```jldoctest
julia> f = TensorPanelField(; name = \"exposures\", axis = \"factor\",
                            labels = [\"mom_12\", \"value\", \"mom_6\"],
                            groups = [\"momentum\", \"value\", \"momentum\"],
                            vals = [1.0 2.0 3.0; 4.0 5.0 6.0]);

julia> pnl = AssetPanel(; pf = [f]);

julia> feature_matrix(pnl, [\"exposures\" => LabelGroup(\"momentum\")])
2×2 Matrix{Float64}:
 1.0  3.0
 4.0  6.0

julia> feature_labels(pnl, [\"exposures\" => LabelGroup(\"momentum\")])
2-element Vector{Pair{String, String}}:
 \"exposures\" => \"mom_12\"
 \"exposures\" => \"mom_6\"
```

# Related

  - [`TensorPanelField`](@ref)
  - [`feature_matrix`](@ref)
  - [`feature_labels`](@ref)
  - [`select_fields`](@ref)
  - [`panel_group_columns!`](@ref)
"""
@concrete struct LabelGroup <: AbstractAlgorithm
    """
    The group that the entry selects, one of the entries of the `groups` of a tensor Panel Field.
    """
    group
    function LabelGroup(group::AbstractString)
        @argcheck(!isempty(group),
                  IsEmptyError("the group of a LabelGroup cannot be empty: it names one entry of the `groups` of a tensor Panel Field"))
        return new{typeof(group)}(group)
    end
end
function LabelGroup(; group::AbstractString)
    return LabelGroup(group)
end
"""
    panel_field_keys(f::NumericPanelField) -> Vector{String}
    panel_field_keys(f::CategoricalPanelField) -> VecStr
    panel_field_keys(f::TensorPanelField) -> VecStr

Return the namespace a Feature Selector entry pairs with a Panel Field's name.

A numeric Panel Field contributes one column and has no second namespace, so it answers with an empty vector. Every key paired with it is then absent, and `strict` decides what that means. A categorical Panel Field answers with its levels and a tensor Panel Field with its labels, and in both the position of a key is the position of the column it selects.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `keys::VecStr`: The levels, the labels, or an empty vector.

# Related

  - [`AbstractPanelField`](@ref)
  - [`select_fields`](@ref)
  - [`feature_labels`](@ref)
  - [`VecStr`](@ref)
"""
function panel_field_keys(::NumericPanelField)
    return String[]
end
function panel_field_keys(f::CategoricalPanelField)
    return f.levels
end
function panel_field_keys(f::TensorPanelField)
    return f.labels
end
"""
    panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer, f::NumericPanelField) -> nothing
    panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer, f::CategoricalPanelField) -> nothing
    panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer, f::TensorPanelField) -> nothing

Push every value column of one Panel Field onto a resolved Feature Selector, in column order.

This is what a bare field name expands to, and what an absent selector expands to for every Panel Field of the panel. The mask is not among them, because a bare name selects the values alone.

# Algorithm

The method that Julia selects is the algorithm. A numeric Panel Field pushes its one column, and the other two push one column per key of [`panel_field_keys`](@ref).

# Arguments

  - `cols`: The resolved columns so far, pushed onto in place.
  - `k`: The Panel Field's position in the panel.
  - `f`: The Panel Field.

# Returns

  - `nothing`. `cols` carries the result.

# Related

  - [`select_fields`](@ref)
  - [`panel_field_keys`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer,
                              ::NumericPanelField)::Nothing
    push!(cols, (Int(k), 0, :vals))
    return nothing
end
function panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer,
                              f::CategoricalPanelField)::Nothing
    for l in eachindex(f.levels)
        push!(cols, (Int(k), Int(l), :vals))
    end
    return nothing
end
function panel_value_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer,
                              f::TensorPanelField)::Nothing
    for l in eachindex(f.labels)
        push!(cols, (Int(k), Int(l), :vals))
    end
    return nothing
end
"""
    panel_selector_msg(name, pool::VecStr) -> String
    panel_selector_msg(f::NumericPanelField, key, keys::VecStr) -> String
    panel_selector_msg(f::AbstractPanelField, key, keys::VecStr) -> String

Build the diagnostic for a Feature Selector entry that resolves against nothing.

The messages name the two namespaces an entry resolves in. The first is the Panel Field names of the panel, and the second is the levels or the labels of one Panel Field. Each method hands its own suggestion pool to [`did_you_mean`](@ref), so the suggestion for a misspelling comes from the names the entry could have meant and not from every column of the panel. A numeric Panel Field has no second namespace. For a key paired with it, the message names the two entry forms that the field takes, and it offers no suggestion.

# Algorithm

 1. Name the entry that resolved against nothing, and the namespace it was resolved in.
 2. Append the nearest match from the pool with [`did_you_mean`](@ref), where there is a pool.

# Arguments

  - `name`: The Panel Field name that resolved against nothing.
  - `pool`: The Panel Field names of the panel. See [`VecStr`](@ref).
  - `f`: The Panel Field whose key resolved against nothing.
  - `key`: The level or label that resolved against nothing.
  - `keys`: The Panel Field's levels or labels. See [`VecStr`](@ref).

# Returns

  - `msg::String`: The diagnostic.

# Related

  - [`select_fields`](@ref)
  - [`did_you_mean`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`VecStr`](@ref)
"""
function panel_selector_msg(name, pool::VecStr)
    return "`sel` names `$(name)`, which is not a Panel Field of the Asset Panel. It holds $(length(pool)): $(join(pool, ", ")). Under `strict = false` the entry is dropped." *
           did_you_mean(string(name), filter(!=(string(name)), pool))
end
function panel_selector_msg(f::NumericPanelField, key, ::VecStr)
    return "`sel` pairs the Panel Field \"$(f.name)\" with `$(key)`, but a numeric Panel Field contributes one column and has no levels or labels to pair with. Select it by its bare name, or pair it with `:observed` for its mask. Under `strict = false` the entry is dropped."
end
function panel_selector_msg(f::AbstractPanelField, key, keys::VecStr)
    what = isa(f, CategoricalPanelField) ? "level" : "label"
    return "`sel` pairs the Panel Field \"$(f.name)\" with `$(key)`, which is not one of its $(length(keys)) $(what)(s)$(isempty(keys) ? "" : ": " * join(keys, ", ")). Under `strict = false` the entry is dropped." *
           did_you_mean(string(key), filter(!=(string(key)), keys))
end
"""
    panel_key_column!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer, f::AbstractPanelField, key::AbstractString, strict::Bool) -> nothing

Push the one value column a Panel Field's level or label names onto a resolved Feature Selector.

# Algorithm

 1. Find `key` among the Panel Field's keys, from [`panel_field_keys`](@ref).
 2. Push the column it names when it is there.
 3. Otherwise hand [`panel_selector_msg`](@ref) to [`strict_diagnostic`](@ref), which throws under `strict` and warns and drops otherwise.

# Arguments

  - `cols`: The resolved columns so far, pushed onto in place.
  - `k`: The Panel Field's position in the panel.
  - `f`: The Panel Field.
  - `key`: One of its levels or labels.
  - $(field_dict[:fdstrict])

# Returns

  - `nothing`. `cols` carries the result.

# Related

  - [`select_fields`](@ref)
  - [`panel_field_keys`](@ref)
  - [`panel_selector_msg`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function panel_key_column!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer,
                           f::AbstractPanelField, key::AbstractString,
                           strict::Bool)::Nothing
    keys::VecStr = panel_field_keys(f)
    l = findfirst(==(key), keys)
    if isnothing(l)
        strict_diagnostic(panel_selector_msg(f, key, keys), strict)
    else
        push!(cols, (Int(k), Int(l), :vals))
    end
    return nothing
end
"""
    panel_group_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer, f::AbstractPanelField, lg::LabelGroup, strict::Bool) -> nothing

Push the value columns of every label in one group of a tensor Panel Field onto a resolved Feature Selector.

# Algorithm

 1. Read the groups of the Panel Field. A numeric or a categorical Panel Field has none, and a tensor Panel Field can carry none.
 2. When there are no groups, hand the message to [`strict_diagnostic`](@ref), which throws under `strict` and warns and drops otherwise.
 3. Find the labels whose group is `lg.group`. When there is none, hand the message to [`strict_diagnostic`](@ref), with the nearest group from [`did_you_mean`](@ref).
 4. Push the value column of each such label, in the order of the labels.

# Arguments

  - `cols`: The resolved columns so far, pushed onto in place.
  - `k`: The Panel Field's position in the panel.
  - `f`: The Panel Field.
  - `lg`: The group to select. See [`LabelGroup`](@ref).
  - $(field_dict[:fdstrict])

# Returns

  - `nothing`. `cols` carries the result.

# Related

  - [`LabelGroup`](@ref)
  - [`select_fields_push!`](@ref)
  - [`panel_key_column!`](@ref)
  - [`strict_diagnostic`](@ref)
  - [`TensorPanelField`](@ref)
"""
function panel_group_columns!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, k::Integer,
                              f::AbstractPanelField, lg::LabelGroup, strict::Bool)::Nothing
    groups = isa(f, TensorPanelField) ? f.groups : nothing
    if isnothing(groups)
        strict_diagnostic("`sel` pairs the Panel Field \"$(f.name)\" with the group `$(lg.group)`, but the field carries no groups. Only a tensor Panel Field built with `groups` has them. Under `strict = false` the entry is dropped.",
                          strict)
        return nothing
    end
    ls = findall(==(lg.group), groups)
    if isempty(ls)
        pool = unique(groups)
        strict_diagnostic("`sel` pairs the Panel Field \"$(f.name)\" with the group `$(lg.group)`, which is not one of its $(length(pool)) group(s): $(join(pool, ", ")). Under `strict = false` the entry is dropped." *
                          did_you_mean(lg.group, pool), strict)
    end
    for l in ls
        push!(cols, (Int(k), Int(l), :vals))
    end
    return nothing
end
"""
    selector_value_ok(v) -> Bool

Check the right side of a paired Feature Selector entry, whose left side names a Panel Field.

# Algorithm

The method that Julia selects is the algorithm.

 1. A `Symbol` is admitted when it is `:observed`.
 2. A string, one level or label, is admitted when it is not empty.
 3. A [`LabelGroup`](@ref) is admitted when its group is a non-empty string. The check reads the field, because a call of the generated positional constructor skips the check of the typed one.
 4. A vector of strings is admitted when it is not empty, holds no empty key and repeats none.
 5. Anything else is refused.

# Arguments

  - `v`: The right side of the pair.

# Returns

  - `ok::Bool`: Whether the right side takes an admitted form.

# Related

  - [`assert_selector_entry`](@ref)
  - [`LabelGroup`](@ref)
"""
function selector_value_ok(::Any)
    return false
end
function selector_value_ok(v::Symbol)
    return v === :observed
end
function selector_value_ok(v::AbstractString)
    return !isempty(v)
end
function selector_value_ok(v::LabelGroup)
    return isa(v.group, AbstractString) && !isempty(v.group)
end
function selector_value_ok(v::AbstractVector{<:AbstractString})
    return !isempty(v) && all(!isempty, v) && allunique(v)
end
"""
    assert_selector_entry(entry) -> nothing

Check that one Feature Selector entry takes one of the five forms the grammar admits.

The five forms are a Panel Field name, a name paired with the levels or labels it keeps, a name paired with one level or label, a name paired with a [`LabelGroup`](@ref), and a name paired with `:observed`. No entry is a column position. Every field, level and label of an Asset Panel carries a name, so an integer has nothing to index.

The check refuses an empty name, and a paired vector that is empty, holds an empty key or repeats a key. Each of those resolves to no column or to a doubled column, and the selector exists to prevent both.

# Algorithm

 1. A string: check that it is not empty.
 2. A pair whose first element is a non-empty string: check its second element with [`selector_value_ok`](@ref).
 3. Refuse anything else.

# Arguments

  - `entry`: One entry of a Feature Selector.

# Validation

  - `entry` takes one of the five forms. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`assert_feature_selector`](@ref)
  - [`selector_value_ok`](@ref)
  - [`select_fields`](@ref)
"""
function assert_selector_entry(entry)::Nothing
    ok = if isa(entry, AbstractString)
        !isempty(entry)
    elseif isa(entry, Pair) && isa(first(entry), AbstractString) && !isempty(first(entry))
        selector_value_ok(last(entry))
    else
        false
    end
    @argcheck(ok,
              ArgumentError("a `sel` entry is a Panel Field name, a name paired with the levels or labels it keeps, a name paired with one level or label, a name paired with a LabelGroup, or a name paired with `:observed`. A name is non-empty, and a paired vector is non-empty, holds no empty key and repeats none. Got\nentry => $(entry)"))
    return nothing
end
"""
    assert_feature_selector(sel::Nothing) -> nothing
    assert_feature_selector(sel::AbstractVector) -> nothing

Check that a Feature Selector is a non-empty vector of distinct entries, each in one of the five admitted forms.

The check refuses an empty `sel` and does not read it as every Panel Field. `nothing` already says that, and a selection that widens to the whole panel without a message is the failure this selector exists to prevent. The check refuses a repeated entry because it doubles the contribution of its column to every distance.

# Algorithm

The method that Julia selects decides whether there is anything to check.

 1. `sel` is `nothing`: it stacks every Panel Field's values, so there is nothing to check.
 2. `sel` is a vector: check that it is non-empty, that it repeats no entry, and that each entry takes an admitted form, with [`assert_selector_entry`](@ref).

# Arguments

  - $(field_dict[:fdsel])

# Validation

  - `!isempty(sel)`. Raises an [`IsEmptyError`](@ref).
  - `allunique(sel)`. Raises an `ArgumentError`.
  - Each entry takes an admitted form. See [`assert_selector_entry`](@ref).

# Returns

  - `nothing`.

# Related

  - [`select_fields`](@ref)
  - [`assert_selector_entry`](@ref)
  - [`FeatureDistance`](@ref)
  - [`IsEmptyError`](@ref)
"""
function assert_feature_selector(::Nothing)::Nothing
    return nothing
end
function assert_feature_selector(sel::AbstractVector)::Nothing
    @argcheck(!isempty(sel),
              IsEmptyError("`sel` cannot be empty. Pass `sel = nothing` to stack every Panel Field's values."))
    @argcheck(allunique(sel),
              ArgumentError("`sel` must not repeat an entry, because a repeated column doubles that column's contribution to every distance. Got\nsel => $(sel)"))
    for entry in sel
        assert_selector_entry(entry)
    end
    return nothing
end
"""
    select_fields_push!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, pnl::AssetPanel, entry, pool, strict::Bool) -> nothing

Resolve one Feature Selector entry against an [`AssetPanel`](@ref)'s Panel Fields.

# Algorithm

 1. Read the Panel Field name off the entry: a paired entry names it first, and a bare entry is the name.
 2. Find the Panel Field. Hand [`panel_selector_msg`](@ref) to [`strict_diagnostic`](@ref) and return when the panel holds none.
 3. A bare name expands to the Panel Field's value columns, through [`panel_value_columns!`](@ref).
 4. A name paired with a `Symbol` claims the Panel Field's one observed-mask column.
 5. A name paired with one key claims that key's column, through [`panel_key_column!`](@ref).
 6. A name paired with a [`LabelGroup`](@ref) claims the column of every label in that group, through [`panel_group_columns!`](@ref).
 7. A name paired with a vector of keys claims one column per key, in the order the vector writes them.

# Arguments

  - `cols`: The resolved columns so far, pushed onto in place.
  - `pnl`: The Asset Panel.
  - `entry`: One entry of the Feature Selector.
  - `pool`: The panel's Panel Field names, the suggestion pool of the first namespace.
  - $(field_dict[:fdstrict])

# Returns

  - `nothing`. `cols` carries the result.

# Related

  - [`select_fields`](@ref)
  - [`panel_value_columns!`](@ref)
  - [`panel_key_column!`](@ref)
  - [`panel_group_columns!`](@ref)
  - [`panel_selector_msg`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function select_fields_push!(cols::AbstractVector{Tuple{Int, Int, Symbol}}, pnl::AssetPanel,
                             entry, pool, strict::Bool)::Nothing
    name = isa(entry, Pair) ? first(entry) : entry
    k = findfirst(f -> f.name == name, pnl.pf)
    if isnothing(k)
        strict_diagnostic(panel_selector_msg(name, pool), strict)
        return nothing
    end
    f = pnl.pf[k]
    if !isa(entry, Pair)
        panel_value_columns!(cols, k, f)
    elseif isa(last(entry), Symbol)
        push!(cols, (Int(k), 0, :observed))
    elseif isa(last(entry), AbstractString)
        panel_key_column!(cols, k, f, last(entry), strict)
    elseif isa(last(entry), LabelGroup)
        panel_group_columns!(cols, k, f, last(entry), strict)
    else
        for key in last(entry)
            panel_key_column!(cols, k, f, key, strict)
        end
    end
    return nothing
end
"""
    select_fields(pnl::AssetPanel, sel, strict::Bool) -> Vector{Tuple{Int, Int, Symbol}}

Resolve a Feature Selector against an [`AssetPanel`](@ref), and return the columns it names.

[`feature_matrix`](@ref) and [`feature_labels`](@ref) both call this one resolution, so the matrix and its labels always name the same columns. It is unexported, because a caller reads the labels and not the positions.

A resolved column is the triple `(k, l, part)`. `k` is the Panel Field's position in the panel. `part` is `:vals` or `:observed`. `l` is the position of the level or the label within the Panel Field, and `0` where the Panel Field contributes one column of that part: a numeric Panel Field's value column, and every Panel Field's observed-mask column.

The order of `sel` is the column order, so a caller decides it. A `nothing` selector takes the panel's own order, and stacks each Panel Field's values without its mask.

# Algorithm

 1. Check that the panel holds a Panel Field. A panel with none carries no feature data. The check refuses it here, where the cause is still known, and not later, where only the empty matrix shows.
 2. Check `sel` with [`assert_feature_selector`](@ref).
 3. `sel` is `nothing`: push every Panel Field's value columns in panel order, and return.
 4. Otherwise resolve each entry with [`select_fields_push!`](@ref), in the order `sel` writes them.
 5. Check that no two entries resolved to one column.
 6. Check that the selection is not empty, which happens when every entry resolved against nothing and was dropped.

# Arguments

  - `pnl`: The Asset Panel.
  - $(field_dict[:fdsel])
  - $(field_dict[:fdstrict])

# Validation

  - `!isempty(pnl.pf)`. Raises an [`IsEmptyError`](@ref) naming the panel, the Feature Matrix and [`asset_panel`](@ref).
  - `sel` is well formed. See [`assert_feature_selector`](@ref).
  - No two entries resolve to one column. Raises an `ArgumentError`.
  - The selection is not empty. Raises an [`IsEmptyError`](@ref).

# Returns

  - `cols::Vector{Tuple{Int, Int, Symbol}}`: One triple per column, in column order.

# Related

  - [`AssetPanel`](@ref)
  - [`feature_matrix`](@ref)
  - [`feature_labels`](@ref)
  - [`select_fields_push!`](@ref)
  - [`assert_feature_selector`](@ref)
"""
function select_fields(pnl::AssetPanel, sel, strict::Bool)
    @argcheck(!isempty(pnl.pf),
              IsEmptyError("a Feature Matrix stacks the Panel Fields of an Asset Panel, and this panel carries none: it states a universe and carries no feature data, which is what a ReturnsResult built from prices alone holds. Build the panel the Feature Matrix is to stack with `asset_panel(inputs)` and pass it as `ReturnsResult(; …, pnl = pnl)`, or set a producer on the estimator, `FeatureDistance(; ape = RegressionPanel())`, which builds one from the prior it is handed."))
    assert_feature_selector(sel)
    cols = Tuple{Int, Int, Symbol}[]
    if isnothing(sel)
        for (k, f) in pairs(pnl.pf)
            panel_value_columns!(cols, k, f)
        end
        return cols
    end
    pool = [String(f.name) for f in pnl.pf]
    for entry in sel
        select_fields_push!(cols, pnl, entry, pool, strict)
    end
    @argcheck(allunique(cols),
              ArgumentError("two entries of `sel` resolve to one column of the Feature Matrix, which doubles that column's contribution to every distance. A bare Panel Field name already claims every one of its value columns, so pairing the same name with one of its levels or labels repeats a column, and a label entry repeats a column of a LabelGroup entry that holds the label. Got\nsel => $(sel)"))
    @argcheck(!isempty(cols),
              IsEmptyError("`sel` selected no column of the Asset Panel: every entry resolved against nothing and was dropped. Set `strict = true` to see which entry, or correct `sel`."))
    return cols
end
"""
    panel_column_label(pnl::AssetPanel, col::Tuple{Int, Int, Symbol})

Return the Feature Selector entry that selects exactly one resolved column.

A label is a selector entry and not a rendered string. The labels of a Feature Matrix are therefore a Feature Selector, and a stack of the panel against them rebuilds the same matrix column for column.

# Algorithm

 1. Read the Panel Field the column belongs to.
 2. An observed-mask column takes the name paired with `:observed`.
 3. A value column of a Panel Field that contributes one takes the bare name.
 4. Any other value column takes the name paired with its level or its label.

# Arguments

  - `pnl`: The Asset Panel.
  - `col`: One resolved column, from [`select_fields`](@ref).

# Returns

  - `label`: The Feature Selector entry naming that column.

# Related

  - [`feature_labels`](@ref)
  - [`select_fields`](@ref)
  - [`panel_field_keys`](@ref)
"""
function panel_column_label(pnl::AssetPanel, col::Tuple{Int, Int, Symbol})
    k, l, part = col
    name = String(pnl.pf[k].name)
    return if part === :observed
        name => :observed
    elseif iszero(l)
        name
    else
        name => String(panel_field_keys(pnl.pf[k])[l])
    end
end
"""
    panel_field_value_column!(zc::AbstractArray, f::NumericPanelField, l::Integer, rows) -> nothing
    panel_field_value_column!(zc::AbstractArray, f::CategoricalPanelField, l::Integer, rows) -> nothing
    panel_field_value_column!(zc::AbstractArray, f::TensorPanelField, l::Integer, rows) -> nothing

Write one Panel Field's value column into a Feature Matrix under construction.

The method cuts the Panel Field to `rows` along its leading axis before it writes the column. A stack of one observation therefore reads one row of the Panel Field, and it reads a lifted static Panel Field, whose values are a [`RepeatedLeading`](@ref), once and not once per observation. The cut is a `selectdim` along the leading axis, so `Colon()` is the whole field. The leading axis of a static panel is its asset axis, so the only cut of a static panel is `Colon()`. [`stacked_axes`](@ref) refuses every other value before this method runs.

# Algorithm

The method that Julia selects is the algorithm. A numeric Panel Field writes its values, a categorical Panel Field writes the `0`/`1` indicator of one level, and a tensor Panel Field writes one label slice of its values, each cut to `rows`.

# Arguments

  - `zc`: The column of the Feature Matrix, a view over the stacked observation rows and the panel's asset axis.
  - `f`: The Panel Field.
  - `l`: The position of the level or the label, and `0` for a numeric Panel Field.
  - $(arg_dict[:fdrows])

# Returns

  - `nothing`. `zc` carries the result.

# Related

  - [`feature_matrix`](@ref)
  - [`stacked_axes`](@ref)
  - [`select_fields`](@ref)
  - [`AbstractPanelField`](@ref)
  - [`RepeatedLeading`](@ref)
"""
function panel_field_value_column!(zc::AbstractArray, f::NumericPanelField, ::Integer,
                                   rows)::Nothing
    zc .= selectdim(f.vals, 1, rows)
    return nothing
end
function panel_field_value_column!(zc::AbstractArray, f::CategoricalPanelField, l::Integer,
                                   rows)::Nothing
    zc .= selectdim(f.codes, 1, rows) .== l
    return nothing
end
function panel_field_value_column!(zc::AbstractArray, f::TensorPanelField, l::Integer,
                                   rows)::Nothing
    zc .= selectdim(selectdim(f.vals, ndims(f.vals), l), 1, rows)
    return nothing
end
"""
    panel_field_observed_column!(zc::AbstractArray, f::NumericPanelField, rows) -> nothing
    panel_field_observed_column!(zc::AbstractArray, f::CategoricalPanelField, rows) -> nothing
    panel_field_observed_column!(zc::AbstractArray, f::TensorPanelField, rows) -> nothing

Write one Panel Field's observed mask into a Feature Matrix under construction, as a `0`/`1` column.

A Panel Field contributes **one** mask column, whatever its kind and however many value columns it contributes. The column shows whether the cell of that Panel Field was observed for that asset. The mask of a tensor Panel Field carries a label axis of its own, so its column holds where every label of that asset was observed.

A Panel Field that carries no mask gives a column of ones, because `omsk === nothing` means that the Panel Field cannot blank. The method cuts a mask to `rows` along its leading axis before it writes it, as [`panel_field_value_column!`](@ref) cuts a value column. A column of ones is the same whatever `rows` holds.

# Algorithm

The method that Julia selects is the algorithm.

 1. Write ones when the Panel Field carries no mask.
 2. A numeric or a categorical Panel Field writes its mask, cut to `rows`.
 3. A tensor Panel Field cuts its mask to `rows`, reduces it over the label axis with `all`, and writes that.

# Arguments

  - `zc`: The column of the Feature Matrix, a view over the stacked observation rows and the panel's asset axis.
  - `f`: The Panel Field.
  - $(arg_dict[:fdrows])

# Returns

  - `nothing`. `zc` carries the result.

# Related

  - [`feature_matrix`](@ref)
  - [`panel_field_value_column!`](@ref)
  - [`select_fields`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_field_observed_column!(zc::AbstractArray, f::NumericPanelField,
                                      rows)::Nothing
    zc .= isnothing(f.omsk) ? true : selectdim(f.omsk, 1, rows)
    return nothing
end
function panel_field_observed_column!(zc::AbstractArray, f::CategoricalPanelField,
                                      rows)::Nothing
    zc .= isnothing(f.omsk) ? true : selectdim(f.omsk, 1, rows)
    return nothing
end
function panel_field_observed_column!(zc::AbstractArray, f::TensorPanelField, rows)::Nothing
    omsk = f.omsk
    if isnothing(omsk)
        zc .= true
    else
        d = ndims(omsk)
        zc .= dropdims(all(selectdim(omsk, 1, rows); dims = d); dims = d)
    end
    return nothing
end
"""
    panel_data_mask(f::AbstractPanelField) -> Option{AbstractArray{Bool}}

Return the mask of the cells of a Panel Field that hold data, or `nothing` when every cell holds data.

A cell holds data where the raw input carried a value, or where a fill policy wrote one. It holds no data where it holds the placeholder of a blank that no fill reached. So the mask is the negation of the placeholder mask `pmsk`. A reader that must not take a placeholder as data, such as [`FeatureDistance`](@ref) and the panel collapse of a meta-optimiser, reads a cell only where this mask is `true`.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `d::Option{AbstractArray{Bool}}`: `.!f.pmsk`, the same size as the values, or `nothing` when `f.pmsk` is `nothing`.

# Related

  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`panel_field_column_mask`](@ref)
"""
function panel_data_mask(f::AbstractPanelField)
    pmsk = f.pmsk
    return isnothing(pmsk) ? nothing : .!pmsk
end
"""
    panel_field_column_mask(f::NumericPanelField, l::Integer, rows = Colon()) -> Option{AbstractArray{Bool}}
    panel_field_column_mask(f::CategoricalPanelField, l::Integer, rows = Colon()) -> Option{AbstractArray{Bool}}
    panel_field_column_mask(f::TensorPanelField, l::Integer, rows = Colon()) -> Option{AbstractArray{Bool}}

Return the mask of the cells of one value column of a Panel Field that hold data, or `nothing` when every cell holds data.

A cell holds data where the raw input carried a value or a fill policy wrote one. Elsewhere it holds the placeholder of a blank that no fill reached, and a reader must not take that placeholder as data. The mask is [`panel_data_mask`](@ref), the negation of the placeholder mask. A numeric Panel Field has one value column, and a categorical Panel Field holds a placeholder or not for the whole cell, never for one level, so both give their whole mask. A tensor Panel Field gives the mask of the label that the column reads. The method cuts the mask to `rows` along its leading axis, as [`panel_field_value_column!`](@ref) cuts a value column.

# Algorithm

The method that Julia selects is the algorithm.

 1. A Panel Field that holds no placeholder gives `nothing`.
 2. A numeric or a categorical Panel Field gives its mask, cut to `rows`.
 3. A tensor Panel Field gives the slice of its mask at label `l`, cut to `rows`.

# Arguments

  - `f`: The Panel Field.
  - `l`: The position of the level or the label, and `0` for a numeric Panel Field.
  - $(arg_dict[:fdrows])

# Returns

  - `m::Option{AbstractArray{Bool}}`: The mask, `assets` or `observations × assets`, or `nothing`.

# Related

  - [`feature_matrix`](@ref)
  - [`feature_data_cells`](@ref)
  - [`panel_data_mask`](@ref)
  - [`panel_field_value_column!`](@ref)
"""
function panel_field_column_mask(f::Union{<:NumericPanelField, <:CategoricalPanelField},
                                 ::Integer, rows = Colon())
    d = panel_data_mask(f)
    return isnothing(d) ? nothing : selectdim(d, 1, rows)
end
function panel_field_column_mask(f::TensorPanelField, l::Integer, rows = Colon())
    d = panel_data_mask(f)
    return isnothing(d) ? nothing : selectdim(selectdim(d, ndims(d), l), 1, rows)
end
"""
    feature_data_cells(pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}}) -> Option{AbstractArray{Bool}}

Mark the cells of an Asset Panel at which every resolved value column holds data.

A cell is one asset of a static panel, or one observation and one asset of a time-varying one. It is `true` where each value column of `cols` holds data, observed or filled, and `false` where one of them holds the placeholder of a blank that no fill reached. An observed-mask column holds data in every cell, so it adds nothing. The answer is `nothing` when no value column of `cols` holds a placeholder. A [`FeatureDistance`](@ref) reads an asset at a cell only where this mask and the active mask are both `true`.

# Algorithm

Take the conjunction of the masks that [`panel_field_column_mask`](@ref) gives for the value columns of `cols`, and skip each `nothing`.

# Arguments

  - `pnl`: The Asset Panel.
  - `cols`: The resolved columns, from [`select_fields`](@ref).

# Returns

  - `d::Option{AbstractArray{Bool}}`: The mask, with the shape of the panel's axes, or `nothing`.

# Related

  - [`feature_matrix`](@ref)
  - [`panel_field_column_mask`](@ref)
  - [`select_fields`](@ref)
  - [`FeatureDistance`](@ref)
"""
function feature_data_cells(pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}})
    d = nothing
    for (k, l, part) in cols
        m = part === :observed ? nothing : panel_field_column_mask(pnl.pf[k], l)
        if !isnothing(m)
            d = isnothing(d) ? BitArray(m) : (d .&= m)
        end
    end
    return d
end
"""
    stacked_axes(ax::Tuple, rows) -> Tuple

Cut the observation axis of an Asset Panel's axes to the rows a Feature Matrix stacks.

A time-varying panel has the axes `(observations, assets)`. Its Feature Matrix stacks the observation rows that `rows` names, so the matrix has one row for each of them. `rows` names a row by its position or by a `true` in a `Bool` mask, because both are the indices that the column writers cut with. `Colon()` is every row, and gives the axes unchanged, on a static panel too. A static panel has the axes `(assets,)` alone. It has no observation axis to cut, so it refuses any other `rows`. Without that check, `rows` would cut the asset axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. `Colon()`: give `ax` unchanged.
 2. A vector: check that `ax` carries an observation axis and that `rows` indexes it. Count the rows it names, which is its length for positions and its count of `true` for a mask, and give `(count, ax[2])`.

# Arguments

  - `ax`: The panel's axes, as [`panel_axes`](@ref) reads them.
  - $(arg_dict[:fdrows])

# Validation

  - `length(ax) == 2` when `rows` is not `Colon()`. Raises an `ArgumentError`.
  - Every position in `rows` lies in `1:ax[1]`, and a `Bool` mask has length `ax[1]`. Raises an `ArgumentError`.

# Returns

  - `ax::Tuple`: `(assets,)` for a static panel, and `(rows named, assets)` for a time-varying one.

# Related

  - [`feature_matrix`](@ref)
  - [`panel_axes`](@ref)
  - [`collapse_rows`](@ref)
"""
function stacked_axes(ax::Tuple, ::Colon)::Tuple
    return ax
end
function stacked_axes(ax::Tuple, rows::AbstractVector{<:Integer})::Tuple
    @argcheck(length(ax) == 2,
              ArgumentError("rows cuts the observation axis, and a static Asset Panel has none. Pass rows = Colon() on a static panel."))
    @argcheck(checkbounds(Bool, 1:ax[1], rows),
              ArgumentError("rows must index the observation axis 1:$(ax[1]): every position lies on it, and a Bool mask has one entry per observation. Got\nrows => $(rows)."))
    return (length(only(to_indices(1:ax[1], (rows,)))), ax[2])
end
"""
    feature_matrix(pnl::AssetPanel, sel = nothing; strict::Bool = false, rows = Colon(),
                   placeholder::Option{<:Real} = NaN) -> Array

Stack the Panel Fields a Feature Selector names into the Feature Matrix a distance measures.

No object keeps the result. The Asset Panel is the data and the Feature Matrix is one view of it, so the caller that measures a distance builds the matrix and drops it after.

A static panel gives an `assets × features` matrix, and a time-varying one an `observations × assets × features` array. A numeric Panel Field gives one column, a categorical Panel Field one `0`/`1` column per level, a tensor Panel Field one column per label, and an observed mask one `0`/`1` column. The order of `sel` is the column order.

A time-varying panel stacks every observation unless `rows` names the rows to stack. Then it stacks those rows alone, one row of the stack for each row that `rows` names. A consumer that reads one row stacks one row this way. A [`FeatureDistance`](@ref) under [`LastObservation`](@ref) passes the last row through [`collapse_rows`](@ref), so it reads a lifted static Panel Field, whose values are a [`RepeatedLeading`](@ref), once and does not copy it once per observation. The stack keeps its observation axis whatever `rows` holds. A one-row stack is therefore a window of one observation, and every collapse algorithm agrees on it. A static panel has no observation axis, so it takes `Colon()` alone.

A cell of a value column holds data where the raw input carried a value or a fill policy wrote one, and the Feature Matrix holds that value. A cell that holds the placeholder of a blank that no fill reached holds no data. The Feature Matrix holds `placeholder` in its place, `NaN` by default, so the answer does not change with the placeholder. `placeholder = nothing` keeps the stored placeholder. An observed-mask column shows where the raw input held a blank, filled or not.

# Mathematical definition

```math
\\begin{align}
Z_{s,\\,i,\\,c} &= \\begin{cases}
u & c \\text{ is a value column of a Panel Field that holds a placeholder, and } d^{(k,l)}_{r_{s},\\,i} = 0 \\\\
v^{(k)}_{r_{s},\\,i} & c \\text{ is the value column of a numeric Panel Field} \\\\
\\mathbf{1}\\left[g^{(k)}_{r_{s},\\,i} = l\\right] & c \\text{ is level } l \\text{ of a categorical Panel Field} \\\\
V^{(k)}_{r_{s},\\,i,\\,l} & c \\text{ is label } l \\text{ of a tensor Panel Field} \\\\
m^{(k)}_{r_{s},\\,i} & c \\text{ is the observed mask of a numeric or a categorical Panel Field} \\\\
\\prod_{j} m^{(k)}_{r_{s},\\,i,\\,j} & c \\text{ is the observed mask of a tensor Panel Field}
\\end{cases}\\,, \\\\
& s = 1, \\ldots, R\\,, \\quad i = 1, \\ldots, N\\,, \\quad c = 1, \\ldots, C\\,.
\\end{align}
```

A static panel has no observation index, so ``s`` and ``r_{s}`` drop out and ``\\mathbf{Z}`` is ``N \\times C``. A Panel Field that cannot blank has ``m^{(k)} = 1`` in every cell. Each code lies on the levels of its Panel Field, so the level columns of one categorical Panel Field sum to one in every row.

Where:

  - ``Z_{s,\\,i,\\,c}``: Entry of the Feature Matrix ``\\mathbf{Z}`` at stacked row ``s``, asset ``i`` and column ``c``.
  - ``r_{s}``: The observation that stacked row ``s`` reads.
  - ``R``: Number of observations the stack reads.
  - ``k``: The Panel Field that column ``c`` reads.
  - ``l``: Position of the level or the label that column ``c`` reads.
  - ``v^{(k)}_{t,\\,i}``: Value of a numeric Panel Field at observation ``t`` and asset ``i``.
  - ``g^{(k)}_{t,\\,i}``: Level position of a categorical Panel Field at observation ``t`` and asset ``i``.
  - ``V^{(k)}_{t,\\,i,\\,l}``: Value of a tensor Panel Field at observation ``t``, asset ``i`` and label ``l``.
  - ``m^{(k)}_{t,\\,i}``, ``m^{(k)}_{t,\\,i,\\,j}``: Observed mask of a Panel Field, ``1`` where the cell was observed. A tensor Panel Field's mask carries the label index ``j``.
  - ``d^{(k,l)}_{t,\\,i}``: Data mask of the value column that ``c`` reads, ``0`` where the cell holds a placeholder, see [`panel_field_column_mask`](@ref). The first case applies first, and it does not apply to a Panel Field that holds no placeholder.
  - ``u``: The value of `placeholder`. With `placeholder = nothing`, the first case does not apply.
  - ``\\mathbf{1}[\\cdot]``: Indicator function, ``1`` when its condition holds and ``0`` otherwise.
  - ``C``: Number of columns the Feature Selector resolves to.
  - $(math_dict[:N])

# Algorithm

 1. Resolve `sel` against the panel with [`select_fields`](@ref).
 2. Derive the element type as the promotion over the Panel Fields whose **value** columns `sel` resolves to. An observed-mask column is a `0`/`1` column that every type holds, so it contributes nothing, and an indicator contributes nothing either. A selection of mask and indicator columns alone stacks in the panel's own type, the promotion over the values of every Panel Field. So the one-hot block of a `Float32` panel is `Float32`. A panel with no numeric or tensor Panel Field stacks in `Float64`. See [`panel_value_eltype`](@ref).
 3. When a value column reads a Panel Field that holds a placeholder, widen the type to hold `placeholder` through [`panel_read_eltype`](@ref). An integer panel with the default `NaN` stacks in a floating-point type.
 4. Allocate the matrix as zeros, over the observation rows that `rows` names, the asset axis of the panel and the resolved column count. See [`stacked_axes`](@ref).
 5. Write each column, cut to `rows`, with [`panel_field_value_column!`](@ref) or [`panel_field_observed_column!`](@ref). Then write `placeholder` into each cell of a value column whose mask from [`panel_field_column_mask`](@ref) is `false`, through [`panel_read_write!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - $(field_dict[:fdsel])
  - $(field_dict[:fdstrict])
  - $(field_dict[:fdrows])
  - $(arg_dict[:fdplc])

# Validation

  - The panel holds a Panel Field, and `sel` resolves to at least one column. See [`select_fields`](@ref).
  - `placeholder` converts to the element type of the matrix. A `Rational` panel that holds a placeholder refuses `NaN` with an `InexactError`.
  - `rows` is `Colon()` on a static panel, and indexes the observation axis of a time-varying one. See [`stacked_axes`](@ref).

# Returns

  - `Z::Array`: The Feature Matrix, in the type derived over the Panel Fields it stacks.

# Related

  - [`AssetPanel`](@ref)
  - [`feature_labels`](@ref)
  - [`select_fields`](@ref)
  - [`stacked_axes`](@ref)
  - [`panel_value_eltype`](@ref)
  - [`panel_field_column_mask`](@ref)
  - [`panel_field_values`](@ref)
  - [`FeatureDistance`](@ref)
  - [`collapse_rows`](@ref)
"""
function feature_matrix(pnl::AssetPanel, sel = nothing; strict::Bool = false,
                        rows::Union{Colon, AbstractVector{<:Integer}} = Colon(),
                        placeholder::Option{<:Real} = NaN)
    return feature_stack(pnl, select_fields(pnl, sel, strict); rows = rows,
                         placeholder = placeholder)
end
"""
    feature_stack(pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}}; rows = Colon(),
                  placeholder::Option{<:Real} = NaN) -> Array

Stack the resolved columns `cols` of an Asset Panel into a Feature Matrix. It is steps 2 to 5 of [`feature_matrix`](@ref), which states them, for a caller that resolved the Feature Selector already. [`FeatureDistance`](@ref) resolves it once and reads the same columns for its mask of the cells that hold data.

# Arguments

  - `pnl`: The Asset Panel.
  - `cols`: The resolved columns, from [`select_fields`](@ref).
  - $(arg_dict[:fdrows])
  - $(arg_dict[:fdplc])

# Returns

  - `Z::Array`: The Feature Matrix.

# Related

  - [`feature_matrix`](@ref)
  - [`feature_data_cells`](@ref)
"""
function feature_stack(pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}};
                       rows::Union{Colon, AbstractVector{<:Integer}} = Colon(),
                       placeholder::Option{<:Real} = NaN)
    Z = zeros(feature_stack_eltype(pnl, cols, placeholder),
              stacked_axes(panel_axes(pnl), rows)..., length(cols))
    for (c, col) in pairs(cols)
        k, l, part = col
        zc = selectdim(Z, ndims(Z), c)
        if part === :observed
            panel_field_observed_column!(zc, pnl.pf[k], rows)
        else
            panel_field_value_column!(zc, pnl.pf[k], l, rows)
            panel_read_write!(zc, panel_field_column_mask(pnl.pf[k], l, rows), placeholder)
        end
    end
    return Z
end
"""
    feature_stack_eltype(pnl::AssetPanel, cols::AbstractVector{Tuple{Int, Int, Symbol}}, placeholder::Option{<:Real}) -> Type

Derive the element type of the Feature Matrix that [`feature_stack`](@ref) stacks from the resolved columns `cols`. It is steps 2 and 3 of [`feature_matrix`](@ref), which states the rule.

# Algorithm

 1. Promote over the Panel Fields whose value columns `cols` holds. An observed-mask column and an indicator contribute nothing, see [`panel_value_eltype`](@ref).
 2. With nothing to promote, take the promotion over every Panel Field of the panel.
 3. When a value column reads a Panel Field that holds a placeholder, widen the type to hold `placeholder` through [`panel_read_eltype`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `cols`: The resolved columns, from [`select_fields`](@ref).
  - $(arg_dict[:fdplc])

# Returns

  - `T::Type`: The element type of the Feature Matrix.

# Related

  - [`feature_stack`](@ref)
  - [`panel_value_eltype`](@ref)
  - [`panel_read_eltype`](@ref)
"""
function feature_stack_eltype(pnl::AssetPanel,
                              cols::AbstractVector{Tuple{Int, Int, Symbol}},
                              placeholder::Option{<:Real})
    vcols = filter(col -> col[3] !== :observed, cols)
    T = mapreduce(col -> panel_value_eltype(pnl.pf[col[1]]), promote_type, vcols;
                  init = Union{})
    T = T === Union{} ? panel_value_eltype(pnl.pf) : T
    blank = any(col -> !isnothing(pnl.pf[col[1]].pmsk), vcols)
    return panel_read_eltype(T, blank ? placeholder : nothing)
end
"""
    feature_labels(pnl::AssetPanel, sel = nothing; strict::Bool = false) -> Vector

Name the columns [`feature_matrix`](@ref) stacks, one Feature Selector entry per column.

A label is the entry that selects its own column and no other. The returned vector is therefore a Feature Selector, and a stack of the panel against it rebuilds the same matrix. So a caller can ask what a distance measured, and no object has to keep the matrix.

The distance kernel never calls this function. It reads the matrix alone, so a path that reads no label allocates none.

# Algorithm

 1. Resolve `sel` against the panel with [`select_fields`](@ref).
 2. Name each resolved column with [`panel_column_label`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - $(field_dict[:fdsel])
  - $(field_dict[:fdstrict])

# Returns

  - `labels::Vector`: One Feature Selector entry per column of [`feature_matrix`](@ref), in column order.

# Related

  - [`AssetPanel`](@ref)
  - [`feature_matrix`](@ref)
  - [`select_fields`](@ref)
  - [`panel_column_label`](@ref)
"""
function feature_labels(pnl::AssetPanel, sel = nothing; strict::Bool = false)
    return [panel_column_label(pnl, col) for col in select_fields(pnl, sel, strict)]
end
export LabelGroup, feature_matrix, feature_labels
