"""
    panel_field_keys(f::NumericPanelField) -> Vector{String}
    panel_field_keys(f::CategoricalPanelField) -> VecStr
    panel_field_keys(f::TensorPanelField) -> VecStr

Return the namespace a Feature Selector entry pairs with a Panel Field's name.

A numeric Panel Field contributes one column and has no second namespace, so it answers with an empty vector: every key paired with it is absent, and `strict` decides what that means. A categorical Panel Field answers with its levels and a tensor Panel Field with its labels, and in both the position of a key is the position of the column it selects.

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

This is what a bare field name expands to, and what an absent selector expands to for every Panel Field of the panel. The mask is not among them: a bare name is the values alone.

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
    panel_selector_msg(f::AbstractPanelField, key, keys::VecStr) -> String

Build the diagnostic a Feature Selector entry that resolves against nothing carries.

The two messages name the two namespaces an entry resolves in. The first is the panel's Panel Field names, the second one Panel Field's levels or labels. Each hands its own suggestion pool to [`did_you_mean`](@ref), so a misspelling is answered against the names it could have meant and not against every column of the panel.

# Algorithm

 1. Name the entry that resolved against nothing, and the namespace it was resolved in.
 2. Append the nearest match from the pool with [`did_you_mean`](@ref).

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
    assert_selector_entry(entry) -> nothing

Check that one Feature Selector entry takes one of the four forms the grammar admits.

The four forms are a Panel Field name, a name paired with the levels or labels it keeps, a name paired with one level or label, and a name paired with `:observed`. No entry is a column position: after the Asset Panel became the one carrier every field, level and label carries a name, so an integer has nothing to index.

A name is refused when it is empty, and a paired vector is refused when it is empty, holds an empty key, or repeats a key. Each of those resolves to no column or to a doubled column, which is the failure the selector exists to remove.

# Algorithm

 1. A string: check that it is not empty.
 2. A pair whose first element is a non-empty string: check its second element. A `Symbol` is `:observed`, a string is non-empty, and a vector of strings is non-empty, holds no empty key and repeats none.
 3. Anything else is refused.

# Arguments

  - `entry`: One entry of a Feature Selector.

# Validation

  - `entry` takes one of the four forms. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`assert_feature_selector`](@ref)
  - [`select_fields`](@ref)
"""
function assert_selector_entry(entry)::Nothing
    ok = if isa(entry, AbstractString)
        !isempty(entry)
    elseif isa(entry, Pair) && isa(first(entry), AbstractString) && !isempty(first(entry))
        v = last(entry)
        if isa(v, Symbol)
            v === :observed
        elseif isa(v, AbstractString)
            !isempty(v)
        elseif isa(v, AbstractVector{<:AbstractString})
            !isempty(v) && all(!isempty, v) && allunique(v)
        else
            false
        end
    else
        false
    end
    @argcheck(ok,
              ArgumentError("a `sel` entry is a Panel Field name, a name paired with the levels or labels it keeps, a name paired with one level or label, or a name paired with `:observed`. A name is non-empty, and a paired vector is non-empty, holds no empty key and repeats none. Got\nentry => $(entry)"))
    return nothing
end
"""
    assert_feature_selector(sel::Nothing) -> nothing
    assert_feature_selector(sel::AbstractVector) -> nothing

Validate a Feature Selector: a non-empty vector of distinct entries, each of the four admitted forms.

An empty `sel` is refused rather than read as "every Panel Field": `nothing` already says that, and a selection that silently widens to the whole panel is the failure this selector exists to remove. A repeated entry is refused because it doubles that column's contribution to every distance.

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
 6. A name paired with a vector of keys claims one column per key, in the order the vector writes them.

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

This is the one resolution. [`feature_matrix`](@ref) and [`feature_labels`](@ref) both read it, so the matrix and its labels cannot disagree about what was selected. It is unexported: a caller reads the labels, not the positions.

A resolved column is the triple `(k, l, part)`. `k` is the Panel Field's position in the panel. `part` is `:vals` or `:observed`. `l` is the position of the level or the label within the Panel Field, and `0` where the Panel Field contributes one column of that part: a numeric Panel Field's value column, and every Panel Field's observed-mask column.

The order of `sel` is the column order, so a caller decides it. A `nothing` selector takes the panel's own order, and stacks each Panel Field's values without its mask.

# Algorithm

 1. Validate `sel` with [`assert_feature_selector`](@ref).
 2. `sel` is `nothing`: push every Panel Field's value columns in panel order, and return.
 3. Otherwise resolve each entry with [`select_fields_push!`](@ref), in the order `sel` writes them.
 4. Check that no two entries resolved to one column.
 5. Check that the selection is not empty, which happens when every entry resolved against nothing and was dropped.

# Arguments

  - `pnl`: The Asset Panel.
  - $(field_dict[:fdsel])
  - $(field_dict[:fdstrict])

# Validation

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
              ArgumentError("two entries of `sel` resolve to one column of the Feature Matrix, which doubles that column's contribution to every distance. A bare Panel Field name already claims every one of its value columns, so pairing the same name with one of its levels or labels repeats a column. Got\nsel => $(sel)"))
    @argcheck(!isempty(cols),
              IsEmptyError("`sel` selected no column of the Asset Panel: every entry resolved against nothing and was dropped. Set `strict = true` to see which entry, or correct `sel`."))
    return cols
end
"""
    panel_column_label(pnl::AssetPanel, col::Tuple{Int, Int, Symbol})

Return the Feature Selector entry that selects exactly one resolved column.

A label is a selector entry, not a rendered string. So the labels of a Feature Matrix are themselves a Feature Selector, and stacking the panel against them rebuilds the same matrix column for column.

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
    panel_field_value_column!(zc::AbstractArray, f::NumericPanelField, l::Integer) -> nothing
    panel_field_value_column!(zc::AbstractArray, f::CategoricalPanelField, l::Integer) -> nothing
    panel_field_value_column!(zc::AbstractArray, f::TensorPanelField, l::Integer) -> nothing

Write one Panel Field's value column into a Feature Matrix under construction.

# Algorithm

The method that Julia selects is the algorithm. A numeric Panel Field writes its values, a categorical Panel Field writes the `0`/`1` indicator of one level, and a tensor Panel Field writes one label slice of its values.

# Arguments

  - `zc`: The column of the Feature Matrix, a view over the panel's observation and asset axes.
  - `f`: The Panel Field.
  - `l`: The position of the level or the label, and `0` for a numeric Panel Field.

# Returns

  - `nothing`. `zc` carries the result.

# Related

  - [`feature_matrix`](@ref)
  - [`select_fields`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_field_value_column!(zc::AbstractArray, f::NumericPanelField,
                                   ::Integer)::Nothing
    zc .= f.vals
    return nothing
end
function panel_field_value_column!(zc::AbstractArray, f::CategoricalPanelField,
                                   l::Integer)::Nothing
    zc .= f.codes .== l
    return nothing
end
function panel_field_value_column!(zc::AbstractArray, f::TensorPanelField,
                                   l::Integer)::Nothing
    zc .= selectdim(f.vals, ndims(f.vals), l)
    return nothing
end
"""
    panel_field_observed_column!(zc::AbstractArray, f::NumericPanelField) -> nothing
    panel_field_observed_column!(zc::AbstractArray, f::CategoricalPanelField) -> nothing
    panel_field_observed_column!(zc::AbstractArray, f::TensorPanelField) -> nothing

Write one Panel Field's observed mask into a Feature Matrix under construction, as a `0`/`1` column.

A Panel Field contributes **one** mask column, whatever its kind and however many value columns it contributes. The column says whether the cell that Panel Field describes was observed for that asset, so a tensor Panel Field, whose mask carries a label axis of its own, holds where every label of that asset was observed.

A Panel Field that carries no mask gives a column of ones: `omsk === nothing` means that the Panel Field cannot blank, so every cell was observed.

# Algorithm

The method that Julia selects is the algorithm.

 1. Write ones when the Panel Field carries no mask.
 2. A numeric or a categorical Panel Field writes its mask, which already has the column's shape.
 3. A tensor Panel Field reduces its mask over the label axis with `all`, and writes that.

# Arguments

  - `zc`: The column of the Feature Matrix, a view over the panel's observation and asset axes.
  - `f`: The Panel Field.

# Returns

  - `nothing`. `zc` carries the result.

# Related

  - [`feature_matrix`](@ref)
  - [`select_fields`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_field_observed_column!(zc::AbstractArray, f::NumericPanelField)::Nothing
    zc .= isnothing(f.omsk) ? true : f.omsk
    return nothing
end
function panel_field_observed_column!(zc::AbstractArray, f::CategoricalPanelField)::Nothing
    zc .= isnothing(f.omsk) ? true : f.omsk
    return nothing
end
function panel_field_observed_column!(zc::AbstractArray, f::TensorPanelField)::Nothing
    omsk = f.omsk
    if isnothing(omsk)
        zc .= true
    else
        d = ndims(omsk)
        zc .= dropdims(all(omsk; dims = d); dims = d)
    end
    return nothing
end
"""
    feature_matrix(pnl::AssetPanel, sel = nothing; strict::Bool = false) -> Array

Stack the Panel Fields a Feature Selector names into the Feature Matrix a distance measures.

Nothing stores the result. The Asset Panel is the data, and the Feature Matrix is one view of it, so it is built where it is measured and thrown away after.

A static panel gives an `assets × features` matrix, and a time-varying one an `observations × assets × features` array. A numeric Panel Field gives one column, a categorical Panel Field one `0`/`1` column per level, a tensor Panel Field one column per label, and an observed mask one `0`/`1` column. The order of `sel` is the column order.

# Algorithm

 1. Resolve `sel` against the panel with [`select_fields`](@ref).
 2. Derive the element type, as the promotion over the Panel Fields whose **value** columns were resolved. An observed-mask column is a `0`/`1` column that every type carries, so it contributes nothing, and neither does an indicator. A selection of mask columns alone stacks in `Float64`. See [`panel_value_eltype`](@ref).
 3. Allocate the matrix as zeros, over the panel's own observation and asset axes and the resolved column count.
 4. Write each column, with [`panel_field_value_column!`](@ref) or [`panel_field_observed_column!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - $(field_dict[:fdsel])
  - $(field_dict[:fdstrict])

# Validation

  - `sel` resolves to at least one column. See [`select_fields`](@ref).

# Returns

  - `Z::Array`: The Feature Matrix, in the type derived over the Panel Fields it stacks.

# Related

  - [`AssetPanel`](@ref)
  - [`feature_labels`](@ref)
  - [`select_fields`](@ref)
  - [`panel_value_eltype`](@ref)
  - [`FeatureDistance`](@ref)
"""
function feature_matrix(pnl::AssetPanel, sel = nothing; strict::Bool = false)
    cols = select_fields(pnl, sel, strict)
    T = mapreduce(promote_type, cols; init = Union{}) do col
        return col[3] === :observed ? Union{} : panel_value_eltype(pnl.pf[col[1]])
    end
    Z = zeros(T === Union{} ? Float64 : T, panel_field_axes(pnl.pf[1])..., length(cols))
    for (c, col) in pairs(cols)
        k, l, part = col
        zc = selectdim(Z, ndims(Z), c)
        if part === :observed
            panel_field_observed_column!(zc, pnl.pf[k])
        else
            panel_field_value_column!(zc, pnl.pf[k], l)
        end
    end
    return Z
end
"""
    feature_labels(pnl::AssetPanel, sel = nothing; strict::Bool = false) -> Vector

Name the columns [`feature_matrix`](@ref) stacks, one Feature Selector entry per column.

A label is the entry that selects exactly its column, so the returned vector is itself a Feature Selector and stacking the panel against it rebuilds the same matrix. That is what lets a caller ask what a distance measured without the matrix being stored anywhere.

The kernel never calls this: it reads the matrix alone, so no label is allocated on a path that does not read one.

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
export feature_matrix, feature_labels
