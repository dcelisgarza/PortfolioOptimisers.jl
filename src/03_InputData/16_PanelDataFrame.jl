"""
    panel_frame_columns(f::NumericPanelField, decode::Bool)
    panel_frame_columns(f::CategoricalPanelField, decode::Bool)
    panel_frame_columns(f::TensorPanelField, decode::Bool)

Return the table columns one Panel Field contributes to a [`panel_dataframe`](@ref), in column order.

A column is the triple `(name, vals, omsk)`. `name` is the column name, `vals` is the slab of values over the axes of the panel, and `omsk` is the slab of the observed mask, or `nothing` when the Panel Field carries no observed mask. Every slab is `assets` on a static panel and `observations × assets` on a time-varying one, so [`panel_array_view`](@ref) slices each of them by asset.

A table and a Feature Matrix give a categorical Panel Field a different number of columns. A Feature Matrix holds numbers, so it gives the Panel Field one indicator column per level. A table column can hold a string, so it gives the Panel Field one column that holds the level. A tensor Panel Field gets the same columns in both, one column per trailing-axis label under the `"<field>=<label>"` name that [`panel_field_labels`](@ref) gives it. A table has two axes, and the labels are a third axis, so each label becomes a column.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref): one column with the name of the Panel Field, which holds its values.
 2. [`CategoricalPanelField`](@ref): one column with the name of the Panel Field. It holds the level that each code names when `decode` is `true`, and the codes when `decode` is `false`.
 3. [`TensorPanelField`](@ref): one column per trailing-axis label, named `"<field>=<label>"`, which holds the slice of that label.

# Arguments

  - `f`: The Panel Field.
  - `decode::Bool`: Whether a categorical Panel Field gives its levels in place of its codes. A numeric and a tensor Panel Field carry no codes, so these methods do not read it.

# Returns

  - `cols::Vector{Tuple{String, Any, Any}}`: One `(name, vals, omsk)` triple per column, in column order. The value slab of a numeric or a tensor Panel Field is the array of the Panel Field or a view of it, not a copy.

# Related

  - [`panel_dataframe`](@ref)
  - [`AbstractPanelField`](@ref)
  - [`panel_field_labels`](@ref)
  - [`panel_array_view`](@ref)
"""
function panel_frame_columns(f::NumericPanelField, ::Bool)
    return Tuple{String, Any, Any}[(String(f.name), f.vals, f.omsk)]
end
function panel_frame_columns(f::CategoricalPanelField, decode::Bool)
    vals = decode ? [f.levels[c] for c in f.codes] : f.codes
    return Tuple{String, Any, Any}[(String(f.name), vals, f.omsk)]
end
function panel_frame_columns(f::TensorPanelField, ::Bool)
    d = ndims(f.vals)
    omsk = f.omsk
    return Tuple{String, Any, Any}[("$(f.name)=$(f.labels[l])", selectdim(f.vals, d, l),
                                    isnothing(omsk) ? nothing : selectdim(omsk, d, l))
                                   for l in eachindex(f.labels)]
end
"""
    panel_frame_fields(pnl::AssetPanel, fields::Nothing)
    panel_frame_fields(pnl::AssetPanel, fields)

Find the Panel Fields that a [`panel_dataframe`](@ref) call names, in the order of the call.

# Algorithm

The method that Julia selects is the algorithm.

 1. `fields` is `nothing`: return the Panel Fields of the panel, in panel order.
 2. Otherwise: [`panel_field`](@ref) finds one Panel Field per entry of `fields`. Return them in the order of `fields`.

# Arguments

  - `pnl`: The Asset Panel.
  - `fields`: The names of the Panel Fields to include, or `nothing` for all Panel Fields. An entry can be a `String` or a `Symbol`.

# Validation

  - The panel holds a Panel Field with each name. Raises a `KeyError`. See [`panel_field`](@ref).

# Returns

  - `fs`: The Panel Fields, in column order. With `fields = nothing` this is `pnl.pf` itself, not a copy.

# Related

  - [`panel_dataframe`](@ref)
  - [`AssetPanel`](@ref)
  - [`panel_field`](@ref)
"""
function panel_frame_fields(pnl::AssetPanel, ::Nothing)
    return pnl.pf
end
function panel_frame_fields(pnl::AssetPanel, fields)
    return [panel_field(pnl, String(name)) for name in fields]
end
"""
    panel_frame_assets(nx::VecStr, assets::Nothing) -> Vector{Int}
    panel_frame_assets(nx::VecStr, assets::AbstractString) -> Vector{Int}
    panel_frame_assets(nx::VecStr, assets) -> Vector{Int}

Convert the asset labels that a [`panel_dataframe`](@ref) call names into positions on the asset axis of the panel.

The panel holds no asset names. Its Panel Fields share an asset axis, and the carrier that holds the panel names that axis. So the caller gives the names in `nx`, and this function finds the position of each label in `nx`. The functions that write the table slice every slab at these positions.

# Algorithm

The method that Julia selects is the algorithm.

 1. `assets` is `nothing`: return every position, in axis order.
 2. `assets` is one string: return the position of that label, as a one-entry vector.
 3. Otherwise: find the first position of each label in `nx`, in the order of `assets`, and check that no position repeats.

# Arguments

  - `nx`: The asset names of the universe of the panel. See [`VecStr`](@ref).
  - `assets`: The asset labels to include, or `nothing` for all assets. An entry can be a `String` or a `Symbol`.

# Validation

  - `nx` holds every entry of `assets`. Raises a `KeyError`.
  - No entry of `assets` repeats. Raises an `ArgumentError`.

# Returns

  - `j::Vector{Int}`: The selected positions on the asset axis, in column order.

# Related

  - [`panel_dataframe`](@ref)
  - [`did_you_mean`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_assets(nx::VecStr, ::Nothing)
    return collect(1:length(nx))
end
function panel_frame_assets(nx::VecStr, assets::AbstractString)
    return panel_frame_assets(nx, [assets])
end
function panel_frame_assets(nx::VecStr, assets)
    j = Int[]
    for a in assets
        k = findfirst(==(String(a)), nx)
        @argcheck(!isnothing(k),
                  KeyError("the panel's universe holds no asset named `$a`$(did_you_mean(String(a), nx)). It holds $(length(nx)): $(join(nx, ", "))"))
        push!(j, something(k))
    end
    @argcheck(allunique(j),
              ArgumentError("`assets` must not repeat a label, because a repeated asset doubles that asset's rows in a long table and collides on its column name in a wide one. Got\nassets => $(assets)"))
    return j
end
"""
    panel_frame_column!(df::DataFrames.DataFrame, name::AbstractString, v::AbstractVector) -> nothing

Add one column to a table that [`panel_dataframe`](@ref) writes, and refuse a column name that the table already holds.

A Panel Field name, a trailing-axis label and an asset name are free text, so two columns can get the same name. For example, a Panel Field named `"asset"` gets the name of the asset key of the long layout, and a numeric Panel Field named `"beta=size"` gets the name of the `"size"` column of a tensor Panel Field `"beta"`. `df[!, name] = v` replaces a column with the same name and raises no error, so the table loses a column. This function raises an error in its place.

The table gets a copy of `v`. A slab can be the array of a Panel Field or a view of it, so without the copy a change to a cell of the table changes the panel.

# Algorithm

 1. Check that `df` holds no column named `name`.
 2. Add a copy of `v` to `df` as its last column, with the name `name`.

# Arguments

  - `df`: The table that the caller writes. This function adds the column to it.
  - `name`: The column name.
  - `v`: The column values.

# Validation

  - `df` holds no column named `name`. Raises an `ArgumentError`.

# Returns

  - `nothing`. `df` holds the new column.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_block!`](@ref)
  - [`panel_frame_field`](@ref)
  - [`panel_frame_long`](@ref)
  - [`panel_frame_wide`](@ref)
"""
function panel_frame_column!(df::DataFrames.DataFrame, name::AbstractString,
                             v::AbstractVector)::Nothing
    @argcheck(iszero(DataFrames.columnindex(df, name)),
              ArgumentError("the table already holds a column named \"$name\", and a second column with this name replaces the first. A Panel Field name, a trailing-axis label or an asset name gives this name to two columns. Rename the Panel Field or the asset, or leave one of them out with `fields` or `assets`."))
    df[:, name] = v
    return nothing
end
"""
    panel_frame_block!(df::DataFrames.DataFrame, A::AbstractVector, names::VecStr) -> nothing
    panel_frame_block!(df::DataFrames.DataFrame, A::AbstractMatrix, names::VecStr) -> nothing

Write one slab of a Panel Field, or one universe mask, into a table as one column per asset.

The single-field shape and the wide layout write their columns through this function.

# Algorithm

The method that Julia selects is the algorithm. A slab is a vector on a static panel and a matrix on a time-varying one.

 1. A vector slab has no observation axis. The value of each asset becomes a column with one row.
 2. A matrix slab has an observation axis. The column of each asset in the slab becomes a column of the table.

Each column goes through [`panel_frame_column!`](@ref).

# Arguments

  - `df`: The table that the caller writes. This function adds the columns to it.
  - `A`: The slab, sliced to the selected assets.
  - `names`: The column name of each selected asset, in the column order of the slab. See [`VecStr`](@ref).

# Validation

  - `df` holds no column with a name in `names`. Raises an `ArgumentError`. See [`panel_frame_column!`](@ref).

# Returns

  - `nothing`. `df` holds the new columns.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_column!`](@ref)
  - [`panel_frame_wide`](@ref)
  - [`panel_frame_field`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_block!(df::DataFrames.DataFrame, A::AbstractVector,
                            names::VecStr)::Nothing
    for (c, name) in pairs(names)
        panel_frame_column!(df, name, view(A, c:c))
    end
    return nothing
end
function panel_frame_block!(df::DataFrames.DataFrame, A::AbstractMatrix,
                            names::VecStr)::Nothing
    for (c, name) in pairs(names)
        panel_frame_column!(df, name, view(A, :, c))
    end
    return nothing
end
"""
    panel_frame_field(pnl::AssetPanel, f::AbstractPanelField, j::VecInt, nxj::VecStr, ts, decode::Bool) -> DataFrames.DataFrame
    panel_frame_field(pnl::AssetPanel, f::TensorPanelField, j::VecInt, nxj::VecStr, ts, decode::Bool)

Write one Panel Field as a table with one row per observation and one column per asset.

A [`panel_dataframe`](@ref) call that names one Panel Field returns this table. The values of a numeric or a categorical Panel Field are `observations × assets`, which is the shape of the table, so the table needs no asset key and no layout. It holds no universe mask and no observed mask. A static panel has no observation axis, so its table has one row.

This function refuses a [`TensorPanelField`](@ref). Its values are `observations × assets × labels`, and a table with one column per asset has no place for the labels. The long and the wide layouts hold it, with one column per label.

# Algorithm

The method that Julia selects refuses a tensor Panel Field. The other method runs these steps.

 1. Add the `"observation"` column with the labels `ts`, unless the panel is static.
 2. Add one column per selected asset with the values of `f`, through [`panel_frame_block!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `f`: The Panel Field.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order of `j`. See [`VecStr`](@ref).
  - `ts`: The observation labels. A static panel does not read it.
  - `decode::Bool`: Whether a categorical Panel Field gives its levels in place of its codes.

# Validation

  - `f` is not a [`TensorPanelField`](@ref). Raises an `ArgumentError`.
  - No asset is named `"observation"` when the panel is time-varying. Raises an `ArgumentError`. See [`panel_frame_column!`](@ref).

# Returns

  - `df::DataFrames.DataFrame`: The table, with the `"observation"` column first when the panel is time-varying. The table holds copies, so a change to the table does not change the panel.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_block!`](@ref)
  - [`panel_frame_column!`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_field(pnl::AssetPanel, f::AbstractPanelField, j::VecInt, nxj::VecStr,
                           ts, decode::Bool)
    df = DataFrames.DataFrame()
    if !panel_is_static(pnl)
        panel_frame_column!(df, "observation", ts)
    end
    _, vals, _ = first(panel_frame_columns(f, decode))
    panel_frame_block!(df, panel_array_view(vals, :, j), nxj)
    return df
end
function panel_frame_field(::AssetPanel, f::TensorPanelField, ::VecInt, ::VecStr, ::Any,
                           ::Bool)
    return throw(ArgumentError("the tensor Panel Field \"$(f.name)\" carries a $(f.axis) axis of $(length(f.labels)) label(s) beside its assets, so it has no observations × assets table of its own. Ask for a layout instead, which spreads the $(f.axis) axis into one column per label: `panel_dataframe(pnl; fields = [\"$(f.name)\"])`."))
end
"""
    panel_frame_long(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool) -> DataFrames.DataFrame

Write the selected Panel Fields in the long layout, with one row per pair of observation and asset and one column per Panel Field column.

On a time-varying panel the table keeps only the rows where the active mask is `true`, so every row is an asset in the universe at that observation. The estimation mask is the `"emsk"` column, because it can be `false` on a row that the active mask keeps. A static panel has no mask, so its table has no `"emsk"` column and has one row per asset.

# Algorithm

 1. Add the key columns. A time-varying panel gets `"observation"` and `"asset"`, where row `(t - 1) n + k` holds observation `t` and asset `k` of the `n` selected assets. A static panel gets `"asset"` alone.
 2. For each Panel Field in order, and each of its columns from [`panel_frame_columns`](@ref), add the slab in row-major order, so that each value is on the row of its observation and asset. When the Panel Field carries an observed mask, add the mask of that column next, as `"<column>::observed"`.
 3. On a time-varying panel, add the `"emsk"` column, and keep the rows where the active mask is `true`.

Each column goes through [`panel_frame_column!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `fs`: The Panel Fields, in column order.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order of `j`. See [`VecStr`](@ref).
  - `ts`: The observation labels. A static panel does not read it.
  - `decode::Bool`: Whether a categorical Panel Field gives its levels in place of its codes.

# Validation

  - No two columns get the same name. A Panel Field named `"observation"`, `"asset"` or `"emsk"` gets the name of a key or mask column. Raises an `ArgumentError`. See [`panel_frame_column!`](@ref).

# Returns

  - `df::DataFrames.DataFrame`: The long table. It holds copies, so a change to the table does not change the panel.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_column!`](@ref)
  - [`panel_frame_wide`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_long(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool)
    static = panel_is_static(pnl)
    df = DataFrames.DataFrame()
    if static
        panel_frame_column!(df, "asset", nxj)
    else
        panel_frame_column!(df, "observation", repeat(collect(ts); inner = length(j)))
        panel_frame_column!(df, "asset", repeat(collect(nxj); outer = length(ts)))
    end
    for f in fs
        for (name, vals, omsk) in panel_frame_columns(f, decode)
            panel_frame_column!(df, name, vec(permutedims(panel_array_view(vals, :, j))))
            if !isnothing(omsk)
                panel_frame_column!(df, "$name::observed",
                                    vec(permutedims(panel_array_view(omsk, :, j))))
            end
        end
    end
    if static
        return df
    end
    panel_frame_column!(df, "emsk", vec(permutedims(view(pnl.emsk, :, j))))
    return df[vec(permutedims(view(pnl.amsk, :, j))), :]
end
"""
    panel_frame_wide(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool) -> DataFrames.DataFrame

Write the selected Panel Fields in the wide layout, with one row per observation and one column per pair of Panel Field column and asset.

A column name is the Panel Field column, an `"@"`, and the asset, for example `"mcap@AAPL"`, `"beta=size@AAPL"` and `"mcap::observed@AAPL"`. The table keeps every cell of the selected assets, and on a time-varying panel it adds both universe masks as the `"amsk@<asset>"` and `"emsk@<asset>"` columns. So the wide layout drops no data, and the long layout drops the rows outside the universe.

To get the Panel Field column and the asset of a cell, read the long layout, which holds them in two key columns. Do not split a wide column name at `"@"`, because a Panel Field name or an asset name can contain `"@"`.

# Algorithm

 1. Add the `"observation"` column, unless the panel is static. A static panel has one row.
 2. For each Panel Field in order, and each of its columns from [`panel_frame_columns`](@ref), add one column per selected asset through [`panel_frame_block!`](@ref). When the Panel Field carries an observed mask, add the mask of that column next, one column per asset.
 3. On a time-varying panel, add the active mask and then the estimation mask, one column per asset each.

# Arguments

  - `pnl`: The Asset Panel.
  - `fs`: The Panel Fields, in column order.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order of `j`. See [`VecStr`](@ref).
  - `ts`: The observation labels. A static panel does not read it.
  - `decode::Bool`: Whether a categorical Panel Field gives its levels in place of its codes.

# Validation

  - No two columns get the same name. A Panel Field named `"amsk"` or `"emsk"` gets the names of a mask column. Raises an `ArgumentError`. See [`panel_frame_column!`](@ref).

# Returns

  - `df::DataFrames.DataFrame`: The wide table. It holds copies, so a change to the table does not change the panel.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_block!`](@ref)
  - [`panel_frame_column!`](@ref)
  - [`panel_frame_long`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_wide(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool)
    static = panel_is_static(pnl)
    df = DataFrames.DataFrame()
    if !static
        panel_frame_column!(df, "observation", ts)
    end
    cols = Vector{String}(undef, length(nxj))
    for f in fs
        for (name, vals, omsk) in panel_frame_columns(f, decode)
            panel_frame_block!(df, panel_array_view(vals, :, j),
                               map!(a -> "$name@$a", cols, nxj))
            if !isnothing(omsk)
                panel_frame_block!(df, panel_array_view(omsk, :, j),
                                   map!(a -> "$name::observed@$a", cols, nxj))
            end
        end
    end
    if static
        return df
    end
    panel_frame_block!(df, view(pnl.amsk, :, j), ["amsk@$a" for a in nxj])
    panel_frame_block!(df, view(pnl.emsk, :, j), ["emsk@$a" for a in nxj])
    return df
end
"""
    panel_dataframe(pnl::AssetPanel; nx::Option{<:VecStr} = nothing,
                    ts::Option{<:AbstractVector} = nothing, fields = nothing,
                    assets = nothing, layout::Symbol = :long,
                    decode::Bool = true) -> DataFrames.DataFrame

Convert an [`AssetPanel`](@ref) to a `DataFrames.DataFrame`.

The library has no file format for a panel. To save, plot or share a panel, convert it to a table with this function, and write the table with a package that reads a `DataFrame`, for example CSV.jl or Arrow.jl.

The panel holds no asset names and no observation labels, because the carrier that holds the panel names them. So the caller gives them in `nx` and `ts`. When one of them is `nothing`, the table names each entry by its position on the axis.

The table has one of three shapes:

  - `fields` is one name: [`panel_frame_field`](@ref) gives that Panel Field as one row per observation and one column per asset. A [`TensorPanelField`](@ref) has a third axis, so this shape refuses it.
  - `layout = :long`: [`panel_frame_long`](@ref) gives one row per pair of observation and asset in the universe. Each Panel Field column is one table column.
  - `layout = :wide`: [`panel_frame_wide`](@ref) gives one row per observation and one column per pair of Panel Field column and asset. It keeps every cell and both universe masks.

In both layouts a tensor Panel Field gives one column per trailing-axis label, with the `"<field>=<label>"` name that the label has in a Feature Matrix.

# Algorithm

 1. Check `layout`.
 2. Name the axes. The asset names are `nx`, or the positions `"1"`, `"2"`, … when `nx` is `nothing`. The observation labels are `ts`, or the positions `1`, `2`, … when `ts` is `nothing` or the panel is static. Check each length against the axes of the panel.
 3. Find the positions of the selected assets with [`panel_frame_assets`](@ref).
 4. `fields` is one string: write that Panel Field with [`panel_frame_field`](@ref), and return the table.
 5. Otherwise find the Panel Fields with [`panel_frame_fields`](@ref), and write them with [`panel_frame_long`](@ref) or [`panel_frame_wide`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `nx`: The asset names of the universe of the panel, or `nothing` to name each asset by its position. See [`VecStr`](@ref).
  - `ts`: The observation labels, or `nothing` to name each observation by its position. A static panel has no observation axis and does not read it.
  - `fields`: One Panel Field name as a string, a collection of names, or `nothing` for all Panel Fields of the panel.
  - `assets`: One asset label as a string, a collection of labels, or `nothing` for all assets of the universe.
  - `layout::Symbol = :long`: `:long` for one row per pair of observation and asset in the universe. `:wide` for one row per observation and one column per pair of Panel Field column and asset. The function checks it in every call, and does not use it when `fields` names one Panel Field.
  - `decode::Bool = true`: Whether a [`CategoricalPanelField`](@ref) gives its levels in place of its integer codes.

# Validation

  - `layout` is `:long` or `:wide`. Raises an `ArgumentError`.
  - `length(nx)` is the length of the asset axis of the panel. Raises a `DimensionMismatch`.
  - `length(ts)` is the length of the observation axis of the panel, when the panel is time-varying. Raises a `DimensionMismatch`.
  - The panel holds every named Panel Field, and `nx` holds every named asset. Raises a `KeyError`.
  - No asset is named twice in `assets`. Raises an `ArgumentError`.
  - `fields` names no [`TensorPanelField`](@ref) as one string. Raises an `ArgumentError`.
  - No two columns of the table get the same name. Raises an `ArgumentError`. See [`panel_frame_column!`](@ref).

# Returns

  - `df::DataFrames.DataFrame`: The table. It holds copies, so a change to the table does not change the panel.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [NumericPanelField(; name = \"mcap\", vals = [1.0, 2.0, 3.0]),
                              CategoricalPanelField(; name = \"sector\", levels = [\"Tech\", \"Energy\"],
                                                    codes = [1, 2, 1])]);

julia> panel_dataframe(pnl; nx = [\"A\", \"B\", \"C\"])
3×3 DataFrame
 Row │ asset   mcap     sector
     │ String  Float64  String
─────┼─────────────────────────
   1 │ A           1.0  Tech
   2 │ B           2.0  Energy
   3 │ C           3.0  Tech
```

# Related

  - [`AssetPanel`](@ref)
  - [`panel_field`](@ref)
  - [`panel_feature_matrix`](@ref)
  - [`panel_frame_field`](@ref)
  - [`panel_frame_long`](@ref)
  - [`panel_frame_wide`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
function panel_dataframe(pnl::AssetPanel; nx::Option{<:VecStr} = nothing,
                         ts::Option{<:AbstractVector} = nothing, fields = nothing,
                         assets = nothing, layout::Symbol = :long, decode::Bool = true)
    @argcheck(layout in (:long, :wide),
              ArgumentError("`layout` is `:long`, one row per (observation, asset), or `:wide`, one column per (Panel Field column, asset). Got layout => :$(layout)"))
    ax = panel_axes(pnl)
    nxa = isnothing(nx) ? string.(1:ax[end]) : nx
    @argcheck(length(nxa) == ax[end],
              DimensionMismatch("`nx` names the assets of the panel's universe, so it is as long as the panel's asset axis, got length(nx) = $(length(nxa)) and $(ax[end]) assets"))
    tsa = if isnothing(ts) || isone(length(ax))
        1:ax[1]
    else
        ts
    end
    @argcheck(isone(length(ax)) || length(tsa) == ax[1],
              DimensionMismatch("`ts` labels the observations of a time-varying panel, so it is as long as the panel's observation axis, got length(ts) = $(length(tsa)) and $(ax[1]) observations"))
    j = panel_frame_assets(nxa, assets)
    nxj = [String(a) for a in view(nxa, j)]
    if isa(fields, AbstractString)
        return panel_frame_field(pnl, panel_field(pnl, fields), j, nxj, tsa, decode)
    end
    fs = panel_frame_fields(pnl, fields)
    return if layout === :long
        panel_frame_long(pnl, fs, j, nxj, tsa, decode)
    else
        panel_frame_wide(pnl, fs, j, nxj, tsa, decode)
    end
end

export panel_dataframe
