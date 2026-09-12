"""
    panel_frame_columns(f::NumericPanelField, decode::Bool)
    panel_frame_columns(f::CategoricalPanelField, decode::Bool)
    panel_frame_columns(f::TensorPanelField, decode::Bool)

Return the table columns one Panel Field contributes to a [`panel_dataframe`](@ref), in column order.

A column is the triple `(name, vals, omsk)`: its name, the slab of values over the panel's own axes, and the slab of its observed mask, or `nothing` when the Panel Field carries none. Every slab is `assets` on a static panel and `observations × assets` on a time-varying one, so [`panel_array_view`](@ref) slices any of them by asset.

The **column count is where a table and a Feature Matrix part company**. A Feature Matrix is numeric, so a categorical Panel Field spreads into one indicator column per level. A table column holds a string, so the same Panel Field is one column carrying the level itself. A tensor Panel Field is the same in both: one column per trailing-axis label, under the `"<field>=<label>"` name [`panel_field_labels`](@ref) gives it, because a label axis is the one thing a flat table cannot carry as an axis of its own.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref): one column, named after the Panel Field, holding its values.
 2. [`CategoricalPanelField`](@ref): one column, named after the Panel Field, holding the level each code names under `decode`, and the codes themselves otherwise.
 3. [`TensorPanelField`](@ref): one column per trailing-axis label, named `"<field>=<label>"`, holding that label's slice.

# Arguments

  - `f`: The Panel Field.
  - `decode::Bool`: Whether a categorical Panel Field renders its levels rather than its codes. The other two kinds carry no codes and ignore it.

# Returns

  - `cols::Vector{Tuple{String, Any, Any}}`: One `(name, vals, omsk)` triple per column, in column order.

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

Resolve the Panel Fields a [`panel_dataframe`](@ref) call names, in the order it names them.

# Algorithm

The method that Julia selects is the algorithm.

 1. `fields` is `nothing`: every Panel Field, in panel order.
 2. Otherwise: one Panel Field per entry, looked up by [`panel_field`](@ref), in the order `fields` writes them.

# Arguments

  - `pnl`: The Asset Panel.
  - `fields`: The Panel Field names to include, or `nothing` for every one of them.

# Validation

  - The panel holds a Panel Field under each name. Raises a `KeyError`. See [`panel_field`](@ref).

# Returns

  - `fs`: The Panel Fields, in column order.

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

Resolve the asset labels a [`panel_dataframe`](@ref) call names into positions on the panel's asset axis.

The panel does not name its own assets: its Panel Fields share an asset axis, and the carrier that holds them names it. So the names are the caller's, and this is where a name becomes the position every slab is sliced by.

# Algorithm

The method that Julia selects is the algorithm.

 1. `assets` is `nothing`: every position, in axis order.
 2. `assets` is one label: that label's position.
 3. Otherwise: one position per label, in the order `assets` writes them.

# Arguments

  - `nx`: The asset names of the panel's universe. See [`VecStr`](@ref).
  - `assets`: The asset labels to include, or `nothing` for every one of them.

# Validation

  - `nx` names every entry of `assets`. Raises a `KeyError`.
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
    panel_frame_block!(df::DataFrames.DataFrame, A::AbstractVector, names::VecStr) -> nothing
    panel_frame_block!(df::DataFrames.DataFrame, A::AbstractMatrix, names::VecStr) -> nothing

Write one slab of a Panel Field into a wide table, as one column per asset.

# Algorithm

The method that Julia selects is the algorithm, and the slab's rank is what says whether the panel is static.

 1. A vector slab is static and carries no observation axis, so each asset's value becomes a one-row column.
 2. A matrix slab is time-varying, so each asset's column of the slab becomes a column of the table.

# Arguments

  - `df`: The table under construction. It is written to.
  - `A`: The slab, already sliced to the selected assets.
  - `names`: The column name of each selected asset, in slab column order. See [`VecStr`](@ref).

# Returns

  - `nothing`. `df` carries the result.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_wide`](@ref)
  - [`panel_frame_field`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_block!(df::DataFrames.DataFrame, A::AbstractVector,
                            names::VecStr)::Nothing
    for (c, name) in pairs(names)
        df[!, name] = [A[c]]
    end
    return nothing
end
function panel_frame_block!(df::DataFrames.DataFrame, A::AbstractMatrix,
                            names::VecStr)::Nothing
    for (c, name) in pairs(names)
        df[!, name] = A[:, c]
    end
    return nothing
end
"""
    panel_frame_field(pnl::AssetPanel, f::AbstractPanelField, j::VecInt, nxj::VecStr, ts, decode::Bool) -> DataFrames.DataFrame
    panel_frame_field(pnl::AssetPanel, f::TensorPanelField, j::VecInt, nxj::VecStr, ts, decode::Bool)

Render one Panel Field as the table of its own shape: one row per observation, one column per asset.

This is what a [`panel_dataframe`](@ref) call that names a single Panel Field returns. A numeric or a categorical Panel Field already **is** observations × assets, so the table needs no second key and no layout, and it carries neither the universe masks nor the observed mask: it is the Panel Field itself, laid out as it stands.

A [`TensorPanelField`](@ref) is refused here. Its values are observations × assets × labels, so no arrangement of one row per observation and one column per asset holds them; a long or a wide layout does, by spreading the label axis into columns.

# Algorithm

The method that Julia selects decides whether the Panel Field has this shape.

 1. Open the table with the observation column, unless the panel is static.
 2. Write the Panel Field's one value column, as one table column per selected asset.

# Arguments

  - `pnl`: The Asset Panel.
  - `f`: The Panel Field.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order `j` writes them. See [`VecStr`](@ref).
  - `ts`: The observation labels, ignored when the panel is static.
  - `decode::Bool`: Whether a categorical Panel Field renders its levels rather than its codes.

# Validation

  - `f` is not a [`TensorPanelField`](@ref). Raises an `ArgumentError`.

# Returns

  - `df::DataFrames.DataFrame`: The observations × assets table, with an `"observation"` column first when the panel is time-varying.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_block!`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_field(pnl::AssetPanel, f::AbstractPanelField, j::VecInt, nxj::VecStr,
                           ts, decode::Bool)
    df = DataFrames.DataFrame()
    if !panel_is_static(pnl)
        df[!, "observation"] = collect(ts)
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

Render the selected Panel Fields in the long layout: one row per `(observation, asset)`, one column per Panel Field column.

The rows a time-varying panel writes are **filtered by the active mask**, so every row of the table is an asset that was in the universe at that observation. The estimation mask stays as its own `"emsk"` column, because it varies within the rows that survive the filter. A static panel has neither mask and neither column, and its rows are its assets.

# Algorithm

 1. Open the table with the `"observation"` and `"asset"` key columns, unless the panel is static, which carries the asset key alone.
 2. Walk the Panel Fields in order, and each one's columns from [`panel_frame_columns`](@ref). Ravel each slab in row order, so an entry lands on the row its two keys name, and follow a Panel Field's value column with its observed-mask column when it carries one.
 3. Append the estimation mask, and drop every row the active mask does not hold.

# Arguments

  - `pnl`: The Asset Panel.
  - `fs`: The Panel Fields, in column order.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order `j` writes them. See [`VecStr`](@ref).
  - `ts`: The observation labels, ignored when the panel is static.
  - `decode::Bool`: Whether a categorical Panel Field renders its levels rather than its codes.

# Returns

  - `df::DataFrames.DataFrame`: The long table.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_wide`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_long(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool)
    static = panel_is_static(pnl)
    df = DataFrames.DataFrame()
    if static
        df[!, "asset"] = collect(nxj)
    else
        df[!, "observation"] = repeat(collect(ts); inner = length(j))
        df[!, "asset"] = repeat(collect(nxj); outer = length(ts))
    end
    for f in fs
        for (name, vals, omsk) in panel_frame_columns(f, decode)
            df[!, name] = vec(permutedims(panel_array_view(vals, :, j)))
            if !isnothing(omsk)
                df[!, "$name::observed"] = vec(permutedims(panel_array_view(omsk, :, j)))
            end
        end
    end
    if static
        return df
    end
    df[!, "emsk"] = vec(permutedims(view(pnl.emsk, :, j)))
    return df[vec(permutedims(view(pnl.amsk, :, j))), :]
end
"""
    panel_frame_wide(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool) -> DataFrames.DataFrame

Render the selected Panel Fields in the wide layout: one row per observation, one column per `(Panel Field column, asset)`.

A column name is the Panel Field column, an `"@"`, and the asset: `"mcap@AAPL"`, `"beta=size@AAPL"`, `"mcap::observed@AAPL"`. The layout keeps every cell of the panel, so both universe masks come with it, as the `"amsk@<asset>"` and `"emsk@<asset>"` blocks. Nothing is filtered, which is what makes the wide layout the lossless one and the long layout the one that reads.

A column name is a rendering, not a key. A caller that needs the two parts back reads the long layout, whose keys are columns of their own, rather than splitting a name on its separators.

# Algorithm

 1. Open the table with the `"observation"` column, unless the panel is static, whose one row is every asset at once.
 2. Walk the Panel Fields in order, and each one's columns from [`panel_frame_columns`](@ref), writing each as a block of one column per selected asset through [`panel_frame_block!`](@ref).
 3. Append the active-mask and estimation-mask blocks, unless the panel is static.

# Arguments

  - `pnl`: The Asset Panel.
  - `fs`: The Panel Fields, in column order.
  - `j`: The selected positions on the asset axis. See [`VecInt`](@ref).
  - `nxj`: The name of each selected asset, in the order `j` writes them. See [`VecStr`](@ref).
  - `ts`: The observation labels, ignored when the panel is static.
  - `decode::Bool`: Whether a categorical Panel Field renders its levels rather than its codes.

# Returns

  - `df::DataFrames.DataFrame`: The wide table.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_frame_columns`](@ref)
  - [`panel_frame_block!`](@ref)
  - [`panel_frame_long`](@ref)
  - [`VecInt`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_wide(pnl::AssetPanel, fs, j::VecInt, nxj::VecStr, ts, decode::Bool)
    static = panel_is_static(pnl)
    df = DataFrames.DataFrame()
    if !static
        df[!, "observation"] = collect(ts)
    end
    for f in fs
        for (name, vals, omsk) in panel_frame_columns(f, decode)
            panel_frame_block!(df, panel_array_view(vals, :, j), ["$name@$a" for a in nxj])
            if !isnothing(omsk)
                panel_frame_block!(df, panel_array_view(omsk, :, j),
                                   ["$name::observed@$a" for a in nxj])
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

Render an [`AssetPanel`](@ref) as a `DataFrames.DataFrame`.

This is the panel's interchange. The library persists no panel of its own, and a table is what every tool that is not this library already reads, so a caller writes the result with whatever they already use and a panel reaches disk, a plot or a spreadsheet without the library owning a format.

A panel does not name its own assets or its own observations, because the carrier that holds it names them. So `nx` and `ts` are the caller's, and each falls back to its own axis positions.

Three shapes are reachable, and `fields` picks between them. A single Panel Field name gives that Panel Field laid out as it stands, through [`panel_frame_field`](@ref). Any other selection gives a layout: `:long` reads, `:wide` is lossless. A [`TensorPanelField`](@ref) has no shape of its own and the first refuses it, and it spreads into one column per trailing-axis label in both layouts, under the `"<field>=<label>"` name it takes in a Feature Matrix.

# Algorithm

 1. Name the axes: `nx` or the asset positions, `ts` or the observation positions, each checked against the shape the panel's Panel Fields agree on.
 2. Resolve the assets with [`panel_frame_assets`](@ref).
 3. `fields` is one name: render that Panel Field with [`panel_frame_field`](@ref) and return.
 4. Otherwise resolve the Panel Fields with [`panel_frame_fields`](@ref), and render them with [`panel_frame_long`](@ref) or [`panel_frame_wide`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `nx`: The asset names of the panel's universe, or `nothing` to name each asset by its position. See [`VecStr`](@ref).
  - `ts`: The observation labels, or `nothing` to name each observation by its position. A static panel has no observation axis and ignores it.
  - `fields`: One Panel Field name, a collection of them, or `nothing` for every Panel Field of the panel.
  - `assets`: One asset label, a collection of them, or `nothing` for every asset of the universe.
  - `layout::Symbol = :long`: `:long` for one row per `(observation, asset)`, filtered by the active mask. `:wide` for one row per observation and one column per `(Panel Field column, asset)`. Ignored when `fields` names a single Panel Field.
  - `decode::Bool = true`: Whether a [`CategoricalPanelField`](@ref) renders its levels rather than its integer codes.

# Validation

  - `layout` is `:long` or `:wide`. Raises an `ArgumentError`.
  - `length(nx)` is the panel's asset axis. Raises a `DimensionMismatch`.
  - `length(ts)` is the panel's observation axis, when the panel is time-varying. Raises a `DimensionMismatch`.
  - The panel holds every named Panel Field, and the universe holds every named asset. Raises a `KeyError`.

# Returns

  - `df::DataFrames.DataFrame`: The table.

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
