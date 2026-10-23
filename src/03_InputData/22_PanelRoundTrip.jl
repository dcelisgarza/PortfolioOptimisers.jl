"""
    panel_frame_axes(pnl::AssetPanel, nx::Option{<:VecStr}, ts::Option{<:AbstractVector}) -> Tuple

Name the asset axis and the observation axis of an [`AssetPanel`](@ref) for a table.

The panel holds no asset names and no observation labels, because the `PricesResult` or the `ReturnsResult` that holds the panel names them. [`panel_dataframe`](@ref) and [`panel_manifest`](@ref) take them from the caller, and name each entry by its position when the caller gives `nothing`. Both functions name the axes through this function, so a table and its manifest name them in the same way.

# Algorithm

 1. The asset names are `nx`, or the positions `"1"`, `"2"`, … when `nx` is `nothing`.
 2. The observation labels are `ts`, or the positions `1`, `2`, … when `ts` is `nothing`. A static panel has no observation axis, so its labels are the positions of its assets, and no caller reads them.
 3. Check each length against the axes of the panel.

# Arguments

  - `pnl`: The Asset Panel.
  - `nx`: The asset names of the universe of the panel, or `nothing`. See [`VecStr`](@ref).
  - `ts`: The observation labels, or `nothing`. A static panel does not read it.

# Validation

  - `length(nx)` is the length of the asset axis of the panel. Raises a `DimensionMismatch`.
  - `length(ts)` is the length of the observation axis of the panel, when the panel is time-varying. Raises a `DimensionMismatch`.

# Returns

  - `(nxa, tsa)`: The asset names and the observation labels. On a static panel, `tsa` is `1:N` for `N` assets.

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_manifest`](@ref)
  - [`panel_axes`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
function panel_frame_axes(pnl::AssetPanel, nx::Option{<:VecStr},
                          ts::Option{<:AbstractVector})
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
    return nxa, tsa
end
"""
    panel_manifest_row!(mf::DataFrames.DataFrame, kind::AbstractString, field, label;
                        group = missing, observed = missing,
                        placeholder = missing) -> nothing

Add one row to a manifest that [`panel_manifest`](@ref) writes.

A cell that the kind of the row does not use holds `missing`. A text reader writes `missing` as an empty cell, so a person who reads the manifest sees only the cells that the row uses.

# Arguments

  - `mf`: The manifest. This function adds the row to it.
  - `kind`: The kind of the row.
  - `field`: The name of the Panel Field of the row, or `missing`.
  - `label`: The label of the row, or `missing`.
  - `group`: The group of a trailing-axis label, `"grouped"` or `"ungrouped"` on a `"tensor"` row, or `missing`.
  - `observed`: Whether the Panel Field carries an observed mask, or `missing`.
  - `placeholder`: Whether the Panel Field carries a placeholder mask, or `missing`.

# Returns

  - `nothing`. `mf` holds the new row.

# Related

  - [`panel_manifest`](@ref)
  - [`panel_manifest_rows!`](@ref)
"""
function panel_manifest_row!(mf::DataFrames.DataFrame, kind::AbstractString, field, label;
                             group = missing, observed = missing,
                             placeholder = missing)::Nothing
    push!(mf, (kind, field, label, group, observed, placeholder))
    return nothing
end
"""
    panel_manifest_rows!(mf::DataFrames.DataFrame, f::NumericPanelField, decode::Bool) -> nothing
    panel_manifest_rows!(mf::DataFrames.DataFrame, f::CategoricalPanelField, decode::Bool) -> nothing
    panel_manifest_rows!(mf::DataFrames.DataFrame, f::TensorPanelField, decode::Bool) -> nothing

Add the rows that describe one Panel Field to a manifest that [`panel_manifest`](@ref) writes.

The table that [`panel_dataframe`](@ref) writes gives the values of a Panel Field. These rows give the parts of the Panel Field that a table column cannot hold: its kind, whether it carries an observed mask and a placeholder mask, whether the column of a categorical Panel Field holds its levels or its codes, the order of its levels, and the axis name, the labels and the groups of a tensor Panel Field.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`NumericPanelField`](@ref): one `"numeric"` row.
 2. [`CategoricalPanelField`](@ref): one `"categorical"` row with the label `"levels"` when `decode` is `true` and `"codes"` when it is `false`, then one `"level"` row per level, in the order of the codes. A level that no cell holds keeps its row.
 3. [`TensorPanelField`](@ref): one `"tensor"` row with the axis name as its label, and the group `"grouped"` when the Panel Field carries groups and `"ungrouped"` when it does not. Then one `"label"` row per trailing-axis label with its group.

The first row of each Panel Field records in `observed` whether it carries an observed mask, and in `placeholder` whether it carries a placeholder mask.

The `"categorical"` row records `decode`, so that [`asset_panel`](@ref) reads the column as the table holds it. The `"tensor"` row records the groups flag because an empty group and no group are both an empty cell in a text file. The flag is a word, so the `group` column always holds one value that is not a number. A reader that guesses column types, such as CSV.jl, then reads the column as text, and the group `"01"` stays `"01"`.

# Arguments

  - `mf`: The manifest. This function adds the rows to it.
  - `f`: The Panel Field.
  - `decode`: The `decode` of the [`panel_dataframe`](@ref) call. Only the method of a categorical Panel Field reads it.

# Returns

  - `nothing`. `mf` holds the new rows.

# Related

  - [`panel_manifest`](@ref)
  - [`panel_manifest_row!`](@ref)
  - [`AbstractPanelField`](@ref)
"""
function panel_manifest_rows!(mf::DataFrames.DataFrame, f::NumericPanelField,
                              ::Bool)::Nothing
    panel_manifest_row!(mf, "numeric", f.name, missing; observed = !isnothing(f.omsk),
                        placeholder = !isnothing(f.pmsk))
    return nothing
end
function panel_manifest_rows!(mf::DataFrames.DataFrame, f::CategoricalPanelField,
                              decode::Bool)::Nothing
    panel_manifest_row!(mf, "categorical", f.name, decode ? "levels" : "codes";
                        observed = !isnothing(f.omsk), placeholder = !isnothing(f.pmsk))
    for level in f.levels
        panel_manifest_row!(mf, "level", f.name, level)
    end
    return nothing
end
function panel_manifest_rows!(mf::DataFrames.DataFrame, f::TensorPanelField,
                              ::Bool)::Nothing
    grouped = !isnothing(f.groups)
    panel_manifest_row!(mf, "tensor", f.name, f.axis;
                        group = grouped ? "grouped" : "ungrouped",
                        observed = !isnothing(f.omsk), placeholder = !isnothing(f.pmsk))
    for (l, label) in pairs(f.labels)
        panel_manifest_row!(mf, "label", f.name, label;
                            group = grouped ? f.groups[l] : missing)
    end
    return nothing
end
"""
    panel_manifest(pnl::AssetPanel; nx::Option{<:VecStr} = nothing,
                   ts::Option{<:AbstractVector} = nothing, fields = nothing,
                   assets = nothing, decode::Bool = true) -> DataFrames.DataFrame

Describe an [`AssetPanel`](@ref) in a table, so that [`asset_panel`](@ref) can read the panel back from a [`panel_dataframe`](@ref) table.

The table of [`panel_dataframe`](@ref) holds the values and the masks. It cannot hold the order of the levels of a categorical Panel Field, a level that no cell holds, whether a categorical column holds the levels or the codes, or the axis name and the groups of a tensor Panel Field. The long layout also drops an asset that is never in the universe and an observation with no asset in the universe. The manifest holds these parts. Write the two tables with a package that writes a `DataFrame`, for example CSV.jl or Arrow.jl, and give both to [`asset_panel`](@ref) to read the panel back.

Call this function with the `nx`, `ts`, `fields`, `assets` and `decode` of the [`panel_dataframe`](@ref) call, so that the two tables describe the same panel.

The manifest has six columns: `kind`, `field`, `label`, `group`, `observed` and `placeholder`. Its rows are, in order:

  - One `"panel"` row. Its label is `"static"` or `"time-varying"`.
  - One `"observation"` row per observation of a time-varying panel. Its label is the observation label as text.
  - One `"asset"` row per selected asset. Its label is the asset name.
  - The rows of each selected Panel Field, from [`panel_manifest_rows!`](@ref).

# Algorithm

 1. Name the axes with [`panel_frame_axes`](@ref), and find the positions of the selected assets with [`panel_frame_assets`](@ref).
 2. Add the `"panel"` row, the `"observation"` rows and the `"asset"` rows.
 3. Find the Panel Fields with [`panel_frame_fields`](@ref), and add the rows of each with [`panel_manifest_rows!`](@ref).

# Arguments

  - `pnl`: The Asset Panel.
  - `nx`: The asset names of the universe of the panel, or `nothing` to name each asset by its position. See [`VecStr`](@ref).
  - `ts`: The observation labels, or `nothing` to name each observation by its position. A static panel has no observation axis and does not read it.
  - `fields`: One Panel Field name as a string, a collection of names, or `nothing` for all Panel Fields of the panel.
  - `assets`: One asset label as a string, a collection of labels, or `nothing` for all assets of the universe.
  - `decode::Bool = true`: Whether the categorical columns of the table hold the levels or the integer codes. The manifest records it, and [`asset_panel`](@ref) reads it.

# Validation

  - `length(nx)` and `length(ts)` match the axes of the panel. Raises a `DimensionMismatch`. See [`panel_frame_axes`](@ref).
  - The panel holds every named Panel Field, and `nx` holds every named asset. Raises a `KeyError`.
  - No asset is named twice in `assets`. Raises an `ArgumentError`.

# Returns

  - `mf::DataFrames.DataFrame`: The manifest.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [CategoricalPanelField(; name = \"sector\",
                                                    levels = [\"Tech\", \"Energy\", \"Retail\"],
                                                    codes = [1, 2, 1])]);

julia> panel_manifest(pnl; nx = [\"A\", \"B\", \"C\"])
8×6 DataFrame
 Row │ kind         field    label    group    observed  placeholder
     │ String       String?  String?  String?  Bool?     Bool?
─────┼───────────────────────────────────────────────────────────────
   1 │ panel        missing  static   missing   missing      missing
   2 │ asset        missing  A        missing   missing      missing
   3 │ asset        missing  B        missing   missing      missing
   4 │ asset        missing  C        missing   missing      missing
   5 │ categorical  sector   levels   missing     false        false
   6 │ level        sector   Tech     missing   missing      missing
   7 │ level        sector   Energy   missing   missing      missing
   8 │ level        sector   Retail   missing   missing      missing
```

# Related

  - [`asset_panel`](@ref)
  - [`panel_dataframe`](@ref)
  - [`AssetPanel`](@ref)
  - [`panel_manifest_rows!`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
function panel_manifest(pnl::AssetPanel; nx::Option{<:VecStr} = nothing,
                        ts::Option{<:AbstractVector} = nothing, fields = nothing,
                        assets = nothing, decode::Bool = true)
    nxa, tsa = panel_frame_axes(pnl, nx, ts)
    j = panel_frame_assets(nxa, assets)
    mf = DataFrames.DataFrame(; kind = String[], field = Union{Missing, String}[],
                              label = Union{Missing, String}[],
                              group = Union{Missing, String}[],
                              observed = Union{Missing, Bool}[],
                              placeholder = Union{Missing, Bool}[])
    static = panel_is_static(pnl)
    panel_manifest_row!(mf, "panel", missing, static ? "static" : "time-varying")
    if !static
        for t in tsa
            panel_manifest_row!(mf, "observation", missing, string(t))
        end
    end
    for k in j
        panel_manifest_row!(mf, "asset", missing, String(nxa[k]))
    end
    for f in panel_frame_fields(pnl, isa(fields, AbstractString) ? [fields] : fields)
        panel_manifest_rows!(mf, f, decode)
    end
    return mf
end
"""
    panel_text(x) -> String

Return a table cell as text, with `missing` as the empty string.

A reader that guesses the type of a column can give a cell as a number, a date or `missing` where the writer wrote text: CSV.jl reads the asset name `"7"` as the integer `7`, and an empty string as `missing`. [`asset_panel`](@ref) compares every name and every label of a table and its manifest as text, so the two tables agree however the reader typed them.

# Arguments

  - `x`: The cell.

# Returns

  - `s::String`: `string(x)`, or `""` when `x` is `missing`.

# Related

  - [`asset_panel`](@ref)
  - [`panel_manifest`](@ref)
"""
function panel_text(x)::String
    return ismissing(x) ? "" : string(x)
end
"""
    panel_table_column(df::DataFrames.DataFrame, name::AbstractString) -> AbstractVector

Return the column of a table with the name `name`, and refuse a name that the table does not hold.

# Arguments

  - `df`: The table.
  - `name`: The column name.

# Validation

  - `df` holds a column named `name`. Raises a `KeyError`.

# Returns

  - `col::AbstractVector`: The column, not a copy.

# Related

  - [`asset_panel`](@ref)
  - [`did_you_mean`](@ref)
"""
function panel_table_column(df::DataFrames.DataFrame, name::AbstractString)
    @argcheck(!iszero(DataFrames.columnindex(df, name)),
              KeyError("the table holds no column named \"$name\"$(did_you_mean(name, names(df))). A table from panel_dataframe with fields or assets left out needs the same fields in `asset_panel(df, mf; fields)`, and a manifest from panel_manifest with the same assets."))
    return df[!, name]
end
"""
    panel_table_cell(col::AbstractVector, r::Integer, name::AbstractString)

Return the value in row `r` of the table column `col`, and refuse a missing value.

A Panel Field holds a finite value in every cell, so a table from [`panel_dataframe`](@ref) holds a value in every cell. A missing value is a change that the table got after the export.

# Arguments

  - `col`: The column.
  - `r`: The row.
  - `name`: The column name, for the error message.

# Validation

  - The cell is not `missing`. Raises an `ArgumentError`.

# Returns

  - `x`: The cell.

# Related

  - [`panel_table_read!`](@ref)
"""
function panel_table_cell(col::AbstractVector, r::Integer, name::AbstractString)
    x = col[r]
    @argcheck(!ismissing(x),
              ArgumentError("the table holds no value in column \"$name\" at row $r, and a Panel Field holds a value in every cell"))
    return x
end
"""
    panel_table_read!(A::AbstractArray, lin::AbstractVector{<:Integer}, tab::NamedTuple, name::AbstractString, g) -> nothing
    panel_table_read!(A::AbstractArray, lin::Nothing, tab::NamedTuple, name::AbstractString, g) -> nothing

Write the cells of one Panel Field column, or of one mask, from a table into the array `A`.

`A` has the shape of the column in the panel. The shape is `assets` on a static panel and `observations × assets` on a time-varying one. The caller fills it before the call, so a cell that the table does not hold keeps that value.

# Algorithm

The method that Julia selects is the algorithm. `lin` says the layout.

 1. The long layout gives `lin`, the linear index into `A` of the cell of each table row. Write `g` of the value of row `r` into `A[lin[r]]`.
 2. The wide layout gives `nothing`. The table holds one column per asset, named `"<name>@<asset>"`, with one row per observation. Write `g` of each value into the cell of its observation and asset.

# Arguments

  - `A`: The array. This function changes it in place.
  - `lin`: The linear index of the cell of each row of a long table, or `nothing` for a wide table.
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `name`: The column name in the long layout, and the part of the column name before `"@"` in the wide layout.
  - `g`: The function that converts a cell to the value that `A` holds.

# Validation

  - The table holds each column that the call reads. Raises a `KeyError`. See [`panel_table_column`](@ref).
  - No cell that the call reads is `missing`. Raises an `ArgumentError`. See [`panel_table_cell`](@ref).

# Returns

  - `nothing`. `A` holds the values.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_column`](@ref)
  - [`panel_table_cell`](@ref)
"""
function panel_table_read!(A::AbstractArray, lin::AbstractVector{<:Integer},
                           tab::NamedTuple, name::AbstractString, g)::Nothing
    col = panel_table_column(tab.df, name)
    for (r, i) in pairs(lin)
        A[i] = g(panel_table_cell(col, r, name))
    end
    return nothing
end
function panel_table_read!(A::AbstractArray, ::Nothing, tab::NamedTuple,
                           name::AbstractString, g)::Nothing
    B = reshape(A, :, length(tab.nx))
    for (k, a) in pairs(tab.nx)
        c = "$name@$a"
        col = panel_table_column(tab.df, c)
        for t in axes(B, 1)
            B[t, k] = g(panel_table_cell(col, t, c))
        end
    end
    return nothing
end
"""
    panel_table_eltype(lin::AbstractVector{<:Integer}, tab::NamedTuple, cols::VecStr) -> Type
    panel_table_eltype(lin::Nothing, tab::NamedTuple, cols::VecStr) -> Type

Return the element type that holds every value of the table columns of one Panel Field.

The function derives the type from the columns, and never forces it. The type is the promoted element type of the columns with `Missing` removed, which [`panel_table_types`](@ref) reads. A reader that gives an integer column and a floating-point column for two labels of one tensor Panel Field gets the floating-point type.

# Arguments

  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `cols`: The column names of the Panel Field in the long layout. See [`VecStr`](@ref).

# Validation

  - The table holds each column. Raises a `KeyError`. See [`panel_table_column`](@ref).
  - The columns state a type that is not `Missing`. A reader such as CSV.jl types a column with no value as `Missing`, for example the columns of a long table with no row. Raises an `ArgumentError`.

# Returns

  - `T::Type`: The element type.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_read!`](@ref)
  - [`panel_table_types`](@ref)
"""
function panel_table_eltype(lin::AbstractVector{<:Integer}, tab::NamedTuple, cols::VecStr)
    T = panel_table_types(lin, tab, cols)
    #! A reader types a column with no value as `Missing`, so the column states no type.
    @argcheck(T !== Union{},
              ArgumentError("the columns $(cols) of the table hold no value, so the reader of the table gave them no type to read. Read them with a type, for example `CSV.File(path; types = Dict(\"$(first(cols))\" => Float64))`."))
    return T
end
function panel_table_eltype(lin::Nothing, tab::NamedTuple, cols::VecStr)
    T = panel_table_types(lin, tab, cols)
    @argcheck(T !== Union{},
              ArgumentError("the columns of $(cols) in the table hold no value, so the reader of the table gave them no type to read. Read them with a type, for example `CSV.File(path; types = Dict(\"$(first(cols))@<asset>\" => Float64))`."))
    return T
end
"""
    panel_table_types(lin::AbstractVector{<:Integer}, tab::NamedTuple, cols::VecStr) -> Type
    panel_table_types(lin::Nothing, tab::NamedTuple, cols::VecStr) -> Type

Return the promoted element type of the table columns of one Panel Field, with `Missing` removed.

A column that holds no value states no type, so the answer is `Union{}` when no column holds a value. [`panel_table_eltype`](@ref) refuses that answer, and [`panel_code_eltype`](@ref) reads it as a column that states no code type.

# Algorithm

The method that Julia selects is the algorithm.

 1. The long layout reads the columns `cols`.
 2. The wide layout reads the column `"<col>@<asset>"` of each column and each asset.

# Arguments

  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `cols`: The column names of the Panel Field in the long layout. See [`VecStr`](@ref).

# Validation

  - The table holds each column. Raises a `KeyError`. See [`panel_table_column`](@ref).

# Returns

  - `T::Type`: The promoted element type, or `Union{}`.

# Related

  - [`panel_table_eltype`](@ref)
  - [`panel_code_eltype`](@ref)
"""
function panel_table_types(::AbstractVector{<:Integer}, tab::NamedTuple, cols::VecStr)
    return mapreduce(c -> nonmissingtype(eltype(panel_table_column(tab.df, c))),
                     promote_type, cols; init = Union{})
end
function panel_table_types(::Nothing, tab::NamedTuple, cols::VecStr)
    return mapreduce(c -> nonmissingtype(eltype(panel_table_column(tab.df, c))),
                     promote_type, ("$c@$a" for c in cols for a in tab.nx); init = Union{})
end
"""
    panel_code_eltype(s::NamedTuple, lin, tab::NamedTuple) -> Type

Return the element type of the codes of a categorical Panel Field that [`asset_panel`](@ref) reads from a table.

The type is derived from the table when the table holds it. A column of codes that a reader typed as an integer type states the type of the codes, so an `Int32` code column gives `Int32` codes. A column of levels holds text, and a code is then the position of its level in the levels. A position in Julia is an `Int`, so such codes are `Int`. A column of codes that holds text or no value states no code type either, and its codes are `Int` too.

# Algorithm

 1. When the manifest entry `s` records that the column holds the levels, return `Int`.
 2. Otherwise read the type of the column with [`panel_table_types`](@ref). Return it when it is an integer type, and `Int` when it is not, or when the column holds no value.

# Arguments

  - `s`: The manifest entry of the categorical Panel Field, from [`panel_manifest_fields`](@ref).
  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.

# Returns

  - `T::Type{<:Integer}`: The element type of the codes.

# Related

  - [`panel_read_field`](@ref)
  - [`panel_code_reader`](@ref)
  - [`panel_table_types`](@ref)
"""
function panel_code_eltype(s::NamedTuple, lin, tab::NamedTuple)
    if s.decode
        return Int
    end
    T = panel_table_types(lin, tab, [s.name])
    return T !== Union{} && T <: Integer ? T : Int
end
"""
    panel_table_mask(lin, tab::NamedTuple, name::AbstractString, sz::Tuple) -> BitArray

Read one mask of the shape `sz` from a table, with `false` in every cell that the table does not hold.

# Arguments

  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `name`: The column name of the mask. See [`panel_table_read!`](@ref).
  - `sz`: The shape of the mask.

# Returns

  - `M::BitArray`: The mask.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_read!`](@ref)
"""
function panel_table_mask(lin, tab::NamedTuple, name::AbstractString, sz::Tuple)
    M = falses(sz)
    panel_table_read!(M, lin, tab, name, identity)
    return M
end
"""
    panel_read_field(s::NamedTuple, lin, tab::NamedTuple, sz::Tuple) -> AbstractPanelField

Read one Panel Field from a table, as its manifest entry `s` describes it.

A cell that a long table does not hold is outside the universe, and the table holds no value for it. It gets the value zero, the first level of a categorical Panel Field, and `false` in the observed mask and in the placeholder mask.

# Algorithm

 1. `"numeric"`: read the column named after the Panel Field into an array of the element type of the column, through [`panel_table_read!`](@ref).
 2. `"categorical"`: read the same column into an array of the code type of [`panel_code_eltype`](@ref). When the manifest records that the column holds the levels, find the code of the level in each cell. When it records the codes, each cell holds the code.
 3. `"tensor"`: read the column `"<field>=<label>"` of each label into its slice of the trailing axis.
 4. Read the observed mask from the columns named `"<column>::observed"` when the manifest records one, and the placeholder mask from the columns named `"<column>::placeholder"` in the same way.

# Arguments

  - `s`: The manifest entry of the Panel Field, from [`panel_manifest_fields`](@ref).
  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `sz`: The shape of one column of the panel: `(assets,)` or `(observations, assets)`.

# Validation

  - When the column holds the levels, each categorical cell is a level of the manifest. Raises a `KeyError`.
  - When the column holds the codes, each categorical cell is an integer. Raises an `ArgumentError`.
  - The constructor of the Panel Field checks the values. See [`NumericPanelField`](@ref), [`CategoricalPanelField`](@ref) and [`TensorPanelField`](@ref).

# Returns

  - `f::AbstractPanelField`: The Panel Field.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_read!`](@ref)
  - [`panel_table_eltype`](@ref)
  - [`panel_code_eltype`](@ref)
  - [`panel_table_mask`](@ref)
  - [`panel_field_table_mask`](@ref)
"""
function panel_read_field(s::NamedTuple, lin, tab::NamedTuple, sz::Tuple)
    if s.kind == "numeric"
        vals = fill(zero(panel_table_eltype(lin, tab, [s.name])), sz)
        panel_table_read!(vals, lin, tab, s.name, identity)
        return NumericPanelField(s.name, vals,
                                 panel_field_table_mask(s.observed, lin, tab,
                                                        "$(s.name)::observed", sz),
                                 panel_field_table_mask(s.placeholder, lin, tab,
                                                        "$(s.name)::placeholder", sz))
    elseif s.kind == "categorical"
        T = panel_code_eltype(s, lin, tab)
        codes = ones(T, sz)
        panel_table_read!(codes, lin, tab, s.name, panel_code_reader(s, T))
        return CategoricalPanelField(s.name, s.labels, codes,
                                     panel_field_table_mask(s.observed, lin, tab,
                                                            "$(s.name)::observed", sz),
                                     panel_field_table_mask(s.placeholder, lin, tab,
                                                            "$(s.name)::placeholder", sz))
    end
    cols = ["$(s.name)=$l" for l in s.labels]
    vals = fill(zero(panel_table_eltype(lin, tab, cols)), sz..., length(cols))
    for (l, c) in pairs(cols)
        panel_table_read!(selectdim(vals, ndims(vals), l), lin, tab, c, identity)
    end
    return TensorPanelField(s.name, s.axis, s.labels, s.grouped ? s.groups : nothing, vals,
                            panel_field_table_mask(s.observed, lin, tab,
                                                   ["$c::observed" for c in cols], sz),
                            panel_field_table_mask(s.placeholder, lin, tab,
                                                   ["$c::placeholder" for c in cols], sz))
end
"""
    panel_field_table_mask(on::Bool, lin, tab::NamedTuple, name::AbstractString, sz::Tuple) -> Option{BitArray}
    panel_field_table_mask(on::Bool, lin, tab::NamedTuple, cols::VecStr, sz::Tuple) -> Option{BitArray}

Read one mask of a Panel Field from a table, the observed mask or the placeholder mask, when its manifest entry records one.

The mask of the column `"<column>"` is the column `"<column>::observed"` or `"<column>::placeholder"`. A numeric or a categorical Panel Field has one mask column, and its mask has the shape `sz`. A tensor Panel Field has one mask column per label, and its mask has a trailing label axis.

# Arguments

  - `on`: Whether the manifest records the mask. With `false`, the answer is `nothing`.
  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `name`, `cols`: The mask column of the Panel Field, or the mask column of each label.
  - `sz`: The shape of one column of the panel.

# Returns

  - `M::Option{BitArray}`: The mask, or `nothing`.

# Related

  - [`panel_read_field`](@ref)
  - [`panel_table_mask`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
"""
function panel_field_table_mask(on::Bool, lin, tab::NamedTuple, name::AbstractString,
                                sz::Tuple)
    return on ? panel_table_mask(lin, tab, name, sz) : nothing
end
function panel_field_table_mask(on::Bool, lin, tab::NamedTuple, cols::VecStr, sz::Tuple)
    if !on
        return nothing
    end
    M = falses(sz..., length(cols))
    for (l, c) in pairs(cols)
        panel_table_read!(selectdim(M, ndims(M), l), lin, tab, c, identity)
    end
    return M
end
"""
    panel_code_reader(s::NamedTuple, ::Type{T}) where {T <: Integer}

Return the function that converts a cell of a categorical column to its integer code of the type `T`.

# Algorithm

 1. When the manifest entry `s` records that the column holds the levels, the cell holds a level. Return the position of the level, as text, in the levels of `s`.
 2. When `s` records the codes, the cell holds the code. Return it as a `T`. A reader that gives the cell as text gets it parsed.

# Arguments

  - `s`: The manifest entry of the categorical Panel Field, from [`panel_manifest_fields`](@ref). Its `decode` tells whether the cell holds the level or the code.
  - `T`: The element type of the codes, from [`panel_code_eltype`](@ref).

# Validation

  - When the column holds the levels, the cell is a level of the manifest. Raises a `KeyError`.
  - When the column holds the codes, the cell is an integer that `T` holds. Raises an `ArgumentError`.

# Returns

  - `g`: The function.

# Related

  - [`panel_read_field`](@ref)
  - [`panel_code_eltype`](@ref)
  - [`panel_text`](@ref)
"""
function panel_code_reader(s::NamedTuple, ::Type{T}) where {T <: Integer}
    if !s.decode
        return function (x)
            c = tryparse(T, panel_text(x))
            @argcheck(!isnothing(c),
                      ArgumentError("the manifest records that the column of the categorical Panel Field \"$(s.name)\" holds the integer codes, as panel_dataframe writes it with decode = false, got $(repr(x)). Write the manifest with the decode of the panel_dataframe call."))
            return something(c)
        end
    end
    code = Dict(level => i for (i, level) in pairs(s.labels))
    return function (x)
        c = get(code, panel_text(x), 0)
        @argcheck(!iszero(c),
                  KeyError("the categorical Panel Field \"$(s.name)\" holds no level \"$(panel_text(x))\"$(did_you_mean(panel_text(x), s.labels)). Its levels are $(join(s.labels, ", ")). A reader that guesses column types can change a level: read the column as text, for example with `CSV.File(path; types = Dict(\"$(s.name)\" => String))`. A table that panel_dataframe wrote with decode = false needs a manifest written with decode = false."))
        return c
    end
end
"""
    panel_manifest_axes(mf::DataFrames.DataFrame) -> NamedTuple

Read the shape, the observation labels and the asset names from a manifest that [`panel_manifest`](@ref) wrote.

# Algorithm

 1. Read the `"panel"` row. Its label is `"static"` or `"time-varying"`.
 2. Read the label of each `"observation"` row and each `"asset"` row as text, in order.

# Arguments

  - `mf`: The manifest.

# Validation

  - `mf` holds the columns `kind` and `label`. Raises a `KeyError`.
  - `mf` holds one `"panel"` row, with the label `"static"` or `"time-varying"`. Raises an `ArgumentError`.
  - A static manifest holds no `"observation"` row. Raises an `ArgumentError`.
  - No asset name repeats. Raises an `ArgumentError`.

# Returns

  - `(; static, ts, nx)`: Whether the panel is static, the observation labels and the asset names, as text.

# Related

  - [`asset_panel`](@ref)
  - [`panel_manifest`](@ref)
  - [`panel_text`](@ref)
"""
function panel_manifest_axes(mf::DataFrames.DataFrame)
    kind = panel_text.(panel_table_column(mf, "kind"))
    label = panel_text.(panel_table_column(mf, "label"))
    shape = label[kind .== "panel"]
    @argcheck(length(shape) == 1 && only(shape) in ("static", "time-varying"),
              ArgumentError("a manifest from panel_manifest holds one \"panel\" row whose label is \"static\" or \"time-varying\", got $(length(shape)) such row(s) with the label(s) $(join(shape, ", "))"))
    static = only(shape) == "static"
    ts = label[kind .== "observation"]
    nx = label[kind .== "asset"]
    @argcheck(!static || isempty(ts),
              ArgumentError("the manifest states a static panel and holds $(length(ts)) \"observation\" row(s), but a static panel has no observation axis"))
    @argcheck(allunique(nx),
              ArgumentError("the manifest names an asset twice, so a table column or a table row cannot name one asset. Got\nnx => $(nx)"))
    return (; static = static, ts = ts, nx = nx)
end
"""
    panel_manifest_fields(mf::DataFrames.DataFrame) -> Vector{<:NamedTuple}

Read the entry of each Panel Field from a manifest that [`panel_manifest`](@ref) wrote.

An entry is the named tuple `(; kind, name, observed, placeholder, decode, grouped, axis, labels, groups)`. `labels` holds the levels of a categorical Panel Field and the trailing-axis labels of a tensor Panel Field. `groups` holds one entry per label: the group of a trailing-axis label, and `""` for a level. `decode` belongs to a categorical Panel Field, and is `false` for the other kinds. `axis` and `grouped` belong to a tensor Panel Field, and are `""` and `false` for the other kinds. A numeric Panel Field has no label and no group.

# Algorithm

 1. Skip the `"panel"`, `"observation"` and `"asset"` rows.
 2. A `"numeric"`, `"categorical"` or `"tensor"` row starts a new entry. The label of a `"categorical"` row gives `decode`, and the group of a `"tensor"` row gives `grouped`.
 3. A `"level"` or a `"label"` row adds its label, and its group, to the entry of its Panel Field.

# Arguments

  - `mf`: The manifest.

# Validation

  - `mf` holds the six columns of a manifest. Raises a `KeyError`.
  - The label of a `"categorical"` row is `"levels"` or `"codes"`, and the group of a `"tensor"` row is `"grouped"` or `"ungrouped"`. Raises an `ArgumentError`.
  - A `"level"` row follows the `"categorical"` row of its Panel Field, and a `"label"` row follows the `"tensor"` row of its Panel Field. Raises an `ArgumentError`.
  - Each row has a kind that a manifest holds. Raises an `ArgumentError`.

# Returns

  - `specs::Vector{<:NamedTuple}`: One entry per Panel Field, in the order of the manifest.

# Related

  - [`asset_panel`](@ref)
  - [`panel_manifest`](@ref)
  - [`panel_manifest_rows!`](@ref)
  - [`panel_manifest_flag`](@ref)
  - [`panel_manifest_attach!`](@ref)
  - [`panel_text`](@ref)
"""
function panel_manifest_fields(mf::DataFrames.DataFrame)
    text = c -> panel_text.(panel_table_column(mf, c))
    kind, field, label, group = text("kind"), text("field"), text("label"), text("group")
    observed, placeholder = text("observed"), text("placeholder")
    specs = @NamedTuple{kind::String, name::String, observed::Bool, placeholder::Bool,
                        decode::Bool, grouped::Bool, axis::String, labels::Vector{String},
                        groups::Vector{String}}[]
    for r in eachindex(kind)
        k = kind[r]
        if k in ("numeric", "categorical", "tensor")
            panel_manifest_flag(r, k, label[r], group[r])
            push!(specs,
                  (; kind = k, name = field[r], observed = observed[r] == "true",
                   placeholder = placeholder[r] == "true",
                   decode = k == "categorical" && label[r] == "levels",
                   grouped = k == "tensor" && group[r] == "grouped",
                   axis = k == "tensor" ? label[r] : "", labels = String[],
                   groups = String[]))
        elseif !(k in ("panel", "observation", "asset"))
            panel_manifest_attach!(specs, r, k, field[r], label[r], group[r])
        end
    end
    return specs
end
"""
    panel_manifest_flag(r::Integer, k::AbstractString, label::AbstractString,
                        group::AbstractString) -> nothing

Check the flag that the first row of a categorical or a tensor Panel Field records as a word.

The label of a `"categorical"` row is `"levels"` or `"codes"`: what the table column of the Panel Field holds. The group of a `"tensor"` row is `"grouped"` or `"ungrouped"`: whether the Panel Field carries groups. A `"numeric"` row records no flag.

# Arguments

  - `r`: The row of the manifest, for the error message.
  - `k`: The kind of the row.
  - `label`: The label of the row.
  - `group`: The group of the row.

# Validation

  - The flag of a `"categorical"` or a `"tensor"` row is one of its two words. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`panel_manifest_fields`](@ref)
  - [`panel_manifest_rows!`](@ref)
"""
function panel_manifest_flag(r::Integer, k::AbstractString, label::AbstractString,
                             group::AbstractString)::Nothing
    @argcheck(k != "categorical" || label in ("levels", "codes"),
              ArgumentError("row $r of the manifest is a \"categorical\" row, so its label is \"levels\" or \"codes\", got \"$label\""))
    @argcheck(k != "tensor" || group in ("grouped", "ungrouped"),
              ArgumentError("row $r of the manifest is a \"tensor\" row, so its group is \"grouped\" or \"ungrouped\", got \"$group\""))
    return nothing
end
"""
    panel_manifest_attach!(specs::AbstractVector{<:NamedTuple}, r::Integer, k::AbstractString,
                           field::AbstractString, label::AbstractString,
                           group::AbstractString) -> nothing

Add the label, and the group, of a `"level"` or a `"label"` row of a manifest to the entry of its Panel Field.

The entry of the Panel Field is the last entry of `specs`, because [`panel_manifest`](@ref) writes the level rows and the label rows of a Panel Field after its first row.

# Arguments

  - `specs`: The entries that [`panel_manifest_fields`](@ref) read so far. The last one gets the label.
  - `r`: The row of the manifest, for the error message.
  - `k`: The kind of the row.
  - `field`: The name of the Panel Field of the row.
  - `label`: The level or the trailing-axis label.
  - `group`: The group of a trailing-axis label, or `""`.

# Validation

  - `k` is `"level"` or `"label"`, and the last entry is the `"categorical"` or the `"tensor"` entry of the Panel Field `field`. Raises an `ArgumentError`.

# Returns

  - `nothing`. The last entry of `specs` holds the label.

# Related

  - [`panel_manifest_fields`](@ref)
  - [`panel_manifest_rows!`](@ref)
"""
function panel_manifest_attach!(specs::AbstractVector{<:NamedTuple}, r::Integer,
                                k::AbstractString, field::AbstractString,
                                label::AbstractString, group::AbstractString)::Nothing
    parent = k == "level" ? "categorical" : "tensor"
    @argcheck(k in ("level", "label") &&
              !isempty(specs) &&
              last(specs).kind == parent &&
              last(specs).name == field,
              ArgumentError("row $r of the manifest has the kind \"$k\" and the field \"$field\". A manifest row is a \"panel\", \"observation\", \"asset\", \"numeric\", \"categorical\", \"level\", \"tensor\" or \"label\" row, and a \"level\" or a \"label\" row follows the \"categorical\" or the \"tensor\" row of its Panel Field."))
    push!(last(specs).labels, label)
    push!(last(specs).groups, group)
    return nothing
end
"""
    panel_long_cells(df::DataFrames.DataFrame, ax::NamedTuple) -> Vector{Int}

Find the cell of the panel that each row of a long table holds.

A long table from [`panel_dataframe`](@ref) holds one row per asset of a static panel, and one row per pair of observation and asset in the universe of a time-varying panel. Its `"asset"` column and, on a time-varying panel, its `"observation"` column name the cell. This function finds the position of each name in the manifest, compared as text, and returns the linear index of the cell in an `assets` or an `observations × assets` array.

# Algorithm

 1. Find the position of the asset of each row in `ax.nx`.
 2. On a time-varying panel, find the position of the observation of each row in `ax.ts`, and combine the two positions into a linear index.
 3. Check that no cell repeats, and that a static table holds every asset.

# Arguments

  - `df`: The long table.
  - `ax`: The axes of the manifest, from [`panel_manifest_axes`](@ref).

# Validation

  - `df` holds the key columns. Raises a `KeyError`.
  - The manifest names no observation twice, when the panel is time-varying. Raises an `ArgumentError`.
  - The manifest names the asset, and the observation, of each row. Raises a `KeyError`.
  - No two rows hold one cell. Raises an `ArgumentError`.
  - A static table holds one row per asset. Raises an `ArgumentError`.

# Returns

  - `lin::Vector{Int}`: The linear index of the cell of each row.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_read!`](@ref)
  - [`panel_text`](@ref)
"""
function panel_long_cells(df::DataFrames.DataFrame, ax::NamedTuple)
    kpos = Dict(a => k for (k, a) in pairs(ax.nx))
    #! A table column states no element type that inference can read, so a comprehension over
    #! it reaches every method that a package adds to `Base.collect_to_with_first!`, and JET
    #! reports one in JuMP. A comprehension over the row positions does not.
    rows = 1:DataFrames.nrow(df)
    acol = panel_table_column(df, "asset")
    k = [panel_long_position(kpos, acol[r], "asset") for r in rows]
    lin = k
    if !ax.static
        @argcheck(allunique(ax.ts),
                  ArgumentError("the manifest names an observation twice, so a row of a long table cannot name one observation. Read the wide layout instead."))
        tpos = Dict(t => i for (i, t) in pairs(ax.ts))
        L = LinearIndices((length(ax.ts), length(ax.nx)))
        ocol = panel_table_column(df, "observation")
        lin = [L[panel_long_position(tpos, ocol[r], "observation"), k[r]] for r in rows]
    end
    @argcheck(allunique(lin),
              ArgumentError("two rows of the long table hold one cell of the panel, so the table is not one that panel_dataframe wrote"))
    @argcheck(!ax.static || length(lin) == length(ax.nx),
              ArgumentError("the long table of a static panel holds one row per asset, got $(length(lin)) rows and $(length(ax.nx)) assets in the manifest"))
    return lin
end
"""
    panel_long_position(pos::AbstractDict, x, key::AbstractString) -> Int

Return the position of the name `x`, as text, on one axis of the manifest.

# Arguments

  - `pos`: The position of each name on the axis.
  - `x`: The cell of the key column.
  - `key`: The name of the key column, for the error message.

# Validation

  - `pos` holds the name. Raises a `KeyError`.

# Returns

  - `i::Int`: The position.

# Related

  - [`panel_long_cells`](@ref)
  - [`panel_text`](@ref)
"""
function panel_long_position(pos::AbstractDict, x, key::AbstractString)::Int
    i = get(pos, panel_text(x), 0)
    @argcheck(!iszero(i),
              KeyError("the manifest names no $key \"$(panel_text(x))\"$(did_you_mean(panel_text(x), collect(keys(pos)))). A reader that guesses column types can change a name, for example the asset \"007\" to the integer 7: read the key columns as text, for example with `CSV.File(path; types = Dict(\"$key\" => String))`."))
    return i
end
"""
    asset_panel(df::DataFrames.DataFrame, mf::DataFrames.DataFrame;
                fields = nothing) -> AssetPanel

Read an [`AssetPanel`](@ref) from a table that [`panel_dataframe`](@ref) wrote and the manifest that [`panel_manifest`](@ref) wrote. This is the inverse of the export.

The two tables can go through any table format and come back, for example through CSV.jl or Arrow.jl, so a panel makes a round trip through a file with no file format of its own. Arrow.jl can map a file into memory, and a column selection before the read gives a subset of the Panel Fields.

The function reads the long and the wide layouts, and tells them apart by the `"asset"` column, which only the long layout holds. It does not read the single-field shape, which holds no mask.

  - A wide table holds every cell and both universe masks, so the panel comes back whole.
  - A long table holds only the rows where the active mask is `true`. The function makes the active mask from the rows that the table holds, and the manifest gives the axes. The value of a cell outside the universe does not come back. The cell gets zero, the first level of a categorical Panel Field, and `false` in the observed mask and in the placeholder mask.

A read that masks the cells outside the universe gives the same answer before and after a long round trip, for example [`panel_field_values`](@ref) with `inactive = NaN`. A read that keeps those cells gets the new values. Two such reads are [`panel_field_values`](@ref) with its default `inactive = nothing`, and the wide layout of [`panel_dataframe`](@ref).

The function compares names and labels as text, so a reader that types a column still matches the manifest. A reader that changes the text of a name does not match it: CSV.jl reads the asset name `"007"` as the integer `7`. Read such a column as text. The manifest records whether each categorical column holds the levels or the codes, so the function reads the column as [`panel_dataframe`](@ref) wrote it.

The codes of a categorical column that holds the codes keep the integer type of the column, so `Int32` codes come back as `Int32` codes when the reader keeps the column type. A column that holds the levels carries no code type, and its codes are the positions of the levels, which are `Int`. See [`panel_code_eltype`](@ref).

A static Panel Field that a time-varying panel lifts, see [`RepeatedLeading`](@ref), comes back as an ordinary array with the same values.

# Algorithm

 1. Read the axes and the Panel Field entries of the manifest with [`panel_manifest_axes`](@ref) and [`panel_manifest_fields`](@ref). Keep the entries that `fields` names with [`panel_manifest_select`](@ref).
 2. Tell the layout with [`panel_table_cells`](@ref). In the long layout, it finds the cell of each row. In the wide layout, it checks the row count: one row on a static panel, one row per observation on a time-varying one.
 3. Read each Panel Field with [`panel_read_field`](@ref).
 4. On a time-varying panel, read the masks. [`panel_active_mask`](@ref) reads the active mask. The estimation mask is the `"emsk"` column of a long table, and the `"emsk@<asset>"` columns of a wide one.
 5. Make the panel with the [`AssetPanel`](@ref) constructor, which checks it.

# Arguments

  - `df`: The table, in the long or the wide layout of [`panel_dataframe`](@ref).
  - `mf`: The manifest, from [`panel_manifest`](@ref).
  - `fields`: One Panel Field name as a string, a collection of names, or `nothing` for all Panel Fields of the manifest.

# Validation

  - The manifest is one that [`panel_manifest`](@ref) wrote. Raises an `ArgumentError` or a `KeyError`. See [`panel_manifest_axes`](@ref) and [`panel_manifest_fields`](@ref).
  - The manifest holds every named Panel Field. Raises a `KeyError`.
  - The table holds each column that the manifest and `fields` name. Raises a `KeyError`.
  - A wide table has one row, or one row per observation. Raises a `DimensionMismatch`.
  - The `"observation"` column of a wide table names the observations of the manifest, as text and in order. Raises an `ArgumentError`. See [`panel_table_cells`](@ref).
  - The rows of a long table name cells of the manifest, each once. See [`panel_long_cells`](@ref).
  - No cell that the function reads is `missing`. Raises an `ArgumentError`.
  - The value columns of a Panel Field state a type. A long table with no row, read back by CSV.jl, types each column as `Missing`. Raises an `ArgumentError`. See [`panel_table_eltype`](@ref).
  - The [`AssetPanel`](@ref) constructor and the constructor of each Panel Field check the result.

# Returns

  - `pnl::AssetPanel`: The Asset Panel.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [NumericPanelField(; name = \"mcap\", vals = [1.0 2.0; 3.0 4.0]),
                              CategoricalPanelField(; name = \"sector\",
                                                    levels = [\"Tech\", \"Energy\", \"Retail\"],
                                                    codes = [1 2; 1 1])], amsk = Bool[1 0; 1 1],
                        emsk = Bool[1 0; 0 1]);

julia> df = panel_dataframe(pnl; nx = [\"A\", \"B\"]);

julia> mf = panel_manifest(pnl; nx = [\"A\", \"B\"]);

julia> pnl2 = asset_panel(df, mf);

julia> pnl2.amsk == pnl.amsk, pnl2.emsk == pnl.emsk
(true, true)

julia> panel_field(pnl2, \"sector\").levels
3-element Vector{String}:
 \"Tech\"
 \"Energy\"
 \"Retail\"
```

# Related

  - [`panel_dataframe`](@ref)
  - [`panel_manifest`](@ref)
  - [`AssetPanel`](@ref)
  - [`panel_read_field`](@ref)
  - [`panel_table_cells`](@ref)
  - [`panel_active_mask`](@ref)
  - [`panel_manifest_select`](@ref)
  - [`panel_manifest_axes`](@ref)
  - [`panel_manifest_fields`](@ref)
"""
function asset_panel(df::DataFrames.DataFrame, mf::DataFrames.DataFrame; fields = nothing)
    ax = panel_manifest_axes(mf)
    specs = panel_manifest_select(panel_manifest_fields(mf), fields)
    sz = ax.static ? (length(ax.nx),) : (length(ax.ts), length(ax.nx))
    tab = (; df = df, nx = ax.nx)
    lin = panel_table_cells(df, ax)
    pf = AbstractPanelField[panel_read_field(s, lin, tab, sz) for s in specs]
    if ax.static
        return AssetPanel(; pf = pf)
    end
    return AssetPanel(; pf = pf, amsk = panel_active_mask(lin, tab, sz),
                      emsk = panel_table_mask(lin, tab, "emsk", sz))
end
"""
    panel_table_cells(df::DataFrames.DataFrame, ax::NamedTuple) -> Option{Vector{Int}}

Tell the layout of a table from [`panel_dataframe`](@ref), and find where its rows go in the panel.

Only the long layout holds an `"asset"` column. The wide layout names a column `"observation"` or `"<name>@<asset>"`, so no column of it is named `"asset"`.

# Algorithm

 1. When the table holds an `"asset"` column, it is a long table. Return the cell of each row, from [`panel_long_cells`](@ref).
 2. Otherwise it is a wide table, whose rows are the observations in order. Check the row count and, on a time-varying panel, the labels of its `"observation"` column against the manifest, as text and in order. Return `nothing`.

# Arguments

  - `df`: The table.
  - `ax`: The axes of the manifest, from [`panel_manifest_axes`](@ref).

# Validation

  - A wide table has one row on a static panel, and one row per observation on a time-varying one. Raises a `DimensionMismatch`.
  - The `"observation"` column of a wide table, when the table holds one, names the observations of the manifest in order. Raises an `ArgumentError`.
  - The rows of a long table name cells of the manifest, each once. See [`panel_long_cells`](@ref).

# Returns

  - `lin::Option{Vector{Int}}`: The linear index of the cell of each row of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).

# Related

  - [`asset_panel`](@ref)
  - [`panel_long_cells`](@ref)
  - [`Option`](@ref)
"""
function panel_table_cells(df::DataFrames.DataFrame, ax::NamedTuple)
    if !iszero(DataFrames.columnindex(df, "asset"))
        return panel_long_cells(df, ax)
    end
    n = ax.static ? 1 : length(ax.ts)
    @argcheck(DataFrames.nrow(df) == n,
              DimensionMismatch("a wide table holds one row on a static panel and one row per observation on a time-varying one, got $(DataFrames.nrow(df)) rows and a manifest that needs $n"))
    #! A wide table is read by position, so its labels must be the ones of the manifest, in
    #! order: a sorted or filtered table would otherwise read back permuted.
    c = DataFrames.columnindex(df, "observation")
    @argcheck(ax.static || iszero(c) || panel_text.(df[!, c]) == ax.ts,
              ArgumentError("a wide table holds the observations of the manifest in order, and its \"observation\" column names them in another order or with other labels. Sort the table by the observation as the manifest orders it, or read the column as text, for example with `CSV.File(path; types = Dict(\"observation\" => String))`."))
    return nothing
end
"""
    panel_active_mask(lin::AbstractVector{<:Integer}, tab::NamedTuple, sz::Tuple) -> BitMatrix
    panel_active_mask(lin::Nothing, tab::NamedTuple, sz::Tuple) -> BitMatrix

Read the active mask of a time-varying panel from a table.

# Algorithm

The method that Julia selects is the algorithm.

 1. A long table holds a row for each cell in the universe and no other row, so the active mask is `true` at the cell of each row.
 2. A wide table holds the active mask in its `"amsk@<asset>"` columns. Read them with [`panel_table_mask`](@ref).

# Arguments

  - `lin`: The linear indices of a long table, or `nothing` for a wide table. See [`panel_table_read!`](@ref).
  - `tab`: The named tuple `(; df, nx)` of the table and the asset names of the manifest.
  - `sz`: The shape `(observations, assets)` of the mask.

# Validation

  - A wide table holds each `"amsk@<asset>"` column. Raises a `KeyError`. See [`panel_table_column`](@ref).

# Returns

  - `amsk::BitMatrix`: The active mask.

# Related

  - [`asset_panel`](@ref)
  - [`panel_table_mask`](@ref)
"""
function panel_active_mask(lin::AbstractVector{<:Integer}, ::NamedTuple, sz::Tuple)
    amsk = falses(sz)
    amsk[lin] .= true
    return amsk
end
function panel_active_mask(::Nothing, tab::NamedTuple, sz::Tuple)
    return panel_table_mask(nothing, tab, "amsk", sz)
end
"""
    panel_manifest_select(specs::AbstractVector{<:NamedTuple}, fields::Nothing)
    panel_manifest_select(specs::AbstractVector{<:NamedTuple}, fields::AbstractString)
    panel_manifest_select(specs::AbstractVector{<:NamedTuple}, fields)

Keep the manifest entries of the Panel Fields that an [`asset_panel`](@ref) call names, in the order of the call.

# Algorithm

The method that Julia selects is the algorithm.

 1. `fields` is `nothing`: return every entry, in the order of the manifest.
 2. `fields` is one string: return its entry, as a one-entry vector.
 3. Otherwise: find the entry of each name with [`panel_manifest_entry`](@ref), in the order of `fields`.

# Arguments

  - `specs`: The entries, from [`panel_manifest_fields`](@ref).
  - `fields`: One Panel Field name as a string, a collection of names, or `nothing`. An entry can be a `String` or a `Symbol`.

# Validation

  - `specs` holds an entry for each name. Raises a `KeyError`. See [`panel_manifest_entry`](@ref).

# Returns

  - `specs`: The entries, in the order of the call.

# Related

  - [`asset_panel`](@ref)
  - [`panel_manifest_entry`](@ref)
"""
function panel_manifest_select(specs::AbstractVector{<:NamedTuple}, ::Nothing)
    return specs
end
function panel_manifest_select(specs::AbstractVector{<:NamedTuple}, fields::AbstractString)
    return panel_manifest_select(specs, [fields])
end
function panel_manifest_select(specs::AbstractVector{<:NamedTuple}, fields)
    return [panel_manifest_entry(specs, String(name)) for name in fields]
end
"""
    panel_manifest_entry(specs::AbstractVector{<:NamedTuple}, name::AbstractString) -> NamedTuple

Return the manifest entry of the Panel Field named `name`, and refuse a name that the manifest does not hold.

# Arguments

  - `specs`: The entries, from [`panel_manifest_fields`](@ref).
  - `name`: The name of the Panel Field.

# Validation

  - `specs` holds an entry with the name. Raises a `KeyError`.

# Returns

  - `s::NamedTuple`: The entry.

# Related

  - [`asset_panel`](@ref)
  - [`panel_manifest_fields`](@ref)
"""
function panel_manifest_entry(specs::AbstractVector{<:NamedTuple}, name::AbstractString)
    k = findfirst(s -> s.name == name, specs)
    @argcheck(!isnothing(k),
              KeyError("the manifest holds no Panel Field named \"$name\"$(did_you_mean(name, [s.name for s in specs])). It holds $(join((s.name for s in specs), ", "))"))
    return specs[something(k)]
end

export panel_manifest
