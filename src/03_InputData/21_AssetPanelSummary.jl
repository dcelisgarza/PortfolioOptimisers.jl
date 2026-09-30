"""
    panel_field_kind(f::NumericPanelField) -> Symbol
    panel_field_kind(f::CategoricalPanelField) -> Symbol
    panel_field_kind(f::TensorPanelField) -> Symbol

Return the kind of a Panel Field as a symbol, for a summary table.

# Algorithm

The method that Julia selects is the algorithm. It returns `:numeric`, `:categorical` or `:tensor`.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `kind::Symbol`: The kind of the Panel Field.

# Related

  - [`NumericPanelField`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`TensorPanelField`](@ref)
  - [`panel_field_coverage`](@ref)
"""
function panel_field_kind(::NumericPanelField)::Symbol
    return :numeric
end
function panel_field_kind(::CategoricalPanelField)::Symbol
    return :categorical
end
function panel_field_kind(::TensorPanelField)::Symbol
    return :tensor
end
"""
    panel_field_width(f::AbstractPanelField) -> Int
    panel_field_width(f::TensorPanelField) -> Int

Return the count of cells that one pair of observation and asset holds in a Panel Field.

A numeric and a categorical Panel Field hold one cell for each pair. A tensor Panel Field holds one cell for each label of its trailing axis.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `K::Int`: The count of cells for each pair of observation and asset.

# Related

  - [`TensorPanelField`](@ref)
  - [`panel_field_missing`](@ref)
"""
function panel_field_width(::AbstractPanelField)::Int
    return 1
end
function panel_field_width(f::TensorPanelField)::Int
    return length(f.labels)
end
"""
    panel_field_missing(f::AbstractPanelField) -> AbstractArray{<:Integer}

Count the cells of a Panel Field that the builder filled, for each pair of observation and asset.

A cell is missing when the observed mask of the Panel Field is false there. A Panel Field with no observed mask cannot blank, so it has no missing cell. A tensor Panel Field holds one cell for each label, so its count for one pair is from `0` to the count of labels.

# Algorithm

 1. Read the axes of the Panel Field with [`panel_field_axes`](@ref).
 2. The observed mask is `nothing`: return zeros over the axes.
 3. Otherwise put the cells of each pair in one row, and count the false cells of each row.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `miss::AbstractArray{<:Integer}`: The count of missing cells, over the observation and asset axes of the Panel Field.

# Related

  - [`panel_field_axes`](@ref)
  - [`panel_field_width`](@ref)
  - [`panel_field_coverage`](@ref)
  - [`panel_align_active`](@ref)
"""
function panel_field_missing(f::AbstractPanelField)
    ax = panel_field_axes(f)
    if isnothing(f.omsk)
        return zeros(Int, ax)
    end
    return reshape(count(!, reshape(f.omsk, prod(ax), :); dims = 2), ax)
end
"""
    panel_active_cells(pnl::AssetPanel) -> AbstractArray{Bool}

Return the active cells of an [`AssetPanel`](@ref) over its observation and asset axes.

A static panel has no active mask. Each asset of a static panel is in its universe, so every cell is active.

# Algorithm

 1. The panel is static: return a true array over [`panel_axes`](@ref).
 2. Otherwise return the active mask.

# Arguments

  - `pnl`: The Asset Panel.

# Returns

  - `act::AbstractArray{Bool}`: The active cells.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_is_static`](@ref)
  - [`panel_field_coverage`](@ref)
"""
function panel_active_cells(pnl::AssetPanel)
    return panel_is_static(pnl) ? trues(panel_axes(pnl)) : pnl.amsk
end
"""
    panel_field_coverage(f::AbstractPanelField, act::AbstractArray{Bool}) -> NamedTuple

Count the cells and the missing cells of a Panel Field, over all cells and over the active cells.

# Algorithm

 1. Count the missing cells of each pair of observation and asset with [`panel_field_missing`](@ref).
 2. Count the cells, the active cells, the missing cells and the missing active cells. A pair holds [`panel_field_width`](@ref) cells.
 3. Count the assets that are active at one observation or more, and whose active cells are all missing.

# Arguments

  - `f`: The Panel Field.
  - `act`: The active cells, see [`panel_active_cells`](@ref).

# Returns

  - `cov::NamedTuple`: `cells`, the count of cells; `missing`, the count of missing cells; `active`, the count of active cells; `active_missing`, the count of missing active cells; `assets_missing`, the count of assets whose active cells are all missing.

# Related

  - [`panel_field_missing`](@ref)
  - [`panel_active_cells`](@ref)
  - [`DataFrames.describe(pnl::AssetPanel)`](@ref)
"""
function panel_field_coverage(f::AbstractPanelField, act::AbstractArray{Bool})
    miss = panel_field_missing(f)
    K = panel_field_width(f)
    N = size(miss)[end]
    M = reshape(miss, :, N)
    A = reshape(act, :, N)
    assets_missing = count(j -> any(view(A, :, j)) &&
                                all(t -> !A[t, j] || M[t, j] == K, axes(A, 1)), 1:N)
    return (; cells = length(miss) * K, missing = sum(miss), active = count(act) * K,
            active_missing = sum(m * a for (m, a) in zip(miss, act); init = 0),
            assets_missing)
end
"""
    DataFrames.describe(pnl::AssetPanel; by::Option{<:AbstractString} = nothing)

Report the share of the cells of each Panel Field of an [`AssetPanel`](@ref) that the builder filled.

A cell is missing when the observed mask of its Panel Field is false. The builder fills that cell by the fill policy of the field, and the observed mask keeps the record, so a filled cell stays missing in this report. A Panel Field with no observed mask cannot blank, and has no missing cell.

With `by = nothing` the table has one row for each Panel Field. With `by` the name of a [`CategoricalPanelField`](@ref), the table has one row for each other Panel Field and each level of `by`. A cell belongs to a level when it is active, and when `by` is observed there and holds that level. A tensor Panel Field counts one cell for each label.

A share over no cell is `NaN`: an empty set has no share.

# Algorithm

 1. Read the active cells with [`panel_active_cells`](@ref).
 2. `by = nothing`: count the cells of each Panel Field with [`panel_field_coverage`](@ref), and divide each missing count by its cell count.
 3. Otherwise read the categorical Panel Field `by` with [`panel_field`](@ref). For each other Panel Field and each level, count the cells of the level with [`panel_level_coverage`](@ref), and divide.

# Arguments

  - `pnl`: The Asset Panel.
  - `by`: The name of a categorical Panel Field to group the cells by, or `nothing`.

# Validation

  - The panel holds a Panel Field named `by`. Raises a `KeyError`.
  - The Panel Field `by` is a [`CategoricalPanelField`](@ref). Raises an `ArgumentError`.

# Returns

  - `df::DataFrames.DataFrame`: With `by = nothing`, the columns `field`, `kind`, `cells`, `missing` (the share of the cells), `active_cells`, `active_missing` (the share of the active cells) and `assets_missing` (the count of assets whose active cells are all missing). With `by`, the columns `field`, `level`, `cells` (the count of cells of the level) and `missing` (their share).

# Examples

```jldoctest
julia> using DataFrames

julia> pnl = AssetPanel(;
                        pf = [NumericPanelField(; name = \"mcap\",
                                                vals = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                                                omsk = Bool[0 1; 1 1; 1 0]),
                              CategoricalPanelField(; name = \"sector\", levels = [\"Tech\", \"Energy\"],
                                                    codes = [1 2; 1 2; 1 2])],
                        amsk = Bool[1 1; 1 1; 1 0], emsk = Bool[1 1; 1 1; 1 0]);

julia> select(describe(pnl), :field, :missing, :active_missing, :assets_missing)
2×4 DataFrame
 Row │ field   missing   active_missing  assets_missing
     │ String  Float64   Float64         Int64
─────┼──────────────────────────────────────────────────
   1 │ mcap    0.333333             0.2               0
   2 │ sector  0.0                  0.0               0

julia> describe(pnl; by = \"sector\")
2×4 DataFrame
 Row │ field   level   cells  missing
     │ String  String  Int64  Float64
─────┼─────────────────────────────────
   1 │ mcap    Tech        3  0.333333
   2 │ mcap    Energy      2  0.0
```

# Related

  - [`AssetPanel`](@ref)
  - [`panel_info`](@ref)
  - [`panel_align_active`](@ref)
  - [`panel_field_coverage`](@ref)
  - [`panel_level_coverage`](@ref)
  - [`panel_dataframe`](@ref)
  - [`Option`](@ref)
"""
function DataFrames.describe(pnl::AssetPanel; by::Option{<:AbstractString} = nothing)
    act = panel_active_cells(pnl)
    if isnothing(by)
        cs = [panel_field_coverage(f, act) for f in pnl.pf]
        return DataFrames.DataFrame(; field = String[f.name for f in pnl.pf],
                                    kind = Symbol[panel_field_kind(f) for f in pnl.pf],
                                    cells = Int[c.cells for c in cs],
                                    missing = [c.missing / c.cells for c in cs],
                                    active_cells = Int[c.active for c in cs],
                                    active_missing = [c.active_missing / c.active
                                                      for c in cs],
                                    assets_missing = Int[c.assets_missing for c in cs])
    end
    g = panel_field(pnl, by)
    @argcheck(g isa CategoricalPanelField,
              ArgumentError("a missing-share report groups the cells by the levels of a categorical Panel Field, and \"$by\" is a $(panel_field_kind(g)) Panel Field"))
    rows = [(; field = f.name, level = l, panel_level_coverage(f, g, k, act)...)
            for f in pnl.pf if f.name != by for (k, l) in pairs(g.levels)]
    return DataFrames.DataFrame(; field = String[r.field for r in rows],
                                level = String[r.level for r in rows],
                                cells = Int[r.cells for r in rows],
                                missing = [r.missing / r.cells for r in rows])
end
"""
    panel_level_coverage(f::AbstractPanelField, g::CategoricalPanelField, k::Integer,
                         act::AbstractArray{Bool}) -> NamedTuple

Count the cells and the missing cells of a Panel Field on the cells of one level of a categorical Panel Field.

A cell belongs to the level when it is active, and when `g` is observed there and holds the code `k`. A filled category is not an observation of its level.

# Algorithm

 1. Mark the cells of the level: active, observed in `g`, and code `k`.
 2. Count the missing cells of `f` with [`panel_field_missing`](@ref) on the marked cells, and the marked cells times [`panel_field_width`](@ref).

# Arguments

  - `f`: The Panel Field to count.
  - `g`: The categorical Panel Field that groups the cells.
  - `k`: The code of the level.
  - `act`: The active cells, see [`panel_active_cells`](@ref).

# Returns

  - `cov::NamedTuple`: `cells`, the count of cells of the level; `missing`, the count of missing cells among them.

# Related

  - [`DataFrames.describe(pnl::AssetPanel)`](@ref)
  - [`panel_field_missing`](@ref)
  - [`CategoricalPanelField`](@ref)
"""
function panel_level_coverage(f::AbstractPanelField, g::CategoricalPanelField, k::Integer,
                              act::AbstractArray{Bool})
    member = panel_level_cells(g, k, act)
    miss = panel_field_missing(f)
    return (; cells = count(member) * panel_field_width(f),
            missing = sum(m * b for (m, b) in zip(miss, member); init = 0))
end
"""
    panel_level_cells(g::CategoricalPanelField, k::Integer, act::AbstractArray{Bool}) -> BitArray

Mark the cells that hold one level of a categorical Panel Field.

A cell holds the level when it is active, and when `g` is observed there and holds the code `k`. A categorical Panel Field with no observed mask is observed at each cell.

# Algorithm

 1. Compare each code of `g` with `k`, and intersect with `act`.
 2. Intersect with the observed mask of `g` when it has one.

# Arguments

  - `g`: The categorical Panel Field.
  - `k`: The code of the level.
  - `act`: The active cells, see [`panel_active_cells`](@ref).

# Returns

  - `member::BitArray`: The cells of the level.

# Related

  - [`panel_level_coverage`](@ref)
  - [`panel_info_levels`](@ref)
"""
function panel_level_cells(g::CategoricalPanelField, k::Integer, act::AbstractArray{Bool})
    member = (g.codes .== k) .& act
    if !isnothing(g.omsk)
        member .&= g.omsk
    end
    return member
end
"""
    panel_spread(v::AbstractVector{<:Real}) -> Option{NamedTuple}

Return the minimum, the median and the maximum of a vector of counts, or `nothing` when it is empty.

# Algorithm

 1. `v` is empty: return `nothing`.
 2. Otherwise return the minimum, the median and the maximum of `v`.

# Arguments

  - `v`: The counts.

# Returns

  - `s::Option{NamedTuple}`: `min`, `median` and `max`, or `nothing`.

# Related

  - [`panel_mask_coverage`](@ref)
  - [`Option`](@ref)
"""
function panel_spread(v::AbstractVector{<:Real})
    if isempty(v)
        return nothing
    end
    return (; min = minimum(v), median = Statistics.median(v), max = maximum(v))
end
"""
    panel_mask_coverage(msk::AbstractMatrix{Bool}) -> NamedTuple

Count the cells of a universe mask, and the spread of its assets for each observation and of its observations for each asset.

# Algorithm

 1. Count the cells and the true cells of `msk`.
 2. Count the true cells of each observation, and take their spread with [`panel_spread`](@ref).
 3. Count the true cells of each asset, keep the assets with one true cell or more, and take their spread.

# Arguments

  - `msk`: The universe mask, observations × assets.

# Returns

  - `cov::NamedTuple`: `cells`, the count of cells; `in_mask`, the count of true cells; `per_observation`, the spread of the count of assets for each observation; `assets`, the count of assets with one true cell or more; `N`, the count of assets; `durations`, the spread of the count of observations for each of those assets.

# Related

  - [`panel_spread`](@ref)
  - [`panel_info`](@ref)
"""
function panel_mask_coverage(msk::AbstractMatrix{Bool})
    per_asset = vec(sum(msk; dims = 1))
    durations = filter(>(0), per_asset)
    return (; cells = length(msk), in_mask = count(msk),
            per_observation = panel_spread(vec(sum(msk; dims = 2))),
            assets = length(durations), N = size(msk, 2),
            durations = panel_spread(durations))
end
"""
    panel_percent(x::Real) -> Real

Return a share as a percentage, rounded to one decimal, for a report.

# Arguments

  - `x`: The share.

# Returns

  - `p::Real`: The percentage.

# Related

  - [`panel_info`](@ref)
"""
function panel_percent(x::Real)
    return round(100 * x; digits = 1)
end
"""
    panel_info_mask(io::IO, title::AbstractString, msk::Nothing) -> nothing
    panel_info_mask(io::IO, title::AbstractString, msk::AbstractMatrix{Bool}) -> nothing

Print the coverage of one universe mask of an Asset Panel.

# Algorithm

The method that Julia selects is the algorithm.

 1. `msk` is `nothing`: the panel is static, so print that it has no mask.
 2. Otherwise read [`panel_mask_coverage`](@ref), and print the true cells, the spread of the assets for each observation, the count of assets in the mask and the spread of their durations.

# Arguments

  - `io`: The stream.
  - `title`: The name of the mask.
  - `msk`: The universe mask, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`panel_info`](@ref)
  - [`panel_mask_coverage`](@ref)
"""
function panel_info_mask(io::IO, title::AbstractString, ::Nothing)::Nothing
    println(io, "\n", title, ": none, the panel is static")
    return nothing
end
function panel_info_mask(io::IO, title::AbstractString, msk::AbstractMatrix{Bool})::Nothing
    c = panel_mask_coverage(msk)
    println(io, "\n", title)
    println(io, "  in mask          : ", c.in_mask, " / ", c.cells, " cells (",
            panel_percent(c.in_mask / c.cells), "%)")
    if !isnothing(c.per_observation)
        s = c.per_observation
        println(io, "  assets per obs.  : min ", s.min, ", median ", s.median, ", max ",
                s.max)
    end
    println(io, "  assets in mask   : ", c.assets, " / ", c.N)
    if !isnothing(c.durations)
        s = c.durations
        println(io, "  duration (obs.)  : min ", s.min, ", median ", s.median, ", max ",
                s.max)
    end
    return nothing
end
"""
    panel_info_levels(io::IO, pnl::AssetPanel, g::CategoricalPanelField,
                      act::AbstractArray{Bool}) -> nothing

Print the levels of one categorical Panel Field, grouped by the smallest count of assets that holds each level over the observations.

A level with few assets gives a noisy cross-sectional estimate, so the report puts each level in one of four groups: fewer than 10 assets, 10 to 19, 20 to 49, and 50 or more. The count of one level at one observation is the count of cells of the level, see [`panel_level_cells`](@ref). A static panel has one count for each level.

# Algorithm

 1. For each level, mark its cells with [`panel_level_cells`](@ref), count them at each observation, and take the minimum.
 2. For each group, print the count of its levels and their names. A group of more than six levels prints the first four names and the count of the others.

# Arguments

  - `io`: The stream.
  - `pnl`: The Asset Panel.
  - `g`: The categorical Panel Field.
  - `act`: The active cells, see [`panel_active_cells`](@ref).

# Returns

  - `nothing`.

# Related

  - [`panel_info`](@ref)
  - [`panel_level_cells`](@ref)
"""
function panel_info_levels(io::IO, pnl::AssetPanel, g::CategoricalPanelField,
                           act::AbstractArray{Bool})::Nothing
    N = panel_axes(pnl)[end]
    lo = [minimum(sum(reshape(panel_level_cells(g, k, act), :, N); dims = 2))
          for k in eachindex(g.levels)]
    println(io, "  ", g.name, ": ", length(g.levels),
            " levels, by the smallest count of assets over the observations")
    for (label, a, b) in (("< 10", 0, 10), ("10 - 19", 10, 20), ("20 - 49", 20, 50),
                          (">= 50", 50, typemax(Int)))
        ls = g.levels[a .<= lo .< b]
        names = if length(ls) <= 6
            join(ls, ", ")
        else
            "$(join(view(ls, 1:4), ", ")), … +$(length(ls) - 4) more"
        end
        println(io, "    ", rpad(label, 8), ": ", length(ls), " levels",
                isempty(ls) ? "" : " ($names)")
    end
    return nothing
end
"""
    panel_info_fields(io::IO, pnl::AssetPanel, cs::AbstractVector) -> nothing

Print one line for each Panel Field of an Asset Panel: its kind, its cells, its missing share of all cells and of the active cells, and the count of assets whose active cells are all missing.

The numbers are the ones of [`DataFrames.describe(pnl::AssetPanel)`](@ref), with the shares as percentages.

# Algorithm

 1. Pad the name column to the longest Panel Field name.
 2. Print the header, then one line for each Panel Field from its counts.

# Arguments

  - `io`: The stream.
  - `pnl`: The Asset Panel.
  - `cs`: The counts of each Panel Field, see [`panel_field_coverage`](@ref).

# Returns

  - `nothing`.

# Related

  - [`panel_info`](@ref)
  - [`panel_field_coverage`](@ref)
  - [`panel_field_kind`](@ref)
  - [`panel_percent`](@ref)
"""
function panel_info_fields(io::IO, pnl::AssetPanel, cs::AbstractVector)::Nothing
    w = maximum(f -> length(f.name), pnl.pf; init = 5)
    println(io, "\nPanel Fields")
    println(io, "  ", rpad("field", w), "  ", rpad("kind", 11), lpad("cells", 7),
            lpad("missing %", 11), lpad("active cells", 14), lpad("active missing %", 18),
            lpad("assets missing", 16))
    for (f, c) in zip(pnl.pf, cs)
        println(io, "  ", rpad(f.name, w), "  ", rpad(string(panel_field_kind(f)), 11),
                lpad(c.cells, 7), lpad(panel_percent(c.missing / c.cells), 11),
                lpad(c.active, 14), lpad(panel_percent(c.active_missing / c.active), 18),
                lpad(c.assets_missing, 16))
    end
    return nothing
end
"""
    panel_info_header(io::IO, pnl::AssetPanel, ts::Option{<:AbstractVector},
                      cs::AbstractVector) -> nothing

Print the dimensions of an Asset Panel and the missing share of its cells, the first lines of [`panel_info`](@ref).

# Algorithm

 1. Check the length of `ts` against the observation axis of a time-varying panel.
 2. Print the count of observations, with the first and the last of `ts` when it is given, the count of assets, the count of Panel Fields and the count of cells.
 3. Add the counts of the Panel Fields, and print the missing share of all cells and of the active cells.

# Arguments

  - `io`: The stream.
  - `pnl`: The Asset Panel.
  - `ts`: The timestamps of the observations, or `nothing`. A static panel does not read it.
  - `cs`: The counts of each Panel Field, see [`panel_field_coverage`](@ref).

# Validation

  - `length(ts)` is the count of observations of a time-varying panel. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`panel_info`](@ref)
  - [`panel_field_coverage`](@ref)
  - [`Option`](@ref)
"""
function panel_info_header(io::IO, pnl::AssetPanel, ts::Option{<:AbstractVector},
                           cs::AbstractVector)::Nothing
    ax = panel_axes(pnl)
    static = isone(length(ax))
    @argcheck(static || isnothing(ts) || length(ts) == ax[1],
              DimensionMismatch("`ts` labels the observations of a time-varying panel, so it is as long as the panel's observation axis, got length(ts) = $(isnothing(ts) ? 0 : length(ts)) and $(ax[1]) observations"))
    span = static || isnothing(ts) || isempty(ts) ? "" : " ($(first(ts)) to $(last(ts)))"
    println(io, "Asset Panel")
    println(io, "  observations : ", static ? "none, the panel is static" : "$(ax[1])$span")
    println(io, "  assets       : ", ax[end])
    println(io, "  Panel Fields : ", length(pnl.pf))
    println(io, "  cells        : ", prod(ax), static ? "" : " (observations × assets)")
    println(io, "  missing      : ",
            panel_percent(sum(c.missing for c in cs; init = 0) /
                          sum(c.cells for c in cs; init = 0)), "% of the cells, ",
            panel_percent(sum(c.active_missing for c in cs; init = 0) /
                          sum(c.active for c in cs; init = 0)), "% of the active cells")
    return nothing
end
"""
    panel_info([io::IO,] pnl::AssetPanel; ts::Option{<:AbstractVector} = nothing) -> nothing

Print a report of an [`AssetPanel`](@ref): its dimensions, the coverage of its two universe masks, the missing share of each Panel Field, and the levels of each categorical Panel Field.

The panel holds no timestamps. Pass the timestamps of the [`ReturnsResult`](@ref) or the [`PricesResult`](@ref) that holds the panel as `ts`, and the report prints the first and the last. A missing cell is a cell that the builder filled, as [`DataFrames.describe(pnl::AssetPanel)`](@ref) states. The `show` of the panel is the tree of its fields, as for each Result, and this report does not replace it.

# Algorithm

 1. Count the cells of each Panel Field with [`panel_field_coverage`](@ref). Print the dimensions and the missing share of the cells with [`panel_info_header`](@ref).
 2. Print the coverage of the active mask with [`panel_info_mask`](@ref). Print the estimation mask the same way, or print that it equals the active mask.
 3. Print the counts of each Panel Field with [`panel_info_fields`](@ref), the numbers of [`DataFrames.describe(pnl::AssetPanel)`](@ref) with the shares as percentages.
 4. Print the levels of each categorical Panel Field with [`panel_info_levels`](@ref).

# Arguments

  - `io`: The stream, `stdout` when it is not given.
  - `pnl`: The Asset Panel.
  - `ts`: The timestamps of the observations, or `nothing`. A static panel does not read it.

# Validation

  - `length(ts)` is the count of observations of a time-varying panel. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [NumericPanelField(; name = \"mcap\",
                                                vals = [1.0 2.0; 3.0 4.0; 5.0 6.0],
                                                omsk = Bool[0 1; 1 1; 1 0]),
                              CategoricalPanelField(; name = \"sector\", levels = [\"Tech\", \"Energy\"],
                                                    codes = [1 2; 1 2; 1 2])],
                        amsk = Bool[1 1; 1 1; 1 0], emsk = Bool[1 1; 1 1; 1 0]);

julia> panel_info(pnl; ts = [\"2024-01\", \"2024-02\", \"2024-03\"])
Asset Panel
  observations : 3 (2024-01 to 2024-03)
  assets       : 2
  Panel Fields : 2
  cells        : 6 (observations × assets)
  missing      : 16.7% of the cells, 10.0% of the active cells

active mask
  in mask          : 5 / 6 cells (83.3%)
  assets per obs.  : min 1, median 2.0, max 2
  assets in mask   : 2 / 2
  duration (obs.)  : min 2, median 2.5, max 3

estimation mask: equal to the active mask

Panel Fields
  field   kind         cells  missing %  active cells  active missing %  assets missing
  mcap    numeric          6       33.3             5              20.0               0
  sector  categorical      6        0.0             5               0.0               0

categorical Panel Fields
  sector: 2 levels, by the smallest count of assets over the observations
    < 10    : 2 levels (Tech, Energy)
    10 - 19 : 0 levels
    20 - 49 : 0 levels
    >= 50   : 0 levels
```

# Related

  - [`AssetPanel`](@ref)
  - [`DataFrames.describe(pnl::AssetPanel)`](@ref)
  - [`panel_align_active`](@ref)
  - [`panel_dataframe`](@ref)
  - [`panel_info_header`](@ref)
  - [`panel_info_fields`](@ref)
  - [`panel_info_mask`](@ref)
  - [`panel_info_levels`](@ref)
  - [`Option`](@ref)
"""
function panel_info(io::IO, pnl::AssetPanel;
                    ts::Option{<:AbstractVector} = nothing)::Nothing
    act = panel_active_cells(pnl)
    cs = [panel_field_coverage(f, act) for f in pnl.pf]
    panel_info_header(io, pnl, ts, cs)
    panel_info_mask(io, "active mask", pnl.amsk)
    if !panel_is_static(pnl) && isequal(pnl.emsk, pnl.amsk)
        println(io, "\nestimation mask: equal to the active mask")
    else
        panel_info_mask(io, "estimation mask", pnl.emsk)
    end
    panel_info_fields(io, pnl, cs)
    gs = filter(g -> g isa CategoricalPanelField, pnl.pf)
    if !isempty(gs)
        println(io, "\ncategorical Panel Fields")
    end
    for g in gs
        panel_info_levels(io, pnl, g, act)
    end
    return nothing
end
function panel_info(pnl::AssetPanel; kwargs...)::Nothing
    return panel_info(stdout, pnl; kwargs...)
end
"""
    panel_observed_cells(f::AbstractPanelField) -> Option{AbstractArray{Bool}}

Return where a Panel Field holds an observed value, over its observation and asset axes.

A tensor Panel Field holds an observed value for a pair of observation and asset when each of its labels is observed there. A Panel Field with no observed mask cannot blank, so it is observed at each cell, and the function returns `nothing` for it.

# Algorithm

 1. The observed mask is `nothing`: return `nothing`.
 2. Otherwise return the pairs that [`panel_field_missing`](@ref) counts no missing cell for.

# Arguments

  - `f`: The Panel Field.

# Returns

  - `obs::Option{AbstractArray{Bool}}`: The observed pairs, or `nothing` when each pair is observed.

# Related

  - [`panel_field_missing`](@ref)
  - [`panel_align_active`](@ref)
  - [`Option`](@ref)
"""
function panel_observed_cells(f::AbstractPanelField)
    return isnothing(f.omsk) ? nothing : iszero.(panel_field_missing(f))
end
"""
    panel_align_active(pnl::AssetPanel, fields) -> NamedTuple

Remove the leading active cells of each asset of an [`AssetPanel`](@ref) until each named Panel Field holds an observed value.

An asset enters the universe of a panel when its active mask is true, and a field of the asset can start later, for example a book value that is first reported months after the listing. The function moves the start of each asset to the first active observation where each named Panel Field is observed. Only the leading cells are removed: a missing cell after that observation stays active. An asset with no such observation leaves the universe. The estimation mask is intersected with the new active mask, so it stays a subset. An observation can be left with no active asset, and the panel admits it, as it admits a view with no asset.

After the alignment, the first active cell of each asset is observed in each named Panel Field. A second call removes nothing, and the order of `fields` does not change the result.

A tensor Panel Field is observed at a pair of observation and asset when each of its labels is observed there. A Panel Field with no observed mask cannot blank, and is observed at each cell. The Panel Fields do not change: a removed cell keeps its value and its observed mask.

# Algorithm

 1. Find each named Panel Field with [`panel_field`](@ref), and read where it is observed with [`panel_observed_cells`](@ref).
 2. For each asset, find the first observation that is active and where each named Panel Field is observed. Remove the active cells before it, or each active cell when there is no such observation, and count them.
 3. Intersect the estimation mask with the new active mask, and make the `AssetPanel` of the Panel Fields and the two new masks.

# Arguments

  - `pnl`: The Asset Panel.
  - `fields`: The name of one Panel Field, or a collection of names. An empty collection removes nothing.

# Validation

  - `pnl` is time-varying. Raises an `ArgumentError`.
  - The panel holds each named Panel Field. Raises a `KeyError`.

# Returns

  - `res::NamedTuple`: `pnl`, the aligned Asset Panel; `n`, the count of cells removed from the active mask.

# Examples

```jldoctest
julia> pnl = AssetPanel(;
                        pf = [NumericPanelField(; name = \"book\",
                                                vals = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0],
                                                omsk = Bool[0 1; 0 0; 1 1; 1 0])],
                        amsk = trues(4, 2), emsk = trues(4, 2));

julia> res = panel_align_active(pnl, \"book\");

julia> res.n
2

julia> res.pnl.amsk
4×2 BitMatrix:
 0  1
 0  1
 1  1
 1  1
```

# Related

  - [`AssetPanel`](@ref)
  - [`panel_observed_cells`](@ref)
  - [`DataFrames.describe(pnl::AssetPanel)`](@ref)
  - [`panel_info`](@ref)
  - [`port_opt_view`](@ref)
"""
function panel_align_active(pnl::AssetPanel, fields)
    @argcheck(!panel_is_static(pnl),
              ArgumentError("an alignment moves the start of each asset in the active mask, and a static Asset Panel has no active mask"))
    names = isa(fields, AbstractString) ? [fields] : fields
    obs = [panel_observed_cells(panel_field(pnl, name)) for name in names]
    amsk = BitMatrix(pnl.amsk)
    T, N = size(amsk)
    n = 0
    for j in 1:N
        s = findfirst(t -> amsk[t, j] && all(o -> isnothing(o) || o[t, j], obs), 1:T)
        for t in 1:(isnothing(s) ? T : s - 1)
            n += amsk[t, j]
            amsk[t, j] = false
        end
    end
    return (; pnl = AssetPanel(; pf = pnl.pf, amsk = amsk, emsk = pnl.emsk .& amsk), n)
end

export panel_info, panel_align_active
