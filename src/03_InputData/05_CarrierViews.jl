"""
    feature_row_indices(pnl::Nothing, ts_new, ts_old) -> Colon
    feature_row_indices(pnl::AssetPanel, ts_new, ts_old) -> Union{Colon, VecInt}

Recover the positional row indices of a time-varying [`AssetPanel`](@ref) from a timestamp window.

A Panel Field holds a plain array, so its observation axis is parallel to the carrier's clock positionally rather than aligned by timestamp. Whenever a routine selects rows of `X` by timestamp, the surviving timestamps are matched back into the original clock to recover the rows the panel must keep. A surviving timestamp absent from that clock throws: it means the row bookkeeping has been broken (a synthesised timestamp, or an outer join that introduced a row `X` never had), and slicing the panel positionally from there would silently pair each asset with another period's values.

Two sites use it. At **price level** the clock is `TimeSeries.timestamp(X)` and the selection is a timestamp window. At the **cross-validation assembly seam** the clock is `ReturnsResult.ts` and the selection is a fold: [`fold_row_indices`](@ref) recovers a fold's rows from the timestamps its view of the returns already carries, which is why `ts` must be unique — it *keys* the observation axis rather than merely labelling it.

The static and absent shapes have no observation axis, so they return `Colon` and cost nothing.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`, or static: return `Colon()`. Neither has an observation axis, so there is no row to recover and the timestamps are not read.
 2. `pnl` is time-varying: match `ts_new` into `ts_old` with [`matched_row_indices`](@ref), which throws when the selection kept no timestamp, or when a surviving timestamp is absent from the original clock.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `ts_new`: Timestamps surviving the selection.
  - `ts_old`: Timestamps of the clock the panel's observation axis is parallel to.

# Validation

  - `ts_new` is not `nothing` when the panel is time-varying.
  - Every entry of `ts_new` appears in `ts_old`.

# Returns

  - `Colon` for a static or absent panel; otherwise the row indices, as a `Vector{Int}`.

# Related

  - [`AssetPanel`](@ref)
  - [`matched_row_indices`](@ref)
  - [`PricesResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`fold_row_indices`](@ref)
"""
function feature_row_indices(::Nothing, ::Any, ::Any)
    return Colon()
end
function feature_row_indices(pnl::AssetPanel, ts_new, ts_old)
    return panel_is_static(pnl) ? Colon() : matched_row_indices(ts_new, ts_old)
end
"""
    matched_row_indices(ts_new::Nothing, ts_old) -> Union{}
    matched_row_indices(ts_new, ts_old) -> Vector{Int}

Match the surviving timestamps back into the original clock, and return the rows they hold.

The body [`feature_row_indices`](@ref) shares between the time-varying feature matrix and the time-varying [`AssetPanel`](@ref): both hold their observation axis parallel to the carrier's clock positionally, so both recover their rows the same way.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ts_new` is `nothing`: throw. The selection kept no timestamp, so the rows to keep cannot be named.
 2. Otherwise match `ts_new` into `ts_old` with `indexin`, check that every surviving timestamp was found, and return the positions as a `Vector{Int}`.

# Arguments

  - `ts_new`: Timestamps that survived the selection.
  - `ts_old`: Timestamps of the clock the observation axis is parallel to.

# Validation

  - `ts_new` is not `nothing`. Raises an `ArgumentError`.
  - Every entry of `ts_new` appears in `ts_old`. Raises an `ArgumentError`.

# Returns

  - `rows::Vector{Int}`: The position each surviving timestamp holds in the original clock.

# Related

  - [`feature_row_indices`](@ref)
  - [`AssetPanel`](@ref)
  - [`prices_to_returns`](@ref)
"""
function matched_row_indices(::Nothing, ::Any)
    return throw(ArgumentError("a time-varying feature axis has its observation axis parallel to the price timestamps, but no timestamps survived the conversion, so the rows to keep cannot be recovered. Pass a static Asset Panel instead."))
end
function matched_row_indices(ts_new, ts_old)
    rows = indexin(ts_new, ts_old)
    missed = findfirst(isnothing, rows)
    @argcheck(isnothing(missed),
              ArgumentError("a time-varying feature axis has its observation axis parallel to the price timestamps, but the timestamp $(ts_new[missed]) selected here is absent from them, so the row it corresponds to cannot be recovered. This happens when the surviving timestamps are not a subset of the original clock — a `collapse_args` timestamp function that synthesises timestamps, or an outer join that introduced rows the asset prices never had. Pass a static Asset Panel, or align the feature axis to the price clock first."))
    return Vector{Int}(rows)
end
"""
    panel_feature_names(pnl::Nothing) -> nothing
    panel_feature_names(pnl::AssetPanel) -> Vector{String}

Name the columns an [`AssetPanel`](@ref) derives, without building the Feature Matrix.

A consumer that needs the column names alone reads them here, and the values are not stacked to answer it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): walk its Panel Fields, appending each one's value column names and then its observed-mask column names.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.

# Returns

  - `nz::Option{Vector{String}}`: One name per derived column, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`panel_feature_matrix`](@ref)
"""
function panel_feature_names(::Nothing)
    return nothing
end
function panel_feature_names(pnl::AssetPanel)
    nz = String[]
    for f in pnl.pf
        append!(nz, panel_field_labels(f))
        if !isnothing(f.omsk)
            append!(nz, panel_field_observed_labels(f))
        end
    end
    return nz
end
"""
    panel_carrier_view(pnl::Nothing, i, j, nx) -> nothing
    panel_carrier_view(pnl::AssetPanel, i, j, nx) -> AssetPanel

View a carrier's [`AssetPanel`](@ref), or return `nothing` when the carrier holds none.

The one-line wrapper every carrier view goes through, so the `nothing` case is written once rather than at each of the six sites that slice a panel.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): return [`port_opt_view`](@ref) of it.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `i`: Observation index.
  - `j`: Asset index.
  - `sq`: Whether the Panel Fields are the assets.

# Returns

  - An Asset Panel over the selected observations and assets, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
  - [`PricesResult`](@ref)
"""
function panel_carrier_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function panel_carrier_view(pnl::AssetPanel, i, j, nx::Option{<:VecStr})
    return port_opt_view(pnl, i, j, nx)
end
