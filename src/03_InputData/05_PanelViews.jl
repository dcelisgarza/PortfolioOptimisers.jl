"""
    feature_row_indices(pnl::Nothing, ts_new, ts_old) -> Colon
    feature_row_indices(pnl::AssetPanel, ts_new, ts_old) -> Union{Colon, VecInt}

Recover the positional row indices of a time-varying [`AssetPanel`](@ref) from a timestamp window.

A Panel Field holds a plain array, so its observation axis follows the clock of the input data by position and not by timestamp. A routine that selects rows of `X` by timestamp calls this function to match the surviving timestamps back into the original clock, and the panel keeps the rows it finds. A surviving timestamp that is absent from that clock throws. Such a timestamp comes from an outer join that adds a row `X` never had. A positional slice past such a timestamp pairs each asset with the values of another period.

Two kinds of caller use it. At the price level, the clock is `TimeSeries.timestamp(X)` and the selection is a timestamp window. Where cross-validation assembles its folds, the clock is `ReturnsResult.ts` and the selection is a fold. There [`fold_row_indices`](@ref) recovers the rows of each fold from the timestamps that the fold's view of the returns holds. This is why `ts` must be unique. It is the key of the observation axis, and a repeated timestamp matches only its first position.

A static panel and an absent panel have no observation axis, so they return `Colon` and read no timestamp.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`, or static: return `Colon()`. Neither has an observation axis, so the function reads no timestamp.
 2. `pnl` is time-varying: match `ts_new` into `ts_old` with [`matched_row_indices`](@ref), giving the rows. That function throws when the selection kept no timestamp, or when a surviving timestamp is absent from the original clock.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `ts_new`: The timestamps that survived the selection.
  - `ts_old`: The timestamps of the clock that the panel's observation axis follows.

# Validation

When the panel is time-varying:

  - `ts_new` is not `nothing`. Raises an `ArgumentError`.
  - Every entry of `ts_new` appears in `ts_old`. Raises an `ArgumentError`.

# Returns

  - `rows::Union{Colon, Vector{Int}}`: `Colon()` for a static or absent panel. Otherwise the position of each surviving timestamp in `ts_old`, in the order of `ts_new`.

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

[`feature_row_indices`](@ref) calls it for a time-varying [`AssetPanel`](@ref), and [`listing_span_view`](@ref) calls it for a Listing Span. Both hold their observation axis parallel to the clock of the input data by position, so both recover their rows the same way.

# Algorithm

The method that Julia selects is the algorithm.

 1. `ts_new` is `nothing`: throw. The selection kept no timestamp, so the function cannot name the rows to keep.
 2. Otherwise match `ts_new` into `ts_old` with `indexin`, giving `rows`. A timestamp that appears twice in `ts_old` matches its first position.
 3. Find the first entry of `rows` that `indexin` did not match, giving `missed`, and throw when there is one.
 4. Return `rows` as a `Vector{Int}`.

# Arguments

  - `ts_new`: The timestamps that survived the selection.
  - `ts_old`: The timestamps of the clock that the observation axis follows.

# Validation

  - `ts_new` is not `nothing`. Raises an `ArgumentError`.
  - Every entry of `ts_new` appears in `ts_old`. Raises an `ArgumentError`.

# Returns

  - `rows::Vector{Int}`: The position of each surviving timestamp in `ts_old`, in the order of `ts_new`. An empty `ts_new` gives an empty vector.

# Related

  - [`feature_row_indices`](@ref)
  - [`listing_span_view`](@ref)
  - [`AssetPanel`](@ref)
  - [`prices_to_returns`](@ref)
"""
function matched_row_indices(::Nothing, ::Any)
    return throw(ArgumentError("a time-varying Asset Panel or Listing Span holds its observation axis parallel to a clock, but no timestamps survived the selection, so the rows to keep cannot be recovered. Supply the surviving timestamps, or pass a static Asset Panel, which has no observation axis."))
end
function matched_row_indices(ts_new, ts_old)
    rows = indexin(ts_new, ts_old)
    missed = findfirst(isnothing, rows)
    @argcheck(isnothing(missed),
              ArgumentError("a time-varying Asset Panel or Listing Span holds its observation axis parallel to a clock, but the timestamp $(ts_new[missed]) selected here is absent from that clock, so the row it corresponds to cannot be recovered. This happens when the surviving timestamps are not a subset of the original clock, for example after an outer join that adds rows the asset prices never had. Pass a static Asset Panel, or align the panel to the clock first."))
    return Vector{Int}(rows)
end
"""
    panel_feature_names(pnl::Nothing) -> nothing
    panel_feature_names(pnl::AssetPanel) -> Vector{String}

Name the columns an [`AssetPanel`](@ref) derives, without building the Feature Matrix.

A consumer that needs only the column names calls this function, which reads no value of the panel.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): read its Panel Fields in order into `nz`. For each field, append the names of its value columns, then the names of its observed-mask columns when the field has an observed mask.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.

# Returns

  - `nz::Option{Vector{String}}`: One name per derived column, in the column order of [`panel_feature_matrix`](@ref), or `nothing`. A panel with no Panel Field gives an empty vector.

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
    asset_panel_view(pnl::Nothing, i, j, nx) -> nothing
    asset_panel_view(pnl::AssetPanel, i, j, nx) -> AssetPanel

View the [`AssetPanel`](@ref) of a `PricesResult` or a `ReturnsResult`, or return `nothing` when the input data holds none.

[`port_opt_view`](@ref) of a [`PricesResult`](@ref) or of a [`ReturnsResult`](@ref) calls it, and [`MissingDataFilter`](@ref) reaches it through that view. So the `nothing` case has one method, and no call site needs a branch for it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `pnl` is `nothing`: return `nothing`.
 2. `pnl` is an [`AssetPanel`](@ref): return [`port_opt_view`](@ref) of it.

# Arguments

  - `pnl`: The Asset Panel, or `nothing`.
  - `i`: Observation index. A static panel has no observation axis and ignores it.
  - `j`: Asset index.
  - `nx`: The asset names, or `nothing`. [`port_opt_view`](@ref) uses them to cut a tensor Panel Field whose labels are the asset names on its label axis too. With `nothing`, every label axis stays whole.

# Returns

  - `pnl′::Option{<:AssetPanel}`: The Asset Panel over the selected observations and assets, or `nothing`.

# Related

  - [`AssetPanel`](@ref)
  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
  - [`PricesResult`](@ref)
  - [`MissingDataFilter`](@ref)
"""
function asset_panel_view(::Nothing, ::Any, ::Any, ::Any)
    return nothing
end
function asset_panel_view(pnl::AssetPanel, i, j, nx::Option{<:VecStr})
    return port_opt_view(pnl, i, j, nx)
end
"""
    label_positions(names, sel, what::AbstractString) -> typeof(sel)
    label_positions(names, sel::VecDate, what::AbstractString) -> Union{}
    label_positions(names::Nothing, sel::VecStr, what::AbstractString) -> Union{}
    label_positions(names::AbstractVector, sel::VecStr, what::AbstractString) -> Vector{Int}

Turn the names that select assets or factors in a view into the positions of those names.

[`port_opt_view`](@ref) of a [`ReturnsResult`](@ref) and of a [`PricesResult`](@ref) calls it on each index that names assets or factors, so a caller can select them by name or by position. An integer index, a range and a `Colon` pass through unchanged.

A name that the input data does not hold is refused. An integer past the end of an axis throws, and a name that selects nothing is the same mistake. If the view dropped the name, a misspelt ticker would give a smaller universe and no error.

# Algorithm

The method that Julia selects is the algorithm.

 1. `sel` is not a vector of names or of timestamps: return it unchanged.
 2. `sel` is a vector of timestamps: throw. A timestamp selects an observation, and this index selects `what`.
 3. `sel` is a vector of names and `names` is `nothing`: throw. The input data carries no names to match.
 4. `sel` is a vector of names: match it into `names` with `indexin`, giving the positions. Throw when a name has no match, and name every such entry.

# Arguments

  - `names`: The names of the axis that `sel` selects, or `nothing`.
  - `sel`: The index of the view: names, integer positions, a range or a `Colon`.
  - `what`: The name of the axis, such as `"asset"`, for the messages.

# Validation

  - `sel` is not a vector of timestamps. Raises an `ArgumentError`.
  - `names` is not `nothing` when `sel` is a vector of names. Raises an `ArgumentError`.
  - Every entry of `sel` is in `names`. Raises an `ArgumentError`.

# Returns

  - `idx`: `sel` unchanged, or the position of each name of `sel` in `names`, in the order of `sel`.

# Related

  - [`port_opt_view`](@ref)
  - [`timestamp_positions`](@ref)
  - [`did_you_mean`](@ref)
  - [`VecStr`](@ref)
  - [`VecDate`](@ref)
"""
function label_positions(::Any, sel, ::AbstractString)
    return sel
end
function label_positions(::Any, sel::VecDate, what::AbstractString)
    return throw(ArgumentError("a timestamp selects an observation, but this index of the view selects the $(what)s. Pass the timestamps as the observation index, which is the first index of port_opt_view(rd, observations, assets) and of port_opt_view(pr, observations, assets)."))
end
function label_positions(::Nothing, sel::VecStr, what::AbstractString)
    return throw(ArgumentError("the view selects the $(what)s by name, but the input data carries no $what names to match. Select the $(what)s by position, or build the input data with its $what names. Got\nsel => $(sel)"))
end
function label_positions(names::AbstractVector, sel::VecStr, what::AbstractString)
    pos = indexin(sel, names)
    missed = findall(isnothing, pos)
    @argcheck(isempty(missed),
              ArgumentError("the view selects the $(what)s by name, and $(length(missed)) name(s) are not among the $(length(names)) $what names of the input data: $(join(sel[missed], ", ")). A view refuses a name that it cannot find, because a dropped name gives a smaller universe with no error." *
                            did_you_mean(sel[first(missed)], names)))
    return Vector{Int}(pos)
end
"""
    timestamp_positions(ts, i) -> typeof(i)
    timestamp_positions(ts, i::VecStr) -> Union{}
    timestamp_positions(ts::Nothing, i::VecDate) -> Union{}
    timestamp_positions(ts::AbstractVector, i::VecDate) -> Vector{Int}

Turn the timestamps that select observations in a view of a [`ReturnsResult`](@ref) into the positions of those observations.

[`port_opt_view`](@ref) of a `ReturnsResult` calls it on its observation index, so a caller can select the observations by timestamp or by position. An integer index, a range and a `Colon` pass through unchanged.

A timestamp selects the observations that the clock holds, as the timestamp view of a [`PricesResult`](@ref) does. The rows come back in clock order whatever the order of `i`, and a timestamp that the clock does not hold selects no row. So a calendar range selects the trading days inside it. A window that keeps no row is refused by the [`ReturnsResult`](@ref) constructor, which refuses empty returns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `i` is not a vector of timestamps or of names: return it unchanged.
 2. `i` is a vector of names: throw. A name selects an asset or a factor, and this index selects observations.
 3. `i` is a vector of timestamps and `ts` is `nothing`: throw. The returns carry no timestamps to match.
 4. `i` is a vector of timestamps: sort it, and keep the position of each timestamp of `ts` that the sorted vector holds, with `insorted`.

# Arguments

  - `ts`: The timestamps of the returns, or `nothing`.
  - `i`: The observation index of the view: timestamps, integer positions, a range or a `Colon`.

# Validation

  - `i` is not a vector of names. Raises an `ArgumentError`.
  - `ts` is not `nothing` when `i` is a vector of timestamps. Raises an `ArgumentError`.

# Returns

  - `idx`: `i` unchanged, or the positions in `ts` of the timestamps that `i` holds, in clock order.

# Related

  - [`port_opt_view`](@ref)
  - [`label_positions`](@ref)
  - [`ReturnsResult`](@ref)
  - [`VecDate`](@ref)
  - [`VecStr`](@ref)
"""
function timestamp_positions(::Any, i)
    return i
end
function timestamp_positions(::Any, i::VecStr)
    return throw(ArgumentError("a name selects an asset or a factor, but the first index of port_opt_view(rd, observations, assets, factors) selects the observations. Pass the asset names as the second index and the factor names as the third. Got\ni => $(i)"))
end
function timestamp_positions(::Nothing, i::VecDate)
    return throw(ArgumentError("the view selects the observations by timestamp, but the ReturnsResult carries no timestamps `ts` to match. Select the observations by position, or build the ReturnsResult with `ts`."))
end
function timestamp_positions(ts::AbstractVector, i::VecDate)
    s = sort(i)
    return findall(t -> insorted(t, s), ts)
end
