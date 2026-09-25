"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for returns-level preprocessing estimators that restrict the asset universe.

An asset selector decides on the training window which asset columns to keep, and that decision is its fitted state. [`apply_preprocessing`](@ref) replays the fitted universe on each later window. The selection reads the training window alone, and the selector never selects again on a test window, so a selector is safe inside cross-validation.

The whole family shares one method of [`fit_preprocessing`](@ref) and one of [`apply_preprocessing`](@ref). That method of [`fit_preprocessing`](@ref) reduces the training window to its Coverage Universe before it calls [`select_assets`](@ref). A selector therefore ranks only the assets that are live at every row of the window, and never an asset that is not yet listed, is delisted, or misses a quote. It needs no finiteness guard of its own, and [`CompleteAssetSelector`](@ref) keeps every column of the reduced window.

A selector drops asset columns and never an observation. [`MissingDataFilter`](@ref) drops observations, from a price window.

# Interfaces

To implement a new asset selector, subtype `AbstractAssetSelector` with its parameters as fields of the struct, and implement the following method:

  - `select_assets(sel::MyAssetSelector, rd::AbstractReturnsResult) -> BitVector`: Return the keep-mask over the asset columns of the reduced training window.

## Arguments

  - `sel`: The concrete asset selector instance.
  - `rd`: The training window, reduced to its Coverage Universe.

## Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for every asset to keep. It holds one entry per asset column of `rd`, and at least one entry is `true`.

# Related

  - [`select_assets`](@ref)
  - [`AssetSelectorResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractAssetSelector <: AbstractReturnsPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of any [`AbstractAssetSelector`](@ref).

It holds the names of the assets that the selector kept on the training window. Every selector stores the same thing, so one result type serves the whole family.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`select_assets`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
@concrete struct AssetSelectorResult <: AbstractReturnsPreprocessingResult
    """
    Names of the assets that the selector kept on the training window, in the column order of that window. This is the fitted universe.
    """
    nx
end
"""
    select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult) -> BitVector

Return the keep-mask over the asset columns of `rd`.

A concrete [`AbstractAssetSelector`](@ref) must implement this method, and no other. [`fit_preprocessing`](@ref) calls it on the Coverage Universe of the training window, and [`apply_preprocessing`](@ref) replays the result on every later window.

`rd` is the reduced carrier, so every column of it is finite and active at every row. A selector needs no finiteness guard. A score that is not finite on a live column is a defect of the score, and [`asset_scores`](@ref) refuses it.

A selector reads `rd.nx` and the `observations × assets` matrix `rd.X`. The reduction reads `rd.pnl`, and [`ClusterGroups`](@ref) gives the whole carrier to [`clusterise`](@ref), so the family reads the fields `nx`, `X` and `pnl` of [`AbstractReturnsResult`](@ref). A selector never sees a prior result.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training window, reduced to its Coverage Universe.

# Validation

  - A subtype that does not implement this method throws an `ArgumentError` that names the subtype.

# Returns

  - `keep::BitVector`: Mask `assets × 1` that is `true` for each asset column to keep, with `length(keep) == size(rd.X, 2)`.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`fit_preprocessing`](@ref)
  - [`coverage_reduction(rd::AbstractReturnsResult)`](@ref)
"""
function select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult)
    return throw(ArgumentError("$(typeof(sel)) subtypes AbstractAssetSelector but does not implement select_assets. Extension authors: every AbstractAssetSelector must define select_assets(sel, rd) returning a keep-mask over the asset columns of rd."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit any [`AbstractAssetSelector`](@ref) by recording the asset universe [`select_assets`](@ref) keeps.

This one method fits every asset selector, and it applies the Coverage Universe. It reduces the window before it calls [`select_assets`](@ref), so every selector ranks only the assets that are live at every row of the training window. A new selector gets the reduction with no code of its own. A window with no live asset throws an `IsEmptyError` from [`coverage_mask`](@ref), before any selector runs.

# Algorithm

 1. Reduce the training window to its Coverage Universe with [`coverage_reduction(rd::AbstractReturnsResult)`](@ref).
 2. Call [`select_assets`](@ref) on the reduced window, giving the keep-mask `keep`.
 3. Check that `keep` holds one entry per asset column of the reduced window.
 4. Check that `keep` keeps at least one asset.
 5. Return an [`AssetSelectorResult`](@ref) holding the names of the kept assets, in the column order of the window.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Validation

  - The carrier must hold an `observations × assets` returns matrix. A carrier that collapsed the asset axis, such as [`PredictionReturnsResult`](@ref), matches no method of the reduction and throws a `MethodError`.
  - At least one asset must be in the Coverage Universe of the training window, else `IsEmptyError`.
  - [`select_assets`](@ref) must return one entry per asset column of the reduced window, else `DimensionMismatch`.
  - The mask must keep at least one asset, else `IsEmptyError`, so no later step receives a problem with no asset.

# Returns

  - `res::AssetSelectorResult`: The fitted asset universe, as names. A name that the reduction dropped is absent from it. [`apply_preprocessing`](@ref) finds each name in the window that it replays on.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`AssetSelectorResult`](@ref)
  - [`apply_preprocessing`](@ref)
"""
function fit_preprocessing(sel::AbstractAssetSelector,
                           rd::AbstractReturnsResult)::AssetSelectorResult
    _, rdc = coverage_reduction(rd)
    keep = select_assets(sel, rdc)
    @argcheck(length(keep) == size(rdc.X, 2),
              DimensionMismatch("select_assets for a $(typeof(sel)) returned a mask of length $(length(keep)) for the $(size(rdc.X, 2)) asset columns of the Coverage Universe of the training window"))
    @argcheck(any(keep),
              IsEmptyError("a $(typeof(sel)) selects no assets from the Coverage Universe of the training window; loosen its configuration"))
    return AssetSelectorResult(collect(rdc.nx[keep]))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Replay a fitted asset universe on a data window.

The result holds the kept columns in the fitted order, not in the column order of the window. The last step computes its weights over the training universe, and `assert_universe_aligned` compares the two name vectors entry by entry.

# Algorithm

 1. For each fitted asset name, in fitted order, find the column of the window that carries it, and record that position in `idx`.
 2. Check that the name is present. A name the window does not carry throws.
 3. Return a [`port_opt_view`](@ref) of the window at `idx`. The positions are in fitted order, so the view reorders the window's columns when the two orders differ.

# Arguments

  - `res`: The fitted asset universe.
  - `rd`: The data window to transform.

# Validation

  - Every fitted asset name must be in the window, else `ArgumentError`, so the universe never shrinks without an error. The message names the missing asset and the size of the window through [`unknown_variable_msg`](@ref), and it does not list the whole universe.

# Returns

  - `rd′::AbstractReturnsResult`: The window restricted to the fitted universe, in fitted order.

# Related

  - [`AssetSelectorResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function apply_preprocessing(res::AssetSelectorResult, rd::AbstractReturnsResult)
    idx = Vector{Int}(undef, length(res.nx))
    for (k, name) in pairs(res.nx)
        j = findfirst(==(name), rd.nx)
        @argcheck(!isnothing(j),
                  ArgumentError(unknown_variable_msg(name, rd.nx, :nx;
                                                     consequence = "the window must contain the whole fitted universe")))
        idx[k] = j
    end
    return port_opt_view(rd, idx)
end
export AssetSelectorResult
