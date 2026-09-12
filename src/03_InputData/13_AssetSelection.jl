"""
    find_complete_indices(X::AbstractMatrix; dims::Int = 1) -> VecInt

Return the indices of columns (or rows) in matrix `X` that do not contain any missing or NaN values.

This function scans the specified dimension of the input matrix and returns the indices of columns (or rows) that are complete, i.e., contain no `missing` or `NaN` values.

Internal machinery — the caller-facing form is [`CompleteAssetSelector`](@ref), which wraps the `dims = 1` (complete-column) mode as a fit/apply estimator. The `dims = 2` (complete-row) mode has no estimator form: dropping observations is a price-level concern ([`MissingDataFilter`](@ref)).

# Algorithm

 1. Orient `X` with `dims_oriented`, so that the axis to test is axis 2 in both modes. `dims = 2` transposes the matrix, and `dims = 1` leaves it alone.
 2. Read the column count `N` of the oriented matrix.
 3. For each column of the oriented matrix, test whether it holds a `missing` entry or a `NaN` entry. Collect the positions of the columns that do, giving `to_remove`. One entry is enough to remove the whole column.
 4. Return `setdiff(1:N, to_remove)`, the positions of the complete columns, in ascending order.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - `dims in (1, 2)`.

# Returns

  - `res::VecInt`: Indices of columns (or rows) in `X` that are complete.

# Examples

```jldoctest
julia> X = [1.0 2.0 NaN; 4.0 missing 6.0];

julia> PortfolioOptimisers.find_complete_indices(X)
1-element Vector{Int64}:
 1

julia> PortfolioOptimisers.find_complete_indices(X; dims = 2)
Int64[]
```

# Related

  - [`CompleteAssetSelector`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`prices_to_returns`](@ref)
"""
function find_complete_indices(X::AbstractMatrix; dims::Int = 1)
    X = dims_oriented(dims, X)
    N = size(X, 2)
    to_remove = Vector{Int}(undef, 0)
    for i in axes(X, 2)
        if any(ismissing, X[:, i]) || any(isnan, X[:, i])
            push!(to_remove, i)
        end
    end
    return setdiff(1:N, to_remove)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for returns-level preprocessing estimators that restrict the asset universe.

An asset selector answers one question on the training window — *which asset columns survive?* — and that answer is its fitted state. [`apply_preprocessing`](@ref) replays the fitted universe on unseen windows, so a selector is safe inside cross-validation: the selection is made on train data alone and never re-decided on test data.

Concrete subtypes implement a single method, [`select_assets`](@ref); the family shares one [`fit_preprocessing`](@ref) and one [`apply_preprocessing`](@ref).

The funnel reduces the training window to its **Coverage Universe** before it calls [`select_assets`](@ref), so a selector ranks among the assets that are live throughout that window and never among an asset that is not yet listed, is delisted, or is missing a quote. A selector therefore needs no finiteness guard of its own, and [`CompleteAssetSelector`](@ref) is the identity on the reduced window.

Selectors restrict *columns only*. Observation filtering is a price-level concern ([`MissingDataFilter`](@ref)), because a fitted transformation cannot decide which rows of an unseen window to drop without breaking the weights/returns alignment `assert_universe_aligned` enforces.

See `docs/adr/0029-asset-selection-is-returns-preprocessing.md` for the design rationale.

# Related

  - [`select_assets`](@ref)
  - [`AssetSelectorResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractAssetSelector <: AbstractReturnsPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of any [`AbstractAssetSelector`](@ref).

Carries the asset universe selected on the training window. One result type serves the whole family: every selector differs in *how* it chooses the universe, never in what it stores.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`select_assets`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
@concrete struct AssetSelectorResult <: AbstractReturnsPreprocessingResult
    """
    Names of the assets that survived the training window, in their original column order (the fitted universe).
    """
    nx
end
"""
    select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult) -> BitVector

Return the keep-mask over the asset columns of `rd`.

This is the single method a concrete [`AbstractAssetSelector`](@ref) must implement. It is called by [`fit_preprocessing`](@ref) on the **Coverage Universe of the training window** only; the resulting universe is then replayed on every later window by [`apply_preprocessing`](@ref).

`rd` is the *reduced* carrier, so every column it carries is finite at every row and active at every row of the panel. A selector ranks among live assets alone, and it needs no finiteness guard: a non-finite score computed from a live column is a defect of the measure, which is why [`asset_scores`](@ref) keeps its refusal.

`rd` is read for `nx` and an `observations × assets` `X`; the funnel itself reads `rd.pnl`, and [`ClusterGroups`](@ref) reads it too, so the implicit contract of the family is `{nx, X, pnl}` (see [`AbstractReturnsResult`](@ref)). A selector is fitted from returns data alone and never sees a prior result, so it reads the data carrier and nothing else.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Returns

  - `keep::BitVector`: `true` for each asset column to retain, `length(keep) == size(rd.X, 2)`, over the reduced window.

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

This is the one funnel of the family, and it is where the Coverage Universe is applied. The window is reduced first, so **every** selector ranks among the assets that are live throughout the training window: a score, a redundancy and a rule all read live columns alone, and a new selector cannot forget the rule. An all-dead window throws an `IsEmptyError` where the mask is derived, so the refusal is [`coverage_mask`](@ref)'s.

# Algorithm

 1. Reduce the training window to its Coverage Universe with [`coverage_reduction(rd::AbstractReturnsResult)`](@ref).
 2. Call [`select_assets`](@ref) on the reduced window, giving the keep-mask `keep`.
 3. Check that `keep` holds one entry per asset column of the reduced window.
 4. Check that `keep` keeps at least one asset.
 5. Return an [`AssetSelectorResult`](@ref) holding the names of the kept assets, in their original column order.

The result records **names**, so the expansion is free: [`apply_preprocessing`](@ref) finds each name in the window it replays on, and a name the reduction dropped is simply absent from the fitted universe.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Validation

  - The carrier must hold an `observations × assets` returns matrix; one that collapsed the asset axis matches no method of the reduction and is named by a `MethodError`.
  - At least one asset must be in the Coverage Universe of the training window.
  - `select_assets` must return a mask whose length matches the number of asset columns of the reduced window.
  - The selection must keep at least one asset; a selector that empties the universe throws rather than passing a zero-asset problem downstream (the [`MissingDataFilter`](@ref) precedent).

# Returns

  - `res::AssetSelectorResult`: The fitted asset universe.

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

The surviving columns are emitted in *fitted* order, not in the window's own column order, because the terminal weights are indexed by the training universe and `assert_universe_aligned` compares the two name vectors elementwise.

# Algorithm

 1. For each fitted asset name, in fitted order, find the column of the window that carries it, and record that position in `idx`.
 2. Check that the name is present. A name the window does not carry throws.
 3. Return a [`port_opt_view`](@ref) of the window at `idx`. The positions are in fitted order, so the view reorders the window's columns when the two orders differ.

# Arguments

  - `res`: The fitted asset universe.
  - `rd`: The data window to transform.

# Validation

  - Every fitted asset name must be present in the window; a missing one throws rather than silently shrinking the universe.

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
                  ArgumentError("the fitted asset \"$name\" is absent from the data window, whose assets are $(collect(rd.nx)); the window must contain the whole fitted universe $(res.nx)"))
        idx[k] = j
    end
    return port_opt_view(rd, idx)
end
export AssetSelectorResult
