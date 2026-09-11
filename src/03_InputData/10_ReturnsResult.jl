"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all returns result types.

All concrete and/or types representing the result of returns calculations should be subtypes of `AbstractReturnsResult`.

## The asset-selector contract

[`select_assets`](@ref) and [`fit_preprocessing`](@ref) dispatch on this supertype, so any subtype reaching an [`AbstractAssetSelector`](@ref) must carry `nx` and an `observations × assets` matrix `X`, plus a [`port_opt_view`](@ref) that replays a selected universe. [`ClusterGroups`](@ref) widens that to `{nx, X, Z}`: it reads the feature matrix `Z` straight off the carrier, because preselection runs before any prior exists and no other source is reachable.

Widening the contract rather than the `Pr_RR` bridge is deliberate — that alias's concreteness is load-bearing at nine routing sites. The cost is that the contract is implicit: it is satisfied by [`ReturnsResult`](@ref) and enforced by nothing. [`PredictionReturnsResult`](@ref) subtypes this supertype, but its `X` is a *portfolio* return vector rather than an asset matrix — the asset axis is already collapsed away — so it satisfies neither the old contract nor the widened one, and every entry point refuses it loudly rather than measuring the wrong axis.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`select_assets`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that asset or factor names and their corresponding returns matrix are provided and consistent.

# Arguments

  - `names`: Asset or factor names.
  - `mat`: Returns matrix.
  - `names_sym`: Symbolic name for the names argument displayed in error messages.
  - `mat_sym`: Symbolic name for the matrix argument displayed in error messages.

# Validation

  - `allunique(names)`, whenever `names` is not `nothing`.

  - If either `names` or `mat` is not `nothing`:

      + `!isnothing(names)` and `!isnothing(mat)`.
      + `!isempty(names)` and `!isempty(mat)`.
      + `length(names) == size(mat, 2)`.

# Returns

  - `nothing`.

# Related

  - [`ReturnsResult`](@ref)
"""
function check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
                                        names_sym::Symbol, mat_sym::Symbol)
    if !isnothing(names)
        @argcheck(allunique(names),
                  ArgumentError("$names_sym names must be unique. Got\nallunique($names_sym) => $(allunique(names))"))
    end
    if !(isnothing(names) && isnothing(mat))
        @argcheck(!isnothing(names),
                  IsNothingError("$names_sym cannot be nothing if $mat_sym is not `nothing`. Got\n!isnothing($names_sym) => $(!isnothing(names))\n!isnothing($mat_sym) => $(!isnothing(mat))"))
        @argcheck(!isnothing(mat),
                  IsNothingError("$mat_sym cannot be nothing if $names_sym is not `nothing`. Got\n!isnothing($names_sym) => $(!isnothing(names))\n!isnothing($mat_sym) => $(!isnothing(mat))"))
        @argcheck(!isempty(names), IsEmptyError("$names_sym cannot be empty."))
        @argcheck(!isempty(mat), IsEmptyError("$mat_sym cannot be empty."))
        @argcheck(length(names) == size(mat, 2),
                  DimensionMismatch("length($names_sym) == size($mat_sym, 2) must hold. Got\nlength($names_sym) => $(length(names))\nsize($mat_sym, 2) => $(size(mat, 2))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the results of asset and factor returns calculations.

`ReturnsResult` is the standard result type returned by returns-processing routines, such as [`prices_to_returns`](@ref).

It supports both asset and factor returns, as well as optional time series and implied volatility information, and is designed for downstream compatibility with optimisation and analysis routines.

It also carries the optional feature matrix `Z` that [`FeatureDistance`](@ref) turns into a distance. `Z` is *data*, not configuration, which is why it is held here rather than on the estimator: the clustering stack is asset-subset-blind by construction, so an estimator-held feature matrix would survive a nested-clustered subproblem or a cross-validation fold unsliced, with its asset axis silently pointing at the full universe. `ReturnsResult` implements [`port_opt_view`](@ref), so a carried `Z` is subselected in step with `X`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReturnsResult(;
        nx::Option{<:VecStr} = nothing,
        X::Option{<:MatNum} = nothing,
        nf::Option{<:VecStr} = nothing,
        F::Option{<:MatNum} = nothing,
        nb::Option{<:VecStr} = nothing,
        B::Option{<:VecNum_MatNum} = nothing,
        ts::Option{<:VecDate} = nothing,
        iv::Option{<:MatNum} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        pnl::Option{<:AssetPanel} = nothing,
    ) -> ReturnsResult

Keywords correspond to the struct's fields.

## Validation

  - If `nx` or `X` is not `nothing`, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is not `nothing`, `!isempty(nf)`, `!isempty(F)`, `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `nb` or `B` is not `nothing` and `B` is a matrix: `!isempty(nb)`, `!isempty(B)`, and `length(nb) == size(B, 2)`.
  - If `nb` or `B` is not `nothing` and `B` is a vector: `length(nb) == 1`.
  - If `X` and `B` are not `nothing`: if `B` is a vector, `size(X, 1) == size(B, 1)`; if `B` is a matrix, `size(X) == size(B)`.
  - If `ts` is not `nothing`, `!isempty(ts)`, `allunique(ts)`, and `length(ts) == size(X, 1)`. Uniqueness is required because `ts` *keys* the observation axis rather than merely labelling it: [`feature_row_indices`](@ref) recovers a subset's rows by matching its surviving timestamps back into this clock, and a repeated timestamp would resolve to the first occurrence and pair an asset with another period's features.
  - If `ts` and `B` are not `nothing`: `length(ts) == size(B, 1)`.
  - If `iv` is not `nothing`, `!isempty(iv)`, every value is non-negative where it is present (an absent one is `NaN`; see [`assert_nonneg_where_present`](@ref)), and `size(iv) == size(X)`.
  - `ivpa` is validated in that same branch, so it is checked only when `iv` is given: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`, and, if a vector, `length(ivpa) == size(iv, 2)`. The bound is strict — a zero adjustment is rejected. An `ivpa` passed without an `iv` reaches no check, because it has no implied volatility to adjust.
  - `pnl`'s asset axis is `length(nx)`, and its observation axis is `size(X, 1)` when it is time-varying. See [`check_asset_panel`](@ref).

# Examples

```jldoctest
julia> ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`AssetPanel`](@ref)
  - [`asset_panel`](@ref)
  - [`check_asset_panel`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
  - [`VecDate`](@ref)
  - [`Num_VecNum`](@ref)
"""
@concrete struct ReturnsResult <: AbstractReturnsResult
    """
    Names or identifiers of asset columns (assets × 1).
    """
    nx
    """
    Asset returns matrix (observations × assets).
    """
    X
    """
    Names or identifiers of factor columns (factors × 1).
    """
    nf
    """
    Factor returns matrix (observations × factors).
    """
    F
    """
    Names or identifiers of benchmark columns (observations × 1) or (observations × assets).
    """
    nb
    """
    Benchmark prices (observations × 1) or (observations × assets).
    """
    B
    """
    Optional timestamps for each observation (observations × 1).
    """
    ts
    """
    Implied volatilities matrix (observations × assets).
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    Optional [`AssetPanel`](@ref): the Panel Fields of the universe, and its two universe masks.
    """
    pnl
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, nb::Option{<:VecStr},
                           B::Option{<:VecNum_MatNum}, ts::Option{<:VecDate},
                           iv::Option{<:MatNum}, ivpa::Option{<:Num_VecNum},
                           pnl::Option{<:AssetPanel})
        check_names_and_returns_matrix(nx, X, :nx, :X)
        check_names_and_returns_matrix(nf, F, :nf, :F)
        if isa(B, VecNum) && !isnothing(nb)
            @argcheck(length(nb) == 1,
                      DimensionMismatch("a single-column benchmark (B) admits exactly one benchmark name (nb), got length(nb) = $(length(nb))"))
        elseif isa(B, MatNum)
            check_names_and_returns_matrix(nb, B, :nb, :B)
        end
        if !isnothing(X) && !isnothing(F)
            @argcheck(size(X, 1) == size(F, 1),
                      DimensionMismatch("asset returns (X) and factor returns (F) must share the same number of observations (rows), got size(X, 1) = $(size(X, 1)) and size(F, 1) = $(size(F, 1))"))
        end
        if !isnothing(X) && !isnothing(B)
            if isa(B, VecNum)
                @argcheck(size(X, 1) == size(B, 1),
                          DimensionMismatch("benchmark returns (B) must match asset returns (X) in number of observations (rows), got size(X, 1) = $(size(X, 1)) and size(B, 1) = $(size(B, 1))"))
            else
                @argcheck(size(X) == size(B),
                          DimensionMismatch("benchmark returns (B) must match asset returns (X) in size, got size(X) = $(size(X)) and size(B) = $(size(B))"))
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            # `ts` is an *index* into the observation axis, not merely a label on it: a
            # subset's surviving timestamps are matched back into it to recover the rows a
            # time-varying feature matrix must keep (see `feature_row_indices`). A repeated
            # timestamp makes that recovery pick the first occurrence and silently pair an
            # asset with another period's features, so the axis must be uniquely keyed.
            @argcheck(allunique(ts),
                      ArgumentError("timestamps (ts) must be unique — they key the observation axis, and a repeated timestamp makes a row unrecoverable by time. Got $(length(ts) - length(unique(ts))) duplicate(s), the first being $(ts[findfirst(i -> ts[i] in view(ts, 1:(i - 1)), eachindex(ts))])"))
            if !isnothing(X)
                @argcheck(length(ts) == size(X, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per asset-returns (X) observation (row), got length(ts) = $(length(ts)) and size(X, 1) = $(size(X, 1))"))
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per factor-returns (F) observation (row), got length(ts) = $(length(ts)) and size(F, 1) = $(size(F, 1))"))
            end
            if !isnothing(B)
                @argcheck(length(ts) == size(B, 1),
                          DimensionMismatch("timestamps (ts) must have one entry per benchmark-returns (B) observation (row), got length(ts) = $(length(ts)) and size(B, 1) = $(size(B, 1))"))
            end
        end
        if !isnothing(iv)
            @argcheck(!isempty(iv), IsEmptyError)
            assert_nonneg_where_present(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X),
                      DimensionMismatch("implied volatilities (iv) must match asset returns (X) in size, got size(iv) = $(size(iv)) and size(X) = $(size(X))"))
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(iv, 2),
                          DimensionMismatch("the implied-volatility risk-premium adjustment (ivpa), when a vector, must have one entry per asset (implied-volatility column), got length(ivpa) = $(length(ivpa)) and size(iv, 2) = $(size(iv, 2))"))
            end
        end
        check_asset_panel(pnl, isnothing(nx) ? nothing : length(nx),
                          isnothing(X) ? nothing : size(X, 1), "length(nx)")
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ts), typeof(iv), typeof(ivpa), typeof(pnl)}(nx, X, nf, F, nb, B,
                                                                      ts, iv, ivpa, pnl)
    end
end
function ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                       nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                       nb::Option{<:VecStr} = nothing, B::Option{<:VecNum_MatNum} = nothing,
                       ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                       ivpa::Option{<:Num_VecNum} = nothing,
                       pnl::Option{<:AssetPanel} = nothing)::ReturnsResult
    return ReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa, pnl)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `ReturnsResult` object for the assets at indices `i`.

This is the [`port_opt_view`](@ref) method for [`ReturnsResult`](@ref) — the View of the library's central data structure, restricting it to a subset of assets.

!!! warning

    This two-argument method indexes **assets**, matching the rest of the `port_opt_view` family. The four-argument method `port_opt_view(rd, i, j, k)` indexes **observations** first and assets second. The two arities therefore give `i` different meanings; see [`port_opt_view(rd::ReturnsResult, i, j, k)`](@ref).

# Algorithm

 1. View the asset names `nx` at `i` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, :, i)`. Axis 2 is the assets, and every observation is kept.
 3. When `B` is a matrix, it holds one column per asset: view `nb` at `i`, and view `B` as `view(rd.B, :, i)`. Otherwise — a single shared benchmark, or none at all — `nb` and `B` both pass through untouched.
 4. View the implied volatilities as `view(rd.iv, :, i)`, and the adjustment `ivpa` at `i`.
 5. View the [`AssetPanel`](@ref) `pnl` with [`panel_carrier_view`](@ref) at `i` on the asset axis, handing it the asset names `rd.nx`. The observation index is a `Colon`, so a time-varying panel keeps every observation. The view slices every Panel Field's values and both universe masks on the asset axis, and a tensor Panel Field whose labels *are* the asset names ([`features_are_assets`](@ref)) on its label axis as well; every other field's label axis addresses features, which an asset view does not reach.
 6. Rebuild the [`ReturnsResult`](@ref). The factor names `nf`, the factor returns `F` and the timestamps `ts` pass through untouched, because none of the three has an asset axis.

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Indices of the assets to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified index.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> PortfolioOptimisers.port_opt_view(rd, 2:2)
ReturnsResult
    nx ┼ SubArray{String, 1, Vector{String}, Tuple{UnitRange{Int64}}, true}: ["B"]
     X ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

* * *

    port_opt_view(
        rd::ReturnsResult,
        i,
        j,
        k = :
    ) -> ReturnsResult

Return a view of the `ReturnsResult` object for assets at indices `j`, observations at indices `i`, and factors at indices `k`.

!!! warning

    Unlike every other [`port_opt_view`](@ref) method — including [`port_opt_view(rd::ReturnsResult, i)`](@ref) — the first index of this method selects **observations**, not assets. Assets are the *second* index. Cross-validation splits observations and assets together, which is why this arity exists at all.

# Algorithm

 1. View the asset names `nx` at `j` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, i, j)`. Axis 1 is the observations, and axis 2 is the assets.
 3. View the factor names `nf` at `k`, unless `k` is a `Colon`, in which case `nf` passes through. View the factor returns as `view(rd.F, i, k)`.
 4. When `B` is a matrix, it holds one column per asset: view `nb` at `j`, and view `B` as `view(rd.B, i, j)`. When `B` is a vector, it is a single shared benchmark: view it as `view(rd.B, i)`, and carry `nb` through.
 5. View the timestamps `ts` at `i`, the implied volatilities as `view(rd.iv, i, j)`, and the adjustment `ivpa` at `j`.
 6. View the [`AssetPanel`](@ref) `pnl` with [`panel_carrier_view`](@ref) at the observations `i` and the assets `j`, handing it the asset names `rd.nx`, which slices both axes of every Panel Field and of both universe masks, and the label axis of a tensor Panel Field whose labels *are* the asset names ([`features_are_assets`](@ref)). A static panel has no observation axis and ignores `i`, which is the same asymmetry `ivpa` has on the asset axis.
 7. Rebuild the [`ReturnsResult`](@ref).

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified indices.

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; 0.3 0.4; 0.5 0.6], nf = [\"F1\"],
                          F = [1.0; 2.0; 3.0;;])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 3×2 Matrix{Float64}
    nf ┼ Vector{String}: ["F1"]
     F ┼ 3×1 Matrix{Float64}
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> PortfolioOptimisers.port_opt_view(rd, 1:2, 2:2)
ReturnsResult
    nx ┼ SubArray{String, 1, Vector{String}, Tuple{UnitRange{Int64}}, true}: ["B"]
     X ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}
    nf ┼ Vector{String}: ["F1"]
     F ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, Base.Slice{Base.OneTo{Int64}}}, false}
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing
```

* * *

    port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)

Erroring tripwire for [`AbstractReturnsResult`](@ref) subtypes that do not implement [`port_opt_view`](@ref).

Without it, the universal leaf fallback `port_opt_view(x, i, args...)` would hand back the returns result *unsubselected*, and a meta-optimiser or cross-validation fold would silently train on the full universe. Returns data is never a leaf value, so an unhandled subtype is a missing method, not a pass-through.

Subtypes carrying an [`AssetPanel`](@ref) owe it the same treatment as `X`: subselect its asset axis on every arity, its observation axis on the arities that take one, and — when a tensor Panel Field's labels *are* the assets ([`features_are_assets`](@ref)) — that field's label axis as well. A panel that survives a fold unsliced is the same silent-wrongness as an unsliced returns matrix, one level down: the distance it produces is finite, plausible, and computed over the wrong universe. [`port_opt_view`](@ref) implements the rule; the [`ReturnsResult`](@ref) methods are the reference.

# Algorithm

 1. Throw an `ArgumentError` naming the concrete type and the number of index arguments the call gave. The method reads neither the indices nor the fields of `rd`.

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractReturnsResult`](@ref)

* * *

    port_opt_view(rd::ReturnsResult, args...; kwargs...)

Erroring tripwire for [`ReturnsResult`](@ref) calls whose *call shape* no supported arity matches.

`ReturnsResult` does implement [`port_opt_view`](@ref), so the [`AbstractReturnsResult`](@ref) tripwire above would misreport a mistyped call as an unimplemented subtype. This method takes the call instead and names the call shape: the supported arities take one, two, or three positional index arguments and no keyword arguments — in particular `factors` is the third *positional* index, not a keyword.

# Algorithm

 1. Count the positional arguments `args`, and read the names of the keyword arguments `kwargs`.
 2. Throw an `ArgumentError` reporting both counts, and naming the three supported call shapes.

# Related

  - [`port_opt_view`](@ref)
  - [`ReturnsResult`](@ref)
"""
function port_opt_view(rd::ReturnsResult, i)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, i)
    B = !isa(rd.B, MatNum) ? rd.B : view(rd.B, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    pnl = panel_carrier_view(rd.pnl, :, i, rd.nx)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, nb = nb, B = B, ts = rd.ts,
                         iv = iv, ivpa = ivpa, pnl = pnl)
end
function port_opt_view(rd::ReturnsResult, i, j, k = :)
    nx = nothing_scalar_array_view(rd.nx, j)
    X = isnothing(rd.X) ? rd.X : view(rd.X, i, j)
    nf = isnothing(rd.nf) || isa(k, Colon) ? rd.nf : view(rd.nf, k)
    F = isnothing(rd.F) ? rd.F : view(rd.F, i, k)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, j)
    B = if isnothing(rd.B)
        nothing
    elseif isa(rd.B, VecNum)
        view(rd.B, i)
    else
        view(rd.B, i, j)
    end
    ts = isnothing(rd.ts) ? rd.ts : view(rd.ts, i)
    iv = isnothing(rd.iv) ? rd.iv : view(rd.iv, i, j)
    ivpa = nothing_scalar_array_view(rd.ivpa, j)
    pnl = panel_carrier_view(rd.pnl, i, j, rd.nx)
    return ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, ts = ts, iv = iv,
                         ivpa = ivpa, pnl = pnl)
end
function port_opt_view(rd::ReturnsResult, args...; kwargs...)
    kws = keys(kwargs)
    kwmsg = isempty(kws) ? "" : " and keyword argument(s) " * join(kws, ", ")
    return throw(ArgumentError("port_opt_view(::ReturnsResult, ...) does not accept this call shape; got $(length(args)) positional index argument(s)$(kwmsg). Supported shapes: port_opt_view(rd, assets) to subselect assets; port_opt_view(rd, observations, assets) or port_opt_view(rd, observations, assets, factors) to subselect observations and assets together (note the reversed index order, and that `factors` is the third positional index, not a keyword)."))
end
function port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)
    return throw(ArgumentError("$(typeof(rd)) subtypes AbstractReturnsResult but does not implement port_opt_view for $(length(args)) index argument(s). Extension authors: implement port_opt_view for the subtype; without it a meta-optimiser or cross-validation fold would silently train on the unsubselected universe. See port_opt_view(rd::ReturnsResult, ...) for the reference implementation."))
end
"""
    const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}

Union of the two data levels cross-validation folds can be computed on: returns-level ([`AbstractReturnsResult`](@ref)) and price-level ([`AbstractPricesResult`](@ref)) data.

Fold generation only needs an observation count ([`cv_nobs`](@ref)) and a timestamp vector ([`cv_timestamps`](@ref)), so [`Base.split`](@ref) and [`n_splits`](@ref) accept either level. Price-level splitting is what lets a `Pipeline` be cross-validated on its *input* rows, keeping stateful preprocessing inside the fold.

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`AbstractPricesResult`](@ref)
  - [`cv_nobs`](@ref)
  - [`cv_timestamps`](@ref)
  - [`n_splits`](@ref)
"""
const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}
"""
    returns_result_picker(rd::ReturnsResult, brt::Bool) -> ReturnsResult

Return a `ReturnsResult` appropriate for benchmark-tracking optimisations.

This helper inspects the `ReturnsResult`'s benchmark field `B` and the boolean flag `brt` (benchmark-tracking). If `brt` is `true` and a benchmark `B` is present it returns a new `ReturnsResult` in which asset returns `X` have the benchmark removed (i.e. `X - B` or broadcast `X .- B` for vector benchmarks). If `brt` is `false` or no benchmark is present, the original `ReturnsResult` is returned unchanged.

# Algorithm

The first step is a method selected on the field type of `B`, so a carrier with no benchmark runs no branch at all.

 1. `rd` carries no benchmark, because its `B` field is `Nothing`: return `rd` itself.
 2. `brt` is `false`: return `rd` itself.
 3. `brt` is `true`: subtract the benchmark from the asset returns, giving `X`. A vector benchmark subtracts by broadcast, `rd.X .- rd.B`, which takes one benchmark value per observation from every asset column. A matrix benchmark subtracts elementwise, `rd.X - rd.B`.
 4. Rebuild the [`ReturnsResult`](@ref) from `X`, and leave `nb` and `B` unset. The benchmark is spent on the subtraction, which is what makes a second call return its argument unchanged. Every other field — `nx`, `nf`, `F`, `ts`, `iv`, `ivpa` and `pnl` — is carried over. The argument itself is never modified.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset, factor and/or benchmark returns.
  - `brt`: Boolean flag indicating whether benchmark-tracking behaviour should be applied. When `true`, asset returns are adjusted by subtracting the benchmark `B` (if present).

# Returns

  - `rd::ReturnsResult`:

      + If `brt` is `true` and a benchmark `B` is present: A new `ReturnsResult` with adjusted asset returns
      + Otherwise: The `rd` is returned unchanged. `nb` and `B` hold `nothing` on an adjusted result, which is what makes the adjustment idempotent.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.10 0.20; 0.30 0.40], nb = [\"BM\"],
                          B = [0.01; 0.02])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ Vector{String}: ["BM"]
     B ┼ Vector{Float64}: [0.01, 0.02]
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd2 = returns_result_picker(rd, false)  # no change when brt is false
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ Vector{String}: ["BM"]
     B ┼ Vector{Float64}: [0.01, 0.02]
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd === rd2
true

julia> rd3 = returns_result_picker(rd, true)
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┼ nothing
   pnl ┴ nothing

julia> rd.X .- rd.B == rd3.X
true
```

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
"""
function returns_result_picker(rd::ReturnsResult{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                 Nothing}, ::Any)
    return rd
end
function returns_result_picker(rd::ReturnsResult{<:Any, <:MatNum, <:Any, <:Any, <:Any,
                                                 <:VecNum_MatNum}, brt::Bool)
    return if !brt
        rd
    else
        X = isa(rd.B, VecNum) ? rd.X .- rd.B : rd.X - rd.B
        ReturnsResult(; nx = rd.nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts, iv = rd.iv,
                      ivpa = rd.ivpa, pnl = rd.pnl)
    end
end
"""
    asset_panel(ape::Nothing, pr, rd::ReturnsResult, X) -> AssetPanel
    asset_panel(ape::Nothing, pr::ReturnsResult, rd::Nothing, X) -> AssetPanel
    asset_panel(ape::Nothing, pr, rd::Nothing, X) -> Union{}

Resolve the [`AssetPanel`](@ref) a [`FeatureDistance`](@ref) with no producer measures.

`nothing` in the `ape` slot says *read the panel the data carrier already holds*. The carriers reach the kernel as the two keywords `pr` and `rd`, and this verb resolves the source by dispatch: a [`ReturnsResult`](@ref) in either slot answers its `pnl`, and `rd` wins when both hold one, because the data carrier is where a panel is data rather than a by-product. `Pr_RR` admits a [`ReturnsResult`](@ref) in the `pr` slot, which is what `clusterise(cle, rd)` and every [`Pipeline`](@ref) step pass, so the second method is not a fallback but the shortest public call.

A prior result alone carries no panel, so it raises an [`IsNothingError`](@ref) naming the two ways forward.

# Algorithm

The method that Julia selects is the algorithm.

 1. `rd` is a [`ReturnsResult`](@ref): answer `rd.pnl`.
 2. `pr` is a [`ReturnsResult`](@ref) and there is no `rd`: answer `pr.pnl`.
 3. Neither slot holds a data carrier: raise.

Each of the first two checks that the carrier it read holds a panel, with [`assert_asset_panel_supplied`](@ref).

# Arguments

  - `ape`: `nothing`, which reads the carrier's panel.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `X`: Returns matrix of the subproblem. Unread here; a producer reads it.

# Validation

  - A data carrier is present, and it holds an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel the data carrier holds.

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`assert_asset_panel_supplied`](@ref)
  - [`FeatureDistance`](@ref)
  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`RegressionPanel`](@ref)
  - [`PhylogenyPanel`](@ref)
"""
function asset_panel(::Nothing, ::Any, rd::ReturnsResult, ::Any)
    return assert_asset_panel_supplied(rd.pnl)
end
function asset_panel(::Nothing, pr::ReturnsResult, ::Nothing, ::Any)
    return assert_asset_panel_supplied(pr.pnl)
end
function asset_panel(::Nothing, ::Any, ::Nothing, ::Any)
    return throw(IsNothingError("`FeatureDistance` with no producer reads the Asset Panel off the data carrier, and this call supplied none: only a prior result reached it, and a prior result carries no panel. Two ways forward:\n  1. Pass the `ReturnsResult` that holds the panel, which every forwarder takes as `rd`.\n  2. Set a producer on the estimator, `FeatureDistance(; ape = RegressionPanel())`, which builds a panel from the prior it is handed."))
end
"""
    assert_asset_panel_supplied(pnl::AssetPanel) -> AssetPanel
    assert_asset_panel_supplied(pnl::Nothing) -> Union{}

Assert that the data carrier a [`FeatureDistance`](@ref) read holds an [`AssetPanel`](@ref), and return it.

The carrier's `pnl` is optional, so a carrier built without one reaches the kernel as `nothing`. This is the one place that turns it into a diagnostic, and it returns the panel so the caller reads one verb rather than a check and an access.

# Algorithm

The method that Julia selects is the algorithm. A panel is returned; `nothing` raises.

# Arguments

  - `pnl`: The carrier's Asset Panel, or `nothing`.

# Validation

  - `!isnothing(pnl)`. Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: The Asset Panel.

# Related

  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`ReturnsResult`](@ref)
  - [`FeatureDistance`](@ref)
  - [`IsNothingError`](@ref)
"""
function assert_asset_panel_supplied(pnl::AssetPanel)
    return pnl
end
function assert_asset_panel_supplied(::Nothing)
    return throw(IsNothingError("`FeatureDistance` with no producer reads the Asset Panel off the data carrier, and the carrier holds none. Build one with `asset_panel(inputs)` and pass it as `ReturnsResult(; …, pnl = pnl)`, or set a producer on the estimator, `FeatureDistance(; ape = RegressionPanel())`."))
end

export ReturnsResult, returns_result_picker
