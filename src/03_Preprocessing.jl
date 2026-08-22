"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all returns result types.

All concrete and/or types representing the result of returns calculations should be subtypes of `AbstractReturnsResult`.

## The asset-selector contract

[`select_assets`](@ref) and [`fit_preprocessing`](@ref) dispatch on this supertype, so any subtype reaching an [`AbstractAssetSelector`](@ref) must carry `nx` and an `observations × assets` matrix `X`, plus a [`port_opt_view`](@ref) that replays a selected universe. [`ClusterGroups`](@ref) widens that to `{nx, X, Z}`: it reads the feature matrix `Z` straight off the carrier, because preselection runs before any prior exists and no other source is reachable.

Widening the contract rather than the `Pr_RR` bridge is deliberate — that alias's concreteness is load-bearing at nine routing sites. The cost is that the contract is implicit: it is satisfied by [`ReturnsResult`](@ref) and enforced by nothing. [`PredictionReturnsResult`](@ref) subtypes this supertype and carries `nz`/`Z`, but its `X` is a *portfolio* return vector rather than an asset matrix — the asset axis is already collapsed away — so it satisfies neither the old contract nor the widened one, and every entry point refuses it loudly rather than measuring the wrong axis.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`select_assets`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all price-level data result types.

All concrete types representing price-level data should be subtypes of `AbstractPricesResult`. Defined alongside [`AbstractReturnsResult`](@ref) so cross-validation splitting, preprocessing, and prediction can dispatch on either data level.

# Related

  - [`AbstractResult`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesResult <: AbstractResult end
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
                  IsNothingError("$names_sym cannot be nothing if $mat_sym is not `nothing`. Got\n!isnothing($names_sym) => $(isnothing(names))\n!isnothing($mat_sym) => $(isnothing(mat))"))
        @argcheck(!isnothing(mat),
                  IsNothingError("$mat_sym cannot be nothing if $names_sym is not `nothing`. Got\n!isnothing($names_sym) => $(isnothing(names))\n!isnothing($mat_sym) => $(isnothing(mat))"))
        @argcheck(!isempty(names), IsEmptyError("$names_sym cannot be empty."))
        @argcheck(!isempty(mat), IsEmptyError("$mat_sym cannot be empty."))
        @argcheck(length(names) == size(mat, 2),
                  DimensionMismatch("length($names_sym) == size($mat_sym, 2) must hold. Got\nlength($names_sym) => $(length(names))\nsize($mat_sym, 2) => $(size(mat, 2))"))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that feature names are provided alongside their feature matrix and are internally consistent.

The **names layer** of [`check_names_and_feature_matrix`](@ref): both-or-neither, non-empty, and unique. Everything that depends only on the matrix — non-emptiness, finiteness and shape — belongs to [`check_feature_matrix`](@ref), so a nameless carrier such as [`LowOrderPrior`](@ref) can reuse it.

# Arguments

  - `nz`: Feature names.
  - `Z`: Feature matrix.

# Validation

  - `!isnothing(nz)`.
  - `!isempty(nz)`.
  - `allunique(nz)`.

# Returns

  - `nothing`.

# Related

  - [`check_names_and_feature_matrix`](@ref)
  - [`check_feature_matrix`](@ref)
  - [`FeatureDistance`](@ref)
"""
function check_feature_names(nz::Option{<:VecStr}, ::MatNum_Arr3Num)::Nothing
    @argcheck(!isnothing(nz),
              IsNothingError("nz cannot be nothing if Z is not `nothing`. Got\n!isnothing(nz) => $(!isnothing(nz))\n!isnothing(Z) => true"))
    @argcheck(!isempty(nz), IsEmptyError("nz cannot be empty."))
    @argcheck(allunique(nz),
              ArgumentError("nz names must be unique. Got\nallunique(nz) => $(allunique(nz))"))
    return nothing
end
"""
    check_feature_matrix(Z::Nothing, na, nobs, na_sym)
    check_feature_matrix(Z::MatNum, na, nobs, na_sym)
    check_feature_matrix(Z::Arr3Num, na, nobs, na_sym)

Validate a feature matrix against the asset and observation axes of the result carrying it, without reference to feature names.

The **shape layer**. It holds every check that depends only on the matrix: non-emptiness, finiteness, and the binding of the asset (and, when time-varying, observation) axis. A feature matrix is never imputed, so a `NaN` or `±Inf` entry is rejected at construction rather than carried into a metric that would map it to a finite, plausible, wrong distance.

Two carriers use it. [`ReturnsResult`](@ref) and [`PricesResult`](@ref) reach it through [`check_names_and_feature_matrix`](@ref), which adds the names layer on top. [`LowOrderPrior`](@ref) calls it directly: a prior-side feature matrix is **nameless** by design — a producer runs inside `prior(pe, X, F; …)` with raw matrices and no names — so it has a shape to check and nothing to check it against by name.

The three methods dispatch on the shape rather than branching on `ndims`.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method.

 1. `Z` is `nothing`: return. The carrier holds no feature matrix, so there is no shape to check.
 2. `Z` is a `MatNum`, which is `assets × features`: check that `Z` is non-empty and that every entry is finite. Check that `na` is not `nothing`. Check that `size(Z, 1) == na`, which binds axis 1 to the assets. The observation count `nobs` is not read, because a static feature matrix has no observation axis.
 3. `Z` is an `Arr3Num`, which is `observations × assets × features`: check that `Z` is non-empty and that every entry is finite. Check that neither `na` nor `nobs` is `nothing`. Check that `size(Z, 1) == nobs`, which binds axis 1 to the observations, and that `size(Z, 2) == na`, which binds axis 2 to the assets.

No method reads the trailing feature axis. That axis binds to `length(nz)`, and the names are the one thing this layer does not hold, so [`check_names_and_feature_matrix`](@ref) makes that check.

# Arguments

  - `Z`: Feature matrix.
  - `na`: Number of assets the feature matrix must bind to, or `nothing` when the carrier has no asset axis.
  - `nobs`: Number of observations a time-varying feature matrix must bind to, or `nothing` when the carrier has no observation axis.
  - `na_sym`: Symbolic name of the asset anchor displayed in error messages.

# Validation

  - `Z` is non-empty and every entry is finite.
  - `size(Z, 1) == na` (static) or `size(Z, 2) == na` (time-varying); `na` must not be `nothing`.
  - `size(Z, 1) == nobs` for a time-varying `Z`; `nobs` must not be `nothing`.

# Returns

  - `nothing`.

# Related

  - [`check_names_and_feature_matrix`](@ref)
  - [`check_feature_names`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`MatNum_Arr3Num`](@ref)
  - [`Sym_Str`](@ref)
"""
function check_feature_matrix(::Nothing, ::Option{<:Integer}, ::Option{<:Integer},
                              ::Sym_Str)::Nothing
    return nothing
end
function check_feature_matrix(Z::MatNum, na::Option{<:Integer}, ::Option{<:Integer},
                              na_sym::Sym_Str)::Nothing
    assert_nonempty(Z, :Z)
    assert_all_finite(Z, :Z)
    @argcheck(!isnothing(na),
              IsNothingError("a static feature matrix (Z) is assets × features, so it needs an asset axis to bind to, but $na_sym is nothing"))
    @argcheck(size(Z, 1) == na,
              DimensionMismatch("a static feature matrix (Z) is assets × features, so its rows must be the assets, got size(Z, 1) = $(size(Z, 1)) and $na_sym = $na. If Z is features × assets, transpose it: the carried feature matrix is always assets-major, because port_opt_view has no `dims` keyword to declare an orientation with."))
    return nothing
end
function check_feature_matrix(Z::Arr3Num, na::Option{<:Integer}, nobs::Option{<:Integer},
                              na_sym::Sym_Str)::Nothing
    assert_nonempty(Z, :Z)
    assert_all_finite(Z, :Z)
    @argcheck(!isnothing(na),
              IsNothingError("a time-varying feature matrix (Z) is observations × assets × features, so it needs an asset axis to bind to, but $na_sym is nothing"))
    @argcheck(!isnothing(nobs),
              IsNothingError("a time-varying feature matrix (Z) is observations × assets × features, so it needs an observation axis to bind to; provide the asset data its observations are parallel to, or pass a static assets × features Z instead"))
    @argcheck(size(Z, 1) == nobs,
              DimensionMismatch("a time-varying feature matrix (Z) is observations × assets × features, so its leading axis must be the observations, got size(Z, 1) = $(size(Z, 1)) and $nobs observations"))
    @argcheck(size(Z, 2) == na,
              DimensionMismatch("a time-varying feature matrix (Z) is observations × assets × features, so its second axis must be the assets, got size(Z, 2) = $(size(Z, 2)) and $na_sym = $na. If the trailing axes are features × assets, permute them: the carried feature matrix is always assets-major, because port_opt_view has no `dims` keyword to declare an orientation with."))
    return nothing
end
"""
    check_names_and_feature_matrix(nz::Option{<:VecStr}, Z::Nothing, na, nobs, na_sym)
    check_names_and_feature_matrix(nz::Option{<:VecStr}, Z::MatNum, na, nobs, na_sym)
    check_names_and_feature_matrix(nz::Option{<:VecStr}, Z::Arr3Num, na, nobs, na_sym)

Validate a feature matrix and its names against the asset and observation axes of the result carrying them.

The carried feature matrix is **canonically assets-major**: `assets × features` when static, `observations × assets × features` when time-varying. Unlike the raw-matrix [`distance`](@ref) entry point, there is no `dims` keyword here to declare an orientation — [`port_opt_view`](@ref) has none to consult either, and it already indexes `X` as `observations × assets` unconditionally. Fixing the layout is what makes the two agree: a transposed non-square `Z` fails here instead of surviving into a fold whose asset axis points at the wrong universe.

This is the **names layer**: it composes [`check_feature_names`](@ref) and [`check_feature_matrix`](@ref) and adds the one check that needs both, binding the trailing feature axis to `length(nz)`. A carrier that has a feature matrix but no names — [`LowOrderPrior`](@ref) — calls [`check_feature_matrix`](@ref) directly instead.

The three methods dispatch on the shape rather than branching on `ndims`.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method.

 1. `Z` is `nothing`: check that `nz` is `nothing` too, and return. A name vector without a matrix names an axis the carrier does not hold.
 2. `Z` is a `MatNum`, which is `assets × features`: apply the names layer with [`check_feature_names`](@ref), then the shape layer with [`check_feature_matrix`](@ref), then check that `size(Z, 2) == length(nz)`, which binds axis 2 to the features. The shape layer is given `nothing` for the observation count, because a static feature matrix has no observation axis.
 3. `Z` is an `Arr3Num`, which is `observations × assets × features`: apply the same two layers, then check that `size(Z, 3) == length(nz)`, which binds axis 3 to the features. The observation count reaches the shape layer, which binds axis 1 to the observations and axis 2 to the assets.

# Arguments

  - `nz`: Feature names.
  - `Z`: Feature matrix.
  - `na`: Number of assets the feature matrix must bind to, or `nothing` when the carrier has no asset axis.
  - `nobs`: Number of observations a time-varying feature matrix must bind to, or `nothing` when the carrier has no observation axis.
  - `na_sym`: Symbolic name of the asset anchor displayed in error messages.

# Validation

  - `nz` and `Z` are both `nothing` or both given (see [`check_feature_names`](@ref)).
  - `size(Z, 1) == na` (static) or `size(Z, 2) == na` (time-varying); `na` must not be `nothing`.
  - `size(Z, 1) == nobs` for a time-varying `Z`; `nobs` must not be `nothing`.
  - `size(Z, ndims(Z)) == length(nz)`.

# Returns

  - `nothing`.

# Related

  - [`check_feature_names`](@ref)
  - [`check_feature_matrix`](@ref)
  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`MatNum_Arr3Num`](@ref)
  - [`Sym_Str`](@ref)
"""
function check_names_and_feature_matrix(nz::Option{<:VecStr}, ::Nothing,
                                        ::Option{<:Integer}, ::Option{<:Integer},
                                        ::Sym_Str)::Nothing
    @argcheck(isnothing(nz),
              IsNothingError("Z cannot be nothing if nz is not `nothing`. Got\n!isnothing(nz) => true\n!isnothing(Z) => false"))
    return nothing
end
function check_names_and_feature_matrix(nz::Option{<:VecStr}, Z::MatNum,
                                        na::Option{<:Integer}, ::Option{<:Integer},
                                        na_sym::Sym_Str)::Nothing
    check_feature_names(nz, Z)
    check_feature_matrix(Z, na, nothing, na_sym)
    @argcheck(size(Z, 2) == length(nz),
              DimensionMismatch("length(nz) == size(Z, 2) must hold. Got\nlength(nz) => $(length(nz))\nsize(Z, 2) => $(size(Z, 2))"))
    return nothing
end
function check_names_and_feature_matrix(nz::Option{<:VecStr}, Z::Arr3Num,
                                        na::Option{<:Integer}, nobs::Option{<:Integer},
                                        na_sym::Sym_Str)::Nothing
    check_feature_names(nz, Z)
    check_feature_matrix(Z, na, nobs, na_sym)
    @argcheck(size(Z, 3) == length(nz),
              DimensionMismatch("length(nz) == size(Z, 3) must hold. Got\nlength(nz) => $(length(nz))\nsize(Z, 3) => $(size(Z, 3))"))
    return nothing
end
"""
    features_are_assets(nz::Option{<:VecStr}, nx::Option{<:VecStr}) -> Bool

Report whether the feature axis *is* the asset axis, so a view must slice both.

True when the feature names equal the asset names, which is what a square phylogeny or adjacency matrix reused as a feature source produces: an `assets × assets` matrix whose features are "adjacent to asset ``k``". Subselecting assets without also subselecting the feature axis would leave the columns pointing at the full universe while the rows point at the subset — a silently wrong distance rather than an error.

Compares the names rather than the axis lengths: a rectangular-by-accident coincidence of counts is not a claim that the axes mean the same thing, and the comparison stays correct under repeated views, since both name vectors are sliced by the same indices.

# Algorithm

 1. If either `nz` or `nx` is `nothing`, return `false`. A carrier that names only one of the two axes makes no claim that the two are the same axis.
 2. Return `nz == nx`, the elementwise equality of the two name vectors.

# Arguments

  - `nz`: Feature names.
  - `nx`: Asset names.

# Returns

  - `Bool`.

# Related

  - [`port_opt_view`](@ref)
  - [`FeatureDistance`](@ref)
"""
function features_are_assets(nz::Option{<:VecStr}, nx::Option{<:VecStr})::Bool
    return !isnothing(nz) && !isnothing(nx) && nz == nx
end
"""
    feature_matrix_view(Z::Nothing, sq::Bool, i, j)
    feature_matrix_view(Z::MatNum, sq::Bool, i, j)
    feature_matrix_view(Z::MatNum, sq::Bool, i, j::Colon)
    feature_matrix_view(Z::Arr3Num, sq::Bool, i, j)
    feature_matrix_view(Z::Arr3Num, sq::Bool, i, j::Colon)

Subselect a carried feature matrix by observations `i` and assets `j`.

`Z` is assets-major, so the asset index `j` addresses axis 1 of a static feature matrix and axis 2 of a time-varying one. The static shape has no observation axis and therefore ignores `i` — the same asymmetry the two [`port_opt_view`](@ref) arities have for `ivpa`.

When `sq` is `true` the feature axis is the asset axis ([`features_are_assets`](@ref)) and is sliced by `j` as well. A `Colon` asset index touches neither axis, so a static feature matrix passes through unchanged rather than being wrapped in a no-op view — the same passthrough `ivpa` gets when only observations are selected.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and no method copies `Z`.

 1. `Z` is `nothing`: return `nothing`.
 2. `Z` is a `MatNum` and `j` is a `Colon`: return `Z` itself. The asset index reaches neither axis, so no view is built.
 3. `Z` is a `MatNum`, which is `assets × features`: return `view(Z, j, j)` when `sq` is `true`, and `view(Z, j, :)` otherwise. Axis 1 is the assets, and axis 2 is the features. The observation index `i` is not read, because a static feature matrix has no observation axis.
 4. `Z` is an `Arr3Num` and `j` is a `Colon`: return `view(Z, i, :, :)`. Only the leading observation axis is selected.
 5. `Z` is an `Arr3Num`, which is `observations × assets × features`: return `view(Z, i, j, j)` when `sq` is `true`, and `view(Z, i, j, :)` otherwise. Axis 1 is the observations, axis 2 is the assets, and axis 3 is the features.

# Arguments

  - `Z`: Feature matrix.
  - `sq`: Whether the feature axis is the asset axis.
  - `i`: Observation indices.
  - `j`: Asset indices.

# Returns

  - A view of `Z`, or `nothing`.

# Related

  - [`features_are_assets`](@ref)
  - [`port_opt_view`](@ref)
"""
function feature_matrix_view(::Nothing, ::Bool, ::Any, ::Any)
    return nothing
end
function feature_matrix_view(Z::MatNum, sq::Bool, ::Any, j)
    return sq ? view(Z, j, j) : view(Z, j, :)
end
function feature_matrix_view(Z::MatNum, ::Bool, ::Any, ::Colon)
    return Z
end
function feature_matrix_view(Z::Arr3Num, sq::Bool, i, j)
    return sq ? view(Z, i, j, j) : view(Z, i, j, :)
end
function feature_matrix_view(Z::Arr3Num, ::Bool, i, ::Colon)
    return view(Z, i, :, :)
end
"""
    feature_row_indices(Z::Nothing, ts_new, ts_old) -> Colon
    feature_row_indices(Z::MatNum, ts_new, ts_old) -> Colon
    feature_row_indices(Z::Arr3Num, ts_new::Nothing, ts_old)
    feature_row_indices(Z::Arr3Num, ts_new, ts_old) -> VecInt

Recover the positional row indices of a time-varying feature matrix from a timestamp window.

A feature matrix is a plain array, so its observation axis is parallel to the carrier's clock positionally rather than aligned by timestamp. Whenever a routine selects rows of `X` by timestamp, the surviving timestamps are matched back into the original clock to recover the rows `Z` must keep. A surviving timestamp absent from that clock throws: it means the row bookkeeping has been broken (a synthesised timestamp, or an outer join that introduced a row `X` never had), and slicing `Z` positionally from there would silently pair each asset with another period's features.

Two sites use it. At **price level** the clock is `TimeSeries.timestamp(X)` and the selection is a timestamp window. At the **cross-validation assembly seam** the clock is `ReturnsResult.ts` and the selection is a fold: [`fold_row_indices`](@ref) recovers a fold's rows from the timestamps its view of the returns already carries, which is why `ts` must be unique — it *keys* the observation axis rather than merely labelling it.

The static and absent shapes have no observation axis, so they return `Colon` and cost nothing.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method.

 1. `Z` is `nothing` or a `MatNum`: return `Colon()`. Neither shape has an observation axis, so there is no row to recover and the timestamps are not read.
 2. `Z` is an `Arr3Num` and `ts_new` is `nothing`: throw. The selection kept no timestamp, so the rows to keep cannot be named.
 3. `Z` is an `Arr3Num`: match `ts_new` into `ts_old` with `indexin`, giving `rows`, the position each surviving timestamp holds in the original clock. Check that no entry of `rows` is `nothing`. Return `rows` as a `Vector{Int}`.

# Arguments

  - `Z`: Feature matrix.
  - `ts_new`: Timestamps surviving the selection.
  - `ts_old`: Timestamps of the clock `Z`'s observation axis is parallel to.

# Validation

  - `ts_new` is not `nothing` when `Z` is time-varying.
  - Every entry of `ts_new` appears in `ts_old`.

# Returns

  - `Colon` for a static or absent `Z`; otherwise the row indices, as a `Vector{Int}`.

# Related

  - [`feature_matrix_view`](@ref)
  - [`PricesResult`](@ref)
  - [`prices_to_returns`](@ref)
  - [`fold_row_indices`](@ref)
"""
function feature_row_indices(::Nothing, ::Any, ::Any)
    return Colon()
end
function feature_row_indices(::MatNum, ::Any, ::Any)
    return Colon()
end
function feature_row_indices(::Arr3Num, ::Nothing, ::Any)
    return throw(ArgumentError("a time-varying feature matrix (Z) has its observation axis parallel to the price timestamps, but no timestamps survived the conversion, so the rows of Z to keep cannot be recovered. Pass a static assets × features Z instead."))
end
function feature_row_indices(::Arr3Num, ts_new, ts_old)
    rows = indexin(ts_new, ts_old)
    missed = findfirst(isnothing, rows)
    @argcheck(isnothing(missed),
              ArgumentError("a time-varying feature matrix (Z) has its observation axis parallel to the price timestamps, but the timestamp $(ts_new[missed]) selected here is absent from them, so the row of Z it corresponds to cannot be recovered. This happens when the surviving timestamps are not a subset of the original clock — a `collapse_args` timestamp function that synthesises timestamps, or an outer join that introduced rows the asset prices never had. Pass a static assets × features Z, or align the feature matrix to the price clock first."))
    return Vector{Int}(rows)
end
"""
$(DocStringExtensions.TYPEDEF)

A container for aligned, time-indexed price-level data.

`PricesResult` is the prices-level mirror of [`ReturnsResult`](@ref): it bundles asset prices with optional factor, benchmark, and implied volatility series, all as `TimeSeries.TimeArray`s. It is the input to price-level preprocessing estimators and prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline cross-validation via [`port_opt_view`](@ref).

The asset price series `X` is the master clock: [`port_opt_view`](@ref) selects observation windows on `X` and aligns the other series to the selected timestamps.

The feature matrix `Z` is the exception to that alignment. It is a plain array, not a `TimeArray` — `TimeSeries.jl` has no 3-dimensional `TimeArray`, and the static shape has no clock at all — so it cannot be aligned by timestamp, only indexed positionally. Its axes are therefore held *parallel* to `X`: the asset axis to `TimeSeries.colnames(X)`, and, for the time-varying shape, the observation axis to `TimeSeries.timestamp(X)` row for row. Every routine that drops an asset or an observation from `X` must drop it from `Z` in the same step, which is what [`port_opt_view`](@ref), [`MissingDataFilter`](@ref) and [`prices_to_returns`](@ref) do.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesResult(;
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing,
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        nz::Option{<:VecStr} = nothing,
        Z::Option{<:MatNum_Arr3Num} = nothing,
    ) -> PricesResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(X)`.
  - If `F` is not `nothing`: `!isempty(F)`.
  - If `B` is not `nothing`: `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`: `!isempty(iv)`, `all(x -> x >= 0, values(iv))`, `all(x -> isfinite(x), values(iv))`, and `size(values(iv), 2) == size(values(X), 2)`.
  - If `ivpa` is not `nothing`: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(values(X), 2)`.
  - `nz` and `Z` are both `nothing` or both given; see [`check_names_and_feature_matrix`](@ref) for the shape rules, which bind `Z`'s asset axis to `size(values(X), 2)` and, for a time-varying `Z`, its observation axis to `size(values(X), 1)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> size(values(pr.X))
(3, 2)
```

# Related

  - [`AbstractPricesResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`Num_VecNum`](@ref)
  - [`MatNum_Arr3Num`](@ref)
  - [`check_names_and_feature_matrix`](@ref)
"""
@concrete struct PricesResult <: AbstractPricesResult
    """
    Asset price data (observations × assets). The master clock for timestamp-window slicing.
    """
    X
    """
    Optional factor price data (observations × factors).
    """
    F
    """
    Optional benchmark price data (observations × 1) or (observations × assets).
    """
    B
    """
    Optional implied volatility data (observations × assets).
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    $(field_dict[:nz_feat])
    """
    nz
    """
    Optional feature matrix, static (assets × features) or time-varying (observations × assets × features). Not a `TimeArray`: its axes are held positionally parallel to `X`.
    """
    Z
    function PricesResult(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray},
                          B::Option{<:TimeSeries.TimeArray},
                          iv::Option{<:TimeSeries.TimeArray}, ivpa::Option{<:Num_VecNum},
                          nz::Option{<:VecStr}, Z::Option{<:MatNum_Arr3Num})
        @argcheck(!isempty(X), IsEmptyError)
        if !isnothing(F)
            @argcheck(!isempty(F), IsEmptyError)
        end
        if !isnothing(B)
            @argcheck(!isempty(B), IsEmptyError)
            @argcheck(size(values(B), 2) in (1, size(values(X), 2)), DimensionMismatch)
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(values(iv), :iv)
            @argcheck(size(values(iv), 2) == size(values(X), 2), DimensionMismatch)
        end
        if !isnothing(ivpa)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(values(X), 2), DimensionMismatch)
            end
        end
        check_names_and_feature_matrix(nz, Z, size(values(X), 2), size(values(X), 1),
                                       "size(values(X), 2)")
        return new{typeof(X), typeof(F), typeof(B), typeof(iv), typeof(ivpa), typeof(nz),
                   typeof(Z)}(X, F, B, iv, ivpa, nz, Z)
    end
end
function PricesResult(; X::TimeSeries.TimeArray,
                      F::Option{<:TimeSeries.TimeArray} = nothing,
                      B::Option{<:TimeSeries.TimeArray} = nothing,
                      iv::Option{<:TimeSeries.TimeArray} = nothing,
                      ivpa::Option{<:Num_VecNum} = nothing, nz::Option{<:VecStr} = nothing,
                      Z::Option{<:MatNum_Arr3Num} = nothing)::PricesResult
    return PricesResult(X, F, B, iv, ivpa, nz, Z)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `PricesResult` for the observation window `i` and the assets `j` of the asset price series `X`.

The asset price series is the master clock: `i` selects rows of `X`, and the factor, benchmark, and implied volatility series are aligned to the selected timestamps (rows whose timestamps are absent from a series are dropped from that series). `j` selects asset columns and defaults to `:`, so a call giving only `i` is an observation window over the whole universe.

# Algorithm

The method that Julia selects is the algorithm. The timestamp methods do the work, and the integer method routes into them.

 1. `i` and `j` are both `Colon`: return `pr` itself. No view is built.

 2. `i` is a vector of timestamps and `j` is a `Colon`: index `X`, `F`, `B` and `iv` by the timestamps `i`. Recover the rows of a time-varying `Z` from the surviving timestamps with [`feature_row_indices`](@ref), and view `Z` on that observation axis with [`feature_matrix_view`](@ref). Carry `ivpa` and `nz` through untouched, because the asset index reaches neither. Rebuild the [`PricesResult`](@ref).

 3. `i` is a vector of timestamps and `j` is a vector of asset indices:

     1. Index `X` by the timestamps `i`, then keep the asset columns `j`.
     2. Index `F` by the timestamps `i` alone. `j` is an asset index, and the factors are a separate axis, so every factor column is kept.
     3. Index `B` by the timestamps `i`. Keep its columns `j` when `B` holds one column per asset, and keep its single column otherwise. The test is `B`'s own width, because a shared benchmark has one column to give whatever `j` asks for.
     4. Index `iv` by the timestamps `i` and the asset columns `j`, and view `ivpa` at `j`.
     5. Read `sq` from [`features_are_assets`](@ref) on `nz` and the asset names of `X`. When `sq` is `true`, view `nz` at `j` as well.
     6. Recover the rows of a time-varying `Z` with [`feature_row_indices`](@ref), and view `Z` at those rows and the assets `j` with [`feature_matrix_view`](@ref). A static `Z` has no observation axis and is viewed on the assets alone.
     7. Rebuild the [`PricesResult`](@ref).

 4. `i` and `j` are integer indices, ranges or `Colon`s: read the timestamps `TimeSeries.timestamp(pr.X)[i]`, and call step 2 or step 3 with them. This is the method a caller reaches with `port_opt_view(pr, 2:3)`.

# Arguments

  - `pr`: A `PricesResult` object.
  - `i`: Observation window into the rows of `pr.X`. Either integer indices (`AbstractVector{<:Integer}`, `AbstractRange`, or `Colon`) or a vector of timestamps (`AbstractVector{<:Dates.AbstractTime}`).
  - `j`: Asset window into the columns of `pr.X`. Integer indices, an `AbstractRange`, or `Colon` for the whole universe.

# Returns

  - `new_pr::PricesResult`: A new `PricesResult` containing only the data for the selected window.

# Details

  - A `Colon` asset index leaves `X`, `B`, `iv` and `ivpa` alone, which is why `ivpa` passes through untouched on the observation-only arity and is viewed at `j` on the other.
  - A static `Z` has no observation axis, so a `Colon` asset index passes it through unchanged rather than wrapping it in a no-op view.
  - When `Z`'s features *are* the assets, the asset subselection slices its feature axis and `nz` too (see [`features_are_assets`](@ref)).

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> pv = PortfolioOptimisers.port_opt_view(pr, 2:3);

julia> first(timestamp(pv.X))
2020-01-02

julia> size(values(pv.X))
(2, 2)
```

# Related

  - [`PricesResult`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::PricesResult, ::Colon, ::Colon)
    return pr
end
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},
                       ::Colon = :)
    X = pr.X[i]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    B = isnothing(pr.B) ? nothing : pr.B[i]
    iv = isnothing(pr.iv) ? nothing : pr.iv[i]
    rows = feature_row_indices(pr.Z, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X))
    Z = feature_matrix_view(pr.Z, false, rows, :)
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = pr.ivpa, nz = pr.nz, Z = Z)
end
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},
                       j::AbstractVector)
    X = pr.X[i][TimeSeries.colnames(pr.X)[j]]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    #! A benchmark is either one column per asset or a single shared column
    #! (the PricesResult constructor admits no other width). Only the first is
    #! indexed by the asset index; slicing the second by `j` reads past its one
    #! column. The test is B's own width, never `length(j)`.
    B = if isnothing(pr.B)
        nothing
    elseif length(TimeSeries.colnames(pr.B)) == size(values(pr.X), 2)
        pr.B[i][TimeSeries.colnames(pr.B)[j]]
    else
        pr.B[i]
    end
    iv = isnothing(pr.iv) ? nothing : pr.iv[i][TimeSeries.colnames(pr.iv)[j]]
    ivpa = nothing_scalar_array_view(pr.ivpa, j)
    sq = features_are_assets(pr.nz, string.(TimeSeries.colnames(pr.X)))
    rows = feature_row_indices(pr.Z, TimeSeries.timestamp(X), TimeSeries.timestamp(pr.X))
    nz = sq ? nothing_scalar_array_view(pr.nz, j) : pr.nz
    Z = feature_matrix_view(pr.Z, sq, rows, j)
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
function port_opt_view(pr::PricesResult,
                       i::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :,
                       j::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :)
    return port_opt_view(pr, TimeSeries.timestamp(pr.X)[i], j)
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
        nz::Option{<:VecStr} = nothing,
        Z::Option{<:MatNum_Arr3Num} = nothing,
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
  - If `iv` is not `nothing`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `all(x -> isfinite(x), iv)`, and `size(iv) == size(X)`.
  - `ivpa` is validated in that same branch, so it is checked only when `iv` is given: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`, and, if a vector, `length(ivpa) == size(iv, 2)`. The bound is strict — a zero adjustment is rejected. An `ivpa` passed without an `iv` reaches no check, because it has no implied volatility to adjust.
  - `nz` and `Z` are both `nothing` or both given; see [`check_names_and_feature_matrix`](@ref) for the shape rules, which bind `Z`'s asset axis to `length(nx)` and, for a time-varying `Z`, its observation axis to `size(X, 1)`.

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
    nz ┼ nothing
     Z ┴ nothing
```

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
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
    $(field_dict[:nz_feat])
    """
    nz
    """
    Optional feature matrix, static (assets × features) or time-varying (observations × assets × features).
    """
    Z
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, nb::Option{<:VecStr},
                           B::Option{<:VecNum_MatNum}, ts::Option{<:VecDate},
                           iv::Option{<:MatNum}, ivpa::Option{<:Num_VecNum},
                           nz::Option{<:VecStr}, Z::Option{<:MatNum_Arr3Num})
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
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X),
                      DimensionMismatch("implied volatilities (iv) must match asset returns (X) in size, got size(iv) = $(size(iv)) and size(X) = $(size(X))"))
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(iv, 2),
                          DimensionMismatch("the implied-volatility risk-premium adjustment (ivpa), when a vector, must have one entry per asset (implied-volatility column), got length(ivpa) = $(length(ivpa)) and size(iv, 2) = $(size(iv, 2))"))
            end
        end
        check_names_and_feature_matrix(nz, Z, isnothing(nx) ? nothing : length(nx),
                                       isnothing(X) ? nothing : size(X, 1), "length(nx)")
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ts), typeof(iv), typeof(ivpa), typeof(nz), typeof(Z)}(nx, X, nf,
                                                                                F, nb, B,
                                                                                ts, iv,
                                                                                ivpa, nz, Z)
    end
end
function ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                       nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                       nb::Option{<:VecStr} = nothing, B::Option{<:VecNum_MatNum} = nothing,
                       ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                       ivpa::Option{<:Num_VecNum} = nothing, nz::Option{<:VecStr} = nothing,
                       Z::Option{<:MatNum_Arr3Num} = nothing)::ReturnsResult
    return ReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa, nz, Z)
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
 5. Read `sq` from [`features_are_assets`](@ref) on `nz` and `nx`. When `sq` is `true`, view `nz` at `i` as well, because the feature axis is the asset axis.
 6. View the feature matrix with [`feature_matrix_view`](@ref) at `i` on the asset axis. The observation index is a `Colon`, so a time-varying `Z` keeps every observation.
 7. Rebuild the [`ReturnsResult`](@ref). The factor names `nf`, the factor returns `F` and the timestamps `ts` pass through untouched, because none of the three has an asset axis.

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Indices of the assets to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified index.

# Details

  - `Z` is carried assets-major, so the asset axis is axis 1 when `Z` is static and axis 2 when `Z` is time-varying.

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
    nz ┼ nothing
     Z ┴ nothing

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
    nz ┼ nothing
     Z ┴ nothing
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
 6. Read `sq` from [`features_are_assets`](@ref) on `nz` and `nx`. When `sq` is `true`, view `nz` at `j` as well.
 7. View the feature matrix with [`feature_matrix_view`](@ref) at the observations `i` and the assets `j`. A static `Z` has no observation axis and ignores `i`.
 8. Rebuild the [`ReturnsResult`](@ref).

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified indices.

# Details

  - A static `Z` ignores the observation index `i`, which is the same asymmetry `ivpa` has.

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
    nz ┼ nothing
     Z ┴ nothing

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
    nz ┼ nothing
     Z ┴ nothing
```

* * *

    port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)

Erroring tripwire for [`AbstractReturnsResult`](@ref) subtypes that do not implement [`port_opt_view`](@ref).

Without it, the universal leaf fallback `port_opt_view(x, i, args...)` would hand back the returns result *unsubselected*, and a meta-optimiser or cross-validation fold would silently train on the full universe. Returns data is never a leaf value, so an unhandled subtype is a missing method, not a pass-through.

Subtypes carrying a feature matrix owe it the same treatment as `X`: subselect its asset axis on every arity, its observation axis on the arities that take one, and — when its features *are* the assets ([`features_are_assets`](@ref)) — its feature axis as well. A feature matrix that survives a fold unsliced is the same silent-wrongness as an unsliced returns matrix, one level down: the distance it produces is finite, plausible, and computed over the wrong universe. [`feature_matrix_view`](@ref) implements the rule; the [`ReturnsResult`](@ref) methods are the reference.

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
    sq = features_are_assets(rd.nz, rd.nx)
    nz = sq ? nothing_scalar_array_view(rd.nz, i) : rd.nz
    Z = feature_matrix_view(rd.Z, sq, :, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, nb = nb, B = B, ts = rd.ts,
                         iv = iv, ivpa = ivpa, nz = nz, Z = Z)
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
    sq = features_are_assets(rd.nz, rd.nx)
    nz = sq ? nothing_scalar_array_view(rd.nz, j) : rd.nz
    Z = feature_matrix_view(rd.Z, sq, i, j)
    return ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, ts = ts, iv = iv,
                         ivpa = ivpa, nz = nz, Z = Z)
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
    Prices_RR

Union of the two data levels cross-validation folds can be computed on: returns-level ([`AbstractReturnsResult`](@ref)) and price-level ([`AbstractPricesResult`](@ref)) data.

Fold generation only needs an observation count ([`cv_nobs`](@ref)) and a timestamp vector ([`cv_timestamps`](@ref)), so [`Base.split`](@ref) and [`n_splits`](@ref) accept either level. Price-level splitting is what lets a `Pipeline` be cross-validated on its *input* rows, keeping stateful preprocessing inside the fold.
"""
const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve one side of a train/test split into a row count.

A size is either an `Integer` count of observations, or an `AbstractFloat` fraction of them in `(0, 1)`. Counts saturate at `N` (asking for more rows than exist takes all of them); the [`safe_index`](@ref) window guards then reject a split that leaves either side empty.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and the two mean different things: a count and a fraction.

 1. `s` is an `Integer`, so it is a count of rows: check that `s > 0`, and return `min(Int(s), N)`. A count larger than the data takes every row.
 2. `s` is an `AbstractFloat`, so it is a fraction of the rows: check that `0 < s < 1`, and return `clamp(floor(Int, s * N), 1, N)`. The fraction rounds **down** to whole rows, and the clamp keeps a small fraction of a short window from resolving to zero rows.

# Arguments

  - `s`: One side of the split, as a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`).
  - `N`: Number of observations available.
  - `name`: Symbolic name of the side, displayed in error messages.

# Validation

  - `s > 0` when `s` is an `Integer`.
  - `0 < s < 1` when `s` is an `AbstractFloat`.

# Returns

  - `n::Int`: The number of rows the side takes, in `1:N`.

# Related

  - [`safe_index`](@ref)
  - [`TrainTestSplit`](@ref)
"""
function split_count(s::Integer, N::Integer, name::Symbol)::Int
    @argcheck(s > zero(s),
              DomainError(s, "the $name of a train/test split must be > 0, got $s"))
    return min(Int(s), N)
end
function split_count(s::AbstractFloat, N::Integer, name::Symbol)::Int
    @argcheck(zero(s) < s < one(s),
              DomainError(s,
                          "the $name of a train/test split must lie in (0, 1) when given as a fraction, got $s"))
    return clamp(floor(Int, s * N), 1, N)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the `(train, test)` observation ranges of a holdout split over `N` time-ordered rows.

Training rows come from the head of the data and test rows from the tail, so the test window is always the most recent one. Each size is a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`), resolved by [`split_count`](@ref).

  - **Neither given**: the split falls at `D` (75 % train, 25 % test).
  - **One given**: the other side is its complement, so the two windows partition the data.
  - **Both given**: the head supplies `lo` training rows, the tail supplies `hi` test rows, and any rows between them are **embargoed** — they belong to neither window. This is how a gap between train and test is expressed. The gap is declared by the two sizes and nothing else; a rule that derives one from the label horizon belongs to the purged cross-validators ([`CombinatorialCrossValidation`](@ref)), not here.

# Algorithm

 1. Resolve the two window lengths `N_l` and `N_h`, through the branch that `lo` and `hi` select:

     1. Neither is given: take `n = clamp(floor(Int, D * N), 1, N)`, then `N_l = n` and `N_h = N - n`.
     2. Only `lo` is given: resolve it with [`split_count`](@ref), then `N_l = n` and `N_h = N - n`.
     3. Only `hi` is given: resolve it with [`split_count`](@ref), then `N_l = N - n` and `N_h = n`.
     4. Both are given: resolve each with [`split_count`](@ref) on its own. Neither is the complement of the other, so the rows between the two windows are embargoed.

 2. Check that both windows are non-empty, and that the two do not overlap.

 3. Return the two ranges `1:N_l` and `(N - N_h + 1):N`. The training window is the head of the data, and the test window is the tail, so the embargoed rows sit between them.

# Arguments

  - `lo`: Training rows, as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `hi`.
  - `hi`: Test rows, likewise; `nothing` takes the complement of `lo`.
  - `N`: Number of observations available.
  - `D = 0.75`: Training fraction taken when neither size is given.

# Validation

  - Both windows are non-empty. A split whose sizes saturate the data on one side (`train_size = N`) leaves nothing to test on and throws.
  - The windows do not overlap: `lo + hi <= N`.

# Returns

  - `(train, test)`: The training and test row ranges, as two `UnitRange{Int}`s.

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`split_count`](@ref)
  - [`CombinatorialCrossValidation`](@ref)
"""
function safe_index(lo::Option{<:Number}, hi::Option{<:Number}, N::Integer, D = 0.75)
    N_l, N_h = if isnothing(lo) && isnothing(hi)
        n = clamp(floor(Int, D * N), 1, N)
        n, N - n
    elseif isnothing(hi)
        n = split_count(lo, N, :train_size)
        n, N - n
    elseif isnothing(lo)
        n = split_count(hi, N, :test_size)
        N - n, n
    else
        split_count(lo, N, :train_size), split_count(hi, N, :test_size)
    end
    @argcheck(N_l > 0 && N_h > 0,
              ArgumentError("a train/test split of $N observations must leave both windows non-empty, got $N_l training and $N_h test observations"))
    @argcheck(N_l + N_h <= N,
              ArgumentError("the training and test windows of a train/test split must not overlap, but $N_l training and $N_h test observations exceed the $N available; rows between the two windows are embargoed, so their sizes may sum to less than $N but never to more"))
    return 1:N_l, (N - N_h + 1):N
end
"""
    train_test_split(rd::ReturnsResult; train_size, test_size) -> (train, test)
    train_test_split(pr::PricesResult; train_size, test_size) -> (train, test)

Cut price- or returns-level data into a training window (the head) and a held-out test window (the tail).

The free-function form of [`TrainTestSplit`](@ref); the windows are [`port_opt_view`](@ref)s, so no data is copied. See [`safe_index`](@ref) for the sizing rules — complement when one side is given, embargo when both are.

# Algorithm

 1. Read the observation count `N` from the asset data: `size(rd.X, 1)` at returns level, and `size(TimeSeries.values(pr.X), 1)` at price level.
 2. Resolve the two row ranges with [`safe_index`](@ref).
 3. Return a [`port_opt_view`](@ref) of each range. The returns-level method passes a `Colon` asset index after the row range, because that arity indexes observations first and assets second.

# Arguments

  - `rd`/`pr`: The data to split.
  - `train_size`: Training rows as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
  - `test_size`: Test rows, likewise; `nothing` takes the complement of `train_size`. With neither given the split is 75/25.

# Returns

  - `(train, test)`: The two windows, of the same type as the input.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = [\"A\"], X = reshape(collect(0.1:0.1:1.0), 10, 1));

julia> train, test = train_test_split(rd; test_size = 0.2);

julia> size(train.X, 1), size(test.X, 1)
(8, 2)
```

# Related

  - [`TrainTestSplit`](@ref)
  - [`safe_index`](@ref)
"""
function train_test_split(rd::ReturnsResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(rd.X, 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train, :), port_opt_view(rd, test, :)
end
function train_test_split(rd::PricesResult; train_size::Option{<:Number} = nothing,
                          test_size::Option{<:Number} = nothing)
    N = size(TimeSeries.values(rd.X), 1)
    train, test = safe_index(train_size, test_size, N)
    return port_opt_view(rd, train), port_opt_view(rd, test)
end
"""
    returns_result_picker(rd::ReturnsResult, brt::Bool) -> ReturnsResult

Return a `ReturnsResult` appropriate for benchmark-tracking optimisations.

This helper inspects the `ReturnsResult`'s benchmark field `B` and the boolean flag `brt` (benchmark-tracking). If `brt` is `true` and a benchmark `B` is present it returns a new `ReturnsResult` in which asset returns `X` have the benchmark removed (i.e. `X - B` or broadcast `X .- B` for vector benchmarks). If `brt` is `false` or no benchmark is present, the original `ReturnsResult` is returned unchanged.

# Algorithm

The first step is a method selected on the field type of `B`, so a carrier with no benchmark runs no branch at all.

 1. `rd` carries no benchmark, because its `B` field is `Nothing`: return `rd` itself.
 2. `brt` is `false`: return `rd` itself.
 3. `brt` is `true`: subtract the benchmark from the asset returns, giving `X`. A vector benchmark subtracts by broadcast, `rd.X .- rd.B`, which takes one benchmark value per observation from every asset column. A matrix benchmark subtracts elementwise, `rd.X - rd.B`.
 4. Rebuild the [`ReturnsResult`](@ref) from `X`, and leave `nb` and `B` unset. The benchmark is spent on the subtraction, which is what makes a second call return its argument unchanged.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset, factor and/or benchmark returns.
  - `brt`: Boolean flag indicating whether benchmark-tracking behaviour should be applied. When `true`, asset returns are adjusted by subtracting the benchmark `B` (if present).

# Returns

  - `rd::ReturnsResult`:

      + If `brt` is `true` and a benchmark `B` is present: A new `ReturnsResult` with adjusted asset returns
      + Otherwise: The `rd` is returned unchanged.

# Details

  - The original `rd` is never modified. An adjustment builds a new [`ReturnsResult`](@ref).
  - Other fields (`nx`, `nf`, `F`, `ts`, `iv`, `ivpa`, `nz`, `Z`) are preserved in the returned object.
  - `nb` and `B` are **not** carried over: the benchmark has been spent on the subtraction, so the returned object holds `nothing` for both. This is what makes the adjustment idempotent — a second call has no benchmark left to subtract and returns its argument unchanged.

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
    nz ┼ nothing
     Z ┴ nothing

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
    nz ┼ nothing
     Z ┴ nothing

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
    nz ┼ nothing
     Z ┴ nothing

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
                      ivpa = rd.ivpa, nz = rd.nz, Z = rd.Z)
    end
end
"""
    apply_impute_method(X, impute_method) -> Any

Apply the `impute_method` given to [`prices_to_returns`](@ref) to the price table `X`.

`Impute` is a weak dependency, so this is the seam that keeps it optional. The identity method on
`nothing` is the default path and lives here; the method accepting an `Impute.Imputor` is supplied
by `PortfolioOptimisersImputeExt` and only exists once the caller has run `using Impute`. Anything
else throws.

# Algorithm

The method that Julia selects is the algorithm. Each step is one method, and the second lives in an extension.

 1. `impute_method` is `nothing`: return `X` unchanged. This is the default path, and it loads nothing.
 2. `impute_method` is an `Impute.Imputor`: apply it to `X`. `PortfolioOptimisersImputeExt` supplies this method, so it exists only after the caller runs `using Impute`.
 3. Any other `impute_method`: read whether the extension is loaded with `Base.get_extension`, and throw an `ArgumentError` that names which of the two mistakes happened. `Impute` not being loaded and a wrong type are different mistakes, and the caller cannot tell them apart from the type alone.

# Arguments

  - `X`: Price table (a `DataFrames.DataFrame` as built by [`prices_to_returns`](@ref)).
  - `impute_method`: `nothing` (no imputation), or an `Impute.Imputor` when `Impute` is loaded.

# Validation

  - Throws an `ArgumentError` for any `impute_method` that is neither `nothing` nor an
    `Impute.Imputor`, distinguishing "`Impute` is not loaded" from "wrong type". Note that
    [`Imputer`](@ref) is a PortfolioOptimisers estimator unrelated to `Impute.jl` and is not
    accepted here.

# Returns

  - `X`: Unchanged when `impute_method` is `nothing`, otherwise the imputed table.

# Related

  - [`prices_to_returns`](@ref)
  - [`Imputer`](@ref)
  - [`Impute`](https://github.com/invenia/Impute.jl)
"""
apply_impute_method(X, ::Nothing) = X
function apply_impute_method(::Any, impute_method)
    reason = if isnothing(Base.get_extension(@__MODULE__, :PortfolioOptimisersImputeExt))
        "`Impute` is not loaded, so no imputor can be recognised; run `using Impute` (adding it to your project if needed)"
    else
        "`Impute` is loaded, but a $(typeof(impute_method)) is not an `Impute.Imputor`"
    end
    return throw(ArgumentError("`impute_method` accepts only `nothing` or an `Impute.Imputor` such as `Impute.LOCF()` or `Impute.Interpolate()`: $(reason). Note that `Imputer` is a PortfolioOptimisers preprocessing estimator, entirely unrelated to `Impute.jl` despite the similar name — it fills gaps with per-asset statistics fitted on a training window and is used as a `Pipeline` step, not passed to `prices_to_returns`."))
end
"""
    prices_to_returns(
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing;
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
        ret_method::Symbol = :simple, padding::Bool = false,
        missing_col_percent::Number = 1.0,
        missing_row_percent::Option{<:Number} = 1.0,
        collapse_args::Tuple = (),
        map_func::Option{<:Function} = nothing,
        join_method::Symbol = :outer,
        impute_method = nothing,
        nz::Option{<:VecStr} = nothing,
        Z::Option{<:MatNum_Arr3Num} = nothing
    ) -> ReturnsResult

Convert `TimeSeries.TimeArray` price data to returns. Handles factor data, missing data,
imputation, and optional implied volatility information.

# Mathematical definition

Returns are computed from prices ``P_{t,i}`` as:

```math
\\begin{align}
r_{t,i} &= \\begin{cases}
(P_{t,i} - P_{t-1,i}) / P_{t-1,i} & \\text{simple} \\\\
\\ln(P_{t,i} / P_{t-1,i}) & \\text{log}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``r_{t,i}``: Return of asset ``i`` at time ``t``.
  - ``P_{t,i}``: Price of asset ``i`` at time ``t``.

`TimeSeries.percentchange` computes both branches through logarithms: the log return is ``\\ln P_{t,i} - \\ln P_{t-1,i}``, and the simple return is ``\\mathrm{expm1}`` of it. The two agree with the forms above to floating point rather than to the last bit, and both need a **positive** price. A negative price throws a `DomainError` from inside the logarithm, on the simple branch as well, and a zero price gives ``\\pm\\infty``.

A benchmark ``B`` is converted by the same rule and **carried alongside** the asset returns in the `B` field of the [`ReturnsResult`](@ref); it is not subtracted here. The subtraction that forms the excess return ``\\tilde{r}_{t,i} = r_{t,i} - b_{t,i}`` is done later, by [`returns_result_picker`](@ref), and only when the optimisation tracks the benchmark.

# Algorithm

 1. Check `X`, `missing_col_percent` and `missing_row_percent`. Read the asset names and the asset timestamps from `X`, and check `nz` and `Z` against them with [`check_names_and_feature_matrix`](@ref).
 2. Merge the factor prices `F` into `X` under `join_method`, and record the factor names.
 3. Merge the benchmark prices `B` into `X` under `join_method`, and record the benchmark names. A benchmark is one shared column, or one column per asset.
 4. Apply `map_func` to every entry, when one is given.
 5. Collapse the time series with `collapse_args`, when they are given. This is the step that changes the frequency.
 6. Convert the table to a `DataFrames.DataFrame`, and replace every `NaN` with `missing`. The two conventions for an absent price become one.
 7. Impute the missing entries with [`apply_impute_method`](@ref), which does nothing unless the caller gives an `Impute.Imputor`.
 8. Count the missing entries of the table, giving one count per row and one count per column.
 9. Drop each row whose count of missing columns exceeds `missing_col_percent` of the column total.
10. Drop each column whose count of missing rows exceeds `missing_row_percent` of the surviving row total. When `missing_row_percent` is `nothing`, keep instead the columns whose count equals the mode of the counts.
11. Drop every column that is still typed as missing, then every row that still holds a missing entry.
12. Convert the surviving prices to returns with `TimeSeries.percentchange` under `ret_method` and `padding`. This is the step that applies the formula above. When `padding` is `true` the first observation is kept and its return is `NaN`, so the returns keep the length of the price clock.
13. Split the surviving column names into the asset names `nx`, the factor names `nf`, the benchmark names `nb`, and the timestamp column, which gives `ts`.
14. Index the implied volatilities `iv` by `ts`, then check `iv` and `ivpa` against the surviving asset count.
15. Subselect the feature matrix. Read the surviving assets' positions `acols` in the original asset names, read `sq` from [`features_are_assets`](@ref), recover the surviving rows with [`feature_row_indices`](@ref), and view `Z` with [`feature_matrix_view`](@ref). Materialise the view with `Array`, and view `nz` at `acols` when `sq` is `true`.
16. Build the asset, factor and benchmark matrices from the surviving columns. A group whose columns all went is `nothing`.
17. Return the [`ReturnsResult`](@ref).

Step 9 counts the missing columns of a row, and step 10 counts the missing rows of a column. The name of each keyword reads as the axis it counts, and the axis it drops is the other one.

# Arguments

  - `X`: Asset price data (observations × assets).
  - `F`: Optional Factor price data (observations × factors).
  - `B`: Optional Benchmark price data (observations × assets) or (observations × 1).
  - `iv`: Optional Implied volatility data.
  - `ivpa`: Optional Implied volatility risk premium adjustment.
  - `ret_method`: Return calculation method (`:simple` or `:log`).
  - `padding`: Whether to pad missing values in returns calculation.
  - `missing_col_percent`: Maximum allowed fraction `(0, 1]` of missing **columns** in an observation row. A row above it is dropped. The name reads as the axis that is counted, not the axis that is dropped.
  - `missing_row_percent`: Maximum allowed fraction `(0, 1]` of missing **rows** in a column. A column above it is dropped. `nothing` keeps the columns whose missing count equals the mode of the counts instead, which is the shape of a panel whose assets share one history.
  - `collapse_args`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func`: Optional function to apply to the data before returns calculation.
  - `join_method`: How to join asset, factor data and benchmark data (`:outer`, `:inner`, etc.).
  - `impute_method`: Optional imputation method for missing data. `nothing`, or an `Impute.Imputor` — which requires `using Impute`, since `Impute` is a weak dependency loaded through `PortfolioOptimisersImputeExt`. Unrelated to [`Imputer`](@ref), which is a PortfolioOptimisers estimator and is not accepted here.
  - `nz`: Optional feature names.
  - `Z`: Optional feature matrix, static (assets × features) or time-varying (observations × assets × features), with its axes parallel to `X`'s columns and rows.

# Validation

  - `!isempty(X)`.
  - `0 < missing_col_percent <= 1`
  - `0 < missing_row_percent <= 1`.
  - If `F` is not `nothing`, `!isempty(F)`.
  - If `B` is not `nothing`, `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`, the timestamps of the merged data matrix must be a subset of `TimeSeries.timestamp(iv)`, then `iv = values(iv)`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `all(x -> isfinite(x), iv)`, and `size(iv) == size(X)`.
  - `ivpa` is validated in that same branch, so it is checked only when `iv` is given: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`, and, if a vector, `length(ivpa) == size(iv, 2)`. The bound is strict — a zero adjustment is rejected.

# Returns

  - `rr::ReturnsResult`: Struct containing asset/factor returns, names, time series, and optional implied volatility data.

# Details

  - Step 10 counts the missing entries of a column over **every** row the table had at step 8, and compares that count against the row total that **survived** step 9. A column can therefore be dropped for missing entries that sit only in rows already gone.

  - The benchmark is converted to returns by the same rule and carried in the `B` field. It is **not** subtracted from the asset returns; [`returns_result_picker`](@ref) does that, for a returns-tracking optimisation.

  - Carries the feature matrix `Z` across, subselected to the assets that survive the conversion — an asset dropped for being entirely missing takes its features with it, or the two matrices would desynchronise silently. A time-varying `Z` is also subselected to the surviving observations, matching them back into the original price timestamps ([`feature_row_indices`](@ref)); a surviving timestamp absent from that clock throws. Under `collapse_args` this gives the aggregated period the features of the row at its representative timestamp — last-observation semantics, matching [`LastObservation`](@ref).

  - Returns a `ReturnsResult` with asset/factor names, returns, timestamps, and optional implied volatility data.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100 101; 102 103; 104 105],
                     [\"A\", \"B\"])
3×2 TimeSeries.TimeArray{Int64, 2, Dates.Date, Matrix{Int64}} 2020-01-01 to 2020-01-03
┌────────────┬─────┬─────┐
│            │ A   │ B   │
├────────────┼─────┼─────┤
│ 2020-01-01 │ 100 │ 101 │
│ 2020-01-02 │ 102 │ 103 │
│ 2020-01-03 │ 104 │ 105 │
└────────────┴─────┴─────┘

julia> prices_to_returns(X)
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ Vector{Dates.Date}: [Dates.Date("2020-01-02"), Dates.Date("2020-01-03")]
    iv ┼ nothing
  ivpa ┼ nothing
    nz ┼ nothing
     Z ┴ nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
  - [`VecDate`](@ref)
  - [`Num_VecNum`](@ref)
  - [`TimeSeries`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type)
  - [`apply_impute_method`](@ref)
  - [`Impute`](https://github.com/invenia/Impute.jl)
"""
function prices_to_returns(X::TimeSeries.TimeArray,
                           F::Option{<:TimeSeries.TimeArray} = nothing;
                           B::Option{<:TimeSeries.TimeArray} = nothing,
                           iv::Option{<:TimeSeries.TimeArray} = nothing,
                           ivpa::Option{<:Num_VecNum} = nothing,
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Number = 1.0,
                           missing_row_percent::Option{<:Number} = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Option{<:Function} = nothing,
                           join_method::Symbol = :outer, impute_method = nothing,
                           nz::Option{<:VecStr} = nothing,
                           Z::Option{<:MatNum_Arr3Num} = nothing)
    @argcheck(!isempty(X), IsEmptyError)
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError)
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent), DomainError)
    end
    asset_names = string.(TimeSeries.colnames(X))
    asset_ts = TimeSeries.timestamp(X)
    check_names_and_feature_matrix(nz, Z, length(asset_names), length(asset_ts),
                                   "the number of asset price columns")
    factor_names = String[]
    benchmark_names = String[]
    if !isnothing(F)
        @argcheck(!isempty(F), IsEmptyError)
        factor_names = string.(TimeSeries.colnames(F))
        X = TimeSeries.merge(X, F; method = join_method)
    end
    if !isnothing(B)
        @argcheck(!isempty(B), IsEmptyError)
        benchmark_names = string.(TimeSeries.colnames(B))
        @argcheck(length(benchmark_names) in (1, length(asset_names)), DimensionMismatch)
        X = TimeSeries.merge(X, B; method = join_method)
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = TimeSeries.collapse(X, collapse_args...)
    end
    X = DataFrames.DataFrame(X)

    DataFrames.transform!(X,
                          2:DataFrames.DataAPI.ncol(X) .=>
                              DataFrames.ByRow((x) -> ifelse((isa(x, Number) && isnan(x)),
                                                             missing, x));
                          renamecols = false)
    X = apply_impute_method(X, impute_method)
    missing_mtx = ismissing.(Matrix(X[!, 2:end]))
    missings_cols = vec(count(missing_mtx; dims = 2))
    keep_rows = missings_cols .<= (DataFrames.DataAPI.ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, :]
    missings_rows = vec(count(missing_mtx; dims = 1))
    keep_cols = if !isnothing(missing_row_percent)
        missings_rows .<= DataFrames.DataAPI.nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, [true; keep_cols]]
    DataFrames.select!(X, DataFrames.InvertedIndices.Not(names(X, Missing)))
    DataFrames.dropmissing!(X)
    X = TimeSeries.percentchange(TimeSeries.TimeArray(X; timestamp = :timestamp),
                                 ret_method; padding = padding)
    X = DataFrames.DataFrame(X)
    col_names = names(X)
    nx = intersect(col_names, asset_names)
    nf = intersect(col_names, factor_names)
    nb = intersect(col_names, benchmark_names)
    oc = setdiff(col_names, union(nx, nf, nb))
    N = length(nx)
    ts = isempty(oc) ? nothing : vec(Matrix(X[!, oc]))
    if !isnothing(ts) && !isnothing(iv)
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)),
                  ArgumentError("ts must be a subset of the timestamps in iv"))
        iv = iv[ts]
    end
    if !isnothing(iv)
        iv = values(iv)
        assert_nonempty_nonneg_finite_val(iv, :iv)
        assert_nonempty_gt0_finite_val(ivpa, :ivpa)
        @argcheck(size(iv) == (DataFrames.DataAPI.nrow(X), N), DimensionMismatch)
        if isa(ivpa, VecNum)
            @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
        end
    end
    if !isnothing(Z)
        @argcheck(!isempty(nx),
                  IsEmptyError("every asset was dropped during the conversion, so the feature matrix (Z) has no asset axis left to bind to"))
        acols = Vector{Int}(indexin(nx, asset_names))
        sq = features_are_assets(nz, asset_names)
        Z = Array(feature_matrix_view(Z, sq, feature_row_indices(Z, ts, asset_ts), acols))
        nz = sq ? nz[acols] : nz
    end
    if isempty(nf)
        nf = nothing
        F = nothing
    else
        F = Matrix(X[!, nf])
    end
    if isempty(nb)
        nb = nothing
        B = nothing
    else
        B = length(nb) == 1 ? X[!, nb[1]] : Matrix(X[!, nb])
    end
    if isempty(nx)
        nx = nothing
        X = nothing
    else
        X = Matrix(X[!, nx])
    end
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, iv = iv,
                         ivpa = ivpa, nz = nz, Z = Z)
end
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

Abstract supertype for all preprocessing estimator types.

Preprocessing estimators transform price or returns data (prices-to-returns conversion, missing-data filtering, imputation) under a fit/apply contract. Fitting one on training data with [`fit_preprocessing`](@ref) produces a result carrying any fitted state — imputation parameters, thresholds, and the selected asset universe — which [`apply_preprocessing`](@ref) then replays on unseen data so train and test windows are transformed consistently. Stateless preprocessing estimators carry no state, and applying them is equivalent to running them.

They are ordinary estimators: they know nothing about pipelines. A `Pipeline` drives them through the same fit/apply verbs any other caller would use.

All concrete preprocessing estimators should subtype one of the two data-level subtypes:

  - [`AbstractPricesPreprocessingEstimator`](@ref): consumes and produces price-level data ([`PricesResult`](@ref)).
  - [`AbstractReturnsPreprocessingEstimator`](@ref): consumes and produces returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
abstract type AbstractPreprocessingEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce price-level data.

Concrete subtypes transform a [`PricesResult`](@ref) into another [`PricesResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
  - [`PricesResult`](@ref)
"""
abstract type AbstractPricesPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing estimators that consume and produce returns-level data.

Concrete subtypes transform a [`ReturnsResult`](@ref) into another [`ReturnsResult`](@ref).

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsPreprocessingEstimator <: AbstractPreprocessingEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all preprocessing result types.

Preprocessing results are produced by [`fit_preprocessing`](@ref) on training data. They carry the fitted state needed to apply the same transformation to unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators produce results that carry only their configuration.

All concrete preprocessing results should subtype one of the two data-level subtypes, [`AbstractPricesPreprocessingResult`](@ref) or [`AbstractReturnsPreprocessingResult`](@ref), so a caller can replay each fitted transformation at the data level it applies to.

# Related

  - [`AbstractResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingResult`](@ref)
"""
abstract type AbstractPreprocessingResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to price-level data ([`PricesResult`](@ref)).

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
"""
abstract type AbstractPricesPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for preprocessing results that apply to returns-level data ([`ReturnsResult`](@ref)).

# Related

  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractReturnsPreprocessingEstimator`](@ref)
"""
abstract type AbstractReturnsPreprocessingResult <: AbstractPreprocessingResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` counts as a missing observation in price-level data.

Price-level data stores absent observations either as `missing` or as `NaN` (the two conventions [`prices_to_returns`](@ref) already unifies).

# Algorithm

 1. Return `true` when `x` is `missing`.
 2. Return `true` when `x` is a `Number` and `isnan(x)` holds. The type test guards the call, because `isnan` is not defined for every value a price table can carry.
 3. Return `false` otherwise.

# Arguments

  - `x`: The value to test.

# Returns

  - `flag::Bool`: `true` when `x` is `missing` or a `NaN` number.

# Related

  - [`MissingDataFilter`](@ref)
  - [`Imputer`](@ref)
"""
function is_missing_value(x)::Bool
    return ismissing(x) || (isa(x, Number) && isnan(x))
end
"""
    fit_preprocessing(est::AbstractPreprocessingEstimator, data) -> fitted

Fit a preprocessing estimator on a data window and return the fitted object consumed by [`apply_preprocessing`](@ref).

The fitted object carries whatever state the transformation needs to be replayed consistently on unseen data — imputation parameters, thresholds, and the selected asset universe. Stateless preprocessing estimators return themselves.

# Interfaces

Concrete preprocessing estimators must implement:

  - `fit_preprocessing(est::MyPreprocessing, data) -> fitted`: Compute the fitted state from the training window.
  - `apply_preprocessing(fitted, data) -> data′`: Transform a data window with the fitted state.

# Arguments

  - `est`: The preprocessing estimator.
  - `data`: The training data window ([`PricesResult`](@ref) or [`ReturnsResult`](@ref) depending on the estimator's level).

# Returns

  - `fitted`: The fitted object, typically an [`AbstractPreprocessingResult`](@ref) or the estimator itself when stateless.

# Related

  - [`apply_preprocessing`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
"""
function fit_preprocessing(est::AbstractPreprocessingEstimator, data)
    return throw(ArgumentError("$(typeof(est)) subtypes AbstractPreprocessingEstimator but does not implement fit_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′."))
end
"""
    apply_preprocessing(fitted, data) -> data′

Transform a data window with a fitted preprocessing object.

Applying the fitted object produced by [`fit_preprocessing`](@ref) on the training window to an unseen (test) window replays the *same* transformation — the same asset universe, the same imputation parameters — so train and test data stay consistent and no information flows from test to train.

# Arguments

  - `fitted`: The fitted object returned by [`fit_preprocessing`](@ref) (an [`AbstractPreprocessingResult`](@ref), or a stateless estimator).
  - `data`: The data window to transform.

# Returns

  - `data′`: The transformed data window.

# Related

  - [`fit_preprocessing`](@ref)
  - [`AbstractPreprocessingEstimator`](@ref)
  - [`AbstractPreprocessingResult`](@ref)
"""
function apply_preprocessing(fitted::Union{<:AbstractPreprocessingEstimator,
                                           <:AbstractPreprocessingResult}, data)
    return throw(ArgumentError("$(typeof(fitted)) subtypes the preprocessing interface but does not implement apply_preprocessing. Extension authors: a preprocessing estimator must implement both halves of the interface, fit_preprocessing(est, data) -> fitted and apply_preprocessing(fitted, data) -> data′; a stateless estimator returns itself from fit_preprocessing and does the work here."))
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator reserving the tail of the observations as a held-out test window.

The estimator form of [`train_test_split`](@ref), and the way the holdout protocol enters a [`Pipeline`](@ref): as the **first** step, it hands the training window to every step downstream and stashes the test window in its fitted [`TrainTestSplitResult`](@ref). `fit_predict(pipe, data)` then evaluates the fitted workflow on that held-out window in one line.

It is the one preprocessing estimator that is not pinned to a data level: it splits whichever level the pipeline input provides, price or returns, since a holdout is a statement about *rows*, not about columns or units.

Replaying a fitted split on an unseen window is a **pass-through** — the fitted rows are training-window state, and applying them to new data would be meaningless — so `predict(res, future_data)` keeps working on genuinely new observations.

!!! warning

    A pipeline containing a `TrainTestSplit` may not also be cross-validated: the split and the cross-validator are two evaluation protocols, and cross-validation already defines its own train/test windows. [`search_cross_validation`](@ref) rejects such a pipeline rather than silently shaving a second holdout off every fold.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrainTestSplit(;
        train_size::Option{<:Number} = nothing,
        test_size::Option{<:Number} = nothing,
    ) -> TrainTestSplit

Keywords correspond to the struct's fields. Sizes follow [`safe_index`](@ref): a row count (`Integer`) or a fraction of the observations (`AbstractFloat` in `(0, 1)`); one side given makes the other its complement; both given embargoes the rows between them; neither given splits 75/25.

# Examples

```jldoctest
julia> pipe = Pipeline(;
                       steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                                EmpiricalPrior(), EqualWeighted()));

julia> pipe.names
("split", "returns", "prior", "opt")
```

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`Pipeline`](@ref)
"""
@concrete struct TrainTestSplit <: AbstractPreprocessingEstimator
    """
    Training observations as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
    """
    train_size
    """
    Test observations, likewise; `nothing` takes the complement of `train_size`.
    """
    test_size
    function TrainTestSplit(train_size::Option{<:Number}, test_size::Option{<:Number})
        return new{typeof(train_size), typeof(test_size)}(train_size, test_size)
    end
end
function TrainTestSplit(; train_size::Option{<:Number} = nothing,
                        test_size::Option{<:Number} = nothing)::TrainTestSplit
    return TrainTestSplit(train_size, test_size)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`TrainTestSplit`](@ref), carrying both windows of the holdout.

The `test` window is the payoff: it is the data the fitted pipeline has never seen, and what `fit_predict(pipe, data)` predicts on. The `train` window is kept alongside it so the raw data the workflow was fitted on is retrievable from the result rather than having to be re-derived.

Both are [`port_opt_view`](@ref)s of the input at whichever level the split ran (price or returns).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`TrainTestSplit`](@ref)
  - [`PipelineResult`](@ref)
"""
@concrete struct TrainTestSplitResult <: AbstractResult
    """
    The training window: the head of the observations, and the data every downstream step is fitted on.
    """
    train
    """
    The held-out test window: the tail of the observations, which no fitted step has seen.
    """
    test
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit a [`TrainTestSplit`](@ref) by cutting the data into its two windows.

Unlike the other preprocessing estimators, the fitted result is *not* replayed on unseen data: a holdout's rows are a fact about the fitting window alone, so [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) passes the window through unchanged.

# Algorithm

 1. [`fit_preprocessing`](@ref) calls [`train_test_split`](@ref) on the data, and returns the [`TrainTestSplitResult`](@ref) that holds both windows.
 2. [`apply_preprocessing`](@ref) on a [`TrainTestSplitResult`](@ref) returns its data argument unchanged.
 3. [`apply_preprocessing`](@ref) on a [`TrainTestSplit`](@ref) returns its data argument unchanged, so an unfitted step is a pass-through as well.

# Related

  - [`TrainTestSplit`](@ref)
  - [`train_test_split`](@ref)
"""
function fit_preprocessing(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    return train_test_split(tts, data)
end
function apply_preprocessing(::TrainTestSplitResult, data::Prices_RR)
    return data
end
function apply_preprocessing(::TrainTestSplit, data::Prices_RR)
    return data
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Split `data` under a [`TrainTestSplit`](@ref), returning both windows as a [`TrainTestSplitResult`](@ref).

The estimator-form counterpart of the keyword form: `train_test_split(rd; test_size = 0.2)` hands back a bare `(train, test)` tuple, while this hands back the same fitted result a pipeline's split step produces, so a holdout configured once can be reused verbatim inside and outside a [`Pipeline`](@ref).

# Algorithm

 1. Call the keyword form of [`train_test_split`](@ref) with `tts.train_size` and `tts.test_size`, giving the two windows.
 2. Wrap the pair in a [`TrainTestSplitResult`](@ref), in the order `(train, test)`.

# Related

  - [`TrainTestSplit`](@ref)
  - [`TrainTestSplitResult`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function train_test_split(tts::TrainTestSplit, data::Prices_RR)::TrainTestSplitResult
    train, test = train_test_split(data; train_size = tts.train_size,
                                   test_size = tts.test_size)
    return TrainTestSplitResult(train, test)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for returns-level preprocessing estimators that restrict the asset universe.

An asset selector answers one question on the training window — *which asset columns survive?* — and that answer is its fitted state. [`apply_preprocessing`](@ref) replays the fitted universe on unseen windows, so a selector is safe inside cross-validation: the selection is made on train data alone and never re-decided on test data.

Concrete subtypes implement a single method, [`select_assets`](@ref); the family shares one [`fit_preprocessing`](@ref) and one [`apply_preprocessing`](@ref). Selectors restrict *columns only*. Observation filtering is a price-level concern ([`MissingDataFilter`](@ref)), because a fitted transformation cannot decide which rows of an unseen window to drop without breaking the weights/returns alignment `assert_universe_aligned` enforces.

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

This is the single method a concrete [`AbstractAssetSelector`](@ref) must implement. It is called by [`fit_preprocessing`](@ref) on the *training* window only; the resulting universe is then replayed on every later window by [`apply_preprocessing`](@ref).

`rd` is read for `nx` and an `observations × assets` `X`; [`ClusterGroups`](@ref) also reads `rd.Z`, widening the implicit contract to `{nx, X, Z}` (see [`AbstractReturnsResult`](@ref)). A selector is fitted from returns data alone and never sees a prior result, so `z_src` has no referent here and no selector carries one.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Returns

  - `keep::BitVector`: `true` for each asset column to retain, `length(keep) == size(rd.X, 2)`.

# Related

  - [`AbstractAssetSelector`](@ref)
  - [`fit_preprocessing`](@ref)
"""
function select_assets(sel::AbstractAssetSelector, rd::AbstractReturnsResult)
    return throw(ArgumentError("$(typeof(sel)) subtypes AbstractAssetSelector but does not implement select_assets. Extension authors: every AbstractAssetSelector must define select_assets(sel, rd) returning a keep-mask over the asset columns of rd."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit any [`AbstractAssetSelector`](@ref) by recording the asset universe [`select_assets`](@ref) keeps.

# Algorithm

 1. Call [`select_assets`](@ref) on the training window, giving the keep-mask `keep`.
 2. Check that `keep` holds one entry per asset column of `rd`.
 3. Check that `keep` keeps at least one asset.
 4. Return an [`AssetSelectorResult`](@ref) holding the names of the kept assets, in their original column order.

# Arguments

  - `sel`: The asset selector.
  - `rd`: The training-window returns data.

# Validation

  - `select_assets` must return a mask whose length matches the number of asset columns.
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
    keep = select_assets(sel, rd)
    @argcheck(length(keep) == size(rd.X, 2),
              DimensionMismatch("select_assets for a $(typeof(sel)) returned a mask of length $(length(keep)) for $(size(rd.X, 2)) asset columns"))
    @argcheck(any(keep),
              IsEmptyError("a $(typeof(sel)) selects no assets from the training window; loosen its configuration"))
    return AssetSelectorResult(collect(rd.nx[keep]))
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
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator converting price-level data into returns-level data.

`PricesToReturns` is the estimator form of [`prices_to_returns`](@ref): it consumes a [`PricesResult`](@ref) and produces a [`ReturnsResult`](@ref). It is stateless — applying it to any window simply runs the conversion — so its fitted object is the estimator itself.

Missing-data filtering is deliberately *not* part of this estimator (the corresponding [`prices_to_returns`](@ref) keywords are held at their permissive defaults); use [`MissingDataFilter`](@ref) and [`Imputer`](@ref) as separate, independently tunable steps.

!!! warning

    Because this step is stateless, it does not define an asset universe. [`prices_to_returns`](@ref) drops assets that are entirely missing in the window being converted, so a training window in which an asset has no history produces a different universe from a clean test window. Precede this estimator with a [`MissingDataFilter`](@ref) (which fits the universe on the training window) and an [`Imputer`](@ref) (which fills the remaining gaps with training statistics) whenever a fitted transformation must be replayed on unseen windows; a `Pipeline` enforces this via `assert_universe_aligned`.

# Algorithm

The estimator is stateless, so both verbs are thin.

 1. [`fit_preprocessing`](@ref) returns the estimator itself. There is no state to fit.
 2. [`apply_preprocessing`](@ref) calls [`prices_to_returns`](@ref) with the five fields as keywords, and with `X`, `F`, `B`, `iv`, `ivpa`, `nz` and `Z` read off the [`PricesResult`](@ref). It returns the [`ReturnsResult`](@ref).

The missing-data keywords of [`prices_to_returns`](@ref) are not fields of this estimator, so they hold their permissive defaults and every row and column reaches the conversion.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesToReturns(;
        ret_method::Symbol = :simple,
        padding::Bool = false,
        collapse_args::Tuple = (),
        map_func::Option{<:Function} = nothing,
        join_method::Symbol = :outer,
    ) -> PricesToReturns

Keywords correspond to the struct's fields.

## Validation

  - `ret_method in (:simple, :log)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> rr = apply_preprocessing(PricesToReturns(), pr);

julia> size(rr.X)
(2, 2)

julia> rr.nx
2-element Vector{String}:
 "A"
 "B"
```

# Related

  - [`AbstractPreprocessingEstimator`](@ref)
  - [`prices_to_returns`](@ref)
  - [`PricesResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
@concrete struct PricesToReturns <: AbstractPreprocessingEstimator
    """
    Return calculation method (`:simple` or `:log`).
    """
    ret_method
    """
    Whether to pad missing values in the returns calculation.
    """
    padding
    """
    Arguments for collapsing the time series (e.g. to lower frequency).
    """
    collapse_args
    """
    Optional function applied to the data before the returns calculation.
    """
    map_func
    """
    How asset, factor, and benchmark data are joined (`:outer`, `:inner`, etc.).
    """
    join_method
    function PricesToReturns(ret_method::Symbol, padding::Bool, collapse_args::Tuple,
                             map_func::Option{<:Function}, join_method::Symbol)
        @argcheck(ret_method in (:simple, :log),
                  ArgumentError("ret_method must be :simple or :log, got :$ret_method"))
        return new{typeof(ret_method), typeof(padding), typeof(collapse_args),
                   typeof(map_func), typeof(join_method)}(ret_method, padding,
                                                          collapse_args, map_func,
                                                          join_method)
    end
end
function PricesToReturns(; ret_method::Symbol = :simple, padding::Bool = false,
                         collapse_args::Tuple = (), map_func::Option{<:Function} = nothing,
                         join_method::Symbol = :outer)::PricesToReturns
    return PricesToReturns(ret_method, padding, collapse_args, map_func, join_method)
end
function prices_to_returns(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(pr.X, pr.F; B = pr.B, iv = pr.iv, ivpa = pr.ivpa,
                             ret_method = ptr.ret_method, padding = ptr.padding,
                             collapse_args = ptr.collapse_args, map_func = ptr.map_func,
                             join_method = ptr.join_method, nz = pr.nz, Z = pr.Z)
end
function fit_preprocessing(ptr::PricesToReturns, ::PricesResult)
    return ptr
end
function apply_preprocessing(ptr::PricesToReturns, pr::PricesResult)::ReturnsResult
    return prices_to_returns(ptr, pr)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator dropping assets and observations with excessive missing data from price-level data.

The *asset universe is fitted state*: the training window decides which assets survive (per-column missing fraction at most `col_thr`), and applying the fitted result to an unseen window subsets it to that same universe — so train weights and test returns always refer to the same assets. Observation (row) filtering is window-local: rows whose missing fraction across the surviving assets exceeds `row_thr` are dropped from whichever window is being transformed.

This estimator supersedes the `missing_col_percent`/`missing_row_percent` keywords of [`prices_to_returns`](@ref), making the thresholds fitted state and independently tunable. Only the asset series `X` (and the matching implied volatility columns, and the feature matrix, whose axes are parallel to `X`) participate; factor and benchmark series pass through unchanged.

# Algorithm

## Fit

 1. Count the missing observations of each asset column with [`is_missing_value`](@ref), and divide each count by the observation total, giving `frac`.
 2. Keep the assets whose fraction does not exceed `col_thr`, and check that one asset at least survives.
 3. Return a [`MissingDataFilterResult`](@ref) holding the surviving asset names and `row_thr`.

## Apply

 1. Find the columns of the window whose names are in the fitted universe, and check that one at least is present.
 2. Count the missing assets of each row over those columns alone, and keep the rows whose count does not exceed `row_thr` of the column total.
 3. Rebuild `X` from the kept rows and the kept columns.
 4. Subselect the implied volatilities on the kept columns, and `ivpa` with them when it is a vector. The implied volatility series keeps every row, because its own clock is not the one that was filtered.
 5. Read `sq` from [`features_are_assets`](@ref), and view `nz` at the kept columns when `sq` is `true`.
 6. View the feature matrix with [`feature_matrix_view`](@ref) at the kept rows and the kept columns.
 7. Rebuild the [`PricesResult`](@ref). The factor series `F` and the benchmark series `B` pass through untouched.

The two thresholds count opposite axes: `col_thr` counts the missing rows of a column and drops columns, and `row_thr` counts the missing columns of a row and drops rows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MissingDataFilter(;
        col_thr::Number = 1.0,
        row_thr::Number = 1.0,
    ) -> MissingDataFilter

Keywords correspond to the struct's fields.

## Validation

  - `0 < col_thr <= 1`.
  - `0 < row_thr <= 1`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 NaN; 102.0 NaN; 104.0 105.0],
                     [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> res = fit_preprocessing(MissingDataFilter(; col_thr = 0.5), pr);

julia> res.nx
1-element Vector{Symbol}:
 :A
```

# Related

  - [`MissingDataFilterResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`Imputer`](@ref)
  - [`PricesResult`](@ref)
"""
@concrete struct MissingDataFilter <: AbstractPricesPreprocessingEstimator
    """
    Maximum allowed fraction `(0, 1]` of missing observations per asset column; assets above it are dropped from the universe at fit time.
    """
    col_thr
    """
    Maximum allowed fraction `(0, 1]` of missing assets per observation row; rows above it are dropped from the window being transformed.
    """
    row_thr
    function MissingDataFilter(col_thr::Number, row_thr::Number)
        @argcheck(zero(col_thr) < col_thr <= one(col_thr), DomainError)
        @argcheck(zero(row_thr) < row_thr <= one(row_thr), DomainError)
        return new{typeof(col_thr), typeof(row_thr)}(col_thr, row_thr)
    end
end
function MissingDataFilter(; col_thr::Number = 1.0,
                           row_thr::Number = 1.0)::MissingDataFilter
    return MissingDataFilter(col_thr, row_thr)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of a [`MissingDataFilter`](@ref).

Carries the asset universe selected on the training window plus the row threshold needed to transform further windows. Produced by [`fit_preprocessing`](@ref), consumed by [`apply_preprocessing`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`MissingDataFilter`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct MissingDataFilterResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets that survived the training window (the fitted universe).
    """
    nx
    """
    Maximum allowed fraction `(0, 1]` of missing assets per observation row.
    """
    row_thr
end
function fit_preprocessing(mdf::MissingDataFilter,
                           pr::PricesResult)::MissingDataFilterResult
    vals = values(pr.X)
    frac = vec(count(is_missing_value, vals; dims = 1)) / size(vals, 1)
    keep = frac .<= mdf.col_thr
    @argcheck(any(keep),
              IsEmptyError("MissingDataFilter with col_thr = $(mdf.col_thr) drops every asset in the training window"))
    return MissingDataFilterResult(TimeSeries.colnames(pr.X)[keep], mdf.row_thr)
end
function apply_preprocessing(res::MissingDataFilterResult, pr::PricesResult)::PricesResult
    cols = findall(in(res.nx), TimeSeries.colnames(pr.X))
    @argcheck(!isempty(cols),
              IsEmptyError("none of the fitted universe assets $(res.nx) are present in the data window"))
    vals = values(pr.X)[:, cols]
    rows = findall(vec(count(is_missing_value, vals; dims = 2)) .<=
                   length(cols) * res.row_thr)
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X)[rows], vals[rows, :],
                             TimeSeries.colnames(pr.X)[cols])
    iv, ivpa = if isnothing(pr.iv)
        nothing, pr.ivpa
    else
        ivv = values(pr.iv)[:, cols]
        ivm = TimeSeries.TimeArray(TimeSeries.timestamp(pr.iv), ivv,
                                   TimeSeries.colnames(pr.iv)[cols])
        ivm, isa(pr.ivpa, VecNum) ? pr.ivpa[cols] : pr.ivpa
    end
    sq = features_are_assets(pr.nz, string.(TimeSeries.colnames(pr.X)))
    nz = sq ? nothing_scalar_array_view(pr.nz, cols) : pr.nz
    Z = feature_matrix_view(pr.Z, sq, rows, cols)
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = iv, ivpa = ivpa, nz = nz, Z = Z)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window.

The *imputation parameters are fitted state*: each asset's fill value is computed from the training window's observed (non-missing) prices with the configured [`Num_VecToScaM`](@ref), and applying the fitted result to an unseen window fills that window's missing observations with the *training* values — never with statistics of the window being transformed, which is exactly the leakage a fit/apply contract exists to prevent.

Assets with no observed values in the training window get no fill value and are left untouched at apply time; combine with [`MissingDataFilter`](@ref) to drop them instead.

# Algorithm

## Fit

 1. For each asset column, collect the observed prices. An entry [`is_missing_value`](@ref) accepts is left out.
 2. Skip an asset whose column holds no observed price. It gets no fill value, and no entry in the result.
 3. Reduce the observed prices of the column to one value with `stat`, giving that asset's fill value.
 4. Return an [`ImputerResult`](@ref) holding the fitted asset names and their fill values, aligned.

## Apply

 1. Copy the price values of the window, so the input is not mutated.
 2. For each fitted asset name, find its column in the window. Skip a name the window does not carry.
 3. Replace every missing entry of that column with that asset's fitted fill value.
 4. Rebuild `X` from the filled values, keeping the timestamps and the column names, then rebuild the [`PricesResult`](@ref). Every other field passes through untouched.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Imputer(;
        stat::Num_VecToScaM = MedianValue(),
    ) -> Imputer

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100.0 1.0; NaN 3.0; 104.0 5.0],
                     [\"A\", \"B\"]);

julia> pr = PricesResult(; X = X);

julia> res = fit_preprocessing(Imputer(), pr);

julia> pv = apply_preprocessing(res, pr);

julia> values(pv.X)[2, 1]
102.0
```

# Related

  - [`ImputerResult`](@ref)
  - [`AbstractPricesPreprocessingEstimator`](@ref)
  - [`MissingDataFilter`](@ref)
  - [`Num_VecToScaM`](@ref)
"""
@concrete struct Imputer <: AbstractPricesPreprocessingEstimator
    """
    Reducer computing an asset's fill value from its observed training prices ([`Num_VecToScaM`](@ref)).
    """
    stat
    function Imputer(stat::Num_VecToScaM)
        return new{typeof(stat)}(stat)
    end
end
function Imputer(; stat::Num_VecToScaM = MedianValue())::Imputer
    return Imputer(stat)
end
"""
$(DocStringExtensions.TYPEDEF)

Fitted result of an [`Imputer`](@ref).

Carries the per-asset fill values computed on the training window. Produced by [`fit_preprocessing`](@ref), consumed by [`apply_preprocessing`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Imputer`](@ref)
  - [`AbstractPricesPreprocessingResult`](@ref)
"""
@concrete struct ImputerResult <: AbstractPricesPreprocessingResult
    """
    Names of the assets with a fitted fill value.
    """
    nx
    """
    Fill values, aligned with `nx`.
    """
    v
end
function fit_preprocessing(imp::Imputer, pr::PricesResult)::ImputerResult
    names = TimeSeries.colnames(pr.X)
    vals = values(pr.X)
    keep = Vector{Int}(undef, 0)
    v = Vector{Any}(undef, 0)
    for i in axes(vals, 2)
        obs = identity.([x for x in view(vals, :, i) if !is_missing_value(x)])
        if isempty(obs)
            continue
        end
        push!(keep, i)
        push!(v, vec_to_real_measure(imp.stat, obs))
    end
    return ImputerResult(names[keep], identity.(v))
end
function apply_preprocessing(res::ImputerResult, pr::PricesResult)::PricesResult
    names = TimeSeries.colnames(pr.X)
    vals = copy(values(pr.X))
    for (name, fill_val) in zip(res.nx, res.v)
        j = findfirst(==(name), names)
        if isnothing(j)
            continue
        end
        for i in axes(vals, 1)
            if is_missing_value(vals[i, j])
                vals[i, j] = fill_val
            end
        end
    end
    X = TimeSeries.TimeArray(TimeSeries.timestamp(pr.X), vals, names)
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa, nz = pr.nz,
                        Z = pr.Z)
end
export PricesResult, ReturnsResult, prices_to_returns, returns_result_picker,
       fit_preprocessing, apply_preprocessing, PricesToReturns, MissingDataFilter,
       MissingDataFilterResult, Imputer, ImputerResult, AssetSelectorResult,
       train_test_split, TrainTestSplit, TrainTestSplitResult
public apply_impute_method
