"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all returns result types.

Every concrete type that holds the result of a returns calculation subtypes `AbstractReturnsResult`. [`ReturnsResult`](@ref) is the member the library builds.

[`select_assets`](@ref), [`fit_preprocessing`](@ref) and the fold generation of cross-validation dispatch on this supertype. [`ClusterGroups`](@ref) reads the Feature Matrix from the [`AssetPanel`](@ref) of the carrier, because preselection runs before any prior exists and the carrier is the only source it can read. The `Pr_RR` alias names the concrete [`ReturnsResult`](@ref), and many methods dispatch on that alias, so these readers take this supertype instead. No check enforces the interface below.

[`PredictionReturnsResult`](@ref) subtypes this supertype, but its `X` is a vector of portfolio returns with no asset axis. It does not meet the interface, and every entry point that needs an asset axis throws an error for it.

# Interfaces

To implement a new returns result that an asset selector, a cluster preselection and a cross-validation fold can read, subtype `AbstractReturnsResult` with these fields:

  - `nx`: The asset names, one per column of `X`.
  - `X`: The asset returns matrix, `observations × assets`. Fold generation reads its row count.
  - `ts`: The timestamps of the rows of `X`, or `nothing`. Fold generation reads it.
  - `pnl`: An [`AssetPanel`](@ref) over the assets of `X`, or `nothing`. [`ClusterGroups`](@ref) reads it.

Then implement the following method:

## `port_opt_view`

  - `port_opt_view(rd::MyReturnsResult, i) -> MyReturnsResult`: Return a view of `rd` on the assets at `i`.
  - `port_opt_view(rd::MyReturnsResult, i, j, k = :) -> MyReturnsResult`: Return a view of `rd` on the observations at `i`, the assets at `j` and the factors at `k`.

A subtype that carries a panel subselects it together with `X`. A subtype that implements no method gets the fallback, which throws an `ArgumentError`.

### Arguments

  - `rd`: The concrete returns result.
  - `i`: Indices of the assets to keep in the two-argument method, and of the observations to keep in the other.
  - `j`: Indices of the assets to keep.
  - `k`: Indices of the factors to keep.

### Returns

  - `rd::MyReturnsResult`: A view of the same type that holds only the selected indices.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
  - [`select_assets`](@ref)
  - [`port_opt_view`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that a list of column names and the returns matrix it names are both given or both `nothing`, and that they agree in size.

[`ReturnsResult`](@ref) calls it for the asset, factor and benchmark pairs.

# Arguments

  - `names`: The column names, or `nothing`.
  - `mat`: The returns matrix, `observations × columns`, or `nothing`.
  - `names_sym`: The name of the names argument, which the error messages print.
  - `mat_sym`: The name of the matrix argument, which the error messages print.

# Validation

  - `allunique(names)`, whenever `names` is not `nothing`. Raises an `ArgumentError`.

  - If either `names` or `mat` is not `nothing`:

      + `!isnothing(names)` and `!isnothing(mat)`. Raises an [`IsNothingError`](@ref).
      + `!isempty(names)` and `!isempty(mat)`. Raises an [`IsEmptyError`](@ref).
      + `length(names) == size(mat, 2)`. Raises a `DimensionMismatch`.

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

Returns data of a universe: the asset returns, and optionally the factor returns, the benchmark returns, the timestamps, the implied volatilities and an Asset Panel.

[`prices_to_returns`](@ref) builds it, and the priors, the optimisers and the cross-validation folds read it. All of its matrices share the observation axis, which is the rows. `X`, `iv` and a matrix `B` also share the asset axis, which is the columns.

The optional [`AssetPanel`](@ref) `pnl` holds the two universe masks of the ingestion layer, and the Panel Fields that [`FeatureDistance`](@ref) turns into a distance. The panel is data, so this type holds it and no estimator does. The clustering code does not know which subset of assets it receives. A Feature Matrix on an estimator would reach a nested clustered subproblem or a cross-validation fold with every asset of the full universe, and its rows would no longer match the assets of the subproblem. `ReturnsResult` implements [`port_opt_view`](@ref), so a view subselects the panel together with `X`.

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

  - If `nx` or `X` is not `nothing`, [`check_names_and_returns_matrix`](@ref) holds for the pair: both are given, `allunique(nx)`, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is not `nothing`, the same checks hold for `nf` and `F`.
  - If `X` and `F` are given, `size(X, 1) == size(F, 1)`.
  - If `B` is a matrix, the same checks hold for `nb` and `B`.
  - If `B` is a vector and `nb` is given, `length(nb) == 1`. A vector `B` needs no name.
  - If `B` is `nothing`, `nb` is `nothing`.
  - If `X` and a vector `B` are given, `size(X, 1) == size(B, 1)`. If `X` and a matrix `B` are given, `size(X) == size(B)`.
  - If `ts` is not `nothing`, `!isempty(ts)`, `X` or `F` is given, and `allunique(ts)`. `length(ts)` equals the row count of each of `X`, `F` and `B` that is given. `ts` must be unique because it keys the observation axis. [`feature_row_indices`](@ref) finds the rows of a subset by matching its timestamps back into this clock, and a repeated timestamp resolves to its first occurrence, which pairs an asset with the features of another period.
  - If `iv` is not `nothing`, `X` is given, `!isempty(iv)`, `size(iv) == size(X)`, and every present value is finite and non-negative. `NaN` marks an absent value. See [`assert_nonneg_where_present`](@ref).
  - `ivpa` is checked only when `iv` is given, because with no implied volatility it adjusts nothing. Then `all(x -> x > 0, ivpa)` and `all(isfinite, ivpa)`, and a vector `ivpa` has `length(ivpa) == size(iv, 2)`. The bound is strict, so a zero adjustment is refused.
  - The asset axis of `pnl` is `length(nx)`. The observation axis of a time-varying `pnl` is `size(X, 1)`. See [`check_asset_panel`](@ref).

A missing partner raises an [`IsNothingError`](@ref), an empty input an [`IsEmptyError`](@ref), a repeated name or timestamp an `ArgumentError`, a size that does not agree a `DimensionMismatch`, and a value outside its domain a `DomainError`.

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
    Names or identifiers of benchmark columns (benchmarks × 1). A vector `B` takes at most one name.
    """
    nb
    """
    Benchmark returns, (observations × 1) for one benchmark that every asset shares, or (observations × assets) for one benchmark per asset.
    """
    B
    """
    Optional timestamps for each observation (observations × 1).
    """
    ts
    """
    Implied volatilities matrix (observations × assets). `NaN` marks an absent value.
    """
    iv
    """
    $(field_dict[:ivpa_iv])
    """
    ivpa
    """
    Optional [`AssetPanel`](@ref) that holds the Panel Fields of the universe and its two universe masks.
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
        elseif !isa(B, VecNum)
            # A matrix `B` and a `nothing` `B` both take the pair check, so a name with no
            # benchmark returns throws as `nx` without `X` does.
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
            @argcheck(!isnothing(X),
                      IsNothingError("X cannot be nothing if iv is not `nothing`: the implied volatilities (iv) are one per asset return (X). Got\n!isnothing(iv) => $(!isnothing(iv))\n!isnothing(X) => $(!isnothing(X))"))
            @argcheck(!isempty(iv), IsEmptyError)
            @argcheck(size(iv) == size(X),
                      DimensionMismatch("implied volatilities (iv) must match asset returns (X) in size, got size(iv) = $(size(iv)) and size(X) = $(size(X))"))
            assert_nonneg_where_present(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
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

This is the [`port_opt_view`](@ref) method for [`ReturnsResult`](@ref). It restricts the returns data to a subset of assets.

!!! warning

    This two-argument method indexes **assets**, as every other `port_opt_view` method does. The four-argument method `port_opt_view(rd, i, j, k)` indexes **observations** first and assets second. So `i` means a different axis in each of the two methods. See [`port_opt_view(rd::ReturnsResult, i, j, k)`](@ref).

# Algorithm

 1. View the asset names `nx` at `i` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, :, i)`. Axis 2 is the assets, and every observation is kept.
 3. When `B` is a matrix, it holds one column per asset, so view `nb` at `i` and view `B` as `view(rd.B, :, i)`. When `B` is a vector or `nothing`, `nb` and `B` pass through unchanged.
 4. View the implied volatilities as `view(rd.iv, :, i)`, and the adjustment `ivpa` at `i`.
 5. View the [`AssetPanel`](@ref) `pnl` with [`panel_carrier_view`](@ref) at `i` on the asset axis, and give it the asset names `rd.nx`. The observation index is a `Colon`, so a time-varying panel keeps every observation. The view slices the values of every Panel Field and both universe masks on the asset axis. It also slices the label axis of a tensor Panel Field whose labels are the asset names, see [`features_are_assets`](@ref). The label axis of every other field holds features, and an asset view does not change it.
 6. Rebuild the [`ReturnsResult`](@ref). The factor names `nf`, the factor returns `F` and the timestamps `ts` pass through unchanged, because none of the three has an asset axis.

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

    The first index of this method selects **observations**, not assets, and the second index selects assets. Every other [`port_opt_view`](@ref) method, [`port_opt_view(rd::ReturnsResult, i)`](@ref) too, takes the assets first. Cross-validation splits observations and assets together, and this method exists for it.

# Algorithm

 1. View the asset names `nx` at `j` with [`nothing_scalar_array_view`](@ref).
 2. View the asset returns as `view(rd.X, i, j)`. Axis 1 is the observations, and axis 2 is the assets.
 3. View the factor names `nf` at `k`. When `k` is a `Colon`, `nf` passes through. View the factor returns as `view(rd.F, i, k)`.
 4. When `B` is a matrix, it holds one column per asset, so view `nb` at `j` and view `B` as `view(rd.B, i, j)`. When `B` is a vector, every asset shares it, so view it as `view(rd.B, i)` and pass `nb` through.
 5. View the timestamps `ts` at `i`, the implied volatilities as `view(rd.iv, i, j)`, and the adjustment `ivpa` at `j`.
 6. View the [`AssetPanel`](@ref) `pnl` with [`panel_carrier_view`](@ref) at the observations `i` and the assets `j`, and give it the asset names `rd.nx`. The view slices both axes of every Panel Field and of both universe masks. It also slices the label axis of a tensor Panel Field whose labels are the asset names, see [`features_are_assets`](@ref). A static panel has no observation axis and ignores `i`, as a scalar `ivpa` ignores `j`.
 7. Rebuild the [`ReturnsResult`](@ref).

Each field that is `nothing` stays `nothing`. No step copies data.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified indices.

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

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

* * *

    port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)

Fallback that throws for an [`AbstractReturnsResult`](@ref) subtype that does not implement [`port_opt_view`](@ref).

Without it, the call reaches the generic fallback `port_opt_view(x, i, args...)`, which throws a `MethodError` for [`nothing_scalar_array_view`](@ref), a function that the author of the subtype never called. This method names the missing method instead. Returns data always has an asset axis, so no subtype can pass through a view unchanged.

A subtype that carries an [`AssetPanel`](@ref) subselects it as it subselects `X`. It subselects the asset axis in every method, and the observation axis in the methods that take one. When the labels of a tensor Panel Field are the assets, see [`features_are_assets`](@ref), it subselects the label axis of that field too. A panel that a fold does not subselect gives a finite distance over the wrong universe, and no check catches it. The [`ReturnsResult`](@ref) methods are the reference implementation.

# Algorithm

 1. Throw an `ArgumentError` that names the concrete type and the number of index arguments of the call. The method reads neither the indices nor the fields of `rd`.

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractReturnsResult`](@ref)

* * *

    port_opt_view(rd::ReturnsResult, args...; kwargs...)

Fallback that throws for a [`ReturnsResult`](@ref) call whose shape matches no supported method.

`ReturnsResult` implements [`port_opt_view`](@ref), so the [`AbstractReturnsResult`](@ref) fallback above would report a wrong call as a missing method. This method takes the call and states its shape. The supported methods take one, two or three positional index arguments and no keyword argument. `factors` is the third positional index, not a keyword.

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
    return throw(ArgumentError("$(typeof(rd)) subtypes AbstractReturnsResult but does not implement port_opt_view for $(length(args)) index argument(s). Extension authors: implement port_opt_view for the subtype, so that a meta-optimiser or a cross-validation fold can subselect its assets and observations. See port_opt_view(rd::ReturnsResult, ...) for the method to mirror."))
end
"""
    const Prices_RR = Union{<:AbstractReturnsResult, <:AbstractPricesResult}

Union of the two data levels cross-validation folds can be computed on: returns-level ([`AbstractReturnsResult`](@ref)) and price-level ([`AbstractPricesResult`](@ref)) data.

Fold generation needs only an observation count, from [`cv_nobs`](@ref), and a timestamp vector, from [`cv_timestamps`](@ref). So [`Base.split`](@ref) and [`n_splits`](@ref) accept either level. A split at the price level lets a `Pipeline` be cross-validated on its input rows, so its stateful preprocessing stays inside the fold.

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

The function reads the benchmark field `B` of `rd` and the flag `brt`, which asks for benchmark tracking. When `brt` is `true` and `rd` carries a benchmark, it returns a new `ReturnsResult` whose asset returns are the excess returns over the benchmark. Otherwise it returns `rd`.

# Algorithm

The method that Julia selects on the type of the field `B` is the first step, so a carrier with no benchmark runs no branch.

 1. `rd` carries no benchmark, because its `B` field is `Nothing`. Return `rd`.
 2. `brt` is `false`. Return `rd`.
 3. `brt` is `true`. Subtract the benchmark from the asset returns, which gives `X`. A vector benchmark subtracts by broadcast, `rd.X .- rd.B`, which takes the benchmark value of each observation from every asset column. A matrix benchmark subtracts elementwise, `rd.X - rd.B`.
 4. Rebuild the [`ReturnsResult`](@ref) from `X`, with `nb` and `B` set to `nothing`. The subtraction uses up the benchmark, so a second call returns its argument unchanged. The fields `nx`, `nf`, `F`, `ts`, `iv`, `ivpa` and `pnl` pass through. The function does not modify `rd`.

# Arguments

  - `rd`: A `ReturnsResult` that holds the asset returns, and optionally the factor and benchmark returns. When it carries a benchmark, it also holds `X`.
  - `brt`: `true` to subtract the benchmark `B` from the asset returns, `false` to keep them. A carrier with no benchmark accepts any value.

# Validation

  - When `rd` carries a benchmark, `rd.X` is a matrix and `brt` is a `Bool`. Otherwise no method matches, and the call raises a `MethodError`.

# Returns

  - `rd::ReturnsResult`:

      + If `brt` is `true` and `rd` carries a benchmark, a new `ReturnsResult` that holds the excess returns in `X`. Its `nb` and `B` are `nothing`, so a second call returns it unchanged.
      + Otherwise, `rd` itself.

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

`nothing` in the `ape` slot tells the kernel to read the panel that the data carrier holds. The carriers reach the kernel as the two keywords `pr` and `rd`, and this function selects the source by dispatch. A [`ReturnsResult`](@ref) in either slot gives its `pnl`. When both slots hold one, `rd` wins, because `rd` is the data carrier. `Pr_RR` admits a [`ReturnsResult`](@ref) in the `pr` slot, and `clusterise(cle, rd)` and every [`Pipeline`](@ref) step pass it there. So the second method serves the shortest public call.

A prior result alone carries no panel, so the call raises an [`IsNothingError`](@ref) that names the two ways to supply one.

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

The `pnl` of a carrier is optional, so a carrier built without one gives `nothing`. This function turns that `nothing` into an error message. It returns the panel, so the caller makes one call for the check and the access.

# Algorithm

The method that Julia selects is the algorithm. A panel returns itself, and `nothing` raises.

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
public AbstractReturnsResult
