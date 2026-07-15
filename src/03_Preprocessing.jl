"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all returns result types in `PortfolioOptimisers.jl`.

All concrete and/or types representing the result of returns calculations should be subtypes of `AbstractReturnsResult`.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all price-level data result types in `PortfolioOptimisers.jl`.

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

# Returns

  - `nothing`.

# Details

  - If either `names` or `mat` is not `nothing`:

      + `!isnothing(names)` and `!isnothing(mat)`.
      + `!isempty(names)` and `!isempty(mat)`.
      + `length(names) == size(mat, 2)`.

# Related

  - [`ReturnsResult`](@ref)
"""
function check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
                                        names_sym::Symbol, mat_sym::Symbol)
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
$(DocStringExtensions.TYPEDEF)

A container for aligned, time-indexed price-level data in `PortfolioOptimisers.jl`.

`PricesResult` is the prices-level mirror of [`ReturnsResult`](@ref): it bundles asset prices with optional factor, benchmark, and implied volatility series, all as `TimeSeries.TimeArray`s. It is the input to price-level preprocessing estimators and prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline cross-validation via [`port_opt_view`](@ref).

The asset price series `X` is the master clock: [`port_opt_view`](@ref) selects observation windows on `X` and aligns the other series to the selected timestamps.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PricesResult(;
        X::TimeSeries.TimeArray,
        F::Option{<:TimeSeries.TimeArray} = nothing,
        B::Option{<:TimeSeries.TimeArray} = nothing,
        iv::Option{<:TimeSeries.TimeArray} = nothing,
        ivpa::Option{<:Num_VecNum} = nothing,
    ) -> PricesResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(X)`.
  - If `F` is not `nothing`: `!isempty(F)`.
  - If `B` is not `nothing`: `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`: `!isempty(iv)`, `all(x -> x >= 0, values(iv))`, `all(x -> isfinite(x), values(iv))`, and `size(values(iv), 2) == size(values(X), 2)`.
  - If `ivpa` is not `nothing`: `all(x -> x > 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(values(X), 2)`.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

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
    Implied volatility risk premium adjustment, if a vector (assets × 1).
    """
    ivpa
    function PricesResult(X::TimeSeries.TimeArray, F::Option{<:TimeSeries.TimeArray},
                          B::Option{<:TimeSeries.TimeArray},
                          iv::Option{<:TimeSeries.TimeArray}, ivpa::Option{<:Num_VecNum})
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
        return new{typeof(X), typeof(F), typeof(B), typeof(iv), typeof(ivpa)}(X, F, B, iv,
                                                                              ivpa)
    end
end
function PricesResult(; X::TimeSeries.TimeArray,
                      F::Option{<:TimeSeries.TimeArray} = nothing,
                      B::Option{<:TimeSeries.TimeArray} = nothing,
                      iv::Option{<:TimeSeries.TimeArray} = nothing,
                      ivpa::Option{<:Num_VecNum} = nothing)::PricesResult
    return PricesResult(X, F, B, iv, ivpa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `PricesResult` for the observation window `i` of the asset price series `X`.

The asset price series is the master clock: `i` selects rows of `X`, and the factor, benchmark, and implied volatility series are aligned to the selected timestamps (rows whose timestamps are absent from a series are dropped from that series).

# Arguments

  - `pr`: A `PricesResult` object.
  - `i`: Observation window into the rows of `pr.X`. Either integer indices (`AbstractVector{<:Integer}`, `AbstractRange`, or `Colon`) or a vector of timestamps (`AbstractVector{<:Dates.AbstractTime}`).

# Returns

  - `new_pr::PricesResult`: A new `PricesResult` containing only the data for the selected window.

# Details

  - `Colon` returns `pr` unchanged.
  - Integer windows index the rows of `pr.X` directly; the selected timestamps are then used to align `F`, `B`, and `iv`.
  - Timestamp windows are applied to all series directly.
  - `ivpa` is per-asset and passes through unchanged.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3),
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

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
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime}, ::Colon)
    X = pr.X[i]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    B = if isnothing(pr.B)
        nothing
    elseif length(TimeSeries.colnames(pr.B)) >= 1
        pr.B[i]
    else
        pr.B[i]
    end
    iv = isnothing(pr.iv) ? nothing : pr.iv[i]
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = pr.ivpa)
end
function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},
                       j::AbstractVector)
    X = pr.X[i][j]
    F = isnothing(pr.F) ? nothing : pr.F[i]
    B = if isnothing(pr.B)
        nothing
    elseif length(j) >= 1
        pr.B[i][j]
    else
        pr.B[i]
    end
    iv = isnothing(pr.iv) ? nothing : pr.iv[i][j]
    ivpa = nothing_scalar_array_view(pr.ivpa, j)
    return PricesResult(; X = X, F = F, B = B, iv = iv, ivpa = ivpa)
end
function port_opt_view(pr::PricesResult,
                       i::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :,
                       j::Union{<:VecInt, <:AbstractRange{<:Integer}, Colon} = :)
    return port_opt_view(pr, TimeSeries.timestamp(pr.X)[i], TimeSeries.colnames(pr.X)[j])
end
"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for storing the results of asset and factor returns calculations in `PortfolioOptimisers.jl`.

`ReturnsResult` is the standard result type returned by returns-processing routines, such as [`prices_to_returns`](@ref).

It supports both asset and factor returns, as well as optional time series and implied volatility information, and is designed for downstream compatibility with optimisation and analysis routines.

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
    ) -> ReturnsResult

Keywords correspond to the struct's fields.

## Validation

  - If `nx` or `X` is not `nothing`, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is not `nothing`, `!isempty(nf)`, `!isempty(F)`, `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `nb` or `B` is not `nothing` and `B` is a matrix: `!isempty(nb)`, `!isempty(B)`, and `length(nb) == size(B, 2)`.
  - If `nb` or `B` is not `nothing` and `B` is a vector: `length(nb) == 1`.
  - If `X` and `B` are not `nothing`: if `B` is a vector, `size(X, 1) == size(B, 1)`; if `B` is a matrix, `size(X) == size(B)`.
  - If `ts` is not `nothing`, `!isempty(ts)`, and `length(ts) == size(X, 1)`.
  - If `ts` and `B` are not `nothing`: `length(ts) == size(B, 1)`.
  - If `iv` is not `nothing`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `size(iv) == size(X)`.
  - If `ivpa` is not `nothing`, `all(x -> x >= 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(iv, 2)`.

# Examples

```jldoctest
julia> ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┴ nothing
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
    Implied volatility risk premium adjustment, if a vector (assets × 1).
    """
    ivpa
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, nb::Option{<:VecStr},
                           B::Option{<:VecNum_MatNum}, ts::Option{<:VecDate},
                           iv::Option{<:MatNum}, ivpa::Option{<:Num_VecNum})
        if !isnothing(nx)
            @argcheck(allunique(nx),
                      ArgumentError("Asset names must be unique. Got\nallunique(nx) => $(allunique(nx))"))
        end
        if !isnothing(nf)
            @argcheck(allunique(nf),
                      ArgumentError("Factor names must be unique. Got\nallunique(nf) => $(allunique(nf))"))
        end
        check_names_and_returns_matrix(nx, X, :nx, :X)
        check_names_and_returns_matrix(nf, F, :nf, :F)
        if isa(B, VecNum) && !isnothing(nb)
            @argcheck(length(nb) == 1, DimensionMismatch)
        elseif isa(B, MatNum)
            check_names_and_returns_matrix(nb, B, :nb, :B)
        end
        if !isnothing(X) && !isnothing(F)
            @argcheck(size(X, 1) == size(F, 1), DimensionMismatch)
        end
        if !isnothing(X) && !isnothing(B)
            if isa(B, VecNum)
                @argcheck(size(X, 1) == size(B, 1), DimensionMismatch)
            else
                @argcheck(size(X) == size(B), DimensionMismatch)
            end
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)), IsNothingError)
            if !isnothing(X)
                @argcheck(length(ts) == size(X, 1), DimensionMismatch)
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
            if !isnothing(B)
                @argcheck(length(ts) == size(B, 1), DimensionMismatch)
            end
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X), DimensionMismatch)
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(nb), typeof(B),
                   typeof(ts), typeof(iv), typeof(ivpa)}(nx, X, nf, F, nb, B, ts, iv, ivpa)
    end
end
function ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                       nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                       nb::Option{<:VecStr} = nothing, B::Option{<:VecNum_MatNum} = nothing,
                       ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                       ivpa::Option{<:Num_VecNum} = nothing)::ReturnsResult
    return ReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `ReturnsResult` object for the assets at indices `i`.

This is the [`port_opt_view`](@ref) method for [`ReturnsResult`](@ref) — the View of the library's central data structure, restricting it to a subset of assets.

!!! warning

    This two-argument method indexes **assets**, matching the rest of the `port_opt_view` family. The four-argument method `port_opt_view(rd, i, j, k)` indexes **observations** first and assets second. The two arities therefore give `i` different meanings; see [`port_opt_view(rd::ReturnsResult, i, j, k)`](@ref).

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Indices of the assets to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified index.

# Details

  - Extracts the asset name, returns, implied volatility, and risk premium adjustment for indices `i`.
  - Preserves factor, timestamp, and other fields from the original object.
  - Returns `nothing` for fields that are not present.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
    nb ┼ nothing
     B ┼ nothing
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┴ nothing

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
  ivpa ┴ nothing
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

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index or indices of the observation(s) to view.
  - `j`: Index or indices of the assets to view.
  - `k`: Index or indices of the factors to view.

# Returns

  - `new_rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified indices.

# Details

  - Extracts the asset name, returns, implied volatility, and risk premium adjustment for indices `j` and observation(s) `i`.
  - Extracts the factor names and returns for indices `k` and observations `i`.
  - Preserves factor names and returns for the selected observations.
  - Preserves timestamps for the selected observations.
  - Returns `nothing` for fields that are not present in the original object.

# Related

  - [`ReturnsResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4; 0.5 0.6], nf = ["F1"],
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
  ivpa ┴ nothing

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
  ivpa ┴ nothing
```

* * *

    port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)

Erroring tripwire for [`AbstractReturnsResult`](@ref) subtypes that do not implement [`port_opt_view`](@ref).

Without it, the universal leaf fallback `port_opt_view(x, i, args...)` would hand back the returns result *unsubselected*, and a meta-optimiser or cross-validation fold would silently train on the full universe. Returns data is never a leaf value, so an unhandled subtype is a missing method, not a pass-through.

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractReturnsResult`](@ref)
"""
function port_opt_view(rd::ReturnsResult, i)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, i)
    B = !isa(rd.B, MatNum) ? rd.B : view(rd.B, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, nb = nb, B = B, ts = rd.ts,
                         iv = iv, ivpa = ivpa)
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
    return ReturnsResult(; nx = nx, X = X, nf = nf, F = F, nb = nb, B = B, ts = ts, iv = iv,
                         ivpa = ivpa)
end
function port_opt_view(rd::AbstractReturnsResult, args...; kwargs...)
    return throw(ArgumentError("port_opt_view is not implemented for $(typeof(rd)) with $(length(args)) index argument(s); implement it for the subtype. ReturnsResult supports port_opt_view(rd, assets) and port_opt_view(rd, observations, assets, factors = :)."))
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
  - **Both given**: the head supplies `lo` training rows, the tail supplies `hi` test rows, and any rows between them are **embargoed** — they belong to neither window. This is how a gap between train and test is expressed.

## Validation

  - Both windows are non-empty. A split whose sizes saturate the data on one side (`train_size = N`) leaves nothing to test on and throws.
  - The windows do not overlap: `lo + hi <= N`.

# Related

  - [`train_test_split`](@ref)
  - [`TrainTestSplit`](@ref)
  - [`split_count`](@ref)
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

# Arguments

  - `rd`/`pr`: The data to split.
  - `train_size`: Training rows as a count (`Integer`) or a fraction (`AbstractFloat` in `(0, 1)`); `nothing` takes the complement of `test_size`.
  - `test_size`: Test rows, likewise; `nothing` takes the complement of `train_size`. With neither given the split is 75/25.

# Returns

  - `(train, test)`: The two windows, of the same type as the input.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A"], X = reshape(collect(0.1:0.1:1.0), 10, 1));

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

# Arguments

  - `rd`: A `ReturnsResult` object containing asset, factor and/or benchmark returns.
  - `brt`: Boolean flag indicating whether benchmark-tracking behaviour should be applied. When `true`, asset returns are adjusted by subtracting the benchmark `B` (if present).

# Returns

  - `rd::ReturnsResult`:

      + If `brt` is `true` and a benchmark `B` is present: A new `ReturnsResult` with adjusted asset returns
      + Otherwise: The `rd` is returned unchanged.

# Details

  - When an adjustment is required and `B` is present, a new `ReturnsResult` is returned leaving the original `rd` unmodified.
  - If no adjustment is required or `B` is `nothing`, the original `ReturnsResult` is returned unchanged.
  - For vector benchmarks (`VecNum`) subtraction uses broadcasting (`X .- B`) to subtract the per-observation benchmark from each asset column (index tracking).
  - For matrix benchmarks (`MatNum`) subtraction uses matrix subtraction (`X - B`).
  - Other fields (`nx`, `nf`, `F`, `ts`, `iv`, `ivpa`) are preserved in the returned object.

# Examples

```jldoctest
julia> rd = ReturnsResult(; nx = ["A", "B"], X = [0.10 0.20; 0.30 0.40], nb = ["BM"],
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
  ivpa ┴ nothing

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
  ivpa ┴ nothing

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
  ivpa ┴ nothing

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
                      ivpa = rd.ivpa)
    end
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
        impute_method::Option{<:Impute.Imputor} = nothing
    ) -> ReturnsResult

Convert price data (and optionally factor data) in `TimeSeries.TimeArray` format to returns, with flexible handling of missing data, imputation, and optional implied volatility information.

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

If a benchmark ``B_{t,i}`` is provided, excess returns are used: ``\\tilde{r}_{t,i} = r_{t,i} - b_{t,i}``.

# Arguments

  - `X`: Asset price data (observations × assets).
  - `F`: Optional Factor price data (observations × factors).
  - `B`: Optional Benchmark price data (observations × assets) or (observations × 1).
  - `iv`: Optional Implied volatility data.
  - `ivpa`: Optional Implied volatility risk premium adjustment.
  - `ret_method`: Return calculation method (`:simple` or `:log`).
  - `padding`: Whether to pad missing values in returns calculation.
  - `missing_col_percent`: Maximum allowed fraction `(0, 1]` of missing values per column (asset + factor).
  - `missing_row_percent`: Maximum allowed fraction `(0, 1]` of missing values per row (timestamp).
  - `collapse_args`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func`: Optional function to apply to the data before returns calculation.
  - `join_method`: How to join asset, factor data and benchmark data (`:outer`, `:inner`, etc.).
  - `impute_method`: Optional imputation method for missing data.

# Returns

  - `rr::ReturnsResult`: Struct containing asset/factor returns, names, time series, and optional implied volatility data.

# Validation

  - `!isempty(X)`.
  - `0 < missing_col_percent <= 1`
  - `0 < missing_row_percent <= 1`.
  - If `F` is not `nothing`, `!isempty(F)`.
  - If `B` is not `nothing`, `!isempty(B)`, and `size(values(B), 2) in (1, size(values(X), 2))`.
  - If `iv` is not `nothing`, the timestamp of the merged data matrix must be a subset of `TimeSeries.timestamp(iv)`, then `iv = values(iv)`, `!isempty(iv)`, `all(x -> x >= 0, iv)`, `size(iv) == size(X)`.
  - If `ivpa` is not `nothing`, `all(x -> x >= 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(iv, 2)`.

# Details

  - Joins asset, factor, and benchmark data as specified.

  - Optionally applies a mapping function and/or collapses the time series.

  - Handles missing values by filtering, imputation, and dropping as configured.

  - Computes returns using the specified method.

      + If `B` is not `nothing`, it is subtracted from asset returns. Used for returns tracking error optimisations.

  - Returns a `ReturnsResult` with asset/factor names, returns, timestamps, and optional implied volatility data.

# Examples

```jldoctest
julia> X = TimeArray(Date(2020, 1, 1):Day(1):Date(2020, 1, 3), [100 101; 102 103; 104 105],
                     ["A", "B"])
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
  ivpa ┴ nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
  - [`VecDate`](@ref)
  - [`Num_VecNum`](@ref)
  - [`TimeSeries`](https://juliastats.org/TimeSeries.jl/stable/timearray/#The-TimeArray-time-series-type)
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
                           join_method::Symbol = :outer,
                           impute_method::Option{<:Impute.Imputor} = nothing)
    @argcheck(!isempty(X), IsEmptyError)
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError)
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent), DomainError)
    end
    asset_names = string.(TimeSeries.colnames(X))
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
    if !isnothing(impute_method)
        X = Impute.impute(X, impute_method)
    end
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
                         ivpa = ivpa)
end
"""
    find_complete_indices(X::AbstractMatrix; dims::Int = 1) -> VecInt

Return the indices of columns (or rows) in matrix `X` that do not contain any missing or NaN values.

This function scans the specified dimension of the input matrix and returns the indices of columns (or rows) that are complete, i.e., contain no `missing` or `NaN` values.

Internal machinery — the caller-facing form is [`CompleteAssetSelector`](@ref), which wraps the `dims = 1` (complete-column) mode as a fit/apply estimator. The `dims = 2` (complete-row) mode has no estimator form: dropping observations is a price-level concern ([`MissingDataFilter`](@ref)).

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:dims])

# Validation

  - `dims in (1, 2)`.

# Returns

  - `res::VecInt`: Indices of columns (or rows) in `X` that are complete.

# Details

  - If `dims == 2`, the matrix is transposed and columns are checked.
  - Any column (or row) containing at least one `missing` or `NaN` value is excluded.
  - The result is a vector of indices of complete columns (or rows).

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
    @argcheck(dims in (1, 2), DomainError)
    if dims == 2
        X = transpose(X)
    end
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

Abstract supertype for all preprocessing estimator types in `PortfolioOptimisers.jl`.

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

Abstract supertype for all preprocessing result types in `PortfolioOptimisers.jl`.

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
    return throw(ArgumentError("fit_preprocessing is not implemented for $(typeof(est)); implement fit_preprocessing(est, data) and apply_preprocessing(fitted, data)"))
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
    return throw(ArgumentError("apply_preprocessing is not implemented for $(typeof(fitted))"))
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
    return throw(ArgumentError("select_assets is not implemented for $(typeof(sel)); every AbstractAssetSelector must define select_assets(sel, rd) returning a keep-mask over the asset columns of rd"))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Fit any [`AbstractAssetSelector`](@ref) by recording the asset universe [`select_assets`](@ref) keeps.

## Validation

  - `select_assets` must return a mask whose length matches the number of asset columns.
  - The selection must keep at least one asset; a selector that empties the universe throws rather than passing a zero-asset problem downstream (the [`MissingDataFilter`](@ref) precedent).

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

## Validation

  - Every fitted asset name must be present in the window; a missing one throws rather than silently shrinking the universe.

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
                     [100.0 101.0; 102.0 103.0; 104.0 105.0], ["A", "B"]);

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
                             join_method = ptr.join_method)
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

This estimator supersedes the `missing_col_percent`/`missing_row_percent` keywords of [`prices_to_returns`](@ref), making the thresholds fitted state and independently tunable. Only the asset series `X` (and the matching implied volatility columns) participate; factor and benchmark series pass through unchanged.

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
                     ["A", "B"]);

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
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = iv, ivpa = ivpa)
end
"""
$(DocStringExtensions.TYPEDEF)

Preprocessing estimator imputing missing price observations from per-asset statistics fitted on the training window.

The *imputation parameters are fitted state*: each asset's fill value is computed from the training window's observed (non-missing) prices with the configured [`Num_VecToScaM`](@ref), and applying the fitted result to an unseen window fills that window's missing observations with the *training* values — never with statistics of the window being transformed, which is exactly the leakage a fit/apply contract exists to prevent.

Assets with no observed values in the training window get no fill value and are left untouched at apply time; combine with [`MissingDataFilter`](@ref) to drop them instead.

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
                     ["A", "B"]);

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
    return PricesResult(; X = X, F = pr.F, B = pr.B, iv = pr.iv, ivpa = pr.ivpa)
end
export PricesResult, ReturnsResult, prices_to_returns, returns_result_picker,
       fit_preprocessing, apply_preprocessing, PricesToReturns, MissingDataFilter,
       MissingDataFilterResult, Imputer, ImputerResult, AssetSelectorResult,
       train_test_split, TrainTestSplit, TrainTestSplitResult
