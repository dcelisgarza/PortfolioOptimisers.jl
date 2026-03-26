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
function _check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
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

A flexible container type for storing the results of asset and factor returns calculations in `PortfolioOptimisers.jl`.

`ReturnsResult` is the standard result type returned by returns-processing routines, such as [`prices_to_returns`](@ref).

It supports both asset and factor returns, as well as optional time series and implied volatility information, and is designed for downstream compatibility with optimisation and analysis routines.

# Fields

$(DocStringExtensions.FIELDS)

# Constructor

    ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                  nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                  ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                  ivpa::Option{<:Num_VecNum} = nothing)

Keywords correspond to the struct's fields.

## Validation

  - If `nx` or `X` is not `nothing`, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is not `nothing`, `!isempty(nf)`, `!isempty(F)`, and `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `ts` is not `nothing`, `!isempty(ts)`, and `length(ts) == size(X, 1)`.
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
    "Names or identifiers of asset columns (assets × 1)."
    nx
    "Asset returns matrix (observations × assets)."
    X
    "Names or identifiers of factor columns (factors × 1)."
    nf
    "Factor returns matrix (observations × factors)."
    F
    "Names or identifiers of benchmark columns (observations × 1) or (observations × assets)."
    nb
    "Benchmark prices (observations × 1) or (observations × assets)."
    B
    "Optional timestamps for each observation (observations × 1)."
    ts
    "Implied volatilities matrix (observations × assets)."
    iv
    "Implied volatility risk premium adjustment, if a vector (assets × 1)."
    ivpa
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, nb::Option{<:VecStr},
                           B::Option{<:VecNum_MatNum}, ts::Option{<:VecDate},
                           iv::Option{<:MatNum}, ivpa::Option{<:Num_VecNum})
        _check_names_and_returns_matrix(nx, X, :nx, :X)
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if isa(B, VecNum) && !isnothing(nb)
            @argcheck(length(nb) == 1, DimensionMismatch)
        elseif isa(B, MatNum)
            _check_names_and_returns_matrix(nb, B, :nb, :B)
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
                       ivpa::Option{<:Num_VecNum} = nothing)
    return ReturnsResult(nx, X, nf, F, nb, B, ts, iv, ivpa)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of the `ReturnsResult` object for the assets at indices `i`.

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

julia> PortfolioOptimisers.returns_result_view(rd, 2:2)
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
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
"""
function returns_result_view(rd::ReturnsResult, i)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    nb = !isa(rd.B, MatNum) ? rd.nb : nothing_scalar_array_view(rd.nb, i)
    B = !isa(rd.B, MatNum) ? rd.B : view(rd.B, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, nb = nb, B = B, ts = rd.ts,
                         iv = iv, ivpa = ivpa)
end
"""
    returns_result_view(
                        rd::ReturnsResult,
                        i,
                        j,
                        k = :
    ) -> ReturnsResult

Return a view of the `ReturnsResult` object for assets at indices `j`, observations at indices `i`, and factors at indices `k`.

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
  - [`returns_result_view`](@ref)
  - [`prices_to_returns`](@ref)
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
"""
function returns_result_view(rd::ReturnsResult, i, j, k = :)
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
  - [`returns_result_view`](@ref)
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
    prices_to_returns(X::TimeSeries.TimeArray,
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

    f(x) = isa(x, Number) && isnan(x) ? missing : x

    DataFrames.transform!(X, 2:DataFrames.DataAPI.ncol(X) .=> DataFrames.ByRow((x) -> f(x));
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
        @argcheck(issubset(ts, TimeSeries.timestamp(iv)))
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

julia> find_complete_indices(X)
1-element Vector{Int64}:
 1

julia> find_complete_indices(X; dims = 2)
Int64[]
```

# Related

  - [`find_uncorrelated_indices`]-(@ref)
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
!!! note

    Not implemented yet, still unexported.
"""
function select_k_extremes(X::MatNum) end

export ReturnsResult, prices_to_returns, find_complete_indices, returns_result_picker
