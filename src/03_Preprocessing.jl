"""
    abstract type AbstractReturnsResult <: AbstractResult end

Abstract supertype for all returns result types in `PortfolioOptimisers.jl`.

All concrete types representing the result of returns calculations (e.g., asset returns, factor returns) should subtype `AbstractReturnsResult`. This enables a consistent interface for downstream analysis and optimization routines.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end
"""
    _check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
                                    names_sym::Symbol, mat_sym::Symbol)

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
  - [`Option`](@ref)
  - [`VecStr`](@ref)
  - [`MatNum`](@ref)
"""
function _check_names_and_returns_matrix(names::Option{<:VecStr}, mat::Option{<:MatNum},
                                         names_sym::Symbol, mat_sym::Symbol)
    if !(isnothing(names) && isnothing(mat))
        @argcheck(!isnothing(names),
                  IsNothingError("$names_sym cannot be nothing if $mat_sym is provided. Got\n!isnothing($names_sym) => $(isnothing(names))\n!isnothing($mat_sym) => $(isnothing(mat))"))
        @argcheck(!isnothing(mat),
                  IsNothingError("$mat_sym cannot be nothing if $names_sym is provided. Got\n!isnothing($names_sym) => $(isnothing(names))\n!isnothing($mat_sym) => $(isnothing(mat))"))
        @argcheck(!isempty(names), IsEmptyError("$names_sym cannot be empty."))
        @argcheck(!isempty(mat), IsEmptyError("$mat_sym cannot be empty."))
        @argcheck(length(names) == size(mat, 2),
                  DimensionMismatch("length($names_sym) == size($mat_sym, 2) must hold. Got\nlength($names_sym) => $(length(names))\nsize($mat_sym, 2) => $(size(mat, 2))"))
    end
end
"""
    struct ReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
        nx::T1
        X::T2
        nf::T3
        F::T4
        ts::T5
        iv::T6
        ivpa::T7
    end

A flexible container type for storing the results of asset and factor returns calculations in `PortfolioOptimisers.jl`.

`ReturnsResult` is the standard result type returned by returns-processing routines, such as [`prices_to_returns`](@ref).
It supports both asset and factor returns, as well as optional time series and implied volatility information, and is designed for downstream compatibility with optimization and analysis routines.

# Fields

  - `nx`: Names or identifiers of asset columns (assets × 1).
  - `X`: Asset returns matrix (observations × assets).
  - `nf`: Names or identifiers of factor columns (factors × 1).
  - `F`: Factor returns matrix (observations × factors).
  - `ts`: Optional timestamps for each observation (observations × 1).
  - `iv`: Implied volatilities matrix (observations × assets).
  - `ivpa`: Implied volatility risk premium adjustment, if a vector (assets × 1).

# Constructor

    ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                  nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                  ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                  ivpa::Option{<:Num_VecNum} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `nx` or `X` is provided, `!isempty(nx)`, `!isempty(X)`, and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is provided, `!isempty(nf)`, `!isempty(F)`, and `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `ts` is provided, `!isempty(ts)`, and `length(ts) == size(X, 1)`.
  - If `iv` is provided, `!isempty(iv)`, `all(x -> x >= 0, iv)`, and `size(iv) == size(X)`.
  - If `ivpa` is provided, `all(x -> x >= 0, ivpa)`, `all(x -> isfinite(x), ivpa)`; if a vector, `length(ivpa) == size(iv, 2)`.

# Examples

```jldoctest
julia> ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx ┼ Vector{String}: ["A", "B"]
     X ┼ 2×2 Matrix{Float64}
    nf ┼ nothing
     F ┼ nothing
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
struct ReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
    function ReturnsResult(nx::Option{<:VecStr}, X::Option{<:MatNum}, nf::Option{<:VecStr},
                           F::Option{<:MatNum}, ts::Option{<:VecDate}, iv::Option{<:MatNum},
                           ivpa::Option{<:Num_VecNum})
        _check_names_and_returns_matrix(nx, X, :nx, :X)
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            @argcheck(size(X, 1) == size(F, 1), DimensionMismatch)
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
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_gt0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X), DimensionMismatch)
            if isa(ivpa, VecNum)
                @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts), typeof(iv),
                   typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
    end
end
function ReturnsResult(; nx::Option{<:VecStr} = nothing, X::Option{<:MatNum} = nothing,
                       nf::Option{<:VecStr} = nothing, F::Option{<:MatNum} = nothing,
                       ts::Option{<:VecDate} = nothing, iv::Option{<:MatNum} = nothing,
                       ivpa::Option{<:Num_VecNum} = nothing)
    return ReturnsResult(nx, X, nf, F, ts, iv, ivpa)
end
"""
    returns_result_view(rd::ReturnsResult, i)

Return a view of the `ReturnsResult` object for the asset or factor at index `i`.

# Arguments

  - `rd`: A `ReturnsResult` object containing asset and/or factor returns.
  - `i`: Index of the asset or factor to view.

# Returns

  - `rr::ReturnsResult`: A new `ReturnsResult` containing only the data for the specified index.

# Details

  - Extracts the asset/factor name, returns, implied volatility, and risk premium adjustment for index `i`.
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
    ts ┼ nothing
    iv ┼ nothing
  ivpa ┴ nothing

julia> PortfolioOptimisers.returns_result_view(rd, 2:2)
ReturnsResult
    nx ┼ SubArray{String, 1, Vector{String}, Tuple{UnitRange{Int64}}, true}: ["B"]
     X ┼ 2×1 SubArray{Float64, 2, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, UnitRange{Int64}}, true}
    nf ┼ nothing
     F ┼ nothing
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
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts, iv = iv,
                         ivpa = ivpa)
end
"""
    prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                      Rb::Option{<:TimeArray} = nothing, iv::Option{<:TimeArray} = nothing,
                      ivpa::Option{<:Num_VecNum} = nothing, ret_method::Symbol = :simple,
                      padding::Bool = false, missing_col_percent::Number = 1.0,
                      missing_row_percent::Option{<:Number} = 1.0, collapse_args::Tuple = (),
                      map_func::Option{<:Function} = nothing, join_method::Symbol = :outer,
                      impute_method::Option{<:Impute.Imputor} = nothing)

Convert price data (and optionally factor data) in `TimeArray` format to returns, with flexible handling of missing data, imputation, and optional implied volatility information.

ReturnsResult a [`ReturnsResult`](@ref) containing asset and factor returns, time series, and optional implied volatility data, suitable for downstream portfolio optimization.

# Arguments

  - `X`: Asset price data (timestamps × assets).
  - `F`: Optional Factor price data (timestamps × factors).
  - `Rb`: Optional Benchmark price data (timestamps × assets) or (timestamps × 1).
  - `iv`: Optional Implied volatility data.
  - `ivpa`: Optional Implied volatility risk premium adjustment.
  - `ret_method`: Return calculation method (`:simple` or `:log`).
  - `padding`: Whether to pad missing values in returns calculation.
  - `missing_col_percent`: Maximum allowed fraction `(0, 1]` of missing values per column (asset + factor).
  - `missing_row_percent`: Maximum allowed fraction `(0, 1]` of missing values per row (timestamp).
  - `collapse_args`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func`: Optional function to apply to the data before returns calculation.
  - `join_method`: How to join asset and factor data (`:outer`, `:inner`, etc.).
  - `impute_method`: Optional imputation method for missing data.

# Returns

  - [`ReturnsResult`](@ref): Struct containing asset/factor returns, names, time series, and optional implied volatility data.

# Examples

```jldoctest
julia> using TimeSeries

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
function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           Rb::Option{<:TimeArray} = nothing,
                           iv::Option{<:TimeArray} = nothing,
                           ivpa::Option{<:Num_VecNum} = nothing,
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Number = 1.0,
                           missing_row_percent::Option{<:Number} = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Option{<:Function} = nothing,
                           join_method::Symbol = :outer,
                           impute_method::Option{<:Impute.Imputor} = nothing)
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError)
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent), DomainError)
    end
    if !isempty(F)
        asset_names = string.(colnames(X))
        factor_names = string.(colnames(F))
        X = merge(X, F; method = join_method)
    else
        asset_names = string.(colnames(X))
        factor_names = String[]
    end
    if !isnothing(map_func)
        X = map(map_func, X)
    end
    if !isempty(collapse_args)
        X = collapse(X, collapse_args...)
    end
    X = DataFrame(X)

    f(x) = isa(x, Number) && isnan(x) ? missing : x

    DataFrames.transform!(X, 2:ncol(X) .=> ByRow((x) -> f(x)); renamecols = false)
    if !isnothing(impute_method)
        X = Impute.impute(X, impute_method)
    end
    missing_mtx = ismissing.(Matrix(X[!, 2:end]))
    missings_cols = vec(count(missing_mtx; dims = 2))
    keep_rows = missings_cols .<= (ncol(X) - 1) * missing_col_percent
    X = X[keep_rows, :]
    missings_rows = vec(count(missing_mtx; dims = 1))
    keep_cols = if !isnothing(missing_row_percent)
        missings_rows .<= nrow(X) * missing_row_percent
    else
        missings_rows .== StatsBase.mode(missings_rows)
    end
    X = X[!, [true; keep_cols]]
    select!(X, Not(names(X, Missing)))
    dropmissing!(X)
    X = percentchange(TimeArray(X; timestamp = :timestamp), ret_method; padding = padding)
    if !isnothing(Rb)
        @argcheck(timestamp(X) == timestamp(Rb))
        Rb_v = values(Rb)
        X_v = values(X)
        X = TimeArray(timestamp(X), isa(Rb_v, AbstractMatrix) ? X_v - Rb_v : X_v .- Rb_v;
                      colnames = colnames(X))
    end
    X = DataFrame(X)
    col_names = names(X)
    nx = intersect(col_names, asset_names)
    nf = intersect(col_names, factor_names)
    oc = setdiff(col_names, union(nx, nf))
    ts = isempty(oc) ? nothing : vec(Matrix(X[!, oc]))
    if !isnothing(ts) && !isnothing(iv)
        iv = iv[ts]
    end
    if isempty(nf)
        nf = nothing
        F = nothing
    else
        F = Matrix(X[!, nf])
    end
    if isempty(nx)
        nx = nothing
        X = nothing
    else
        X = Matrix(X[!, nx])
    end
    return ReturnsResult(; ts = ts, nx = nx, X = X, nf = nf, F = F, iv = values(iv),
                         ivpa = ivpa)
end
"""
    find_complete_indices(X::AbstractMatrix; dims::Int = 1)

Return the indices of columns (or rows) in matrix `X` that do not contain any missing or NaN values.

This function scans the specified dimension of the input matrix and returns the indices of columns (or rows) that are complete, i.e., contain no `missing` or `NaN` values.

# Arguments

  - `X`: Input matrix of numeric values (observations × assets).
  - `dims`: Dimension along which to check for completeness (`1` for columns, `2` for rows). Default is `1`.

# Returns

  - `res::Vector{Int}`: Indices of columns (or rows) in `X` that are complete.

# Validation

  - `dims in (1, 2)`.

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

  - [`find_correlated_indices`](@ref)
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
        if any(map(ismissing, X[:, i])) || any(map(isnan, X[:, i]))
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

export ReturnsResult, prices_to_returns, find_complete_indices
