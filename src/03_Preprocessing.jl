function drop_correlated(X::AbstractMatrix; threshold::Real = 0.95, absolute::Bool = false)
    N = size(X, 2)
    rho = !absolute ? cor(X) : abs.(cor(X))
    mean_rho = mean(rho; dims = 1)
    tril_idx = findall(tril!(trues(size(rho)), -1))
    candidate_idx = findall(rho[tril_idx] .>= threshold)
    candidate_idx = candidate_idx[sortperm(rho[tril_idx][candidate_idx]; rev = true)]
    to_remove = sizehint!(Set{Int}(), div(length(candidate_idx), 2))
    for idx in candidate_idx
        i, j = tril_idx[idx][1], tril_idx[idx][2]
        if i ∉ to_remove && j ∉ to_remove
            if mean_rho[i] > mean_rho[j]
                push!(to_remove, i)
            else
                push!(to_remove, j)
            end
        end
    end
    return setdiff(1:N, to_remove)
end
function drop_incomplete(X::AbstractMatrix, any_missing::Bool = true)
    N = size(X, 2)
    return if any_missing
        to_remove = Vector{Int}(undef, 0)
        for i in axes(X, 2)
            if any(isnan, view(X, :, i)) || any(ismissing, view(X, :, i))
                push!(to_remove, i)
            end
        end
        setdiff(1:N, to_remove)
    else
        (1:N)[(.!isnan.(X[1, :]) .|| ismissing.(X[1, :])) .& (.!isnan.(X[end, :]) .|| ismissing.(X[end,
                                                                                                   :]))]
    end
end
function select_kextremes(X::AbstractMatrix) end
function _check_names_and_returns_matrix(names, mat, names_sym, mat_sym)
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

A flexible container type for storing the results of asset and factor returns calculations in PortfolioOptimisers.jl.

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

    ReturnsResult(; nx::Union{Nothing, <:AbstractVector{<:AbstractString}} = nothing,
                  X::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                  nf::Union{Nothing, <:AbstractVector{<:AbstractString}} = nothing,
                  F::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                  ts::Union{Nothing, <:AbstractVector{<:Dates.AbstractTime}} = nothing,
                  iv::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                  ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)

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
"""
struct ReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
    function ReturnsResult(nx::Union{Nothing, <:AbstractVector{<:AbstractString}},
                           X::Union{Nothing, <:AbstractMatrix{<:Real}},
                           nf::Union{Nothing, <:AbstractVector{<:AbstractString}},
                           F::Union{Nothing, <:AbstractMatrix{<:Real}},
                           ts::Union{Nothing, <:AbstractVector{<:Dates.AbstractTime}},
                           iv::Union{Nothing, <:AbstractMatrix{<:Real}},
                           ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}})
        _check_names_and_returns_matrix(nx, X, :nx, :X)
        _check_names_and_returns_matrix(nf, F, :nf, :F)
        if !isnothing(X) && !isnothing(F)
            @argcheck(size(X, 1) == size(F, 1), DimensionMismatch)
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError)
            @argcheck(!(isnothing(X) && isnothing(F)),
                      IsNothingError("at least one of `X` or `F` must be provided. Got\n!isnothing(X) => $(isnothing(X))\n!isnothing(F) => $(isnothing(F))"))
            if !isnothing(X)
                @argcheck(length(ts) == size(X, 1), DimensionMismatch)
            end
            if !isnothing(F)
                @argcheck(length(ts) == size(F, 1), DimensionMismatch)
            end
        end
        if !isnothing(iv)
            assert_nonempty_nonneg_finite_val(iv, :iv)
            assert_nonempty_geq0_finite_val(ivpa, :ivpa)
            @argcheck(size(iv) == size(X), DimensionMismatch)
            if isa(ivpa, AbstractVector)
                @argcheck(length(ivpa) == size(iv, 2), DimensionMismatch)
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts), typeof(iv),
                   typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
    end
end
function ReturnsResult(; nx::Union{Nothing, <:AbstractVector{<:AbstractString}} = nothing,
                       X::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                       nf::Union{Nothing, <:AbstractVector{<:AbstractString}} = nothing,
                       F::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                       ts::Union{Nothing, <:AbstractVector{<:Dates.AbstractTime}} = nothing,
                       iv::Union{Nothing, <:AbstractMatrix{<:Real}} = nothing,
                       ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    return ReturnsResult(nx, X, nf, F, ts, iv, ivpa)
end
function returns_result_view(rd::ReturnsResult, i::AbstractVector)
    nx = nothing_scalar_array_view(rd.nx, i)
    X = isnothing(rd.X) ? nothing : view(rd.X, :, i)
    iv = isnothing(rd.iv) ? nothing : view(rd.iv, :, i)
    ivpa = nothing_scalar_array_view(rd.ivpa, i)
    return ReturnsResult(; nx = nx, X = X, nf = rd.nf, F = rd.F, ts = rd.ts, iv = iv,
                         ivpa = ivpa)
end
"""
    prices_to_returns(X::TimeArray; F::TimeArray = TimeArray(TimeType[], []),
                      iv::Union{Nothing, <:TimeArray} = nothing,
                      ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      ret_method::Symbol = :simple, padding::Bool = false,
                      missing_col_percent::Real = 1.0,
                      missing_row_percent::Union{Nothing, <:Real} = 1.0,
                      collapse_args::Tuple = (), map_func::Union{Nothing, Function} = nothing,
                      join_method::Symbol = :outer,
                      impute_method::Union{Nothing, <:Impute.Imputor} = nothing)

Convert price data (and optionally factor data) in `TimeArray` format to returns, with flexible handling of missing data, imputation, and optional implied volatility information.

ReturnsResult a [`ReturnsResult`](@ref) containing asset and factor returns, time series, and optional implied volatility data, suitable for downstream portfolio optimization.

# Arguments

  - `X`: Asset price data (timestamps × assets).
  - `F`: Optional Factor price data (timestamps × factors).
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
"""
function prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                           iv::Union{Nothing, <:TimeArray} = nothing,
                           ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                           ret_method::Symbol = :simple, padding::Bool = false,
                           missing_col_percent::Real = 1.0,
                           missing_row_percent::Union{Nothing, <:Real} = 1.0,
                           collapse_args::Tuple = (),
                           map_func::Union{Nothing, Function} = nothing,
                           join_method::Symbol = :outer,
                           impute_method::Union{Nothing, <:Impute.Imputor} = nothing)
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError("0 < missing_col_percent <= 1 must hold. Got\nmissing_col_percent => $missing_col_percent"))
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent),
                  DomainError("0 < missing_row_percent <= 1 must hold. Got\nmissing_row_percent => $missing_row_percent"))
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
    X = DataFrame(percentchange(TimeArray(X; timestamp = :timestamp), ret_method;
                                padding = padding))
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

export drop_correlated, drop_incomplete, ReturnsResult, prices_to_returns
