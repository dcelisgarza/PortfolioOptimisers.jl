"""
    AbstractReturnsResult <: AbstractResult

Abstract supertype for all returns result types in PortfolioOptimisers.jl.

All concrete types representing the result of returns calculations (e.g., asset returns, factor returns)
should subtype `AbstractReturnsResult`. This enables a consistent interface for downstream analysis
and optimization routines.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end

"""
    assert_matrix_issquare(A::AbstractMatrix)

Assert that `A` is a square matrix.
"""
function assert_matrix_issquare(A::AbstractMatrix)
    @smart_assert(size(A, 1) == size(A, 2))
end
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

"""
    struct ReturnsResult{T1 <: Union{Nothing, <:AbstractVector},
                         T2 <: Union{Nothing, <:AbstractMatrix},
                         T3 <: Union{Nothing, <:AbstractVector},
                         T4 <: Union{Nothing, <:AbstractMatrix},
                         T5 <: Union{Nothing, <:AbstractVector},
                         T6 <: Union{Nothing, <:AbstractMatrix},
                         T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
           AbstractReturnsResult
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

  - `nx::Union{Nothing, AbstractVector}`: Names or identifiers of asset columns.
  - `X::Union{Nothing, AbstractMatrix}`: Asset returns matrix (observations × assets).
  - `nf::Union{Nothing, AbstractVector}`: Names or identifiers of factor columns.
  - `F::Union{Nothing, AbstractMatrix}`: Factor returns matrix (observations × factors).
  - `ts::Union{Nothing, AbstractVector}`: Optional time series (e.g., timestamps) for each observation.
  - `iv::Union{Nothing, AbstractMatrix}`: Implied volatilities matrix.
  - `ivpa::Union{Nothing, Real, AbstractVector{<:Real}}`: Implied volatility risk premium adjustment.

# Constructor

    ReturnsResult(; nx=nothing, X=nothing, nf=nothing, F=nothing, ts=nothing, iv=nothing, ivpa=nothing)

Keyword arguments correspond to the fields above. The constructor performs internal consistency checks (e.g., matching dimensions, non-emptiness, positivity for variances).

# Related

  - [`AbstractReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
"""
struct ReturnsResult{T1 <: Union{Nothing, <:AbstractVector},
                     T2 <: Union{Nothing, <:AbstractMatrix},
                     T3 <: Union{Nothing, <:AbstractVector},
                     T4 <: Union{Nothing, <:AbstractMatrix},
                     T5 <: Union{Nothing, <:AbstractVector},
                     T6 <: Union{Nothing, <:AbstractMatrix},
                     T7 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}}} <:
       AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
end
"""
    ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
                    X::Union{Nothing, <:AbstractMatrix} = nothing,
                    nf::Union{Nothing, <:AbstractVector} = nothing,
                    F::Union{Nothing, <:AbstractMatrix} = nothing,
                    ts::Union{Nothing, <:AbstractVector} = nothing,
                    iv::Union{Nothing, <:AbstractMatrix} = nothing,
                    ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)

Construct a [`ReturnsResult`](@ref) object, validating dimensions and types for asset and factor returns, time series, and instrument variance data.

# Arguments

  - `nx::Union{Nothing, AbstractVector}`: Asset names or identifiers.
  - `X::Union{Nothing, AbstractMatrix}`: Asset returns matrix.
  - `nf::Union{Nothing, AbstractVector}`: Factor names or identifiers.
  - `F::Union{Nothing, AbstractMatrix}`: Factor returns matrix.
  - `ts::Union{Nothing, AbstractVector}`: Time series (e.g., timestamps).
  - `iv::Union{Nothing, AbstractMatrix}`: Implied volatility matrix.
  - `ivpa::Union{Nothing, Real, AbstractVector{<:Real}}`: Implied volatility risk premium adjustment.

# Validation

  - If `nx` or `X` is provided, both must be non-empty and `length(nx) == size(X, 2)`.
  - If `nf` or `F` is provided, both must be non-empty and `length(nf) == size(F, 2)`, and `size(X, 1) == size(F, 1)`.
  - If `ts` is provided, must be non-empty and `length(ts) == size(X, 1)`.
  - If `iv` is provided, must be non-empty, positive, and `size(iv) == size(X)`.
  - If `ivpa` is provided, must be positive and finite; if a vector, `length(ivpa) == size(iv, 2)`.

# Examples

```jldoctest
julia> ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 0.4])
ReturnsResult
    nx | Vector{String}: ["A", "B"]
     X | 2×2 Matrix{Float64}
    nf | nothing
     F | nothing
    ts | nothing
    iv | nothing
  ivpa | nothing
```

# Related

  - [`ReturnsResult`](@ref)
  - [`prices_to_returns`](@ref)
"""
function ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
                       X::Union{Nothing, <:AbstractMatrix} = nothing,
                       nf::Union{Nothing, <:AbstractVector} = nothing,
                       F::Union{Nothing, <:AbstractMatrix} = nothing,
                       ts::Union{Nothing, <:AbstractVector} = nothing,
                       iv::Union{Nothing, <:AbstractMatrix} = nothing,
                       ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
    nxs_flag = !isnothing(nx)
    X_flag = !isnothing(X)
    if nxs_flag || X_flag
        @smart_assert(nxs_flag && X_flag)
        @smart_assert(!isempty(nx) && !isempty(X))
        @smart_assert(length(nx) == size(X, 2))
    end
    nfs_flag = !isnothing(nf)
    F_flag = !isnothing(F)
    if (nfs_flag || F_flag) && nxs_flag
        @smart_assert(nfs_flag && F_flag)
        @smart_assert(!isempty(nf) && !isempty(F))
        @smart_assert(length(nf) == size(F, 2))
        @smart_assert(size(X, 1) == size(F, 1))
    end
    if !isnothing(ts)
        @smart_assert(!isempty(ts))
        @smart_assert(length(ts) == size(X, 1))
    end
    if !isnothing(iv)
        @smart_assert(!isempty(iv))
        @smart_assert(size(iv) == size(X))
        @smart_assert(all(x -> x > zero(eltype(iv)), iv))
        if !isnothing(ivpa)
            if isa(ivpa, Real)
                @smart_assert(isfinite(ivpa) && ivpa > zero(ivpa))
            elseif isa(ivpa, AbstractVector)
                @smart_assert(!isempty(ivpa))
                @smart_assert(all(isfinite, ivpa) && all(x -> x > zero(eltype(ivpa)), ivpa))
                @smart_assert(length(ivpa) == size(iv, 2))
            end
        end
    end
    return ReturnsResult{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts),
                         typeof(iv), typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
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
    prices_to_returns(X::TimeArray, F::TimeArray = TimeArray(TimeType[], []);
                      iv::Union{Nothing, <:TimeArray} = nothing,
                      ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                      ret_method::Symbol = :simple, padding::Bool = false,
                      missing_col_percent::Real = 1.0,
                      missing_row_percent::Union{Nothing, <:Real} = 1.0,
                      collapse_args::Tuple = (),
                      map_func::Union{Nothing, Function} = nothing,
                      join_method::Symbol = :outer,
                      impute_method::Union{Nothing, <:Impute.Imputor} = nothing)

Convert price data (and optionally factor data) in `TimeArray` format to returns, with flexible handling of missing data, imputation, and optional implied volatility information.

Returns a [`ReturnsResult`](@ref) containing asset and factor returns, time series, and optional implied volatility data, suitable for downstream portfolio optimization.

# Arguments

  - `X::TimeArray`: Asset price data (timestamps × assets).
  - `F::TimeArray`: (Optional) Factor price data (timestamps × factors).
  - `iv::Union{Nothing, TimeArray}`: (Optional) Implied volatility data.
  - `ivpa::Union{Nothing, Real, AbstractVector{<:Real}}`: (Optional) Implied volatility risk premium adjustment.
  - `ret_method::Symbol`: Return calculation method (`:simple` or `:log`).
  - `padding::Bool`: Whether to pad missing values in returns calculation.
  - `missing_col_percent::Real`: Maximum allowed fraction of missing values per column (asset/factor).
  - `missing_row_percent::Union{Nothing, Real}`: Maximum allowed fraction of missing values per row (timestamp).
  - `collapse_args::Tuple`: Arguments for collapsing the time series (e.g., to lower frequency).
  - `map_func::Union{Nothing, Function}`: Optional function to apply to the data before returns calculation.
  - `join_method::Symbol`: How to join asset and factor data (`:outer`, `:inner`, etc.).
  - `impute_method::Union{Nothing, Impute.Imputor}`: Optional imputation method for missing data.

# Returns

  - [`ReturnsResult`](@ref): Struct containing asset/factor returns, names, time series, and optional implied volatility data.

# Validation

  - Ensures consistency of asset/factor names and dimensions.
  - Handles missing data according to specified thresholds and imputation method.
  - Validates implied volatility and risk premium adjustment if provided.

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

julia> rr = prices_to_returns(X)
ReturnsResult
    nx | Vector{String}: ["A", "B"]
     X | 2×2 Matrix{Float64}
    nf | nothing
     F | nothing
    ts | Vector{Dates.Date}: [Dates.Date("2020-01-02"), Dates.Date("2020-01-03")]
    iv | nothing
  ivpa | nothing
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
    @smart_assert(zero(missing_col_percent) <
                  missing_col_percent <=
                  one(missing_col_percent))
    if !isnothing(missing_row_percent)
        @smart_assert(zero(missing_row_percent) <
                      missing_row_percent <=
                      one(missing_row_percent))
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
    f(x) =
        if isa(x, Number) && isnan(x)
            missing
        else
            x
        end
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
        @smart_assert(ts == timestamp(iv))
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
    brinson_attribution(X::TimeArray, w::AbstractVector, wb::AbstractVector,
                        asset_classes::DataFrame, col, date0 = nothing,
                        date1 = nothing)
"""
function brinson_attribution(X::TimeArray, w::AbstractVector, wb::AbstractVector,
                             asset_classes::DataFrame, col, date0 = nothing,
                             date1 = nothing)
    # Efficient filtering of date range
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = timestamp(X)
        idx = (DateTime(date0) .<= timestamps) .& (timestamps .<= DateTime(date1))
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)
    ret_b = dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrame(;
                   index = ["Asset Allocation", "Security Selection", "Interaction",
                            "Total Excess Return"])

    # Precompute class membership matrix for efficiency
    sets_mat = [class_j == class_i for class_j in classes, class_i in unique_classes]

    for (i, class_i) in enumerate(unique_classes)
        sets_i = view(sets_mat, :, i)

        w_i = dot(sets_i, w)
        wb_i = dot(sets_i, wb)

        ret_i = dot(ret .* sets_i, w) / w_i
        ret_b_i = dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end

"""
    ⊗(A::AbstractArray, B::AbstractArray)

Tensor product of two arrays. Returns a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊗([1, 2], [3, 4])
2×2 Matrix{Int64}:
 3  4
 6  8
```
"""
⊗(A::AbstractArray, B::AbstractArray) = reshape(kron(B, A), (length(A), length(B)))

"""
    ⊙(A, B)

Elementwise multiplication.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊙([1, 2], [3, 4])
2-element Vector{Int64}:
 3
 8

julia> PortfolioOptimisers.:⊙([1, 2], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊙(2, [3, 4])
2-element Vector{Int64}:
 6
 8

julia> PortfolioOptimisers.:⊙(2, 3)
6
```
"""
⊙(A::AbstractArray, B::AbstractArray) = A .* B
⊙(A::AbstractArray, B) = A * B
⊙(A, B::AbstractArray) = A * B
⊙(A, B) = A * B

"""
    ⊘(A, B)

Elementwise division.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊘([4, 9], [2, 3])
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘([4, 6], 2)
2-element Vector{Float64}:
 2.0
 3.0

julia> PortfolioOptimisers.:⊘(8, [2, 4])
2-element Vector{Float64}:
 4.0
 2.0

julia> PortfolioOptimisers.:⊘(8, 2)
4.0
```
"""
⊘(A::AbstractArray, B::AbstractArray) = A ./ B
⊘(A::AbstractArray, B) = A / B
⊘(A, B::AbstractArray) = A ./ B
⊘(A, B) = A / B

"""
    ⊕(A, B)

Elementwise addition.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊕([1, 2], [3, 4])
2-element Vector{Int64}:
 4
 6

julia> PortfolioOptimisers.:⊕([1, 2], 2)
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊕(2, [3, 4])
2-element Vector{Int64}:
 5
 6

julia> PortfolioOptimisers.:⊕(2, 3)
5
```
"""
⊕(A::AbstractArray, B::AbstractArray) = A .+ B
⊕(A::AbstractArray, B) = A .+ B
⊕(A, B::AbstractArray) = A .+ B
⊕(A, B) = A + B

"""
    ⊖(A, B)

Elementwise subtraction.

# Examples

```jldoctest
julia> PortfolioOptimisers.:⊖([4, 6], [1, 2])
2-element Vector{Int64}:
 3
 4

julia> PortfolioOptimisers.:⊖([4, 6], 2)
2-element Vector{Int64}:
 2
 4

julia> PortfolioOptimisers.:⊖(8, [2, 4])
2-element Vector{Int64}:
 6
 4

julia> PortfolioOptimisers.:⊖(8, 2)
6
```
"""
⊖(A::AbstractArray, B::AbstractArray) = A - B
⊖(A::AbstractArray, B) = A .- B
⊖(A, B::AbstractArray) = A .- B
⊖(A, B) = A - B

"""
    dot_scalar(a::Real, b::AbstractVector)
    dot_scalar(a::AbstractVector, b::Real)
    dot_scalar(a::AbstractVector, b::AbstractVector)

Efficient scalar and vector dot product utility.

  - If one argument is a scalar and the other a vector, returns the scalar times the sum of the vector.
  - If both arguments are vectors, returns their dot product.

# Arguments

  - `a::Real`, `b::AbstractVector`: Multiplies `a` by the sum of `b`.
  - `a::AbstractVector`, `b::Real`: Multiplies the sum of `a` by `b`.
  - `a::AbstractVector`, `b::AbstractVector`: Computes the dot product of `a` and `b`.

# Returns

  - `Real`: The resulting scalar.

# Examples

```jldoctest
julia> PortfolioOptimisers.dot_scalar(2.0, [1.0, 2.0, 3.0])
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], 2.0)
12.0

julia> PortfolioOptimisers.dot_scalar([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
32.0
```
"""
function dot_scalar(a::Real, b::AbstractVector)
    return a * sum(b)
end
function dot_scalar(a::AbstractVector, b::Real)
    return sum(a) * b
end
function dot_scalar(a::AbstractVector, b::AbstractVector)
    return dot(a, b)
end

"""
    nothing_scalar_array_view(x, i)

Utility for safely viewing or indexing into possibly `nothing`, scalar, or array values.

  - If `x` is `nothing`, returns `nothing`.
  - If `x` is a scalar, returns `x`.
  - If `x` is a vector, returns `view(x, i)`.
  - If `x` is a vector of vectors, returns `view.(x, Ref(i))`.
  - If `x` is a matrix or higher array, returns `view(x, i, i)`.

# Arguments

  - `x`: Input value, which may be `nothing`, a scalar, vector, or array.
  - `i`: Index or indices to view.

# Returns

  - The corresponding view or value, or `nothing` if `x` is `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_view(nothing, 1:2)

julia> PortfolioOptimisers.nothing_scalar_array_view(3.0, 1:2)
3.0

julia> PortfolioOptimisers.nothing_scalar_array_view([1.0, 2.0, 3.0], 2:3)
2-element view(::Vector{Float64}, 2:3) with eltype Float64:
 2.0
 3.0

julia> PortfolioOptimisers.nothing_scalar_array_view([[1, 2], [3, 4]], 1)
2-element Vector{SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 fill(1)
 fill(3)
```
"""
function nothing_scalar_array_view(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_view(x::Real, ::Any)
    return x
end
function nothing_scalar_array_view(x::AbstractVector, i)
    return view(x, i)
end
function nothing_scalar_array_view(x::AbstractVector{<:AbstractVector}, i)
    return view.(x, Ref(i))
end
function nothing_scalar_array_view(x::AbstractArray, i)
    return view(x, i, i)
end

"""
    nothing_scalar_array_view_odd_order(x, i, j)

Utility for safely viewing or indexing into possibly `nothing` or array values with two indices.

  - If `x` is `nothing`, returns `nothing`.
  - Otherwise, returns `view(x, i, j)`.

# Arguments

  - `x`: Input value, which may be `nothing` or an array.
  - `i`, `j`: Indices to view.

# Returns

  - The corresponding view or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_view_odd_order(nothing, 1, 2)

julia> PortfolioOptimisers.nothing_scalar_array_view_odd_order([1 2; 3 4], 1, 2)
0-dimensional view(::Matrix{Int64}, 1, 2) with eltype Int64:
2
```
"""
function nothing_scalar_array_view_odd_order(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_view_odd_order(x::AbstractArray, i, j)
    return view(x, i, j)
end

"""
    nothing_scalar_array_getindex(x, i)
    nothing_scalar_array_getindex(x, i, j)

Utility for safely indexing into possibly `nothing`, scalar, vector, or array values.

  - If `x` is `nothing`, returns `nothing`.
  - If `x` is a scalar, returns `x`.
  - If `x` is a vector, returns `x[i]`.
  - If `x` is a matrix, returns `x[i, i]` or `x[i, j]`.

# Arguments

  - `x`: Input value, which may be `nothing`, a scalar, vector, or matrix.
  - `i`, `j`: Indices.

# Returns

  - The corresponding value or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.nothing_scalar_array_getindex(nothing, 1)

julia> PortfolioOptimisers.nothing_scalar_array_getindex(3.0, 1)
3.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1.0, 2.0, 3.0], 2)
2.0

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1 2; 3 4], 2)
4

julia> PortfolioOptimisers.nothing_scalar_array_getindex([1 2; 3 4], 1, 2)
2
```
"""
function nothing_scalar_array_getindex(x::Real, ::Any)
    return x
end
function nothing_scalar_array_getindex(::Nothing, ::Any)
    return nothing
end
function nothing_scalar_array_getindex(x::AbstractVector, i)
    return x[i]
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i)
    return x[i, i]
end
function nothing_scalar_array_getindex(::Nothing, i, j)
    return nothing
end
function nothing_scalar_array_getindex(x::AbstractMatrix, i, j)
    return x[i, j]
end

"""
    nothing_asset_sets_view(x, i)

Utility for safely viewing or indexing into possibly `nothing` or DataFrame values.

  - If `x` is `nothing`, returns `nothing`.
  - If `x` is a DataFrame, returns `view(x, i, :)`.

# Arguments

  - `x`: Input value, which may be `nothing` or a DataFrame.
  - `i`: Indices.

# Returns

  - The corresponding view or `nothing`.

# Examples

```jldoctest
julia> using DataFrames

julia> df = DataFrame(; A = 1:5, B = 6:10, C = 11:15)
5×3 DataFrame
 Row │ A      B      C
     │ Int64  Int64  Int64
─────┼─────────────────────
   1 │     1      6     11
   2 │     2      7     12
   3 │     3      8     13
   4 │     4      9     14
   5 │     5     10     15

julia> PortfolioOptimisers.nothing_asset_sets_view(df, 1:2)
2×3 SubDataFrame
 Row │ A      B      C
     │ Int64  Int64  Int64
─────┼─────────────────────
   1 │     1      6     11
   2 │     2      7     12
```
"""
function nothing_asset_sets_view(::Nothing, ::Any)
    return nothing
end
function nothing_asset_sets_view(x::AbstractDataFrame, i)
    return view(x, i, :)
end

"""
    fourth_moment_index_factory(N::Integer, i::AbstractVector)

Constructs an index vector for extracting the fourth moment submatrix corresponding to indices `i` from a covariance matrix of size `N × N`.

# Arguments

  - `N::Integer`: Size of the full covariance matrix.
  - `i::AbstractVector`: Indices of the variables of interest.

# Returns

  - `Vector{Int}`: Indices for extracting the fourth moment submatrix.

# Examples

```jldoctest
julia> PortfolioOptimisers.fourth_moment_index_factory(3, [1, 2])
4-element Vector{Int64}:
 1
 2
 4
 5
```
"""
function fourth_moment_index_factory(N::Integer, i)
    idx = sizehint!(Int[], length(i)^2)
    for c in i
        append!(idx, (((c - 1) * N + 1):(c * N))[i])
    end
    return idx
end

"""
    traverse_concrete_subtypes(t, ctarr::Union{Nothing, <:AbstractVector} = nothing)

Recursively traverse all subtypes of the given abstract type `t` and collect all concrete struct types into `ctarr`.

# Arguments

  - `t`: An abstract type whose subtypes will be traversed.
  - `ctarr::Union{Nothing, <:AbstractVector}`: (Optional) An array to collect the concrete types. If not provided, a new empty array is created.

# Returns

An array containing all concrete struct types that are subtypes (direct or indirect) of `types`.

# Examples

```julia
julia> abstract type MyAbstract end

julia> struct MyConcrete1 <: MyAbstract end

julia> struct MyConcrete2 <: MyAbstract end

julia> traverse_concrete_subtypes(MyAbstract)
2-element Vector{Any}:
 MyConcrete1
 MyConcrete2
```
"""
function traverse_concrete_subtypes(t, ctarr::Union{Nothing, <:AbstractVector} = nothing)
    if isnothing(ctarr)
        ctarr = []
    end
    sts = subtypes(t)
    for st in sts
        if !isstructtype(st)
            traverse_concrete_subtypes(st, ctarr)
        else
            push!(ctarr, st)
        end
    end
    return ctarr
end

"""
    concrete_typed_array(A::AbstractArray)

Convert an `AbstractArray` `A` to a concrete typed array, where each element is of the same type as the elements of `A`.

This is useful for converting arrays with abstract element types to arrays with concrete element types, which can improve performance in some cases.

# Arguments

  - `A::AbstractArray`: The input array.

# Returns

A new array with the same shape as `A`, but with a concrete element type inferred from the elements of `A`.

# Examples

```jldoctest
julia> A = Any[1, 2.0, 3];

julia> PortfolioOptimisers.concrete_typed_array(A)
3-element reshape(::Vector{Union{Float64, Int64}}, 3) with eltype Union{Float64, Int64}:
 1
 2.0
 3
```
"""
function concrete_typed_array(A::AbstractArray)
    return reshape(Union{typeof.(A)...}[A...], size(A))
end
function factory(::Nothing, args...; kwargs...)
    return nothing
end

export drop_correlated, drop_incomplete, ReturnsResult, prices_to_returns,
       brinson_attribution, factory, traverse_concrete_subtypes
