"""
```julia
abstract type AbstractReturnsResult <: AbstractResult end
```

Abstract supertype for all returns result types in PortfolioOptimisers.jl.

All concrete types representing the result of returns calculations (e.g., asset returns, factor returns)
should subtype `AbstractReturnsResult`. This enables a consistent interface for downstream analysis
and optimization routines.

# Related

  - [`AbstractResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
abstract type AbstractReturnsResult <: AbstractResult end

function assert_nonneg_finite_val(val::AbstractDict)
    @argcheck(!isempty(val))
    @argcheck(any(isfinite, values(val)))
    @argcheck(all(x -> x >= zero(x), values(val)))
    return nothing
end
function assert_nonneg_finite_val(val::AbstractVector{<:Pair})
    @argcheck(!isempty(val))
    @argcheck(any(isfinite, getindex.(val, 2)))
    @argcheck(all(x -> x[2] >= zero(x[2]), val))
    return nothing
end
function assert_nonneg_finite_val(val::AbstractVector{<:Real})
    @argcheck(!isempty(val))
    @argcheck(any(isfinite, val))
    @argcheck(all(x -> x >= zero(x), val))
    return nothing
end
function assert_nonneg_finite_val(val::Pair)
    @argcheck(isfinite(val[2]) && val[2] >= zero(eltype(val[2])))
    return nothing
end
function assert_nonneg_finite_val(val::Real)
    @argcheck(isfinite(val) && val >= zero(eltype(val)))
    return nothing
end
function assert_nonneg_finite_val(::Nothing)
    return nothing
end

"""
```julia
assert_matrix_issquare(A::AbstractMatrix)
```

Assert that `A` is a square matrix.
"""
function assert_matrix_issquare(A::AbstractMatrix)
    @argcheck(size(A, 1) == size(A, 2),
              DimensionMismatch("matrix is not square: dimensions are $(size(A))"))
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
```julia
struct ReturnsResult{T1, T2, T3, T4, T5, T6, T7} <: AbstractReturnsResult
    nx::T1
    X::T2
    nf::T3
    F::T4
    ts::T5
    iv::T6
    ivpa::T7
end
```

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

```julia
ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
              X::Union{Nothing, <:AbstractMatrix} = nothing,
              nf::Union{Nothing, <:AbstractVector} = nothing,
              F::Union{Nothing, <:AbstractMatrix} = nothing,
              ts::Union{Nothing, <:AbstractVector} = nothing,
              iv::Union{Nothing, <:AbstractMatrix} = nothing,
              ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing)
```

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
    nx | Vector{String}: ["A", "B"]
     X | 2×2 Matrix{Float64}
    nf | nothing
     F | nothing
    ts | nothing
    iv | nothing
  ivpa | nothing
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
    function ReturnsResult(nx::Union{Nothing, <:AbstractVector},
                           X::Union{Nothing, <:AbstractMatrix},
                           nf::Union{Nothing, <:AbstractVector},
                           F::Union{Nothing, <:AbstractMatrix},
                           ts::Union{Nothing, <:AbstractVector},
                           iv::Union{Nothing, <:AbstractMatrix},
                           ivpa::Union{Nothing, <:Real, <:AbstractVector})
        nxs_flag = !isnothing(nx)
        X_flag = !isnothing(X)
        if nxs_flag || X_flag
            @argcheck((nxs_flag && X_flag),
                      IsNothingError(uppercasefirst(mul_cond_msg(some_msg("`nx`", nx),
                                                                 some_msg("`X`", X)))))
            @argcheck(!isempty(nx) && !isempty(X),
                      IsEmptyError(uppercasefirst(mul_cond_msg(non_empty_msg(`nx`, nx),
                                                               non_empty_msg("`X`", X)))))
            @argcheck(length(nx) == size(X, 2),
                      DimensionMismatch(comp_msg("length of `nx`",
                                                 "number of columns of `X`", :eq,
                                                 length(nx), size(X, 2)) * "."))
        end
        nfs_flag = !isnothing(nf)
        F_flag = !isnothing(F)
        if nfs_flag || F_flag
            @argcheck(nfs_flag && F_flag,
                      IsNothingError(uppercasefirst(mul_cond_msg(some_msg("`nf`", nf),
                                                                 some_msg("`F`", F)))))
            @argcheck(!isempty(nf) && !isempty(F),
                      IsEmptyError(uppercasefirst(mul_cond_msg(non_empty_msg(`nf`, nf),
                                                               non_empty_msg("`F`", F)))))
            @argcheck(length(nf) == size(F, 2),
                      DimensionMismatch(comp_msg("length of `nf`",
                                                 "number of columns of `F`", :eq,
                                                 length(nf), size(F, 2)) * "."))
        end
        if X_flag && F_flag
            @argcheck(size(X, 1) == size(F, 1),
                      DimensionMismatch(comp_msg("number of rows of `X`",
                                                 "number of rows of `F`", :eq, size(X, 1),
                                                 size(F, 1)) * "."))
        end
        if !isnothing(ts)
            @argcheck(!isempty(ts), IsEmptyError(non_empty_msg("`ts`") * "."))
            if X_flag
                @argcheck(length(ts) == size(X, 1),
                          DimensionMismatch(uppercasefirst(comp_msg("length of `ts`",
                                                                    "number of rows of `X`",
                                                                    :eq, length(ts),
                                                                    size(X, 1))) * "."))
            elseif F_flag
                @argcheck(length(ts) == size(F, 1),
                          DimensionMismatch(uppercasefirst(comp_msg("length of `ts`",
                                                                    "number of rows of `F`",
                                                                    :eq, length(ts),
                                                                    size(F, 1))) * "."))
            else
                throw(IsNothingEmptyError(uppercasefirst("at least one of " *
                                                         mul_cond_msg(nothing_non_empty_msg("`X`",
                                                                                            X),
                                                                      nothing_non_empty_msg("`F`",
                                                                                            F))) *
                                          "."))
            end
        end
        if !isnothing(iv)
            @argcheck(!isempty(iv), IsEmptyError(non_empty_msg("`iv`") * "."))
            @argcheck(size(iv) == size(X),
                      DimensionMismatch(uppercasefirst(comp_msg("size of `iv`",
                                                                "size of `X`", :eq,
                                                                size(iv), size(X))) * "."))
            @argcheck(all(x -> x >= zero(eltype(iv)), iv),
                      DomainError(iv, "all entries of " * non_neg_msg("`iv`") * "."))
            if isa(ivpa, Real)
                @argcheck(isfinite(ivpa), DomainError(ivpa, non_finite_msg("`ivpa`") * "."))
                @argcheck(ivpa > zero(ivpa),
                          DomainError(ivpa, comp_msg("`ivpa`", zero(ivpa), :gt) * "."))
            elseif isa(ivpa, AbstractVector)
                @argcheck(!isempty(ivpa), IsEmptyError(non_empty_msg("`ivpa`") * "."))
                @argcheck(length(ivpa) == size(iv, 2),
                          DimensionMismatch(uppercasefirst(comp_msg("length of `ivpa`",
                                                                    "number of columns of `iv`",
                                                                    :eq, length(ivpa),
                                                                    size(iv, 2))) * "."))
                @argcheck(all(x -> isfinite(x), ivpa),
                          DomainError(ivpa,
                                      "all entries of " * non_finite_msg("`ivpa`") * "."))
                @argcheck(all(x -> x > zero(eltype(ivpa)), ivpa),
                          DomainError(ivpa,
                                      "all entries of " *
                                      comp_msg("`ivpa`", "0", :geq) *
                                      "."))
            end
        end
        return new{typeof(nx), typeof(X), typeof(nf), typeof(F), typeof(ts), typeof(iv),
                   typeof(ivpa)}(nx, X, nf, F, ts, iv, ivpa)
    end
end
function ReturnsResult(; nx::Union{Nothing, <:AbstractVector} = nothing,
                       X::Union{Nothing, <:AbstractMatrix} = nothing,
                       nf::Union{Nothing, <:AbstractVector} = nothing,
                       F::Union{Nothing, <:AbstractMatrix} = nothing,
                       ts::Union{Nothing, <:AbstractVector} = nothing,
                       iv::Union{Nothing, <:AbstractMatrix} = nothing,
                       ivpa::Union{Nothing, <:Real, <:AbstractVector} = nothing)
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
```julia
prices_to_returns(X::TimeArray; F::TimeArray = TimeArray(TimeType[], []),
                  iv::Union{Nothing, <:TimeArray} = nothing,
                  ivpa::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                  ret_method::Symbol = :simple, padding::Bool = false,
                  missing_col_percent::Real = 1.0,
                  missing_row_percent::Union{Nothing, <:Real} = 1.0,
                  collapse_args::Tuple = (), map_func::Union{Nothing, Function} = nothing,
                  join_method::Symbol = :outer,
                  impute_method::Union{Nothing, <:Impute.Imputor} = nothing)
```

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
    @argcheck(zero(missing_col_percent) < missing_col_percent <= one(missing_col_percent),
              DomainError(missing_col_percent,
                          range_msg("`missing_col_percent`", zero(missing_col_percent),
                                    one(missing_col_percent), nothing, false, true) * "."))
    if !isnothing(missing_row_percent)
        @argcheck(zero(missing_row_percent) <
                  missing_row_percent <=
                  one(missing_row_percent),
                  DomainError(missing_row_percent,
                              range_msg("`missing_row_percent`", nothing,
                                        zero(missing_row_percent), one(missing_row_percent),
                                        false, true) * "."))
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

"""
```julia
brinson_attribution(X::TimeArray, w::AbstractVector, wb::AbstractVector,
                    asset_classes::DataFrame, col; date0 = nothing, date1 = nothing)
```
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
```julia
⊗(A::AbstractArray, B::AbstractArray)
```

Tensor product of two arrays. ReturnsResult a matrix of size `(length(A), length(B))` where each element is the product of elements from `A` and `B`.

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
```julia
⊙(A, B)
```

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
```julia
⊘(A, B)
```

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
```julia
⊕(A, B)
```

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
```julia
⊖(A, B)
```

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
```julia
dot_scalar(a::Real, b::AbstractVector)
dot_scalar(a::AbstractVector, b::Real)
dot_scalar(a::AbstractVector, b::AbstractVector)
```

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
```julia
nothing_scalar_array_view(x, i)
```

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
```julia
nothing_scalar_array_view_odd_order(x, i, j)
```

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
```julia
nothing_scalar_array_getindex(x, i)
nothing_scalar_array_getindex(x, i, j)
```

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
```julia
fourth_moment_index_factory(N::Integer, i::AbstractVector)
```

Constructs an index vector for extracting the fourth moment submatrix corresponding to indices `i` from a covariance matrix of size `N × N`.

# Arguments

  - `N`: Size of the full covariance matrix.
  - `i`: Indices of the variables of interest.

# Returns

  - `idx::Vector{Int}`: Indices for extracting the fourth moment submatrix.

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
```julia
traverse_concrete_subtypes(t; ctarr::Union{Nothing, <:AbstractVector} = nothing)
```

Recursively traverse all subtypes of the given abstract type `t` and collect all concrete struct types into `ctarr`.

# Arguments

  - `t`: An abstract type whose subtypes will be traversed.
  - `ctarr`: Optional An array to collect the concrete types. If not provided, a new empty array is created.

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
```julia
concrete_typed_array(A::AbstractArray)
```

Convert an `AbstractArray` `A` to a concrete typed array, where each element is of the same type as the elements of `A`.

This is useful for converting arrays with abstract element types to arrays with concrete element types, which can improve performance in some cases.

# Arguments

  - `A`: The input array.

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
