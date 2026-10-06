"""
    assert_observed_factor_returns(fx::Nothing, csr, K::Integer) -> nothing
    assert_observed_factor_returns(fx::MatNum, csr, K::Integer) -> nothing

Check the observed factor returns of a [`CrossSectionalFactorModel`](@ref) against the fit and the factor axis.

The observed factors are the trailing columns of the reduced axis, after the factors that the fit estimated. So `csr.f` and `fx` together fill the reduced axis, row for row, and every consumer that reads the two side by side with [`cross_sectional_factor_returns`](@ref) meets one matrix of the width of the loadings. A block with no observed factor takes the method over `Nothing`.

# Arguments

  - `fx`: The observed factor returns, `observations × factors`, or `nothing`.
  - `csr`: The nested cross-sectional regression result, or `nothing`.
  - `K`: Number of factors on the reduced axis, the column count of `L`, or of `M` when `L` is unset.

# Validation

  - `csr` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - `!isempty(fx)`, and `size(fx, 1) == size(csr.f, 1)`.
  - `size(csr.f, 2) + size(fx, 2) == K`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`cross_sectional_factor_returns`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
"""
function assert_observed_factor_returns(::Nothing, ::Any, ::Integer)::Nothing
    return nothing
end
function assert_observed_factor_returns(fx::MatNum, csr::Option{<:CrossSectionalRegression},
                                        K::Integer)::Nothing
    @argcheck(!isnothing(csr),
              IsNothingError("fx holds the returns of the observed factors beside the factor returns of the fit, so it needs the fit: csr cannot be nothing when fx is given"))
    @argcheck(!isempty(fx), IsEmptyError("fx cannot be empty"))
    @argcheck(size(fx, 1) == size(csr.f, 1),
              DimensionMismatch("fx ($(size(fx, 1)) rows) must match csr.f ($(size(csr.f, 1)) rows), because the observed factors are observed on the rows the fit estimated"))
    @argcheck(size(csr.f, 2) + size(fx, 2) == K,
              DimensionMismatch("csr.f ($(size(csr.f, 2)) columns) and fx ($(size(fx, 2)) columns) must fill the reduced factor axis ($K columns), because the observed factors are its trailing columns"))
    return nothing
end
"""
    assert_idiosyncratic_covariance(esigma::Nothing, N::Integer)
    assert_idiosyncratic_covariance(esigma::VecNum, N::Integer)
    assert_idiosyncratic_covariance(esigma::MatNum, N::Integer)

Check the idiosyncratic covariance of a [`CrossSectionalFactorModel`](@ref) against the asset count `N`.

The idiosyncratic covariance takes either of two shapes, and the shape is the dispatch rather than a branch: a vector holds the idiosyncratic variances alone, and a matrix holds the full covariance an idiosyncratic correlation threshold produces. An absent covariance is checked by the method over `Nothing`, so no caller writes an `isnothing` test.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `N`: Number of assets the model carries.

# Validation

  - `!isempty(esigma)`.
  - `length(esigma) == N` when `esigma` is a vector.
  - `esigma` is square, and `size(esigma, 1) == N`, when `esigma` is a matrix.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
  - [`assert_matrix_issquare`](@ref)
"""
function assert_idiosyncratic_covariance(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_idiosyncratic_covariance(esigma::VecNum, N::Integer)::Nothing
    @argcheck(!isempty(esigma), IsEmptyError("esigma cannot be empty"))
    @argcheck(length(esigma) == N,
              DimensionMismatch("esigma ($(length(esigma))) must match the asset count ($N)"))
    return nothing
end
function assert_idiosyncratic_covariance(esigma::MatNum, N::Integer)::Nothing
    @argcheck(!isempty(esigma), IsEmptyError("esigma cannot be empty"))
    assert_matrix_issquare(esigma, :esigma)
    @argcheck(size(esigma, 1) == N,
              DimensionMismatch("esigma ($(size(esigma, 1))) must match the asset count ($N)"))
    return nothing
end
"""
    idiosyncratic_covariance_view(esigma::Nothing, i)
    idiosyncratic_covariance_view(esigma::VecNum, i)
    idiosyncratic_covariance_view(esigma::MatNum, i)

Return a view of an idiosyncratic covariance, selecting only the assets indexed by `i`.

The shape is the dispatch, as it is in [`assert_idiosyncratic_covariance`](@ref): a vector of variances is indexed once, and a full covariance matrix is indexed on both of its axes, which keeps the selected block square.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `i`: Indices of the assets to select.

# Returns

  - `esigma::Option{<:VecNum_MatNum}`: A view over the selected assets, or `nothing` when the model carries no idiosyncratic covariance.

# Examples

```jldoctest
julia> PortfolioOptimisers.idiosyncratic_covariance_view([1.0, 2.0, 3.0], [1, 3])
2-element view(::Vector{Float64}, [1, 3]) with eltype Float64:
 1.0
 3.0

julia> PortfolioOptimisers.idiosyncratic_covariance_view([1.0 0.0; 0.0 2.0], [2])
1×1 view(::Matrix{Float64}, [2], [2]) with eltype Float64:
 2.0
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_idiosyncratic_covariance`](@ref)
  - [`port_opt_view`](@ref)
"""
function idiosyncratic_covariance_view(::Nothing, args...)::Nothing
    return nothing
end
function idiosyncratic_covariance_view(esigma::VecNum, i)
    return view(esigma, i)
end
function idiosyncratic_covariance_view(esigma::MatNum, i)
    return view(esigma, i, i)
end
"""
    assert_idiosyncratic_count(x::Nothing, N::Integer, sym::Symbol)
    assert_idiosyncratic_count(x::VecNum, N::Integer, sym::Symbol)

Check a per-asset count of a loadings block, `edof` or `ediv`, against the asset count `N`.

The count is a vector with one entry per asset, or `nothing` when the fit recorded none. The method over `Nothing` accepts the absent count, so no caller writes an `isnothing` test. The check reads the length alone. An entry that is not finite or not positive is a valid record of a short fit, and the consumer that reads the count refuses it.

# Arguments

  - `x`: The count, or `nothing`.
  - `N`: Number of assets the block carries.
  - `sym`: Name of the field, for the error message.

# Validation

  - `!isempty(x)`, and `length(x) == N`.

# Returns

  - `nothing`.

# Related

  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_idiosyncratic_covariance`](@ref)
"""
function assert_idiosyncratic_count(::Nothing, ::Integer, ::Symbol)::Nothing
    return nothing
end
function assert_idiosyncratic_count(x::VecNum, N::Integer, sym::Symbol)::Nothing
    @argcheck(!isempty(x), IsEmptyError("$(sym) cannot be empty"))
    @argcheck(length(x) == N,
              DimensionMismatch("$(sym) ($(length(x))) must match the asset count ($N)"))
    return nothing
end
"""
    cs_history_assets(A::Nothing, N::Integer, sym::Symbol)
    cs_history_assets(A::MatNum, N::Integer, sym::Symbol)

Check the asset axis of an optional per-asset history of a [`CrossSectionalFactorModel`](@ref), and return its observation count.

A per-asset history holds one row per observation and one column per asset, so the asset axis is the second one. The count comes back so that the caller pins the observation axis of two histories against each other with [`assert_cs_history_obs`](@ref) and needs no `isnothing` test of its own.

# Arguments

  - `A`: A per-asset history, or `nothing`.
  - `N`: Number of assets the model carries.
  - `sym`: Name of the field, which the raise reports.

# Validation

  - `!isempty(A)`.
  - `size(A, 2) == N`.

# Returns

  - `T::Option{<:Integer}`: The observation count of `A`, or `nothing` when `A` is absent.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_cs_history_obs`](@ref)
"""
function cs_history_assets(::Nothing, ::Integer, ::Symbol)::Nothing
    return nothing
end
function cs_history_assets(A::MatNum, N::Integer, sym::Symbol)
    @argcheck(!isempty(A), IsEmptyError("$sym cannot be empty"))
    @argcheck(size(A, 2) == N,
              DimensionMismatch("$sym ($(size(A, 2)) columns) must match the asset count ($N)"))
    return size(A, 1)
end
"""
    assert_cs_history_obs(a::Option{<:Integer}, b::Option{<:Integer}, asym::Symbol, bsym::Symbol)
    assert_cs_history_obs(a::Integer, b::Integer, asym::Symbol, bsym::Symbol)

Check that two per-asset histories of a [`CrossSectionalFactorModel`](@ref) agree on the observation axis.

A history that the model does not carry constrains nothing, so the pair is checked only when [`cs_history_assets`](@ref) returned a count for both.

# Arguments

  - `a`: Observation count of the first history, or `nothing`.
  - `b`: Observation count of the second history, or `nothing`.
  - `asym`: Name of the first field, which the raise reports.
  - `bsym`: Name of the second field, which the raise reports.

# Validation

  - `a == b`, when both counts are present.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`cs_history_assets`](@ref)
"""
function assert_cs_history_obs(::Option{<:Integer}, ::Option{<:Integer}, ::Symbol,
                               ::Symbol)::Nothing
    return nothing
end
function assert_cs_history_obs(a::Integer, b::Integer, asym::Symbol, bsym::Symbol)::Nothing
    @argcheck(a == b,
              DimensionMismatch("$asym ($a rows) and $bsym ($b rows) must agree on the observation axis"))
    return nothing
end
"""
    assert_exposure_history(Ms::Nothing, N::Integer, K::Integer)
    assert_exposure_history(Ms::Arr3Num, N::Integer, K::Integer)

Check the exposure history of a [`CrossSectionalFactorModel`](@ref) against the asset count `N` and the factor count `K`.

The exposure history holds one slice per observation, and a slice has the shape of the loadings matrix, so its second and third axes are the asset axis and the factor axis.

# Arguments

  - `Ms`: An exposure history `observations × assets × factors`, or `nothing`.
  - `N`: Number of assets the model carries.
  - `K`: Number of raw factors the model carries.

# Validation

  - `!isempty(Ms)`.
  - `size(Ms, 2) == N`.
  - `size(Ms, 3) == K`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
"""
function assert_exposure_history(::Nothing, ::Integer, ::Integer)::Nothing
    return nothing
end
function assert_exposure_history(Ms::Arr3Num, N::Integer, K::Integer)::Nothing
    @argcheck(!isempty(Ms), IsEmptyError("Ms cannot be empty"))
    @argcheck(size(Ms, 2) == N,
              DimensionMismatch("Ms ($(size(Ms, 2)) rows per slice) must match the asset count ($N)"))
    @argcheck(size(Ms, 3) == K,
              DimensionMismatch("Ms ($(size(Ms, 3)) columns per slice) must match the factor count ($K)"))
    return nothing
end
"""
    assert_cs_regression_assets(csr::Nothing, N::Integer)
    assert_cs_regression_assets(csr::CrossSectionalRegression, N::Integer)

Check the asset axis of the fit a [`CrossSectionalFactorModel`](@ref) nests against the asset count `N`.

The residuals of a [`CrossSectionalRegression`](@ref) hold one row per observation and one column per asset, so the asset axis is the second one, as it is for every per-asset history of the model.

# Arguments

  - `csr`: The nested cross-sectional regression result, or `nothing`.
  - `N`: Number of assets the model carries.

# Validation

  - `size(csr.eps, 2) == N`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function assert_cs_regression_assets(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_cs_regression_assets(csr::CrossSectionalRegression, N::Integer)::Nothing
    @argcheck(size(csr.eps, 2) == N,
              DimensionMismatch("csr.eps ($(size(csr.eps, 2)) columns) must match the asset count ($N)"))
    return nothing
end
"""
    assert_return_forecast_assets(rf::Nothing, N::Integer)
    assert_return_forecast_assets(rf::AbstractReturnForecastResult, N::Integer)

Check the Return Forecast of a [`CrossSectionalFactorModel`](@ref) against the asset count `N`.

A model carries the Return Forecast its prior fitted, or `nothing`, and the absent case is the method over `Nothing`. The check reads the two fields every member of the family answers, `mu` and `hist`, and never the member's own fields, so a new member needs no new method here.

# Arguments

  - `rf`: Return Forecast Result, or `nothing`.
  - `N`: Number of assets the model carries.

# Validation

  - `length(rf.mu) == N`.
  - The rules of [`cs_history_assets`](@ref) on `rf.hist`.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
  - [`cs_history_assets`](@ref)
"""
function assert_return_forecast_assets(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_return_forecast_assets(rf::AbstractReturnForecastResult,
                                       N::Integer)::Nothing
    @argcheck(length(rf.mu) == N,
              DimensionMismatch("rf.mu ($(length(rf.mu))) must match the asset count ($N)"))
    cs_history_assets(rf.hist, N, Symbol("rf.hist"))
    return nothing
end
"""
    assert_row_key_part(key::Nothing, T, sym::Symbol)
    assert_row_key_part(key::VecInt, T::Option{<:Integer}, sym::Symbol)
    assert_row_key_part(key::VecDate, T::Option{<:Integer}, sym::Symbol)

Check one part of the row key of a factor model block.

A row key names the observation of the returns data that each row of a factor model block describes. It has two parts: the position of each row in the returns data that the prior read, and the timestamp of each row. [`CrossSectionalFactorModel`](@ref) and [`Regression`](@ref) carry both parts, and each one checks them with this function. A [`factor_attribution`](@ref) of a cross-validation reads the key in the order of time, so each part must increase strictly.

# Arguments

  - `key`: The positions or the timestamps of the rows, or `nothing`.
  - `T`: The number of rows that the key describes, or `nothing` when the block does not state it.
  - `sym`: The name of the field that holds `key`, for the error message.

# Validation

  - [`assert_row_key_length`](@ref) on `key` and `T`.
  - The entries of a vector of positions are positive and increase strictly. Raises an `ArgumentError`.
  - The entries of a vector of timestamps increase strictly. Raises an `ArgumentError`.

# Returns

  - `nothing`.

# Related

  - [`assert_cs_block_rows`](@ref)
  - [`attribution_row_key`](@ref)
"""
function assert_row_key_part(::Nothing, ::Any, ::Symbol)::Nothing
    return nothing
end
function assert_row_key_part(key::VecInt, T::Option{<:Integer}, sym::Symbol)::Nothing
    assert_row_key_length(key, T, sym)
    @argcheck(isempty(key) || (first(key) >= 1 && all(>(0), diff(key))),
              ArgumentError("$sym must hold positive positions that increase strictly"))
    return nothing
end
function assert_row_key_part(key::VecDate, T::Option{<:Integer}, sym::Symbol)::Nothing
    assert_row_key_length(key, T, sym)
    @argcheck(issorted(key; lt = <=), ArgumentError("$sym must increase strictly"))
    return nothing
end
"""
    assert_row_key_length(key::AbstractVector, T::Option{<:Integer}, sym::Symbol)

Check that one part of a row key has one entry for each row that it describes.

# Arguments

  - `key`: The positions or the timestamps of the rows.
  - `T`: The number of rows that the key describes, or `nothing` when the block does not state it.
  - `sym`: The name of the field that holds `key`, for the error message.

# Validation

  - When `T` is an integer, `length(key) == T`. Raises a `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`assert_row_key_part`](@ref)
"""
function assert_row_key_length(key::AbstractVector, T::Option{<:Integer},
                               sym::Symbol)::Nothing
    @argcheck(isnothing(T) || length(key) == T,
              DimensionMismatch("$sym ($(length(key))) must have one entry for each of the $T rows that the key describes"))
    return nothing
end
"""
    row_key_length(key::Nothing)
    row_key_length(key::AbstractVector)

Return the number of rows that one part of a row key describes, or `nothing` when the part is absent.

A block that states no row count of its own, such as a [`Regression`](@ref), checks the timestamps of its key against the length of its positions. This function gives that length.

# Arguments

  - `key`: The positions or the timestamps of the rows, or `nothing`.

# Returns

  - `T::Option{<:Integer}`: `length(key)`, or `nothing`.

# Related

  - [`assert_row_key_part`](@ref)
  - [`Regression`](@ref)
"""
function row_key_length(::Nothing)::Nothing
    return nothing
end
function row_key_length(key::AbstractVector)::Int
    return length(key)
end
"""
    assert_cs_block_rows(idx::Option{<:VecInt}, ts::Option{<:VecDate},
                         csr::Option{<:CrossSectionalRegression})

Check the row key of a [`CrossSectionalFactorModel`](@ref) against its fit.

The row key names the observation of the returns data that each row of the block describes. `idx` holds the position of each row in the returns data that the prior read, and `ts` holds its timestamp. A key describes the rows of the fit, so a block that carries a key carries a fit, and the key has one entry for each row of the fit.

# Arguments

  - `idx`: Position of each block row in the returns data, or `nothing`.
  - `ts`: Timestamp of each block row, or `nothing`.
  - `csr`: The cross-sectional fit of the block, or `nothing`.

# Validation

  - When `idx` or `ts` is present, `csr` is present. Raises an `ArgumentError`.
  - [`assert_row_key_part`](@ref) on `idx` and on `ts`, with the `size(csr.f, 1)` rows of the fit.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`assert_row_key_part`](@ref)
"""
function assert_cs_block_rows(idx::Option{<:VecInt}, ts::Option{<:VecDate},
                              csr::Option{<:CrossSectionalRegression})::Nothing
    if isnothing(idx) && isnothing(ts)
        return nothing
    end
    @argcheck(!isnothing(csr),
              ArgumentError("idx and ts name the observation of each row of the fit, so a block that carries either one must carry csr"))
    Tb = size(csr.f, 1)
    assert_row_key_part(idx, Tb, :idx)
    assert_row_key_part(ts, Tb, :ts)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the loadings, the factor-orthogonal expected return and the fitted history of a factor model fitted per observation across the assets.

The result is the cross-sectional member of [`AbstractLoadingsRegressionResult`](@ref), so a consumer that re-bases a constraint or decomposes risk in the factor basis reads it exactly as it reads a [`Regression`](@ref). `M` carries the **raw** loadings, whose columns are the named original factors, because a constraint must be written in names a caller can put in an equation. Every field after `b` is optional, so a model that keeps its loadings and drops its histories is a member of the family, and a caller that asks for a dropped history reads `nothing`.

**An unset `L` reads back as `M`.** A [`@forward_properties`](@ref) `swap(L, M)` rule makes `csfm.L` return `csfm.M` whenever `L` was not given, as [`Regression`](@ref) behaves, so a consumer that decomposes risk in the factor basis needs no `Nothing` branch, and `isnothing(csfm.L)` is never true. Read `getfield(csfm, :L)` when the unset case must be told apart, as [`port_opt_view`](@ref) does.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x}_{t} &= b_{t} \\boldsymbol{1} + \\mathbf{M}_{t} \\boldsymbol{f}_{t} + \\boldsymbol{\\varepsilon}_{t}\\,, \\\\
\\boldsymbol{\\mu} &= \\mathbf{M} \\boldsymbol{\\mu}_{f} + \\boldsymbol{b}\\,, \\\\
\\mathbf{M} &= \\mathbf{M}_{T}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_obs])
  - ``b_{t}``: Intercept of observation ``t``, the ``t``-th entry of the intercept vector [`CrossSectionalRegression`](@ref) carries. The term is absent when the fit carries no intercept.
  - ``\\boldsymbol{1}``: Vector of ones ``N \\times 1``.
  - ``\\mathbf{M}_{t}``: Exposure slice of observation ``t``, ``N \\times K``, the ``t``-th slice of `Ms`.
  - ``\\boldsymbol{f}_{t}``: Factor returns of observation ``t``, on the axis of ``\\mathbf{M}_{t}``.
  - ``\\boldsymbol{\\varepsilon}_{t}``: Idiosyncratic returns of observation ``t``, the part of ``\\boldsymbol{x}_{t}`` the exposures and the intercept do not explain.
  - $(math_dict[:mu_er])
  - ``\\boldsymbol{\\mu}_{f}``: Expected factor returns ``K \\times 1``. A factor prior carries it, and this result does not.
  - ``\\boldsymbol{b}``: Factor-orthogonal expected return ``N \\times 1``, `b`. It is the part of ``\\boldsymbol{\\mu}`` the factors do not span, so it is a term of the expected return and never a term of one observation.
  - ``\\mathbf{M}``: Loadings matrix ``N \\times K`` of the factor model, `M`. It is the last slice of the exposure history.
  - $(math_dict[:N])
  - $(math_dict[:K])
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalFactorModel(;
        M::MatNum,
        L::Option{<:MatNum} = nothing,
        b::VecNum,
        csr::Option{<:CrossSectionalRegression} = nothing,
        Ms::Option{<:Arr3Num} = nothing,
        vs::Option{<:MatNum} = nothing,
        esigma::Option{<:VecNum_MatNum} = nothing,
        edof::Option{<:VecNum} = nothing,
        ediv::Option{<:VecNum} = nothing,
        rw::Option{<:MatNum} = nothing,
        bw::Option{<:MatNum} = nothing,
        nf::Option{<:VecStr} = nothing,
        fam::Option{<:VecStr} = nothing,
        fcb::Option{<:AbstractFactorFamilyBasis} = nothing,
        lag::Option{<:Integer} = nothing,
        rf::Option{<:AbstractReturnForecastResult} = nothing,
        fx::Option{<:MatNum} = nothing,
        fr::Option{<:MatNum} = nothing,
        lambda::Option{<:Number} = nothing,
        c::Option{<:Number} = nothing,
        idx::Option{<:VecInt} = nothing,
        ts::Option{<:VecDate} = nothing
    ) -> CrossSectionalFactorModel

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(M)`, `!isempty(b)`, and `length(b) == size(M, 1)`.
  - `L` and `fcb` are present together, or absent together.
  - If provided, `!isempty(L)`, and `size(L, 1) == size(M, 1)`.
  - If provided, `!isempty(nf)`, `length(nf) == size(M, 2)`, and `nf` repeats no name.
  - If provided, `!isempty(fam)`, and `length(fam) == size(M, 2)`.
  - If provided, `!isempty(Ms)`, `size(Ms, 2) == size(M, 1)`, and `size(Ms, 3) == size(M, 2)`.
  - If provided, `size(csr.eps, 2) == size(M, 1)`.
  - If provided, `!isempty(vs)`, `!isempty(rw)`, `!isempty(bw)`, and each carries `size(M, 1)` columns.
  - Every two of `vs`, `rw` and `bw` that are present agree on the observation axis, so `size(rw) == size(bw) == size(vs)` when all three are present.
  - If provided, `!isempty(esigma)`, and `esigma` carries `size(M, 1)` entries when it is a vector, or is square with `size(M, 1)` rows when it is a matrix.
  - If provided, `edof` and `ediv` each carry `size(M, 1)` entries.
  - If provided, `lag >= 0`.
  - If provided, `length(rf.mu) == size(M, 1)`, and `rf.hist` carries `size(M, 1)` columns when the member computes one.
  - If provided, `fx` passes [`assert_observed_factor_returns`](@ref): `csr` is given, `fx` has the rows of `csr.f`, and the two fill the column count of `L` (or of `M` when `L` is unset).
  - If provided, `fr` needs `fcb` and `csr`, and `size(fr) == (size(csr.f, 1), size(M, 2))`.
  - The rules of [`assert_cs_block_rows`](@ref) on `idx` and `ts`.

## View parameters

`CrossSectionalFactorModel` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `M`, `L` and `b` are sliced on their **first** axis, which is the asset axis of a loadings result.
  - `csr` is viewed by its own [`port_opt_view`](@ref) method.
  - `Ms` is sliced on its **second** axis, which is the asset axis of a slice.
  - `vs`, `rw` and `bw` are sliced on their **second** axis, which is the asset axis of a per-asset history.
  - `esigma` is sliced by [`idiosyncratic_covariance_view`](@ref), on one axis or on both.
  - `edof` and `ediv` are sliced on their only axis, which is the asset axis.
  - `rf` is viewed by its own [`port_opt_view`](@ref) method, which cuts `mu` and `hist` on the asset axis.
  - `nf`, `fam`, `fcb`, `lag`, `fx`, `fr`, `lambda`, `c`, `idx` and `ts` pass through unchanged. Each is indexed by factor, by observation, or by nothing at all, and none follows an asset selection.

# Examples

```jldoctest
julia> CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0], b = [0.1, 0.2, 0.3],
                                 esigma = [0.4, 0.5, 0.6], fam = [\"style\", \"style\"], lag = 1)
CrossSectionalFactorModel
       M ┼ 3×2 Matrix{Float64}
       L ┼ 3×2 Matrix{Float64}
       b ┼ Vector{Float64}: [0.1, 0.2, 0.3]
     csr ┼ nothing
      Ms ┼ nothing
      vs ┼ nothing
  esigma ┼ Vector{Float64}: [0.4, 0.5, 0.6]
    edof ┼ nothing
    ediv ┼ nothing
      rw ┼ nothing
      bw ┼ nothing
      nf ┼ nothing
     fam ┼ Vector{String}: ["style", "style"]
     fcb ┼ nothing
     lag ┼ Int64: 1
      rf ┼ nothing
      fx ┼ nothing
      fr ┼ nothing
  lambda ┼ nothing
       c ┼ nothing
     idx ┼ nothing
      ts ┴ nothing
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalRegression`](@ref)
  - [`port_opt_view`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
  - [`AbstractReturnForecastResult`](@ref)
"""
@concrete struct CrossSectionalFactorModel <: AbstractLoadingsRegressionResult
    """
    $(field_dict[:M]) Its columns are the named original factors, so a constraint written in a factor's name resolves against it.
    """
    M
    """
    $(field_dict[:L]) It is set only after a family re-basis, and an unset `L` reads back as `M`.
    """
    L
    """
    Factor-orthogonal expected return, one entry per asset. It is the part of the expected return the factors do not span, and it is not the per-observation intercept of the fit, which `csr` carries.
    """
    b
    """
    The cross-sectional fit the model was built around. It carries the factor returns, the idiosyncratic returns, the eligible asset counts and the per-observation intercepts.
    """
    csr
    """
    Exposure history `observations × assets × factors`. Its last slice is the loadings matrix `M`. The constructor checks the two axes it shares with `M` and never the entries, so a caller that builds both keeps them in step itself.
    """
    Ms
    """
    Idiosyncratic variance history `observations × assets`. Row `t` holds the variances estimated from the observations up to `t`.
    """
    vs
    """
    $(field_dict[:esigma])
    """
    esigma
    """
    $(field_dict[:edof])
    """
    edof
    """
    $(field_dict[:ediv])
    """
    ediv
    """
    Regression weight history `observations × assets`. Entry `(t, i)` is the weight asset `i` carried in the fit of observation `t`, and a weight of zero excluded the pair.
    """
    rw
    """
    Benchmark weight history `observations × assets`. Entry `(t, i)` is the weight asset `i` carried in the benchmark of observation `t`.
    """
    bw
    """
    Name of each raw factor, one entry per column of `M`. The prior derives the axis from its Exposure Estimators and stores the answer here, so a consumer that names a factor reads one list.
    """
    nf
    """
    Family label of each raw factor, one entry per column of `M`.
    """
    fam
    """
    The family re-basis `L` is written in. It is present exactly when `L` is present. Its rows are the rows of `Ms`, one basis for the exposures of each observation, and its last row re-bases `M` into `L`. The factor return of row `t` of `csr.f` was fitted on the exposures of row `t - lag`, so it is written in the basis of row `t - lag`: a transform of a factor-return history reads the basis rows `1:(T - lag)` against the return rows `(1 + lag):T`.
    """
    fcb
    """
    Number of observations by which the exposures lag the returns.
    """
    lag
    """
    The Return Forecast the prior fitted, or `nothing`. Its `mu` is the forecast `b` was split out of, so a consumer reads the forecast the split consumed rather than refitting it.
    """
    rf
    """
    Returns of the observed factors, `observations × factors`, on the rows of `csr.f`, or `nothing` when the model has no observed factor. The fit observes them, for example the Currency Excess Returns of the Currency Factors, and does not estimate them, so `csr` does not carry them. The observed factors are the trailing columns of the raw axis and of the reduced axis, and [`cross_sectional_factor_returns`](@ref) reads them beside `csr.f`.
    """
    fx
    """
    Factor returns on the raw factor axis, `observations × factors`, on the rows of `csr.f` and with one column per column of `M`, or `nothing`. A block with a family re-basis carries it, and a block with none leaves it `nothing`, because its raw axis is the axis of [`cross_sectional_factor_returns`](@ref). The fit states the factor returns in the reduced basis of row `t - lag`, and `fcb` holds the basis of the rows of the block alone, so it cannot expand the first `lag` rows. The prior expands every row with the basis of its own history, and a consumer that needs a dropped factor's return, for example [`factor_model_summary`](@ref) and [`factor_attribution`](@ref), reads it here.
    """
    fr
    """
    Spanned Shrinkage the prior resolved, or `nothing` when the block was not built by a prior. A rule in the slot and a stated number are recorded alike, so a caller reads back the number the mean used.
    """
    lambda
    """
    Orthogonal Forecast Scale the prior resolved, or `nothing` when the block was not built by a prior. `b` is this scale times the unscaled orthogonal part of the Return Forecast.
    """
    c
    """
    Position of each row of the fit in the returns data that the prior read, or `nothing`. The fit drops the Descriptor warm-up and the exposure lag, so its first row is a later row of the data. A [`factor_attribution`](@ref) of a cross-validation reads it to find the rows of the block that each fold covers. A prior folded online counts the positions over every observation it folded.
    """
    idx
    """
    Timestamp of each row of the fit, or `nothing` when the returns data that the prior read carries no timestamps. When the folds of a cross-validation carry timestamps too, a [`factor_attribution`](@ref) matches the two by timestamp, so a prior fitted on other returns data still finds its rows.
    """
    ts
    function CrossSectionalFactorModel(M::MatNum, L::Option{<:MatNum}, b::VecNum,
                                       csr::Option{<:CrossSectionalRegression},
                                       Ms::Option{<:Arr3Num}, vs::Option{<:MatNum},
                                       esigma::Option{<:VecNum_MatNum},
                                       edof::Option{<:VecNum}, ediv::Option{<:VecNum},
                                       rw::Option{<:MatNum}, bw::Option{<:MatNum},
                                       nf::Option{<:VecStr}, fam::Option{<:VecStr},
                                       fcb::Option{<:AbstractFactorFamilyBasis},
                                       lag::Option{<:Integer},
                                       rf::Option{<:AbstractReturnForecastResult},
                                       fx::Option{<:MatNum}, fr::Option{<:MatNum},
                                       lambda::Option{<:Number}, c::Option{<:Number},
                                       idx::Option{<:VecInt}, ts::Option{<:VecDate})
        @argcheck(!isempty(M), IsEmptyError("M cannot be empty"))
        @argcheck(!isempty(b), IsEmptyError("b cannot be empty"))
        N = size(M, 1)
        K = size(M, 2)
        @argcheck(length(b) == N,
                  DimensionMismatch("b ($(length(b))) must match M ($N rows)"))
        @argcheck(isnothing(L) == isnothing(fcb),
                  ArgumentError("L and fcb must be present together or absent together"))
        if !isnothing(L)
            @argcheck(!isempty(L), IsEmptyError("L cannot be empty"))
            @argcheck(size(L, 1) == N,
                      DimensionMismatch("L ($(size(L, 1)) rows) must match M ($N rows)"))
        end
        if !isnothing(nf)
            @argcheck(!isempty(nf), IsEmptyError("nf cannot be empty"))
            @argcheck(length(nf) == K,
                      DimensionMismatch("nf ($(length(nf))) must match M ($K columns)"))
            @argcheck(allunique(nf), ArgumentError("nf must not repeat a factor name"))
        end
        if !isnothing(fam)
            @argcheck(!isempty(fam), IsEmptyError("fam cannot be empty"))
            @argcheck(length(fam) == K,
                      DimensionMismatch("fam ($(length(fam))) must match M ($K columns)"))
        end
        if !isnothing(lag)
            assert_nonneg(lag, :lag)
        end
        assert_exposure_history(Ms, N, K)
        assert_cs_regression_assets(csr, N)
        assert_idiosyncratic_covariance(esigma, N)
        assert_idiosyncratic_count(edof, N, :edof)
        assert_idiosyncratic_count(ediv, N, :ediv)
        assert_return_forecast_assets(rf, N)
        assert_observed_factor_returns(fx, csr, isnothing(L) ? K : size(L, 2))
        if !isnothing(fr)
            @argcheck(!isnothing(fcb) && !isnothing(csr),
                      ArgumentError("fr is the raw-axis factor return history of a re-based fit, so it needs fcb and csr"))
            @argcheck(size(fr) == (size(csr.f, 1), K),
                      DimensionMismatch("fr ($(size(fr))) must have the rows of csr.f ($(size(csr.f, 1))) and the columns of M ($K)"))
        end
        tvs = cs_history_assets(vs, N, :vs)
        trw = cs_history_assets(rw, N, :rw)
        tbw = cs_history_assets(bw, N, :bw)
        assert_cs_history_obs(trw, tbw, :rw, :bw)
        assert_cs_history_obs(trw, tvs, :rw, :vs)
        assert_cs_history_obs(tbw, tvs, :bw, :vs)
        assert_cs_block_rows(idx, ts, csr)
        return new{typeof(M), typeof(L), typeof(b), typeof(csr), typeof(Ms), typeof(vs),
                   typeof(esigma), typeof(edof), typeof(ediv), typeof(rw), typeof(bw),
                   typeof(nf), typeof(fam), typeof(fcb), typeof(lag), typeof(rf),
                   typeof(fx), typeof(fr), typeof(lambda), typeof(c), typeof(idx),
                   typeof(ts)}(M, L, b, csr, Ms, vs, esigma, edof, ediv, rw, bw, nf, fam,
                               fcb, lag, rf, fx, fr, lambda, c, idx, ts)
    end
end
function CrossSectionalFactorModel(; M::MatNum, L::Option{<:MatNum} = nothing, b::VecNum,
                                   csr::Option{<:CrossSectionalRegression} = nothing,
                                   Ms::Option{<:Arr3Num} = nothing,
                                   vs::Option{<:MatNum} = nothing,
                                   esigma::Option{<:VecNum_MatNum} = nothing,
                                   edof::Option{<:VecNum} = nothing,
                                   ediv::Option{<:VecNum} = nothing,
                                   rw::Option{<:MatNum} = nothing,
                                   bw::Option{<:MatNum} = nothing,
                                   nf::Option{<:VecStr} = nothing,
                                   fam::Option{<:VecStr} = nothing,
                                   fcb::Option{<:AbstractFactorFamilyBasis} = nothing,
                                   lag::Option{<:Integer} = nothing,
                                   rf::Option{<:AbstractReturnForecastResult} = nothing,
                                   fx::Option{<:MatNum} = nothing,
                                   fr::Option{<:MatNum} = nothing,
                                   lambda::Option{<:Number} = nothing,
                                   c::Option{<:Number} = nothing,
                                   idx::Option{<:VecInt} = nothing,
                                   ts::Option{<:VecDate} = nothing)::CrossSectionalFactorModel
    return CrossSectionalFactorModel(M, L, b, csr, Ms, vs, esigma, edof, ediv, rw, bw, nf,
                                     fam, fcb, lag, rf, fx, fr, lambda, c, idx, ts)
end
"""
    cross_sectional_factor_returns(csfm::CrossSectionalFactorModel) -> MatNum

Return the factor returns of a [`CrossSectionalFactorModel`](@ref) on the axis of its loadings `L`: the factors the fit estimated, then the observed factors.

The fit estimates `csr.f` on its own design axis, which is the reduced axis less the observed factors, and it observes the returns `fx`. The two side by side are the factor returns that the loadings `L` (or `M` when the model has no family re-basis) multiply. So the loadings of observation `t - lag` times row `t` of the answer, plus the idiosyncratic return of `t`, give the returns of observation `t` in the base currency.

# Algorithm

 1. Refuse a block that carries no fit.
 2. Return `csr.f` when `fx` is `nothing`, and `hcat(csr.f, fx)` otherwise.

# Arguments

  - `csfm`: A cross-sectional factor model block.

# Validation

  - `csfm.csr` is not `nothing`. Raises an [`IsNothingError`](@ref).

# Returns

  - `f::MatNum`: The factor returns, `observations × factors`, on the axis of `L`.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`AbstractObservedExposureEstimator`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function cross_sectional_factor_returns(csfm::CrossSectionalFactorModel)
    return cross_sectional_factor_returns(csfm.csr, csfm.fx)
end
function cross_sectional_factor_returns(::Nothing, ::Any)
    return throw(IsNothingError("csr cannot be nothing: the factor returns of a cross-sectional factor model are the returns its fit estimated, and this block carries no fit"))
end
function cross_sectional_factor_returns(csr::CrossSectionalRegression, ::Nothing)
    return csr.f
end
function cross_sectional_factor_returns(csr::CrossSectionalRegression, fx::MatNum)
    return hcat(csr.f, fx)
end
"""
    idiosyncratic_variances(rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::VecNum, rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::MatNum, rr::AbstractLoadingsRegressionResult)
    idiosyncratic_variances(esigma::Nothing, rr::Regression)
    idiosyncratic_variances(esigma::Nothing, rr::CrossSectionalFactorModel)

Read the idiosyncratic variance vector off a loadings block, whatever shape the block stores it in.

Both members of [`AbstractLoadingsRegressionResult`](@ref) carry `esigma` under one name, and both admit the two shapes: a vector of variances, or a full covariance. A consumer that needs the variances alone — an [`AbstractUncertaintySetEstimator`](@ref) that weights the cross-section by the inverse idiosyncratic variance is the first one — asks for them here rather than testing the shape at its own site.

The shape is the dispatch, as it is in [`assert_idiosyncratic_covariance`](@ref) and [`idiosyncratic_covariance_view`](@ref). The one-argument entry reads the field and forwards it beside the block, so the two refusals name the block they came from: a [`Regression`](@ref) is filled by the prior that lifts the factor moments, so its message names `rsd`, and a [`CrossSectionalFactorModel`](@ref) is filled by its own fit, so its message names the field.

There is no fallback. A block that carries no idiosyncratic covariance cannot answer, and an answer of ones or of zeros is a different weighting rather than a missing one.

# Arguments

  - `esigma`: Idiosyncratic covariance, a vector of variances, a square matrix, or `nothing`.
  - `rr`: The loadings block the field was read from, which the raise reports.

# Validation

  - `!isnothing(esigma)`, raising an `IsNothingError`.

# Returns

  - `esigma::VecNum`: The idiosyncratic variances, one per asset. A vector comes back unchanged, and a matrix comes back as its diagonal.

# Examples

```jldoctest
julia> re = Regression(; M = [1.0 2.0; 3.0 4.0], esigma = [0.1, 0.2]);

julia> PortfolioOptimisers.idiosyncratic_variances(re)
2-element Vector{Float64}:
 0.1
 0.2
```

# Related

  - [`AbstractLoadingsRegressionResult`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`assert_idiosyncratic_covariance`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
"""
function idiosyncratic_variances(rr::AbstractLoadingsRegressionResult)
    return idiosyncratic_variances(rr.esigma, rr)
end
function idiosyncratic_variances(esigma::VecNum, ::AbstractLoadingsRegressionResult)
    return esigma
end
function idiosyncratic_variances(esigma::MatNum, ::AbstractLoadingsRegressionResult)
    return LinearAlgebra.diag(esigma)
end
function idiosyncratic_variances(::Nothing, rr::Regression)
    return throw(IsNothingError("`esigma` is unset on this loadings block, so it carries no idiosyncratic variances to read. A time-series factor prior writes them only when it adds a residual block, and this block was built by a fit that added none.\nFit the prior with `rsd = true`, so that the lift measures the residual variances and writes them onto the block.\nGot\nrr => $(nameof(typeof(rr)))\nesigma => nothing"))
end
function idiosyncratic_variances(::Nothing, rr::CrossSectionalFactorModel)
    return throw(IsNothingError("`esigma` is unset on this loadings block, so it carries no idiosyncratic variances to read. A cross-sectional factor model fills the field from its own fit, and this block was built without it.\nBuild the block with `esigma` set, so that the idiosyncratic variances travel with the loadings.\nGot\nrr => $(nameof(typeof(rr)))\nesigma => nothing"))
end
# When `L` is unset (`Nothing` type parameter), `:L` falls back to the loadings matrix `M`;
# when `L` is a stored matrix the default field access already returns it, so only the
# `Nothing` specialisation needs a rule (see [`@forward_properties`](@ref)'s `swap`).
@forward_properties CrossSectionalFactorModel{<:Any, Nothing, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any} begin
    swap(L, M)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

State whether the model was fitted in a re-based factor family.

`L` and `fcb` are present together and absent together, and the pair is present exactly when a Factor Family was re-based before the fit. The answer is therefore the presence of `L`, read with `getfield` rather than through property access: the `swap(L, M)` rule of [`CrossSectionalFactorModel`](@ref) makes `csfm.L` return `csfm.M` when `L` is unset, so a property read would answer `true` for every model.

# Arguments

  - `csfm`: A cross-sectional factor model result.

# Returns

  - `val::Bool`: `true` when `L` is set, so the raw factor axis of `M` is a linear image of the re-based one and a factor covariance stated on it is singular; `false` when the fit ran in the raw basis.

# Examples

```jldoctest
julia> PortfolioOptimisers.has_family_rebasis(CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0],
                                                                        b = [0.1, 0.2]))
false

julia> fcb = FactorFamilyBasis(; fnm = [\"f\"], fi = [[1, 2]], di = [2],
                               ratios = reshape([1.0], 1, 1), K = 2);

julia> PortfolioOptimisers.has_family_rebasis(CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0],
                                                                        L = reshape([1.0, 2.0], 2,
                                                                                    1),
                                                                        b = [0.1, 0.2], fcb = fcb))
true
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`AbstractLoadingsRegressionResult`](@ref)
"""
function has_family_rebasis(csfm::CrossSectionalFactorModel)::Bool
    return !isnothing(getfield(csfm, :L))
end
"""
    port_opt_view(csfm::CrossSectionalFactorModel, i, args...)

Return a view of a [`CrossSectionalFactorModel`](@ref) result, selecting only the assets indexed by `i`.

# Algorithm

 1. Read `L` with `getfield`, never through property access. The `swap(L, M)` rule of [`CrossSectionalFactorModel`](@ref) makes `csfm.L` return `csfm.M` when `L` is unset, so a property read would materialise `L` as a copy of `M` and lose the unset-ness.
 2. Take a row view of `M`, of `L` when step 1 found a matrix, and an element view of `b`, giving the loadings and the factor-orthogonal expected return of the selected assets.
 3. View the nested fit with its own [`port_opt_view`](@ref) method, which cuts its residuals on the asset axis.
 4. Take a view of `Ms` on its second axis, and of `vs`, `rw` and `bw` on their second axis, giving the histories of the selected assets.
 5. View `esigma` with [`idiosyncratic_covariance_view`](@ref), which reads its shape, and view `edof` and `ediv` with [`nothing_scalar_array_view`](@ref).
 6. View the Return Forecast with its own [`port_opt_view`](@ref) method, which cuts `mu` and `hist` on the asset axis.
 7. Build a new [`CrossSectionalFactorModel`](@ref) from the views, passing `nf`, `fam`, `fcb`, `lag`, `fx`, `fr`, `lambda`, `c`, `idx` and `ts` through, which re-runs every guard of the constructor.

# Arguments

  - `csfm`: A cross-sectional factor model result.
  - `i`: Indices of the assets to select.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `csfm::CrossSectionalFactorModel`: A new result whose per-asset fields are restricted to the selected assets.

# Examples

```jldoctest
julia> csfm = CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0; 5.0 6.0], b = [0.1, 0.2, 0.3],
                                        esigma = [0.4, 0.5, 0.6]);

julia> PortfolioOptimisers.port_opt_view(csfm, [1, 3]).b
2-element view(::Vector{Float64}, [1, 3]) with eltype Float64:
 0.1
 0.3

julia> isnothing(getfield(PortfolioOptimisers.port_opt_view(csfm, [1, 3]), :L))
true
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`idiosyncratic_covariance_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(csfm::CrossSectionalFactorModel, i,
                       args...)::CrossSectionalFactorModel
    # `L` must be read with `getfield`: the `swap(L, M)` property rule above makes
    # `csfm.L` return `csfm.M` when `L` is unset, so `isnothing(csfm.L)` is never true and
    # a viewed result would materialise `L` as a copy of `M`, silently losing the
    # unset-ness the rule exists to express.
    L = getfield(csfm, :L)
    csr = csfm.csr
    Ms = csfm.Ms
    vs = csfm.vs
    rw = csfm.rw
    bw = csfm.bw
    rf = csfm.rf
    return CrossSectionalFactorModel(; M = view(csfm.M, i, :),
                                     L = isnothing(L) ? nothing : view(L, i, :),
                                     b = view(csfm.b, i),
                                     csr = if isnothing(csr)
                                         nothing
                                     else
                                         port_opt_view(csr, i, args...)
                                     end, Ms = isnothing(Ms) ? nothing : view(Ms, :, i, :),
                                     vs = isnothing(vs) ? nothing : view(vs, :, i),
                                     esigma = idiosyncratic_covariance_view(csfm.esigma, i),
                                     edof = nothing_scalar_array_view(csfm.edof, i),
                                     ediv = nothing_scalar_array_view(csfm.ediv, i),
                                     rw = isnothing(rw) ? nothing : view(rw, :, i),
                                     bw = isnothing(bw) ? nothing : view(bw, :, i),
                                     nf = csfm.nf, fam = csfm.fam, fcb = csfm.fcb,
                                     lag = csfm.lag, rf = if isnothing(rf)
                                         nothing
                                     else
                                         port_opt_view(rf, i, args...)
                                     end, fx = csfm.fx, fr = csfm.fr, lambda = csfm.lambda,
                                     c = csfm.c, idx = csfm.idx, ts = csfm.ts)
end
"""
    regression(csfm::CrossSectionalFactorModel, args...)

Return the cross-sectional factor model unchanged.

This method is a pass-through for [`CrossSectionalFactorModel`](@ref) results, as the method over [`Regression`](@ref) is for that result. A consumer that binds `RegE_Reg` takes either a result or an estimator, and calls `regression` on both.

# Arguments

  - `csfm`: A cross-sectional factor model result.
  - `args...`: Additional arguments (ignored).

# Returns

  - The input `csfm`, unchanged.

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`RegE_Reg`](@ref)
"""
function regression(csfm::CrossSectionalFactorModel, args...)
    return csfm
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the answer of a Factor Model Diagnostic together with the names and the family labels of its factor axis.

A diagnostic verb called on a [`CrossSectionalFactorModel`](@ref) returns it. The method of the same verb over bare histories returns the bare array, because it has no names to carry. [`port_opt_view`](@ref) selects factors of the answer by position, by name, or by family with a [`LabelGroup`](@ref). It cuts every factor dimension of `X`, and `nf` and `fam` with it. So a caller computes a diagnostic once, over every factor, and then selects the factors it reads.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorDiagnosticResult(;
        X::AbstractArray,
        nf::Option{<:VecStr} = nothing,
        fam::Option{<:VecStr} = nothing,
        dims::Tuple{Vararg{Integer}} = (ndims(X),)
    ) -> FactorDiagnosticResult

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(dims)`, and `dims` names distinct dimensions of `X`. Raises a `DomainError` otherwise.
  - Every dimension of `X` that `dims` names has one length, the factor count `K`. Raises a `DimensionMismatch` otherwise.
  - If provided, `length(nf) == K`, and `nf` repeats no name.
  - If provided, `length(fam) == K`.

# Examples

```jldoctest
julia> r = FactorDiagnosticResult(; X = [1.0 2.0 3.0; 4.0 5.0 6.0], nf = [\"value\", \"size\", \"tech\"],
                                  fam = [\"style\", \"style\", \"industry\"]);

julia> PortfolioOptimisers.port_opt_view(r, LabelGroup(\"style\")).X
2×2 view(::Matrix{Float64}, :, [1, 2]) with eltype Float64:
 1.0  2.0
 4.0  5.0

julia> PortfolioOptimisers.port_opt_view(r, [\"tech\"]).nf
1-element view(::Vector{String}, [3]) with eltype String:
 "tech"
```

# Related

  - [`port_opt_view`](@ref)
  - [`LabelGroup`](@ref)
  - [`cs_diagnostic_factor_names`](@ref)
  - [`FactorSummaryResult`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
@concrete struct FactorDiagnosticResult <: AbstractResult
    """
    The answer of the diagnostic: a vector with one entry per factor, a matrix `observations × factors`, or a matrix `factors × factors`.
    """
    X
    """
    $(field_dict[:fd_nf])
    """
    nf
    """
    $(field_dict[:fd_fam])
    """
    fam
    """
    The dimensions of `X` that run over the factors: `(ndims(X),)` for a series of the factors, and `(1, 2)` for a matrix of factor pairs.
    """
    dims
    function FactorDiagnosticResult(X::AbstractArray, nf::Option{<:VecStr},
                                    fam::Option{<:VecStr}, dims::Tuple{Vararg{Integer}})
        @argcheck(!isempty(dims) && allunique(dims) && all(d -> 1 <= d <= ndims(X), dims),
                  DomainError(dims,
                              "dims must name distinct dimensions of X, which has $(ndims(X))"))
        K = size(X, first(dims))
        @argcheck(all(d -> size(X, d) == K, dims),
                  DimensionMismatch("every factor dimension of X must have one length. Got size(X) => $(size(X)) and dims => $(dims)"))
        if !isnothing(nf)
            @argcheck(length(nf) == K,
                      DimensionMismatch("nf ($(length(nf))) must match the factor axis of X ($K)"))
            @argcheck(allunique(nf), ArgumentError("nf must not repeat a factor name"))
        end
        if !isnothing(fam)
            @argcheck(length(fam) == K,
                      DimensionMismatch("fam ($(length(fam))) must match the factor axis of X ($K)"))
        end
        return new{typeof(X), typeof(nf), typeof(fam), typeof(dims)}(X, nf, fam, dims)
    end
end
function FactorDiagnosticResult(; X::AbstractArray, nf::Option{<:VecStr} = nothing,
                                fam::Option{<:VecStr} = nothing,
                                dims::Tuple{Vararg{Integer}} = (ndims(X),))
    return FactorDiagnosticResult(X, nf, fam, dims)
end
"""
    factor_axis_positions(nf::Option{<:VecStr}, fam::Option{<:VecStr}, i) -> typeof(i)
    factor_axis_positions(nf::Option{<:VecStr}, fam::Option{<:VecStr}, i::Integer) -> UnitRange{Int}
    factor_axis_positions(nf::Option{<:VecStr}, fam::Nothing, i::LabelGroup) -> Union{}
    factor_axis_positions(nf::Option{<:VecStr}, fam::VecStr, i::LabelGroup) -> Vector{Int}

Turn the factor index of a view of a diagnostic answer into positions on its factor axis.

A position, a range and a `Colon` pass through. A vector of names goes through [`label_positions`](@ref), which refuses a name it cannot find. One integer becomes a range of one position, so the answer keeps its factor axis. A [`LabelGroup`](@ref) selects every factor whose family label is the group, in the order of the axis.

# Algorithm

The method that Julia selects is the algorithm.

 1. `i` is not an integer and not a [`LabelGroup`](@ref): return [`label_positions`](@ref) of `nf` and `i`.
 2. `i` is an integer: return `i:i`.
 3. `i` is a [`LabelGroup`](@ref) and `fam` is `nothing`: throw.
 4. `i` is a [`LabelGroup`](@ref): return the positions of the entries of `fam` that equal the group, and throw when there is none.

# Arguments

  - $(arg_dict[:fd_nf])
  - $(arg_dict[:fd_fam])
  - `i`: The factor index of the view: positions, a range, a `Colon`, a vector of names, or a [`LabelGroup`](@ref).

# Validation

  - `nf` is not `nothing` when `i` is a vector of names. Raises an `ArgumentError`.
  - Every name of `i` is in `nf`. Raises an `ArgumentError` that suggests the nearest name.
  - `fam` is not `nothing` when `i` is a [`LabelGroup`](@ref). Raises an `ArgumentError`.
  - At least one entry of `fam` is the group of `i`. Raises an `ArgumentError` that suggests the nearest family.

# Returns

  - `pos`: The positions of the selected factors, or `i` unchanged.

# Related

  - [`FactorDiagnosticResult`](@ref)
  - [`label_positions`](@ref)
  - [`LabelGroup`](@ref)
  - [`port_opt_view`](@ref)
"""
function factor_axis_positions(nf::Option{<:VecStr}, ::Option{<:VecStr}, i)
    return label_positions(nf, i, "factor")
end
function factor_axis_positions(::Option{<:VecStr}, ::Option{<:VecStr}, i::Integer)
    return i:i
end
function factor_axis_positions(::Option{<:VecStr}, ::Nothing, i::LabelGroup)
    return throw(ArgumentError("the view selects the factors of the family $(i.group), but the answer carries no family labels. Build the block with its family labels `fam`, or select the factors by position or by name."))
end
function factor_axis_positions(::Option{<:VecStr}, fam::VecStr, i::LabelGroup)
    pos = findall(==(i.group), fam)
    @argcheck(!isempty(pos),
              ArgumentError("the view selects the factors of the family $(i.group), and no factor of the answer carries that label. The families are $(join(unique(fam), ", "))." *
                            did_you_mean(i.group, unique(fam))))
    return pos
end
"""
    port_opt_view(r::FactorDiagnosticResult, i, args...)

Return a view of a [`FactorDiagnosticResult`](@ref) that keeps only the factors that `i` selects.

# Algorithm

 1. Turn `i` into positions on the factor axis with [`factor_axis_positions`](@ref).
 2. View `X` at those positions on each dimension that `dims` names, and in full on each other dimension.
 3. View `nf` and `fam` at those positions with [`nothing_scalar_array_view`](@ref).
 4. Build a new [`FactorDiagnosticResult`](@ref) from the views, which runs the guards of the constructor again.

# Arguments

  - `r`: A diagnostic answer.
  - `i`: The factor index: positions, a range, a `Colon`, a vector of names, or a [`LabelGroup`](@ref).
  - `args...`: Additional positional arguments (ignored).

# Validation

  - The rules of [`factor_axis_positions`](@ref).

# Returns

  - `r::FactorDiagnosticResult`: The answer on the selected factors.

# Examples

```jldoctest
julia> r = FactorDiagnosticResult(; X = [1.0 0.5; 0.5 1.0], nf = [\"value\", \"size\"], dims = (1, 2));

julia> PortfolioOptimisers.port_opt_view(r, 2).X
1×1 view(::Matrix{Float64}, 2:2, 2:2) with eltype Float64:
 1.0
```

# Related

  - [`FactorDiagnosticResult`](@ref)
  - [`factor_axis_positions`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(r::FactorDiagnosticResult, i, args...)::FactorDiagnosticResult
    p = factor_axis_positions(r.nf, r.fam, i)
    idx = ntuple(d -> d in r.dims ? p : Colon(), ndims(r.X))
    return FactorDiagnosticResult(view(r.X, idx...), nothing_scalar_array_view(r.nf, p),
                                  nothing_scalar_array_view(r.fam, p), r.dims)
end
"""
    factor_table_fields(r::AbstractResult, i) -> Tuple

Return every field of a per-factor table, cut to the factors that `i` selects.

A per-factor table is a Result whose fields are vectors with one entry per factor, scalars, or `nothing`, and which carries the names `nf` and the family labels `fam` of its factors. The answer is in the order of the fields, so the positional constructor of the table takes it.

# Algorithm

 1. Turn `i` into positions on the factor axis with [`factor_axis_positions`](@ref).
 2. View each field at those positions with [`nothing_scalar_array_view`](@ref), which keeps a scalar and `nothing` unchanged.

# Arguments

  - `r`: A per-factor table.
  - `i`: The factor index: positions, a range, a `Colon`, a vector of names, or a [`LabelGroup`](@ref).

# Validation

  - The rules of [`factor_axis_positions`](@ref).

# Returns

  - `fields::Tuple`: One entry per field of `r`, in the order of the fields.

# Related

  - [`FactorSummaryResult`](@ref)
  - [`ExposureICSummaryResult`](@ref)
  - [`factor_axis_positions`](@ref)
"""
function factor_table_fields(r::AbstractResult, i)
    p = factor_axis_positions(r.nf, r.fam, i)
    return ntuple(k -> nothing_scalar_array_view(getfield(r, k), p), fieldcount(typeof(r)))
end
"""
    cs_diagnostic_factor_names(csfm::CrossSectionalFactorModel)
    cs_diagnostic_factor_names(fcb::Option{<:AbstractFactorFamilyBasis}, nf::Nothing)
    cs_diagnostic_factor_names(fcb::Nothing, nf::VecStr)
    cs_diagnostic_factor_names(fcb::AbstractFactorFamilyBasis, nf::VecStr)

Return the factor names of the axis a cross-sectional regression diagnostic answers on.

A regression diagnostic that carries a factor axis answers on the design of the regression: the reduced axis when the block carries a family re-basis, less the observed factors, which are its last columns and which the regression did not estimate. So the names of the raw axis do not label it. The one-argument verb maps them, and the [`FactorDiagnosticResult`](@ref) of a regression diagnostic carries its answer. The two-argument verb maps the names onto the whole reduced axis, observed factors included, which is the axis of the loadings `L`. It maps family labels the same way. A block that names no factor answers `nothing`.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `fcb`: The family re-basis of the block, or `nothing`.
  - `nf`: The names, or the family labels, of the raw factor axis, or `nothing`.

# Returns

  - `nf::Option{<:Vector{String}}`: The names of the answer's factor axis, or `nothing` when the block names no factor.

# Examples

```jldoctest
julia> csfm = CrossSectionalFactorModel(; M = [1.0 2.0; 3.0 4.0], b = [0.1, 0.2],
                                        esigma = [0.4, 0.5], nf = [\"value\", \"size\"]);

julia> PortfolioOptimisers.cs_diagnostic_factor_names(csfm)
2-element Vector{String}:
 "value"
 "size"
```

# Related

  - [`CrossSectionalFactorModel`](@ref)
  - [`FactorDiagnosticResult`](@ref)
  - [`reduce_factor_names`](@ref)
  - [`cs_design_labels`](@ref)
  - [`cs_regression_t_stats`](@ref)
  - [`exposure_vif`](@ref)
"""
function cs_diagnostic_factor_names(csfm::CrossSectionalFactorModel)
    return cs_design_labels(csfm, csfm.nf)
end
function cs_diagnostic_factor_names(::Option{<:AbstractFactorFamilyBasis},
                                    ::Nothing)::Nothing
    return nothing
end
function cs_diagnostic_factor_names(::Nothing, nf::VecStr)::Vector{String}
    return String[String(n) for n in nf]
end
function cs_diagnostic_factor_names(fcb::AbstractFactorFamilyBasis,
                                    nf::VecStr)::Vector{String}
    return reduce_factor_names(fcb, nf)
end
"""
    cs_design_labels(csfm::CrossSectionalFactorModel, labels::Option{<:VecStr})

Map labels of the raw factor axis onto the design axis of the cross-sectional regression.

The design axis is the reduced axis less the observed factors, which [`cs_diagnostic_factor_names`](@ref) states. The labels are the names or the family labels of the raw axis.

# Algorithm

 1. Map the labels onto the whole reduced axis with the two-argument [`cs_diagnostic_factor_names`](@ref).
 2. Drop the last `size(csfm.fx, 2)` entries when the block carries observed factor returns.

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `labels`: The names or the family labels of the raw factor axis, or `nothing`.

# Returns

  - `labels::Option{<:Vector{String}}`: The labels of the design axis, or `nothing` when `labels` is `nothing`.

# Related

  - [`cs_diagnostic_factor_names`](@ref)
  - [`cs_design_result`](@ref)
"""
function cs_design_labels(csfm::CrossSectionalFactorModel, labels::Option{<:VecStr})
    lb = cs_diagnostic_factor_names(csfm.fcb, labels)
    return isnothing(lb) || isnothing(csfm.fx) ? lb : lb[1:(end - size(csfm.fx, 2))]
end
"""
    cs_design_result(csfm::CrossSectionalFactorModel, X::AbstractArray)

Wrap the answer of a regression diagnostic of a block in a [`FactorDiagnosticResult`](@ref) on the design axis.

The last dimension of `X` is the factor axis. The names and the family labels are those of the design axis, from [`cs_design_labels`](@ref).

# Arguments

  - `csfm`: A cross-sectional factor model block.
  - `X`: The answer of the diagnostic, with the factors on its last dimension.

# Returns

  - `r::FactorDiagnosticResult`: The answer with the labels of its factor axis.

# Related

  - [`FactorDiagnosticResult`](@ref)
  - [`cs_design_labels`](@ref)
  - [`exposure_vif`](@ref)
  - [`cs_regression_t_stats`](@ref)
"""
function cs_design_result(csfm::CrossSectionalFactorModel, X::AbstractArray)
    return FactorDiagnosticResult(X, cs_design_labels(csfm, csfm.nf),
                                  cs_design_labels(csfm, csfm.fam), (ndims(X),))
end

export CrossSectionalFactorModel, cross_sectional_factor_returns, FactorDiagnosticResult
public cs_diagnostic_factor_names
