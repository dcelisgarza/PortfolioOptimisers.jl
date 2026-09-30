"""
$(DocStringExtensions.TYPEDSIGNATURES)

Shared preamble for windowed moment estimators (matrix input).

Whenever a window is given — an `Int`, which resolves to a range, or an explicit index
vector — `iv`, and the `active_mask` and `estimation_mask` keywords, are subset to the same
rows, or columns when `dims = 2`, so they stay aligned with the windowed returns. Only
`window = nothing`, which resolves to a `Colon`, leaves them unchanged.

# Algorithm

 1. Resolve `window` with [`get_window`](@ref), giving `win`. `nothing` resolves to a `Colon`, an `Int` to the range of the last `window` observations, and an index vector passes through.
 2. Apply `win` to `X` and rebind the observation weights to it with [`moment_window_and_weights`](@ref), giving the windowed `X` and `w_new`.
 3. Build `inner`, a copy of `est` that carries `w_new`, with [`factory`](@ref).
 4. Cut `iv` to `win` with [`windowed_rows`](@ref), and cut the row-aligned keywords with [`windowed_keywords`](@ref), giving `kw`.
 5. Return `inner`, the windowed `X`, `iv` and `kw`.

# Arguments

  - `est`: Wrapped moment estimator to be cloned with updated weights.
  - `w`: Optional observation weights applied after windowing.
  - `window`: Window specification — `nothing` (full data), an `Int` (last `window`
    observations), or a `VecInt` of explicit row/column indices.
  - `X`: Data matrix of asset returns.
  - `iv`: Optional instrument variable matrix; subsetted to the window when `window` is a
    `VecInt`.
  - `dims`: Observation dimension — 1 for rows (default), 2 for columns. Checked by
    [`assert_dims`](@ref), so every generated windowed method rejects an out-of-range `dims`
    instead of silently resolving a one-observation window.
  - `kwargs...`: Passed through to [`moment_window_and_weights`](@ref), and returned as `kw` with the masks cut.

# Validation

  - $(val_dict[:dims])

# Returns

  - `(inner, X, iv, kw)`: Weight-updated estimator, windowed returns matrix, (possibly
    subsetted) instrument variable matrix, and the keywords with the masks cut to the window.

# Related

  - [`get_window`](@ref)
  - [`moment_window_and_weights`](@ref)
  - [`factory`](@ref)
  - [`windowed_rows`](@ref)
  - [`windowed_keywords`](@ref)
  - [`@windowed_estimator`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
function windowed_preamble(est, w::Option{<:ObsWeights}, window::Option{<:Int_VecInt},
                           X::MatNum; iv::Option{<:MatNum} = nothing, dims::Int = 1,
                           kwargs...)
    assert_dims(dims)
    win = get_window(window, X, dims)
    X, w_new = moment_window_and_weights(X, w, win; dims = dims, kwargs...)
    inner = factory(est, w_new)
    return inner, X, windowed_rows(iv, win, dims), windowed_keywords(win, dims; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Shared preamble for windowed moment estimators (vector input).

This method takes no `dims`, because a vector carries one axis, and no `iv`, because no
vector generic of the family declares one.

# Algorithm

 1. Resolve `window` with [`get_window`](@ref), giving `win`. `nothing` resolves to a `Colon`, an `Int` to the range of the last `window` observations, and an index vector passes through.
 2. Apply `win` to `X` and rebind the observation weights to it with [`moment_window_and_weights`](@ref), giving the windowed `X` and `w_new`.
 3. Build `inner`, a copy of `est` that carries `w_new`, with [`factory`](@ref).
 4. Return `inner` and the windowed `X`.

# Arguments

  - `est`: Wrapped moment estimator to be cloned with updated weights.
  - `w`: Optional observation weights applied after windowing.
  - `window`: Window specification — `nothing` (full data), an `Int` (last `window`
    observations), or a `VecInt` of explicit indices.
  - `X`: Data vector of returns.

# Returns

  - `(inner, X)`: Weight-updated estimator and windowed returns vector.

# Related

  - [`get_window`](@ref)
  - [`moment_window_and_weights`](@ref)
  - [`factory`](@ref)
  - [`@windowed_estimator`](@ref)
  - [`WindowedVariance`](@ref)
"""
function windowed_preamble(est, w::Option{<:ObsWeights}, window::Option{<:Int_VecInt},
                           X::VecNum)
    win = get_window(window, X)
    X, w_new = moment_window_and_weights(X, w, win)
    inner = factory(est, w_new)
    return inner, X
end
"""
    windowed_rows(x::Nothing, win, dims::Int) -> nothing
    windowed_rows(x::AbstractMatrix, win::Colon, dims::Int) -> AbstractMatrix
    windowed_rows(x::AbstractMatrix, win::VecInt, dims::Int) -> SubArray

Cut a matrix that pairs each observation of the returns with one row, or with one column when `dims = 2`, to the observations of a resolved window.

The instrument variables and the two universe masks carry one entry per observation, so a window that cuts the returns must cut them too, or they no longer line up.

# Algorithm

The method that Julia selects is the algorithm.

 1. `x` is `nothing`: return `nothing`.
 2. `win` is a `Colon`: return `x` unchanged, so the full-data case never copies it.
 3. `win` is an index vector: return a view of the rows of `x` in `win`, or of its columns when `dims == 2`.

# Arguments

  - `x`: The matrix to cut, or `nothing`.
  - `win`: The resolved window, from [`get_window`](@ref).
  - $(arg_dict[:dims])

# Returns

  - `x′`: `x` over the observations of the window, or `nothing`.

# Related

  - [`windowed_preamble`](@ref)
  - [`windowed_keywords`](@ref)
  - [`get_window`](@ref)
"""
function windowed_rows(::Nothing, ::Any, ::Int)
    return nothing
end
function windowed_rows(x::AbstractMatrix, ::Colon, ::Int)
    return x
end
function windowed_rows(x::AbstractMatrix, win::VecInt, dims::Int)
    return isone(dims) ? view(x, win, :) : view(x, :, win)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Cut the row-aligned keywords of a windowed moment call to the observations of a resolved window.

A mask-aware estimator, such as [`ExpWeightedCovariance`](@ref), takes an `active_mask` keyword, and a regime-adjusted one also takes an `estimation_mask` keyword. Each has the size of the returns. The wrapper cuts the returns to the window, so it cuts these two keywords with them. Every other keyword passes through unchanged, and a keyword the caller did not give is not added.

# Algorithm

 1. Read the keywords as a `NamedTuple`, giving `kw`.
 2. Cut the entries named `active_mask` and `estimation_mask` with [`windowed_rows`](@ref), and keep every other entry.
 3. Return the entries under the names of `kw`.

# Arguments

  - `win`: The resolved window, from [`get_window`](@ref).
  - $(arg_dict[:dims])
  - `kwargs...`: The keywords of the windowed call.

# Returns

  - `kw::NamedTuple`: The keywords, with the two masks cut to the window.

# Related

  - [`windowed_preamble`](@ref)
  - [`windowed_rows`](@ref)
"""
function windowed_keywords(win, dims::Int; kwargs...)
    kw = values(kwargs)
    cut = map(keys(kw), values(kw)) do k, v
        return k in (:active_mask, :estimation_mask) ? windowed_rows(v, win, dims) : v
    end
    return NamedTuple{keys(kw)}(cut)
end
"""
    windowed_panel(pnl::Option{<:AssetPanel}, win::Colon) -> Option{<:AssetPanel}
    windowed_panel(pnl::Option{<:AssetPanel}, win::VecInt) -> Option{<:AssetPanel}

Cut an Asset Panel to the observations of a resolved window.

The panel method of a windowed estimator cuts the returns to the window, and gives the inner estimator the panel over the same observations. So a mask-aware inner estimator reads the active mask of the rows it fits, and a plain inner estimator reduces to the Coverage Universe of the window, not of the whole sample. The panel always holds its observations on the rows, whatever the orientation of the returns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `win` is a `Colon`: return `pnl` unchanged.
 2. `win` is an index vector: return the view of `pnl` over the rows in `win` and every asset, with [`asset_panel_view`](@ref). A `nothing` panel stays `nothing`.

# Arguments

  - $(arg_dict[:pnl_moment])
  - `win`: The resolved window, from [`get_window`](@ref).

# Returns

  - `pnl′::Option{<:AssetPanel}`: The Asset Panel over the observations of the window, or `nothing`.

# Related

  - [`asset_panel_view`](@ref)
  - [`windowed_preamble`](@ref)
  - [`AssetPanel`](@ref)
"""
function windowed_panel(pnl::Option{<:AssetPanel}, ::Colon)
    return pnl
end
function windowed_panel(pnl::Option{<:AssetPanel}, win::VecInt)
    return asset_panel_view(pnl, win, :, nothing)
end
"""
    windowed_series_rows(window::Nothing, t::Integer) -> UnitRange
    windowed_series_rows(window::Integer, t::Integer) -> UnitRange
    windowed_series_rows(window::VecInt, t::Integer) -> VecInt

The observations that a windowed estimator keeps when it is fitted on observations `1` to `t`.

Row `t` of a variance series is a fit on the first `t` observations. A windowed estimator keeps the observations of its window among them, so row `t` of its series reads nothing after `t`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `window` is `nothing`: keep `1:t`.
 2. `window` is an `Int`: keep the last `window` observations of `1:t`, that is `max(1, t - window + 1):t`.
 3. `window` is an index vector: keep its entries that are at most `t`, in their order. The result is empty before the first entry.

# Arguments

  - `window`: The window of the estimator.
  - `t`: The last observation of the fit.

# Returns

  - `rows`: The observations that the fit keeps.

# Related

  - [`windowed_variance_series`](@ref)
  - [`get_window`](@ref)
"""
function windowed_series_rows(::Nothing, t::Integer)
    return 1:t
end
function windowed_series_rows(window::Integer, t::Integer)
    return max(1, t - window + 1):t
end
function windowed_series_rows(window::VecInt, t::Integer)
    return filter(<=(t), window)
end
"""
    windowed_series_row(est, w, rows, X, pnl::Nothing; dims::Int = 1, kwargs...)
    windowed_series_row(est, w, rows, X, pnl::AssetPanel; dims::Int = 1, kwargs...)

Fit the variance of a windowed estimator on the observations `rows`, as one row of its variance series.

# Algorithm

 1. Cut `X`, the observation weights and the row-aligned keywords to `rows` with [`windowed_preamble`](@ref), giving `inner`, `Xt` and `kw`.
 2. `pnl` is `nothing`: call `Statistics.var(inner, Xt; dims, kw...)`. `pnl` is an Asset Panel: call its panel method on the panel over `rows`, from [`windowed_panel`](@ref).
 3. Return the result as a vector.

# Arguments

  - `est`: The inner estimator of the windowed estimator.
  - `w`: Its observation weights, one per observation of `X`, or `nothing`.
  - `rows`: The observations of the fit, from [`windowed_series_rows`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the variance estimator. The two universe masks are cut to `rows`.

# Returns

  - `v::VecNum`: The variance of each asset.

# Related

  - [`windowed_variance_series`](@ref)
  - [`windowed_preamble`](@ref)
"""
function windowed_series_row(est, w::Option{<:ObsWeights}, rows::VecInt, X::MatNum,
                             ::Nothing; dims::Int = 1, kwargs...)
    inner, Xt, _, kw = windowed_preamble(est, w, rows, X; dims = dims, kwargs...)
    return vec(Statistics.var(inner, Xt; dims = dims, kw...))
end
function windowed_series_row(est, w::Option{<:ObsWeights}, rows::VecInt, X::MatNum,
                             pnl::AssetPanel; dims::Int = 1, kwargs...)
    inner, Xt, _, kw = windowed_preamble(est, w, rows, X; dims = dims, kwargs...)
    return vec(Statistics.var(inner, Xt, windowed_panel(pnl, rows); dims = dims, kw...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the point-in-time variance series of a windowed estimator.

Row `t` of a variance series is the estimate that the estimator gives when it is fitted on the observations `1` to `t`. A windowed estimator keeps the last observations of its window, so row `t` is a fit on the window that ends at `t`: the window rolls with `t`. Row `t` thus reads nothing after `t`, and the last row equals the batch fit of the estimator on the whole sample.

Before the window is full, row `t` keeps every observation from `1` to `t`. An index-vector window keeps its entries up to `t`, and a row before its first entry is `NaN`.

# Mathematical definition

```math
\\begin{align}
v_{t,\\,i} &= \\left[ \\hat{\\boldsymbol{\\sigma}}^{2}\\left( \\mathbf{X}_{\\mathcal{W}_{t},\\,\\cdot} \\right) \\right]_{i}\\,, \\\\
\\mathcal{W}_{t} &= \\left\\{ \\max(1,\\, t - n + 1),\\, \\ldots,\\, t \\right\\}\\,.
\\end{align}
```

Where:

  - ``v_{t,\\,i}``: Entry of the series for asset ``i`` at observation ``t``.
  - ``\\mathcal{W}_{t}``: Observations that the window keeps at ``t``, for a window of ``n`` observations.
  - ``\\mathbf{X}_{\\mathcal{W}_{t},\\,\\cdot}``: Rows of the returns matrix in ``\\mathcal{W}_{t}``, with the active mask of the same rows.
  - ``\\hat{\\boldsymbol{\\sigma}}^{2}(\\cdot)``: Variance vector that the inner estimator fits on a block, indexed by asset.

# Algorithm

 1. Fit row `T`, the last observation, with [`windowed_series_row`](@ref) on the rows from [`windowed_series_rows`](@ref), and allocate the frame from its element type.
 2. For each earlier observation `t`, fit row `t` the same way. A row that keeps no observation is `NaN`.
 3. Return the series, transposed when `dims == 2`.

# Arguments

  - `est`: The inner estimator of the windowed estimator.
  - `w`: Its observation weights, one per observation of `X`, or `nothing`.
  - `window`: The window of the estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the variance estimator. The two universe masks are cut to the rows of each fit.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`windowed_series_rows`](@ref)
  - [`windowed_series_row`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function windowed_variance_series(est, w::Option{<:ObsWeights},
                                  window::Option{<:Int_VecInt}, X::MatNum,
                                  pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    assert_dims(dims)
    T = size(X, dims)
    vT = windowed_series_row(est, w, windowed_series_rows(window, T), X, pnl; dims = dims,
                             kwargs...)
    val = Matrix{eltype(vT)}(undef, T, length(vT))
    val[T, :] = vT
    for t in 1:(T - 1)
        rows = windowed_series_rows(window, t)
        if isempty(rows)
            val[t, :] .= NaN
        else
            val[t, :] = windowed_series_row(est, w, rows, X, pnl; dims = dims, kwargs...)
        end
    end
    return isone(dims) ? val : permutedims(val)
end
