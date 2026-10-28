"""
$(DocStringExtensions.TYPEDSIGNATURES)

Cut the returns matrix, the observation weights and the per-observation arguments of a windowed moment call to its window.

A window that is an `Int` or an index vector cuts `iv`, `active_mask` and `estimation_mask` to the rows that it keeps in `X`, or to the columns when `dims = 2`. These arguments then align with the windowed returns. A `window` of `nothing` cuts nothing.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. Resolve `window` to `win` with [`get_window`](@ref). `nothing` resolves to a `Colon`, an `Int` to the range of the last `window` observations, and an index vector to itself.
 3. Cut `X` to `win`, and resolve the observation weights `w_new` over the cut `X`, with [`moment_window_and_weights`](@ref).
 4. Make `inner`, a copy of `est` that holds `w_new`, with [`factory`](@ref).
 5. Cut `iv` to `win` with [`windowed_rows`](@ref).
 6. Cut the two universe masks in `kwargs` to `win` with [`windowed_keywords`](@ref). The result is `kw`.
 7. Return `inner`, the cut `X`, the cut `iv` and `kw`.

# Arguments

  - `est`: Inner moment estimator. The function copies it with the windowed observation weights.
  - $(arg_dict[:oow]) The function cuts it to the window before it resolves it.
  - $(arg_dict[:window]) `nothing` selects every observation.
  - $(arg_dict[:X])
  - $(arg_dict[:oiv]) The function cuts it to the window.
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords for [`moment_window_and_weights`](@ref). The function also returns them as `kw`, with the two universe masks cut to the window.

# Validation

  - $(val_dict[:dims])

# Returns

  - `(inner, X, iv, kw)`: The copy of `est` with the windowed weights, the windowed returns, the windowed `iv` or `nothing`, and the keywords with the two masks cut to the window.

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

Cut the returns vector and the observation weights of a windowed moment call to its window.

This method takes no `dims`, because a vector has one axis. It takes no `iv`, because no vector method of a windowed estimator takes one.

# Algorithm

 1. Resolve `window` to `win` with [`get_window`](@ref), as the matrix method does.
 2. Cut `X` to `win`, and resolve the observation weights `w_new` over the cut `X`, with [`moment_window_and_weights`](@ref).
 3. Make `inner`, a copy of `est` that holds `w_new`, with [`factory`](@ref).
 4. Return `inner` and the cut `X`.

# Arguments

  - `est`: Inner moment estimator. The function copies it with the windowed observation weights.
  - $(arg_dict[:oow]) The function cuts it to the window before it resolves it.
  - $(arg_dict[:window]) `nothing` selects every observation.
  - $(arg_dict[:Xv])

# Returns

  - `(inner, X)`: The copy of `est` with the windowed weights, and the windowed returns.

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

Cut a matrix with one row for each observation, or one column when `dims = 2`, to the observations of a resolved window.

The implied volatility matrix `iv` and the two universe masks have one entry for each observation. A window that cuts the returns must cut them too. If it does not, their rows do not align with the rows of the returns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `x` is `nothing`: return `nothing`.
 2. `win` is a `Colon`: return `x` unchanged.
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

Cut the two universe mask keywords of a windowed moment call to the observations of a resolved window.

A mask-aware estimator, such as [`ExpWeightedCovariance`](@ref), takes an `active_mask` keyword. A regime-adjusted estimator also takes an `estimation_mask` keyword. Each mask has the size of the returns. The windowed estimator cuts the returns to the window, so the function cuts the two masks with them. It passes every other keyword unchanged, and it adds no keyword that the caller did not give.

# Algorithm

 1. Read the keywords as the `NamedTuple` `kw`.
 2. Cut the entries `active_mask` and `estimation_mask` with [`windowed_rows`](@ref). Keep every other entry.
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

The panel method of a windowed estimator cuts the returns to the window. It gives the inner estimator the panel over the same observations. So a mask-aware inner estimator reads the active mask of the rows that it fits. A plain inner estimator reduces to the Coverage Universe of the window, not of the whole sample. The panel holds its observations on the rows for each orientation of the returns.

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

Return the observations that a windowed estimator keeps in a fit on the observations `1` to `t`.

Row `t` of a variance series is a fit on the first `t` observations. A windowed estimator keeps the observations of its window among them, so row `t` of its series reads no observation after `t`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `window` is `nothing`: keep `1:t`.
 2. `window` is an `Int`: keep the last `window` observations of `1:t`, which are `max(1, t - window + 1):t`.
 3. `window` is an index vector: keep its entries that are at most `t`, in their order. The result is empty when `t` is less than the first entry.

# Arguments

  - $(arg_dict[:window]) `nothing` selects every observation.
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

Fit the variance of the inner estimator of a windowed estimator on the observations `rows`, as one row of its variance series.

# Algorithm

 1. Cut `X`, the observation weights and the two universe masks to `rows` with [`windowed_preamble`](@ref). The results are `inner`, `Xt` and `kw`.
 2. `pnl` is `nothing`: call `Statistics.var(inner, Xt; dims, kw...)`. `pnl` is an Asset Panel: call the panel method of `Statistics.var` with the panel over `rows`, from [`windowed_panel`](@ref).
 3. Return the result as a vector.

# Arguments

  - `est`: The inner estimator of the windowed estimator.
  - $(arg_dict[:oow])
  - `rows`: The observations of the fit, from [`windowed_series_rows`](@ref).
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords for the variance estimator. The function cuts the two universe masks to `rows`.

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

Row `t` of a variance series is the estimate of the estimator on the observations `1` to `t`. A windowed estimator keeps the last observations of its window, so row `t` is a fit on the window that ends at `t`. The window moves with `t`. Row `t` reads no observation after `t`, and the last row equals the batch fit of the estimator on the whole sample.

# Mathematical definition

```math
\\begin{align}
\\mathbf{V}_{tj} &= \\hat{\\sigma}^2_j\\left(\\mathbf{X}_{\\mathcal{W}_{t}}\\right)\\,, \\\\
\\mathcal{W}_{t} &= \\begin{cases}
\\left\\{1,\\, \\ldots,\\, t\\right\\} & \\text{without a window}\\,, \\\\
\\left\\{\\max(1,\\, t - n + 1),\\, \\ldots,\\, t\\right\\} & \\text{for a window of } n \\text{ observations}\\,, \\\\
\\left\\{s \\in \\mathcal{S} : s \\leq t\\right\\} & \\text{for a window of the indices } \\mathcal{S}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:V_tj_series])
  - $(math_dict[:sigma2_hat_j]) Here the inner estimator estimates it.
  - ``\\mathbf{X}_{\\mathcal{W}_{t}}``: Rows of the returns matrix in ``\\mathcal{W}_{t}``, with the active mask of the same rows.
  - ``\\mathcal{W}_{t}``: Observations that the window keeps at ``t``.
  - ``n``: Number of observations in the window.
  - ``\\mathcal{S}``: Indices of the observations that an index window selects.

``\\mathbf{V}_{tj}`` is `NaN` when ``\\mathcal{W}_{t}`` is empty. Before an integer window is full, ``\\mathcal{W}_{t}`` holds every observation from ``1`` to ``t``.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. Fit row `T`, the last observation, with [`windowed_series_row`](@ref) on the rows from [`windowed_series_rows`](@ref). The result is `vT`.
 3. If row `1` keeps no observation, refuse an `Integer` or a `Rational` element type of `vT`. The rows that a window keeps grow with `t`, so a row holds `NaN` only in this case.
 4. Allocate the series `val` with the element type of `vT`, and write `vT` into row `T`.
 5. For each earlier observation `t`, fit row `t` the same way. A row that keeps no observation is `NaN`.
 6. Return `val`, transposed when `dims == 2`.

# Arguments

  - `est`: The inner estimator of the windowed estimator.
  - $(arg_dict[:oow])
  - $(arg_dict[:window]) `nothing` selects every observation.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords for the variance estimator. The function cuts the two universe masks to the rows of each fit.

# Validation

  - $(val_dict[:dims])
  - $(val_dict[:nan_series_window])

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
    # The rows that a window keeps grow with `t`, so a row with no observation exists when
    # the first row has none.
    @argcheck(!isempty(windowed_series_rows(window, 1)) ||
              !(eltype(vT) <: Union{Integer, Rational}),
              ArgumentError("the window keeps no observation of the first rows of the variance series, and those rows hold NaN. The element type $(eltype(vT)) of the variance cannot hold NaN. Convert the sample to a floating-point type, or start the index window at the first observation."))
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
"""
    const SeedWindowed = Union{WindowedExpectedReturns{<:Any, <:Any, <:Any, <:SeedWindow},
                               WindowedCovariance{<:Any, <:Any, <:Any, <:SeedWindow},
                               WindowedVariance{<:Any, <:Any, <:Any, <:SeedWindow},
                               WindowedCoskewness{<:Any, <:Any, <:Any, <:SeedWindow},
                               WindowedCokurtosis{<:Any, <:Any, <:Any, <:SeedWindow}}

Groups the windowed estimators under a [`SeedWindow`](@ref), which are the windowed estimators that fold.

The fourth type parameter of a windowed estimator is the type of its `rule` field, so the alias selects the rule by dispatch. A windowed estimator under a [`RollingWindow`](@ref) is not in the group, and it does not fold.

# Related

  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
  - [`SeedWindow`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
const SeedWindowed = Union{WindowedExpectedReturns{<:Any, <:Any, <:Any, <:SeedWindow},
                           WindowedCovariance{<:Any, <:Any, <:Any, <:SeedWindow},
                           WindowedVariance{<:Any, <:Any, <:Any, <:SeedWindow},
                           WindowedCoskewness{<:Any, <:Any, <:Any, <:SeedWindow},
                           WindowedCokurtosis{<:Any, <:Any, <:Any, <:SeedWindow}}
"""
    partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)
    partial_fit!(est::SeedWindowed, x::VecNum; kwargs...)

Fold a block of observations into a windowed estimator under a [`SeedWindow`](@ref).

The first fold cuts the block to the window, as a batch fit does, and folds the rows that remain into the inner estimator. It returns the estimator with `window = nothing`. Each later fold then passes every row to the inner estimator. The windowed estimator holds no state of its own. Its inner estimator holds the state of the fold, and the window of `nothing` records that the first fold happened.

The vector method folds one observation as a matrix of one row. It also changes each `Bool` vector keyword, such as a mask, to a matrix of one row. A first fold of one observation keeps it when the window keeps the last row.

# Algorithm

 1. Read `inner`, the inner estimator. The declaration of each windowed type sets it as the first field.
 2. Resolve the window against the block to `win` with [`get_window`](@ref). After the first fold, the window is `nothing` and `win` is a `Colon`.
 3. Cut the block with [`windowed_rows`](@ref). Cut the `active_mask` and `estimation_mask` keywords with [`windowed_keywords`](@ref).
 4. Fold the cut block into `inner` with [`partial_fit!`](@ref).
 5. Return the estimator with the folded `inner` and `window = nothing`.

# Arguments

  - `est`: Windowed estimator under a seed window.
  - $(arg_dict[:X])
  - `x::VecNum`: One observation, with one entry for each asset.
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords for the fold of the inner estimator. The function cuts the two universe masks to the window.

# Validation

  - `est.w` is `nothing`, because a vector of observation weights gives a new weight to each past observation when a new observation arrives. The function throws an `ArgumentError` otherwise.
  - The inner estimator folds, by [`supports_partial_fit`](@ref). The function throws an `ArgumentError` otherwise.
  - $(val_dict[:dims])

# Returns

  - `est`: The windowed estimator, with its inner estimator folded and its window spent.

# Related

  - [`SeedWindow`](@ref)
  - [`SeedWindowed`](@ref)
  - [`windowed_preamble`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
function partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)
    # The declaration of every windowed type sets the inner estimator as the first field.
    inner = getfield(est, 1)
    @argcheck(isnothing(est.w),
              ArgumentError("`$(typeof(est).name.name)` under a `SeedWindow` holds observation weights, and the fold cannot honour them: a weight vector reweights every past observation when a new one arrives. Set `w = nothing`, or fit the estimator in batch."))
    @argcheck(supports_partial_fit(inner),
              ArgumentError("`$(typeof(est).name.name)` under a `SeedWindow` folds its inner estimator, and `$(typeof(inner).name.name)` does not fold. Use an inner estimator that folds, such as `ExpWeightedCovariance`, or set the rule to `RollingWindow()`, so that a prior that carries the observations refits the estimator over its carried rows."))
    assert_dims(dims)
    win = get_window(est.window, X, dims)
    inner = partial_fit!(inner, windowed_rows(X, win, dims); dims = dims,
                         windowed_keywords(win, dims; kwargs...)...)
    return typeof(est).name.wrapper(inner, est.w, nothing, est.rule, est.cache)
end
function partial_fit!(est::SeedWindowed, x::VecNum; kwargs...)
    kw = map(v -> isa(v, AbstractVector{Bool}) ? permutedims(v) : v, values(kwargs))
    return partial_fit!(est, permutedims(x)::MatNum; dims = 1, kw...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`, because a windowed estimator under a [`SeedWindow`](@ref) folds.

In a Carry Fold, a prior such as [`EmpiricalPrior`](@ref) folds each member that folds. It fits each other member again over the rows that it carries. A fit again over those rows cuts the window again, which is the rule of a [`RollingWindow`](@ref). So a seed window must fold, and the answer is `true` for each inner estimator. If the inner estimator does not fold, [`partial_fit!`](@ref) throws an error that names it, and the prior does not fit the estimator again.

# Arguments

  - `est`: Windowed estimator under a seed window.

# Returns

  - `true`.

# Related

  - [`supports_partial_fit`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function supports_partial_fit(::SeedWindowed)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`SeedWindowed`](@ref) method of [`observation_count`](@ref).

A windowed estimator under a [`SeedWindow`](@ref) keeps no state of its own, and its inner estimator holds the state of the fold. So the method reads the count of the inner estimator. That count is the number of rows that the inner estimator folded: the rows of the first block that the window kept, and every later row.

# Arguments

  - `est`: Windowed estimator under a seed window.

# Validation

  - The inner estimator holds a partial-fit state with a count, as [`observation_count`](@ref) states. An `ArgumentError` is thrown otherwise.

# Returns

  - `n::Integer`: The number of observations that the inner estimator folded.

# Related

  - [`observation_count`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function observation_count(est::SeedWindowed)
    # The declaration of every windowed type sets the inner estimator as the first field.
    return observation_count(getfield(est, 1))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `"rule"`, the path to the seed window inside a windowed estimator under a [`SeedWindow`](@ref).

The walk of [`seed_window_path`](@ref) over an estimator tree stops at this method. The walk puts the names of the fields above the estimator before `"rule"`.

# Arguments

  - `est`: Windowed estimator under a seed window.

# Returns

  - `path::String`: `"rule"`.

# Related

  - [`seed_window_path`](@ref)
  - [`Online`](@ref)
"""
function seed_window_path(::SeedWindowed)
    return "rule"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the expected returns from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there.

# Arguments

  - `me`: Windowed expected returns estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:mu])

# Related

  - [`WindowedExpectedReturns`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::WindowedExpectedReturns{<:Any, <:Any, <:Any, <:SeedWindow};
                         kwargs...)
    return Statistics.mean(me.me; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the covariance from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there.

# Arguments

  - `ce`: Windowed covariance estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`WindowedCovariance`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::WindowedCovariance{<:Any, <:Any, <:Any, <:SeedWindow};
                        kwargs...)
    return Statistics.cov(ce.ce; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the correlation from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there. The generic `cor` with no data reads the `cache` of the windowed estimator, which stays `nothing`.

# Arguments

  - `ce`: Windowed covariance estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`WindowedCovariance`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::WindowedCovariance{<:Any, <:Any, <:Any, <:SeedWindow};
                        kwargs...)
    return Statistics.cor(ce.ce; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the variance from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there.

# Arguments

  - `ve`: Windowed variance estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:vararr])

# Related

  - [`WindowedVariance`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ve::WindowedVariance{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
    return Statistics.var(ve.ve; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the standard deviation from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there. The generic `std` with no data reads the `cache` of the windowed estimator, which stays `nothing`.

# Arguments

  - `ve`: Windowed variance estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:stdarr])

# Related

  - [`WindowedVariance`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ve::WindowedVariance{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
    return Statistics.std(ve.ve; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the coskewness from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there.

# Arguments

  - `ske`: Windowed coskewness estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`WindowedCoskewness`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function coskewness(ske::WindowedCoskewness{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
    return coskewness(ske.ske; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the cokurtosis from the fold of a windowed estimator under a [`SeedWindow`](@ref).

The inner estimator holds the state of the fold, so the method reads it there.

# Arguments

  - `kte`: Windowed cokurtosis estimator under a seed window.
  - `kwargs...`: Keywords for the read of the inner estimator.

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`WindowedCokurtosis`](@ref)
  - [`partial_fit!(est::SeedWindowed, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function cokurtosis(kte::WindowedCokurtosis{<:Any, <:Any, <:Any, <:SeedWindow}; kwargs...)
    return cokurtosis(kte.kte; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the fields of a windowed estimator that the printout shows, which are every field except `cache`.

The `cache` field holds the Partial Fit State of an online fit. It is not configuration. Without this method, the state prints under the estimator each time a Prior or an optimiser that holds the estimator prints. To show the field for one type, call `set_show_nothing_fields!` with the name of the type and `true`, for example `set_show_nothing_fields!(:WindowedCovariance, true)`.

# Arguments

  - `est`: Windowed estimator. The function reads its type alone.

# Returns

  - `fields::Tuple`: The field names to show, which are every field name except `:cache`.

# Related

  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(est::Union{WindowedExpectedReturns, WindowedCovariance,
                                WindowedVariance, WindowedCoskewness, WindowedCokurtosis})
    return filter(!=(:cache), fieldnames(typeof(est)))
end
