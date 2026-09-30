@windowed_estimator WindowedCovariance <: AbstractCovarianceEstimator begin
    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    noun = "Covariance"
    forward = [Statistics.cov(::MatNum; mean) => :sigma,
               Statistics.cor(::MatNum; mean) => :rho]
    doctest = """
    julia> WindowedCovariance()
    WindowedCovariance
          ce ┼ PortfolioOptimisersCovariance
             │   ce ┼ Covariance
             │      │    me ┼ SimpleExpectedReturns
             │      │       │   w ┴ nothing
             │      │    ce ┼ GeneralCovariance
             │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │      │       │    w ┴ nothing
             │      │   alg ┼ FullMoment()
             │      │     w ┴ nothing
             │   mp ┼ MatrixProcessing
             │      │     pdm ┼ Posdef
             │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      │      dn ┼ nothing
             │      │      dt ┼ nothing
             │      │     alg ┼ nothing
             │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
           w ┼ nothing
      window ┴ nothing
    """
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the point-in-time variance series of a windowed covariance estimator.

Row `t` is the fit of the estimator on the observations `1` to `t`. The estimator keeps the last observations of its window among them, so the window rolls with `t`, and the last row equals the batch fit. Before the window is full, row `t` keeps every observation from `1` to `t`. [`windowed_variance_series`](@ref) states the rule.

A prior calls this method with the `active_mask` and `estimation_mask` keywords, and each fit cuts them to its own rows.

# Arguments

  - `ce`: Windowed covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the inner estimator. The two universe masks are cut to the rows of each fit.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`WindowedCovariance`](@ref)
  - [`windowed_variance_series`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::WindowedCovariance, X::MatNum; dims::Int = 1, kwargs...)
    return windowed_variance_series(ce.ce, ce.w, ce.window, X, nothing; dims = dims,
                                    kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the point-in-time variance series of a windowed covariance estimator from an Asset Panel.

Row `t` is the fit of the estimator on the observations `1` to `t` of the returns and of the panel, as in [`variance_series(ce::WindowedCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Each fit gives the inner estimator the panel over its own rows, so a mask-aware inner estimator reads the active mask of the window, and a plain one reduces to the Coverage Universe of the window.

# Arguments

  - `ce`: Windowed covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the inner estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`WindowedCovariance`](@ref)
  - [`windowed_variance_series`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::WindowedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    return windowed_variance_series(ce.ce, ce.w, ce.window, X, pnl; dims = dims, kwargs...)
end
