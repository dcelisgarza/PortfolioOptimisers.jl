@windowed_estimator WindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun = "Variance"
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.var(::VecNum; mean) => :varnum,
               Statistics.std(::MatNum; mean) => :stdarr,
               Statistics.std(::VecNum; mean) => :stdnum]
    doctest = """
    julia> WindowedVariance()
    WindowedVariance
          ve ┼ SimpleVariance
             │          me ┼ SimpleExpectedReturns
             │             │   w ┴ nothing
             │           w ┼ nothing
             │   corrected ┴ Bool: true
           w ┼ nothing
      window ┼ nothing
        rule ┴ RollingWindow()
    """
end
function variance_count(ve::WindowedVariance, X::MatNum)
    inner, Xw, _ = windowed_preamble(ve.ve, ve.w, ve.window, X)
    return variance_count(inner, Xw)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the point-in-time variance series of a windowed variance estimator.

Row `t` is the fit of the estimator on the observations `1` to `t`. The estimator keeps the last observations of its window among them, so the window rolls with `t`, and the last row equals the batch fit. Before the window is full, row `t` keeps every observation from `1` to `t`. [`windowed_variance_series`](@ref) states the rule.

A prior calls this method in its variance slot, with the `active_mask` and `estimation_mask` keywords, and each fit cuts them to its own rows.

# Arguments

  - `ve`: Windowed variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the inner estimator. The two universe masks are cut to the rows of each fit.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`WindowedVariance`](@ref)
  - [`windowed_variance_series`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ve::WindowedVariance, X::MatNum; dims::Int = 1, kwargs...)
    return windowed_variance_series(ve.ve, ve.w, ve.window, X, nothing; dims = dims,
                                    kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the point-in-time variance series of a windowed variance estimator from an Asset Panel.

Row `t` is the fit of the estimator on the observations `1` to `t` of the returns and of the panel, as in [`variance_series(ve::WindowedVariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Each fit gives the inner estimator the panel over its own rows, so a mask-aware inner estimator reads the active mask of the window, and a plain one reduces to the Coverage Universe of the window.

# Arguments

  - `ve`: Windowed variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Keywords passed to the inner estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`WindowedVariance`](@ref)
  - [`windowed_variance_series`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ve::WindowedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    return windowed_variance_series(ve.ve, ve.w, ve.window, X, pnl; dims = dims, kwargs...)
end
