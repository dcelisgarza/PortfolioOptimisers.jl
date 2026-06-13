"""
$(DocStringExtensions.TYPEDEF)

Covariance estimator that restricts computation to a rolling or indexed observation window.

`WindowedCovariance` wraps another covariance estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted covariance estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedCovariance(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

# Examples

```jldoctest
julia> WindowedCovariance()
WindowedCovariance
      ce ┼ PortfolioOptimisersCovariance
         │   ce ┼ Covariance
         │      │    me ┼ SimpleExpectedReturns
         │      │       │   w ┴ nothing
         │      │    ce ┼ GeneralCovariance
         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │      │       │    w ┴ nothing
         │      │   alg ┴ Full()
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
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct WindowedCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    ce
    """
    $(field_dict[:oow])
    """
    w
    """
    Window specification: an integer (last `window` observations) or a vector of indices.
    """
    window
    function WindowedCovariance(ce::StatsBase.CovarianceEstimator, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ce), typeof(w), typeof(window)}(ce, w, window)
    end
end
function WindowedCovariance(;
                            ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)::WindowedCovariance
    return WindowedCovariance(ce, w, window)
end
"""
    factory(ce::WindowedCovariance, w::ObsWeights) -> WindowedCovariance

Return a new [`WindowedCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator and stored as the windowed weights.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`WindowedCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::WindowedCovariance, w::ObsWeights)::WindowedCovariance
    return WindowedCovariance(; ce = factory(ce.ce, w), w = w, window = ce.window)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the covariance estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ce])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:cev])

# Related

  - [`WindowedCovariance`](@ref)
"""
function moment_view(ce::WindowedCovariance, i)::WindowedCovariance
    return WindowedCovariance(; ce = moment_view(ce.ce, i), w = ce.w, window = ce.window)
end
"""
    Statistics.cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                   kwargs...)

Compute the covariance matrix using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying covariance estimator.

# Arguments

  - `ce`: Windowed covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean passed to the underlying estimator.
  - $(arg_dict[:oiv])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - $(ret_dict[:sigma])

# Related

  - [`WindowedCovariance`](@ref)
  - [`cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        iv::Option{<:MatNum} = nothing, kwargs...)
    window = get_window(ce.window, X, dims)
    X, w = moment_window_and_weights(X, ce.w, window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    if !isnothing(iv) && isa(window, VecInt)
        iv = isone(dims) ? view(iv, window, :) : view(iv, :, window)
    end
    return Statistics.cov(ce, X; dims = dims, mean = mean, iv = iv, kwargs...)
end
"""
    Statistics.cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, iv::Option{<:MatNum} = nothing,
                   kwargs...)

Compute the correlation matrix using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying covariance estimator's `cor` method.

# Arguments

  - `ce`: Windowed covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean passed to the underlying estimator.
  - $(arg_dict[:oiv])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`WindowedCovariance`](@ref)
  - [`cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1,
                        iv::Option{<:MatNum} = nothing, kwargs...)
    window = get_window(ce.window, X, dims)
    X, w = moment_window_and_weights(X, ce.w, window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    if !isnothing(iv) && isa(window, VecInt)
        iv = isone(dims) ? view(iv, window, :) : view(iv, :, window)
    end
    return Statistics.cor(ce, X; dims = dims, iv = iv, kwargs...)
end

export WindowedCovariance
