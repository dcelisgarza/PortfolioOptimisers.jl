"""
$(DocStringExtensions.TYPEDEF)

Variance estimator that restricts computation to a rolling or indexed observation window.

`WindowedVariance` wraps another variance estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted variance estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedVariance(;
        ce::AbstractVarianceEstimator = SimpleVariance(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
@concrete struct WindowedVariance <: AbstractVarianceEstimator
    "$(field_dict[:me])"
    ce
    "$(field_dict[:oow])"
    w
    "Window specification: an integer (last `window` observations) or a vector of indices."
    window
    function WindowedVariance(ce::AbstractVarianceEstimator, w::Option{<:ObsWeights},
                              window::Option{<:Int_VecInt})
        assert_nonempty_nonneg_finite_val(w, :w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ce), typeof(w), typeof(window)}(ce, w, window)
    end
end
function WindowedVariance(; ce::AbstractVarianceEstimator = SimpleVariance(),
                          w::Option{<:ObsWeights} = nothing,
                          window::Option{<:Int_VecInt} = nothing)
    return WindowedVariance(ce, w, window)
end
"""
    factory(ce::WindowedVariance, w::ObsWeights) -> WindowedVariance

Return a new [`WindowedVariance`](@ref) estimator with observation weights `w` applied to the underlying variance estimator and stored as the windowed weights.

# Arguments

  - `ce`: Windowed variance estimator.
  - $(arg_dict[:ow])

# Returns

  - `ce::WindowedVariance`: Updated estimator with weights applied.

# Related

  - [`WindowedVariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::WindowedVariance, w::ObsWeights)
    return WindowedVariance(; ce = factory(ce.ce, w), w = w, window = ce.window)
end
"""
    Statistics.var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                   kwargs...)

Compute the variance vector using a rolling or indexed observation window (matrix input).

This method selects a window of observations from `X`, applies observation weights if specified, then delegates to the underlying variance estimator.

# Arguments

  - `ce`: Windowed variance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean passed to the underlying estimator.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `var::Matrix{<:Number}`: Variance vector for the selected window.

# Related

  - [`WindowedVariance`](@ref)
  - [`std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.var(ce, X; dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.var(ce::WindowedVariance, X::VecNum; mean = nothing)

Compute the variance using a rolling or indexed observation window (vector input).

This method selects a window of observations from the vector `X`, applies observation weights if specified, then delegates to the underlying variance estimator.

# Arguments

  - `ce`: Windowed variance estimator.
  - `X`: Data vector of returns.
  - `mean`: Optional pre-computed mean passed to the underlying estimator.

# Returns

  - `var::Number`: Variance for the selected window.

# Related

  - [`WindowedVariance`](@ref)
  - [`var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ce::WindowedVariance, X::VecNum; mean = nothing)
    X, w = moment_window_and_weights(X, ce.w, ce.window)
    ce = factory(ce.ce, w)
    return Statistics.var(ce, X; mean = mean)
end
"""
    Statistics.std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                   kwargs...)

Compute the standard deviation vector using a rolling or indexed observation window (matrix input).

This method selects a window of observations from `X`, applies observation weights if specified, then delegates to the underlying variance estimator's `std` method.

# Arguments

  - `ce`: Windowed variance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `mean`: Optional pre-computed mean passed to the underlying estimator.
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `sd::Matrix{<:Number}`: Standard deviation vector for the selected window.

# Related

  - [`WindowedVariance`](@ref)
  - [`var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.std(ce, X; dims = dims, mean = mean, kwargs...)
end
"""
    Statistics.std(ce::WindowedVariance, X::VecNum; mean = nothing)

Compute the standard deviation using a rolling or indexed observation window (vector input).

This method selects a window of observations from the vector `X`, applies observation weights if specified, then delegates to the underlying variance estimator's `std` method.

# Arguments

  - `ce`: Windowed variance estimator.
  - `X`: Data vector of returns.
  - `mean`: Optional pre-computed mean passed to the underlying estimator.

# Returns

  - `sd::Number`: Standard deviation for the selected window.

# Related

  - [`WindowedVariance`](@ref)
  - [`Statistics.std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.std(ce::WindowedVariance, X::VecNum; mean = nothing)
    X, w = moment_window_and_weights(X, ce.w, ce.window)
    ce = factory(ce.ce, w)
    return Statistics.std(ce, X; mean = mean)
end

export WindowedVariance
