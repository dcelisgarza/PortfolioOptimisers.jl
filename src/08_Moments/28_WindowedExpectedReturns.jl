"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that restricts computation to a rolling or indexed observation window.

`WindowedExpectedReturns` wraps another expected returns estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted expected returns estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
"""
@concrete struct WindowedExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:oow])"
    w
    "Window specification: an integer (last `window` observations) or a vector of indices."
    window
    function WindowedExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                     w::Option{<:ObsWeights}, window::Option{<:Int_VecInt})
        validate_observation_weights(w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(me), typeof(w), typeof(window)}(me, w, window)
    end
end
function WindowedExpectedReturns(;
                                 me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                 w::Option{<:ObsWeights} = nothing,
                                 window::Option{<:Int_VecInt} = nothing)
    return WindowedExpectedReturns(me, w, window)
end
"""
    factory(me::WindowedExpectedReturns, w::ObsWeights) -> WindowedExpectedReturns

Return a new [`WindowedExpectedReturns`](@ref) estimator with observation weights `w` applied to the underlying mean estimator and stored as the windowed weights.

# Arguments

  - `me`: Windowed expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::WindowedExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`WindowedExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(me::WindowedExpectedReturns, w::ObsWeights)
    return WindowedExpectedReturns(; me = factory(me.me, w), w = w, window = me.window)
end
"""
    Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute expected returns using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying mean estimator.

# Arguments

  - `me`: Windowed expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `mu::ArrNum`: Expected returns vector for the selected window.

# Related

  - [`WindowedExpectedReturns`](@ref)
"""
function Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    X, w = moment_window_and_weights(X, me.w, me.window; dims = dims, kwargs...)
    me = factory(me.me, w)
    return Statistics.mean(me, X; dims = dims, kwargs...)
end

export WindowedExpectedReturns
