"""
$(DocStringExtensions.TYPEDEF)

Coskewness estimator that restricts computation to a rolling or indexed observation window.

`WindowedCoskewness` wraps another coskewness estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted coskewness estimation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowedCoskewness(;
        ske::CoskewnessEstimator = Coskewness(),
        w::Option{<:ObsWeights} = nothing,
        window::Option{<:Int_VecInt} = nothing
    ) -> WindowedCoskewness

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - If `window` is provided, it must be nonempty, nonnegative, and finite.

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`Coskewness`](@ref)
"""
@concrete struct WindowedCoskewness <: CoskewnessEstimator
    "$(field_dict[:ske])"
    ske
    "$(field_dict[:oow])"
    w
    "Window specification: an integer (last `window` observations) or a vector of indices."
    window
    function WindowedCoskewness(ske::CoskewnessEstimator, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        validate_observation_weights(w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ske), typeof(w), typeof(window)}(ske, w, window)
    end
end
function WindowedCoskewness(; ske::CoskewnessEstimator = Coskewness(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)
    return WindowedCoskewness(ske, w, window)
end
"""
    factory(ske::WindowedCoskewness, w::ObsWeights) -> WindowedCoskewness

Return a new [`WindowedCoskewness`](@ref) estimator with observation weights `w` applied to the underlying coskewness estimator and stored as the windowed weights.

# Arguments

  - `ske`: Windowed coskewness estimator.
  - $(arg_dict[:ow])

# Returns

  - `ske::WindowedCoskewness`: Updated estimator with weights applied.

# Related

  - [`WindowedCoskewness`](@ref)
  - [`factory`](@ref)
"""
function factory(ske::WindowedCoskewness, w::ObsWeights)
    return WindowedCoskewness(; ske = factory(ske.ske, w), w = w, window = ske.window)
end
"""
    coskewness(ske::WindowedCoskewness, X::MatNum; dims::Int = 1, kwargs...)

Compute the coskewness tensor and processed matrix using a rolling or indexed observation window.

This method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying coskewness estimator.

# Arguments

  - `ske`: Windowed coskewness estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - `cskew::Matrix{<:Number}`: Coskewness tensor (assets × assets²).
  - `V::Matrix{<:Number}`: Processed coskewness matrix (assets × assets).

# Related

  - [`WindowedCoskewness`](@ref)
  - [`Coskewness`](@ref)
  - [`coskewness`](@ref)
"""
function coskewness(ske::WindowedCoskewness, X::MatNum; dims::Int = 1, kwargs...)
    X, w = moment_window_and_weights(X, ske.w, ske.window; dims = dims, kwargs...)
    ske = factory(ske.ske, w)
    return coskewness(ske, X; dims = dims, kwargs...)
end

export WindowedCoskewness
