@concrete struct WindowedVariance <: AbstractVarianceEstimator
    "$(field_dict[:me])"
    ce
    "$(field_dict[:oow])"
    w
    window
    function WindowedVariance(ce::AbstractVarianceEstimator, w::Option{<:ObsWeights},
                              window::Option{<:Int_VecInt})
        validate_observation_weights(w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ce), typeof(w), typeof(window)}(ce, w, window)
    end
end
function WindowedVariance(; ce::AbstractVarianceEstimator = SimpleVariance(),
                          w::Option{<:ObsWeights} = nothing,
                          window::Option{<:Int_VecInt} = nothing)
    return WindowedVariance(ce, w, window)
end
function factory(ce::WindowedVariance, w::ObsWeights)
    return WindowedVariance(; ce = factory(ce, w), w = w, window = ce.window)
end
function Statistics.var(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.var(ce, X; dims = dims, mean = mean, kwargs...)
end
function Statistics.var(ce::WindowedVariance, X::VecNum; mean = nothing)
    X, w = moment_window_and_weights(X, ce.w, ce.window)
    ce = factory(ce.ce, w)
    return Statistics.var(ce, X; mean = mean)
end
function Statistics.std(ce::WindowedVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.std(ce, X; dims = dims, mean = mean, kwargs...)
end
function Statistics.std(ce::WindowedVariance, X::VecNum; mean = nothing)
    X, w = moment_window_and_weights(X, ce.w, ce.window)
    ce = factory(ce.ce, w)
    return Statistics.std(ce, X; mean = mean)
end

export WindowedVariance
