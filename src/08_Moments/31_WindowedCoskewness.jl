@concrete struct WindowedCoskewness <: CoskewnessEstimator
    "$(field_dict[:ske])"
    ske
    "$(field_dict[:oow])"
    w
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
function factory(ske::WindowedCoskewness, w::ObsWeights)
    return WindowedCoskewness(; ske = factory(ske, w), w = w, window = ske.window)
end
function coskewness(ske::WindowedCoskewness, X::MatNum; dims::Int = 1, kwargs...)
    X, w = moment_window_and_weights(X, ske.w, ske.window; dims = dims, kwargs...)
    ske = factory(ske.ske, w)
    return coskewness(ske, X; dims = dims, kwargs...)
end

export WindowedCoskewness
