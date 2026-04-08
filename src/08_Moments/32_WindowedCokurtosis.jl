@concrete struct WindowedCokurtosis <: CokurtosisEstimator
    "$(field_dict[:ke])"
    ke
    "$(field_dict[:oow])"
    w
    window
    function WindowedCokurtosis(ke::Cokurtosis, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        validate_observation_weights(w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ke), typeof(w), typeof(window)}(ke, w, window)
    end
end
function WindowedCokurtosis(; ke::Cokurtosis = Cokurtosis(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)
    return WindowedCokurtosis(ke, w, window)
end
function factory(ke::WindowedCokurtosis, w::ObsWeights)
    return WindowedCokurtosis(; ke = factory(ke, w), w = w, window = ke.window)
end
function cokurtosis(ke::WindowedCokurtosis, X::MatNum; dims::Int = 1, kwargs...)
    X, w = moment_window_and_weights(X, ke.w, ke.window; dims = dims, kwargs...)
    ke = factory(ke.ke, w)
    return cokurtosis(ke, X; dims = dims, kwargs...)
end

export WindowedCokurtosis
