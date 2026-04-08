@concrete struct WindowedCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:oow])"
    w
    window
    function WindowedCovariance(ce::AbstractCovarianceEstimator, w::Option{<:ObsWeights},
                                window::Option{<:Int_VecInt})
        validate_observation_weights(w)
        assert_nonempty_nonneg_finite_val(window, :window)
        return new{typeof(ce), typeof(w), typeof(window)}(ce, w, window)
    end
end
function WindowedCovariance(;
                            ce::AbstractCovarianceEstimator = PortfolioOptimisersCovariance(),
                            w::Option{<:ObsWeights} = nothing,
                            window::Option{<:Int_VecInt} = nothing)
    return WindowedCovariance(ce, w, window)
end
function factory(ce::WindowedCovariance, w::ObsWeights)
    return WindowedCovariance(; ce = factory(ce.ce, w), w = w, window = ce.window)
end
function Statistics.cov(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.cov(ce, X; dims = dims, mean = mean, kwargs...)
end
function Statistics.cor(ce::WindowedCovariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    X, w = moment_window_and_weights(X, ce.w, ce.window; dims = dims, kwargs...)
    ce = factory(ce.ce, w)
    return Statistics.cor(ce, X; dims = dims, mean = mean, kwargs...)
end

export WindowedCovariance
