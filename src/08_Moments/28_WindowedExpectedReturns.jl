@concrete struct WindowedExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:oow])"
    w
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
function factory(me::WindowedExpectedReturns, w::ObsWeights)
    return WindowedExpectedReturns(; me = factory(me, w), w = w, window = me.window)
end
function Statistics.mean(me::WindowedExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    X, w = moment_window_and_weights(X, me.w, me.window; dims = dims, kwargs...)
    me = factory(me.me, w)
    return Statistics.mean(me, X; dims = dims, kwargs...)
end

export WindowedExpectedReturns
