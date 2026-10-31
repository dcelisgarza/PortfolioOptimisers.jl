@windowed_estimator WindowedExpectedReturns <: AbstractExpectedReturnsEstimator begin
    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns()
    noun = "Expected returns"
    forward = [Statistics.mean(::MatNum) => :mu]
    doctest = """
    julia> WindowedExpectedReturns()
    WindowedExpectedReturns
          me ┼ SimpleExpectedReturns
             │   w ┴ nothing
           w ┼ nothing
      window ┼ nothing
        rule ┴ RollingWindow()
    """
end
function effective_observation_count(me::WindowedExpectedReturns, X::MatNum; kwargs...)
    inner, Xw, iv, kw = windowed_preamble(me.me, me.w, me.window, X; kwargs...)
    return effective_observation_count(inner, Xw; iv = iv, kw...)
end
