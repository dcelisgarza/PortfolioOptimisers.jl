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
function effective_observation_count(me::WindowedExpectedReturns, X::MatNum)
    inner, Xw, _ = windowed_preamble(me.me, me.w, me.window, X)
    return effective_observation_count(inner, Xw)
end
