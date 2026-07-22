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
      window ┴ nothing
    """
end
