@windowed_estimator WindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun = "Variance"
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.var(::VecNum; mean) => :varnum,
               Statistics.std(::MatNum; mean) => :stdarr,
               Statistics.std(::VecNum; mean) => :stdnum]
    doctest = """
    julia> WindowedVariance()
    WindowedVariance
          ve ┼ SimpleVariance
             │          me ┼ SimpleExpectedReturns
             │             │   w ┴ nothing
             │           w ┼ nothing
             │   corrected ┴ Bool: true
           w ┼ nothing
      window ┴ nothing
    """
end
function variance_count(ve::WindowedVariance, X::MatNum)
    inner, Xw, _ = windowed_preamble(ve.ve, ve.w, ve.window, X)
    return variance_count(inner, Xw)
end
