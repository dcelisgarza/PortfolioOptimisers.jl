@windowed_estimator WindowedCoskewness <: CoskewnessEstimator begin
    ske::CoskewnessEstimator = Coskewness()
    noun = "Coskewness"
    forward = [coskewness(::MatNum; mean) => (:cskew, :cskewV)]
    doctest = """
    julia> WindowedCoskewness()
    WindowedCoskewness
         ske ┼ Coskewness
             │    me ┼ SimpleExpectedReturns
             │       │   w ┴ nothing
             │    mp ┼ MatrixProcessing
             │       │     pdm ┼ Posdef
             │       │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │       │      dn ┼ nothing
             │       │      dt ┼ nothing
             │       │     alg ┼ nothing
             │       │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │   alg ┼ FullMoment()
             │     w ┴ nothing
           w ┼ nothing
      window ┴ nothing
    """
end
