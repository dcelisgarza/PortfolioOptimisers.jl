@windowed_estimator WindowedCokurtosis <: CokurtosisEstimator begin
    kte::CokurtosisEstimator = Cokurtosis()
    noun = "Cokurtosis"
    forward = [cokurtosis(::MatNum; mean) => :kte]
    doctest = """
    julia> WindowedCokurtosis()
    WindowedCokurtosis
         kte ┼ Cokurtosis
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
