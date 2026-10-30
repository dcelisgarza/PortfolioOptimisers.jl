@windowed_estimator WindowedCokurtosis <: CokurtosisEstimator begin
    kte::CokurtosisEstimator = Cokurtosis()
    noun = "Cokurtosis"
    forward = [cokurtosis(::MatNum; mean) => :ckurt]
    doctest = """
    julia> WindowedCokurtosis()
    WindowedCokurtosis
         kte ┼ Cokurtosis
             │      me ┼ SimpleExpectedReturns
             │         │   w ┴ nothing
             │      mp ┼ MatrixProcessing
             │         │     pdm ┼ Posdef
             │         │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │         │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │         │      dn ┼ nothing
             │         │      dt ┼ nothing
             │         │     alg ┼ nothing
             │         │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
             │     alg ┼ FullMoment()
             │       w ┼ nothing
             │   cache ┴ nothing
           w ┼ nothing
      window ┼ nothing
        rule ┴ RollingWindow()
    """
end
function effective_observation_count(kte::WindowedCokurtosis, X::MatNum; mean = nothing,
                                     kwargs...)
    inner, Xw, iv, kw = windowed_preamble(kte.kte, kte.w, kte.window, X; kwargs...)
    return effective_observation_count(inner, Xw; mean = mean, iv = iv, kw...)
end
