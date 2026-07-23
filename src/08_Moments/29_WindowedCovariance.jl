@windowed_estimator WindowedCovariance <: AbstractCovarianceEstimator begin
    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    noun = "Covariance"
    forward = [Statistics.cov(::MatNum; mean) => :sigma,
               Statistics.cor(::MatNum; mean) => :rho]
    doctest = """
    julia> WindowedCovariance()
    WindowedCovariance
          ce ┼ PortfolioOptimisersCovariance
             │   ce ┼ Covariance
             │      │    me ┼ SimpleExpectedReturns
             │      │       │   w ┴ nothing
             │      │    ce ┼ GeneralCovariance
             │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
             │      │       │    w ┴ nothing
             │      │   alg ┴ FullMoment()
             │   mp ┼ MatrixProcessing
             │      │     pdm ┼ Posdef
             │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
             │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
             │      │      dn ┼ nothing
             │      │      dt ┼ nothing
             │      │     alg ┼ nothing
             │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
           w ┼ nothing
      window ┴ nothing
    """
end
