"""
    DetoneCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        dt::Detone = Detone(),
        pdm::Option{<:Posdef} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    DetoneCovariance(
        ce::StatsBase.CovarianceEstimator,
        dt::Detone,
        pdm::Option{<:Posdef},
    ) -> PortfolioOptimisersCovariance

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then detoning, in that order, via [`MatrixProcessing`](@ref).

# Examples

```jldoctest
julia> DetoneCovariance()
PortfolioOptimisersCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┴ FullMoment()
  mp ┼ MatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ nothing
     │      dt ┼ Detone
     │         │   pdm ┼ Posdef
     │         │       │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │         │     n ┴ Int64: 1
     │     alg ┼ nothing
     │   order ┴ Tuple{Symbol, Symbol}: (:pdm, :dt)
```

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
function DetoneCovariance(ce::StatsBase.CovarianceEstimator, dt::Detone,
                          pdm::Option{<:Posdef})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = ce,
                                         mp = MatrixProcessing(; pdm = pdm, dt = dt,
                                                               order = (:pdm, :dt)))
end
function DetoneCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                          dt::Detone = Detone(),
                          pdm::Option{<:Posdef} = Posdef())::PortfolioOptimisersCovariance
    return DetoneCovariance(ce, dt, pdm)
end

export DetoneCovariance
