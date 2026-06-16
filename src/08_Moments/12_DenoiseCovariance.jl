"""
    DenoiseCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        dn::Denoise = Denoise(),
        pdm::Option{<:Posdef} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    DenoiseCovariance(
        ce::StatsBase.CovarianceEstimator,
        dn::Denoise,
        pdm::Option{<:Posdef},
    ) -> PortfolioOptimisersCovariance

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then denoising, in that order, via [`MatrixProcessing`](@ref).

# Examples

```jldoctest
julia> DenoiseCovariance()
PortfolioOptimisersCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ SimpleCovariance: SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┴ Full()
  mp ┼ MatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ Denoise
     │         │      pdm ┼ Posdef
     │         │          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │         │      alg ┼ ShrunkDenoise
     │         │          │   alpha ┴ Float64: 0.0
     │         │     args ┼ Tuple{}: ()
     │         │   kwargs ┼ @NamedTuple{}: NamedTuple()
     │         │   kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
     │         │        m ┼ Int64: 10
     │         │        n ┴ Int64: 1000
     │      dt ┼ nothing
     │     alg ┼ nothing
     │   order ┴ Tuple{Symbol, Symbol}: (:pdm, :dn)
```

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
function DenoiseCovariance(ce::StatsBase.CovarianceEstimator, dn::Denoise,
                           pdm::Option{<:Posdef})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = ce,
                                         mp = MatrixProcessing(; pdm = pdm, dn = dn,
                                                               order = (:pdm, :dn)))
end
function DenoiseCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           dn::Denoise = Denoise(),
                           pdm::Option{<:Posdef} = Posdef())::PortfolioOptimisersCovariance
    return DenoiseCovariance(ce, dn, pdm)
end

export DenoiseCovariance
