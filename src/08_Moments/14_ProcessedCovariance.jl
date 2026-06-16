"""
    ProcessedCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
        pdm::Option{<:Posdef} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    ProcessedCovariance(
        ce::StatsBase.CovarianceEstimator,
        alg::Option{<:AbstractMatrixProcessingAlgorithm},
        pdm::Option{<:Posdef},
    ) -> PortfolioOptimisersCovariance

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then a custom matrix processing algorithm, in that order, via
[`MatrixProcessing`](@ref).

# Examples

```jldoctest
julia> ProcessedCovariance()
PortfolioOptimisersCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┴ Full()
  mp ┼ MatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ nothing
     │      dt ┼ nothing
     │     alg ┼ nothing
     │   order ┴ Tuple{Symbol, Symbol}: (:pdm, :alg)
```

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`MatrixProcessing`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
function ProcessedCovariance(ce::StatsBase.CovarianceEstimator,
                             alg::Option{<:AbstractMatrixProcessingAlgorithm},
                             pdm::Option{<:Posdef})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(ce,
                                         MatrixProcessing(; pdm = pdm, alg = alg,
                                                          order = (:pdm, :alg)))
end
function ProcessedCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:Posdef} = Posdef())::PortfolioOptimisersCovariance
    return ProcessedCovariance(ce, alg, pdm)
end

export ProcessedCovariance
