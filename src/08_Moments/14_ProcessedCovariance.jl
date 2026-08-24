"""
    ProcessedCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    ProcessedCovariance(
        ce::StatsBase.CovarianceEstimator,
        alg::Option{<:AbstractMatrixProcessingAlgorithm},
        pdm::Option{<:AbstractPosdefEstimator},
    ) -> PortfolioOptimisersCovariance

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then a custom matrix processing algorithm, in that order, via
[`MatrixProcessing`](@ref).

[`matrix_processing!`](@ref) in `src/07_MatrixProcessing.jl` states what an
[`AbstractMatrixProcessingAlgorithm`](@ref) does to the matrix, and [`posdef!`](@ref) in
`src/04_PosdefMatrix.jl` states the mathematics of the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `alg`, with `order = (:pdm, :alg)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) carrying `ce` and that estimator.

`order` is what fixes the composition: `Statistics.cov` runs `ce` first, then projects the matrix
onto the positive definite cone, and applies `alg` last. A caller who needs the reverse order builds
the [`MatrixProcessing`](@ref) itself.

# Arguments

  - $(arg_dict[:ce])
  - `alg`: Optional matrix processing algorithm applied after the projection. If `nothing`, the projection is the only step.
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: Composite estimator that applies `alg` to the matrix `ce` computes.

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
     │   alg ┴ FullMoment()
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
                             pdm::Option{<:AbstractPosdefEstimator})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(ce,
                                         MatrixProcessing(; pdm = pdm, alg = alg,
                                                          order = (:pdm, :alg)))
end
function ProcessedCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:AbstractPosdefEstimator} = Posdef())::PortfolioOptimisersCovariance
    return ProcessedCovariance(ce, alg, pdm)
end

export ProcessedCovariance
