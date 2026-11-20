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

Returns a [`PortfolioOptimisersCovariance`](@ref) that projects the covariance matrix to the
positive definite cone and then applies a matrix processing algorithm of the caller. A
[`MatrixProcessing`](@ref) applies the two steps in that order.

[`matrix_processing!`](@ref) in `src/04_MatrixProcessing/04_MatrixProcessing.jl` states what an
[`AbstractMatrixProcessingAlgorithm`](@ref) does to the matrix. [`posdef!`](@ref) in
`src/04_MatrixProcessing/01_PosdefMatrix.jl` states the mathematics of the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `alg`, with `order = (:pdm, :alg)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) that holds `ce` and that estimator.

The `order` field sets the composition. `Statistics.cov` runs `ce` first. Then it projects the
matrix onto the positive definite cone, and it applies `alg` last. For the reverse order, build the
[`MatrixProcessing`](@ref) directly.

# Arguments

  - $(arg_dict[:ce])
  - `alg`: The optional matrix processing algorithm that runs after the projection. If `alg` is `nothing`, the projection is the only step.
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: The composite estimator that applies `alg` to the matrix that `ce` computes.

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
     │   alg ┼ FullMoment()
     │     w ┴ nothing
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
    return PortfolioOptimisersCovariance(; ce = ce,
                                         mp = MatrixProcessing(; pdm = pdm, alg = alg,
                                                               order = (:pdm, :alg)))
end
function ProcessedCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:AbstractPosdefEstimator} = Posdef())::PortfolioOptimisersCovariance
    return ProcessedCovariance(ce, alg, pdm)
end

export ProcessedCovariance
