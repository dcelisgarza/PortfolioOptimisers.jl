"""
    DenoiseCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        dn::AbstractDenoiseEstimator = Denoise(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    DenoiseCovariance(
        ce::StatsBase.CovarianceEstimator,
        dn::AbstractDenoiseEstimator,
        pdm::Option{<:AbstractPosdefEstimator},
    ) -> PortfolioOptimisersCovariance

Returns a [`PortfolioOptimisersCovariance`](@ref) that projects the covariance matrix to the
positive definite cone and then denoises it. A [`MatrixProcessing`](@ref) applies the two steps in
that order.

[`denoise!`](@ref) in `src/04_MatrixProcessing/02_Denoise.jl` states the mathematics of the
denoising step. [`posdef!`](@ref) in `src/04_MatrixProcessing/01_PosdefMatrix.jl` states the
mathematics of the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `dn`, with `order = (:pdm, :dn)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) that holds `ce` and that estimator.

The `order` field sets the composition. `Statistics.cov` runs `ce` first. Then it projects the
matrix onto the positive definite cone, and it denoises the matrix last. For the reverse order,
build the [`MatrixProcessing`](@ref) directly.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:dn])
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: The composite estimator that denoises the matrix that `ce` computes.

# Examples

```jldoctest
julia> DenoiseCovariance()
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
function DenoiseCovariance(ce::StatsBase.CovarianceEstimator, dn::AbstractDenoiseEstimator,
                           pdm::Option{<:AbstractPosdefEstimator})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = ce,
                                         mp = MatrixProcessing(; pdm = pdm, dn = dn,
                                                               order = (:pdm, :dn)))
end
function DenoiseCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                           dn::AbstractDenoiseEstimator = Denoise(),
                           pdm::Option{<:AbstractPosdefEstimator} = Posdef())::PortfolioOptimisersCovariance
    return DenoiseCovariance(ce, dn, pdm)
end

export DenoiseCovariance
