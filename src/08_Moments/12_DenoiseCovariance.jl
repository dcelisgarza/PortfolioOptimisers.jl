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

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then denoising, in that order, via [`MatrixProcessing`](@ref).

[`denoise!`](@ref) in `src/05_Denoise.jl` states the mathematics of the denoising step, and
[`posdef!`](@ref) in `src/04_PosdefMatrix.jl` that of the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `dn`, with `order = (:pdm, :dn)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) carrying `ce` and that estimator.

`order` is what fixes the composition: `Statistics.cov` runs `ce` first, then projects the matrix
onto the positive definite cone, and denoises last. A caller who needs the reverse order builds the
[`MatrixProcessing`](@ref) itself.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:dn])
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: Composite estimator that denoises the matrix `ce` computes.

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
