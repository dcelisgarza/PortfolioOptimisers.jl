"""
    DetoneCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        dt::AbstractDetoneEstimator = Detone(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
    ) -> PortfolioOptimisersCovariance

    DetoneCovariance(
        ce::StatsBase.CovarianceEstimator,
        dt::AbstractDetoneEstimator,
        pdm::Option{<:AbstractPosdefEstimator},
    ) -> PortfolioOptimisersCovariance

Convenience constructor. Returns a [`PortfolioOptimisersCovariance`](@ref) configured to apply
positive definite projection then detoning, in that order, via [`MatrixProcessing`](@ref).

[`detone!`](@ref) in `src/06_Detone.jl` states the mathematics of the detoning step, and
[`posdef!`](@ref) in `src/04_PosdefMatrix.jl` that of the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `dt`, with `order = (:pdm, :dt)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) carrying `ce` and that estimator.

`order` is what fixes the composition: `Statistics.cov` runs `ce` first, then projects the matrix
onto the positive definite cone, and detones last. A caller who needs the reverse order builds the
[`MatrixProcessing`](@ref) itself.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:dt])
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: Composite estimator that detones the matrix `ce` computes.

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
     │   alg ┼ FullMoment()
     │     w ┴ nothing
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
function DetoneCovariance(ce::StatsBase.CovarianceEstimator, dt::AbstractDetoneEstimator,
                          pdm::Option{<:AbstractPosdefEstimator})::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = ce,
                                         mp = MatrixProcessing(; pdm = pdm, dt = dt,
                                                               order = (:pdm, :dt)))
end
function DetoneCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                          dt::AbstractDetoneEstimator = Detone(),
                          pdm::Option{<:AbstractPosdefEstimator} = Posdef())::PortfolioOptimisersCovariance
    return DetoneCovariance(ce, dt, pdm)
end

export DetoneCovariance
