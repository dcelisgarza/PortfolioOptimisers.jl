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

Returns a [`PortfolioOptimisersCovariance`](@ref) that projects the covariance matrix to the
positive definite cone and then detones it. A [`MatrixProcessing`](@ref) applies the two steps in
that order.

[`detone!`](@ref) in `src/04_MatrixProcessing/03_Detone.jl` states the mathematics of the detoning
step. [`posdef!`](@ref) in `src/04_MatrixProcessing/01_PosdefMatrix.jl` states the mathematics of
the projection.

# Algorithm

 1. Build a [`MatrixProcessing`](@ref) from `pdm` and `dt`, with `order = (:pdm, :dt)`.
 2. Return a [`PortfolioOptimisersCovariance`](@ref) that holds `ce` and that estimator.

The `order` field sets the composition. `Statistics.cov` runs `ce` first. Then it projects the
matrix onto the positive definite cone, and it detones the matrix last. For the reverse order,
build the [`MatrixProcessing`](@ref) directly.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:dt])
  - $(arg_dict[:opdm])

# Returns

  - `ce::PortfolioOptimisersCovariance`: The composite estimator that detones the matrix that `ce` computes.

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
