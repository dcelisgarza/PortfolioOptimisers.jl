"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset standard deviations.

`StandardDeviationExpectedReturns` computes "expected returns" as the standard deviation of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's volatility.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardDeviationExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> StandardDeviationExpectedReturns

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> StandardDeviationExpectedReturns()
StandardDeviationExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ DenoiseDetoneAlgMatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ DenoiseDetoneAlg()
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct StandardDeviationExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:ce])"
    ce
    function StandardDeviationExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function StandardDeviationExpectedReturns(;
                                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())
    return StandardDeviationExpectedReturns(ce)
end
"""
    factory(ce::StandardDeviationExpectedReturns, w::ObsWeights) -> StandardDeviationExpectedReturns

Return a new [`StandardDeviationExpectedReturns`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - `ce`: Standard deviation expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::StandardDeviationExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::StandardDeviationExpectedReturns, w::ObsWeights)
    return StandardDeviationExpectedReturns(; ce = factory(ce.ce, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
"""
function moment_view(me::StandardDeviationExpectedReturns, i)
    return StandardDeviationExpectedReturns(; me = moment_view(me.ce, i))
end
"""
    Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the standard deviation of each asset.

This method returns the standard deviation vector of `X` as estimated by the covariance estimator `me.ce`.

# Arguments

  - `me`: Standard deviation expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::Matrix{<:Number}`: Standard deviation vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
"""
function Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    return Statistics.std(me.ce, X; dims = dims, kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset variances.

`VarianceExpectedReturns` computes "expected returns" as the variance of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's variance. Variance is the square of volatility (standard deviation).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> VarianceExpectedReturns

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> VarianceExpectedReturns()
VarianceExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ DenoiseDetoneAlgMatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ DenoiseDetoneAlg()
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
"""
@concrete struct VarianceExpectedReturns <: AbstractExpectedReturnsEstimator
    "$(field_dict[:ce])"
    ce
    function VarianceExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function VarianceExpectedReturns(;
                                 ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())
    return VarianceExpectedReturns(ce)
end
"""
    factory(ce::VarianceExpectedReturns, w::ObsWeights) -> VarianceExpectedReturns

Return a new [`VarianceExpectedReturns`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - `ce`: variance expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::VarianceExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`VarianceExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::VarianceExpectedReturns, w::ObsWeights)
    return VarianceExpectedReturns(; ce = factory(ce.ce, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`VarianceExpectedReturns`](@ref)
"""
function moment_view(me::VarianceExpectedReturns, i)
    return VarianceExpectedReturns(; me = moment_view(me.ce, i))
end
"""
    Statistics.mean(me::VarianceExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the variance of each asset.

This method returns the variance vector of `X` as estimated by the covariance estimator `me.ce`.

# Arguments

  - `me`: Variance expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::Matrix{<:Number}`: Variance vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`VarianceExpectedReturns`](@ref)
"""
function Statistics.mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    return Statistics.var(me.ce, X; dims = dims, kwargs...)
end

export StandardDeviationExpectedReturns, VarianceExpectedReturns
