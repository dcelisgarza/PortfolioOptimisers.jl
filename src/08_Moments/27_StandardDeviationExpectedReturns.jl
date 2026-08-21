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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

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
     │      │   alg ┴ FullMoment()
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct StandardDeviationExpectedReturns <:
                               AbstractExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    function StandardDeviationExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function StandardDeviationExpectedReturns(;
                                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())::StandardDeviationExpectedReturns
    return StandardDeviationExpectedReturns(ce)
end
"""
    Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the standard deviation of each asset.

This method returns the standard deviation vector of `X` as estimated by the covariance estimator `me.ce`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_j &= \\hat{\\sigma}_j = \\sqrt{\\hat{\\mathbf{\\Sigma}}_{jj}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: Expected return proxy of asset ``j``.
  - ``\\hat{\\sigma}_j``: Standard deviation of asset ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}``: Covariance matrix that `me.ce` estimates.
  - ``\\hat{\\mathbf{\\Sigma}}_{jj}``: ``j``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `me`: Standard deviation expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: Standard deviation vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Details

  - The method reads the diagonal of the matrix that `me.ce` returns, not a formula of its own. Every choice inside `me.ce` therefore reaches the result: the moment algorithm, the observation weights, and the matrix processing.

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
  - [`std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

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
     │      │   alg ┴ FullMoment()
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct VarianceExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    function VarianceExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function VarianceExpectedReturns(;
                                 ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())::VarianceExpectedReturns
    return VarianceExpectedReturns(ce)
end
"""
    Statistics.mean(me::VarianceExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the variance of each asset.

This method returns the variance vector of `X` as estimated by the covariance estimator `me.ce`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_j &= \\hat{\\sigma}_j^2 = \\hat{\\mathbf{\\Sigma}}_{jj}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: Expected return proxy of asset ``j``.
  - ``\\hat{\\sigma}_j^2``: Variance of asset ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}``: Covariance matrix that `me.ce` estimates.
  - ``\\hat{\\mathbf{\\Sigma}}_{jj}``: ``j``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `me`: Variance expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: Variance vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Details

  - The method reads the diagonal of the matrix that `me.ce` returns, not a formula of its own. Every choice inside `me.ce` therefore reaches the result: the moment algorithm, the observation weights, and the matrix processing.

# Related

  - [`VarianceExpectedReturns`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    return Statistics.var(me.ce, X; dims = dims, kwargs...)
end

export StandardDeviationExpectedReturns, VarianceExpectedReturns
