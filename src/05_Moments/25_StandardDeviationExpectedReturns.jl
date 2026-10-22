"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset standard deviations.

`StandardDeviationExpectedReturns` returns the standard deviation of each asset in the place of an expected return. The covariance estimator `ce` estimates it. The result is in the units of the returns. Use it where a per-asset characteristic need not be a mean, for example to rank the assets on volatility with an [`L1UncertaintySet`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardDeviationExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> StandardDeviationExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

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
     │      │   alg ┼ FullMoment()
     │      │     w ┴ nothing
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
  - [`VarianceExpectedReturns`](@ref): The sibling that returns the square of this result.
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

This method returns the standard deviation of each asset in `X`, which the covariance estimator `me.ce` estimates. The result is in the units of the returns in `X`. It is the elementwise square root of what [`mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) returns on the same data and the same `me.ce`.

The method takes the square root of the diagonal of the matrix that `me.ce` returns. A choice inside `me.ce` that changes that diagonal, such as the observation weights, changes the result. The method calls `me.ce` through [`library_covariance_estimator`](@ref). That function wraps an estimator that the library does not own, such as `StatsBase.SimpleCovariance()`, in a [`GeneralCovariance`](@ref), because `StatsBase` defines no `std` for such an estimator.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_j &= \\hat{\\sigma}_j = \\sqrt{\\hat{\\mathbf{\\Sigma}}_{jj}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: The value that takes the place of the expected return of asset ``j``.
  - ``\\hat{\\sigma}_j``: The standard deviation of asset ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}``: The covariance matrix that `me.ce` estimates.
  - ``\\hat{\\mathbf{\\Sigma}}_{jj}``: The ``j``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `me`: The standard deviation expected returns estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method passes to the covariance estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: The standard deviations, a `(1, N)` matrix if `dims == 1` or an `(N, 1)` matrix if `dims == 2`.

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
  - [`std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): Returns the square of this result.
"""
function Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
    return Statistics.std(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset variances.

`VarianceExpectedReturns` returns the variance of each asset in the place of an expected return. The covariance estimator `ce` estimates it. The result is in the squared units of the returns. The variance is the square of the standard deviation, so this type returns the elementwise square of what [`StandardDeviationExpectedReturns`](@ref) returns on the same data and the same `ce`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> VarianceExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

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
     │      │   alg ┼ FullMoment()
     │      │     w ┴ nothing
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
  - [`StandardDeviationExpectedReturns`](@ref): The sibling that returns the square root of this result.
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

This method returns the variance of each asset in `X`, which the covariance estimator `me.ce` estimates. The result is in the squared units of the returns in `X`. It is the elementwise square of what [`mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref) returns on the same data and the same `me.ce`.

The method reads the diagonal of the matrix that `me.ce` returns. A choice inside `me.ce` that changes that diagonal, such as the observation weights, changes the result. The method calls `me.ce` through [`library_covariance_estimator`](@ref). That function wraps an estimator that the library does not own, such as `StatsBase.SimpleCovariance()`, in a [`GeneralCovariance`](@ref), because `StatsBase` defines no `var` for such an estimator.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_j &= \\hat{\\sigma}_j^2 = \\hat{\\mathbf{\\Sigma}}_{jj}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mu}_j``: The value that takes the place of the expected return of asset ``j``.
  - ``\\hat{\\sigma}_j^2``: The variance of asset ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}``: The covariance matrix that `me.ce` estimates.
  - ``\\hat{\\mathbf{\\Sigma}}_{jj}``: The ``j``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `me`: The variance expected returns estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method passes to the covariance estimator.

# Validation

  - $(val_dict[:dims])

# Returns

  - `mu::Matrix{<:Number}`: The variances, a `(1, N)` matrix if `dims == 1` or an `(N, 1)` matrix if `dims == 2`.

# Related

  - [`VarianceExpectedReturns`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
  - [`mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): Returns the square root of this result.
"""
function Statistics.mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    return Statistics.var(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
end

export StandardDeviationExpectedReturns, VarianceExpectedReturns
