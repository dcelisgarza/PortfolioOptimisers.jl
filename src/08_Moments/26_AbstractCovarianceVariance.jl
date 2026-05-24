"""
    Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the variance vector from the diagonal of the covariance matrix.

This method extracts the diagonal of the covariance matrix returned by `Statistics.cov(ce, X; dims = dims, kwargs...)` and reshapes it into a row or column vector depending on `dims`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i^2 &= \\hat{\\mathbf{\\Sigma}}_{ii}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_i^2``: Variance of asset ``i``.
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - ``\\hat{\\mathbf{\\Sigma}}_{ii}``: ``i``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `ce`: Covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `var::Matrix{<:Number}`: Variance vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    val = LinearAlgebra.diag(Statistics.cov(ce, X; dims = dims, kwargs...))
    return isone(dims) ? reshape(val, 1, length(val)) : reshape(val, length(val), 1)
end
"""
    Statistics.std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the standard deviation vector from the diagonal of the covariance matrix.

This method extracts the diagonal of the covariance matrix returned by `Statistics.cov(ce, X; dims = dims, kwargs...)`, takes the element-wise square root, and reshapes it into a row or column vector depending on `dims`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i &= \\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_i``: Standard deviation of asset ``i``.
  - ``\\hat{\\mathbf{\\Sigma}}``: Estimated covariance matrix.
  - ``\\hat{\\mathbf{\\Sigma}}_{ii}``: ``i``-th diagonal element of ``\\hat{\\mathbf{\\Sigma}}``.

# Arguments

  - `ce`: Covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `sd::Matrix{<:Number}`: Standard deviation vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    val = sqrt.(LinearAlgebra.diag(Statistics.cov(ce, X; dims = dims, kwargs...)))
    return isone(dims) ? reshape(val, 1, length(val)) : reshape(val, length(val), 1)
end
