"""
    Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the variance vector from the diagonal of the covariance matrix.

This method is the fallback that gives every [`AbstractCovarianceEstimator`](@ref) a marginal variance, so a caller needs no separate variance estimator to read the diagonal of the matrix the covariance estimator already builds.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i^2 &= \\hat{\\mathbf{\\Sigma}}_{ii}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_i^2``: Variance of asset ``i``.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:Sigma_hat_ii])

# Algorithm

 1. Compute the covariance matrix with `Statistics.cov(ce, X; dims = dims, kwargs...)`.
 2. Read its diagonal into `val`, the variance of each asset.
 3. Reshape `val` to a `1 × N` row vector when `dims == 1`, and to an `N × 1` column vector otherwise.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `var::Matrix{<:Number}`: Variance vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> var(Covariance(), X)
1×2 Matrix{Float64}:
 0.0001  0.0001
```

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

This method is the fallback that gives every [`AbstractCovarianceEstimator`](@ref) a marginal standard deviation, so a caller needs no separate variance estimator to read the diagonal of the matrix the covariance estimator already builds.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i &= \\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma_hat_i])
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:Sigma_hat_ii])

# Algorithm

 1. Compute the covariance matrix with `Statistics.cov(ce, X; dims = dims, kwargs...)`.
 2. Read its diagonal, take the element-wise square root, and store the result in `val`.
 3. Reshape `val` to a `1 × N` row vector when `dims == 1`, and to an `N × 1` column vector otherwise.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `sd::Matrix{<:Number}`: Standard deviation vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> std(Covariance(), X)
1×2 Matrix{Float64}:
 0.01  0.01
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                        kwargs...)
    val = sqrt.(LinearAlgebra.diag(Statistics.cov(ce, X; dims = dims, kwargs...)))
    return isone(dims) ? reshape(val, 1, length(val)) : reshape(val, length(val), 1)
end
