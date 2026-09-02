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
"""
    variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the point-in-time variance series, one row per observation.

Row `t` holds the variance of each asset estimated from observations `1` to `t` alone, so a caller that reads row `t - 1` holds a variance that observation `t` did not enter. That is what a weight paired with observation `t` needs: a variance carrying the date-`t` squared residual would down-weight an asset for its own shock, and would correlate the weights with the residuals.

This method is the fallback that gives every [`AbstractCovarianceEstimator`](@ref) a series, so a member needs no method of its own to answer correctly. It refits on an expanding window, at a cost of one fit per observation. A member whose estimate is a recursion overrides it with a single forward pass.

# Mathematical definition

```math
\\begin{align}
\\mathbf{V}_{ti} &= \\hat{\\sigma}_i^2\\left(\\mathbf{X}_{1:t}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}_{ti}``: Variance of asset ``i`` after observation ``t``.
  - ``\\hat{\\sigma}_i^2``: Variance of asset ``i``, as `ce` estimates it.
  - ``\\mathbf{X}_{1:t}``: First ``t`` observations of the data matrix.
  - $(math_dict[:T])

# Algorithm

 1. Orient `X` so that the observations lie on the rows.
 2. For each observation `t`, call `Statistics.var(ce, X[1:t, :]; dims = 1, kwargs...)` and write the result into row `t`.
 3. Return the series, transposed when `dims == 2`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the variance estimator. A keyword carrying one entry per observation is passed unsliced, so an estimator that takes one must override this method.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if `dims == 2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> PortfolioOptimisers.variance_series(SimpleVariance(), X)
3×2 Matrix{Float64}:
 NaN       NaN
   0.0002    0.0002
   0.0001    0.0001
```

Row `1` is a fit on a single observation, so an estimator that needs two returns `NaN` there.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1,
                         kwargs...)
    X = dims_oriented(dims, X)
    v1 = vec(Statistics.var(ce, view(X, 1:1, :); dims = 1, kwargs...))
    val = Matrix{eltype(v1)}(undef, size(X, 1), length(v1))
    val[1, :] = v1
    for t in 2:size(X, 1)
        val[t, :] = vec(Statistics.var(ce, view(X, 1:t, :); dims = 1, kwargs...))
    end
    return isone(dims) ? val : permutedims(val)
end
