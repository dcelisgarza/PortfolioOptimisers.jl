"""
    Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the variance vector from the diagonal of the covariance matrix.

This method is the fallback that gives a marginal variance to each [`AbstractCovarianceEstimator`](@ref). A caller does not need a separate variance estimator to read the diagonal of the covariance matrix.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i^2 &= \\hat{\\mathbf{\\Sigma}}_{ii}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_i^2``: The variance of asset ``i``.
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
  - `kwargs...`: More keyword arguments for the covariance estimator.

# Returns

  - `var::Matrix{<:Number}`: The variance vector. Its shape is `(1, N)` if `dims == 1`, or `(N, 1)` if `dims == 2`.

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

This method is the fallback that gives a marginal standard deviation to each [`AbstractCovarianceEstimator`](@ref). A caller does not need a separate variance estimator to read the diagonal of the covariance matrix.

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
  - `kwargs...`: More keyword arguments for the covariance estimator.

# Returns

  - `sd::Matrix{<:Number}`: The standard deviation vector. Its shape is `(1, N)` if `dims == 1`, or `(N, 1)` if `dims == 2`.

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
    Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Always throw a `MethodError`. An [`AbstractVarianceEstimator`](@ref) estimates one marginal variance for each asset. It has no structure across assets, so it has no covariance matrix to return.

This method stops a recursion. Without it, `cov` falls through to [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref), which calls `cor`. A variance estimator has no `cor` method of its own, so a generic fallback of `cor` calls `cov` again. The two calls then recurse without end. This method throws before that fallback runs.

Use a covariance estimator, for example [`Covariance`](@ref), for a covariance matrix.

# Examples

```jldoctest
julia> try
           Statistics.cov(SimpleVariance(), [0.01 0.02; 0.03 0.04])
       catch e
           e isa MethodError
       end
true
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`Statistics.cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return throw(MethodError(Statistics.cov, (ve, X)))
end
"""
    Statistics.cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Always throw a `MethodError`. An [`AbstractVarianceEstimator`](@ref) estimates one marginal variance for each asset. It has no structure across assets, so it has no correlation matrix to return.

Without this method, `cor` falls through to a generic fallback that calls [`Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref), and that method throws a `MethodError`. This method throws the same kind of error one call before. The error names `cor`, which is the verb that the caller used.

Use a covariance estimator, for example [`Covariance`](@ref), for a correlation matrix.

# Examples

```jldoctest
julia> try
           Statistics.cor(SimpleVariance(), [0.01 0.02; 0.03 0.04])
       catch e
           e isa MethodError
       end
true
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`Statistics.cov(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return throw(MethodError(Statistics.cor, (ve, X)))
end
"""
    Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the standard deviation vector as the element-wise square root of the variance vector.

This method is the fallback for an [`AbstractVarianceEstimator`](@ref) that defines only [`Statistics.var`](@ref). It computes `std` from `var`, so it does not call [`Statistics.cov`](@ref), which throws for an [`AbstractVarianceEstimator`](@ref). A member that computes its own standard deviation overrides this method. [`SimpleVariance`](@ref) is an example.

# Mathematical definition

```math
\\begin{align}
\\hat{\\sigma}_i &= \\sqrt{\\hat{\\sigma}_i^2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:sigma_hat_i])

# Algorithm

 1. Compute the variance vector with `Statistics.var(ve, X; dims = dims, kwargs...)`.
 2. Take the element-wise square root.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the variance estimator.

# Returns

  - $(ret_dict[:stdvar])

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return sqrt.(Statistics.var(ve, X; dims = dims, kwargs...))
end
"""
    variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute the point-in-time variance series, with one row for each observation.

Row `t` holds the variance of each asset, as estimated from observations `1` to `t` only, so row `t - 1` holds a variance that does not contain observation `t`. A weight for observation `t` needs this variance. A variance that contains the squared residual of date `t` gives less weight to an asset for its own shock, and it correlates the weights with the residuals.

This method is the fallback that gives a series to each [`AbstractCovarianceEstimator`](@ref), so a member does not need a method of its own. It fits the estimator again on an expanding window, which costs one fit for each observation. A member whose estimate is a recursion overrides this method with one forward pass.

# Mathematical definition

```math
\\begin{align}
\\mathbf{V}_{tj} &= \\hat{\\sigma}^2_j\\left(\\mathbf{X}_{1:t}\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:V_tj_series])
  - $(math_dict[:sigma2_hat_j]) Here `ce` estimates it.
  - ``\\mathbf{X}_{1:t}``: The first ``t`` observations of the data matrix.
  - $(math_dict[:T])

# Algorithm

 1. Orient `X` so that the observations lie on the rows.
 2. For each observation `t`, call `Statistics.var(ce, X[1:t, :]; dims = 1, kwargs...)` and write the result into row `t`.
 3. Return the series, transposed when `dims == 2`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the variance estimator. The method does not slice a keyword that has one entry for each observation, so an estimator that takes such a keyword must override this method.

# Validation

  - $(val_dict[:dims])

# Returns

  - `val::Matrix{<:Number}`: The variance series. Its shape is `(T, N)` if `dims == 1`, or `(N, T)` if `dims == 2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> PortfolioOptimisers.variance_series(SimpleVariance(), X)
3×2 Matrix{Float64}:
 NaN       NaN
   0.0002    0.0002
   0.0001    0.0001
```

Row `1` is a fit on one observation. An estimator that needs two observations returns `NaN` in that row.

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
"""
    variance_count(ve::AbstractVarianceEstimator, X::MatNum)
    variance_count(ve::SimpleVariance, X::MatNum)
    variance_count(ve::ExpWeightedVariance, X::MatNum)
    variance_count(ve::RegimeAdjustedExpWeightedVariance, X::MatNum)
    variance_count(ve::WindowedVariance, X::MatNum)

Return the effective count and the divisor of the variance that `ve` measures on each column of `X`.

A consumer that prices the sampling error of a variance needs two facts that only the estimator knows. The first fact is the number of observations that the estimate is worth. The second fact is the divisor of the sum of squares. This function returns both. A prior that writes the idiosyncratic variances of a loadings block calls this function on the residuals. The prior writes the result next to the variances, so that [`ResidualInflation`](@ref) reads it from the block.

The method for [`AbstractVarianceEstimator`](@ref) returns `nothing`, which means that the estimator states no count. `nothing` is not a default count. A consumer that needs a count refuses a block that has none, and its error names the fix.

# Mathematical definition

Let the estimate of asset ``i`` be a weighted sum of squares, ``\\hat{d}_{i} = \\sum_{t} a_{t} e_{t,i}^{2}``, of residuals with variance ``d_{i}``. A scaled chi-squared law with the same mean and the same variance approximates the law of the estimate:

```math
\\begin{align}
\\dfrac{m_{i} \\hat{d}_{i}}{d_{i}} &\\sim \\chi^{2}_{n_{i}}\\,, \\\\
n_{i} &= \\dfrac{\\left(\\sum_{t} a_{t}\\right)^{2}}{\\sum_{t} a_{t}^{2}}\\,, \\\\
m_{i} &= \\dfrac{n_{i}}{\\sum_{t} a_{t}}\\,.
\\end{align}
```

Where:

  - ``n_{i}``: The effective count of the observations of asset ``i``. It is the Kish count of the weights ``a_{t}``.
  - ``m_{i}``: The divisor of the variance of asset ``i`` for that count. The estimate ``\\hat{d}_{i}`` equals the sum of squares of ``n_{i}`` observations of equal weight, divided by ``m_{i}``.
  - ``a_{t}``: The weight of observation ``t`` in the estimate.
  - ``e_{t,i}``: The residual of asset ``i`` at observation ``t``.
  - ``d_{i}``: The true variance of asset ``i``.

On an unweighted sample of ``T`` observations, ``n_{i} = T``. Then ``m_{i}`` is `T - 1` if the estimator corrects the bias, and `T` if it does not. For Gaussian residuals the relation is then exact. For all other weights it is an approximation.

# Algorithm

 1. On [`AbstractVarianceEstimator`](@ref), return `nothing`.
 2. On [`SimpleVariance`](@ref), take the finite rows of each column, and read the count from the observation weights with [`simple_variance_count`](@ref).
 3. On [`ExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedVariance`](@ref), take the Kish count of the exponential weights over the finite rows of each column with [`exp_weighted_variance_count`](@ref). The weights sum to one after the correction of the cold start, so the divisor equals the count. The count ignores the regime multiplier and the lags of a Heteroskedasticity and Autocorrelation Consistent (HAC) adjustment.
 4. On [`WindowedVariance`](@ref), apply the window and the weights of the wrapper with [`windowed_preamble`](@ref). Then return the count of the inner estimator.

# Arguments

  - $(arg_dict[:ve])
  - `X`: The data matrix, of shape `observations × assets`.

# Returns

  - `cnt::Option{<:NamedTuple}`: The named tuple `(; n, m)` of two vectors, with one entry for each column of `X`. `n` is the effective count and `m` is the divisor. The value is `nothing` if the estimator states no count.

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`factor_lift`](@ref)
  - [`ResidualInflation`](@ref)
  - [`Regression`](@ref)
"""
function variance_count(::AbstractVarianceEstimator, ::MatNum)::Nothing
    return nothing
end
"""
    effective_observation_count(ce::StatsBase.CovarianceEstimator, X::MatNum) -> Real
    effective_observation_count(ce) -> Real
    effective_observation_count(ce, state::AbstractPartialFitState) -> Real
    effective_observation_count(ce, state::SampleBufferState) -> Real

Return the number of observations that the covariance of `ce` is worth, as one number for the whole matrix.

The denoising step of [`MatrixProcessing`](@ref) reads the ratio `T / N` of the sample. An unweighted moment reads every row of `X` with the same weight, so `T` is `size(X, 1)`. A weighted moment is worth fewer observations than `X` has rows. A denoiser that reads `size(X, 1)` then gets a ratio that is too large, and it removes too little noise. So [`PortfolioOptimisersCovariance`](@ref) reads `T` from this function.

The form with `X` is the batch fit. Its default is `size(X, 1)`, so an unweighted estimator pays nothing. The form with no data is the fit from a fold. It reads the state of `ce` with [`partial_fit_cache`](@ref), and the form with a state reads the count from it. The default of a state is the folded row count of [`observation_count`](@ref). A [`SampleBufferState`](@ref) holds the observations, so it reads the batch form over the rows of the buffer.

The count covers every row of `X`, the rows with a `NaN` too, as `size(X, 1)` does. It does not count the rows of each asset.

# Interfaces

An estimator that weights its observations adds a method of the batch form. A method of the state form is necessary when the estimator folds into a state that holds no observations.

# Mathematical definition

```math
\\begin{align}
T_{e} &= \\dfrac{\\left(\\sum_{t} w_{t}\\right)^{2}}{\\sum_{t} w_{t}^{2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

The count is ``T`` under equal weights. Frequency weights count replicated observations, so their count is ``\\sum_{t} w_{t}``.

# Algorithm

The method that Julia selects is the algorithm.

 1. On a covariance estimator with no method of its own, return `size(X, 1)`.
 2. On [`Covariance`](@ref), [`GeneralCovariance`](@ref) and [`DistanceCovariance`](@ref), return the count of the observation weights with [`observation_weight_count`](@ref). A [`Covariance`](@ref) with no weights reads the count of its inner estimator, because the inner estimator can hold weights of its own.
 3. On [`ExpWeightedCovariance`](@ref), return the count of [`exp_weighted_covariance_count`](@ref) over the rows of `X`. On [`RegimeAdjustedExpWeightedCovariance`](@ref), do the same with the decay and the centring of its correlation. The denoising step processes the correlation, so it reads the count of the correlation.
 4. On [`WindowedCovariance`](@ref), apply the window and the weights of the wrapper with [`windowed_preamble`](@ref). Then return the count of the inner estimator over the rows that the window keeps.
 5. On [`PortfolioOptimisersCovariance`](@ref), [`CorrelationCovariance`](@ref), [`GeodesicShrinkageCovariance`](@ref) and [`ImpliedVolatility`](@ref), return the count of the inner estimator, because these estimators pass `X` to it without change.
 6. With no data, read the state of `ce`, and return the count of the state. A composite that keeps no state of its own reads its inner estimator, as [`observation_count`](@ref) does. The two exponentially weighted states return the count of step 3 over the folded rows.

# Arguments

  - $(arg_dict[:ce])
  - `X`: The data matrix, of shape `observations × assets`.
  - `state`: The partial-fit state of `ce`.

# Returns

  - `T::Real`: The effective number of observations. It is an `Integer` for an unweighted estimator.

# Related

  - [`sample_matrix_processing!`](@ref)
  - [`observation_count`](@ref)
  - [`variance_count`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
function effective_observation_count(::StatsBase.CovarianceEstimator, X::MatNum)
    return size(X, 1)
end
function effective_observation_count(ce::Union{AbstractEstimator,
                                               StatsBase.CovarianceEstimator})
    return effective_observation_count(ce, partial_fit_cache(ce))
end
function effective_observation_count(::Union{AbstractEstimator,
                                             StatsBase.CovarianceEstimator},
                                     state::AbstractPartialFitState)
    return observation_count(state)
end
function effective_observation_count(ce::Union{AbstractEstimator,
                                               StatsBase.CovarianceEstimator},
                                     state::SampleBufferState)
    return effective_observation_count(ce, sample_buffer(state))
end
"""
    Statistics.cov(ce::AbstractCovarianceEstimator; kwargs...)
    Statistics.cor(ce::AbstractCovarianceEstimator; kwargs...)
    Statistics.var(ve::AbstractVarianceEstimator)
    Statistics.std(ve::AbstractVarianceEstimator)

Reads a moment from the partial-fit state that a covariance or variance estimator holds.

This method is the call with no data for an estimator that has no such call of its own. It reads the state from the `cache` field with [`partial_fit_cache`](@ref). Then it gives the state to the method of the same verb that takes a state. For a [`SampleBufferState`](@ref), which [`Online`](@ref) seeds, that method runs the batch verb over the rows of the buffer. A family that folds exactly has its own call with no data, so this method serves the estimators that fold only through a buffer, for example [`GerberCovariance`](@ref) and [`WindowedCovariance`](@ref).

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ve])
  - `kwargs...`: The keyword arguments for the method that reads the state, for example `strict`.

# Validation

  - The estimator holds a state. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The covariance matrix for `cov`, or the correlation matrix `rho` for `cor`.
  - `sigma::ArrNum`: The variance for `var`, or the standard deviation for `std`, with one entry for each asset.

# Related

  - [`partial_fit_cache`](@ref)
  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.cov(ce::AbstractCovarianceEstimator; kwargs...)
    return Statistics.cov(ce, partial_fit_cache(ce); kwargs...)
end
function Statistics.cor(ce::AbstractCovarianceEstimator; kwargs...)
    return Statistics.cor(ce, partial_fit_cache(ce); kwargs...)
end
function Statistics.var(ve::AbstractVarianceEstimator)
    return Statistics.var(ve, partial_fit_cache(ve))
end
function Statistics.std(ve::AbstractVarianceEstimator)
    return Statistics.std(ve, partial_fit_cache(ve))
end
public effective_observation_count
