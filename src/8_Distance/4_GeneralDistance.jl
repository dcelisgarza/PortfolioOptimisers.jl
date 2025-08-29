"""
    struct GeneralDistance{T1, T2} <: AbstractDistanceEstimator
        power::T1
        alg::T2
    end

A flexible distance estimator that generalizes distance transformations for portfolio optimization.

`GeneralDistance` allows you to raise the base correlation or distance matrix to an arbitrary integer power before applying a distance transformation. This enables the construction of custom distance metrics, such as higher-order or nonlinear distances, by combining a power transformation with any supported base distance algorithm (e.g., `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, etc.).

# Fields

  - `power`: The integer power to which the base correlation or distance matrix is raised.
  - `alg`: The base distance algorithm to use (e.g., `SimpleDistance()`).

# Constructor

    GeneralDistance(; power::Integer = 1, alg::AbstractDistanceAlgorithm = SimpleDistance())

# Related

  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
struct GeneralDistance{T1, T2} <: AbstractDistanceEstimator
    power::T1
    alg::T2
end
"""
    GeneralDistance(; power::Integer = 1,
                     alg::AbstractDistanceAlgorithm = SimpleDistance())

Construct a [`GeneralDistance`](@ref) estimator with the specified power and base distance algorithm.

This constructor creates a `GeneralDistance` object that will raise the base correlation or distance matrix to the given integer power before applying the specified distance transformation.

# Arguments

  - `power`: The integer power to which the base correlation or distance matrix is raised.
  - `alg`: The base distance algorithm to use.

# Returns

  - `GeneralDistance`: A configured general distance estimator.

# Validation

  - Asserts that `power` is at least 1.

# Examples

```jldoctest
julia> GeneralDistance()
GeneralDistance
  power | Int64: 1
    alg | SimpleDistance()
```

# Related

  - [`GeneralDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function GeneralDistance(; power::Integer = 1,
                         alg::AbstractDistanceAlgorithm = SimpleDistance())
    @argcheck(power >= one(power))
    return GeneralDistance(power, alg)
end

"""
    distance(de::GeneralDistance{<:Any, <:SimpleDistance},
             ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered simple distance matrix from data using a covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises it to the power specified in `de.power`, and then applies the simple distance transformation. The result is a matrix of pairwise distances suitable for clustering or portfolio construction.

# Arguments

  - `de`: General distance estimator with a power and `SimpleDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances.

# Details

  - The correlation matrix is raised to the specified integer power.
  - The distance is computed as `sqrt(clamp!((1 - rho) * scale, 0, 1))`, where `scale` is `0.5` if the power is odd, otherwise `1.0`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
"""
    distance(de::GeneralDistance{<:Any, <:SimpleDistance},
             rho::AbstractMatrix, args...; kwargs...)

Compute a powered simple distance matrix from a correlation or covariance matrix.

This method takes a correlation or covariance matrix `rho`, raises it to the power specified in `de.power`, and applies the simple distance transformation. If `rho` is a covariance matrix, it is first converted to a correlation matrix.

# Arguments

  - `de`: General distance estimator with a power and `SimpleDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances.

# Details

  - If the diagonal of `rho` is not all ones, it is assumed to be a covariance matrix and is converted to a correlation matrix.
  - The distance is computed as `sqrt(clamp!((1 - rho^power) * scale, 0, 1))`, where `scale` is `0.5` if the power is odd, otherwise `1.0`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:SimpleDistance},
                 ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered correlation matrix and the corresponding simple distance matrix from data.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises it to the power specified in `de.power`, and returns both the powered correlation matrix and the corresponding simple distance matrix.

# Arguments

  - `de`: General distance estimator with a power and `SimpleDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered correlation matrix and distance matrix.

# Details

  - The distance is computed as `sqrt(clamp!((1 - rho) * scale, 0, 1))`, where `scale` is `0.5` if the power is odd, otherwise `1.0`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end

"""
    distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
             ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered simple absolute distance matrix from data using a covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, takes the absolute value, raises it to the power specified in `de.power`, and then applies the simple absolute distance transformation. The result is a matrix of pairwise distances suitable for clustering or portfolio construction.

# Arguments

  - `de`: General distance estimator with a power and `SimpleAbsoluteDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances.

# Details

  - The correlation matrix is computed, then the absolute value is taken and raised to the specified integer power.
  - The distance is computed as `sqrt(clamp!(1 - |rho|^power, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = abs.(cor(ce, X; dims = dims), kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
"""
    distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
             rho::AbstractMatrix, args...; kwargs...)

Compute a powered simple absolute distance matrix from a correlation or covariance matrix.

This method takes a correlation or covariance matrix `rho`, takes the absolute value, raises it to the power specified in `de.power`, and applies the simple absolute distance transformation. If `rho` is a covariance matrix, it is first converted to a correlation matrix.

# Arguments

  - `de`: General distance estimator with a power and `SimpleAbsoluteDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances.

# Details

  - If the diagonal of `rho` is not all ones, it is assumed to be a covariance matrix and is converted to a correlation matrix.
  - The distance is computed as `sqrt(clamp!(1 - |rho|^power, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho) .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                 ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered absolute correlation matrix and the corresponding simple absolute distance matrix from data.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, takes the absolute value, raises it to the power specified in `de.power`, and returns both the powered absolute correlation matrix and the corresponding simple absolute distance matrix.

# Arguments

  - `de`: General distance estimator with a power and `SimpleAbsoluteDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered absolute correlation matrix and distance matrix.

# Details

  - The distance is computed as `sqrt(clamp!(1 - |rho|^power, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = abs.(cor(ce, X; dims = dims), kwargs...) .^ de.power
    return rho, sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end

"""
    distance(de::GeneralDistance{<:Any, <:LogDistance},
             ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered log-distance matrix from data using a covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises the absolute value to the power specified in `de.power`, and then applies the negative logarithm transformation. The result is a matrix of pairwise log-distances.

# Arguments

  - `de`: General distance estimator with a power and `LogDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Details

  - The correlation matrix is computed, absolute value is taken, raised to the specified power, and the negative logarithm is applied: `-log(|rho|^power)`.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...)) .^ de.power
    return -log.(rho)
end
"""
    distance(de::GeneralDistance{<:Any, <:LogDistance},
             ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered log-distance matrix from data using an LTD covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided LTD covariance estimator `ce`, raises it to the power specified in `de.power`, and then applies the negative logarithm transformation. The result is a matrix of pairwise log-distances.

# Arguments

  - `de`: General distance estimator with a power and `LogDistance` algorithm.
  - `ce`: LTD covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Details

  - The correlation matrix is computed, raised to the specified power, and the negative logarithm is applied: `-log(rho^power)`.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return -log.(rho)
end
"""
    distance(de::GeneralDistance{<:Any, <:LogDistance},
             rho::AbstractMatrix, args...; kwargs...)

Compute a powered log-distance matrix from a correlation or covariance matrix.

This method takes a correlation or covariance matrix `rho`, converts to a correlation matrix if needed, takes the absolute value, raises it to the power specified in `de.power`, and applies the negative logarithm transformation.

# Arguments

  - `de`: General distance estimator with a power and `LogDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Details

  - If the diagonal of `rho` is not all ones, it is assumed to be a covariance matrix and is converted to a correlation matrix.
  - The distance is computed as `-log(|rho|^power)`.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:LogDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho) .^ de.power)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                 ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered absolute correlation matrix and the corresponding log-distance matrix from data.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, takes the absolute value, raises it to the power specified in `de.power`, and returns both the powered absolute correlation matrix and the corresponding log-distance matrix.

# Arguments

  - `de`: General distance estimator with a power and `LogDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered absolute correlation matrix and log-distance matrix.

# Details

  - The distance is computed as `-log(rho)` where `rho = |cor(X)|^power`.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = abs.(cor(ce, X; dims = dims, kwargs...)) .^ de.power
    return rho, -log.(rho)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                 ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered correlation matrix and the corresponding log-distance matrix from data using an LTD covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided LTD covariance estimator `ce`, raises it to the power specified in `de.power`, and returns both the powered correlation matrix and the corresponding log-distance matrix.

# Arguments

  - `de`: General distance estimator with a power and `LogDistance` algorithm.
  - `ce`: LTD covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered correlation matrix and log-distance matrix.

# Details

  - The distance is computed as `-log(rho)` where `rho = cor(X)^power`.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, -log.(rho)
end

"""
    distance(de::GeneralDistance{<:Any, <:VariationInfoDistance},
             ::Any, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered variation of information (VI) distance matrix from a data matrix.

This method computes the VI distance matrix for the input data matrix `X` using the configuration in the `VariationInfoDistance` algorithm, then raises the result to the power specified in `de.power`. The VI distance is a measure of dissimilarity between random variables based on their mutual information, estimated via histogram binning.

# Arguments

  - `de`: General distance estimator with a power and `VariationInfoDistance` algorithm.
  - `::Any`: Placeholder for compatibility; ignored.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `dist::Matrix{<:Real}`: Matrix of powered pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the `VariationInfoDistance` algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.
  - The resulting VI distance matrix is raised to the specified power.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:VariationInfoDistance}, ::Any,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:VariationInfoDistance},
                 ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered correlation matrix and the powered variation of information (VI) distance matrix from data.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises it to the power specified in `de.power`, and returns both the powered correlation matrix and the corresponding powered VI distance matrix. The VI distance is computed using the configuration in the `VariationInfoDistance` algorithm and is also raised to the specified power.

# Arguments

  - `de`: General distance estimator with a power and `VariationInfoDistance` algorithm.
  - `ce`: Covariance estimator (used to compute the correlation matrix).
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered correlation matrix and powered VI distance matrix.

# Details

  - The number of bins and normalisation are taken from the `VariationInfoDistance` algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.
  - Both the correlation matrix and the VI distance matrix are raised to the specified power.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end

"""
    distance(de::GeneralDistance{<:Any, <:CorrelationDistance},
             ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute a powered correlation distance matrix from data using a covariance estimator.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises it to the power specified in `de.power`, and then applies the correlation distance transformation. The result is a matrix of pairwise correlation distances.

# Arguments

  - `de`: General distance estimator with a power and `CorrelationDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise correlation distances.

# Details

  - The correlation matrix is raised to the specified integer power.
  - The distance is computed as `sqrt(clamp!(1 - rho, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
"""
    distance(de::GeneralDistance{<:Any, <:CorrelationDistance},
             rho::AbstractMatrix, args...; kwargs...)

Compute a powered correlation distance matrix from a correlation or covariance matrix.

This method takes a correlation or covariance matrix `rho`, raises it to the power specified in `de.power`, and applies the correlation distance transformation. If `rho` is a covariance matrix, it is first converted to a correlation matrix.

# Arguments

  - `de`: General distance estimator with a power and `CorrelationDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise correlation distances.

# Details

  - If the diagonal of `rho` is not all ones, it is assumed to be a covariance matrix and is converted to a correlation matrix.
  - The distance is computed as `sqrt(clamp!(1 - rho^power, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    assert_matrix_issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CorrelationDistance},
                 ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the powered correlation matrix and the corresponding correlation distance matrix from data.

This method computes the correlation matrix from the data matrix `X` using the provided covariance estimator `ce`, raises it to the power specified in `de.power`, and returns both the powered correlation matrix and the corresponding correlation distance matrix.

# Arguments

  - `de`: General distance estimator with a power and `CorrelationDistance` algorithm.
  - `ce`: Covariance estimator to use for correlation computation.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation estimator.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of powered correlation matrix and correlation distance matrix.

# Details

  - The distance is computed as `sqrt(clamp!(1 - rho, 0, 1))`.

# Related

  - [`GeneralDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CorrelationDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end

"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the provided `MutualInfoCovariance` estimator.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matri{<Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power,
                                    alg = VariationInfoDistance(; bins = ce.bins,
                                                                normalise = ce.normalise)),
                    ce, X; dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
             ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a wrapped mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the wrapped `MutualInfoCovariance` estimator.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Wrapped mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power,
                                    alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                                normalise = ce.ce.normalise)),
                    ce, X; dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
             ce::Union{<:LTDCovariance,
                       <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a lower tail dependence covariance estimator and data matrix.

This method dispatches to the [`LogDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: LTD covariance estimator or a wrapped LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`LTDCovariance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = LogDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
             ce::Union{<:DistanceCovariance,
                       <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a distance covariance estimator and data matrix.

This method dispatches to the [`CorrelationDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Distance covariance estimator or a wrapped distance covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::Union{<:DistanceCovariance,
                            <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
             ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
             kwargs...)

Compute the canonical distance matrix using a generic covariance estimator and data matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, rho::AbstractMatrix,
             args...; kwargs...)

Compute the canonical distance matrix from a correlation or covariance matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), rho;
                    kwargs...)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                 ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the provided `MutualInfoCovariance` estimator.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power,
                                        alg = VariationInfoDistance(; bins = ce.bins,
                                                                    normalise = ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                 ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a wrapped mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the wrapped `MutualInfoCovariance` estimator.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Wrapped mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power,
                                        alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                                    normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                 ce::Union{<:LTDCovariance,
                           <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a lower tail dependence covariance estimator and data matrix.

This method dispatches to the [`LogDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: LTD covariance estimator or a wrapped LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`LTDCovariance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = LogDistance()), ce, X;
                        dims = dims, kwargs...)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                 ce::Union{<:DistanceCovariance,
                           <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a distance covariance estimator and data matrix.

This method dispatches to the [`CorrelationDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Distance covariance estimator or a wrapped distance covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::Union{<:DistanceCovariance,
                                <:PortfolioOptimisersCovariance{<:DistanceCovariance,
                                                                <:Any}}, X::AbstractMatrix;
                      dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = CorrelationDistance()),
                        ce, X; dims = dims, kwargs...)
end
"""
    cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                 ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                 kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a generic covariance estimator and data matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `de`: General distance estimator with a power and `CanonicalDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = SimpleDistance()), ce, X;
                        dims = dims, kwargs...)
end

export GeneralDistance
