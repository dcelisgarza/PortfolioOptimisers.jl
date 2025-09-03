"""
    struct Distance{T1} <: AbstractDistanceEstimator
        alg::T1
    end

Distance estimator for portfolio optimization.

`Distance` is a flexible container type for configuring and applying distance-based estimators in PortfolioOptimisers.jl. It encapsulates a distance algorithm (such as `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `CanonicalDistance`, or `VariationInfoDistance`) and provides a unified interface for computing distance matrices and related quantities.

# Fields

  - `alg`: The distance algorithm to use (e.g., `SimpleDistance()`).

# Constructor

    Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
"""
struct Distance{T1} <: AbstractDistanceEstimator
    alg::T1
end
"""
    Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())

Construct a [`Distance`](@ref) estimator with the specified distance algorithm.

This constructor creates a `Distance` object using the provided distance algorithm.

# Arguments

  - `alg`: The distance algorithm to use.

# Returns

  - `Distance`: A configured distance estimator.

# Examples

```jldoctest
julia> Distance()
Distance
  alg | SimpleDistance()
```

# Related

  - [`Distance`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`distance`](@ref)
"""
function Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())
    return Distance(alg)
end

"""
    distance(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the simple distance matrix from a covariance estimator and data matrix.

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, then transforms it into a distance matrix using the formula `distance = sqrt(clamp((1 - ρ) / 2, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleDistance}`: Distance estimator with `SimpleDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise Euclidean distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:SimpleDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
"""
    distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)

Compute the simple distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is first converted to a correlation matrix. The distance matrix is then computed using the formula `distance = sqrt(clamp((1 - ρ) / 2, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleDistance}`: Distance estimator with `SimpleDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise Euclidean distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(
        clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)), one(eltype(rho))),
    )
end
"""
    cor_and_dist(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the simple distance matrix from a covariance estimator and data matrix.

This method returns a tuple containing the correlation matrix and the corresponding distance matrix, where the distance is computed as `sqrt(clamp((1 - ρ) / 2, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleDistance}`: Distance estimator with `SimpleDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and distance matrix.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:SimpleDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
    sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end

"""
    distance(::Distance{<:SimpleAbsoluteDistance}, ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the simple absolute distance matrix from a covariance estimator and data matrix.

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, takes the absolute value of the correlation coefficients, and transforms them into a distance matrix using the formula `distance = sqrt(clamp(1 - |ρ|, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleAbsoluteDistance}`: Distance estimator with `SimpleAbsoluteDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise absolute distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:SimpleAbsoluteDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
"""
    distance(::Distance{<:SimpleAbsoluteDistance}, rho::AbstractMatrix, args...; kwargs...)

Compute the simple absolute distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is first converted to a correlation matrix. The distance matrix is then computed using the formula `distance = sqrt(clamp(1 - |ρ|, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleAbsoluteDistance}`: Distance estimator with `SimpleAbsoluteDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise absolute distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using [`StatsBase.cov2cor`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.cov2cor).

# Related

  - [`Distance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:SimpleAbsoluteDistance},
    rho::AbstractMatrix,
    args...;
    kwargs...,
)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- abs.(rho), zero(eltype(rho)), one(eltype(rho))))
end
"""
    cor_and_dist(::Distance{<:SimpleAbsoluteDistance},
                 ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                 kwargs...)

Compute both the absolute correlation matrix and the simple absolute distance matrix from a covariance estimator and data matrix.

This method returns a tuple containing the absolute correlation matrix and the corresponding distance matrix, where the distance is computed as `sqrt(clamp(1 - |ρ|, 0, 1))`, with `ρ` the correlation coefficient.

# Arguments

  - `::Distance{<:SimpleAbsoluteDistance}`: Distance estimator with `SimpleAbsoluteDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of absolute correlation matrix and distance matrix.

# Related

  - [`Distance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:SimpleAbsoluteDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return rho, sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end

"""
    distance(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the log-distance matrix from a covariance estimator and data matrix.

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, takes the absolute value of the correlation coefficients, and transforms them into a distance matrix using the formula `distance = -log(|ρ|)`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with `LogDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:LogDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return -log.(rho)
end
"""
    distance(::Distance{<:LogDistance},
             ce::Union{<:LTDCovariance,
                       <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

This method computes the correlation matrix using the provided LTD covariance estimator `ce` and data matrix `X`, then transforms it into a distance matrix using the formula `distance = -log(ρ)`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with `LogDistance` algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:LogDistance},
    ce::Union{<:LTDCovariance,<:PortfolioOptimisersCovariance{<:LTDCovariance,<:Any}},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end
"""
    distance(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)

Compute the log-distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is first converted to a correlation matrix. The distance matrix is then computed using the formula `distance = -log(|ρ|)`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with `LogDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(abs.(rho))
end
"""
    cor_and_dist(::Distance{<:LogDistance},
                 ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                 kwargs...)

Compute both the absolute correlation matrix and the log-distance matrix from a covariance estimator and data matrix.

This method returns a tuple containing the absolute correlation matrix and the corresponding log-distance matrix, where the distance is computed as `-log(|ρ|)`, with `ρ` the correlation coefficient.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with `LogDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of absolute correlation matrix and log-distance matrix.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:LogDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = abs.(cor(ce, X; dims = dims, kwargs...))
    return rho, -log.(rho)
end
"""
    cor_and_dist(::Distance{<:LogDistance},
                 ce::Union{<:LTDCovariance,
                           <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

This method returns a tuple containing the correlation matrix and the corresponding log-distance matrix, where the distance is computed as `-log(ρ)`, with `ρ` the correlation coefficient.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with `LogDistance` algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and log-distance matrix.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:LogDistance},
    ce::Union{<:LTDCovariance,<:PortfolioOptimisersCovariance{<:LTDCovariance,<:Any}},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho)
end

"""
    distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix;
             dims::Int = 1, kwargs...)

Compute the variation of information (VI) distance matrix from a data matrix.

This method computes the VI distance matrix for the input data matrix `X` using the configuration in the `VariationInfoDistance` algorithm. The VI distance is a measure of dissimilarity between random variables based on their mutual information, estimated via histogram binning.

# Arguments

  - `de::Distance{<:VariationInfoDistance}`: Distance estimator with `VariationInfoDistance` algorithm.
  - `::Any`: Placeholder for compatibility; ignored.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the `VariationInfoDistance` algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
function distance(
    de::Distance{<:VariationInfoDistance},
    ::Any,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
"""
    cor_and_dist(de::Distance{<:VariationInfoDistance},
                 ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                 kwargs...)

Compute both the correlation matrix and the variation of information (VI) distance matrix from a covariance estimator and data matrix.

This method returns a tuple containing the correlation matrix and the corresponding VI distance matrix, where the VI distance is a measure of dissimilarity between random variables based on their mutual information, estimated via histogram binning.

# Arguments

  - `de::Distance{<:VariationInfoDistance}`: Distance estimator with `VariationInfoDistance` algorithm.
  - `ce`: Covariance estimator (used to compute the correlation matrix).
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and VI distance matrix.

# Details

  - The number of bins and normalisation are taken from the `VariationInfoDistance` algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    de::Distance{<:VariationInfoDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...)
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise)
end

"""
    distance(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the correlation distance matrix from a covariance estimator and data matrix.

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, then transforms it into a distance matrix using the formula `distance = sqrt(clamp(1 - ρ, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:CorrelationDistance}`: Distance estimator with `CorrelationDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise correlation distances.

# Related

  - [`Distance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:CorrelationDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
"""
    distance(::Distance{<:CorrelationDistance}, rho::AbstractMatrix, args...; kwargs...)

Compute the correlation distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is first converted to a correlation matrix. The distance matrix is then computed using the formula `distance = sqrt(clamp(1 - ρ, 0, 1))`, where `ρ` is the correlation coefficient.

# Arguments

  - `::Distance{<:CorrelationDistance}`: Distance estimator with `CorrelationDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise correlation distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.

# Related

  - [`Distance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(
    ::Distance{<:CorrelationDistance},
    rho::AbstractMatrix,
    args...;
    kwargs...,
)
    assert_matrix_issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
"""
    cor_and_dist(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the correlation distance matrix from a covariance estimator and data matrix.

This method returns a tuple containing the correlation matrix and the corresponding correlation distance matrix, where the distance is computed as `sqrt(clamp(1 - ρ, 0, 1))`, with `ρ` the correlation coefficient.

# Arguments

  - `::Distance{<:CorrelationDistance}`: Distance estimator with `CorrelationDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and correlation distance matrix.

# Related

  - [`Distance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CorrelationDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end

"""
    distance(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the provided `MutualInfoCovariance` estimator.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce::MutualInfoCovariance`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function distance(
    ::Distance{<:CanonicalDistance},
    ce::MutualInfoCovariance,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return distance(
        Distance(; alg = VariationInfoDistance(; bins = ce.bins, normalise = ce.normalise)),
        ce,
        X;
        dims = dims,
        kwargs...,
    )
end
"""
    distance(::Distance{<:CanonicalDistance},
             ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a wrapped mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the wrapped `MutualInfoCovariance` estimator.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Wrapped mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function distance(
    ::Distance{<:CanonicalDistance},
    ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance,<:Any},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return distance(
        Distance(;
            alg = VariationInfoDistance(; bins = ce.ce.bins, normalise = ce.ce.normalise),
        ),
        ce,
        X;
        dims = dims,
        kwargs...,
    )
end
"""
    distance(::Distance{<:CanonicalDistance},
             ce::Union{<:LTDCovariance,
                       <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a lower tail dependence covariance estimator and data matrix.

This method dispatches to the [`LogDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: LTD covariance estimator or a wrapped LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`LTDCovariance`](@ref)
"""
function distance(
    ::Distance{<:CanonicalDistance},
    ce::Union{<:LTDCovariance,<:PortfolioOptimisersCovariance{<:LTDCovariance,<:Any}},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return distance(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
"""
    distance(::Distance{<:CanonicalDistance},
             ce::Union{<:DistanceCovariance,
                       <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a distance covariance estimator and data matrix.

This method dispatches to the [`CorrelationDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Distance covariance estimator or a wrapped distance covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function distance(
    ::Distance{<:CanonicalDistance},
    ce::Union{
        <:DistanceCovariance,
        <:PortfolioOptimisersCovariance{<:DistanceCovariance,<:Any},
    },
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return distance(Distance(; alg = CorrelationDistance()), ce, X; dims = dims, kwargs...)
end
"""
    distance(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using a generic covariance estimator and data matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function distance(
    ::Distance{<:CanonicalDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return distance(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end
"""
    distance(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...; kwargs...)

Compute the canonical distance matrix from a correlation or covariance matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function distance(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...; kwargs...)
    return distance(Distance(; alg = SimpleDistance()), rho; kwargs...)
end
"""
    cor_and_dist(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the provided `MutualInfoCovariance` estimator.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CanonicalDistance},
    ce::MutualInfoCovariance,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return cor_and_dist(
        Distance(; alg = VariationInfoDistance(; bins = ce.bins, normalise = ce.normalise)),
        ce,
        X;
        dims = dims,
        kwargs...,
    )
end
"""
    cor_and_dist(::Distance{<:CanonicalDistance},
                 ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a wrapped mutual information covariance estimator and data matrix.

This method dispatches to the [`VariationInfoDistance`](@ref) algorithm, using the number of bins and normalisation from the wrapped `MutualInfoCovariance` estimator.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Wrapped mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CanonicalDistance},
    ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance,<:Any},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return cor_and_dist(
        Distance(;
            alg = VariationInfoDistance(; bins = ce.ce.bins, normalise = ce.ce.normalise),
        ),
        ce,
        X;
        dims = dims,
        kwargs...,
    )
end
"""
    cor_and_dist(::Distance{<:CanonicalDistance},
                 ce::Union{<:LTDCovariance,
                           <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a lower tail dependence covariance estimator and data matrix.

This method dispatches to the [`LogDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: LTD covariance estimator or a wrapped LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`LTDCovariance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CanonicalDistance},
    ce::Union{<:LTDCovariance,<:PortfolioOptimisersCovariance{<:LTDCovariance,<:Any}},
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return cor_and_dist(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
"""
    cor_and_dist(::Distance{<:CanonicalDistance},
                 ce::Union{<:DistanceCovariance,
                           <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a distance covariance estimator and data matrix.

This method dispatches to the [`CorrelationDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Distance covariance estimator or a wrapped distance covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`Distance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CanonicalDistance},
    ce::Union{
        <:DistanceCovariance,
        <:PortfolioOptimisersCovariance{<:DistanceCovariance,<:Any},
    },
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return cor_and_dist(
        Distance(; alg = CorrelationDistance()),
        ce,
        X;
        dims = dims,
        kwargs...,
    )
end
"""
    cor_and_dist(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute both the correlation matrix and the canonical distance matrix using a generic covariance estimator and data matrix.

This method dispatches to the [`SimpleDistance`](@ref) algorithm.

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator with `CanonicalDistance` algorithm.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and canonical distance matrix.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
"""
function cor_and_dist(
    ::Distance{<:CanonicalDistance},
    ce::StatsBase.CovarianceEstimator,
    X::AbstractMatrix;
    dims::Int = 1,
    kwargs...,
)
    return cor_and_dist(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end

export Distance
