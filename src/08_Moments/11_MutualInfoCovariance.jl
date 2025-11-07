"""
    struct MutualInfoCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
        ve::T1
        bins::T2
        normalise::T3
    end

Covariance estimator based on mutual information.

`MutualInfoCovariance` implements a robust covariance estimator that uses mutual information (MI) to capture both linear and nonlinear dependencies between asset returns. This estimator is particularly useful for identifying complex relationships that are not detected by traditional correlation-based methods. The MI matrix is optionally normalised and then rescaled by marginal standard deviations to produce a covariance matrix.

# Fields

  - `ve`: Variance estimator used to compute marginal standard deviations.
  - `bins`: Binning algorithm or fixed number of bins for histogram-based MI estimation.
  - `normalise`: Whether to normalise the MI matrix.

# Constructor

    MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                         bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                         normalise::Bool = true)

Keyword arguments correspond to the fields above.

## Validation

  - If `bins` is an integer, `bins > 0`.

# Examples

```jldoctest
julia> MutualInfoCovariance()
MutualInfoCovariance
         ve ┼ SimpleVariance
            │          me ┼ SimpleExpectedReturns
            │             │   w ┴ nothing
            │           w ┼ nothing
            │   corrected ┴ Bool: true
       bins ┼ HacineGharbiRavier()
  normalise ┴ Bool: true
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractBins`](@ref)
"""
struct MutualInfoCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
    function MutualInfoCovariance(ve::AbstractVarianceEstimator,
                                  bins::Union{<:AbstractBins, <:Integer}, normalise::Bool)
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins)
        end
        return new{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins, normalise)
    end
end
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                              normalise::Bool = true)
    return MutualInfoCovariance(ve, bins, normalise)
end
function factory(ce::MutualInfoCovariance, w::WeightsType = nothing)
    return MutualInfoCovariance(; ve = factory(ce.ve, w), bins = ce.bins,
                                normalise = ce.normalise)
end
"""
    cor(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)

Compute the mutual information (MI) correlation matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information correlation matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI correlation captures both linear and nonlinear dependencies between asset returns, making it robust to complex relationships that may not be detected by traditional correlation measures.

# Arguments

  - `ce`: Mutual information-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of mutual information-based correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cov(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
"""
    cov(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)

Compute the mutual information (MI) covariance matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information covariance matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI covariance matrix is obtained by rescaling the MI correlation matrix by the marginal standard deviations, as estimated by the variance estimator in `ce`.

# Arguments

  - `ce`: Mutual information-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of mutual information-based covariances.

# Validation

  - `dims` is either `1` or `2`.

# Examples

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cor(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::MutualInfoCovariance, X::NumMat; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return mutual_info(X, ce.bins, ce.normalise) ⊙ (std_vec ⊗ std_vec)
end

export MutualInfoCovariance
