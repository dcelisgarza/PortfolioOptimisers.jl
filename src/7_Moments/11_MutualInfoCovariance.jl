"""
    struct MutualInfoCovariance{T1 <: AbstractVarianceEstimator,
                                T2 <: Union{<:AbstractBins, <:Integer}, T3 <: Bool} <: AbstractCovarianceEstimator

Covariance estimator based on mutual information.

`MutualInfoCovariance` implements a robust covariance estimator that uses mutual information (MI) to capture both linear and nonlinear dependencies between asset returns. This estimator is particularly useful for identifying complex relationships that are not detected by traditional correlation-based methods. The MI matrix is optionally normalised and then rescaled by marginal standard deviations to produce a covariance matrix.

# Fields

  - `ve::AbstractVarianceEstimator`: Variance estimator used to compute marginal standard deviations.
  - `bins::Union{<:AbstractBins, <:Integer}`: Binning algorithm or fixed number of bins for histogram-based MI estimation.
  - `normalise::Bool`: Whether to normalise the MI matrix.

# Constructor

    MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                          bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                          normalise::Bool = true)

Creates a `MutualInfoCovariance` object with the specified variance estimator, binning strategy, and normalisation option.

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractBins`](@ref)
"""
struct MutualInfoCovariance{T1 <: AbstractVarianceEstimator,
                            T2 <: Union{<:AbstractBins, <:Integer}, T3 <: Bool} <:
       AbstractCovarianceEstimator
    ve::T1
    bins::T2
    normalise::T3
end
"""
    MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                          bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                          normalise::Bool = true)

Construct a [`MutualInfoCovariance`](@ref) estimator for robust covariance or correlation estimation using mutual information.

This constructor creates a `MutualInfoCovariance` object using the specified variance estimator, binning algorithm (or fixed bin count), and normalisation flag. The estimator computes the covariance matrix by combining the mutual information matrix (optionally normalised) with the marginal standard deviations.

# Arguments

  - `ve::AbstractVarianceEstimator`: Variance estimator.
  - `bins::Union{<:AbstractBins, <:Integer}`: Binning algorithm or fixed number of bins for MI estimation.
  - `normalise::Bool`: Whether to normalise the MI matrix.

# Returns

  - `MutualInfoCovariance`: A configured mutual information-based covariance estimator.

# Validation

  - If `bins` is an integer, asserts that `bins > 0`.

# Examples

```jldoctest
julia> ce = MutualInfoCovariance()
MutualInfoCovariance
         ve | SimpleVariance
            |          me | SimpleExpectedReturns
            |             |   w | nothing
            |           w | nothing
            |   corrected | Bool: true
       bins | HacineGharbiRavier()
  normalise | Bool: true
```

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractBins`](@ref)
"""
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                              normalise::Bool = true)
    if isa(bins, Integer)
        @smart_assert(bins > zero(bins))
    end
    return MutualInfoCovariance{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins,
                                                                             normalise)
end
function factory(ce::MutualInfoCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return MutualInfoCovariance(; ve = factory(ce.ve, w), bins = ce.bins,
                                normalise = ce.normalise)
end

"""
    cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the mutual information (MI) correlation matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information correlation matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI correlation captures both linear and nonlinear dependencies between asset returns, making it robust to complex relationships that may not be detected by traditional correlation measures.

# Arguments

  - `ce::MutualInfoCovariance`: Mutual information-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{Float64}`: Symmetric matrix of mutual information-based correlation coefficients.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Details

If `dims == 2`, the input matrix is transposed before computation. The correlation matrix is computed using [`mutual_info`](@ref), which estimates the pairwise mutual information between all variables, optionally normalised.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end

"""
    cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the mutual information (MI) covariance matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information covariance matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI covariance matrix is obtained by rescaling the MI correlation matrix by the marginal standard deviations, as estimated by the variance estimator in `ce`.

# Arguments

  - `ce::MutualInfoCovariance`: Mutual information-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{Float64}`: Symmetric matrix of mutual information-based covariances.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Details

If `dims == 2`, the input matrix is transposed before computation. The covariance matrix is computed as the elementwise product of the MI correlation matrix and the outer product of marginal standard deviations.

# Examples

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cor(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return mutual_info(X, ce.bins, ce.normalise) ⊙ (std_vec ⊗ std_vec)
end

export MutualInfoCovariance
