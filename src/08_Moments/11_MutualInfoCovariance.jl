"""
$(DocStringExtensions.TYPEDEF)

Covariance estimator based on mutual information.

`MutualInfoCovariance` implements a robust covariance estimator that uses mutual information (MI) to capture both linear and nonlinear dependencies between asset returns. This estimator is particularly useful for identifying complex relationships that are not detected by traditional correlation-based methods. The MI matrix is optionally normalised and then rescaled by marginal standard deviations to produce a covariance matrix.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{\\rho}}_{ij} &= \\mathrm{MI}(X_i,\\, X_j)\\,, \\\\
\\hat{\\mathbf{\\Sigma}}_{ij} &= \\hat{\\boldsymbol{\\rho}}_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\rho}}_{ij}``: Mutual information-based correlation between assets ``i`` and ``j``.
  - ``\\hat{\\mathbf{\\Sigma}}_{ij}``: Covariance between assets ``i`` and ``j``.
  - ``\\mathrm{MI}(X_i, X_j)``: (Optionally normalised) mutual information between assets ``i`` and ``j``.
  - ``\\hat{\\sigma}_i``: Marginal standard deviation of asset ``i`` from the variance estimator `ve`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MutualInfoCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        bins::Int_Bin = HacineGharbiRavier(),
        normalise::Bool = true
    ) -> MutualInfoCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:bins])

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
@propagatable @concrete struct MutualInfoCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:bins])
    """
    bins
    """
    $(field_dict[:normalise])
    """
    normalise
    function MutualInfoCovariance(ve::AbstractVarianceEstimator, bins::Int_Bin,
                                  normalise::Bool)
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins, DomainError(bins, "bins must be positive"))
            assert_resource_cap(bins, RESOURCE_LIMITS[].max_bins, :bins, :max_bins)
        end
        return new{typeof(ve), typeof(bins), typeof(normalise)}(ve, bins, normalise)
    end
end
function MutualInfoCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                              bins::Int_Bin = HacineGharbiRavier(),
                              normalise::Bool = true)::MutualInfoCovariance
    return MutualInfoCovariance(ve, bins, normalise)
end
"""
    Statistics.cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the mutual information (MI) correlation matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information correlation matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI correlation captures both linear and nonlinear dependencies between asset returns, making it robust to complex relationships that may not be detected by traditional correlation measures.

# Arguments

  - `ce`: Mutual information-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of mutual information-based correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cov(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    return mutual_info(X, ce.bins, ce.normalise)
end
"""
    Statistics.cov(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the mutual information (MI) covariance matrix using a [`MutualInfoCovariance`](@ref) estimator.

This method computes the pairwise mutual information covariance matrix for the input data matrix `X`, using the binning strategy and normalisation specified in `ce`. The MI covariance matrix is obtained by rescaling the MI correlation matrix by the marginal standard deviations, as estimated by the variance estimator in `ce`.

# Arguments

  - `ce`: Mutual information-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of mutual information-based covariances.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sigma = mutual_info(X, ce.bins, ce.normalise)
    return StatsBase.cor2cov!(sigma, sd)
end

export MutualInfoCovariance
