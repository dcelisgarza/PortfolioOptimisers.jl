"""
$(DocStringExtensions.TYPEDEF)

Covariance estimator based on mutual information.

`MutualInfoCovariance` implements a robust covariance estimator that uses mutual information (MI) to capture both linear and nonlinear dependencies between asset returns. This estimator is particularly useful for identifying complex relationships that are not detected by traditional correlation-based methods. The MI matrix is optionally normalised and then rescaled by marginal standard deviations to produce a covariance matrix.

# Mathematical definition

```math
\\hat{\\boldsymbol{\\rho}}_{ij} = \\mathrm{MI}(X_i,\\, X_j), \\qquad
\\hat{\\mathbf{\\Sigma}}_{ij} = \\hat{\\boldsymbol{\\rho}}_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j
```

where ``\\mathrm{MI}(X_i, X_j)`` is the (optionally normalised) mutual information between assets ``i`` and ``j``, and ``\\hat{\\sigma}_i`` are marginal standard deviations from the variance estimator `ve`.

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
@concrete struct MutualInfoCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:bins])"
    bins
    "$(field_dict[:normalise])"
    normalise
    function MutualInfoCovariance(ve::AbstractVarianceEstimator, bins::Int_Bin,
                                  normalise::Bool)
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins)
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
    factory(ce::MutualInfoCovariance, w::ObsWeights) -> MutualInfoCovariance

Return a new [`MutualInfoCovariance`](@ref) estimator with observation weights `w` applied to the underlying variance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::MutualInfoCovariance, w::ObsWeights)::MutualInfoCovariance
    return MutualInfoCovariance(; ve = factory(ce.ve, w), bins = ce.bins,
                                normalise = ce.normalise)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the covariance estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ce])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:cev])

# Related

  - [`MutualInfoCovariance`](@ref)
"""
function moment_view(ce::MutualInfoCovariance, i)::MutualInfoCovariance
    return MutualInfoCovariance(; ve = moment_view(ce.ve, i), bins = ce.bins,
                                normalise = ce.normalise)
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
    @argcheck(dims in (1, 2))
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

# Examples

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`mutual_info`](@ref)
  - [`cor(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::MutualInfoCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sigma = mutual_info(X, ce.bins, ce.normalise)
    return StatsBase.cor2cov!(sigma, sd)
end

export MutualInfoCovariance
