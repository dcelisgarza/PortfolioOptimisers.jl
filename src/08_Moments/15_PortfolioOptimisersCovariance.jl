"""
$(DocStringExtensions.TYPEDEF)

Composite covariance estimator with post-processing.

`PortfolioOptimisersCovariance` is a flexible container type that combines any covariance estimator with a matrix post-processing step.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PortfolioOptimisersCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing()
    ) -> PortfolioOptimisersCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PortfolioOptimisersCovariance()
PortfolioOptimisersCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┴ Full()
  mp ┼ DenoiseDetoneAlgMatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ nothing
     │      dt ┼ nothing
     │     alg ┼ nothing
     │   order ┴ DenoiseDetoneAlg()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
"""
@concrete struct PortfolioOptimisersCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:mp])"
    mp
    function PortfolioOptimisersCovariance(ce::StatsBase.CovarianceEstimator,
                                           mp::AbstractMatrixProcessingEstimator)
        return new{typeof(ce), typeof(mp)}(ce, mp)
    end
end
function PortfolioOptimisersCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = DenoiseDetoneAlgMatrixProcessing())::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(ce, mp)
end
"""
    factory(ce::PortfolioOptimisersCovariance, w::ObsWeights) -> PortfolioOptimisersCovariance

Return a new [`PortfolioOptimisersCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::PortfolioOptimisersCovariance,
                 w::ObsWeights)::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = factory(ce.ce, w), mp = ce.mp)
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

  - [`PortfolioOptimisersCovariance`](@ref)
"""
function moment_view(ce::PortfolioOptimisersCovariance, i)::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(; ce = moment_view(ce.ce, i), mp = ce.mp)
end
"""
    Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)

Compute the covariance matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `sigma::Matrix{<:Number}`: The processed covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)

Compute the correlation matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the correlation matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `rho::Matrix{<:Number}`: The processed correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cor)
"""
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end
"""
    find_uncorrelated_indices(X::MatNum;
                              ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                              t::Number = 0.95, absolute::Bool = false,
                              measure::VectorToScalarMeasure = MeanValue())

Find indices of a maximally uncorrelated subset of assets from a data matrix.

This function identifies a subset of asset columns in `X` such that no two assets in the subset have a pairwise (absolute) correlation exceeding the threshold `t`. When two assets are too correlated, the one with the higher summary correlation measure (across all assets) is removed. The function returns the indices of the remaining uncorrelated assets.

# Arguments

  - `X`: Data matrix of asset returns (observations × assets).
  - `ce`: Covariance estimator used to compute the correlation matrix.
  - `t`: Correlation threshold above which two assets are considered too correlated.
  - `absolute`: If `true`, the absolute value of the correlation is used for comparison.
  - `measure`: Summary measure applied to each column of the correlation matrix (e.g., mean) to decide which asset to remove when two are too correlated.

# Returns

  - `idx::Vector{Int}`: Indices of assets that form a maximally uncorrelated subset.

# Details

  - Computes the (absolute) correlation matrix for all assets.
  - Identifies pairs of assets with correlation at or above `t`, sorted from most to least correlated.
  - For each correlated pair (not yet removed), removes the asset with the higher summary correlation value. If both assets have equal summary values, both are removed.
  - Returns the indices of assets not in the removed set.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
"""
function find_uncorrelated_indices(X::MatNum;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   t::Number = 0.95, absolute::Bool = false,
                                   measure::VectorToScalarMeasure = MeanValue())
    N = size(X, 2)
    rho = !absolute ? Statistics.cor(ce, X) : abs.(Statistics.cor(ce, X))
    summary_rho = [vec_to_real_measure(measure, x) for x in eachcol(rho)]
    tril_idx = findall(LinearAlgebra.tril!(trues(size(rho)), -1))
    candidate_idx = findall(x -> x >= t, rho[tril_idx])
    candidate_idx = candidate_idx[sortperm(rho[tril_idx][candidate_idx]; rev = true)]
    to_remove = sizehint!(Set{Int}(), div(length(candidate_idx), 2))
    for idx in candidate_idx
        i, j = tril_idx[idx][1], tril_idx[idx][2]
        if i ∉ to_remove && j ∉ to_remove
            if summary_rho[i] > summary_rho[j]
                push!(to_remove, i)
            elseif summary_rho[i] < summary_rho[j]
                push!(to_remove, j)
            else
                push!(to_remove, i)
                push!(to_remove, j)
            end
        end
    end
    return setdiff(1:N, to_remove)
end

export PortfolioOptimisersCovariance, find_uncorrelated_indices
