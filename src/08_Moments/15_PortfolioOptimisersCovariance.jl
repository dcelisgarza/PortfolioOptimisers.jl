"""
$(DocStringExtensions.TYPEDEF)

Composite covariance estimator with post-processing.

`PortfolioOptimisersCovariance` is a flexible container type that combines any covariance estimator with a matrix post-processing step.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PortfolioOptimisersCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing()
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
     │   alg ┴ FullMoment()
  mp ┼ MatrixProcessing
     │     pdm ┼ Posdef
     │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      dn ┼ nothing
     │      dt ┼ nothing
     │     alg ┼ nothing
     │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
"""
@propagatable @concrete struct PortfolioOptimisersCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:mp])
    """
    mp
    function PortfolioOptimisersCovariance(ce::StatsBase.CovarianceEstimator,
                                           mp::AbstractMatrixProcessingEstimator)
        return new{typeof(ce), typeof(mp)}(ce, mp)
    end
end
function PortfolioOptimisersCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = MatrixProcessing())::PortfolioOptimisersCovariance
    return PortfolioOptimisersCovariance(ce, mp)
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
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
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
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
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
                              measure::Num_VecToScaM = MeanValue(),
                              scores::Option{<:VecNum} = nothing)

Find indices of a maximally uncorrelated subset of assets from a data matrix.

This function identifies a subset of asset columns in `X` such that no two assets in the subset have a pairwise (absolute) correlation exceeding the threshold `t`. When two assets are too correlated, the one with the higher *drop score* is removed. The function returns the indices of the remaining uncorrelated assets.

By default the drop score is each asset's summary correlation to every other asset, so the asset that is redundant with the *most* of the universe goes first. Supplying `scores` replaces that criterion — higher means "drop me" — which is how [`RedundancySelector`](@ref) makes the survivor of each correlated pair the better-scoring asset under a risk measure.

Internal machinery — the caller-facing form is [`RedundancySelector`](@ref) with a [`PairwiseCorrelation`](@ref) algorithm.

# Arguments

  - `X`: Data matrix of asset returns (observations × assets).
  - `ce`: Covariance estimator used to compute the correlation matrix.
  - `t`: Correlation threshold above which two assets are considered too correlated.
  - `absolute`: If `true`, the absolute value of the correlation is used for comparison.
  - `measure`: Summary measure applied to each column of the correlation matrix (e.g., mean) to produce the default drop score. Ignored when `scores` is given.
  - `scores`: Per-asset drop scores; the asset with the *higher* score is removed from a correlated pair.

# Returns

  - `idx::Vector{Int}`: Indices of assets that form a maximally uncorrelated subset.

# Details

  - Computes the (absolute) correlation matrix for all assets.
  - Identifies pairs of assets with correlation at or above `t`, sorted from most to least correlated.
  - For each correlated pair (not yet removed), removes the asset with the higher drop score. **If both assets score equally, both are removed** — the library's "if we cannot tell them apart, trust neither" tie policy, which is why two identical columns leave no survivor.
  - Returns the indices of assets not in the removed set.

# Related

  - [`RedundancySelector`](@ref)
  - [`PairwiseCorrelation`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`Num_VecToScaM`](@ref)
  - [`MeanValue`](@ref)
"""
function find_uncorrelated_indices(X::MatNum;
                                   ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                   t::Number = 0.95, absolute::Bool = false,
                                   measure::Num_VecToScaM = MeanValue(),
                                   scores::Option{<:VecNum} = nothing)
    N = size(X, 2)
    rho = !absolute ? Statistics.cor(ce, X) : abs.(Statistics.cor(ce, X))
    if !isnothing(scores)
        @argcheck(length(scores) == N,
                  DimensionMismatch("find_uncorrelated_indices got $(length(scores)) scores for $N assets"))
    end
    summary_rho = if isnothing(scores)
        [vec_to_real_measure(measure, x) for x in eachcol(rho)]
    else
        scores
    end
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

export PortfolioOptimisersCovariance
