"""
$(DocStringExtensions.TYPEDEF)

Runs any covariance estimator, then applies a matrix post-processing step to its result.

`ce` computes the raw matrix and `mp` repairs or filters it — positive-definite repair, denoising, and detoning — so the composite is the estimator the rest of the library takes as its default.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PortfolioOptimisersCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        mp::AbstractMatrixProcessingEstimator = MatrixProcessing()
    ) -> PortfolioOptimisersCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
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

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Compute `sigma` with `Statistics.cov(ce.ce, X; kwargs...)`.
 3. When `sigma` is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Apply [`matrix_processing!`](@ref) with `ce.mp` to `sigma`, in place.
 5. Return `sigma`.

`ce.ce` runs before `ce.mp`, and `ce.mp.order` fixes the order of the steps inside the
post-processing. Step 1 orients `X` once, so the estimator and the post-processing both read the
same orientation and neither takes a `dims` of its own.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `sigma::Matrix{<:Number}`: The processed covariance matrix.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
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

# Algorithm

 1. Check `dims` and orient `X` to `observations × assets`, transposing it when `dims == 2`.
 2. Compute `rho` with `Statistics.cor(ce.ce, X; kwargs...)`.
 3. When `rho` is immutable, copy it into a `Matrix`, because step 4 writes in place.
 4. Apply [`matrix_processing!`](@ref) with `ce.mp` to `rho`, in place.
 5. Return `rho`.

`ce.ce` runs before `ce.mp`, and `ce.mp.order` fixes the order of the steps inside the
post-processing. Step 1 orients `X` once, so the estimator and the post-processing both read the
same orientation and neither takes a `dims` of its own.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Validation

  - $(val_dict[:dims])

# Returns

  - `rho::Matrix{<:Number}`: The processed correlation matrix.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cor)
"""
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::MatNum; dims = 1, kwargs...)
    X = dims_oriented(dims, X)
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

# Algorithm

 1. Compute the correlation matrix `rho` with `ce`, and take its absolute value when `absolute` is `true`.
 2. When `scores` is `nothing`, collapse each column of `rho` with `measure` into `summary_rho`, the default drop score. Otherwise take `summary_rho` from `scores`.
 3. Read the strict lower triangle of `rho` into `tril_idx`, giving each pair once.
 4. Keep the pairs whose correlation is at least `t`, and sort them from the most to the least correlated.
 5. Walk that list. For a pair whose two assets are both still present, remove the one with the higher drop score. **When the two scores are equal, remove both** — the library's "if we cannot tell them apart, trust neither" tie policy, which is why two identical columns leave no survivor.
 6. Return the indices that step 5 did not remove, in ascending order.

Step 5 skips a pair whose assets are already removed, so the result depends on the order step 4
fixes.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:ce])
  - `t`: Correlation threshold above which two assets are considered too correlated.
  - `absolute`: If `true`, the absolute value of the correlation is used for comparison.
  - `measure`: Summary measure applied to each column of the correlation matrix (e.g., mean) to produce the default drop score. Ignored when `scores` is given.
  - `scores`: Per-asset drop scores; the asset with the *higher* score is removed from a correlated pair.

# Validation

  - If `scores` is not `nothing`, `length(scores) == size(X, 2)`, else a `DimensionMismatch` is raised.

# Returns

  - `idx::Vector{Int}`: Indices of assets that form a maximally uncorrelated subset.

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
