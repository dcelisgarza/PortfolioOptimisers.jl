"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for low order prior estimators.

`AbstractLowOrderPriorEstimator` is the base type for estimators that compute low order moments (mean and covariance) from asset and/or factor returns. All concrete low order prior estimators should subtype this type for consistent moment estimation and integration.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using only asset returns.

`AbstractLowOrderPriorEstimator_A` is the base type for estimators that compute low order moments (mean and covariance) using only asset returns data. All concrete asset-only prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using factor returns.

`AbstractLowOrderPriorEstimator_F` is the base type for estimators that compute low order moments (mean and covariance) requiring the use of both asset and factor returns data. All concrete factor-adjusted prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Low order prior estimator using both asset and factor returns.

`AbstractLowOrderPriorEstimator_AF` is the base type for estimators that compute low order moments (mean and covariance) using both asset and optionally factor returns data. All concrete prior estimators which may optionally use factor returns should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
"""
    const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_AF` is a union type that allows dispatch on both asset-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using asset returns, with or without factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                      <:AbstractLowOrderPriorEstimator_AF}

Union type for factor-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_F_AF` is a union type that allows dispatch on both factor-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using factor returns, with or without asset returns.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_F_AF = Union{<:AbstractLowOrderPriorEstimator_F,
                                                  <:AbstractLowOrderPriorEstimator_AF}
"""
    const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                        <:AbstractLowOrderPriorEstimator_F,
                                                        <:AbstractLowOrderPriorEstimator_AF}

Union type for asset-only, factor-only, and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_F_AF` is a union type that allows dispatch on asset-only, factor-only, and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using any combination of asset and factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_F_AF = Union{<:AbstractLowOrderPriorEstimator_A,
                                                    <:AbstractLowOrderPriorEstimator_F,
                                                    <:AbstractLowOrderPriorEstimator_AF}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for high order prior estimators.

`AbstractHighOrderPriorEstimator` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) from asset and/or factor returns. All concrete high order prior estimators should subtype this type to ensure a consistent interface for higher moment estimation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
"""
$(DocStringExtensions.TYPEDEF)

High order prior estimator using factor returns.

`AbstractHighOrderPriorEstimator_F` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) requiring both asset and factor returns data. All concrete factor-based high order prior estimators should subtype this type.

# Related

  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHiLoOrderPriorEstimator_F`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator_F <: AbstractHighOrderPriorEstimator end
"""
    const AbstractHiLoOrderPriorEstimator_F = Union{<:AbstractLowOrderPriorEstimator_F,
                                                    <:AbstractHighOrderPriorEstimator_F}

Alias for a union of low-order and high-order factor prior estimator types.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractHighOrderPriorEstimator_F`](@ref)
"""
const AbstractHiLoOrderPriorEstimator_F = Union{<:AbstractLowOrderPriorEstimator_F,
                                                <:AbstractHighOrderPriorEstimator_F}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all prior result types.

`AbstractPriorResult` is the base type for all result objects produced by prior estimators, containing computed prior information such as moments, asset returns, and factor returns. All concrete prior result types should subtype this to ensure a consistent interface for integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractPriorResult <: AbstractResult end
"""
    const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}

Alias for a union of prior estimator and prior result types.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}
"""
    const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}

Alias for a union of prior result and returns result types.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`ReturnsResult`](@ref)
"""
const Pr_RR = Union{<:AbstractPriorResult, <:ReturnsResult}
"""
    prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)

Compute prior information from asset and/or factor returns using a prior estimator.

`prior` applies the specified prior estimator to a `ReturnsResult` object, extracting asset and factor returns and passing them, along with any additional information, to the estimator. Returns a prior result containing computed moments and other prior information for use in portfolio optimisation workflows.

# Arguments

  - `pr`: Prior estimator.
  - `rd`: Asset and/or factor returns result.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `pr::AbstractPriorResult`: Result object containing computed prior information.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`ReturnsResult`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
    @argcheck(!isnothing(rd.X), IsNothingError)
    if isa(pr, AbstractHiLoOrderPriorEstimator_F)
        @argcheck(!isnothing(rd.F),
                  IsNothingError("this is a factor prior; it needs factor returns. ReturnsResult.F is nothing — populate F (e.g. via prices_to_returns on factor prices)."))
    end
    return prior(pr, rd.X, rd.F; iv = rd.iv, ivpa = rd.ivpa, kwargs...)
end
"""
    prior(pr::AbstractPriorResult, args...; kwargs...)

Propagate or pass through prior result objects.

`prior` returns the input prior result object unchanged. This method is used to propagate already constructed prior results or enable uniform interface handling in workflows that accept either estimators or results.

# Arguments

  - `pr`: Prior result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `pr::AbstractPriorResult`: The input prior result object, unchanged.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
"""
function prior(pr::AbstractPriorResult, args...; kwargs...)::AbstractPriorResult
    return pr
end
"""
    port_opt_view(pr, args...; kwargs...)

Get a view or subset of a prior estimator or result for slicing.

Returns the prior unchanged for estimators (they are not sliceable), or returns a sliced prior result for a given cluster or asset index. Used in hierarchical optimisation to provide cluster-specific priors.

# Arguments

  - `pr`: Prior estimator or result.
  - `args...`: Additional arguments (index, etc.).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Sliced prior result or unchanged estimator.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function port_opt_view(pr::Option{<:AbstractPriorEstimator}, ::Any, args...;
                       kwargs...)::Option{<:AbstractPriorEstimator}
    return pr
end
function port_opt_view(pr::AbstractVector{<:Union{<:AbstractPriorResult,
                                                  <:AbstractPriorEstimator}}, ::Any,
                       args...; kwargs...)
    return pr
end
"""
    clusterise(cle::AbstractClustersEstimator, pr::AbstractPriorResult; kwargs...)

Clusterise asset or factor returns from a prior result using a clustering estimator.

`clusterise` applies the specified clustering estimator to the asset returns matrix contained in the prior result object, producing a clustering result for use in phylogeny analysis, constraint generation, or portfolio construction.

# Arguments

  - `cle`: Clustering estimator.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator.

# Returns

  - `clr::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClustersEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`clusterise`](@ref)
"""
function clusterise(cle::AbstractClustersEstimator, pr::Pr_RR;
                    rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true, kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return clusterise(cle, X; kwargs...)
end
"""
    phylogeny_matrix(pl::NwE_ClE_Cl, pr::AbstractPriorResult;
                     kwargs...)

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Arguments

  - `pl`: Network estimator, res estimator, or clustering result.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `plr::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(pl::NwE_ClE_Cl, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                          cle_pr::Bool = true, kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return phylogeny_matrix(pl, X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute phylogeny constraints from asset returns in a prior result using a phylogeny constraint estimator.

`phylogeny_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `cle_pr` is false).

# Arguments

  - `plc`: Phylogeny constraint estimator.
  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result (used when `cle_pr = false`).
  - `cle_pr`: If `true`, use asset returns from `pr`; otherwise, use `rd`. Default is `true`.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - Phylogeny constraint result.

# Related

  - [`AbstractPhylogenyConstraintEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`phylogeny_constraints`](@ref)
"""
function phylogeny_constraints(plc::AbstractPhylogenyConstraintEstimator, pr::Pr_RR;
                               rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                               kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return phylogeny_constraints(plc, X; kwargs...)
end
"""
    centrality_vector(cte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a centrality estimator and prior result.

`centrality_vector` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, returning centrality scores for each asset.

# Arguments

  - `cte`: Centrality estimator.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(cte::CentralityEstimator, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                           kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return centrality_vector(cte, X; kwargs...)
end
"""
    centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Arguments

  - `pl`: Network estimator, res estimator, or clustering result.
  - `ct`: Centrality algorithm.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `plr::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, pr::Pr_RR;
                           rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                           kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return centrality_vector(pl, ct, X; kwargs...)
end
"""
    average_centrality(pl::NwE_Pl_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum,
                       pr::AbstractPriorResult; kwargs...)

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Arguments

  - `pl`: Network estimator or phylogeny result.
  - `ct`: Centrality algorithm.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(pl::NwE_Pl_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,
                            cle_pr::Bool = true, kwargs...)
    return LinearAlgebra.dot(centrality_vector(pl, ct, pr; rd = rd, cle_pr = cle_pr,
                                               kwargs...).X, w)
end
"""
    average_centrality(cte::CentralityEstimator, w::VecNum, pr::AbstractPriorResult;
                       kwargs...)

Compute the weighted average centrality for a centrality estimator.

`average_centrality` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, then returns the weighted average using the provided portfolio weights.

# Arguments

  - `cte`: Centrality estimator.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Number`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::VecNum, pr::Pr_RR;
                            rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                            kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return average_centrality(cte, w, X; kwargs...)
end
"""
    asset_phylogeny(pl::NwE_ClE_Cl,
                    w::VecNum, pr::AbstractPriorResult; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a portfolio allocation using a phylogeny estimator or clustering result and a prior result.

This function computes the phylogeny matrix from the asset returns in the prior result using the specified phylogeny estimator or clustering result, then evaluates the asset phylogeny score for the given portfolio weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Arguments

  - `pl`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object containing asset returns.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the phylogeny matrix computation.

# Returns

  - `score::Number`: Asset phylogeny score.

# Details

  - Computes the phylogeny matrix from the asset returns in `pr` using `pl`.
  - Evaluates the weighted sum of the phylogeny matrix using the weights `w`.
  - Normalises the score by the sum of absolute weights.
  - Returns a real-valued score quantifying the phylogenetic structure of the allocation.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(pl::NwE_ClE_Cl, w::VecNum, pr::Pr_RR;
                         rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                         kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return asset_phylogeny(pl, w, X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute centrality constraints from asset returns in a prior result using a centrality constraint estimator.

`centrality_constraints` delegates to the asset-returns variant by extracting `X` from `pr` (or `rd` if provided and `cle_pr` is false).

# Arguments

  - `ccs`: Centrality constraint estimator or vector thereof.
  - `pr`: Prior result or returns result object.
  - `rd`: Optional returns result (used when `cle_pr = false`).
  - `cle_pr`: If `true`, use asset returns from `pr`; otherwise, use `rd`. Default is `true`.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - Centrality constraint result.

# Related

  - [`AbstractPriorResult`](@ref)
  - [`centrality_constraints`](@ref)
"""
function centrality_constraints(ccs::CC_VecCC, pr::Pr_RR;
                                rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true,
                                kwargs...)
    X = isnothing(rd) || cle_pr ? pr.X : rd.X
    return centrality_constraints(ccs, X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Container type for low order prior results in `PortfolioOptimisers.jl`.

`LowOrderPrior` stores the output of low order prior estimation routines, including asset returns, mean vector, covariance matrix, Cholesky factor, weights, entropy, Kullback-Leibler divergence, outlier weights, regression results, and optional factor moments. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowOrderPrior(;
        X::MatNum,
        mu::VecNum,
        sigma::MatNum,
        chol::Option{<:MatNum} = nothing,
        w::Option{<:ObsWeights} = nothing,
        ens::Option{<:Number} = nothing,
        kld::Option{<:Num_VecNum} = nothing,
        ow::Option{<:VecNum} = nothing,
        rr::Option{<:Regression} = nothing,
        f_mu::Option{<:VecNum} = nothing,
        f_sigma::Option{<:MatNum} = nothing,
        f_w::Option{<:VecNum} = nothing
    ) -> LowOrderPrior

Keywords correspond to the struct's fields.

## Validation

  - `X`, `mu`, and `sigma` must be non-empty.
  - `size(sigma, 1) == size(sigma, 2)`.
  - `size(X, 2) == length(mu) == size(sigma, 1)`.
  - If `w` is not `nothing`, `!isempty(w)` and `length(w) == size(X, 1)`.
  - If `kld` is an `AbstractVector`, `!isempty(kld)`.
  - If `ow` is not `nothing`, `!isempty(ow)`.
  - If any of `rr`, `f_mu`, or `f_sigma` are provided, all must be provided and non-empty, `size(rr.M, 2) == length(f_mu) == size(f_sigma, 1)`, and `size(rr.M, 1) == length(mu)`.
  - If `f_sigma` is not `nothing`, it must be square and `size(f_sigma, 1) == size(rr.M, 2)`.
  - If `chol` is not `nothing`, `!isempty(chol)` and `length(mu) == size(chol, 2)`.
  - If `f_w` is not `nothing`, `!isempty(f_w)` and `length(f_w) == size(X, 1)`.

# Examples

```jldoctest
julia> LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                     sigma = [0.0001 0.0002; 0.0002 0.0003])
LowOrderPrior
        X ┼ 2×2 Matrix{Float64}
       mu ┼ Vector{Float64}: [0.02, 0.03]
    sigma ┼ 2×2 Matrix{Float64}
     chol ┼ nothing
        w ┼ nothing
      ens ┼ nothing
      kld ┼ nothing
       ow ┼ nothing
       rr ┼ nothing
     f_mu ┼ nothing
  f_sigma ┼ nothing
      f_w ┴ nothing
```

# Related

  - [`AbstractPriorResult`](@ref)
  - [`prior`](@ref)
  - [`HighOrderPrior`](@ref)
"""
@concrete struct LowOrderPrior <: AbstractPriorResult
    """
    $(field_dict[:X])
    """
    X
    """
    $(field_dict[:mu])
    """
    mu
    """
    $(field_dict[:sigma])
    """
    sigma
    """
    $(field_dict[:chol])
    """
    chol
    """
    $(field_dict[:w_prior])
    """
    w
    """
    $(field_dict[:ens])
    """
    ens
    """
    $(field_dict[:kld])
    """
    kld
    """
    $(field_dict[:op_w])
    """
    ow
    """
    $(field_dict[:reg_rr])
    """
    rr
    """
    $(field_dict[:f_mu])
    """
    f_mu
    """
    $(field_dict[:f_sigma])
    """
    f_sigma
    """
    $(field_dict[:f_w])
    """
    f_w
    function LowOrderPrior(X::MatNum, mu::VecNum, sigma::MatNum, chol::Option{<:MatNum},
                           w::Option{<:ObsWeights}, ens::Option{<:Number},
                           kld::Option{<:Num_VecNum}, ow::Option{<:VecNum},
                           rr::Option{<:Regression}, f_mu::Option{<:VecNum},
                           f_sigma::Option{<:MatNum}, f_w::Option{<:VecNum})
        @argcheck(!isempty(X), IsEmptyError("X cannot be empty"))
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        @argcheck(!isempty(sigma), IsEmptyError("sigma cannot be empty"))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(size(X, 2) == length(mu) == size(sigma, 1),
                  DimensionMismatch("size(X, 2) ($(size(X, 2))), length(mu) ($(length(mu))), and size(sigma, 1) ($(size(sigma, 1))) must all match"))
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(w, StatsBase.AbstractWeights)
            @argcheck(length(w) == size(X, 1),
                      DimensionMismatch("length(w) ($(length(w))) must match size(X, 1) ($(size(X, 1)))"))
        end
        if isa(kld, VecNum)
            @argcheck(!isempty(kld), IsEmptyError("kld cannot be empty"))
        end
        if !isnothing(ow)
            @argcheck(!isempty(ow), IsEmptyError("ow cannot be empty"))
        end
        loadings_flag = !isnothing(rr)
        f_mu_flag = !isnothing(f_mu)
        f_sigma_flag = !isnothing(f_sigma)
        if loadings_flag || f_mu_flag || f_sigma_flag
            @argcheck(loadings_flag,
                      ArgumentError("rr must be provided when f_mu or f_sigma is provided, isnothing(rr) = $(loadings_flag), isnothing(f_mu) = $(f_mu_flag), isnothing(f_sigma) = $(f_sigma_flag)"))
            @argcheck(f_mu_flag,
                      ArgumentError("f_mu must be provided when rr or f_sigma is provided, isnothing(rr) = $(loadings_flag), isnothing(f_mu) = $(f_mu_flag), isnothing(f_sigma) = $(f_sigma_flag)"))
            @argcheck(f_sigma_flag,
                      ArgumentError("f_sigma must be provided when rr or f_mu is provided, isnothing(rr) = $(loadings_flag), isnothing(f_mu) = $(f_mu_flag), isnothing(f_sigma) = $(f_sigma_flag)"))
            @argcheck(!isempty(f_mu), IsEmptyError("f_mu cannot be empty"))
            @argcheck(!isempty(f_sigma), IsEmptyError("f_sigma cannot be empty"))
            assert_matrix_issquare(f_sigma, :f_sigma)
            @argcheck(size(rr.M, 2) == length(f_mu) == size(f_sigma, 1),
                      DimensionMismatch("size(rr.M, 2) = $(size(rr.M, 2)), length(f_mu) = $(length(f_mu)), and size(f_sigma, 1) = $(size(f_sigma, 1)) must all match"))
            @argcheck(size(rr.M, 1) == length(mu),
                      DimensionMismatch("size(rr.M, 1) = $(size(rr.M, 1)) must match length(mu) = $(length(mu))"))
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol), IsEmptyError("chol cannot be empty"))
            @argcheck(length(mu) == size(chol, 2),
                      DimensionMismatch("length(mu) ($(length(mu))) must match size(chol, 2) ($(size(chol, 2)))"))
        end
        if !isnothing(f_w)
            @argcheck(!isempty(f_w), IsEmptyError("f_w cannot be empty"))
            @argcheck(length(f_w) == size(X, 1),
                      DimensionMismatch("length(f_w) ($(length(f_w))) must match size(X, 1) ($(size(X, 1)))"))
        end
        return new{typeof(X), typeof(mu), typeof(sigma), typeof(chol), typeof(w),
                   typeof(ens), typeof(kld), typeof(ow), typeof(rr), typeof(f_mu),
                   typeof(f_sigma), typeof(f_w)}(X, mu, sigma, chol, w, ens, kld, ow, rr,
                                                 f_mu, f_sigma, f_w)
    end
end
function LowOrderPrior(; X::MatNum, mu::VecNum, sigma::MatNum,
                       chol::Option{<:MatNum} = nothing, w::Option{<:ObsWeights} = nothing,
                       ens::Option{<:Number} = nothing, kld::Option{<:Num_VecNum} = nothing,
                       ow::Option{<:VecNum} = nothing, rr::Option{<:Regression} = nothing,
                       f_mu::Option{<:VecNum} = nothing,
                       f_sigma::Option{<:MatNum} = nothing,
                       f_w::Option{<:VecNum} = nothing)::LowOrderPrior
    return LowOrderPrior(X, mu, sigma, chol, w, ens, kld, ow, rr, f_mu, f_sigma, f_w)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`LowOrderPrior`](@ref) restricted to assets at index `i`.

# Related

  - [`LowOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::LowOrderPrior, i, args...)::LowOrderPrior
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    return LowOrderPrior(; X = view(pr.X, :, i), mu = view(pr.mu, i),
                         sigma = view(pr.sigma, i, i), chol = chol, w = pr.w, ens = pr.ens,
                         kld = pr.kld, ow = pr.ow, rr = port_opt_view(pr.rr, i),
                         f_mu = pr.f_mu, f_sigma = pr.f_sigma, f_w = pr.f_w)
end
"""
$(DocStringExtensions.TYPEDEF)

Container type for high order prior results in `PortfolioOptimisers.jl`.

`HighOrderPrior` stores the output of high order prior estimation routines, including low order prior results, cokurtosis tensor, elimination and summation matrices, coskewness tensor, quadratic skewness matrix, and matrix processing estimator. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics involving higher moments.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderPrior(;
        pr::AbstractPriorResult,
        kt::Option{<:MatNum} = nothing,
        D2::Option{<:MatNum} = nothing,
        L2::Option{<:MatNum} = nothing,
        S2::Option{<:MatNum} = nothing,
        sk::Option{<:MatNum} = nothing,
        V::Option{<:MatNum} = nothing,
        skmp::Option{<:AbstractMatrixProcessingEstimator} = MatrixProcessing()
    ) -> HighOrderPrior

Keywords correspond to the struct's fields.

## Validation

Defining `N = length(pr.mu)`.

  - If any of `kt`, `L2`, or `S2` are provided, all must be provided, non-empty, and `size(kt) == (N^2, N^2)`, `size(L2) == size(S2) == (div(N * (N + 1), 2), N^2)`.
  - If `sk` or `V` are provided, both must be provided, non-empty, and `size(sk) == (N, N^2)`, `size(V) == (N, N)`.

# Examples

```jldoctest
julia> HighOrderPrior(;
                      pr = LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                                         sigma = [0.0001 0.0002; 0.0002 0.0003]), kt = rand(4, 4),
                      D2 = PortfolioOptimisers.duplication_matrix(2),
                      L2 = PortfolioOptimisers.elimination_matrix(2),
                      S2 = PortfolioOptimisers.summation_matrix(2), sk = rand(2, 4),
                      V = rand(2, 2))
HighOrderPrior
    pr ┼ LowOrderPrior
       │         X ┼ 2×2 Matrix{Float64}
       │        mu ┼ Vector{Float64}: [0.02, 0.03]
       │     sigma ┼ 2×2 Matrix{Float64}
       │      chol ┼ nothing
       │         w ┼ nothing
       │       ens ┼ nothing
       │       kld ┼ nothing
       │        ow ┼ nothing
       │        rr ┼ nothing
       │      f_mu ┼ nothing
       │   f_sigma ┼ nothing
       │       f_w ┴ nothing
    kt ┼ 4×4 Matrix{Float64}
    D2 ┼ 4×3 SparseArrays.SparseMatrixCSC{Int64, Int64}
    L2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    S2 ┼ 3×4 SparseArrays.SparseMatrixCSC{Int64, Int64}
    sk ┼ 2×4 Matrix{Float64}
     V ┼ 2×2 Matrix{Float64}
  skmp ┼ nothing
  f_kt ┼ nothing
  f_sk ┼ nothing
   f_V ┴ nothing
```

# Related

  - [`AbstractPriorResult`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`HighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
@concrete struct HighOrderPrior <: AbstractPriorResult
    """
    $(field_dict[:pr])
    """
    pr
    """
    $(field_dict[:kt])
    """
    kt
    """
    $(field_dict[:D2])
    """
    D2
    """
    $(field_dict[:L2])
    """
    L2
    """
    $(field_dict[:S2])
    """
    S2
    """
    $(field_dict[:sk])
    """
    sk
    """
    $(field_dict[:V])
    """
    V
    """
    $(field_dict[:skmp])
    """
    skmp
    """
    $(field_dict[:f_kt])
    """
    f_kt
    # chol_kt
    """
    $(field_dict[:f_sk])
    """
    f_sk
    """
    $(field_dict[:f_V])
    """
    f_V
    function HighOrderPrior(pr::AbstractPriorResult, kt::Option{<:MatNum},
                            D2::Option{<:MatNum}, L2::Option{<:MatNum},
                            S2::Option{<:MatNum}, sk::Option{<:MatNum}, V::Option{<:MatNum},
                            skmp::Option{<:AbstractMatrixProcessingEstimator},
                            f_kt::Option{<:MatNum}, #chol_kt::Option{<:MatNum},
                            f_sk::Option{<:MatNum}, f_V::Option{<:MatNum})
        N = length(pr.mu)
        sk_flag = isa(sk, MatNum)
        kt_flag = isa(kt, MatNum)
        L2_flag = isa(L2, MatNum)
        S2_flag = isa(S2, MatNum)
        if kt_flag || L2_flag || S2_flag
            @argcheck(kt_flag,
                      ArgumentError("kt must be provided when L2 or S2 is provided, isnothing(kt) = $(kt_flag), isnothing(L2) = $(L2_flag), isnothing(S2) = $(S2_flag)"))
            @argcheck(L2_flag,
                      ArgumentError("L2 must be provided when kt or S2 is provided, isnothing(kt) = $(kt_flag), isnothing(L2) = $(L2_flag), isnothing(S2) = $(S2_flag)"))
            @argcheck(S2_flag,
                      ArgumentError("S2 must be provided when kt or L2 is provided, isnothing(kt) = $(kt_flag), isnothing(L2) = $(L2_flag), isnothing(S2) = $(S2_flag)"))
            @argcheck(!isempty(kt), IsEmptyError("kt cannot be empty"))
            @argcheck(!isempty(L2), IsEmptyError("L2 cannot be empty"))
            @argcheck(!isempty(S2), IsEmptyError("S2 cannot be empty"))
            @argcheck(size(kt) == (N^2, N^2),
                      DimensionMismatch("size(kt) ($(size(kt))) must be ($(N^2), $(N^2))"))
            @argcheck(size(L2) == size(S2) == (div(N * (N + 1), 2), N^2),
                      DimensionMismatch("size(L2) ($(size(L2))) and size(S2) ($(size(S2))) must be ($(div(N * (N + 1), 2)), $(N^2))"))
            if sk_flag
                @argcheck(isa(D2, MatNum),
                          ArgumentError("D2 must be provided when sk is provided, isnothing(D2) = $(isnothing(D2)), isnothing(sk) = $(sk_flag)"))
                @argcheck(!isempty(D2), IsEmptyError("D2 cannot be empty"))
                @argcheck(size(D2) == size(transpose(L2)),
                          DimensionMismatch("size(D2) = $(size(D2)) must match size(L2') = $(size(transpose(L2)))"))
            end
        end
        V_flag = isa(V, MatNum)
        if sk_flag || V_flag
            @argcheck(sk_flag,
                      ArgumentError("sk must be provided when V is provided, isnothing(sk) = $(sk_flag), isnothing(V) = $(V_flag)"))
            @argcheck(V_flag,
                      ArgumentError("V must be provided when sk is provided, isnothing(sk) = $(sk_flag), isnothing(V) = $(V_flag)"))
            @argcheck(!isempty(sk), IsEmptyError("sk cannot be empty"))
            @argcheck(!isempty(V), IsEmptyError("V cannot be empty"))
            @argcheck(size(V) == (N, N),
                      DimensionMismatch("size(V) = $(size(V)) must be ($N, $N)"))
            @argcheck(size(sk) == (N, N^2),
                      DimensionMismatch("size(sk) = $(size(sk)) must be ($N, $(N^2))"))
        end
        f_kt_flag = !isnothing(f_kt)
        f_sk_flag = !isnothing(f_sk)
        if f_kt_flag || f_sk_flag
            rr = pr.rr
            @argcheck(!isnothing(rr),
                      IsNothingError("pr.rr cannot be nothing when f_kt or f_sk is provided"))
            Nf = size(rr.M, 2)
            if f_kt_flag
                @argcheck(!isempty(f_kt), IsEmptyError("f_kt cannot be empty"))
                # @argcheck(!isempty(chol_kt))
                assert_matrix_issquare(f_kt, :f_kt)
                @argcheck(Nf^2 == size(f_kt, 1),
                          DimensionMismatch("Nf^2 ($( Nf^2)) must match size(f_kt, 1) ($(size(f_kt, 1)))"))
                # @argcheck(N^2 == Nfa^2 == size(chol_kt, 2))
            end
            if f_sk_flag
                @argcheck(!isempty(f_sk), IsEmptyError("f_sk cannot be empty"))
                @argcheck(!isempty(f_V), IsEmptyError("f_V cannot be empty"))
                @argcheck(size(f_sk) == (Nf, Nf^2),
                          DimensionMismatch("size(f_sk) ($(size(f_sk))) must be ($Nf, $(Nf^2))"))
                @argcheck(size(f_V) == (Nf, Nf),
                          DimensionMismatch("size(f_V) ($(size(f_V))) must be ($Nf, $Nf)"))
            end
        end
        return new{typeof(pr), typeof(kt), typeof(D2), typeof(L2), typeof(S2), typeof(sk),
                   typeof(V), typeof(skmp), typeof(f_kt), #typeof(chol_kt),
                   typeof(f_sk), typeof(f_V)}(pr, kt, D2, L2, S2, sk, V, skmp, f_kt,
                                              #  chol_kt,
                                              f_sk, f_V)
    end
end
function HighOrderPrior(; pr::AbstractPriorResult, kt::Option{<:MatNum} = nothing,
                        D2::Option{<:MatNum} = nothing, L2::Option{<:MatNum} = nothing,
                        S2::Option{<:MatNum} = nothing, sk::Option{<:MatNum} = nothing,
                        V::Option{<:MatNum} = nothing,
                        skmp::Option{<:AbstractMatrixProcessingEstimator} = nothing,
                        f_kt::Option{<:MatNum} = nothing,
                        # chol_kt::Option{<:MatNum} = nothing,
                        f_sk::Option{<:MatNum} = nothing,
                        f_V::Option{<:MatNum} = nothing)::HighOrderPrior
    return HighOrderPrior(pr, kt, D2, L2, S2, sk, V, skmp, f_kt, #chol_kt,
                          f_sk, f_V)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`HighOrderPrior`](@ref) restricted to assets at index `i`, slicing all relevant moment tensors accordingly.

# Related

  - [`HighOrderPrior`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pr::HighOrderPrior, i, args...)
    idx = fourth_moment_index_generator(length(pr.mu), i)
    kt = pr.kt
    sk = pr.sk
    skmp = pr.skmp
    sk = nothing_scalar_array_view_odd_order(sk, i, idx)
    if !isnothing(sk)
        V = negative_spectral_coskewness(sk, view(pr.X, :, i), skmp)
    else
        V = nothing
    end
    if !isnothing(pr.D2)
        D2, L2, S2 = dup_elim_sum_view(kt, length(i))
    elseif !isnothing(pr.S2)
        D2 = nothing
        L2, S2 = dup_elim_sum_view(kt, length(i))[2:3]
    else
        D2, L2, S2 = (nothing, nothing, nothing)
    end
    return HighOrderPrior(; pr = port_opt_view(pr.pr, i),
                          kt = nothing_scalar_array_view(kt, idx), D2 = D2, L2 = L2,
                          S2 = S2, sk = sk, V = V, skmp = skmp, f_kt = pr.f_kt,
                          f_sk = pr.f_sk, f_V = pr.f_V)
end
# Forward unknown property names to the embedded `pr` prior, allowing transparent access to
# low-order moment fields (see [`@forward_properties`](@ref)).
@forward_properties HighOrderPrior begin
    forward(pr)
end

export prior, LowOrderPrior, HighOrderPrior
