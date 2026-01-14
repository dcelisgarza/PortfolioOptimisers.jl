"""
    abstract type AbstractPriorEstimator <: AbstractEstimator end

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
    abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end

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
    abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end

Low order prior estimator using only asset returns.

`AbstractLowOrderPriorEstimator_A` is the base type for estimators that compute low order moments (mean and covariance) using only asset returns data. All concrete asset-only prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
    abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end

Low order prior estimator using factor returns.

`AbstractLowOrderPriorEstimator_F` is the base type for estimators that compute low order moments (mean and covariance) requiring the use of both asset and factor returns data. All concrete factor-adjusted prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
    abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end

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
    abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end

Abstract supertype for high order prior estimators.

`AbstractHighOrderPriorEstimator` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) from asset and/or factor returns. All concrete high order prior estimators should subtype this type to ensure a consistent interface for higher moment estimation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
abstract type AbstractHighOrderPriorEstimator_F <: AbstractHighOrderPriorEstimator end
"""
    abstract type AbstractPriorResult <: AbstractResult end

Abstract supertype for all prior result types.

`AbstractPriorResult` is the base type for all result objects produced by prior estimators, containing computed prior information such as moments, asset returns, and factor returns. All concrete prior result types should subtype this to ensure a consistent interface for integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractPriorResult <: AbstractResult end
const PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}
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
function prior(pr::AbstractPriorResult, args...; kwargs...)
    return pr
end
function prior_view(pr::AbstractPriorEstimator, args...; kwargs...)
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
function clusterise(cle::AbstractClustersEstimator, pr::AbstractPriorResult; kwargs...)
    return clusterise(cle, pr.X; kwargs...)
end
"""
    phylogeny_matrix(necle::NwE_ClE_Cl, pr::AbstractPriorResult;
                     kwargs...)

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Arguments

  - `necle`: Network estimator, res estimator, or clustering result.
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
function phylogeny_matrix(necle::NwE_ClE_Cl, pr::AbstractPriorResult; kwargs...)
    return phylogeny_matrix(necle, pr.X; kwargs...)
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
function centrality_vector(cte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)
    return centrality_vector(cte, pr.X; kwargs...)
end
"""
    centrality_vector(nt::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                      pr::AbstractPriorResult; kwargs...)

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Arguments

  - `nt`: Network estimator, res estimator, or clustering result.
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
function centrality_vector(nt::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm,
                           pr::AbstractPriorResult; kwargs...)
    return centrality_vector(nt, ct, pr.X; kwargs...)
end
"""
    average_centrality(nt::NwE_Ph_ClE_Cl,
                       ct::AbstractCentralityAlgorithm, w::VecNum,
                       pr::AbstractPriorResult; kwargs...)

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Arguments

  - `nt`: Network estimator or phylogeny result.
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
function average_centrality(nt::NwE_Ph_ClE_Cl, ct::AbstractCentralityAlgorithm, w::VecNum,
                            pr::AbstractPriorResult; kwargs...)
    return LinearAlgebra.dot(centrality_vector(nt, ct, pr; kwargs...).X, w)
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
function average_centrality(cte::CentralityEstimator, w::VecNum, pr::AbstractPriorResult;
                            kwargs...)
    return average_centrality(cte, w, pr.X; kwargs...)
end
"""
    asset_phylogeny(cle::NwE_ClE_Cl,
                    w::VecNum, pr::AbstractPriorResult; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a portfolio allocation using a phylogeny estimator or clustering result and a prior result.

This function computes the phylogeny matrix from the asset returns in the prior result using the specified phylogeny estimator or clustering result, then evaluates the asset phylogeny score for the given portfolio weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation.

# Arguments

  - `cle`: Phylogeny estimator or clustering result used to compute the phylogeny matrix.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object containing asset returns.
  - `dims`: Dimension along which to compute the phylogeny matrix.
  - `kwargs...`: Additional keyword arguments passed to the phylogeny matrix computation.

# Returns

  - `score::Number`: Asset phylogeny score.

# Details

  - Computes the phylogeny matrix from the asset returns in `pr` using `cle`.
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
function asset_phylogeny(cle::NwE_ClE_Cl, w::VecNum, pr::AbstractPriorResult; dims::Int = 1,
                         kwargs...)
    return asset_phylogeny(cle, w, pr.X; dims = dims, kwargs...)
end
"""
    struct LowOrderPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12} <:
           AbstractPriorResult
        X::T1
        mu::T2
        sigma::T3
        chol::T4
        w::T5
        ens::T6
        kld::T7
        ow::T8
        rr::T9
        f_mu::T10
        f_sigma::T11
        f_w::T12
    end

Container type for low order prior results in PortfolioOptimisers.jl.

`LowOrderPrior` stores the output of low order prior estimation routines, including asset returns, mean vector, covariance matrix, Cholesky factor, weights, entropy, Kullback-Leibler divergence, outlier weights, regression results, and optional factor moments. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics.

# Fields

  - `X`: Asset returns matrix.
  - `mu`: Mean vector.
  - `sigma`: Covariance matrix.
  - `chol`: Cholesky factorisation of the factor-adjusted covariance matrix. Factor models sparsify the covariance matrix, so using their smaller, sparser Cholesky factor makes for more numerically stable and efficient optimisations.
  - `w`: Asset weights.
  - `ens`: Effective number of scenarios.
  - `kld`: Kullback-Leibler divergence.
  - `ow`: Opinion pooling weights.
  - `rr`: Regression result.
  - `f_mu`: Factor mean vector.
  - `f_sigma`: Factor covariance matrix.
  - `f_w`: Factor weights.

# Constructor

    LowOrderPrior(; X::MatNum, mu::VecNum, sigma::MatNum,
                  chol::Option{<:MatNum} = nothing,
                  w::Option{<:StatsBase.AbstractWeights} = nothing,
                  ens::Option{<:Number} = nothing,
                  kld::Option{<:Num_VecNum} = nothing,
                  ow::Option{<:VecNum} = nothing,
                  rr::Option{<:Regression} = nothing,
                  f_mu::Option{<:VecNum} = nothing,
                  f_sigma::Option{<:MatNum} = nothing,
                  f_w::Option{<:VecNum} = nothing)

Keyword arguments correspond to the fields above.

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
struct LowOrderPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12} <:
       AbstractPriorResult
    X::T1
    mu::T2
    sigma::T3
    chol::T4
    w::T5
    ens::T6
    kld::T7
    ow::T8
    rr::T9
    f_mu::T10
    f_sigma::T11
    f_w::T12
    function LowOrderPrior(X::MatNum, mu::VecNum, sigma::MatNum, chol::Option{<:MatNum},
                           w::Option{<:StatsBase.AbstractWeights}, ens::Option{<:Number},
                           kld::Option{<:Num_VecNum}, ow::Option{<:VecNum},
                           rr::Option{<:Regression}, f_mu::Option{<:VecNum},
                           f_sigma::Option{<:MatNum}, f_w::Option{<:VecNum})
        @argcheck(!isempty(X))
        @argcheck(!isempty(mu))
        @argcheck(!isempty(sigma))
        assert_matrix_issquare(sigma, :sigma)
        @argcheck(size(X, 2) == length(mu) == size(sigma, 1))
        if !isnothing(w)
            @argcheck(!isempty(w))
            @argcheck(length(w) == size(X, 1))
        end
        if isa(kld, VecNum)
            @argcheck(!isempty(kld))
        end
        if !isnothing(ow)
            @argcheck(!isempty(ow))
        end
        loadings_flag = !isnothing(rr)
        f_mu_flag = !isnothing(f_mu)
        f_sigma_flag = !isnothing(f_sigma)
        if loadings_flag || f_mu_flag || f_sigma_flag
            @argcheck(loadings_flag)
            @argcheck(f_mu_flag)
            @argcheck(f_sigma_flag)
            @argcheck(!isempty(f_mu))
            @argcheck(!isempty(f_sigma))
            assert_matrix_issquare(f_sigma, :f_sigma)
            @argcheck(size(rr.M, 2) == length(f_mu) == size(f_sigma, 1))
            @argcheck(size(rr.M, 1) == length(mu))
        end
        if !isnothing(chol)
            @argcheck(!isempty(chol))
            @argcheck(length(mu) == size(chol, 2))
        end
        if !isnothing(f_w)
            @argcheck(!isempty(f_w))
            @argcheck(length(f_w) == size(X, 1))
        end
        return new{typeof(X), typeof(mu), typeof(sigma), typeof(chol), typeof(w),
                   typeof(ens), typeof(kld), typeof(ow), typeof(rr), typeof(f_mu),
                   typeof(f_sigma), typeof(f_w)}(X, mu, sigma, chol, w, ens, kld, ow, rr,
                                                 f_mu, f_sigma, f_w)
    end
end
function LowOrderPrior(; X::MatNum, mu::VecNum, sigma::MatNum,
                       chol::Option{<:MatNum} = nothing,
                       w::Option{<:StatsBase.AbstractWeights} = nothing,
                       ens::Option{<:Number} = nothing, kld::Option{<:Num_VecNum} = nothing,
                       ow::Option{<:VecNum} = nothing, rr::Option{<:Regression} = nothing,
                       f_mu::Option{<:VecNum} = nothing,
                       f_sigma::Option{<:MatNum} = nothing, f_w::Option{<:VecNum} = nothing)
    return LowOrderPrior(X, mu, sigma, chol, w, ens, kld, ow, rr, f_mu, f_sigma, f_w)
end
function prior_view(pr::LowOrderPrior, i)
    chol = isnothing(pr.chol) ? nothing : view(pr.chol, :, i)
    return LowOrderPrior(; X = view(pr.X, :, i), mu = view(pr.mu, i),
                         sigma = view(pr.sigma, i, i), chol = chol, w = pr.w, ens = pr.ens,
                         kld = pr.kld, ow = pr.ow, rr = regression_view(pr.rr, i),
                         f_mu = pr.f_mu, f_sigma = pr.f_sigma, f_w = pr.f_w)
end
"""
    struct HighOrderPrior{T1, T2, T3, T4, T5, T6, T7} <: AbstractPriorResult
        pr::T1
        kt::T2
        L2::T3
        S2::T4
        sk::T5
        V::T6
        skmp::T7
    end

Container type for high order prior results in PortfolioOptimisers.jl.

`HighOrderPrior` stores the output of high order prior estimation routines, including low order prior results, cokurtosis tensor, elimination and summation matrices, coskewness tensor, quadratic skewness matrix, and matrix processing estimator. It is used throughout the package to represent validated prior information for portfolio optimisation and analytics involving higher moments.

# Fields

  - `pr`: Prior result for low order moments.
  - `kt`: Cokurtosis tensor.
  - `L2`: Elimination matrix.
  - `S2`: Summation matrix.
  - `sk`: Coskewness tensor.
  - `V`: Negative quadratic skewness matrix.
  - `skmp`: Matrix processing estimator for post-processing quadratic skewness.

# Constructor

    HighOrderPrior(; pr::AbstractPriorResult, kt::Option{<:MatNum} = nothing,
                   L2::Option{<:MatNum} = nothing,
                   S2::Option{<:MatNum} = nothing,
                   sk::Option{<:MatNum} = nothing,
                   V::Option{<:MatNum} = nothing,
                   skmp::Option{<:AbstractMatrixProcessingEstimator} = DenoiseDetoneAlgMatrixProcessing())

Keyword arguments correspond to the fields above.

## Validation

Defining `N = length(pr.mu)`.

  - If any of `kt`, `L2`, or `S2` are provided, all must be provided, non-empty, and `size(kt) == (N^2, N^2)`, `size(L2) == size(S2) == (div(N * (N + 1), 2), N^2)`.
  - If `sk` or `V` are provided, both must be provided, non-empty, and `size(sk) == (N, N^2)`, `size(V) == (N, N)`.

# Examples

```jldoctest
julia> HighOrderPrior(;
                      pr = LowOrderPrior(; X = [0.01 0.02; 0.03 0.04], mu = [0.02, 0.03],
                                         sigma = [0.0001 0.0002; 0.0002 0.0003]), kt = rand(4, 4),
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
struct HighOrderPrior{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <: AbstractPriorResult
    pr::T1
    kt::T2
    L2::T3
    S2::T4
    sk::T5
    V::T6
    skmp::T7
    f_kt::T8
    # chol_kt::T9
    f_sk::T9
    f_V::T10
    function HighOrderPrior(pr::AbstractPriorResult, kt::Option{<:MatNum},
                            L2::Option{<:MatNum}, S2::Option{<:MatNum},
                            sk::Option{<:MatNum}, V::Option{<:MatNum},
                            skmp::Option{<:AbstractMatrixProcessingEstimator},
                            f_kt::Option{<:MatNum}, #chol_kt::Option{<:MatNum},
                            f_sk::Option{<:MatNum}, f_V::Option{<:MatNum})
        N = length(pr.mu)
        kt_flag = isa(kt, MatNum)
        L2_flag = isa(L2, MatNum)
        S2_flag = isa(S2, MatNum)
        if kt_flag || L2_flag || S2_flag
            @argcheck(kt_flag)
            @argcheck(L2_flag)
            @argcheck(S2_flag)
            @argcheck(!isempty(kt))
            @argcheck(!isempty(L2))
            @argcheck(!isempty(S2))
            @argcheck(size(kt) == (N^2, N^2))
            @argcheck(size(L2) == size(S2) == (div(N * (N + 1), 2), N^2))
        end
        sk_flag = isa(sk, MatNum)
        V_flag = isa(V, MatNum)
        if sk_flag || V_flag
            @argcheck(sk_flag)
            @argcheck(V_flag)
            @argcheck(!isempty(sk))
            @argcheck(!isempty(V))
            @argcheck(size(V) == (N, N))
            @argcheck(size(sk) == (N, N^2))
        end
        f_kt_flag = !isnothing(f_kt)
        f_sk_flag = !isnothing(f_sk)
        if f_kt_flag || f_sk_flag
            rr = pr.rr
            @argcheck(!isnothing(rr))
            Nf = size(rr.M, 2)
            if f_kt_flag
                @argcheck(!isempty(f_kt))
                # @argcheck(!isempty(chol_kt))
                assert_matrix_issquare(f_kt, :f_kt)
                @argcheck(Nf^2 == size(f_kt, 1))
                # @argcheck(N^2 == Nfa^2 == size(chol_kt, 2))
            end
            if f_sk_flag
                @argcheck(!isempty(f_sk))
                @argcheck(!isempty(f_V))
                @argcheck(size(f_sk) == (Nf, Nf^2))
                @argcheck(size(f_V) == (Nf, Nf))
            end
        end
        return new{typeof(pr), typeof(kt), typeof(L2), typeof(S2), typeof(sk), typeof(V),
                   typeof(skmp), typeof(f_kt), #typeof(chol_kt), 
                   typeof(f_sk), typeof(f_V)}(pr, kt, L2, S2, sk, V, skmp, f_kt,
                                              #  chol_kt,
                                              f_sk, f_V)
    end
end
function HighOrderPrior(; pr::AbstractPriorResult, kt::Option{<:MatNum} = nothing,
                        L2::Option{<:MatNum} = nothing, S2::Option{<:MatNum} = nothing,
                        sk::Option{<:MatNum} = nothing, V::Option{<:MatNum} = nothing,
                        skmp::Option{<:AbstractMatrixProcessingEstimator} = nothing,
                        f_kt::Option{<:MatNum} = nothing,
                        # chol_kt::Option{<:MatNum} = nothing,
                        f_sk::Option{<:MatNum} = nothing, f_V::Option{<:MatNum} = nothing)
    return HighOrderPrior(pr, kt, L2, S2, sk, V, skmp, f_kt, #chol_kt, 
                          f_sk, f_V)
end

export prior, LowOrderPrior, HighOrderPrior
