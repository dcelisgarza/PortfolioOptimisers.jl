"""
```julia
abstract type AbstractPriorEstimator <: AbstractEstimator end
```

Abstract supertype for all prior estimators.

`AbstractPriorEstimator` is the base type for all estimators that compute prior information from asset and/or factor returns. All concrete prior estimators should subtype this type to ensure a consistent interface for prior computation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractHighOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractPriorEstimator <: AbstractEstimator end
"""
```julia
abstract type AbstractLowOrderPriorEstimator <: AbstractPriorEstimator end
```

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
```julia
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
```

Low order prior estimator using only asset returns.

`AbstractLowOrderPriorEstimator_A` is the base type for estimators that compute low order moments (mean and covariance) using only asset returns data. All concrete asset-only prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_A <: AbstractLowOrderPriorEstimator end
"""
```julia
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
```

Low order prior estimator using factor returns.

`AbstractLowOrderPriorEstimator_F` is the base type for estimators that compute low order moments (mean and covariance) requiring the use of both asset and factor returns data. All concrete factor-adjusted prior estimators should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_F <: AbstractLowOrderPriorEstimator end
"""
```julia
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
```

Low order prior estimator using both asset and factor returns.

`AbstractLowOrderPriorEstimator_AF` is the base type for estimators that compute low order moments (mean and covariance) using both asset and optionally factor returns data. All concrete prior estimators which may optionally use factor returns should subtype this type.

# Related

  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
"""
abstract type AbstractLowOrderPriorEstimator_AF <: AbstractLowOrderPriorEstimator end
"""
```julia
const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                  AbstractLowOrderPriorEstimator_AF}
```

Union type for asset-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_AF` is a union type that allows dispatch on both asset-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using asset returns, with or without factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                  AbstractLowOrderPriorEstimator_AF}
"""
```julia
const AbstractLowOrderPriorEstimator_F_AF = Union{AbstractLowOrderPriorEstimator_F,
                                                  AbstractLowOrderPriorEstimator_AF}
```

Union type for factor-only and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_F_AF` is a union type that allows dispatch on both factor-only and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using factor returns, with or without asset returns.

# Related

  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_F_AF = Union{AbstractLowOrderPriorEstimator_F,
                                                  AbstractLowOrderPriorEstimator_AF}
"""
```julia
const AbstractLowOrderPriorEstimator_A_F_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                    AbstractLowOrderPriorEstimator_F,
                                                    AbstractLowOrderPriorEstimator_AF}
```

Union type for asset-only, factor-only, and asset-and-factor low order prior estimators.

`AbstractLowOrderPriorEstimator_A_F_AF` is a union type that allows dispatch on asset-only, factor-only, and asset-and-factor prior estimators. This is useful for generic algorithms that operate on estimators using any combination of asset and factor returns.

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`AbstractLowOrderPriorEstimator_F`](@ref)
  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_A_AF`](@ref)
  - [`AbstractLowOrderPriorEstimator_F_AF`](@ref)
"""
const AbstractLowOrderPriorEstimator_A_F_AF = Union{AbstractLowOrderPriorEstimator_A,
                                                    AbstractLowOrderPriorEstimator_F,
                                                    AbstractLowOrderPriorEstimator_AF}
"""
```julia
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
```

Abstract supertype for high order prior estimators.

`AbstractHighOrderPriorEstimator` is the base type for estimators that compute high order moments (such as coskewness and cokurtosis) from asset and/or factor returns. All concrete high order prior estimators should subtype this type to ensure a consistent interface for higher moment estimation and integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`AbstractLowOrderPriorEstimator`](@ref)
  - [`prior`](@ref)
"""
abstract type AbstractHighOrderPriorEstimator <: AbstractPriorEstimator end
"""
```julia
abstract type AbstractPriorResult <: AbstractResult end
```

Abstract supertype for all prior result types.

`AbstractPriorResult` is the base type for all result objects produced by prior estimators, containing computed prior information such as moments, asset returns, and factor returns. All concrete prior result types should subtype this to ensure a consistent interface for integration with portfolio optimisation workflows.

# Related

  - [`AbstractPriorEstimator`](@ref)
  - [`prior`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractPriorResult <: AbstractResult end
"""
```julia
prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
```

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
```julia
prior(pr::AbstractPriorResult, args...; kwargs...)
```

Propagate or pass through prior result objects.

`prior` returns the input prior result object unchanged. This method is used to propagate already constructed prior results or enable uniform interface handling in workflows that accept either estimators or results.

# Arguments

  - `pr`: Prior result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `res::AbstractPriorResult`: The input prior result object, unchanged.

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
```julia
clusterise(cle::ClusteringEstimator, pr::AbstractPriorResult; kwargs...)
```

Clusterise asset or factor returns from a prior result using a clustering estimator.

`clusterise` applies the specified clustering estimator to the asset returns matrix contained in the prior result object, producing a clustering result for use in phylogeny analysis, constraint generation, or portfolio construction.

# Arguments

  - `cle`: Clustering estimator.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the clustering estimator.

# Returns

  - `res::AbstractClusteringResult`: Result object containing clustering information.

# Related

  - [`ClusteringEstimator`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`clusterise`](@ref)
"""
function clusterise(cle::ClusteringEstimator, pr::AbstractPriorResult; kwargs...)
    return clusterise(cle, pr.X; kwargs...)
end
"""
```julia
phylogeny_matrix(necle::Union{<:AbstractNetworkEstimator, <:AbstractClusteringEstimator,
                              <:AbstractClusteringResult}, pr::AbstractPriorResult;
                 kwargs...)
```

Compute the phylogeny matrix from asset returns in a prior result using a network or clustering estimator.

`phylogeny_matrix` applies the specified network or clustering estimator to the asset returns matrix contained in the prior result object, producing a phylogeny matrix for use in constraint generation, centrality analysis, or portfolio construction.

# Arguments

  - `necle`: Network estimator, clustering estimator, or clustering result.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments passed to the estimator.

# Returns

  - `res::PhylogenyResult`: Result object containing the phylogeny matrix.

# Related

  - [`NetworkEstimator`](@ref)
  - [`ClusteringEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(necle::Union{<:AbstractNetworkEstimator,
                                       <:AbstractClusteringEstimator,
                                       <:AbstractClusteringResult}, pr::AbstractPriorResult;
                          kwargs...)
    return phylogeny_matrix(necle, pr.X; kwargs...)
end
"""
```julia
centrality_vector(necte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)
```

Compute the centrality vector for a centrality estimator and prior result.

`centrality_vector` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, returning centrality scores for each asset.

# Arguments

  - `necte`: Centrality estimator.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `res::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(necte::CentralityEstimator, pr::AbstractPriorResult; kwargs...)
    return centrality_vector(necte, pr.X; kwargs...)
end
"""
```julia
centrality_vector(ne::Union{<:AbstractNetworkEstimator, <:AbstractClusteringEstimator,
                            <:AbstractClusteringResult}, cent::AbstractCentralityAlgorithm,
                  pr::AbstractPriorResult; kwargs...)
```

Compute the centrality vector for a network or clustering estimator and centrality algorithm.

`centrality_vector` constructs the phylogeny matrix from the asset returns in the prior result, builds a graph, and computes node centrality scores using the specified centrality algorithm.

# Arguments

  - `ne`: Network estimator, clustering estimator, or clustering result.
  - `cent`: Centrality algorithm.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `res::PhylogenyResult`: Result object containing the centrality vector.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(ne::Union{<:AbstractNetworkEstimator,
                                     <:AbstractClusteringEstimator,
                                     <:AbstractClusteringResult},
                           cent::AbstractCentralityAlgorithm, pr::AbstractPriorResult;
                           kwargs...)
    return centrality_vector(ne, cent, pr.X; kwargs...)
end
"""
```julia
average_centrality(ne::Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult},
                   cent::AbstractCentralityAlgorithm, w::AbstractVector,
                   pr::AbstractPriorResult; kwargs...)
```

Compute the weighted average centrality for a network or phylogeny result.

`average_centrality` computes the centrality vector using the specified network or phylogeny estimator and centrality algorithm, then returns the weighted average using the provided portfolio weights.

# Arguments

  - `ne`: Network estimator or phylogeny result.
  - `cent`: Centrality algorithm.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Real`: Weighted average centrality.

# Related

  - [`NetworkEstimator`](@ref)
  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(ne::Union{<:AbstractPhylogenyEstimator,
                                      <:AbstractPhylogenyResult},
                            cent::AbstractCentralityAlgorithm, w::AbstractVector,
                            pr::AbstractPriorResult; kwargs...)
    return dot(centrality_vector(ne, cent, pr.X; kwargs...).X, w)
end
"""
```julia
average_centrality(cte::CentralityEstimator, w::AbstractVector, pr::AbstractPriorResult;
                   kwargs...)
```

Compute the weighted average centrality for a centrality estimator.

`average_centrality` applies the centrality algorithm in the estimator to the network constructed from the asset returns in the prior result, then returns the weighted average using the provided portfolio weights.

# Arguments

  - `cte`: Centrality estimator.
  - `w`: Portfolio weights vector.
  - `pr`: Prior result object.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `ac::Real`: Weighted average centrality.

# Related

  - [`CentralityEstimator`](@ref)
  - [`centrality_vector`](@ref)
  - [`average_centrality`](@ref)
"""
function average_centrality(cte::CentralityEstimator, w::AbstractVector,
                            pr::AbstractPriorResult; kwargs...)
    return average_centrality(cte.ne, cte.cent, w, pr.X; kwargs...)
end

export prior
