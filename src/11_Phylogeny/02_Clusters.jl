"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering estimator types.

All concrete and/or abstract types implementing clustering-based estimation algorithms should be subtypes of `AbstractClustersEstimator`.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersEstimator <: AbstractPhylogenyEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering algorithm types.

All concrete and/or abstract types implementing specific clustering algorithms should be subtypes of `AbstractClustersAlgorithm`.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
abstract type AbstractClustersAlgorithm <: AbstractPhylogenyAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the clustering algorithm `alg` unchanged.

Identity pass-through used when a clustering algorithm is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::AbstractClustersAlgorithm, args...; kwargs...)
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all hierarchical clustering algorithm types.

All concrete and/or abstract types implementing hierarchical clustering algorithms (such as hierarchical agglomerative clustering or DBHT) should be subtypes of `AbstractHierarchicalClusteringAlgorithm`.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`HClustAlgorithm`](@ref)
  - [`DBHT`](@ref)
"""
abstract type AbstractHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all non-hierarchical clustering algorithm types.

All concrete and/or abstract types implementing non-hierarchical clustering algorithms (such as k-means) should be subtypes of `AbstractNonHierarchicalClusteringAlgorithm`.

# Related

  - [`AbstractClustersAlgorithm`](@ref)
  - [`KMeansAlgorithm`](@ref)
"""
abstract type AbstractNonHierarchicalClusteringAlgorithm <: AbstractClustersAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters estimator types.

All concrete and/or abstract types implementing algorithms to estimate the optimal number of clusters should be subtypes of `AbstractOptimalNumberClustersEstimator`.

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
"""
abstract type AbstractOptimalNumberClustersEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all optimal number of clusters algorithm types.

All concrete and/or abstract types implementing specific algorithms for determining the optimal number of clusters should be subtypes of `AbstractOptimalNumberClustersAlgorithm`.

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
"""
abstract type AbstractOptimalNumberClustersAlgorithm <: AbstractAlgorithm end
"""
    const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}

Alias for an integer or optimal number of clusters algorithm.

Matches either a plain integer (specifying the number of clusters directly) or an [`AbstractOptimalNumberClustersAlgorithm`](@ref) (which determines the optimal number automatically).

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
"""
const Int_ONC = Union{<:Integer, <:AbstractOptimalNumberClustersAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all clustering result types.

All concrete and/or abstract types representing the result of a clustering estimation should be subtypes of `AbstractClusteringResult`.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClustersAlgorithm`](@ref)
"""
abstract type AbstractClusteringResult <: AbstractPhylogenyResult end
"""
    const ClTypes = Union{<:Clustering.ClusteringResult, <:Clustering.Hclust}

Alias for clustering result types from the Clustering.jl package.

Matches either a `Clustering.ClusteringResult` or `Clustering.Hclust`. Used internally to accept output from the Clustering.jl library for both flat and hierarchical clustering results.

# Related

  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
const ClTypes = Union{<:Clustering.ClusteringResult, <:Clustering.Hclust}
"""
$(DocStringExtensions.TYPEDEF)

Carries a clustering of the asset universe together with the matrices it was computed from.

Holds the result of either family, because `res` takes both shapes [`clusterise`](@ref) produces: a `Clustering.Hclust` from a hierarchical algorithm and a `Clustering.ClusteringResult` from a non-hierarchical one. `k` is the number of clusters selected by an [`AbstractOptimalNumberClustersEstimator`](@ref), and [`assignments`](@ref) is what reads a cluster label per asset off the pair — cutting the tree at `k` on the hierarchical branch, reading the field on the other.

# `D` is what the clustering saw, unless `P` is present

`S` and `D` are the similarity and distance matrices of the universe, and an ordinary [`ClustersEstimator`](@ref) clusters `D` itself and leaves `P` as `nothing`. A [`NetworkClustersEstimator`](@ref) instead accumulates a **pseudo-distance** matrix out of the network structure and clusters *that*, so it fills `P` and `res` is the clustering of `P`. Both matrices are kept because a consumer wants the network's clustering and the universe's own distances.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Clusters(;
        res::ClTypes,
        S::MatNum,
        D::MatNum,
        P::Option{<:MatNum} = nothing,
        k::Integer
    ) -> Clusters

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:S])
  - $(val_dict[:D])
  - $(val_dict[:S_D])
  - $(val_dict[:S_P])
  - $(val_dict[:ck])

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`NetworkClustersEstimator`](@ref)
  - [`clusterise`](@ref)
  - [`assignments`](@ref)
"""
@concrete struct Clusters <: AbstractClusteringResult
    """
    $(field_dict[:clres])
    """
    res
    """
    $(field_dict[:S])
    """
    S
    """
    $(field_dict[:D])
    """
    D
    """
    $(field_dict[:clP])
    """
    P
    """
    $(field_dict[:ck])
    """
    k
    function Clusters(res::ClTypes, S::MatNum, D::MatNum, P::Option{<:MatNum}, k::Integer)
        @argcheck(!isempty(S), IsEmptyError)
        @argcheck(!isempty(D), IsEmptyError)
        @argcheck(size(S) == size(D), DimensionMismatch)
        if !isnothing(P)
            @argcheck(!isempty(P), IsEmptyError)
            @argcheck(size(S) == size(P), DimensionMismatch)
        end
        @argcheck(one(k) <= k, DomainError)
        return new{typeof(res), typeof(S), typeof(D), typeof(P), typeof(k)}(res, S, D, P, k)
    end
end
function Clusters(; res::ClTypes, S::MatNum, D::MatNum, P::Option{<:MatNum} = nothing,
                  k::Integer)::Clusters
    return Clusters(res, S, D, P, k)
end
"""
    clusterise(cle::AbstractClusteringResult, args...; kwargs...)

Return the clustering result `cle` unchanged.

Identity pass-through, so that every consumer takes an estimator or a precomputed result through one call. A [`ClustersEstimator`](@ref) reaching this function has already been run; a result reaching it is not run again.

# Arguments

  - `cle`: Clustering result.
  - `args...`: Additional positional arguments, ignored.
  - `kwargs...`: Additional keyword arguments, ignored.

# Returns

  - `cle::AbstractClusteringResult`: The input, unchanged.

# Related

  - [`AbstractClusteringResult`](@ref)
  - [`Clusters`](@ref)
  - [`ClustersEstimator`](@ref)
"""
function clusterise(cle::AbstractClusteringResult, args...; kwargs...)
    return cle
end
"""
$(DocStringExtensions.TYPEDEF)

Picks the number of clusters at which the within-cluster dispersion curve bends most sharply.

The two-difference gap statistic. It reads the second-order difference of the within-cluster dispersion across cluster counts, so it finds the elbow of that curve without the Monte Carlo simulation the original gap statistic needs.

# Mathematical definition

```math
\\begin{align}
c^{\\star} &= \\underset{c}{\\arg\\max} \\left(W_{c+2} + W_{c} - 2 W_{c+1}\\right)\\,,\\\\
&\\mathrm{s.t.} \\quad 1 \\leq c \\leq \\sqrt{N}\\,,\\\\
W_{c} &= \\sum_{j=1}^{c} g\\left(\\left\\{d_{uv} : u, v \\in \\mathcal{C}_{j},\\, u < v\\right\\}\\right)\\,.
\\end{align}
```

Where:

  - ``c^{\\star}``: Selected number of clusters.
  - ``W_{c}``: Within-cluster dispersion of a cut into ``c`` clusters.
  - ``\\mathcal{C}_{j}``: Set of assets in the ``j``-th cluster of that cut.
  - ``d_{uv}``: Entry of the distance matrix the clustering ran on.
  - ``g``: Vector-to-scalar measure `alg`, applied to one cluster's pairwise distances. A cluster of one asset has no pairwise distance and contributes ``0``.
  - ``N``: Number of assets.

The selected ``c`` is the **left end** of the triple, not its centre. That is the source's own statement of the problem, and the code maximises it as written.

# The measure decides what "dispersion" means, and the default is not the source's

The source takes ``W_{c}`` as the **mean** of a cluster's pairwise distances, which is `alg = MeanValue()`. The default `alg = StandardisedValue()` divides that mean by the corrected standard deviation of the same distances, which is a different statistic and selects a different ``c``: on a 400x40 sample the two answer **6 and 4**.

!!! warning

    `StandardisedValue()` is undefined for a cluster of exactly two assets. Two assets carry **one** pairwise distance, and the corrected standard deviation of a single value is `NaN`, so ``W_{c}`` is `NaN` for every cut in which such a cluster appears. The gap series is then `NaN` throughout and the returned `k` is the **length of that series** rather than a maximiser of anything. A universe with a tightly correlated pair reaches this: over 20 assets with two such pairs every cut from `2` to `6` clusters carries a two-asset cluster, the default answers `k = 4` off the fallback, and `MeanValue()` answers `k = 2` off a real argmax. Pass `alg = MeanValue()` when the universe may cluster that finely.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SecondOrderDifference(;
        alg::Num_VecToScaM = StandardisedValue()
    ) -> SecondOrderDifference

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> SecondOrderDifference()
SecondOrderDifference
  alg ┼ StandardisedValue
      │   mv ┼ MeanValue
      │      │   w ┴ nothing
      │   sv ┼ StdValue
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`SilhouetteScore`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`MeanValue`](@ref)
  - [`StandardisedValue`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.2.1.2, Equation 12.15.
  - $(ref_dict[:yue2008])
  - $(ref_dict[:tibshirani2001])
"""
@propagatable @concrete struct SecondOrderDifference <:
                               AbstractOptimalNumberClustersAlgorithm
    """
    $(field_dict[:vsalg])
    """
    @fprop alg
    function SecondOrderDifference(alg::Num_VecToScaM)
        return new{typeof(alg)}(alg)
    end
end
function SecondOrderDifference(;
                               alg::Num_VecToScaM = StandardisedValue())::SecondOrderDifference
    return SecondOrderDifference(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Picks the number of clusters whose assets sit best inside their own cluster.

Each asset gets a silhouette, which compares how far it sits from its own cluster against how far it sits from the nearest other one. `alg` reduces the whole vector of them to one number per cluster count, and the count with the largest number wins.

# Mathematical definition

```math
\\begin{align}
s_{i} &= \\frac{b_{i} - a_{i}}{\\max\\left(a_{i},\\, b_{i}\\right)}\\,,\\\\
a_{i} &= \\sum_{j \\in \\mathcal{C}_{I},\\, j \\neq i} \\frac{d_{ij}}{\\left|\\mathcal{C}_{I}\\right| - 1}\\,,\\\\
b_{i} &= \\underset{J \\neq I}{\\min} \\sum_{j \\in \\mathcal{C}_{J}} \\frac{d_{ij}}{\\left|\\mathcal{C}_{J}\\right|}\\,,\\\\
c^{\\star} &= \\underset{c}{\\arg\\max} \\; g\\left(\\boldsymbol{s}\\right)\\,, \\quad 1 \\leq c \\leq \\sqrt{N}\\,.
\\end{align}
```

Where:

  - ``s_{i}``: Silhouette of asset ``i``, in ``\\left[-1,\\, 1\\right]``. It is ``1`` when the asset is clustered well and ``-1`` when it is clustered wrongly.
  - ``a_{i}``: Mean distance from asset ``i`` to the other members of its own cluster ``\\mathcal{C}_{I}``.
  - ``b_{i}``: Smallest mean distance from asset ``i`` to the members of any other cluster.
  - ``d_{ij}``: Entry of the distance matrix the clustering ran on.
  - ``\\boldsymbol{s}``: Vector of silhouettes over all assets, one per asset.
  - ``g``: Vector-to-scalar measure `alg`.
  - ``c^{\\star}``: Selected number of clusters.
  - ``N``: Number of assets.

# The default reduction is the source's standardised score

`alg = StandardisedValue()` divides the mean of ``\\boldsymbol{s}`` by its corrected standard deviation, which is the source's *quality measure* term for term. The standardisation is what makes scores comparable across cluster counts, so it is the reduction the selection wants; the field is a knob only because a caller may want the plain mean instead. The two disagree in general, and the standardised form is the one this default computes.

Unlike [`SecondOrderDifference`](@ref), the reduction here runs over **one vector of length** ``N``, never over one cluster at a time, so no cluster size makes it undefined.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SilhouetteScore(;
        alg::Num_VecToScaM = StandardisedValue()
    ) -> SilhouetteScore

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> SilhouetteScore()
SilhouetteScore
  alg ┼ StandardisedValue
      │   mv ┼ MeanValue
      │      │   w ┴ nothing
      │   sv ┼ StdValue
      │      │           w ┼ nothing
      │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`OptimalNumberClusters`](@ref)
  - [`SecondOrderDifference`](@ref)
  - [`VectorToScalarMeasure`](@ref)
  - [`StandardisedValue`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.2.1.1, Equations 12.13-12.14.
  - $(ref_dict[:rousseeuw1987])
  - $(ref_dict[:lopezdeprado2019])
"""
@propagatable @concrete struct SilhouetteScore <: AbstractOptimalNumberClustersAlgorithm
    """
    $(field_dict[:vsalg])
    """
    @fprop alg
    function SilhouetteScore(alg::Num_VecToScaM)
        return new{typeof(alg)}(alg)
    end
end
function SilhouetteScore(; alg::Num_VecToScaM = StandardisedValue())::SilhouetteScore
    return SilhouetteScore(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Decides how many clusters to cut a dendrogram or a partition into.

Pairs a selection rule with a ceiling on the answer. `alg` is either an [`AbstractOptimalNumberClustersAlgorithm`](@ref), which chooses the count from the data, or an `Integer`, which states it outright.

# A stated `k` is a request, not a guarantee

`max_k` caps every branch, and the cap is itself capped: the ceiling in force is `min(floor(Int, sqrt(N)), max_k)`, so a `max_k` above ``\\sqrt{N}`` buys nothing. A stated `k` above that ceiling is lowered to it.

A stated `k` is also checked against the tree on the hierarchical branch. A cut into `k` clusters that no node of the dendrogram supports is not honoured; [`optimal_number_clusters`](@ref) searches upward and downward for the nearest `k` that is, and takes the nearer of the two. The data-driven branches apply the same test through [`valid_k_clusters`](@ref), which walks the scoring array from its largest entry down until one passes.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OptimalNumberClusters(;
        max_k::Option{<:Integer} = nothing,
        alg::Int_ONC = SecondOrderDifference()
    ) -> OptimalNumberClusters

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:max_k])
  - $(val_dict[:kalg])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> OptimalNumberClusters(; max_k = 10)
OptimalNumberClusters
  max_k ┼ Int64: 10
    alg ┼ SecondOrderDifference
        │   alg ┼ StandardisedValue
        │       │   mv ┼ MeanValue
        │       │      │   w ┴ nothing
        │       │   sv ┼ StdValue
        │       │      │           w ┼ nothing
        │       │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`AbstractOptimalNumberClustersAlgorithm`](@ref)
  - [`SecondOrderDifference`](@ref)
  - [`SilhouetteScore`](@ref)
  - [`optimal_number_clusters`](@ref)
  - [`valid_k_clusters`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.2.1.
"""
@propagatable @concrete struct OptimalNumberClusters <:
                               AbstractOptimalNumberClustersEstimator
    """
    $(field_dict[:max_k])
    """
    max_k
    """
    $(field_dict[:kalg])
    """
    @fprop alg
    function OptimalNumberClusters(max_k::Option{<:Integer}, alg::Int_ONC)
        if !isnothing(max_k)
            @argcheck(one(max_k) <= max_k, DomainError)
        end
        if isa(alg, Integer)
            @argcheck(one(alg) <= alg, DomainError)
        end
        return new{typeof(max_k), typeof(alg)}(max_k, alg)
    end
end
function OptimalNumberClusters(; max_k::Option{<:Integer} = nothing,
                               alg::Int_ONC = SecondOrderDifference())::OptimalNumberClusters
    return OptimalNumberClusters(max_k, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Builds a dendrogram by merging the two nearest clusters until one remains.

`linkage` is the criterion that says how near two clusters are once they hold more than one asset. `:single` takes the smallest distance between their members and `:complete` the largest; `:average` takes the mean over all cross pairs; `:ward` merges the pair that raises the total within-cluster variance least. The symbol is passed straight to `Clustering.hclust`, so every criterion that package accepts is available.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HClustAlgorithm(;
        linkage::Symbol = :ward
    ) -> HClustAlgorithm

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> HClustAlgorithm(; linkage = :average)
HClustAlgorithm
  linkage ┴ Symbol: :average
```

# Related

  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`ClustersEstimator`](@ref)
  - [`DBHT`](@ref)
  - [`clusterise`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.1.1, Equations 12.6-12.12.
  - $(ref_dict[:mullner2011])
"""
@concrete struct HClustAlgorithm <: AbstractHierarchicalClusteringAlgorithm
    """
    Linkage method for hierarchical clustering from [`Clustering.jl`](https://juliastats.org/Clustering.jl/stable/hclust.html).
    """
    linkage
    function HClustAlgorithm(linkage::Symbol)
        return new{typeof(linkage)}(linkage)
    end
end
function HClustAlgorithm(; linkage::Symbol = :ward)::HClustAlgorithm
    return HClustAlgorithm(linkage)
end
"""
$(DocStringExtensions.TYPEDEF)

Turns a return matrix into a clustering of the asset universe.

Holds the four steps in the order they run: `ce` estimates the covariance, `de` turns it into a distance matrix, `alg` clusters that matrix, and `onc` decides how many clusters to keep. [`clusterise`](@ref) runs them and returns a [`Clusters`](@ref).

# The universe is clustered, not the observations

Every step reads an `assets x assets` matrix, so the number of observations leaves the picture at `ce`. That is why `alg` and `onc` carry no `@fprop` tag: an [`ObsWeights`](@ref) weights observations, and there is no observation left for it to weight once `de` has run. `ce` and `de` are tagged, and they are where a weight belongs.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ClustersEstimator(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        alg::AbstractClustersAlgorithm = HClustAlgorithm(),
        onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters()
    ) -> ClustersEstimator

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `de`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> ClustersEstimator()
ClustersEstimator
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┴ FullMoment()
      │   mp ┼ MatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  alg ┼ HClustAlgorithm
      │   linkage ┴ Symbol: :ward
  onc ┼ OptimalNumberClusters
      │   max_k ┼ nothing
      │     alg ┼ SecondOrderDifference
      │         │   alg ┼ StandardisedValue
      │         │       │   mv ┼ MeanValue
      │         │       │      │   w ┴ nothing
      │         │       │   sv ┼ StdValue
      │         │       │      │           w ┼ nothing
      │         │       │      │   corrected ┴ Bool: true
```

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractNonHierarchicalClusteringAlgorithm`](@ref)
  - [`AbstractOptimalNumberClustersEstimator`](@ref)
  - [`Clusters`](@ref)
  - [`clusterise`](@ref)
  - [`factory`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 12.1.1.
"""
@propagatable @concrete struct ClustersEstimator <: AbstractClustersEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    """
    $(field_dict[:de])
    """
    @fprop de
    """
    $(field_dict[:clalg])
    """
    alg
    """
    $(field_dict[:onc])
    """
    onc
    function ClustersEstimator(ce::StatsBase.CovarianceEstimator,
                               de::AbstractDistanceEstimator,
                               alg::AbstractClustersAlgorithm,
                               onc::AbstractOptimalNumberClustersEstimator)
        return new{typeof(ce), typeof(de), typeof(alg), typeof(onc)}(ce, de, alg, onc)
    end
end
function ClustersEstimator(;
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           de::AbstractDistanceEstimator = Distance(;
                                                                    alg = CanonicalDistance()),
                           alg::AbstractClustersAlgorithm = HClustAlgorithm(),
                           onc::AbstractOptimalNumberClustersEstimator = OptimalNumberClusters())::ClustersEstimator
    return ClustersEstimator(ce, de, alg, onc)
end
"""
    const ClE_Cl = Union{<:AbstractClustersEstimator, <:AbstractClusteringResult}

Alias for a clustering estimator or result.

Matches either an [`AbstractClustersEstimator`](@ref) or an [`AbstractClusteringResult`](@ref). Used for dispatch in phylogeny and network estimation workflows that accept either form.

# Related

  - [`AbstractClustersEstimator`](@ref)
  - [`AbstractClusteringResult`](@ref)
"""
const ClE_Cl = Union{<:AbstractClustersEstimator, <:AbstractClusteringResult}

export Clusters, clusterise, SecondOrderDifference, SilhouetteScore, OptimalNumberClusters,
       HClustAlgorithm, ClustersEstimator
