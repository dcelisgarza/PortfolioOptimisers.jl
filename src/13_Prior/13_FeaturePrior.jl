"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for feature matrix estimators.

An `AbstractFeatureMatrixEstimator` is a **producer**: it computes the derived feature matrix `Z` that [`FeaturePrior`](@ref) attaches to a prior result. Producers exist so that a feature matrix on an [`AbstractPriorResult`](@ref) means exactly one thing — *someone declared these are features* — rather than being a second live spelling of a matrix that is also reachable elsewhere on the same result.

All producers implement [`feature_matrix`](@ref).

# Related

  - [`feature_matrix`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
"""
abstract type AbstractFeatureMatrixEstimator <: AbstractEstimator end
"""
    feature_matrix(ze::MatNum_Arr3Num, pr::AbstractPriorResult, X, F, sets; kwargs...)
    feature_matrix(ze::RegressionFeatures, pr::AbstractPriorResult, X, F, sets; kwargs...)

Compute the derived feature matrix.

Returns the `Z` that [`FeaturePrior`](@ref) writes onto the prior result. `Z` is **canonically assets-major** — `assets × features` when static, `observations × assets × features` when time-varying — and is validated as such by [`LowOrderPrior`](@ref).

A producer declares a matrix to be features and nothing else. It does *not* declare the feature axis to be the asset axis, even when it happens to build a square matrix: the prior carrier has no squareness vocabulary, because a derived `Z` is never sliced down its feature axis. A producer runs inside `prior(pe, X, F; …)`, so it refits on whatever universe the subproblem hands it — which recomputes a square matrix on that universe rather than cutting one describing a larger one. Exogenous square structure travels on the *data* carrier instead, where [`features_are_assets`](@ref) derives squareness from `nz` against `nx`.

`pr` is the **already-computed** wrapped prior result, not an estimator. That ordering is required: [`RegressionFeatures`](@ref) reads `pr.rr`, which does not exist until the wrapped prior has run.

# Arguments

  - `ze`: Feature matrix producer, or a literal feature matrix.
  - `pr`: Prior result returned by the wrapped estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Factor returns matrix, or `nothing`.
  - `sets`: Asset sets supplying asset names, or `nothing`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Returns

  - `Z::MatNum_Arr3Num`.

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`FeaturePrior`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`features_are_assets`](@ref)
"""
function feature_matrix end
function feature_matrix(ze::MatNum_Arr3Num, ::AbstractPriorResult, args...; kwargs...)
    return ze
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer that reads the regression loadings off the wrapped prior result.

`RegressionFeatures` treats `pr.rr.L` — the coordinate system a factor model places each asset in — as an `assets × features` feature matrix. It is the cheapest real feature source in the library: a factor prior already computes it, and already refits it per fold, so the features track the fold with no extra plumbing.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, set by [`DimensionReductionRegression`](@ref): the low-dimensional coordinate system the asset actually lives in. `M` is the *reconstructed* full-factor loadings. `pr.rr.L` always resolves — [`Regression`](@ref) swaps in `M` when `L` is unset — so this producer needs no branch and works behind every regression estimator.

Both are assets-major and both are row-sliced by `port_opt_view(re, i)`, so the carried layout holds with no transpose. Its feature axis is the reduced dimensions, which are not assets.

# Validation

  - The wrapped prior must carry a regression (see [`assert_prior_regression`](@ref)). Nesting order does not matter: every wrapping estimator forwards `rr` and the factor block `fpr` (ADR 0046), so `FeaturePrior(; pe = BlackLittermanPrior(; pe = FactorPrior(…)), ze = RegressionFeatures())` and `BlackLittermanPrior(; pe = FeaturePrior(; pe = FactorPrior(…), ze = RegressionFeatures()))` both resolve. What throws is a wrapped prior that never computed a regression at all, such as `EmpiricalPrior`.

# Examples

```jldoctest
julia> RegressionFeatures()
RegressionFeatures()
```

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`feature_matrix`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FactorPrior`](@ref)
  - [`Regression`](@ref)
  - [`assert_prior_regression`](@ref)
"""
struct RegressionFeatures <: AbstractFeatureMatrixEstimator end
function feature_matrix(::RegressionFeatures, pr::AbstractPriorResult, args...; kwargs...)
    assert_prior_regression(pr, :pe)
    return pr.rr.L
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny feature algorithms.

A phylogeny feature algorithm is the rule turning the graph a [`PhylogenyFeatures`](@ref) source describes into an `assets × assets` feature matrix, `Z[i, k] = f(hops(i, k))`. The family is open: a user needing a different rule defines a member and a [`phylogeny_features`](@ref) method for it.

A different *fall-off* needs neither — that is [`AbstractSeparationDecayAlgorithm`](@ref), which [`GradedNeighbourhood`](@ref) carries as a field.

Every member includes **self**, so `f(0)` is the top of its scale — see [`PhylogenyFeatures`](@ref) for why the diagonal is load-bearing rather than cosmetic.

# Related

  - [`BinaryNeighbourhood`](@ref)
  - [`GradedNeighbourhood`](@ref)
  - [`PhylogenyFeatures`](@ref)
  - [`phylogeny_features`](@ref)
"""
abstract type AbstractPhylogenyFeatureAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Phylogeny feature algorithm giving each asset its `n`-hop neighbourhood indicator.

`Z[i, k] = 1` when asset `k` is within `n` hops of asset `i` and `0` otherwise, with `Z[i, i] = 1`. This is exactly [`phylogeny_matrix`](@ref)'s output with the diagonal restored, so the binary variant reuses the existing routine rather than reimplementing it — the reuse map decision 6 names.

# Related

  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`GradedNeighbourhood`](@ref)
  - [`PhylogenyFeatures`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
struct BinaryNeighbourhood <: AbstractPhylogenyFeatureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Phylogeny feature algorithm grading each asset's neighbourhood by hop count.

`Z[i, k] = decay(hops(i, k))` inside `n` hops and `0` beyond, so the score falls off with distance instead of flattening to an indicator. It is strictly richer than [`BinaryNeighbourhood`](@ref), which is its `n = 1` special case up to scale: `phylogeny_matrix` accumulates `sum(A^i for i in 0:n)` and then `clamp!(P, 0, 1)` **destroys the step count**, which is the information this algorithm keeps.

The score is a function of the hop count rather than of the un-clamped walk count `sum(A^i)`. A walk count is degree-biased — a hub accumulates walks combinatorially — so two assets' scores would encode how busy their neighbourhoods are as much as how close they are.

# The two knobs

`decay` shapes the fall-off; `n` on the source [`NetworkEstimator`](@ref) truncates it. They are deliberately separate — an exponential never reaches zero, so a budget cannot be expressed as a fall-off — and `n` is the only place truncation happens. Under the default [`LinearDecay`](@ref) the two coincide in appearance: a direct neighbour scores `n`, a two-hop neighbour `n - 1`, the asset itself `n + 1`, and the score would hit `0` exactly one hop past the budget that already cut it. Under [`ExponentialDecay`](@ref) or [`ReciprocalDecay`](@ref) the diagonal is `1` and the fall-off is set by the member's own parameter, independently of how far `n` looks.

Because no decay emits zero inside the budget, **a zero entry means unreachable-or-beyond-`n` and nothing else**.

# Unreachable pairs

`hops` is `typemax` for an unreachable vertex, so the budget comparison both selects the `0` and **guards the decay call**: `separation_decay` is never evaluated at `typemax`. The guard is load-bearing rather than tidy — `ReciprocalDecay` overflows `1 + d` there, and for a fractional `power` that is a `DomainError` rather than a discarded number.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GradedNeighbourhood(;
        decay::AbstractSeparationDecayAlgorithm = LinearDecay()
    ) -> GradedNeighbourhood

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> GradedNeighbourhood(; decay = ExponentialDecay(; rate = 0.5))
GradedNeighbourhood
  decay ┼ ExponentialDecay
        │   rate ┴ Float64: 0.5
```

# Related

  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`BinaryNeighbourhood`](@ref)
  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`PhylogenyFeatures`](@ref)
  - [`calc_adjacency`](@ref)
"""
@concrete struct GradedNeighbourhood <: AbstractPhylogenyFeatureAlgorithm
    """
    $(field_dict[:sdecay])
    """
    decay
    function GradedNeighbourhood(decay::AbstractSeparationDecayAlgorithm)
        return new{typeof(decay)}(decay)
    end
end
function GradedNeighbourhood(;
                             decay::AbstractSeparationDecayAlgorithm = LinearDecay())::GradedNeighbourhood
    return GradedNeighbourhood(decay)
end
"""
    phylogeny_features(alg::BinaryNeighbourhood, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)
    phylogeny_features(alg::GradedNeighbourhood, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)
    phylogeny_features(alg::AbstractPhylogenyFeatureAlgorithm,
                       pl::AbstractClustersEstimator, X::MatNum; kwargs...)

Turn a graph source into a square `assets × assets` feature matrix.

The kernel behind [`PhylogenyFeatures`](@ref). Every method returns a `Float64` matrix — not the `Int` or `BitMatrix` the phylogeny routines produce — so that [`AngularDist`](@ref) keeps its BLAS `gemm` path.

# The source is always refit

`pl` is an estimator — a [`NetworkEstimator`](@ref) or a [`ClustersEstimator`](@ref) — never a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref), because an Estimator does not hold a Result (see `CONTEXT.md` §1). The structure is therefore rebuilt from `X` on every call, so it tracks a cross-validation fold or a meta-optimiser's subproblem instead of describing a universe it no longer sees.

# `alg` applies to a graph, and is inert for a partition

A **graph** source has hop structure, so `alg` selects the decay over it. Its adjacency is `0`/`1` by construction — `Graphs.adjacency_matrix` of a minimum spanning tree or a PMFG — so there are no edge weights to preserve or discard.

A **partition** has none: two assets are in the same cluster or they are not, and there is nothing between them to decay. Both algorithms therefore give the same co-membership matrix, and `alg` is inert rather than an error — the same treatment `FeatureDistance`'s collapse `alg` gets on a static feature matrix.

# The diagonal

`Z[i, i]` is the top of the scale, never zero: `1` for [`BinaryNeighbourhood`](@ref) and for any clustering source, and `separation_decay(decay, 0, n)` for [`GradedNeighbourhood`](@ref) over a graph — `n + 1` under the default [`LinearDecay`](@ref), `1` for the members that pin `f(0) = 1`. That the diagonal is maximal is a contract on [`AbstractSeparationDecayAlgorithm`](@ref), checked before the loop by [`assert_separation_decay`](@ref).

# Arguments

  - `alg`: Phylogeny feature algorithm.
  - `pl`: Structure source — a network estimator (a graph) or a clustering estimator (a partition).
  - `X`: Asset returns matrix `observations × assets`.
  - `kwargs...`: Additional keyword arguments passed to the underlying phylogeny routines.

# Returns

  - `Z::Matrix{Float64}`: Square `assets × assets` feature matrix.

# Related

  - [`PhylogenyFeatures`](@ref)
  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`calc_adjacency`](@ref)
"""
function phylogeny_features end
function phylogeny_features(::BinaryNeighbourhood, pl::AbstractNetworkEstimator, X::MatNum;
                            kwargs...)::Matrix
    return Matrix{eltype(X)}(phylogeny_matrix(pl, X; dims = 1, kwargs...).X) +
           LinearAlgebra.I
end
# `alg` is inert here: a partition is flat, so there is no hop structure to decay. See the
# docstring's caveat -- the resulting distance carries the partition and nothing more.
function phylogeny_features(::AbstractPhylogenyFeatureAlgorithm,
                            pl::AbstractClustersEstimator, X::MatNum; kwargs...)::Matrix
    return Matrix{eltype(X)}(phylogeny_matrix(pl, X; dims = 1, kwargs...).X) +
           LinearAlgebra.I
end
function phylogeny_features(alg::GradedNeighbourhood, pl::AbstractNetworkEstimator,
                            X::MatNum; kwargs...)::Matrix
    A = calc_adjacency(pl, X; dims = 1, kwargs...)
    g = Graphs.SimpleGraph(A)
    n = pl.n
    et = eltype(X)
    dk = alg.decay
    # A hop count only ever takes values in `0:n` here, so probing that range is exhaustive
    # rather than a spot check -- `n + 1` evaluations before an `assets^2` loop.
    assert_separation_decay(dk, 0:n, n)
    Z = zeros(et, Graphs.nv(g), Graphs.nv(g))
    for v in Graphs.vertices(g)
        h = Graphs.gdistances(g, v)
        for u in Graphs.vertices(g)
            # `gdistances` reports `typemax` for an unreachable vertex, so the budget
            # comparison also guards the decay call and must short-circuit -- `ifelse` would
            # evaluate it, and `ReciprocalDecay` overflows `1 + d` at `typemax`.
            @inbounds Z[u, v] = h[u] <= n ? et(separation_decay(dk, h[u], n)) : zero(et)
        end
    end
    return Z
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer reusing a square phylogeny or adjacency matrix as a feature source.

An `assets × assets` adjacency matrix *is* an `assets × features` feature matrix whose feature `k` reads "is related to asset `k`", so `pairwise` over its rows measures **neighbourhood overlap** — a standard notion of topological similarity — and needs almost no new estimation code. It is the only producer whose feature axis *is* the asset axis, which needs no flag on the carrier: it refits, so a subproblem gets its own square matrix over its own universe rather than a slice of a larger one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyFeatures(;
        pl::NwE_ClE = NetworkEstimator(),
        alg::AbstractPhylogenyFeatureAlgorithm = GradedNeighbourhood()
    ) -> PhylogenyFeatures

Keywords correspond to the struct's fields.

# Why the diagonal includes self

The diagonal is not a convention, it selects between two different algorithms. On a three-node path `1 - 2 - 3` under the default [`AngularDist`](@ref):

| pair            | zero diagonal | self included |
|:--------------- | -------------:| -------------:|
| `1`-`3`, 2 hops | **0.0**       | 0.333         |
| `1`-`2`, 1 hop  | 0.5           | 0.196         |

With a zero diagonal the two **non-adjacent** endpoints come out identical and the adjacent pairs maximally far: rows are compared on who their neighbours are, never on whether they are each other's. That is *structural equivalence* — similarity of role — which is a real notion but the opposite of the proximity the name promises.

Including self also keeps subproblems well defined. An asset view of a spanning tree routinely isolates a vertex, and a zero-diagonal row for an isolated asset is a **zero row**: [`AngularDist`](@ref)'s zero-vector convention then puts every isolated asset at distance `0` from every other isolated asset, clustering them together for no reason. With self included they sit at maximal distance from everything, including each other.

# A clustering source is admitted, with a caveat worth reading

`pl` is bound by [`NwE_ClE`](@ref): a graph ([`NetworkEstimator`](@ref)) or a partition ([`ClustersEstimator`](@ref)). Both are *estimators*, so both refit.

A partition carries much less than a graph, and the shortfall is measurable rather than stylistic. Its matrix is `P * transpose(P)` with the diagonal restored, so row `i` is the co-membership indicator of asset `i` and the distance between two rows depends on nothing but cluster membership and cluster size. On a seven-asset universe clustered `[1, 1, 1, 2, 2, 3, 3]` the whole distance matrix takes **three** distinct values. Worse, `phylogeny_matrix`'s `- I` makes each row of a *pair* a lone `1` pointing at the other member, so the two rows are orthogonal and a size-two cluster's **within**-cluster distance equals its **across**-cluster distance — this producer restores the diagonal, which repairs exactly that case, but the coarseness remains.

This is the single-partition case of the round-trip argument that closed the endogenous branch for [`AssetSetsFeatures`](@ref): clustering a re-encoded clustering largely returns the clustering. Prefer a graph source unless the partition is what you actually want to measure.

# Provenance

The source is a [`NetworkEstimator`](@ref) and never a precomputed [`PhylogenyResult`](@ref), because an Estimator does not hold a Result (see `CONTEXT.md` §1). This producer is therefore **endogenous**: the graph is filtered from the returns correlation, so it measures topology the correlation implies rather than structure outside it, and it refits on every fold and every subproblem.

A `FeatureDistance` nested inside the source's own `de` does not recurse: the producer runs inside `prior(pe, X, F; …)`, before `pr.Z` exists, so it fails loudly with an `IsNothingError` naming the missing feature matrix.

# Validation

  - `Z` binds to the asset axis of `X` (see [`check_feature_matrix`](@ref) on [`LowOrderPrior`](@ref)). Squareness is guaranteed by [`phylogeny_features`](@ref) rather than checked on the carrier.

# Examples

```jldoctest
julia> PhylogenyFeatures(; alg = BinaryNeighbourhood())
PhylogenyFeatures
   pl ┼ NetworkEstimator
      │    ce ┼ PortfolioOptimisersCovariance
      │       │   ce ┼ Covariance
      │       │      │    me ┼ SimpleExpectedReturns
      │       │      │       │   w ┴ nothing
      │       │      │    ce ┼ GeneralCovariance
      │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │      │       │    w ┴ nothing
      │       │      │   alg ┴ FullMoment()
      │       │   mp ┼ MatrixProcessing
      │       │      │     pdm ┼ Posdef
      │       │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │       │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │       │      │      dn ┼ nothing
      │       │      │      dt ┼ nothing
      │       │      │     alg ┼ nothing
      │       │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │    de ┼ Distance
      │       │   power ┼ nothing
      │       │     alg ┴ CanonicalDistance()
      │   alg ┼ KruskalTree
      │       │     args ┼ Tuple{}: ()
      │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │     n ┴ Int64: 1
  alg ┴ BinaryNeighbourhood()
```

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`phylogeny_features`](@ref)
  - [`feature_matrix`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
  - [`NwE_ClE`](@ref)
"""
@propagatable @concrete struct PhylogenyFeatures <: AbstractFeatureMatrixEstimator
    """
    $(field_dict[:plfe])
    """
    @fprop @vprop pl
    """
    $(field_dict[:plfalg])
    """
    alg
    function PhylogenyFeatures(pl::NwE_ClE, alg::AbstractPhylogenyFeatureAlgorithm)
        return new{typeof(pl), typeof(alg)}(pl, alg)
    end
end
function PhylogenyFeatures(; pl::NwE_ClE = NetworkEstimator(),
                           alg::AbstractPhylogenyFeatureAlgorithm = GradedNeighbourhood())::PhylogenyFeatures
    return PhylogenyFeatures(pl, alg)
end
function feature_matrix(ze::PhylogenyFeatures, ::AbstractPriorResult, X::MatNum, args...;
                        kwargs...)
    # `prior(pe::FeaturePrior, …)` has already transposed `X` to observations x assets and
    # consumed `dims` as a named keyword, so it is not in `kwargs` and the hop routines are
    # called at `dims = 1`, matching the canonically assets-major carried layout.
    return phylogeny_features(ze.alg, ze.pl, X; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer reading exogenous taxonomy memberships off a [`UniverseSets`](@ref).

`AssetSetsFeatures` is the thin producer wrapping [`asset_sets_features`](@ref), so a sector, industry or country classification reaches [`FeatureDistance`](@ref) through the *derived* carrier as well as through the user's own [`ReturnsResult`](@ref). It reads its taxonomy from [`FeaturePrior`](@ref)'s `sets` field.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetSetsFeatures(;
        vals::Union{<:AbstractVector{<:AbstractString}, <:AbstractVector{<:Pair}},
        strict::Bool = false
    ) -> AssetSetsFeatures

Keywords correspond to the struct's fields.

# One type, two contracts

`vals` is dispatched on element type, and the two readings are genuinely different animals:

  - `AbstractVector{<:AbstractString}` — **group name keys**. Every key is a partition, so every row has exactly `length(vals)` ones and the equal-row-norm identity `cos = shared / L` holds exactly. The feature axis is *derived* by stacking, names are qualified `\"<key>=<group>\"`, and a view rebuilds it.
  - `AbstractVector{<:Pair}` — a **graded program**. An ordered, last-wins list of authored edges over the axis *declared* at `sets.dict[sets.zkey]`. Names are bare, the axis is fold-invariant, and the equal-norm identity does not hold — no normalisation could restore it, since it needs `0`/`1` entries and not merely equal norms.

They share a type rather than splitting into two producers because the grammar strictly **subsumes** the key list: `\"nx_sector\" => 2.0` emits the same columns as `\"nx_sector\"`, only scaled. The cost is accepted and stated: one type now carries two contracts, and only one of them has the identity.

# The only exogenous rectangular source

Every other producer in this file derives `Z` from the returns: [`RegressionFeatures`](@ref) from a factor fit, [`PhylogenyFeatures`](@ref) from a graph filtered out of the correlation. A taxonomy comes from outside the price history entirely, which is the structure a feature distance exists to bring in — see [`asset_sets_features`](@ref) for the nested-versus-crossed reading and the exact `cos = shared / L` identity.

# Two entry points, because a producer alone would not reach the default

Producers run inside `prior(pe::FeaturePrior, …)`, so they can only ever populate the **derived** carrier — and `z_src` defaults to `:data`. [`asset_sets_features`](@ref) is therefore public in its own right, for building an `assets × features` matrix to hand to [`ReturnsResult`](@ref) directly. The two routes give the identical matrix; they differ only in which carrier holds it, and hence in what a fold does to it.

# The feature axis is groups, never assets — on the key path

Even when the group values happen to number as many as the assets. There the feature axis indexes *groups*, so a coincidence of counts is not a claim about what the columns mean, and an asset view rebuilds the axis from the viewed taxonomy rather than slicing it, so a group with no members left in the view simply disappears.

A graded program is the other case: its axis is **declared**, part asset node and part taxonomy node, and a view passes it through untouched. `size(Z, 2)` is therefore fold-invariant, and an asset node whose asset the view dropped survives as an all-zero column.

# `strict` is a field, not a keyword

[`feature_matrix`](@ref)'s producer interface is `feature_matrix(ze, pr, X, F, sets)`, so there is nowhere to pass a keyword through. It rides on the estimator instead, and appears as a keyword on the public [`asset_sets_features`](@ref) — which matters because the public function is the only route to the default `z_src = :data` carrier, so a grammar landing on the producer alone would be reachable from one carrier and not the other.

# Validation

  - On the **key** path: `length(vals) >= 2` and `allunique(vals)` (see [`assert_feature_keys`](@ref)), at construction. Neither carries to the graded path — a one-column matrix is legal there, and repeating a key is the whole point of last-wins.
  - On the **graded** path: `!isempty(vals)`, at construction.
  - `FeaturePrior.sets` is not `nothing`, and resolves against it, when the producer runs.

# Examples

```jldoctest
julia> AssetSetsFeatures(; vals = [\"nx_sector\", \"nx_country\"])
AssetSetsFeatures
    vals ┼ Vector{String}: ["nx_sector", "nx_country"]
  strict ┴ Bool: false
```

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`asset_sets_features`](@ref)
  - [`assert_feature_keys`](@ref)
  - [`Scale`](@ref)
  - [`UniverseSets`](@ref)
  - [`feature_matrix`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
"""
@concrete struct AssetSetsFeatures <: AbstractFeatureMatrixEstimator
    """
    $(field_dict[:asets_vals])
    """
    vals
    """
    $(field_dict[:asets_strict])
    """
    strict
    function AssetSetsFeatures(vals::AbstractVector{<:AbstractString},
                               strict::Bool)::AssetSetsFeatures
        assert_feature_keys(vals)
        return new{typeof(vals), typeof(strict)}(vals, strict)
    end
    function AssetSetsFeatures(vals::AbstractVector{<:Pair},
                               strict::Bool)::AssetSetsFeatures
        @argcheck(!isempty(vals),
                  IsEmptyError("`vals` cannot be empty: a program with no entries can only produce the all-zero matrix, which declares every asset identical"))
        return new{typeof(vals), typeof(strict)}(vals, strict)
    end
end
function AssetSetsFeatures(;
                           vals::Union{<:AbstractVector{<:AbstractString},
                                       <:AbstractVector{<:Pair}},
                           strict::Bool = false)::AssetSetsFeatures
    return AssetSetsFeatures(vals, strict)
end
# The taxonomy is the whole input, so an absent `sets` is a missing argument rather than a
# defaulted one: there is nothing to fall back to, and a returns-derived substitute would be
# the exact endogeneity this producer exists to escape.
function feature_matrix(ze::AssetSetsFeatures, ::AbstractPriorResult, ::Any, ::Any,
                        sets::Option{<:UniverseSets}; kwargs...)
    @argcheck(!isnothing(sets),
              IsNothingError("AssetSetsFeatures reads its taxonomy from `FeaturePrior.sets`, which is nothing. Pass `FeaturePrior(; ze = AssetSetsFeatures(; vals = $(ze.vals)), sets = UniverseSets(…))`."))
    return asset_sets_features(ze.vals, sets; strict = ze.strict)
end
"""
    feature_estimator_view(ze::AbstractFeatureMatrixEstimator, i, args...)
    feature_estimator_view(ze::MatNum_Arr3Num, i, args...)

Subselect a [`FeaturePrior`](@ref)'s `ze` slot by assets `i`.

A producer is usually *configuration*: it recomputes from the viewed prior on the next call, so it passes through unchanged — which the universal [`port_opt_view`](@ref) fallback already does for an estimator with nothing to slice. A producer that *embeds data* is the exception, and delegating to [`port_opt_view`](@ref) rather than returning `ze` is what lets the exception exist without every other producer opting in.

A literal feature matrix is *data* and must be sliced on its asset axis, exactly as the carried matrix is — otherwise its columns would keep pointing at the full universe while the rows point at a cluster.

The literal path slices with `sq = false`, matching the prior carrier's own view: a derived `Z`'s feature axis is never sliced, because a producer refits rather than being cut down.

# Related

  - [`FeaturePrior`](@ref)
  - [`PhylogenyFeatures`](@ref)
  - [`feature_matrix_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function feature_estimator_view(ze::AbstractFeatureMatrixEstimator, i, args...)
    return port_opt_view(ze, i, args...)
end
function feature_estimator_view(ze::MatNum_Arr3Num, i, args...)
    return feature_matrix_view(ze, false, :, i)
end
"""
$(DocStringExtensions.TYPEDEF)

Prior estimator that attaches a feature matrix to the prior it wraps.

`FeaturePrior` delegates every moment to the wrapped estimator `pe` and adds nothing but `Z`, so it is a **provably pure addition**: the moments it returns are the wrapped estimator's, unchanged. That is what makes every existing prior feature-capable without any of them knowing about features.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeaturePrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
        sets::Option{<:UniverseSets} = nothing
    ) -> FeaturePrior

Keywords correspond to the struct's fields.

# Details

  - **Moments first, features second.** The wrapped prior is computed before `ze` runs, because a producer may need the result — [`RegressionFeatures`](@ref) reads `pr.rr`.
  - **Nesting order does not matter.** Every wrapping prior estimator forwards `Z`, so `BlackLittermanPrior(; pe = FeaturePrior(…))` and `FeaturePrior(; pe = BlackLittermanPrior(…))` both arrive at [`distance`](@ref) with the same feature matrix. The exception is the estimators whose wrapped prior is fit on *factors* — [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) — which drop it deliberately; wrap those from the outside.
  - **The outermost declaration wins.** Nesting one `FeaturePrior` inside another overwrites the inner feature matrix rather than merging.
  - **A literal `ze` must be static to survive a fold.** An `assets × features` matrix is observation-independent and is correct under any cross-validation fold. A time-varying literal is not: folds slice observations *before* the prior is fit and never touch the estimator, so its observation axis would no longer match — which [`LowOrderPrior`](@ref) rejects at construction. Use a producer to derive a time-varying `Z` per fold.

# Validation

  - If `ze` is a literal matrix, it is non-empty.
  - If `sets` is not `nothing`, `length(sets.dict[sets.xkey]) == size(X, 2)`.

# Examples

```jldoctest
julia> pe = FeaturePrior(; pe = EmpiricalPrior(), ze = RegressionFeatures());

julia> pe.ze
RegressionFeatures()

julia> isnothing(pe.sets)
true

julia> pe.pe
EmpiricalPrior
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
       me ┼ SimpleExpectedReturns
          │   w ┴ nothing
  horizon ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_AF`](@ref)
  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`feature_matrix`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`FeatureDistance`](@ref)
  - [`prior`](@ref)
"""
@propagatable @concrete struct FeaturePrior <: AbstractLowOrderPriorEstimator_AF
    """
    $(field_dict[:pe])
    """
    @fprop pe
    """
    $(field_dict[:ze])
    """
    @fprop ze
    """
    $(field_dict[:sets])
    """
    sets
    function FeaturePrior(pe::AbstractLowOrderPriorEstimator_A_F_AF,
                          ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
                          sets::Option{<:UniverseSets})
        if isa(ze, MatNum_Arr3Num)
            assert_nonempty(ze, :ze)
        end
        return new{typeof(pe), typeof(ze), typeof(sets)}(pe, ze, sets)
    end
end
function FeaturePrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                      ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
                      sets::Option{<:UniverseSets} = nothing)::FeaturePrior
    return FeaturePrior(pe, ze, sets)
end
# Expose `:me` and `:ce` from the embedded prior estimator `pe` for transparent access
# (see [`@forward_properties`](@ref)).
@forward_properties FeaturePrior begin
    forward(pe, me, ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`FeaturePrior`](@ref) restricted to assets at index `i`.

Hand-written rather than generated by [`@vprop`](@ref): the `ze` slot holds either a producer or a literal feature matrix, and the two need different treatment — see [`feature_estimator_view`](@ref).

# Related

  - [`FeaturePrior`](@ref)
  - [`feature_estimator_view`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(pe::FeaturePrior, i, args...)::FeaturePrior
    return FeaturePrior(; pe = port_opt_view(pe.pe, i, args...),
                        ze = feature_estimator_view(pe.ze, i, args...),
                        sets = port_opt_view(pe.sets, i, args...))
end
"""
    prior(pe::FeaturePrior, X::MatNum, F::Option{<:MatNum} = nothing;
          dims::Int = 1, kwargs...)

Compute the wrapped prior's moments and attach a feature matrix to them.

Every moment is the wrapped estimator's, untouched. The only addition is `Z`, produced by `pe.ze` from the *already-computed* prior result — the ordering [`RegressionFeatures`](@ref) needs, since it reads `pr.rr`.

# Arguments

  - `pe`: Feature prior estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Optional factor matrix (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the wrapped estimator and the producer.

# Returns

  - `pr::LowOrderPrior`: The wrapped result, with `Z` set.

# Validation

  - `dims in (1, 2)`.
  - If `pe.sets` is not `nothing`, `length(pe.sets.dict[pe.sets.xkey]) == size(X, 2)`.
  - `Z` is validated against `X` by [`LowOrderPrior`](@ref).

# Related

  - [`FeaturePrior`](@ref)
  - [`feature_matrix`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::FeaturePrior, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1,
               kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
        if !isnothing(F)
            F = transpose(F)
        end
    end
    if !isnothing(pe.sets)
        @argcheck(length(pe.sets.dict[pe.sets.xkey]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.xkey]) ($(length(pe.sets.dict[pe.sets.xkey]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    pr = prior(pe.pe, X, F; kwargs...)
    Z = feature_matrix(pe.ze, pr, X, F, pe.sets; kwargs...)
    # This estimator only attaches a feature matrix, so `Z` is the single deviation from the
    # wrapped result and everything else forwards untouched (see [`forward_prior`](@ref)).
    return forward_prior(pr; Z = Z)
end

export AbstractFeatureMatrixEstimator, RegressionFeatures, FeaturePrior, feature_matrix,
       AbstractPhylogenyFeatureAlgorithm, BinaryNeighbourhood, GradedNeighbourhood,
       PhylogenyFeatures, phylogeny_features, AssetSetsFeatures
