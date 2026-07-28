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

Compute the derived feature matrix and its squareness flag.

Returns the pair `(Z, z_sq)` that [`FeaturePrior`](@ref) writes onto the prior result. `Z` is **canonically assets-major** — `assets × features` when static, `observations × assets × features` when time-varying — and is validated as such by [`LowOrderPrior`](@ref).

`z_sq` is **stated here, by the producer**, because a producer runs inside `prior(pe, X, F; …)` with raw matrices and no feature names: a prior result carries none, so [`features_are_assets`](@ref) is structurally unavailable and squareness cannot be derived. A producer that has both name vectors — one whose features *are* the assets, given `sets` — must still run the `nx == nz` comparison rather than assume, because a `z_sq` that lies is silent, not loud: [`feature_matrix_view`](@ref) slices the feature axis that is not there.

`pr` is the **already-computed** wrapped prior result, not an estimator. That ordering is required: [`RegressionFeatures`](@ref) reads `pr.rr`, which does not exist until the wrapped prior has run.

# Arguments

  - `ze`: Feature matrix producer, or a literal feature matrix.
  - `pr`: Prior result returned by the wrapped estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Factor returns matrix, or `nothing`.
  - `sets`: Asset sets supplying asset names, or `nothing`.
  - `kwargs...`: Additional keyword arguments passed to underlying estimators.

# Returns

  - `(Z, z_sq)::Tuple{<:MatNum_Arr3Num, Bool}`.

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`FeaturePrior`](@ref)
  - [`RegressionFeatures`](@ref)
  - [`features_are_assets`](@ref)
"""
function feature_matrix end
# A literal matrix declares nothing about its own axes, so it is never treated as square. To
# reuse an `assets × assets` adjacency matrix as features — where an asset view must slice
# the feature axis too — use a producer that states `z_sq` rather than passing the matrix in
# bare.
function feature_matrix(ze::MatNum_Arr3Num, ::AbstractPriorResult, args...; kwargs...)
    return ze, false
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer that reads the regression loadings off the wrapped prior result.

`RegressionFeatures` treats `pr.rr.L` — the coordinate system a factor model places each asset in — as an `assets × features` feature matrix. It is the cheapest real feature source in the library: a factor prior already computes it, and already refits it per fold, so the features track the fold with no extra plumbing.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, set by [`DimensionReductionRegression`](@ref): the low-dimensional coordinate system the asset actually lives in. `M` is the *reconstructed* full-factor loadings. `pr.rr.L` always resolves — [`Regression`](@ref) swaps in `M` when `L` is unset — so this producer needs no branch and works behind every regression estimator.

Both are assets-major and both are row-sliced by `port_opt_view(re, i)`, so the carried layout holds with no transpose. Sets `z_sq = false`: reduced dimensions are not assets.

# Validation

  - The wrapped prior must carry a regression (see [`assert_prior_regression`](@ref)). `BlackLittermanPrior` forwards neither `rr` nor `f_mu`/`f_sigma`, so `FeaturePrior(; pe = BlackLittermanPrior(…), ze = RegressionFeatures())` throws; nest the other way round instead.

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
    return pr.rr.L, false
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny feature algorithms.

A phylogeny feature algorithm is the rule turning the graph a [`PhylogenyFeatures`](@ref) source describes into an `assets × assets` feature matrix, `Z[i, k] = f(hops(i, k))`. The family is open: a user needing a different decay defines a member and a [`phylogeny_features`](@ref) method for it.

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

`Z[i, k] = max(0, n + 1 - hops(i, k))`, so a direct neighbour scores `n`, a two-hop neighbour `n - 1`, anything beyond `n` hops scores `0`, and the asset itself scores `n + 1`. It is strictly richer than [`BinaryNeighbourhood`](@ref), which is its `n = 1` special case up to scale: `phylogeny_matrix` accumulates `sum(A^i for i in 0:n)` and then `clamp!(P, 0, 1)` **destroys the step count**, which is the information this algorithm keeps.

The decay is linear in hops rather than the un-clamped walk count `sum(A^i)`. A walk count is degree-biased — a hub accumulates walks combinatorially — so two assets' scores would encode how busy their neighbourhoods are as much as how close they are.

Unreachable pairs score `0`, which needs no special case: `hops` is `typemax` there and the comparison against `n` runs before the subtraction, so nothing overflows.

# Related

  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`BinaryNeighbourhood`](@ref)
  - [`PhylogenyFeatures`](@ref)
  - [`calc_adjacency`](@ref)
"""
struct GradedNeighbourhood <: AbstractPhylogenyFeatureAlgorithm end
"""
    phylogeny_features(alg::AbstractPhylogenyFeatureAlgorithm,
                       pl::PhylogenyResult{<:MatNum}, args...; kwargs...)
    phylogeny_features(alg::BinaryNeighbourhood, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)
    phylogeny_features(alg::GradedNeighbourhood, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)

Turn a graph source into a square `assets × assets` feature matrix.

The kernel behind [`PhylogenyFeatures`](@ref). Every method returns a `Float64` matrix — not the `Int` or `BitMatrix` the phylogeny routines produce — so that [`AngularDist`](@ref) keeps its BLAS `gemm` path.

# The two source kinds are treated differently, on purpose

  - **A precomputed [`PhylogenyResult`](@ref) is used as given.** Its matrix already *is* the relatedness matrix; `alg` is inert, no hop transform runs, and a weighted graph keeps its weights rather than being binarised. This is the only exogenous path in the whole producer family for square structure — a supply chain, a shared-ownership graph, a country adjacency — and a literal `ze` cannot serve it, because a bare matrix is `z_sq = false` by dispatch (see [`feature_matrix`](@ref)).
  - **A [`NetworkEstimator`](@ref) is refit from `X`,** so the graph tracks a cross-validation fold, and `alg` selects the decay. Its adjacency is already `0`/`1` by construction — `Graphs.adjacency_matrix` of a minimum spanning tree or a PMFG — so there are no edge weights left to preserve or discard on this path.

# The diagonal

`Z[i, i]` is the top of the scale, never zero: `1` for [`BinaryNeighbourhood`](@ref), `n + 1` for [`GradedNeighbourhood`](@ref), and `maximum(pl.X)` for a precomputed matrix, which has no decay function to continue and whose diagonal [`PhylogenyResult`](@ref) forces to zero. An all-zero precomputed matrix falls back to `1`, leaving the identity: with no edges at all, no two assets share anything, which is the honest answer rather than a degenerate one.

# Arguments

  - `alg`: Phylogeny feature algorithm.
  - `pl`: Graph source — a network estimator, or a precomputed square phylogeny result.
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
# `alg` is inert here, the same way `FeatureDistance`'s collapse `alg` is inert on a static
# feature matrix: there is no hop structure to decay, only a relatedness matrix to carry
# through with its diagonal restored.
function phylogeny_features(::AbstractPhylogenyFeatureAlgorithm,
                            pl::PhylogenyResult{<:MatNum}, args...;
                            kwargs...)::Matrix{Float64}
    Z = Matrix{Float64}(pl.X)
    d = maximum(Z)
    return Z + ifelse(iszero(d), one(d), d) * LinearAlgebra.I
end
function phylogeny_features(::BinaryNeighbourhood, pl::AbstractNetworkEstimator, X::MatNum;
                            kwargs...)::Matrix{Float64}
    return Matrix{Float64}(phylogeny_matrix(pl, X; dims = 1, kwargs...).X) + LinearAlgebra.I
end
function phylogeny_features(::GradedNeighbourhood, pl::AbstractNetworkEstimator, X::MatNum;
                            kwargs...)::Matrix{Float64}
    A = calc_adjacency(pl, X; dims = 1, kwargs...)
    g = Graphs.SimpleGraph(A)
    n = pl.n
    Z = zeros(Float64, Graphs.nv(g), Graphs.nv(g))
    for v in Graphs.vertices(g)
        h = Graphs.gdistances(g, v)
        for u in Graphs.vertices(g)
            # `gdistances` reports `typemax` for an unreachable vertex, so the comparison
            # must run before the subtraction or `n + 1 - h[u]` would overflow.
            @inbounds Z[u, v] = ifelse(h[u] <= n, Float64(n + 1 - h[u]), 0.0)
        end
    end
    return Z
end
"""
$(DocStringExtensions.TYPEDEF)

Feature matrix producer reusing a square phylogeny or adjacency matrix as a feature source.

An `assets × assets` adjacency matrix *is* an `assets × features` feature matrix whose feature `k` reads "is related to asset `k`", so `pairwise` over its rows measures **neighbourhood overlap** — a standard notion of topological similarity — and needs almost no new estimation code. It is the only producer that sets `z_sq = true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyFeatures(;
        pl::NwE_PlM = NetworkEstimator(),
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

# Why a clustering source is rejected

`pl` is bound by [`NwE_PlM`](@ref), which admits a graph and excludes a partition — a [`ClustersEstimator`](@ref) or [`Clusters`](@ref) is a `MethodError` here, not a silent degeneracy. Its square matrix is `P * transpose(P) - I`, whose row `i` is the co-membership indicator of asset `i`, so the distance depends on nothing but cluster size. On a seven-asset universe clustered `[1, 1, 1, 2, 2, 3, 3]` it takes three values in total, and the size-two cluster's **within**-cluster distance is `0.5` — exactly its across-cluster distance, because `- I` leaves each row of a pair as a lone `1` pointing at the other member, making the two rows orthogonal. The partition is recoded and then lost. This is the single-partition case of the round-trip argument that closed the endogenous branch for [`AssetSetsFeatures`](@ref).

# Provenance

The [`NetworkEstimator`](@ref) source is **endogenous** — the graph is filtered from the returns correlation — so it measures topology the correlation implies rather than structure outside it, and it refits per fold. The [`PhylogenyResult`](@ref) source is **exogenous** and is the only route by which user-supplied square structure reaches a distance at all.

A `FeatureDistance` nested inside the source's own `de` does not recurse: the producer runs inside `prior(pe, X, F; …)`, before `pr.Z` exists, so it fails loudly with an `IsNothingError` naming the missing feature matrix.

# Validation

  - `Z` is square and binds to the asset axis of `X` (see [`check_feature_matrix`](@ref) and [`assert_square_feature_axis`](@ref) on [`LowOrderPrior`](@ref)).

# Examples

```jldoctest
julia> PhylogenyFeatures(; pl = PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0]))
PhylogenyFeatures
   pl ┼ PhylogenyResult
      │   X ┴ 3×3 Matrix{Int64}
  alg ┴ GradedNeighbourhood()
```

# Related

  - [`AbstractFeatureMatrixEstimator`](@ref)
  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`phylogeny_features`](@ref)
  - [`feature_matrix`](@ref)
  - [`FeaturePrior`](@ref)
  - [`FeatureDistance`](@ref)
  - [`NwE_PlM`](@ref)
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
    function PhylogenyFeatures(pl::NwE_PlM, alg::AbstractPhylogenyFeatureAlgorithm)
        if isa(pl, PhylogenyResult)
            assert_matrix_issquare(pl.X, :pl)
        end
        return new{typeof(pl), typeof(alg)}(pl, alg)
    end
end
function PhylogenyFeatures(; pl::NwE_PlM = NetworkEstimator(),
                           alg::AbstractPhylogenyFeatureAlgorithm = GradedNeighbourhood())::PhylogenyFeatures
    return PhylogenyFeatures(pl, alg)
end
function feature_matrix(ze::PhylogenyFeatures, ::AbstractPriorResult, X::MatNum, args...;
                        kwargs...)
    # `prior(pe::FeaturePrior, …)` has already transposed `X` to observations x assets and
    # consumed `dims` as a named keyword, so it is not in `kwargs` and the hop routines are
    # called at `dims = 1`, matching the canonically assets-major carried layout.
    Z = phylogeny_features(ze.alg, ze.pl, X; kwargs...)
    @argcheck(size(Z, 1) == size(X, 2),
              DimensionMismatch("a square feature matrix (Z) is assets x assets, so it must bind to the asset axis of X, got size(Z) = $(size(Z)) and size(X, 2) = $(size(X, 2)). A precomputed PhylogenyResult is indexed by the full asset universe: it must cover exactly the assets the prior is fit on."))
    return Z, true
end
"""
    feature_estimator_view(ze::AbstractFeatureMatrixEstimator, i, args...)
    feature_estimator_view(ze::MatNum_Arr3Num, i, args...)

Subselect a [`FeaturePrior`](@ref)'s `ze` slot by assets `i`.

A producer is usually *configuration*: it recomputes from the viewed prior on the next call, so it passes through unchanged — which the universal [`port_opt_view`](@ref) fallback already does for an estimator with nothing to slice. A producer that *embeds data* is the exception, and must slice it: [`PhylogenyFeatures`](@ref) over a precomputed [`PhylogenyResult`](@ref) holds a matrix indexed by the full asset universe. Delegating to [`port_opt_view`](@ref) rather than returning `ze` is what lets the exception exist without every other producer opting in.

A literal feature matrix is *data* and must be sliced on its asset axis, exactly as the carried matrix is — otherwise its columns would keep pointing at the full universe while the rows point at a cluster.

The literal path slices with `sq = false`, matching the `z_sq = false` that [`feature_matrix`](@ref) reports for a bare matrix.

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

`FeaturePrior` delegates every moment to the wrapped estimator `pe` and adds nothing but `Z` and `z_sq`, so it is a **provably pure addition**: the moments it returns are the wrapped estimator's, unchanged. That is what makes every existing prior feature-capable without any of them knowing about features.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeaturePrior(;
        pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
        ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
        sets::Option{<:AssetSets} = nothing
    ) -> FeaturePrior

Keywords correspond to the struct's fields.

# Details

  - **Moments first, features second.** The wrapped prior is computed before `ze` runs, because a producer may need the result — [`RegressionFeatures`](@ref) reads `pr.rr`.
  - **Nesting order does not matter.** Every wrapping prior estimator forwards `Z` and `z_sq`, so `BlackLittermanPrior(; pe = FeaturePrior(…))` and `FeaturePrior(; pe = BlackLittermanPrior(…))` both arrive at [`distance`](@ref) with the same feature matrix. The exception is the estimators whose wrapped prior is fit on *factors* — [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) — which drop it deliberately; wrap those from the outside.
  - **The outermost declaration wins.** Nesting one `FeaturePrior` inside another overwrites the inner feature matrix rather than merging.
  - **A literal `ze` must be static to survive a fold.** An `assets × features` matrix is observation-independent and is correct under any cross-validation fold. A time-varying literal is not: folds slice observations *before* the prior is fit and never touch the estimator, so its observation axis would no longer match — which [`LowOrderPrior`](@ref) rejects at construction. Use a producer to derive a time-varying `Z` per fold.

# Validation

  - If `ze` is a literal matrix, it is non-empty.
  - If `sets` is not `nothing`, `length(sets.dict[sets.key]) == size(X, 2)`.

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
                          sets::Option{<:AssetSets})
        if isa(ze, MatNum_Arr3Num)
            assert_nonempty(ze, :ze)
        end
        return new{typeof(pe), typeof(ze), typeof(sets)}(pe, ze, sets)
    end
end
function FeaturePrior(; pe::AbstractLowOrderPriorEstimator_A_F_AF = EmpiricalPrior(),
                      ze::Union{<:MatNum_Arr3Num, <:AbstractFeatureMatrixEstimator},
                      sets::Option{<:AssetSets} = nothing)::FeaturePrior
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

Every moment is the wrapped estimator's, untouched. The only additions are `Z` and `z_sq`, produced by `pe.ze` from the *already-computed* prior result — the ordering [`RegressionFeatures`](@ref) needs, since it reads `pr.rr`.

# Arguments

  - `pe`: Feature prior estimator.
  - `X`: Asset returns matrix `observations × assets`.
  - `F`: Optional factor matrix (default: `nothing`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the wrapped estimator and the producer.

# Returns

  - `pr::LowOrderPrior`: The wrapped result, with `Z` and `z_sq` set.

# Validation

  - `dims in (1, 2)`.
  - If `pe.sets` is not `nothing`, `length(pe.sets.dict[pe.sets.key]) == size(X, 2)`.
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
        @argcheck(length(pe.sets.dict[pe.sets.key]) == size(X, 2),
                  DimensionMismatch("length(pe.sets.dict[pe.sets.key]) ($(length(pe.sets.dict[pe.sets.key]))) must match size(X, 2) ($(size(X, 2)))"))
    end
    pr = prior(pe.pe, X, F; kwargs...)
    Z, z_sq = feature_matrix(pe.ze, pr, X, F, pe.sets; kwargs...)
    return LowOrderPrior(; X = pr.X, mu = pr.mu, sigma = pr.sigma, chol = pr.chol, w = pr.w,
                         ens = pr.ens, kld = pr.kld, ow = pr.ow, rr = pr.rr, f_mu = pr.f_mu,
                         f_sigma = pr.f_sigma, f_w = pr.f_w, Z = Z, z_sq = z_sq)
end

export AbstractFeatureMatrixEstimator, RegressionFeatures, FeaturePrior, feature_matrix,
       AbstractPhylogenyFeatureAlgorithm, BinaryNeighbourhood, GradedNeighbourhood,
       PhylogenyFeatures, phylogeny_features
