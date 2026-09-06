"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny feature algorithms.

A phylogeny feature algorithm is the rule turning the structure a [`PhylogenyPanel`](@ref) source describes into an `assets × assets` proximity matrix, `Z[i, k] = f(separation(i, k))`. The family is open: a user needing a different rule defines a member and a [`phylogeny_features`](@ref) method for it.

Two neighbouring choices need neither. A different *fall-off* is an [`AbstractSeparationDecayAlgorithm`](@ref), which [`Proximity`](@ref) carries as a field; a different *notion of far* is an [`AbstractSeparationAlgorithm`](@ref), which the source [`NetworkEstimator`](@ref) carries as `sep`. Between them those two knobs span every neighbourhood rule the family has needed so far, which is why exactly one member ships.

**One member is an extension point, not a taxonomy.** The type exists so that a rule which is *not* a decayed separation — a role-similarity matrix, say, or a rule reading structure the separation kernels do not expose — has a place to dispatch from. It is not a partition of anything, and nothing infers a second member's existence from the first.

Every member includes **self**, so `f(0)` is the top of its scale — see [`PhylogenyPanel`](@ref) for why the diagonal is load-bearing rather than cosmetic.

# Related

  - [`Proximity`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`phylogeny_features`](@ref)
  - [`AbstractSeparationAlgorithm`](@ref)
  - [`AbstractSeparationDecayAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyFeatureAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Phylogeny feature algorithm scoring each pair by how far apart it sits.

`Z[i, k] = decay(separation(i, k))` inside the budget and `0` beyond, so the score falls off with distance instead of flattening to an indicator. The separation and its budget come from the source [`NetworkEstimator`](@ref)'s `sep` — hops under [`HopCount`](@ref), a summed shortest path under [`PathLength`](@ref) — and the fall-off from `decay`.

The score is a function of the separation rather than of the un-clamped walk count `sum(A^i)`. A walk count is degree-biased — a hub accumulates walks combinatorially — so two assets' scores would encode how busy their neighbourhoods are as much as how close they are. It is for the same reason strictly richer than [`phylogeny_matrix`](@ref)'s output, which accumulates that walk count and then `clamp!(P, 0, 1)` **destroys the step count**: this is the information this algorithm keeps.

# The two knobs

`decay` shapes the fall-off; `sep` on the source [`NetworkEstimator`](@ref) truncates it. They are deliberately separate — an exponential never reaches zero, so a budget cannot be expressed as a fall-off — and the budget is the only place truncation happens. Under the default [`LinearDecay`](@ref) and [`HopCount`](@ref) the two coincide in appearance: a direct neighbour scores `n`, a two-hop neighbour `n - 1`, the asset itself `n + 1`, and the score would hit `0` exactly one hop past the budget that already cut it. Under [`ExponentialDecay`](@ref) or [`ReciprocalDecay`](@ref) the diagonal is `1` and the fall-off is set by the member's own parameter, independently of how far the budget looks.

[`NoDecay`](@ref) is the flat end of that dial and is worth stating on its own, because it is what a binary neighbourhood indicator now *is*: the budget still cuts, so `Proximity(; decay = NoDecay())` gives `1` inside it and `0` outside — an indicator, not a matrix of ones.

A zero entry means **functionally unreachable**: either the pair is disconnected or outside the budget, or the decay has fallen to nothing — the same claim about the pair, since [`AbstractSeparationDecayAlgorithm`](@ref) forbids anything below zero inside the budget. No shipped decay other than the flat one emits zero there, so for what ships a zero is unreachable-or-out-of-budget and nothing else.

# Two separations, and what may be compared across them

A `Z` graded over hops and a `Z` graded over path lengths are **interchangeable as inputs** — both satisfy the same contract, so every consumer takes either — and are **not comparable as values**. The budgets are in different units, the supports differ, and under [`LinearDecay`](@ref) so do the scales.

On any real universe they will nevertheless *look* interchangeable: measured over twenty assets, `rho = 0.99` on a minimum spanning tree and `0.95` to `0.98` on a PMFG. That is empirical rather than guaranteed — both structures are selected by distance, so their two readings of it rarely disagree — and it is not a licence to compare one run's numbers against another's.

Under [`PathLength`](@ref)'s default `dmax = nothing` the budget is the **observed** diameter, so `f(0)` is data-dependent for [`LinearDecay`](@ref) and a diameter that moves between cross-validation folds *shifts* every entry of `Z` rather than rescaling it. A fixed `dmax` buys back the fold-stability, and the decays that pin `f(0) = 1` never had the exposure.

# Unreachable pairs

An unreachable pair carries [`separation_matrix`](@ref)'s sentinel — `typemax`, which is `typemax(Int)` for a hop count and `Inf` for a path length over `Float64` weights — so the budget comparison both selects the `0` and **guards the decay call**: `separation_decay` is never evaluated at the sentinel. The guard is load-bearing rather than tidy — `ReciprocalDecay` overflows `1 + d` there, and for a fractional `power` that is a `DomainError` rather than a discarded number.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Proximity(;
        decay::AbstractSeparationDecayAlgorithm = LinearDecay()
    ) -> Proximity

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> Proximity(; decay = ExponentialDecay(; rate = 0.5))
Proximity
  decay ┼ ExponentialDecay
        │   rate ┴ Float64: 0.5
```

# Related

  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`AbstractSeparationAlgorithm`](@ref)
  - [`HopCount`](@ref)
  - [`PathLength`](@ref)
  - [`AbstractSeparationDecayAlgorithm`](@ref)
  - [`LinearDecay`](@ref)
  - [`NoDecay`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`separation_matrix`](@ref)
"""
@concrete struct Proximity <: AbstractPhylogenyFeatureAlgorithm
    """
    $(field_dict[:sdecay])
    """
    decay
    function Proximity(decay::AbstractSeparationDecayAlgorithm)
        return new{typeof(decay)}(decay)
    end
end
function Proximity(; decay::AbstractSeparationDecayAlgorithm = LinearDecay())::Proximity
    return Proximity(decay)
end
"""
    phylogeny_features(alg::Proximity, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)
    phylogeny_features(alg::AbstractPhylogenyFeatureAlgorithm,
                       pl::AbstractClustersEstimator, X::MatNum; kwargs...)

Turn a graph source into a square `assets × assets` feature matrix.

The kernel behind [`PhylogenyPanel`](@ref). Every method returns a `Float64` matrix — not the `Int` or `BitMatrix` the phylogeny routines produce — so that [`AngularDist`](@ref) keeps its BLAS `gemm` path.

# The source is always refit

`pl` is an estimator — a [`NetworkEstimator`](@ref) or a [`ClustersEstimator`](@ref) — never a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref), because an Estimator does not hold a Result (see `CONTEXT.md` §1). The structure is therefore rebuilt from `X` on every call, so it tracks a cross-validation fold or a meta-optimiser's subproblem instead of describing a universe it no longer sees.

# `alg` applies to a graph, and is inert for a partition

A **graph** source has separation structure, so `alg` decays it — over the separations its `sep` measures.

A **partition** has none: two assets are in the same cluster or they are not, and there is nothing between them to decay. Every algorithm therefore gives the same co-membership matrix, and `alg` is inert rather than an error — the same treatment `FeatureDistance`'s collapse `alg` gets on a static feature matrix.

The inert surface is `alg` alone. `sep` lives on [`NetworkEstimator`](@ref), which a clustering source does not have, so there is no second field going quiet here.

# The diagonal

`Z[i, i]` is the top of the scale, never zero: `1` for any clustering source, and `separation_decay(decay, 0, dmax)` for [`Proximity`](@ref) over a graph — `n + 1` under the default [`LinearDecay`](@ref) and [`HopCount`](@ref), the observed diameter plus one under [`PathLength`](@ref)'s default budget, `1` for the members that pin `f(0) = 1`. That the diagonal is maximal is a contract on [`AbstractSeparationDecayAlgorithm`](@ref), checked before the loop by [`assert_separation_decay`](@ref).

# Algorithm

Over a graph source, under [`Proximity`](@ref):

 1. Build the structure from `X` through [`separation_graph`](@ref), giving `g`.
 2. Resolve the separation algorithm against the structure through [`resolve_separation`](@ref), giving `sep`.
 3. Measure the separations through [`separation_matrix`](@ref), giving `d`.
 4. Read the budget through [`separation_budget`](@ref), and score `d` through [`_proximity_features`](@ref).

Over a partition source, under any algorithm:

 1. Build the co-membership matrix from `X` through [`phylogeny_matrix`](@ref), and convert it to `eltype(X)`.
 2. Add the identity, restoring the diagonal that [`phylogeny_matrix`](@ref) subtracts.

# Arguments

  - `alg`: Phylogeny feature algorithm.
  - `pl`: Structure source — a network estimator (a graph) or a clustering estimator (a partition).
  - `X`: Asset returns matrix `observations × assets`.
  - `kwargs...`: Additional keyword arguments passed to the underlying phylogeny routines.

# Returns

  - `Z::Matrix{Float64}`: Square `assets × assets` feature matrix.

# Related

  - [`PhylogenyPanel`](@ref)
  - [`AbstractPhylogenyFeatureAlgorithm`](@ref)
  - [`Proximity`](@ref)
  - [`separation_graph`](@ref)
  - [`separation_matrix`](@ref)
  - [`separation_budget`](@ref)
  - [`is_related`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_features end
# `alg` is inert here: a partition is flat, so there is no separation structure to decay. See
# the docstring's caveat -- the resulting distance carries the partition and nothing more.
function phylogeny_features(::AbstractPhylogenyFeatureAlgorithm,
                            pl::AbstractClustersEstimator, X::MatNum; kwargs...)::Matrix
    return Matrix{eltype(X)}(phylogeny_matrix(pl, X; dims = 1, kwargs...).X) +
           LinearAlgebra.I
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Score a separation matrix under a decay, inside a budget.

The loop behind [`phylogeny_features`](@ref)'s [`Proximity`](@ref) method, split out because it is a
function of the **separations** alone: the structure, the estimator and the data are all spent by the
time it runs. Handing it a matrix is how the unreachable branch is tested — every structure a shipped
estimator builds is connected, so a disconnected one arrives as an argument rather than through a
test double that answers [`calc_adjacency`](@ref).

# Algorithm

 1. Probe the decay over `0:dmax` through [`assert_separation_decay`](@ref), before the pair loop.
 2. Allocate `Z`, of the element type `et` and the size of `d`, filled with zeros.
 3. For each pair, read its separation `duv` and ask [`is_related`](@ref) whether it is inside the
    budget, giving `rel`. This is the sentinel test as well as the budget rule, so it is the guard
    on the next step.
 4. Where `rel` holds, write `separation_decay(dk, duv, dmax)` into `Z[u, v]`. Where it does not,
    leave the zero of step 2.

# Arguments

  - `alg`: Proximity algorithm carrying the decay.
  - `sep`: Separation algorithm the matrix was measured under, forwarded to [`is_related`](@ref).
  - `d`: Separation matrix from [`separation_matrix`](@ref).
  - `dmax`: Separation budget from [`separation_budget`](@ref).
  - `et`: Element type of the result. [`phylogeny_features`](@ref) passes `eltype(X)`, so that
    [`AngularDist`](@ref) keeps its BLAS `gemm` path.

# Returns

  - `Z::Matrix`: Square `assets × assets` feature matrix.

# Related

  - [`phylogeny_features`](@ref)
  - [`Proximity`](@ref)
  - [`is_related`](@ref)
  - [`separation_decay`](@ref)
  - [`assert_separation_decay`](@ref)
"""
function _proximity_features(alg::Proximity, sep::AbstractSeparationAlgorithm, d::MatNum,
                             dmax::Number, et::Type)::Matrix
    dk = alg.decay
    # Under `HopCount` a separation only ever takes values in `0:dmax` here, so probing that
    # range is exhaustive rather than a spot check -- `dmax + 1` evaluations before an
    # `assets^2` loop. Under a non-integral budget the same range is a unit-spaced sample,
    # which is all `assert_separation_decay` needs: its endpoint evaluation at `dmax` closes
    # non-negativity over the continuum, and monotonicity stays sampled by design.
    assert_separation_decay(dk, 0:dmax, dmax)
    Z = zeros(et, size(d))
    for v in axes(d, 2), u in axes(d, 1)
        duv = @inbounds d[u, v]
        # `is_related` is the budget rule and the sentinel test at once -- `separation_matrix`
        # reports a sentinel for an unreachable pair, and both are unrelated. The branch stays
        # a short-circuiting `?:` because it also guards the decay call: `ifelse` would
        # evaluate it, and `ReciprocalDecay` overflows `1 + d` at `typemax`.
        rel = is_related(sep, duv, dmax)
        @inbounds Z[u, v] = rel ? et(separation_decay(dk, duv, dmax)) : zero(et)
    end
    return Z
end
function phylogeny_features(alg::Proximity, pl::AbstractNetworkEstimator, X::MatNum;
                            kwargs...)::Matrix
    # One structure per call. A rule in the budget field is answered against it, here, where
    # the data is in hand: `separation_budget` below cannot do it, because it takes `d`
    # rather than the structure by design.
    g = separation_graph(pl.sep, pl, X; dims = 1, kwargs...)
    sep = resolve_separation(pl.sep, pl, X, g; dims = 1, kwargs...)
    d = separation_matrix(sep, g)
    return _proximity_features(alg, sep, d, separation_budget(sep, pl, d), eltype(X))
end
"""
    panel_axis_labels(names::Nothing, n::Integer) -> Vector{String}
    panel_axis_labels(names::VecStr, n::Integer) -> Vector{String}

Name the trailing axis of a produced [`TensorPanelField`](@ref), from the carrier or positionally.

A producer's field has a labelled trailing axis, and neither a loadings matrix nor a proximity matrix names that axis as data. So the **data carrier** is the one place a name exists: `rd.nx` names a proximity field's assets, and `rd.nf` names a loadings field's factors. Where the carrier gives no name, or gives the wrong number of them, the labels are positional, `"1"` to `"n"`.

The count is checked rather than assumed. A reduced or re-based `L` has fewer columns than the carrier has factors, and a positional label is then the only honest one.

# Algorithm

The method that Julia selects is the algorithm.

 1. `names` is `nothing`: return the positional labels.
 2. `names` is a vector: return it as `String`s when it holds `n` of them, and the positional labels otherwise.

# Arguments

  - `names`: The carrier's names for the axis, or `nothing`.
  - `n`: Length of the axis to label.

# Returns

  - `labels::Vector{String}`: One label per trailing-axis entry.

# Related

  - [`RegressionPanel`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`TensorPanelField`](@ref)
  - [`asset_panel`](@ref)
"""
function panel_axis_labels(::Nothing, n::Integer)
    return [string(k) for k in 1:n]
end
function panel_axis_labels(names::VecStr, n::Integer)
    return length(names) == n ? String[String(s) for s in names] : [string(k) for k in 1:n]
end
"""
    carrier_asset_names(rd::Nothing) -> nothing
    carrier_asset_names(rd::AbstractReturnsResult) -> Option{<:VecStr}

Read the asset names a producer labels its trailing axis with, or `nothing`.

A producer is handed the two carriers the kernel received. Only the data carrier names the assets, so this is the one read, and it answers `nothing` when there is no data carrier.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `rd`: The data carrier, or `nothing`.

# Returns

  - `nx::Option{<:VecStr}`: The asset names, or `nothing`.

# Related

  - [`PhylogenyPanel`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`ReturnsResult`](@ref)
"""
function carrier_asset_names(::Nothing)
    return nothing
end
function carrier_asset_names(rd::AbstractReturnsResult)
    return rd.nx
end
"""
    regression_factor_names(rr::Regression{<:Any, Nothing, <:Any, <:Any},
                            rd::AbstractReturnsResult) -> Option{<:VecStr}
    regression_factor_names(rr, rd) -> nothing

Read the factor names a [`RegressionPanel`](@ref) labels its loadings axis with, or `nothing`.

The names are the carrier's `nf`, and they are read **only** where the loadings axis is the carrier's factor axis: a [`Regression`](@ref) whose `L` is unset, whose loadings are therefore the raw `M`, one column per factor the carrier holds. A reduced or re-based `L` has its own axis, and a [`CrossSectionalFactorModel`](@ref)'s factors are exposures that no data names, so both are labelled positionally by [`panel_axis_labels`](@ref).

# Algorithm

The method that Julia selects is the algorithm. The `Nothing` type parameter of `L` is the raw-`M` case, which is the same parameter the `swap(L, M)` rule of [`Regression`](@ref) fires on.

# Arguments

  - `rr`: The regression result the loadings came from.
  - `rd`: The data carrier, or `nothing`.

# Returns

  - `nf::Option{<:VecStr}`: The factor names, or `nothing`.

# Related

  - [`RegressionPanel`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`Regression`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression_factor_names(::Regression{<:Any, Nothing, <:Any, <:Any},
                                 rd::AbstractReturnsResult)
    return rd.nf
end
function regression_factor_names(::Any, ::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel holding the factor loadings the wrapped prior fitted.

`RegressionPanel` treats `pr.rr.L` — the coordinate system a factor model places each asset in — as the values of one tensor Panel Field, `"loadings"` on the axis `"factor"`. It is the cheapest real feature source in the library: a factor prior already computes the loadings, and already refits them per fold, so the features track the fold with no extra plumbing.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, set by [`DimensionReductionRegression`](@ref): the low-dimensional coordinate system the asset actually lives in. `M` is the *reconstructed* full-factor loadings. `pr.rr.L` always resolves — [`Regression`](@ref) swaps in `M` when `L` is unset — so this producer needs no branch and works behind every regression estimator, the time-series one and the cross-sectional one alike.

Both are assets-major, so the panel's asset axis is the carrier's with no transpose. The trailing axis is the reduced dimensions, which are not assets, so a produced loadings panel is never square in the sense [`features_are_assets`](@ref) means.

# Validation

  - The call supplies a prior result. Raises an [`IsNothingError`](@ref) naming the site.
  - The wrapped prior carries a regression (see [`assert_prior_regression`](@ref)). Nesting order does not matter: every wrapping estimator forwards `rr` and the factor block `fpr` (ADR 0046). What throws is a prior that never computed a regression at all, such as [`EmpiricalPrior`](@ref).

# Examples

```jldoctest
julia> RegressionPanel()
RegressionPanel()
```

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`asset_panel`](@ref)
  - [`FeatureDistance`](@ref)
  - [`FactorPrior`](@ref)
  - [`Regression`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`panel_axis_labels`](@ref)
"""
struct RegressionPanel <: AbstractAssetPanelEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel holding a square proximity matrix graded from a graph or a partition.

An `assets × assets` proximity matrix *is* a feature block whose feature `k` reads "is close to asset `k`", so a metric over its rows measures **neighbourhood overlap** — a standard notion of topological similarity — and needs almost no new estimation code. `PhylogenyPanel` returns it as one tensor Panel Field, `"proximity"` on the axis `"asset"`.

It is the one producer whose trailing axis *is* the asset axis. That needs no flag anywhere: it refits, so a subproblem gets its own square matrix over its own universe rather than a slice of a larger one, and [`features_are_assets`](@ref) compares the field's labels against the carrier's asset names where a hand-supplied panel meets a view.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyPanel(;
        pl::NwE_ClE = NetworkEstimator(),
        alg::AbstractPhylogenyFeatureAlgorithm = Proximity()
    ) -> PhylogenyPanel

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pl`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `pl`: Recursively viewed via [`port_opt_view`](@ref).

# Why the diagonal includes self

The diagonal is not a convention, it selects between two different algorithms. Measured on a three-node path `1 - 2 - 3` under the default [`AngularDist`](@ref), with the decay held flat at [`NoDecay`](@ref) so that the diagonal is the only thing that changes between the two columns:

| pair            | zero diagonal | self included |
|:--------------- | -------------:| -------------:|
| `1`-`3`, 2 hops | **0.0**       | 0.333         |
| `1`-`2`, 1 hop  | 0.5           | 0.196         |

The default [`LinearDecay`](@ref) grades the same graph rather than flattening it, so its numbers differ — `0.436` and `0.239` over a two-hop budget — while the ordering is the one the right-hand column shows.

With a zero diagonal the two **non-adjacent** endpoints come out identical and the adjacent pairs maximally far: rows are compared on who their neighbours are, never on whether they are each other's. That is *structural equivalence* — similarity of role — which is a real notion but the opposite of the proximity the name promises.

Including self also keeps subproblems well defined. An asset view of a spanning tree routinely isolates a vertex, and a zero-diagonal row for an isolated asset is a **zero row**: [`AngularDist`](@ref)'s zero-vector convention then puts every isolated asset at distance `0` from every other isolated asset, clustering them together for no reason. With self included they sit at maximal distance from everything, including each other.

# A clustering source is admitted, with a caveat worth reading

`pl` is bound by [`NwE_ClE`](@ref): a graph ([`NetworkEstimator`](@ref)) or a partition ([`ClustersEstimator`](@ref)). Both are *estimators*, so both refit.

A partition carries much less than a graph, and the shortfall is measurable rather than stylistic. Its matrix is `P * transpose(P)` with the diagonal restored, so row `i` is the co-membership indicator of asset `i` and two rows are either identical or disjoint. On a seven-asset universe clustered `[1, 1, 1, 2, 2, 3, 3]` the whole distance matrix takes **two** distinct values under the default [`AngularDist`](@ref): `0.0` within a cluster and `0.5` across one, whatever the cluster sizes. The raw `phylogeny_matrix` output, whose `- I` this producer undoes, takes three — `0.0`, `0.333` and `0.5` — because that `- I` makes each row of a *pair* a lone `1` pointing at the other member, so the two rows are orthogonal and a size-two cluster's **within**-cluster distance equals its **across**-cluster distance. Restoring the diagonal repairs exactly that case; the coarseness remains.

Clustering a re-encoded clustering largely returns the clustering. Prefer a graph source unless the partition is what you actually want to measure.

# Provenance

The source is a [`NetworkEstimator`](@ref) and never a precomputed [`PhylogenyResult`](@ref), because an Estimator does not hold a Result (see `CONTEXT.md` §1). This producer is therefore **endogenous**: the graph is filtered from the returns correlation, so it measures topology the correlation implies rather than structure outside it, and it refits on every fold and every subproblem.

It reads no prior result, so it is the one producer that runs at a pre-prior site such as preselection.

# Examples

```jldoctest
julia> PhylogenyPanel()
PhylogenyPanel
   pl ┼ NetworkEstimator
      │    ce ┼ PortfolioOptimisersCovariance
      │       │   ce ┼ Covariance
      │       │      │    me ┼ SimpleExpectedReturns
      │       │      │       │   w ┴ nothing
      │       │      │    ce ┼ GeneralCovariance
      │       │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │      │       │    w ┴ nothing
      │       │      │   alg ┼ FullMoment()
      │       │      │     w ┴ nothing
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
      │   sep ┼ HopCount
      │       │   n ┴ Int64: 1
  alg ┼ Proximity
      │   decay ┴ LinearDecay()
```

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`asset_panel`](@ref)
  - [`phylogeny_features`](@ref)
  - [`Proximity`](@ref)
  - [`FeatureDistance`](@ref)
  - [`NwE_ClE`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct PhylogenyPanel <: AbstractAssetPanelEstimator
    """
    $(field_dict[:plfe])
    """
    @fprop @vprop pl
    """
    $(field_dict[:plfalg])
    """
    alg
    function PhylogenyPanel(pl::NwE_ClE, alg::AbstractPhylogenyFeatureAlgorithm)
        return new{typeof(pl), typeof(alg)}(pl, alg)
    end
end
function PhylogenyPanel(; pl::NwE_ClE = NetworkEstimator(),
                        alg::AbstractPhylogenyFeatureAlgorithm = Proximity())::PhylogenyPanel
    return PhylogenyPanel(pl, alg)
end
"""
    asset_panel(ape::RegressionPanel, pr, rd, X) -> AssetPanel
    asset_panel(ape::PhylogenyPanel, pr, rd, X) -> AssetPanel

Build the static [`AssetPanel`](@ref) a producer returns, at the point of use.

Each method returns a panel holding **one** [`TensorPanelField`](@ref), because a loadings matrix and a proximity matrix are each one quantity with a labelled third axis. The trailing axis is labelled off the data carrier where a name exists there, and positionally otherwise; [`panel_axis_labels`](@ref) states the rule.

A producer runs on the subproblem's own prior and returns, so nothing views what it built and a fold refits it.

# Algorithm

A [`RegressionPanel`](@ref) takes four steps:

 1. Check that a prior result reached the call, with [`assert_producer_prior`](@ref).
 2. Check that the prior carries a regression, with [`assert_prior_regression`](@ref).
 3. Label the loadings axis with [`panel_axis_labels`](@ref), from [`regression_factor_names`](@ref).
 4. Return the panel holding `pr.rr.L` as the field `"loadings"` on the axis `"factor"`.

A [`PhylogenyPanel`](@ref) takes three steps:

 1. Grade the structure into an `assets × assets` matrix with [`phylogeny_features`](@ref).
 2. Label the trailing axis with [`panel_axis_labels`](@ref), from [`carrier_asset_names`](@ref).
 3. Return the panel holding that matrix as the field `"proximity"` on the axis `"asset"`.

# Arguments

  - `ape`: The producer.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for the axis names alone.
  - `X`: Returns matrix of the subproblem, observations × assets.

# Validation

  - A [`RegressionPanel`](@ref) needs a prior result carrying a regression. Raises an [`IsNothingError`](@ref).

# Returns

  - `pnl::AssetPanel`: A static Asset Panel holding one tensor Panel Field.

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`RegressionPanel`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`TensorPanelField`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`FeatureDistance`](@ref)
"""
function asset_panel(ape::RegressionPanel, pr, rd, ::Any)
    assert_producer_prior(ape, pr)
    assert_prior_regression(pr, :pe)
    L = pr.rr.L
    return AssetPanel(;
                      pf = [TensorPanelField(; name = "loadings", axis = "factor",
                                             labels = panel_axis_labels(regression_factor_names(pr.rr,
                                                                                                rd),
                                                                        size(L, 2)),
                                             vals = L)])
end
function asset_panel(ape::PhylogenyPanel, ::Any, rd, X::MatNum)
    Zp = phylogeny_features(ape.alg, ape.pl, X)
    return AssetPanel(;
                      pf = [TensorPanelField(; name = "proximity", axis = "asset",
                                             labels = panel_axis_labels(carrier_asset_names(rd),
                                                                        size(Zp, 2)),
                                             vals = Zp)])
end
"""
    assert_producer_prior(ape::AbstractAssetPanelEstimator, pr::AbstractPriorResult) -> nothing
    assert_producer_prior(ape::AbstractAssetPanelEstimator, pr) -> Union{}

Assert that a producer that reads a prior result was handed one, and name the site when it was not.

A producer runs wherever the estimator holding it runs, and one of those sites has no prior by construction: preselection is fitted from the returns data alone, before any prior exists. So does the shortest public call, `clusterise(cle, rd)`, which puts a data carrier in the `pr` slot. This turns both into a diagnostic that names the site rather than a `nothing` field access or a `MethodError`.

# Algorithm

The method that Julia selects is the algorithm. A prior result returns; anything else, `nothing` and a data carrier alike, raises.

# Arguments

  - `ape`: The producer, named in the message.
  - `pr`: The prior result, or whatever reached the slot.

# Validation

  - `isa(pr, AbstractPriorResult)`. Raises an [`IsNothingError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`asset_panel`](@ref)
  - [`RegressionPanel`](@ref)
  - [`ClusterGroups`](@ref)
  - [`IsNothingError`](@ref)
"""
function assert_producer_prior(::AbstractAssetPanelEstimator,
                               ::AbstractPriorResult)::Nothing
    return nothing
end
function assert_producer_prior(ape::AbstractAssetPanelEstimator, ::Any)::Nothing
    return throw(IsNothingError("$(nameof(typeof(ape))) reads the prior result it is handed, and this call supplied none. A pre-prior site supplies none by construction: preselection is fitted from the returns data alone, and `clusterise(cle, rd)` puts a data carrier in the prior slot. Two ways forward:\n  1. Build the Asset Panel from data and put it on the `ReturnsResult`, then leave `FeatureDistance.ape` at `nothing`.\n  2. Use a producer that reads no prior, such as `PhylogenyPanel`."))
end

export Proximity, phylogeny_features, RegressionPanel, PhylogenyPanel
