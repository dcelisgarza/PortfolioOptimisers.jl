"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny feature algorithms.

A phylogeny feature algorithm is the rule that turns the structure of a [`PhylogenyPanel`](@ref) source into an `assets × assets` proximity matrix. The one member, [`Proximity`](@ref), scores each pair of assets by a decay of their separation.

Two other choices need no new member. A different fall-off is an [`AbstractSeparationDecayAlgorithm`](@ref), which [`Proximity`](@ref) holds in its `decay` field. A different measure of separation is an [`AbstractSeparationAlgorithm`](@ref), which the source [`NetworkEstimator`](@ref) holds in its `sep` field. A new member is for a rule that is not a decay of a separation, for example a matrix of role similarity.

Every member puts each asset at the top of its own scale, so the diagonal of the matrix holds the largest entries. [`PhylogenyPanel`](@ref) states why the diagonal matters.

# Interfaces

To add a rule, subtype `AbstractPhylogenyFeatureAlgorithm` and implement the method below. A clustering source needs no new method, because the co-membership method of [`phylogeny_features`](@ref) accepts every member.

## `phylogeny_features`

  - `phylogeny_features(alg::MyPhylogenyFeatureAlgorithm, pl::AbstractNetworkEstimator, X::MatNum; kwargs...) -> Matrix`: Build the feature matrix of the graph that `pl` fits on `X`.

### Arguments

  - `alg`: The new phylogeny feature algorithm.
  - `pl`: Network estimator that fits the graph.
  - `X`: Asset returns matrix, `observations × assets`.
  - `kwargs...`: Keyword arguments for the phylogeny routines.

### Returns

  - `Z::Matrix`: Square `assets × assets` feature matrix, with the largest entry of each row on the diagonal.

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

Phylogeny feature algorithm that scores each pair of assets by a decay of their separation.

The `sep` field of the source [`NetworkEstimator`](@ref) sets the separation and its budget. [`HopCount`](@ref) counts the edges of a shortest path, and [`PathLength`](@ref) sums their weights. The `decay` field sets the fall-off. A pair inside the budget scores the decay of its separation, and every other pair scores zero.

The score reads the separation, not the walk count `sum(A^i)` of the adjacency matrix `A`. A hub gathers walks faster than other assets, so a walk count mixes the number of neighbours of two assets into their closeness. [`phylogeny_matrix`](@ref) sums that walk count and then clamps it to `[0, 1]`, which removes the number of steps. `Proximity` keeps it.

# Mathematical definition

```math
\\begin{align}
Z_{i,\\,k} &= \\begin{cases}
f\\left(D_{i,\\,k}\\right) & D_{i,\\,k} \\leq d_{\\mathrm{max}}\\,, \\\\
0 & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:Z_prox])
  - $(math_dict[:D_sep])
  - $(math_dict[:dmax_sep])
  - $(math_dict[:f_sep_decay])

The diagonal is ``f(0)``, the largest entry of ``\\mathbf{Z}``, because a decay is largest at a separation of zero. An unreachable pair has ``D_{i,\\,k} = +\\infty``, which exceeds every budget, so it scores zero. A decay is non-negative inside the budget, so a zero entry means that the pair is unreachable, outside the budget, or scored zero by the decay. No shipped decay scores zero inside the budget.

# The fall-off and the budget

`decay` sets the fall-off, and `sep` on the source sets the budget. The budget is the only truncation. The two are separate because an exponential never reaches zero, so a fall-off cannot state a budget. Under the default [`LinearDecay`](@ref) and a [`HopCount`](@ref) budget of `n`, the asset itself scores `n + 1`, a direct neighbour scores `n`, and a pair `n` hops apart scores `1`. Under [`ExponentialDecay`](@ref) or [`ReciprocalDecay`](@ref) the diagonal is `1`, and the parameter of the decay sets the fall-off whatever the budget.

[`NoDecay`](@ref) scores `1` at every separation, so `Proximity(; decay = NoDecay())` gives `1` inside the budget and `0` outside it. The result is the indicator of the neighbourhood that the budget selects, not a matrix of ones.

# Hop counts and path lengths

A `Z` graded over hop counts and a `Z` graded over path lengths are both valid inputs to every consumer, but their values do not compare. The budgets have different units, the supports differ, and under [`LinearDecay`](@ref) the scales differ too. On real data the two separations order the pairs almost alike, and [`PathLength`](@ref) states the measurement. That agreement does not make the numbers of one run comparable with the numbers of another.

Under the default `dmax = nothing` of [`PathLength`](@ref), the budget is the observed diameter of the graph. Under [`LinearDecay`](@ref) the diagonal ``d_{\\mathrm{max}} + 1`` then depends on the data, and a diameter that changes between cross-validation folds shifts every entry of `Z`. A fixed `dmax` keeps the scale fixed across folds. A decay with `f(0) = 1` does not have this problem.

# Unreachable pairs

[`separation_matrix`](@ref) writes a sentinel for an unreachable pair. The sentinel is `typemax(Int)` for a hop count, and `Inf` for a path length over `Float64` weights. The budget test fails at the sentinel, so the kernel never evaluates the decay there. [`ReciprocalDecay`](@ref) needs that guard, because it overflows `1 + d` at `typemax(Int)`, and a fractional `power` then raises a `DomainError`.

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

Turn a graph source or a partition source into a square `assets × assets` feature matrix.

This is the kernel of [`PhylogenyPanel`](@ref). Every method returns a matrix of element type `float_if_integer(eltype(X))`, which is the element type of the returns, or `Float64` for integer returns. It is never the `Int` or `BitMatrix` that the phylogeny routines give, so [`AngularDist`](@ref) keeps its BLAS `gemm` path, a `Float32` history gives a `Float32` matrix, and a fractional score fits into the matrix of an integer history.

`pl` is an estimator, a [`NetworkEstimator`](@ref) or a [`ClustersEstimator`](@ref). It is never a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref), because an Estimator does not hold a Result, as `CONTEXT.md` §1 states. So each call builds the structure from `X`, and the structure follows a cross-validation fold or the subproblem of a meta-optimiser.

A graph has separations, and `alg` decays them. A partition has none, because two assets are in one cluster or they are not. So every algorithm gives the same co-membership matrix of a partition, and `alg` has no effect there. `FeatureDistance` treats its collapse `alg` the same way on a static feature matrix. `sep` is a field of [`NetworkEstimator`](@ref) alone, so no other field loses its effect on a partition.

# Mathematical definition

```math
\\begin{align}
Z_{i,\\,k} &= \\begin{cases}
f\\left(D_{i,\\,k}\\right) & D_{i,\\,k} \\leq d_{\\mathrm{max}}\\,, \\\\
0 & \\text{otherwise}\\,,
\\end{cases} \\quad \\text{for a graph}\\,, \\\\
Z_{i,\\,k} &= \\begin{cases}
1 & c_{i} = c_{k}\\,, \\\\
0 & \\text{otherwise}\\,,
\\end{cases} \\quad \\text{for a partition}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Z_prox])
  - $(math_dict[:D_sep])
  - $(math_dict[:dmax_sep])
  - $(math_dict[:f_sep_decay])
  - ``c_{i}``: Cluster of asset ``i`` in the partition that `pl` fits.

The diagonal holds the largest entry of each row. It is ``f(0)`` for a graph, and ``1`` for a partition. [`Proximity`](@ref) states ``f(0)`` for each decay. [`assert_separation_decay`](@ref) checks that ``f(0)`` is the largest score of a decay before the kernel scores a pair.

# Algorithm

Over a graph source, under [`Proximity`](@ref):

 1. Build the structure from `X` through [`separation_graph`](@ref), giving `g`.
 2. Resolve the separation algorithm against the structure through [`resolve_separation`](@ref), giving `sep`.
 3. Measure the separations through [`separation_matrix`](@ref), giving `d`.
 4. Read the budget through [`separation_budget`](@ref), and score `d` in the element type `float_if_integer(eltype(X))` through [`_proximity_features`](@ref), giving `Z`.

Over a partition source, under any algorithm:

 1. Build the co-membership matrix from `X` through [`phylogeny_matrix`](@ref), and convert it to the element type `float_if_integer(eltype(X))`.
 2. Add the identity, which restores the diagonal that [`phylogeny_matrix`](@ref) subtracts, giving `Z`.

# Arguments

  - `alg`: Phylogeny feature algorithm.
  - `pl`: Structure source, a network estimator for a graph or a clustering estimator for a partition.
  - `X`: Asset returns matrix, `observations × assets`.
  - `kwargs...`: Keyword arguments for the phylogeny routines.

# Returns

  - `Z::Matrix`: Square `assets × assets` feature matrix of element type `float_if_integer(eltype(X))`.

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
    return Matrix{float_if_integer(eltype(X))}(phylogeny_matrix(pl, X; dims = 1, kwargs...).X) +
           LinearAlgebra.I
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Score a separation matrix under a decay, inside a budget.

This is the loop of the [`Proximity`](@ref) method of [`phylogeny_features`](@ref). It is a function of its own because it reads the separations alone, and the structure, the estimator and the data play no part in it. A test hands it a disconnected separation matrix to reach the unreachable branch, because every structure that a shipped estimator builds is connected.

# Algorithm

 1. Probe the decay over `0:dmax` through [`assert_separation_decay`](@ref), before the pair loop.
 2. Allocate `Z`, of the element type `et` and the size of `d`, filled with zeros.
 3. For each pair, read its separation `duv`, and ask [`is_related`](@ref) whether it is inside the budget, giving `rel`. This test also rejects the sentinel of an unreachable pair, so it guards the next step.
 4. Where `rel` holds, write `separation_decay(dk, duv, dmax)` into `Z[u, v]`, which is the form that [`Proximity`](@ref) states. Where it does not hold, keep the zero of step 2.

# Arguments

  - `alg`: Proximity algorithm that holds the decay.
  - `sep`: Separation algorithm that measured `d`, forwarded to [`is_related`](@ref).
  - `d`: Separation matrix from [`separation_matrix`](@ref).
  - `dmax`: Separation budget from [`separation_budget`](@ref).
  - `et`: Element type of the result. [`phylogeny_features`](@ref) passes `float_if_integer(eltype(X))`, so that [`AngularDist`](@ref) keeps its BLAS `gemm` path.

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
    return _proximity_features(alg, sep, d, separation_budget(sep, pl, d),
                               float_if_integer(eltype(X)))
end
"""
    panel_axis_labels(names::Nothing, n::Integer) -> Vector{String}
    panel_axis_labels(names::VecStr, n::Integer) -> Vector{String}

Name the trailing axis of a [`TensorPanelField`](@ref) that a producer builds, from the carrier or by position.

The field of a producer has a labelled trailing axis, and neither a loadings matrix nor a proximity matrix holds names for that axis. The data carrier is the one place that holds them. `rd.nx` names the assets of a proximity field, and `rd.nf` names the factors of a loadings field. When the carrier gives no names, or a wrong number of them, the labels are the positions `"1"` to `"n"`.

The function checks the count. A reduced or re-based `L` has fewer columns than the carrier has factors, and the carrier's names then belong to other columns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `names` is `nothing`: return the positional labels.
 2. `names` is a vector: return it as `String`s when it holds `n` names, and the positional labels otherwise.

# Arguments

  - `names`: The names that the carrier gives the axis, or `nothing`.
  - `n`: Length of the axis to label.

# Returns

  - `labels::Vector{String}`: One label per entry of the trailing axis.

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

Read the asset names that label the trailing axis of a producer's field, or `nothing`.

The kernel hands a producer the two carriers that it received. Only the data carrier names the assets, so this function reads the data carrier, and returns `nothing` when there is none.

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
    regression_factor_names(rr::CrossSectionalFactorModel, rd) -> Option{<:VecStr}
    regression_factor_names(rr, rd) -> nothing

Read the factor names that label the loadings axis of a [`RegressionPanel`](@ref), or `nothing`.

The function reads a name wherever one exists for the axis that `pr.rr.L` spans. Otherwise [`panel_axis_labels`](@ref) labels the axis by position.

  - A time-series [`Regression`](@ref) holds no factor names, so the names are the `nf` of the carrier. The function reads them only when the loadings axis is the factor axis of the carrier. That is a `Regression` whose `L` is unset, so that its loadings are the raw `M`, with one column for each factor of the carrier. A reduced or re-based `L` has an axis of its own, and no data names it.
  - A [`CrossSectionalFactorModel`](@ref) names its own factors. The prior derives the raw axis from its Exposure Estimators and stores it as `nf` on the block. When the block holds a family re-basis, [`cs_diagnostic_factor_names`](@ref) maps that list onto the reduced axis, which is the axis that `L` spans. So the names come from the block, and the function does not read the data carrier.

# Algorithm

The method that Julia selects is the algorithm. The `Nothing` type parameter of `L` selects the raw `M`. The `swap(L, M)` rule of [`Regression`](@ref) reads the same parameter.

# Arguments

  - `rr`: The regression result that holds the loadings.
  - `rd`: The data carrier, or `nothing`.

# Returns

  - `nf::Option{<:VecStr}`: The factor names, or `nothing`.

# Related

  - [`RegressionPanel`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`Regression`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`cs_diagnostic_factor_names`](@ref)
  - [`ReturnsResult`](@ref)
"""
function regression_factor_names(::Regression{<:Any, Nothing, <:Any, <:Any},
                                 rd::AbstractReturnsResult)
    return rd.nf
end
function regression_factor_names(rr::CrossSectionalFactorModel, ::Any)
    return cs_diagnostic_factor_names(rr)
end
function regression_factor_names(::Any, ::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel that holds the factor loadings of the wrapped prior.

`RegressionPanel` takes `pr.rr.L`, the coordinates that a factor model gives each asset, as the values of one tensor Panel Field, `"loadings"` on the axis `"factor"`. It needs no estimation of its own. A factor prior computes the loadings already, and refits them in each fold, so the features follow the fold.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, and [`DimensionReductionRegression`](@ref) sets it. It holds the coordinates of each asset in the reduced space. `M` holds the loadings on every factor, rebuilt from `L`. `pr.rr.L` always has a value, because [`Regression`](@ref) returns `M` when `L` is unset. So this producer needs no branch, and it works behind the time-series and the cross-sectional regression estimators alike.

Both matrices have one row per asset, so the asset axis of the panel is the asset axis of the carrier, with no transpose. The trailing axis holds factors or reduced dimensions, not assets, so a loadings panel is never square in the sense of [`features_are_assets`](@ref).

# Validation

  - The call supplies a prior result. Raises an [`IsNothingError`](@ref) that names the call site.
  - The wrapped prior holds a regression, see [`assert_prior_regression`](@ref). The order of nesting has no effect, because every wrapping estimator forwards `rr` and the factor block `fpr`. A prior that computes no regression, such as [`EmpiricalPrior`](@ref), raises the error.

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

Builds an Asset Panel that holds a square proximity matrix, graded from a graph or a partition.

In an `assets × assets` proximity matrix, feature `k` of a row reads "is close to asset `k`". So a metric over the rows measures the overlap of two neighbourhoods, which is a standard measure of topological similarity, and it needs almost no new estimation code. `PhylogenyPanel` returns the matrix as one tensor Panel Field, `"proximity"` on the axis `"asset"`.

It is the one producer whose trailing axis is the asset axis, and it needs no flag for that. It refits, so a subproblem gets its own square matrix over its own assets, not a slice of a larger matrix. Where a panel supplied by hand meets a view, [`features_are_assets`](@ref) compares the labels of the field with the asset names of the carrier.

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

The diagonal selects between two different algorithms. The table measures a three-node path `1 - 2 - 3` under the default [`AngularDist`](@ref) and the default one-hop budget. The decay is [`NoDecay`](@ref), so the diagonal is the only difference between the two columns.

| pair            | zero diagonal | self included |
|:--------------- | -------------:| -------------:|
| `1`-`3`, 2 hops | 0.0           | 0.333         |
| `1`-`2`, 1 hop  | 0.5           | 0.196         |

The default [`LinearDecay`](@ref) grades the path over the same budget, so its distances are `0.436` for `1`-`3` and `0.239` for `1`-`2`. The order of the two pairs is the order of the right-hand column.

With a zero diagonal, the two endpoints, which are not adjacent, have a distance of zero, and each adjacent pair has the largest distance. The metric then compares two rows by their shared neighbours, and ignores whether the two assets are neighbours of each other. That is structural equivalence, a similarity of role, and it is the opposite of proximity.

The diagonal also keeps a subproblem well defined. A view of a spanning tree on a subset of the assets can isolate a vertex. With a zero diagonal, the row of an isolated asset is a zero row, and the zero-vector convention of [`AngularDist`](@ref) puts every isolated asset at distance `0` from every other isolated asset. With self included, each isolated asset is at distance `0.5` from every other asset, the largest distance between two non-negative rows.

# A clustering source

[`NwE_ClE`](@ref) bounds `pl`, so it is a graph ([`NetworkEstimator`](@ref)) or a partition ([`ClustersEstimator`](@ref)). Both are estimators, so both refit.

A partition holds less information than a graph. Its matrix is the co-membership indicator `P * transpose(P)`, where `P` is the `assets × clusters` membership matrix. Row `i` marks the assets in the cluster of asset `i`, so two rows are equal or disjoint. On seven assets in the clusters `[1, 1, 1, 2, 2, 3, 3]`, the distance matrix under the default [`AngularDist`](@ref) takes two values, `0.0` inside a cluster and `0.5` across clusters, whatever the sizes of the clusters. The raw output of [`phylogeny_matrix`](@ref) is `P * transpose(P) - I`, and it takes three values, `0.0`, `0.333` and `0.5`. In a cluster of two assets, the `- I` leaves each row with a single `1` at the other member, so the two rows are orthogonal, and the distance inside that cluster equals the distance across clusters. This producer adds `I` back, which repairs that case, but the matrix stays coarse.

A clustering of a feature matrix built from a clustering gives back much the same clustering. Use a graph source unless the partition is the thing to measure.

# Provenance

The source is a [`NetworkEstimator`](@ref) and never a precomputed [`PhylogenyResult`](@ref), because an Estimator does not hold a Result, as `CONTEXT.md` §1 states. The producer is therefore endogenous. It filters the graph from the correlation of the returns, so it measures the topology that the correlation implies and no structure outside it. It refits on every fold and on every subproblem.

It reads no prior result, so it is the one producer that runs where no prior exists yet, for example in preselection.

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

Build the static [`AssetPanel`](@ref) of a producer, at the point of use.

Each method returns a panel that holds one [`TensorPanelField`](@ref), because a loadings matrix and a proximity matrix are each one quantity with a labelled third axis. The labels of the trailing axis come from the data carrier or from the regression block where a name exists, and are positions otherwise. [`panel_axis_labels`](@ref) and [`regression_factor_names`](@ref) state the rule.

A producer runs on the prior and the returns of its own subproblem, so no view reads the panel that it builds, and a fold refits it. Called on its own, on a prior fitted on a point-in-time Asset Panel, a [`RegressionPanel`](@ref) reads the loadings on the Investable Mask of the prior and answers for the full universe. Outside the mask each asset gets a zero row and a false observed mask, so an optimiser can take the panel back. [`expand_investable_loadings`](@ref) states the rule.

# Algorithm

A [`RegressionPanel`](@ref) takes seven steps:

 1. Check that a prior result reached the call, with [`assert_producer_prior`](@ref).
 2. Check that the prior holds a regression, with [`assert_prior_regression`](@ref).
 3. Reduce the prior to its Investable Mask through [`investable_mask`](@ref) and [`port_opt_view`](@ref), giving `prr`, and read its loadings `L`. A prior fitted on a point-in-time Asset Panel writes `NaN` into the loadings of every asset outside its mask, and a Panel Field admits no `NaN`. Inside an optimiser the prior arrives reduced, and the view covers the whole universe.
 4. Count the assets on the mask whose loadings are not all finite, giving `nnf`, and refuse a count above zero. Such an asset has a finite moment and a loadings row that is not finite, which is a defect of the regression.
 5. Label the loadings axis with [`panel_axis_labels`](@ref), from [`regression_factor_names`](@ref).
 6. Expand the loadings back onto the full universe with [`expand_investable_loadings`](@ref), giving `vals` and `omsk`.
 7. Return the panel that holds them as the field `"loadings"` on the axis `"factor"`.

A [`PhylogenyPanel`](@ref) takes three steps:

 1. Grade the structure into an `assets × assets` matrix with [`phylogeny_features`](@ref), giving `Zp`.
 2. Label the trailing axis with [`panel_axis_labels`](@ref), from [`carrier_asset_names`](@ref).
 3. Return the panel that holds `Zp` as the field `"proximity"` on the axis `"asset"`.

# Arguments

  - `ape`: The producer.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) Read for the axis names alone.
  - `X`: Returns matrix of the subproblem, observations × assets.

# Validation

  - A [`RegressionPanel`](@ref) needs a prior result that holds a regression. Raises an [`IsNothingError`](@ref).
  - A [`RegressionPanel`](@ref) needs every loading inside the Investable Mask of the prior to be finite. Raises an [`IsNonFiniteError`](@ref) that counts the assets whose loadings are not finite.

# Returns

  - `pnl::AssetPanel`: A static Asset Panel that holds one tensor Panel Field.

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
    imsk = investable_mask(pr)
    prr = isnothing(imsk) ? pr : port_opt_view(pr, findall(imsk))
    L = prr.rr.L
    nnf = count(i -> !all(isfinite, view(L, i, :)), axes(L, 1))
    @argcheck(iszero(nnf),
              IsNonFiniteError("`RegressionPanel` reads the factor loadings `pr.rr.L` as a Panel Field, and $(nnf) of the $(size(L, 1)) assets inside the prior's Investable Mask carry a loading that is not finite, so the field cannot be built over them. An asset outside the mask is written as a zero row with a false observed mask and never reaches this check, so every asset it counts has a finite moment and a loadings row that is not: a defect of the regression, not of the universe.\nGot\npr => $(nameof(typeof(pr)))\nrr => $(nameof(typeof(pr.rr)))\nassets with a non-finite loading => $(nnf)"))
    vals, omsk = expand_investable_loadings(L, imsk)
    return AssetPanel(;
                      pf = [TensorPanelField(; name = "loadings", axis = "factor",
                                             labels = panel_axis_labels(regression_factor_names(prr.rr,
                                                                                                rd),
                                                                        size(L, 2)),
                                             vals = vals, omsk = omsk)])
end
"""
    expand_investable_loadings(L::MatNum, imsk::Nothing) -> (L, nothing)
    expand_investable_loadings(L::MatNum, imsk::BitVector) -> (vals, omsk)

Write the loadings that a [`RegressionPanel`](@ref) read on the Investable Mask back onto the full asset universe.

A prior fitted on a point-in-time Asset Panel writes `NaN` into the loadings of every asset outside its Investable Mask, and a Panel Field admits no `NaN`. So the producer reads the loadings on the mask, and this function expands them. Each asset outside the mask gets a zero row and a false observed mask. Every uncertainty set that a caller fits on its own on such a prior follows the same rule, and a Panel Field already has that shape for a cell that a fill policy wrote.

A zero row is a zero feature vector. [`AngularDist`](@ref) puts it at distance `1` from every asset that has loadings, and at distance `0` from every other asset that has none. The selector `"loadings" => :observed` reads the mask as a column.

# Mathematical definition

```math
\\begin{align}
\\mathbf{V}_{\\mathcal{M},\\,\\cdot} &= \\mathbf{L}\\,, \\\\
\\mathbf{V}_{i,\\,\\cdot} &= \\boldsymbol{0}\\,, \\quad i \\notin \\mathcal{M}\\,, \\\\
O_{i,\\,j} &= \\mathbb{1}\\left[i \\in \\mathcal{M}\\right]\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}``: Loadings over every asset, of size ``N \\times K``.
  - ``\\mathbf{L}``: Loadings on the Investable Mask, of size ``\\lvert \\mathcal{M} \\rvert \\times K``, with rows in the order of the assets.
  - ``\\mathcal{M}``: Assets inside the Investable Mask.
  - ``O_{i,\\,j}``: Entry ``(i, j)`` of the observed mask, of the size of ``\\mathbf{V}``.
  - $(math_dict[:N])
  - ``K``: Number of columns of the loadings.

The rows of ``\\mathbf{V}`` on the mask are ``\\mathbf{L}``, so a view of the expanded field at the mask gives back the reduced loadings. An optimiser on the full universe can therefore take a panel that was built on its own.

# Algorithm

The method that Julia selects is the algorithm.

 1. `imsk` is `nothing`, or every asset is investable: the loadings are those of the full universe, and the field holds no mask.
 2. Otherwise allocate a zero frame `vals` in `eltype(L)` over every asset, write `L` into the rows of the mask, and build the observed mask `omsk`, `true` on those rows and `false` elsewhere.

# Arguments

  - `L`: The loadings on the Investable Mask, `investable assets × factors`.
  - `imsk`: The Investable Mask of the prior over every asset, or `nothing`.

# Returns

  - `vals::MatNum`: The loadings over every asset.
  - `omsk::Option{<:AbstractMatrix{Bool}}`: The observed mask of the same shape, or `nothing` when every asset is investable.

# Related

  - [`RegressionPanel`](@ref)
  - [`asset_panel`](@ref)
  - [`investable_mask`](@ref)
  - [`TensorPanelField`](@ref)
"""
function expand_investable_loadings(L::MatNum, ::Nothing)
    return L, nothing
end
function expand_investable_loadings(L::MatNum, imsk::BitVector)
    if all(imsk)
        return L, nothing
    end
    vals = zeros(eltype(L), length(imsk), size(L, 2))
    vals[imsk, :] = L
    omsk = falses(size(vals))
    omsk[imsk, :] .= true
    return vals, omsk
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

Check that a producer that reads a prior result received one, and name the call site when it did not.

A producer runs wherever the estimator that holds it runs, and one of those sites has no prior. Preselection fits from the returns alone, before a prior exists. The shortest public call, `clusterise(cle, rd)`, also has none, because it puts a data carrier in the `pr` slot. This function turns both into an error that names the site, in place of a field access on `nothing` or a `MethodError`.

# Algorithm

The method that Julia selects is the algorithm. A prior result returns. Anything else, `nothing` and a data carrier alike, raises.

# Arguments

  - `ape`: The producer, named in the message.
  - `pr`: The prior result, or the value in the prior slot.

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
public AbstractPhylogenyFeatureAlgorithm
