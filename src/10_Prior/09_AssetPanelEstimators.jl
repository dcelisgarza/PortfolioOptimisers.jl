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

`decay = nothing` scores `1` at every separation, so `Proximity(; decay = nothing)` gives `1` inside the budget and `0` outside it. The result is the indicator of the neighbourhood that the budget selects, not a matrix of ones.

!!! note "What the proximity gives out of sample"

    Under [`LinearDecay`](@ref) and under [`ExponentialDecay`](@ref), the proximity did not give a lower risk or a higher Sharpe ratio out of sample than the distances that read the returns alone. [`PhylogenyPanel`](@ref) states the measurement.

# Hop counts and path lengths

A `Z` graded over hop counts and a `Z` graded over path lengths are both valid inputs to every consumer, but their values do not compare. The budgets have different units, the supports differ, and under [`LinearDecay`](@ref) the scales differ too. On real data the two separations order the pairs almost alike, and [`PathLength`](@ref) states the measurement. That agreement does not make the numbers of one run comparable with the numbers of another.

Under the default `dmax = nothing` of [`PathLength`](@ref), the budget is the observed diameter of the graph. Under [`LinearDecay`](@ref) the diagonal ``d_{\\mathrm{max}} + 1`` then depends on the data, and a diameter that changes between cross-validation folds shifts every entry of `Z`. A fixed `dmax` keeps the scale fixed across folds. A decay with `f(0) = 1` does not have this problem.

# Unreachable pairs

[`separation_matrix`](@ref) writes a sentinel for an unreachable pair. The sentinel is `typemax(Int)` for a hop count, and `Inf` for a path length over `Float64` weights. The budget test fails at the sentinel, so the kernel never evaluates the decay there. [`ReciprocalDecay`](@ref) needs that guard, because it overflows `1 + d` at `typemax(Int)`, and a fractional `power` then raises a `DomainError`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Proximity(;
        decay::Option{<:AbstractSeparationDecayAlgorithm} = LinearDecay()
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
  - [`separation_decay`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`separation_matrix`](@ref)
  - [`Option`](@ref)
"""
@concrete struct Proximity <: AbstractPhylogenyFeatureAlgorithm
    """
    $(field_dict[:sdecay])
    """
    decay
    function Proximity(decay::Option{<:AbstractSeparationDecayAlgorithm})
        return new{typeof(decay)}(decay)
    end
end
function Proximity(;
                   decay::Option{<:AbstractSeparationDecayAlgorithm} = LinearDecay())::Proximity
    return Proximity(decay)
end
"""
    phylogeny_features(alg::Proximity, pl::AbstractNetworkEstimator,
                       X::MatNum; kwargs...)
    phylogeny_features(alg::AbstractPhylogenyFeatureAlgorithm,
                       pl::AbstractClustersEstimator, X::MatNum; kwargs...)

Turn a graph source or a partition source into a square `assets × assets` feature matrix.

This is the kernel of [`PhylogenyPanel`](@ref). Every method returns a matrix of element type `eltype(X)`, which is the element type of the returns. It is never the `Int` or `BitMatrix` that the phylogeny routines give, so [`ChordDist`](@ref) and [`AngularDist`](@ref) keep their BLAS `gemm` path and a `Float32` history gives a `Float32` matrix.

`pl` is an estimator, a [`NetworkEstimator`](@ref) or a [`ClustersEstimator`](@ref). It is never a precomputed [`PhylogenyResult`](@ref) or [`Clusters`](@ref), because an Estimator does not hold a Result. So each call builds the structure from `X`, and the structure follows a cross-validation fold or the subproblem of a meta-optimiser.

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
 4. Read the budget through [`separation_budget`](@ref), and score `d` in the element type `eltype(X)` through [`_proximity_features`](@ref), giving `Z`.

Over a partition source, under any algorithm:

 1. Build the co-membership matrix from `X` through [`phylogeny_matrix`](@ref), and convert it to the element type `eltype(X)`.
 2. Add the identity, which restores the diagonal that [`phylogeny_matrix`](@ref) subtracts, giving `Z`.

# Arguments

  - `alg`: Phylogeny feature algorithm.
  - `pl`: Structure source, a network estimator for a graph or a clustering estimator for a partition.
  - `X`: Asset returns matrix, `observations × assets`.
  - `kwargs...`: Keyword arguments for the phylogeny routines.

# Returns

  - `Z::Matrix`: Square `assets × assets` feature matrix of element type `eltype(X)`.

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
  - `et`: Element type of the result. [`phylogeny_features`](@ref) passes `eltype(X)`, so that [`ChordDist`](@ref) and [`AngularDist`](@ref) keep their BLAS `gemm` path.

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

Name the label axis of a [`TensorPanelField`](@ref) that a producer builds, from the returns data or by position.

The field of a producer has a labelled label axis, and neither a loadings matrix nor a proximity matrix holds names for that axis. The returns data is the one place that holds them. `rd.nx` names the assets of a proximity field, and `rd.nf` names the factors of a loadings field. When the returns data gives no names, or a wrong number of them, the labels are the positions `"1"` to `"n"`.

The function checks the count. A reduced or re-based `L` has fewer columns than the returns data has factors, and the names of the returns data then belong to other columns.

# Algorithm

The method that Julia selects is the algorithm.

 1. `names` is `nothing`: return the positional labels.
 2. `names` is a vector: return it as `String`s when it holds `n` names, and the positional labels otherwise.

# Arguments

  - `names`: The names that the returns data gives the axis, or `nothing`.
  - `n`: Length of the axis to label.

# Returns

  - `labels::Vector{String}`: One label per entry of the label axis.

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
    returns_asset_names(rd::Nothing) -> nothing
    returns_asset_names(rd::AbstractReturnsResult) -> Option{<:VecStr}

Read the asset names that label the label axis of a producer's field, or `nothing`.

The kernel hands a producer the two values that it received, `pr` and `rd`. Only `rd` names the assets, so this function reads `rd`, and returns `nothing` when there is none.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - `rd`: The returns data, or `nothing`.

# Returns

  - `nx::Option{<:VecStr}`: The asset names, or `nothing`.

# Related

  - [`PhylogenyPanel`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`ReturnsResult`](@ref)
"""
function returns_asset_names(::Nothing)
    return nothing
end
function returns_asset_names(rd::AbstractReturnsResult)
    return rd.nx
end
"""
    regression_factor_names(rr::Regression{<:Any, Nothing, <:Any, <:Any},
                            rd::AbstractReturnsResult) -> Option{<:VecStr}
    regression_factor_names(rr::CrossSectionalFactorModel, rd) -> Option{<:VecStr}
    regression_factor_names(rr, rd) -> nothing

Read the factor names that label the loadings axis of a [`RegressionPanel`](@ref), or `nothing`.

The function reads a name wherever one exists for the axis that `pr.rr.L` spans. Otherwise [`panel_axis_labels`](@ref) labels the axis by position.

  - A time-series [`Regression`](@ref) holds no factor names, so the names are the `nf` of the returns data. The function reads them only when the loadings axis is the factor axis of the returns data. That is a `Regression` whose `L` is unset, so that its loadings are the raw `M`, with one column for each factor of the returns data. A reduced or re-based `L` has an axis of its own, and no data names it.
  - A [`CrossSectionalFactorModel`](@ref) names its own factors. The prior derives the raw axis from its Exposure Estimators and stores it as `nf` on the block. When the block holds a family re-basis, [`cs_diagnostic_factor_names`](@ref) maps that list onto the reduced axis, which is the axis that `L` spans. So the names come from the block, and the function does not read the returns data.

# Algorithm

The method that Julia selects is the algorithm. The `Nothing` type parameter of `L` selects the raw `M`. The `swap(L, M)` rule of [`Regression`](@ref) reads the same parameter.

# Arguments

  - `rr`: The regression result that holds the loadings.
  - `rd`: The returns data, or `nothing`.

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
    return cs_diagnostic_factor_names(rr.fcb, rr.nf)
end
function regression_factor_names(::Any, ::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel that holds the factor loadings of the wrapped prior.

`RegressionPanel` takes `pr.rr.L`, the coordinates that a factor model gives each asset, as the values of one tensor Panel Field on the axis `"factor"`. The field is named `"loadings"`, or `name` when it is a string. It needs no estimation of its own. A factor prior computes the loadings already, and refits them in each fold, so the features follow the fold.

# The matrix is `L`, not `M`

`L` is `assets × reduced_dimensions`, and [`DimensionReductionRegression`](@ref) sets it. It holds the coordinates of each asset in the reduced space. `M` holds the loadings on every factor, rebuilt from `L`. `pr.rr.L` always has a value, because [`Regression`](@ref) returns `M` when `L` is unset. So this producer needs no branch, and it works behind the time-series and the cross-sectional regression estimators alike.

Both matrices have one row per asset, so the asset axis of the panel is the asset axis of the returns data, with no transpose. The label axis holds factors or reduced dimensions, not assets, so a loadings panel is never square in the sense of [`features_are_assets`](@ref).

!!! note "What the loadings give out of sample"

    A walk-forward measurement compared the loadings of this producer, under [`FactorPrior`](@ref), with the distances that read the returns alone under the same prior. [`NestedClustered`](@ref) with the loadings had a lower risk on one data set, over 31 folds. It was one pass in about 75 comparisons of feature distances, so chance can explain it. Under [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref) and [`SchurComplementHierarchicalRiskParity`](@ref), the loadings gave no lower risk and no higher Sharpe ratio. [`FeatureDistance`](@ref) states the whole measurement.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegressionPanel(;
        name::Option{<:AbstractString} = nothing
    ) -> RegressionPanel

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:pfname])

# Validation

  - The call supplies a prior result. Raises an [`IsNothingError`](@ref) that names the call site.
  - The wrapped prior holds a regression, see [`assert_prior_regression`](@ref). The order of nesting has no effect, because every wrapping estimator forwards `rr` and the factor block `fpr`. A prior that computes no regression, such as [`EmpiricalPrior`](@ref), raises the error.

# Examples

```jldoctest
julia> RegressionPanel()
RegressionPanel
  name ┴ nothing
```

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`asset_panel`](@ref)
  - [`FeatureDistance`](@ref)
  - [`FactorPrior`](@ref)
  - [`Regression`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`panel_axis_labels`](@ref)
  - [`resolve_panel_field_name`](@ref)
"""
@concrete struct RegressionPanel <: AbstractAssetPanelEstimator
    """
    $(field_dict[:pfname])
    """
    name
    function RegressionPanel(name::Option{<:AbstractString})
        assert_panel_field_name(name)
        return new{typeof(name)}(name)
    end
end
function RegressionPanel(; name::Option{<:AbstractString} = nothing)::RegressionPanel
    return RegressionPanel(name)
end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel that holds a square proximity matrix, graded from a graph or a partition.

In an `assets × assets` proximity matrix, feature `k` of a row reads "is close to asset `k`". So a metric over the rows measures the overlap of two neighbourhoods, which is a standard measure of topological similarity, and it needs almost no new estimation code. `PhylogenyPanel` returns the matrix as one tensor Panel Field on the axis `"asset"`. The field is named `"proximity"`, or `name` when it is a string.

It is the one producer whose label axis is the asset axis, and it needs no flag for that. It refits, so a subproblem gets its own square matrix over its own assets, not a slice of a larger matrix. Where a panel supplied by hand meets a view, [`features_are_assets`](@ref) compares the labels of the field with the asset names of the returns data.

!!! note "What the producer gives out of sample"

    The graph of this producer comes from the correlation of the returns, so the producer derives from the returns-only distances. A walk-forward measurement on two data sets compared it with them, under [`Proximity`](@ref) with [`LinearDecay`](@ref) and with [`ExponentialDecay`](@ref). Under [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref), [`SchurComplementHierarchicalRiskParity`](@ref) and [`NestedClustered`](@ref), it gave no lower risk and no higher Sharpe ratio out of sample on both data sets. [`FeatureDistance`](@ref) states the whole measurement.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyPanel(;
        pl::NwE_ClE = NetworkEstimator(),
        alg::AbstractPhylogenyFeatureAlgorithm = Proximity(),
        name::Option{<:AbstractString} = nothing
    ) -> PhylogenyPanel

Keywords correspond to the struct's fields.

## Validation

  - `isnothing(returns_source(pl))`, else an `ArgumentError`, through [`assert_inner_reader`](@ref).
  - $(val_dict[:pfname])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `pl`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `pl`: Recursively viewed through [`port_opt_view`](@ref).

# Why the diagonal includes self

The diagonal selects between two different algorithms. The table measures a three-node path `1 - 2 - 3` under the default [`ChordDist`](@ref) and the default one-hop budget. The decay is `nothing`, so the diagonal is the only difference between the two columns.

| pair            | zero diagonal | self included |
|:--------------- | -------------:| -------------:|
| `1`-`3`, 2 hops | 0.0           | 0.5           |
| `1`-`2`, 1 hop  | 0.707         | 0.303         |

The default [`LinearDecay`](@ref) grades the path over the same budget, so its distances are `0.632` for `1`-`3` and `0.367` for `1`-`2`. The order of the two pairs is the order of the right-hand column.

With a zero diagonal, the two endpoints, which are not adjacent, have a distance of zero, and each adjacent pair has the largest distance. The metric then compares two rows by their shared neighbours, and ignores whether the two assets are neighbours of each other. That is structural equivalence, a similarity of role, and it is the opposite of proximity.

The diagonal also keeps a subproblem well defined. A view of a spanning tree on a subset of the assets can isolate a vertex. With a zero diagonal, the row of an isolated asset is a zero row, and the zero-vector convention of [`ChordDist`](@ref) puts every isolated asset at distance `0` from every other isolated asset. With self included, each isolated asset is at distance ``1 / \\sqrt{2} \\approx 0.707`` from every other asset, the largest distance between two non-negative rows.

# A clustering source

[`NwE_ClE`](@ref) bounds `pl`, so it is a graph ([`NetworkEstimator`](@ref)) or a partition ([`ClustersEstimator`](@ref)). Both are estimators, so both refit.

A partition holds less information than a graph. Its matrix is the co-membership indicator `P * transpose(P)`, where `P` is the `assets × clusters` membership matrix. Row `i` marks the assets in the cluster of asset `i`, so two rows are equal or disjoint. On seven assets in the clusters `[1, 1, 1, 2, 2, 3, 3]`, the distance matrix under the default [`ChordDist`](@ref) takes two values, `0.0` inside a cluster and `0.707` across clusters, whatever the sizes of the clusters. The raw output of [`phylogeny_matrix`](@ref) is `P * transpose(P) - I`, and it takes three values, `0.0`, `0.5` and `0.707`. In a cluster of two assets, the `- I` leaves each row with a single `1` at the other member, so the two rows are orthogonal, and the distance inside that cluster equals the distance across clusters. This producer adds `I` back, which repairs that case, but the matrix stays coarse.

A clustering of a feature matrix built from a clustering gives back much the same clustering. Use a graph source unless the partition is the thing to measure.

# Provenance

The source is a [`NetworkEstimator`](@ref) and never a precomputed [`PhylogenyResult`](@ref), because an Estimator does not hold a Result. The producer is therefore endogenous. It filters the graph from the correlation of the returns, so it measures the topology that the correlation implies and no structure outside it. It refits on every fold and on every subproblem.

It reads no prior result, so it is the one producer that runs where no prior exists yet, for example in preselection.

`pl` reads the returns matrix that the reader of the panel picked, so it states no source of its own, and the constructor refuses an `x_src` on it that is not `nothing`, see [`assert_inner_reader`](@ref).

# Examples

```jldoctest
julia> PhylogenyPanel()
PhylogenyPanel
    pl ┼ NetworkEstimator
       │      ce ┼ PortfolioOptimisersCovariance
       │         │   ce ┼ Covariance
       │         │      │    me ┼ SimpleExpectedReturns
       │         │      │       │   w ┴ nothing
       │         │      │    ce ┼ GeneralCovariance
       │         │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
       │         │      │       │    w ┴ nothing
       │         │      │   alg ┼ FullMoment()
       │         │      │     w ┴ nothing
       │         │   mp ┼ MatrixProcessing
       │         │      │     pdm ┼ Posdef
       │         │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
       │         │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │         │      │      dn ┼ nothing
       │         │      │      dt ┼ nothing
       │         │      │     alg ┼ nothing
       │         │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
       │      de ┼ Distance
       │         │   power ┼ nothing
       │         │     alg ┴ CanonicalDistance()
       │     alg ┼ KruskalTree
       │         │     args ┼ Tuple{}: ()
       │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
       │     sep ┼ HopCount
       │         │   n ┼ CountOf
       │         │     │   val ┴ Int64: 1
       │   x_src ┴ nothing
   alg ┼ Proximity
       │   decay ┴ LinearDecay()
  name ┴ nothing
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
    """
    $(field_dict[:pfname])
    """
    name
    function PhylogenyPanel(pl::NwE_ClE, alg::AbstractPhylogenyFeatureAlgorithm,
                            name::Option{<:AbstractString})
        assert_inner_reader(pl, :pl, :PhylogenyPanel)
        assert_panel_field_name(name)
        return new{typeof(pl), typeof(alg), typeof(name)}(pl, alg, name)
    end
end
function PhylogenyPanel(; pl::NwE_ClE = NetworkEstimator(),
                        alg::AbstractPhylogenyFeatureAlgorithm = Proximity(),
                        name::Option{<:AbstractString} = nothing)::PhylogenyPanel
    return PhylogenyPanel(pl, alg, name)
end
"""
    asset_panel(ape::RegressionPanel, pr, rd, X) -> AssetPanel
    asset_panel(ape::PhylogenyPanel, pr, rd, X) -> AssetPanel
    asset_panel(ape::WalkProfilePanel, pr, rd, X) -> AssetPanel

Build the static [`AssetPanel`](@ref) of a producer, at the point of use.

The methods of [`RegressionPanel`](@ref) and [`PhylogenyPanel`](@ref) return a panel that holds one [`TensorPanelField`](@ref), because a loadings matrix and a proximity matrix are each one quantity with a labelled second axis. The field takes the resolved name of the producer, see [`resolve_panel_field_name`](@ref). The method of [`WalkProfilePanel`](@ref) returns one [`TensorPanelField`](@ref) for each graph and each single direction, labelled by the nodes of the graph. The labels of the label axis come from the returns data or from the regression block where a name exists, and are positions otherwise. [`panel_axis_labels`](@ref) and [`regression_factor_names`](@ref) state the rule.

A producer runs on the prior and the returns of its own subproblem, so no view reads the panel that it builds, and a fold refits it. Called on its own, on a prior fitted on a point-in-time Asset Panel, a [`RegressionPanel`](@ref) reads the loadings on the Investable Mask of the prior and answers for the full universe. Outside the mask each asset gets a zero row and a false observed mask, so an optimiser can take the panel back. [`expand_investable_loadings`](@ref) states the rule.

# Algorithm

A [`RegressionPanel`](@ref) takes seven steps:

 1. Check that a prior result reached the call, with [`assert_producer_prior`](@ref).
 2. Check that the prior holds a regression, with [`assert_prior_regression`](@ref).
 3. Reduce the prior to its Investable Mask through [`investable_mask`](@ref) and [`port_opt_view`](@ref), giving `prr`, and read its loadings `L`. A prior fitted on a point-in-time Asset Panel writes `NaN` into the loadings of every asset outside its mask, and a Panel Field admits no `NaN`. Inside an optimiser the prior arrives reduced, and the view covers the whole universe.
 4. Count the assets on the mask whose loadings are not all finite, giving `nnf`, and refuse a count above zero. Such an asset has a finite moment and a loadings row that is not finite, which is a defect of the regression.
 5. Label the loadings axis with [`panel_axis_labels`](@ref), from [`regression_factor_names`](@ref).
 6. Expand the loadings back onto the full universe with [`expand_investable_loadings`](@ref), giving `vals` and `omsk`.
 7. Return the panel that holds them as one field on the axis `"factor"`, named `"loadings"` or `ape.name`.

A [`PhylogenyPanel`](@ref) takes three steps:

 1. Grade the structure into an `assets × assets` matrix with [`phylogeny_features`](@ref), giving `Zp`.
 2. Label the label axis with [`panel_axis_labels`](@ref), from [`returns_asset_names`](@ref).
 3. Return the panel that holds `Zp` as one field on the axis `"asset"`, named `"proximity"` or `ape.name`.

A [`WalkProfilePanel`](@ref) takes four steps:

 1. Read the returns data with [`walk_returns`](@ref), from `rd`, or from `pr` when `rd` is `nothing`.
 2. Read the parts of the walk with [`walk_parts_of`](@ref), one for each graph and each single direction, and check that `X` has one column for each asset.
 3. Mark the assets with a zero profile in every field with [`walk_unwalked`](@ref), giving `zall`.
 4. Build the field of each part with [`walk_field`](@ref), and return the panel that holds them.

# Arguments

  - `ape`: The producer.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd]) [`RegressionPanel`](@ref) and [`PhylogenyPanel`](@ref) read it for the axis names alone. [`WalkProfilePanel`](@ref) reads its Exogenous Graphs and its asset names.
  - `X`: Returns matrix of the subproblem, observations × assets.

# Validation

  - A [`RegressionPanel`](@ref) needs a prior result that holds a regression. Raises an [`IsNothingError`](@ref).
  - A [`RegressionPanel`](@ref) needs every loading inside the Investable Mask of the prior to be finite. Raises an [`IsNonFiniteError`](@ref) that counts the assets whose loadings are not finite.
  - A [`WalkProfilePanel`](@ref) needs the returns data, with asset names, a graph in force at the last row of each name of `ng`, and one column of `X` for each asset. Raises an [`IsNothingError`](@ref), an `ArgumentError` or a `DimensionMismatch`. See [`walk_parts_of`](@ref).

# Returns

  - `pnl::AssetPanel`: A static Asset Panel that holds one tensor Panel Field, or one for each part of a walk.

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`RegressionPanel`](@ref)
  - [`PhylogenyPanel`](@ref)
  - [`WalkProfilePanel`](@ref)
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
              IsNonFiniteError("`RegressionPanel` reads the factor loadings `pr.rr.L` as a Panel Field, and $(nnf) of the $(size(L, 1)) assets inside the prior's Investable Mask carry a loading that is not finite, so the field cannot be built over them. An asset outside the mask is written as a zero placeholder row and never reaches this check, so every asset it counts has a finite moment and a loadings row that is not: a defect of the regression, not of the universe.\nGot\npr => $(nameof(typeof(pr)))\nrr => $(nameof(typeof(pr.rr)))\nassets with a non-finite loading => $(nnf)"))
    vals, omsk = expand_investable_loadings(L, imsk)
    return AssetPanel(;
                      pf = [TensorPanelField(;
                                             name = resolve_panel_field_name(ape.name,
                                                                             "loadings"),
                                             axis = "factor",
                                             labels = panel_axis_labels(regression_factor_names(prr.rr,
                                                                                                rd),
                                                                        size(L, 2)),
                                             vals = vals, omsk = omsk,
                                             pmsk = isnothing(omsk) ? nothing : .!omsk)])
end
"""
    expand_investable_loadings(L::MatNum, imsk::Nothing) -> (L, nothing)
    expand_investable_loadings(L::MatNum, imsk::BitVector) -> (vals, omsk)

Write the loadings that a [`RegressionPanel`](@ref) read on the Investable Mask back onto the full asset universe.

A prior fitted on a point-in-time Asset Panel writes `NaN` into the loadings of every asset outside its Investable Mask, and a Panel Field admits no `NaN`. So the producer reads the loadings on the mask, and this function expands them. Each asset outside the mask gets a zero row and a false observed mask. Every uncertainty set that a caller fits on its own on such a prior follows the same rule. The producer marks the row as a placeholder too: its placeholder mask is the negation of the observed mask, because no loading exists there.

A zero row is a placeholder, not a zero feature vector. A [`FeatureDistance`](@ref) reads no placeholder: a direct call refuses such an asset by name. An optimiser hands the producer the prior reduced to the mask, so a fit meets no such row. The selector `"loadings" => :observed` reads the mask as a column. `"loadings"` is the derived name, and the selector names the resolved name of the producer, see [`resolve_panel_field_name`](@ref).

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
                      pf = [TensorPanelField(;
                                             name = resolve_panel_field_name(ape.name,
                                                                             "proximity"),
                                             axis = "asset",
                                             labels = panel_axis_labels(returns_asset_names(rd),
                                                                        size(Zp, 2)),
                                             vals = Zp)])
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that read the edge weights of a walk.

[`WalkProfilePanel`](@ref) holds a member, or `nothing` for the stored weights, in its `wgt` field. The rule runs at the point of use, on the one snapshot that a fit reads, so a user with a graph series builds no second copy of each snapshot.

# Interfaces

To add a rule, subtype `AbstractEdgeWeightRule` and implement the following method.

## `read_edge_weights`

  - `read_edge_weights(wgt::MyEdgeWeightRule, M::MatNum) -> MatNum`: Return the adjacency of the walk with the weights that the rule reads. It keeps the zero of each pair with no edge, and the kind of storage of `M`.

### Arguments

  - `wgt`: The rule.
  - `M`: The adjacency of the walk.

### Returns

  - `M::MatNum`: The adjacency with the weights that the rule reads.

# Related

  - [`UnitEdgeWeights`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`read_edge_weights`](@ref)
"""
abstract type AbstractEdgeWeightRule <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Reads each edge as a weight of one, so the walk counts the paths and ignores their weights.

Under [`WeightShare`](@ref), each step goes to each neighbour at the share one over the number of edges of the node, the simple random walk. It costs one pass over the stored entries of the snapshot that the fit reads.

# Related

  - [`AbstractEdgeWeightRule`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`read_edge_weights`](@ref)
"""
struct UnitEdgeWeights <: AbstractEdgeWeightRule end
"""
    read_edge_weights(wgt::Nothing, M::MatNum) -> MatNum
    read_edge_weights(wgt::UnitEdgeWeights, M::MatNum) -> MatNum

Read the edge weights of the adjacency of a walk under a rule, or keep them under `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: return `M`.
 2. [`UnitEdgeWeights`](@ref): map each entry that is not zero to `one` of its type, and keep each zero. A sparse `M` stays sparse, because the map keeps zero at zero.

# Arguments

  - `wgt`: The rule, or `nothing`.
  - `M`: The adjacency of the walk.

# Returns

  - `M::MatNum`: The adjacency with the weights that the rule reads.

# Related

  - [`AbstractEdgeWeightRule`](@ref)
  - [`UnitEdgeWeights`](@ref)
  - [`walk_operator`](@ref)
"""
function read_edge_weights(::Nothing, M::MatNum)
    return M
end
function read_edge_weights(::UnitEdgeWeights, M::MatNum)
    return map(x -> iszero(x) ? zero(x) : one(x), M)
end
"""
$(DocStringExtensions.TYPEDEF)

Builds an Asset Panel that holds the Walk Profile of each asset over an Exogenous Graph.

The Walk Profile of an asset is the weighted sum of the nodes that the walks of `1` to `n` hops from the asset reach in an [`ExogenousGraph`](@ref), each walk weighted by the decay of its number of hops. A metric over the profiles measures how much two assets share their direct and their indirect dependencies, such as a supplier of a supplier. The producer reads the graph in force at the last row of its window in `rd.G`, so a fold reads the graph in force at its own last row. It returns one tensor Panel Field for each graph and each single direction, `"walk_<dir>_<graph>"` on the axis `"node"`, with each profile scaled to unit length.

The nodes are the label axis, and they can be entities that are not assets. The walk finds each asset among the nodes by its name in `rd.nx`.

A [`GraphUnion`](@ref) in `ng` is one graph of the walk, the union of its parts in force at the last row, built for each single direction of `dir`. Its fields are `"<prefix>_<dir>_" * name`, with the `name` of the union. A vector in `ng` keeps each entry apart, a name or a union, as one Panel Field for each single direction.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{p}_{i} &= \\mathbb{1}_{\\mathrm{self}}\\, w(0)\\, \\boldsymbol{e}_{i} + \\sum_{h=1}^{n} w(h)\\, \\tilde{\\mathbf{M}}^{h} \\boldsymbol{e}_{i}\\,, \\\\
\\hat{\\boldsymbol{p}}_{i} &= \\dfrac{\\boldsymbol{p}_{i}}{\\lVert \\boldsymbol{p}_{i} \\rVert}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}_{i}``: Walk Profile of asset ``i`` over the nodes of the graph.
  - ``\\hat{\\boldsymbol{p}}_{i}``: The profile scaled to unit length, which the Panel Field holds. A zero profile stays zero.
  - $(math_dict[:Mt_walk])
  - ``\\boldsymbol{e}_{i}``: Unit vector of the node of asset ``i``.
  - ``w(h)``: Weight of a walk of ``h`` hops, [`separation_decay`](@ref) of `decay` at the separation ``h`` and the budget ``n``.
  - ``n``: Hop budget, the resolved `sep`.
  - ``\\mathbb{1}_{\\mathrm{self}}``: `1` when `self` is `true`, and `0` otherwise.

The weights are non-negative, so each profile is in the non-negative orthant and each cosine of two profiles is in ``[0,\\,1]``. [`OrthantChordDist`](@ref) is the metric for that domain, and [`WalkDistance`](@ref) pairs it with this producer. Under [`ExponentialDecay`](@ref) at the rate ``\\log(1 / \\alpha)``, ``w(h) = \\alpha^{h}``. That is ``\\alpha^{h - 1}`` times one factor, so the unit profile equals the profile of the sum ``\\sum_{h} \\alpha^{h - 1} \\tilde{\\mathbf{M}}^{h}``.

# The budget

`sep` is a [`HopCount`](@ref), because a matrix power counts edges and a walk has no radius. A [`PathLength`](@ref) fails at dispatch. A [`CountOf`](@ref) is the budget. A [`FractionOf`](@ref) takes its quantile over the hop separations of the pairs of an asset and a node that the walk direction reaches, and rounds it to the nearest `Int`. A rule is called as `n(ape, X, g; dims = 1)`, with this producer in place of a network estimator and the directed graph of the walk as `g`. [`resolve_separation`](@ref) resolves each graph and each single direction on its own.

# The scale of the weights

Under [`WeightShare`](@ref), each `h`-step distribution has a total of at most one, so the walk needs no scaling. Under `nrm = nothing`, the walk reads the stored weights, and a power of a large weight can overflow. So that path keeps each hop vector divided by its largest entry, with the log of the scale apart, and joins the terms relative to the largest scale. It is exact to rounding, it cannot overflow, and an underflow drops only the terms that are smaller than the smallest float relative to the largest term. With `nrm = nothing`, the self term is in the units of the raw weights.

# Assets that the walk cannot read

  - An asset that is not a node of a graph has a placeholder in each field of that graph, and it is unreadable.
  - An asset with a zero profile in one field and a profile that is not zero in another field keeps a zero row in the first field, and it is readable.
  - An asset with a zero profile in every field has a placeholder in every field, and it is unreadable. Under the decays of the library, this is an asset with no edge along the walk directions. `self = true` gives every node a profile.

[`readable_assets`](@ref) finds the unreadable assets from `rd` alone, so a fit drops them at its entry, and a direct call refuses them. The rule reads every field of the producer, and not only the fields that `sel` of a [`FeatureDistance`](@ref) picks.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WalkProfilePanel(;
        ng::Option{<:Union{AbstractString, GraphUnion, AbstractVector}} = nothing,
        sep::HopCount = HopCount(; n = CountOf(2)),
        decay::Option{<:AbstractSeparationDecayAlgorithm} = ExponentialDecay(; rate = log(2)),
        dir::AbstractEdgeDirection = OutEdges(),
        nrm::Option{<:AbstractEdgeNormalisation} = WeightShare(),
        wgt::Option{<:AbstractEdgeWeightRule} = nothing,
        self::Bool = false,
        prefix::Option{<:AbstractString} = nothing
    ) -> WalkProfilePanel

Keywords correspond to the struct's fields. The defaults walk two hops along the out-edges with the weight ``2^{-h}``, on the shares of the weights.

## Validation

  - A `String` in `ng` is not empty, and a vector in `ng` is not empty, holds names and [`GraphUnion`](@ref)s alone, and holds no empty name and no name twice. The name of a union is its `name`. Raises an [`IsEmptyError`](@ref) or an `ArgumentError`. See [`assert_graph_names`](@ref).
  - A `String` in `prefix` is not empty. Raises an [`IsEmptyError`](@ref). See [`assert_panel_field_name`](@ref).

# Examples

```jldoctest
julia> WalkProfilePanel(; ng = \"supply\", dir = BothEdges())
WalkProfilePanel
      ng ┼ String: "supply"
     sep ┼ HopCount
         │   n ┼ CountOf
         │     │   val ┴ Int64: 2
   decay ┼ ExponentialDecay
         │   rate ┴ Float64: 0.6931471805599453
     dir ┼ BothEdges()
     nrm ┼ WeightShare()
     wgt ┼ nothing
    self ┼ Bool: false
  prefix ┴ nothing
```

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`asset_panel`](@ref)
  - [`readable_assets`](@ref)
  - [`WalkDistance`](@ref)
  - [`OrthantChordDist`](@ref)
  - [`ExogenousGraph`](@ref)
  - [`graph_in_force`](@ref)
  - [`HopCount`](@ref)
  - [`separation_decay`](@ref)
  - [`AbstractEdgeDirection`](@ref)
  - [`AbstractEdgeNormalisation`](@ref)
  - [`AbstractEdgeWeightRule`](@ref)
"""
@concrete struct WalkProfilePanel <: AbstractAssetPanelEstimator
    """
    $(field_dict[:walk_ng])
    """
    ng
    """
    $(field_dict[:walk_sep])
    """
    sep
    """
    $(field_dict[:walk_decay])
    """
    decay
    """
    $(field_dict[:walk_dir])
    """
    dir
    """
    $(field_dict[:walk_nrm])
    """
    nrm
    """
    $(field_dict[:walk_wgt])
    """
    wgt
    """
    $(field_dict[:walk_self])
    """
    self
    """
    $(field_dict[:walk_prefix])
    """
    prefix
    function WalkProfilePanel(ng::Option{<:Union{AbstractString, GraphUnion,
                                                 AbstractVector}}, sep::HopCount,
                              decay::Option{<:AbstractSeparationDecayAlgorithm},
                              dir::AbstractEdgeDirection,
                              nrm::Option{<:AbstractEdgeNormalisation},
                              wgt::Option{<:AbstractEdgeWeightRule}, self::Bool,
                              prefix::Option{<:AbstractString})
        assert_graph_names(ng, :ng)
        assert_panel_field_name(prefix)
        return new{typeof(ng), typeof(sep), typeof(decay), typeof(dir), typeof(nrm),
                   typeof(wgt), typeof(self), typeof(prefix)}(ng, sep, decay, dir, nrm, wgt,
                                                              self, prefix)
    end
end
function WalkProfilePanel(;
                          ng::Option{<:Union{AbstractString, GraphUnion, AbstractVector}} = nothing,
                          sep::HopCount = HopCount(; n = CountOf(2)),
                          decay::Option{<:AbstractSeparationDecayAlgorithm} = ExponentialDecay(;
                                                                                               rate = log(2)),
                          dir::AbstractEdgeDirection = OutEdges(),
                          nrm::Option{<:AbstractEdgeNormalisation} = WeightShare(),
                          wgt::Option{<:AbstractEdgeWeightRule} = nothing,
                          self::Bool = false,
                          prefix::Option{<:AbstractString} = nothing)::WalkProfilePanel
    return WalkProfilePanel(ng, sep, decay, dir, nrm, wgt, self, prefix)
end
function graph_requirements(::WalkProfilePanel)
    return (NonNegativeWeights(),)
end
function graph_names(ape::WalkProfilePanel)
    return ape.ng
end
"""
    walk_returns(pr, rd::ReturnsResult) -> ReturnsResult
    walk_returns(pr::ReturnsResult, rd::Nothing) -> ReturnsResult
    walk_returns(pr, rd::Nothing) -> Union{}

Return the returns data that a [`WalkProfilePanel`](@ref) reads, for the graphs and the asset names.

`clusterise(cle, rd)` puts the returns data in the prior slot, so the function reads `pr` when `rd` is `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

# Arguments

  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])

# Validation

  - `rd` or `pr` is a [`ReturnsResult`](@ref). Raises an [`IsNothingError`](@ref).

# Returns

  - `rd::ReturnsResult`: The returns data.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`feature_asset_names`](@ref)
"""
function walk_returns(::Any, rd::ReturnsResult)
    return rd
end
function walk_returns(pr::ReturnsResult, ::Nothing)
    return pr
end
function walk_returns(::Any, ::Nothing)
    return throw(IsNothingError("WalkProfilePanel reads the Exogenous Graphs and the asset names of the returns data, and this call supplied no ReturnsResult. Pass the returns data as `rd`."))
end
"""
    walk_asset_names(rd::ReturnsResult) -> VecStr

Return the asset names that a [`WalkProfilePanel`](@ref) finds among the nodes of a graph.

# Arguments

  - $(arg_dict[:rd])

# Validation

  - `rd.nx` is not `nothing`. Raises an [`IsNothingError`](@ref).

# Returns

  - `nx::VecStr`: The asset names.

# Related

  - [`WalkProfilePanel`](@ref)
"""
function walk_asset_names(rd::ReturnsResult)
    @argcheck(!isnothing(rd.nx),
              IsNothingError("WalkProfilePanel finds each asset among the nodes of an Exogenous Graph by its name, and the returns data names no asset. Give the ReturnsResult the names `nx`."))
    return rd.nx
end
"""
    walk_positions(nodes::VecStr, nx::VecStr) -> Vector{Int}

Return the position of the node of each asset in a graph, or `0` for an asset that is not a node.

# Arguments

  - `nodes`: The names of the nodes of the graph.
  - `nx`: The asset names.

# Returns

  - `ia::Vector{Int}`: One position for each asset.

# Related

  - [`WalkProfilePanel`](@ref)
"""
function walk_positions(nodes::VecStr, nx::VecStr)
    pos = Dict(n => k for (k, n) in enumerate(nodes))
    return [get(pos, s, 0) for s in nx]
end
"""
    walk_operator(ape::WalkProfilePanel, d::AbstractEdgeDirection, At::MatNum) -> MatNum

Return the adjacency that one single direction of a walk multiplies, with the start of each step in its columns.

# Algorithm

 1. Orient `At` along `d` with [`directed_weights`](@ref).
 2. Read its weights under `ape.wgt` with [`read_edge_weights`](@ref).
 3. Normalise them under `ape.nrm` with [`normalise_edge_weights`](@ref), giving `M`.

# Arguments

  - `ape`: The producer.
  - `d`: The single direction.
  - `At`: The adjacency of the graph, with the sources in its columns.

# Returns

  - `M::MatNum`: The adjacency of the walk, ``\\tilde{\\mathbf{M}}``.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`walk_profiles`](@ref)
"""
function walk_operator(ape::WalkProfilePanel, d::AbstractEdgeDirection, At::MatNum)
    return normalise_edge_weights(ape.nrm,
                                  read_edge_weights(ape.wgt, directed_weights(d, At)))
end
"""
    walk_parts_of(ape::WalkProfilePanel, rd::ReturnsResult) -> Vector{Tuple}

Return one tuple for each graph and each single direction that a [`WalkProfilePanel`](@ref) walks.

Each tuple holds the name of the graph, the single direction, the names of the nodes, the adjacency of the walk from [`walk_operator`](@ref), and the node positions of the assets from [`walk_positions`](@ref). [`asset_panel`](@ref) and [`readable_assets`](@ref) read the same tuples, so the two always agree.

# Algorithm

 1. Read the asset names with [`walk_asset_names`](@ref).
 2. For each entry of [`graph_name_list`](@ref) of `ape.ng`, add its tuples with [`walk_entry_parts!`](@ref).

# Arguments

  - `ape`: The producer.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`walk_asset_names`](@ref) and [`walk_entry_parts!`](@ref).

# Returns

  - `parts::Vector{Tuple}`: The parts of the walk.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`walk_field`](@ref)
"""
function walk_parts_of(ape::WalkProfilePanel, rd::ReturnsResult)
    nx = walk_asset_names(rd)
    parts = Tuple{String, AbstractEdgeDirection, Vector{String}, MatNum, Vector{Int}}[]
    for entry in graph_name_list(ape.ng)
        walk_entry_parts!(parts, ape, rd, entry, nx)
    end
    return parts
end
"""
    walk_entry_parts!(parts::AbstractVector, ape::WalkProfilePanel, rd::ReturnsResult, name::Option{<:AbstractString}, nx::VecStr) -> parts
    walk_entry_parts!(parts::AbstractVector, ape::WalkProfilePanel, rd::ReturnsResult, gu::GraphUnion, nx::VecStr) -> parts

Add to `parts` the tuple of each single direction of one entry of the `ng` field of a [`WalkProfilePanel`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. A name, or `nothing` for the only graph: read the graph in force at the last row with [`consumer_graph`](@ref), which checks it against [`graph_requirements`](@ref), one time. Read the name of the graph in `rd.ng`, through [`graph_position`](@ref). Build the tuple of each single direction of [`walk_parts`](@ref) from that graph.
 2. A [`GraphUnion`](@ref): for each single direction of [`walk_parts`](@ref), build the union along that direction with [`consumer_graph`](@ref), and its tuple. The name of the graph is the name of the union, [`graph_entry_label`](@ref).

# Arguments

  - `parts`: The tuples of the walk, which the function extends.
  - `ape`: The producer.
  - $(arg_dict[:rd])
  - `name`, `gu`: The entry.
  - `nx`: The asset names.

# Validation

  - The rules of [`graph_position`](@ref) and [`consumer_graph`](@ref).

# Returns

  - `parts::AbstractVector`: The tuples of the walk.

# Related

  - [`walk_parts_of`](@ref)
  - [`GraphUnion`](@ref)
"""
function walk_entry_parts!(parts::AbstractVector, ape::WalkProfilePanel, rd::ReturnsResult,
                           name::Option{<:AbstractString}, nx::VecStr)
    g = consumer_graph(ape, rd, name)
    gname = rd.ng[graph_position(rd.ng, name)]
    ia = walk_positions(g.nodes, nx)
    for d in walk_parts(ape.dir)
        push!(parts, (gname, d, g.nodes, walk_operator(ape, d, g.At), ia))
    end
    return parts
end
function walk_entry_parts!(parts::AbstractVector, ape::WalkProfilePanel, rd::ReturnsResult,
                           gu::GraphUnion, nx::VecStr)
    gname = graph_entry_label(gu)
    for d in walk_parts(ape.dir)
        g = consumer_graph(ape, rd, gu, d)
        push!(parts,
              (gname, d, g.nodes, walk_operator(ape, d, g.At), walk_positions(g.nodes, nx)))
    end
    return parts
end
"""
    walk_unwalked(ape::WalkProfilePanel, parts::AbstractVector, N::Integer) -> BitVector

Mark the assets whose Walk Profile is zero in every field of a [`WalkProfilePanel`](@ref).

An asset has a profile that is not zero in a field when it is a node of the graph and its column of the adjacency of the walk holds an edge, because the decays of the library weigh a walk of one hop above zero. `self = true` gives every node a profile, so the function then marks no asset. The rule costs one column sum of each adjacency, and it runs no walk.

# Arguments

  - `ape`: The producer.
  - `parts`: The parts of the walk, from [`walk_parts_of`](@ref).
  - `N`: The number of assets.

# Returns

  - `zall::BitVector`: `true` for each asset with a zero profile in every field.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`readable_assets`](@ref)
"""
function walk_unwalked(ape::WalkProfilePanel, parts::AbstractVector, N::Integer)
    walks = falses(N)
    for (_, _, _, M, ia) in parts
        deg = vec(sum(M; dims = 1))
        for i in eachindex(ia)
            walks[i] |= ia[i] > 0 && !iszero(deg[ia[i]])
        end
    end
    return ape.self ? falses(N) : .!walks
end
"""
    walk_graph(M::MatNum) -> Graphs.SimpleDiGraph

Build the directed graph of the steps of a walk, with an edge from node `i` to node `j` at each entry `M[j, i]` that is not zero.

A [`FractionOf`](@ref) budget and a rule read this graph.

# Arguments

  - `M`: The adjacency of the walk, with the start of each step in its columns.

# Returns

  - `g::Graphs.SimpleDiGraph`: The graph of the steps.

# Related

  - [`resolve_separation`](@ref)
  - [`stored_entries`](@ref)
"""
function walk_graph(M::MatNum)
    I, J, _ = stored_entries(M)
    g = Graphs.SimpleDiGraph(size(M, 1))
    for k in eachindex(I, J)
        Graphs.add_edge!(g, J[k], I[k])
    end
    return g
end
function resolve_separation(sep::HopCount{<:CountOf}, ::WalkProfilePanel, ::MatNum,
                            ::MatNum, ::VecInt)::HopCount
    return sep
end
function resolve_separation(sep::HopCount, ape::WalkProfilePanel, X::MatNum, M::MatNum,
                            ia::VecInt)::HopCount
    return resolve_separation(sep, ape, X, walk_graph(M), ia)
end
function resolve_separation(sep::HopCount{<:FractionOf}, ::WalkProfilePanel, ::MatNum,
                            g::Graphs.AbstractGraph, ia::VecInt)::HopCount
    v = Int[]
    for s in ia
        for (t, d) in enumerate(Graphs.gdistances(g, s))
            if t != s && d < typemax(d)
                push!(v, d)
            end
        end
    end
    @argcheck(!isempty(v),
              ArgumentError("a FractionOf hop budget of a WalkProfilePanel takes a quantile of the hops from each asset to each node that the walk reaches, and the walk from the $(length(ia)) assets of this graph reaches no other node."))
    return HopCount(round(Int, Statistics.quantile(v, sep.n.val)))
end
function resolve_separation(sep::HopCount{<:HopCountRule}, ape::WalkProfilePanel, X::MatNum,
                            g::Graphs.AbstractGraph, ::VecInt)::HopCount
    n = sep.n(ape, X, g; dims = 1)
    @argcheck(isa(n, Integer),
              ArgumentError("a hop count rule must return an Integer, because the walk sums the matrix powers 1:n.\nGot $(n)::$(typeof(n)) from $(typeof(sep.n))."))
    return HopCount(n)
end
"""
    walk_profiles(nrm::WeightShare, M::MatNum, ia::VecInt, ws::AbstractVector, self::Bool, T::Type) -> Matrix
    walk_profiles(nrm, M::MatNum, ia::VecInt, ws::AbstractVector, self::Bool, T::Type) -> Matrix

Sum the walks of `1` to `n` hops from the node of each asset, each weighted by its decay, before the scale to unit length.

The column `j` of the result is the profile of the asset at the node `ia[j]`, over every node. The first method is the walk on the shares of [`WeightShare`](@ref). The total of each `h`-step distribution is at most one, so it needs no scaling. The second method serves every other rule and `nothing`, and its weights can have any scale. It keeps each hop vector divided by its largest entry, with the log of its scale apart, and it joins the terms in the frame of the largest scale. The frame drops out at the scale to unit length, so the result is a positive multiple of the profile, exact to rounding, with no overflow.

# Algorithm

Both methods start from the unit vectors `E` of the nodes of the assets, from [`walk_starts`](@ref). The self term is the term of zero hops, with the weight `self * ws[1]`, so it is zero when `self` is `false`.

The plain walk, under [`WeightShare`](@ref):

 1. Weigh `E` by `self * ws[1]`, giving the sum `P`, and set the hop vectors `V` to `E`.
 2. For each hop `h` in `1:n`, multiply `V` by `M` from the left, and add `ws[h + 1] * V` to `P`.

The scaled walk, under any other rule:

 1. Take the largest entry `mM` of `M`, or `1` when `M` holds no edge, and divide `M` by it, giving `Ms`. Keep `log(mM)` as `lM`.
 2. Join `E` to the empty sum `A` and its log scale `La`, at the scale `0` and with the weight `self * ws[1]`, through [`walk_join!`](@ref). Set `V` to `E`, with the log scale `L` of zero for each column.
 3. For each hop `h` in `1:n`, multiply `V` by `Ms` from the left, add `lM` to `L`, rescale each column with [`walk_rescale!`](@ref), and join with the weight `ws[h + 1]`.

# Arguments

  - `nrm`: The normalisation of the walk, which selects the method.
  - `M`: The adjacency of the walk, with the start of each step in its columns and non-negative weights.
  - `ia`: The node positions of the assets that are nodes of the graph.
  - `ws`: The weights `w(0)` to `w(n)`.
  - $(arg_dict[:walk_self])
  - `T`: The element type of the walk.

# Returns

  - `P::Matrix`: The profiles, `nodes × assets`, of element type `T`.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`walk_field`](@ref)
"""
function walk_profiles(::WeightShare, M::MatNum, ia::VecInt, ws::AbstractVector, self::Bool,
                       T::Type)
    V = walk_starts(ia, size(M, 1), T)
    P = (self * ws[1]) * V
    for h in 1:(length(ws) - 1)
        V = M * V
        P .+= ws[h + 1] * V
    end
    return P
end
function walk_profiles(::Any, M::MatNum, ia::VecInt, ws::AbstractVector, self::Bool,
                       T::Type)
    mM = mapreduce(identity, max, M; init = zero(eltype(M)))
    sM = iszero(mM) ? one(mM) : mM
    Ms = M / sM
    lM = log(T(sM))
    V = walk_starts(ia, size(M, 1), T)
    L = zeros(typeof(lM), length(ia))
    A = zeros(T, size(V))
    La = fill(typemin(typeof(lM)), length(ia))
    walk_join!(A, La, V, L, self * ws[1])
    for h in 1:(length(ws) - 1)
        V = Ms * V
        L .+= lM
        walk_rescale!(V, L)
        walk_join!(A, La, V, L, ws[h + 1])
    end
    return A
end
"""
    walk_starts(ia::VecInt, n::Integer, T::Type) -> Matrix

Return the unit vectors of the nodes `ia` among `n` nodes, one column for each asset, of element type `T`.

# Arguments

  - `ia`: The node positions of the assets.
  - `n`: The number of nodes.
  - `T`: The element type.

# Returns

  - `E::Matrix`: `nodes × assets`, with a one at `E[ia[j], j]` and zeros elsewhere.

# Related

  - [`walk_profiles`](@ref)
"""
function walk_starts(ia::VecInt, n::Integer, T::Type)
    E = zeros(T, n, length(ia))
    for (j, k) in enumerate(ia)
        E[k, j] = one(T)
    end
    return E
end
"""
    walk_rescale!(V::AbstractMatrix, L::AbstractVector) -> V

Divide each hop vector, a column of `V`, by its largest entry, and add the log of that entry to its log scale in `L`.

A column of zeros is a walk that reached no node. Its log scale becomes `typemin`, the log of zero, and it stays zero at each later hop.

# Arguments

  - `V`: The hop vectors, `nodes × assets`, with non-negative entries.
  - `L`: The log scale of each column.

# Returns

  - `V::AbstractMatrix`: The rescaled hop vectors, with the largest entry of each column `1` or `0`.

# Related

  - [`walk_profiles`](@ref)
  - [`walk_join!`](@ref)
"""
function walk_rescale!(V::AbstractMatrix, L::AbstractVector)
    for j in axes(V, 2)
        m = maximum(view(V, :, j))
        if iszero(m)
            L[j] = typemin(eltype(L))
        else
            view(V, :, j) ./= m
            L[j] += log(m)
        end
    end
    return V
end
"""
    walk_join!(A::AbstractMatrix, La::AbstractVector, V::AbstractMatrix, L::AbstractVector, w::Number) -> A

Add the term ``w \\exp(L_{j}) \\boldsymbol{v}_{j}`` to each column of the sum ``\\exp(La_{j}) \\boldsymbol{a}_{j}``, and keep the larger of the two scales.

The sum keeps the frame of its largest term, so the smaller term enters with a factor ``\\exp(\\cdot) \\leq 1``, and no factor overflows. A term below the smallest float relative to the frame underflows to zero.

# Algorithm

For each column `j`:

 1. Take the log scale of the term, `lt = L[j] + log(w)`. Skip the term when it is `typemin`, a zero term.
 2. Take the new frame `m = max(La[j], lt)`.
 3. Multiply the sum by `exp(La[j] - m)` and the term by `exp(lt - m)`, add them, and set `La[j]` to `m`. An empty sum has `La[j] == typemin`, so its factor is zero.

# Arguments

  - `A`: The sum, `nodes × assets`.
  - `La`: The log scale of each column of the sum.
  - `V`: The term, `nodes × assets`.
  - `L`: The log scale of each column of the term.
  - `w`: The weight of the term.

# Returns

  - `A::AbstractMatrix`: The sum.

# Related

  - [`walk_profiles`](@ref)
  - [`walk_rescale!`](@ref)
"""
function walk_join!(A::AbstractMatrix, La::AbstractVector, V::AbstractMatrix,
                    L::AbstractVector, w::Number)
    lw = iszero(w) ? typemin(eltype(L)) : log(convert(eltype(A), w))
    for j in axes(A, 2)
        lt = L[j] + lw
        if lt == typemin(lt)
            continue
        end
        m = max(La[j], lt)
        view(A, :, j) .= view(A, :, j) .* exp(La[j] - m) .+ view(V, :, j) .* exp(lt - m)
        La[j] = m
    end
    return A
end
"""
    walk_field(ape::WalkProfilePanel, X::MatNum, gname::AbstractString, d::AbstractEdgeDirection,
               nodes::VecStr, M::MatNum, ia::VecInt, zall::AbstractVector{Bool}) -> TensorPanelField

Build the Panel Field of one graph and one single direction of a [`WalkProfilePanel`](@ref).

# Algorithm

 1. Find the assets that are nodes of the graph, `j`, and their node positions, `ka`.
 2. Resolve the budget with [`resolve_separation`](@ref), giving the count of hops `n`.
 3. Check the decay over `0:n` with [`assert_separation_decay`](@ref), and score the weights `ws`, `w(0)` to `w(n)`, with [`separation_decay`](@ref).
 4. Derive the element type `T` from `eltype(M)` and `eltype(ws)`, with `promote_type`.
 5. Walk with [`walk_profiles`](@ref), giving `P`, and scale each column of `P` to unit length. A zero column stays zero.
 6. Write the profiles into the rows of `vals`, `assets × nodes`, at the rows `j`, with a zero row for every other asset.
 7. Mark a placeholder at each row of an asset that is not a node of the graph, or that `zall` marks, giving `pm`. When `pm` marks a row, the field holds the placeholder mask of the rows and its negation as the observed mask.

# Arguments

  - `ape`: The producer.
  - `X`: Returns matrix of the subproblem, observations × assets, for a rule of the budget.
  - `gname`: The name of the graph.
  - `d`: The single direction.
  - `nodes`: The names of the nodes, the labels of the field.
  - `M`: The adjacency of the walk.
  - `ia`: The node position of each asset, `0` for an asset that is not a node.
  - `zall`: The assets with a zero profile in every field, from [`walk_unwalked`](@ref).

# Validation

  - The rules of [`resolve_separation`](@ref) and [`assert_separation_decay`](@ref).

# Returns

  - `pf::TensorPanelField`: The field `"<prefix>_<dir>_<graph>"` on the axis `"node"`.

# Related

  - [`WalkProfilePanel`](@ref)
  - [`walk_profiles`](@ref)
  - [`resolve_panel_field_name`](@ref)
"""
function walk_field(ape::WalkProfilePanel, X::MatNum, gname::AbstractString,
                    d::AbstractEdgeDirection, nodes::VecStr, M::MatNum, ia::VecInt,
                    zall::AbstractVector{Bool})
    j = findall(>(0), ia)
    ka = ia[j]
    n = resolve_separation(ape.sep, ape, X, M, ka).n.val
    assert_separation_decay(ape.decay, 0:n, n)
    ws = [separation_decay(ape.decay, h, n) for h in 0:n]
    T = promote_type(eltype(M), eltype(ws))
    P = walk_profiles(ape.nrm, M, ka, ws, ape.self, T)
    for c in axes(P, 2)
        s = LinearAlgebra.norm(view(P, :, c))
        if !iszero(s)
            view(P, :, c) ./= s
        end
    end
    vals = zeros(T, length(ia), length(nodes))
    vals[j, :] = transpose(P)
    pm = (ia .== 0) .| zall
    pmsk = any(pm) ? repeat(pm, 1, length(nodes)) : nothing
    return TensorPanelField(;
                            name = string(resolve_panel_field_name(ape.prefix, "walk"), "_",
                                          edge_direction_label(d), "_", gname),
                            axis = "node", labels = nodes, vals = vals,
                            omsk = isnothing(pmsk) ? nothing : .!pmsk, pmsk = pmsk)
end
"""
    readable_assets(ape::AbstractAssetPanelEstimator, rd) -> nothing
    readable_assets(ape::WalkProfilePanel, rd::ReturnsResult) -> BitVector

Mark the assets that the panel of a producer can read, from the returns data alone, or return `nothing` when the producer cannot tell before it fits.

A fit composes this mask with its Investable Mask before [`investable_reduction`](@ref), through [`feature_readable_mask`](@ref). So an unreadable asset departs as a non-investable asset, the reduction announces it, and the fit gives it a zero weight. A direct call of the distance has no entry of a fit, so it refuses the asset through [`assert_feature_readable`](@ref).

The fallback returns `nothing`, for a producer whose unreadable assets the data does not show before the fit, such as [`RegressionPanel`](@ref) and [`PhylogenyPanel`](@ref). A [`WalkProfilePanel`](@ref) knows its unreadable assets from the graphs in force and the asset names, in one column sum of each adjacency.

# Algorithm

The method that Julia selects is the algorithm.

 1. The fallback: return `nothing`.
 2. A [`WalkProfilePanel`](@ref): read the parts of the walk with [`walk_parts_of`](@ref), and mark the assets with a zero profile in every field with [`walk_unwalked`](@ref), giving `zall`. An asset is readable when `zall` does not mark it and it is a node of every graph.

# Arguments

  - `ape`: The producer.
  - $(arg_dict[:rd])

# Validation

  - The rules of [`walk_parts_of`](@ref) for a [`WalkProfilePanel`](@ref).

# Returns

  - `keep::Option{BitVector}`: `true` for each readable asset, or `nothing`.

# Related

  - [`AbstractAssetPanelEstimator`](@ref)
  - [`WalkProfilePanel`](@ref)
  - [`feature_readable_mask`](@ref)
  - [`assert_feature_readable`](@ref)
"""
function readable_assets(::AbstractAssetPanelEstimator, ::Any)
    return nothing
end
function readable_assets(ape::WalkProfilePanel, rd::ReturnsResult)
    parts = walk_parts_of(ape, rd)
    keep = .!walk_unwalked(ape, parts, length(rd.nx))
    for (_, _, _, _, ia) in parts
        keep .&= ia .> 0
    end
    return keep
end
function asset_panel(ape::WalkProfilePanel, pr, rd, X::MatNum)
    r = walk_returns(pr, rd)
    parts = walk_parts_of(ape, r)
    N = length(r.nx)
    @argcheck(size(X, 2) == N,
              DimensionMismatch("WalkProfilePanel reads one column of X for each asset of the returns data, got size(X, 2) = $(size(X, 2)) and $(N) asset names"))
    zall = walk_unwalked(ape, parts, N)
    return AssetPanel(; pf = [walk_field(ape, X, p..., zall) for p in parts])
end
"""
    assert_producer_prior(ape::AbstractAssetPanelEstimator, pr::AbstractPriorResult) -> nothing
    assert_producer_prior(ape::AbstractAssetPanelEstimator, pr) -> Union{}

Check that a producer that reads a prior result received one, and name the call site when it did not.

A producer runs wherever the estimator that holds it runs, and one of those sites has no prior. Preselection fits from the returns alone, before a prior exists. The shortest public call, `clusterise(cle, rd)`, also has none, because it puts the returns data in the `pr` slot. This function turns both into an error that names the site, in place of a field access on `nothing` or a `MethodError`.

# Algorithm

The method that Julia selects is the algorithm. A prior result returns. Anything else, `nothing` and the returns data alike, raises.

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
    return throw(IsNothingError("$(nameof(typeof(ape))) reads the prior result it is handed, and this call supplied none. A pre-prior site supplies none by construction: preselection is fitted from the returns data alone, and `clusterise(cle, rd)` puts the returns data in the prior slot. Two ways forward:\n  1. Build the Asset Panel from data and put it on the `ReturnsResult`, then leave `FeatureDistance.ape` at `nothing`.\n  2. Use a producer that reads no prior, such as `PhylogenyPanel`."))
end

export Proximity, phylogeny_features, RegressionPanel, PhylogenyPanel, WalkProfilePanel,
       UnitEdgeWeights
public AbstractPhylogenyFeatureAlgorithm, AbstractEdgeWeightRule, readable_assets,
       read_edge_weights
