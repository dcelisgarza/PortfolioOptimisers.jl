"""
$(DocStringExtensions.TYPEDEF)

Normalised angular distance metric.

Unlike `Distances.CosineDist` (``1 - \\cos``), the angular distance satisfies the triangle inequality, so it is a true metric and the hierarchies built from it are well defined. It maps ``[-1,\\,1] \\to [1,\\,0]``, so it is bounded, scale-invariant per asset, and admits signed features. Its exact similarity counterpart is [`AngularSimilarity`](@ref), which recovers the cosine from the distance alone.

A zero feature vector has no direction, so the cosine is undefined. By convention two zero vectors are at distance `0` from each other (they are identical) and at distance `1` from every non-zero vector (maximally dissimilar), which keeps ``S = \\cos(\\pi D)`` true on every entry of the matching similarity matrix.

# Mathematical definition

```math
\\begin{align}
d_{i,\\,j} &= \\dfrac{1}{\\pi}\\arccos\\left(\\dfrac{\\boldsymbol{z}_{i} \\cdot \\boldsymbol{z}_{j}}{\\lVert\\boldsymbol{z}_{i}\\rVert \\lVert\\boldsymbol{z}_{j}\\rVert}\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_ij_dist])
  - $(math_dict[:z_i_feature])

# Algorithm

The metric carries two paths, and both are its contract. The elementwise method answers one pair of feature vectors, and `Distances._pairwise!` answers a whole matrix.

The elementwise method, `AngularDist()(a, b)`:

 1. Promote the element types of `a` and `b` with `Float64`, giving `T`.
 2. Take the norms of `a` and `b`, giving `na` and `nb`.
 3. Return `zero(T)` when both norms are zero, and `one(T)` when exactly one of them is. This is the zero-feature-vector convention above.
 4. Divide the dot product of `a` and `b` by `na * nb`, giving the cosine.
 5. Clamp the cosine to ``[-1,\\,1]``, take its ``\\arccos``, and divide by ``\\pi``.

The matrix method, `Distances._pairwise!(::AngularDist, r, a)`. It receives `a` already permuted to columns-as-observations, so a zero *column* of `a` is a zero feature vector:

 1. Delegate the whole matrix to the `Distances.CosineDist` kernel, which writes ``1 - \\cos`` into `r` with one BLAS `gemm` call. That kernel divides by the norm, so a zero column of `a` leaves `NaN` in its row and its column of `r`.
 2. Mark the zero columns of `a`, giving `z`.
 3. Rewrite every entry of `r` in place: the diagonal to `zero(T)`; a pair of zero columns to `zero(T)`; a zero column against a non-zero one to `one(T)`; every other entry to ``\\arccos(1 - r_{i,\\,j}) / \\pi``.

One matrix multiplication replaces ``N^{2}`` scalar calls, and it is the faster path from three assets upward: `1.2` times at ``N = 3``, `2.3` at ``N = 5``, and `4` to `9` times from ``N = 8``. It loses only at ``N = 2``, where the single distance it saves does not pay for the call. So there is one matrix path and nothing to tune.

!!! note "The two paths differ on the diagonal, and the matrix path is the correct one"

    ``\\arccos(1 - r) / \\pi`` is the algebraic identity of the elementwise method, not its floating-point result. Off the diagonal the two paths agree to a few units in the last place. On the diagonal they differ by up to `6.707879276254074e-9`: the cosine of a vector with itself rounds to `0.9999999999999999`, ``\\arccos`` has an infinite derivative at `1`, and a `1e-16` error there becomes a `1e-8` error in the distance. The matrix path writes an exact zero instead.

    `Distances.pairwise` writes an exact zero diagonal, so the matrix entry points — which are the only route [`FeatureDistance`](@ref) takes — never see the residual. Call the metric directly on a pair of identical vectors and it is there. The `"AngularDist gemm path matches the elementwise method"` testset pins the two paths together, and that is why it pins them with a tolerance.

# Related

  - [`AngularSimilarity`](@ref)
  - [`FeatureDistance`](@ref)
  - [`default_similarity`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)

# References

  - $(ref_dict[:vandongen2012])
"""
struct AngularDist <: Distances.Metric end
function (::AngularDist)(a, b)
    T = promote_type(eltype(a), eltype(b), Float64)
    na = LinearAlgebra.norm(a)
    nb = LinearAlgebra.norm(b)
    if iszero(na) && iszero(nb)
        return zero(T)
    elseif iszero(na) || iszero(nb)
        return one(T)
    end
    return acos(clamp(T(LinearAlgebra.dot(a, b) / (na * nb)), -one(T), one(T))) / T(pi)
end
Distances.result_type(::AngularDist, a::Type, b::Type) = promote_type(a, b, Float64)
# The `AngularDist` docstring's `# Algorithm` section states both paths, the measurement that
# separates them on the diagonal, and why only one matrix path exists.
function Distances._pairwise!(::AngularDist, r::AbstractMatrix, a::AbstractMatrix)
    Distances._pairwise!(Distances.CosineDist(), r, a)
    T = eltype(r)
    z = [iszero(LinearAlgebra.norm(view(a, :, j))) for j in axes(a, 2)]
    @inbounds for j in axes(r, 2), i in axes(r, 1)
        r[i, j] = if i == j
            zero(T)
        elseif z[i] && z[j]
            zero(T)
        elseif z[i] || z[j]
            one(T)
        else
            acos(clamp(one(T) - r[i, j], -one(T), one(T))) / T(pi)
        end
    end
    return r
end
function default_similarity(::AngularDist)::AngularSimilarity
    return AngularSimilarity()
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all collapse algorithms.

A collapse algorithm is the aggregator applied along the observation axis of a window of time-varying features. It is consumed through [`AggregateFeatures`](@ref) and [`AggregateDistances`](@ref), which differ in *what* they aggregate, not in *how*.

# Related

  - [`MeanCollapse`](@ref)
  - [`MedianCollapse`](@ref)
  - [`AbstractFeatureCollapseAlgorithm`](@ref)
"""
abstract type AbstractCollapseAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Aggregates along the observation axis with the possibly weighted arithmetic mean.

This is the only collapse algorithm [`AggregateDistances`](@ref) accepts, because a convex combination of metrics is itself a metric. [`AggregateFeatures`](@ref) accepts it too, so it is the one member both consumers share, and the default of both.

# Mathematical definition

```math
\\begin{align}
\\bar{z}_{i,\\,k} &= \\dfrac{\\sum\\limits_{t=1}^{T} w_{t} z_{t,\\,i,\\,k}}{\\sum\\limits_{t=1}^{T} w_{t}}\\,,
\\end{align}
```

Where:

  - $(math_dict[:zbar_ik_feature])
  - $(math_dict[:z_tik_feature])
  - $(math_dict[:w_t_obsweight])
  - $(math_dict[:T])

The weights are non-negative and the denominator normalises them, so the aggregate is a convex combination of the window. That is what makes it a metric when it is applied to distance matrices.

# Algorithm

 1. Reduce the leading observation axis of `Z` with `Statistics.mean`, weighted by `w` when `w` is not `nothing`.
 2. Drop the reduced axis, giving an `assets × features` matrix.

# Related

  - [`AbstractCollapseAlgorithm`](@ref)
  - [`MedianCollapse`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
"""
struct MeanCollapse <: AbstractCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Aggregates along the observation axis with the possibly weighted median, which resists an outlying observation.

Only [`AggregateFeatures`](@ref) accepts it: it aggregates the features and applies the metric afterwards, so the result is a metric. [`AggregateDistances`](@ref) rejects it at construction, because an entrywise median of distance matrices need not satisfy the triangle inequality.

A quantile interpolates, so the aggregate need not be an element of the window. `Statistics.median(v, w)` is the `StatsBase` ``0.5``-quantile rather than an order statistic: on the window `[0, 1, 2, 3]` under the weights `[1, 2, 3, 4]` it returns `11/6`, which lies strictly between the second and the third value. Interpolation is what the quantile *may* do rather than what it always does — the same window under the weights `[4, 3, 2, 1]` returns `1`, an element. The unweighted median of an even window averages the two central values for the same reason.

# Mathematical definition

```math
\\begin{align}
\\bar{z}_{i,\\,k} &= Q_{0.5}\\left(\\left\\{z_{t,\\,i,\\,k}\\right\\}_{t=1}^{T},\\, \\left\\{w_{t}\\right\\}_{t=1}^{T}\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:zbar_ik_feature])
  - $(math_dict[:z_tik_feature])
  - $(math_dict[:w_t_obsweight])
  - $(math_dict[:T])
  - ``Q_{0.5}``: The ``0.5``-quantile of the window under those weights.

# Algorithm

 1. For each asset `j` and each feature `k`, take the observation series `view(Z, :, j, k)`.
 2. Reduce that series with `Statistics.median`, weighted by `w` when `w` is not `nothing`, giving the entry of the collapsed matrix.

# Related

  - [`AbstractCollapseAlgorithm`](@ref)
  - [`MeanCollapse`](@ref)
  - [`AggregateFeatures`](@ref)
"""
struct MedianCollapse <: AbstractCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all feature collapse algorithms.

A feature collapse algorithm reduces a window of time-varying features, `observations × assets × features`, to a single `assets × assets` distance matrix. It is the [`FeatureDistance`](@ref) `alg` field, and is inert when the feature matrix is 2-D — a static feature matrix has no observation axis to collapse. At `observations == 1` every algorithm in the family agrees exactly.

# Related

  - [`LastObservation`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
  - [`StackObservations`](@ref)
  - [`FeatureDistance`](@ref)
"""
abstract type AbstractFeatureCollapseAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Discards the window and measures the last observation's feature matrix alone.

The cheapest member of the family and its default, because it is the only one whose result depends on no aggregation choice.

# Algorithm

 1. Take the last slice of the observation axis, `view(Z, size(Z, 1), :, :)`, giving an `assets × features` matrix.
 2. Apply the metric to that matrix once.

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
  - [`StackObservations`](@ref)
"""
struct LastObservation <: AbstractFeatureCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Collapses the window to one `assets × features` matrix, then applies the metric once.

Each feature is aggregated along the observation axis. The metric runs *after* the aggregation, so the result is a metric for both [`MeanCollapse`](@ref) and [`MedianCollapse`](@ref), and this is the only consumer that takes the median.

# Algorithm

 1. Resolve `w` against `Z` with [`collapse_weights`](@ref), giving a weight vector of one entry per observation, or `nothing`.
 2. Collapse the observation axis of `Z` with `alg`, giving one `assets × features` matrix.
 3. Apply the metric to that matrix once, and apply the zero-feature-vector convention to the result.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AggregateFeatures(;
        w::Option{<:ObsWeights} = nothing,
        alg::AbstractCollapseAlgorithm = MeanCollapse()
    ) -> AggregateFeatures

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> AggregateFeatures()
AggregateFeatures
    w ┼ nothing
  alg ┴ MeanCollapse()
```

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractCollapseAlgorithm`](@ref)
  - [`AggregateDistances`](@ref)
  - [`FeatureDistance`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct AggregateFeatures <: AbstractFeatureCollapseAlgorithm
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:calg])
    """
    alg
    function AggregateFeatures(w::Option{<:ObsWeights},
                               alg::AbstractCollapseAlgorithm)::AggregateFeatures
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(alg)}(w, alg)
    end
end
function AggregateFeatures(; w::Option{<:ObsWeights} = nothing,
                           alg::AbstractCollapseAlgorithm = MeanCollapse())::AggregateFeatures
    return AggregateFeatures(w, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures every observation, then aggregates the resulting distance matrices.

Produces one distance matrix per observation and combines them into a single `assets × assets` matrix. Costs `observations` metric evaluations against [`AggregateFeatures`](@ref)'s one, and accumulates into a single buffer rather than materialising the whole stack.

Only [`MeanCollapse`](@ref) is accepted: a convex combination of metrics is a metric, an entrywise median of them is not. Because the metric is applied *before* the aggregation, the zero-feature convention is applied per observation — an asset that is zero at some observations but not others is treated as zero only in the observations where it is.

# Algorithm

 1. Resolve `w` against `Z` with [`collapse_weights`](@ref), giving a weight vector of one entry per observation, or `nothing`.
 2. Allocate the accumulator `D` and the single per-observation buffer `Dt`, both `assets × assets`, and set the weight total `sw` to zero.
 3. For each observation `t`: measure that slice of `Z` into `Dt`; apply the zero-feature-vector convention to `Dt`; read the observation's weight `wt`, which is `one(T)` when `w` is `nothing`; add `wt .* Dt` to `D`; and add `wt` to `sw`.
 4. Divide `D` by `sw`, giving the convex combination of the per-observation distance matrices.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AggregateDistances(;
        w::Option{<:ObsWeights} = nothing,
        alg::AbstractCollapseAlgorithm = MeanCollapse()
    ) -> AggregateDistances

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - `alg` is not a [`MedianCollapse`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> AggregateDistances()
AggregateDistances
    w ┼ nothing
  alg ┴ MeanCollapse()

julia> AggregateDistances(; alg = MedianCollapse())
ERROR: ArgumentError: alg must not be a MedianCollapse: an entrywise median of distance matrices need not satisfy the triangle inequality, so the result would not be a metric. Use MeanCollapse, or aggregate the features instead with AggregateFeatures.
[...]
```

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractCollapseAlgorithm`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`FeatureDistance`](@ref)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
"""
@propagatable @concrete struct AggregateDistances <: AbstractFeatureCollapseAlgorithm
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:calg])
    """
    alg
    function AggregateDistances(w::Option{<:ObsWeights},
                                alg::AbstractCollapseAlgorithm)::AggregateDistances
        assert_nonempty_nonneg_finite_val(w, :w)
        @argcheck(!isa(alg, MedianCollapse),
                  ArgumentError("alg must not be a MedianCollapse: an entrywise median of distance matrices need not satisfy the triangle inequality, so the result would not be a metric. Use MeanCollapse, or aggregate the features instead with AggregateFeatures."))
        return new{typeof(w), typeof(alg)}(w, alg)
    end
end
function AggregateDistances(; w::Option{<:ObsWeights} = nothing,
                            alg::AbstractCollapseAlgorithm = MeanCollapse())::AggregateDistances
    return AggregateDistances(w, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Concatenates the window into one long feature vector per asset, so nothing is averaged away.

Turns `observations × assets × features` into an `assets × (observations · features)` matrix along the feature axis, and applies the metric once. Two assets are close only when their whole trajectories agree — which is also why the result is dominated by whichever observations carry the largest magnitudes, and why heterogeneous features should be standardised before it is used.

Equals none of the other members of the family in general, but agrees with all of them when `observations == 1`.

# Algorithm

 1. Permute `Z` so the asset axis leads: `(2, 1, 3)` at `dims = 1`, and `(3, 1, 2)` at `dims = 2`.
 2. Reshape the permuted array to `assets × (observations · features)`, giving one long feature vector per asset.
 3. Apply the metric to that matrix once, along its first axis.

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`LastObservation`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
"""
struct StackObservations <: AbstractFeatureCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns a feature matrix into a distance matrix, by applying a metric to the rows of that matrix.

A feature matrix describes assets by their exposures, memberships, loadings or adjacencies rather than by their returns. This estimator is a peer of [`Distance`](@ref) and [`DistanceDistance`](@ref): unlike them, it never consults a correlation matrix, so it is usable where returns are uninformative or unavailable.

Any `Distances.SemiMetric` is accepted, including a user-defined one, and every metric yields a similarity, so no combination throws **on this path**. The remarks below are about the metric a caller chooses, not about this type.

!!! warning "A metric is not automatically in the similarity's domain"

    A metric returning a distance above `1` gives similarities outside ``[-1,\\,1]`` under the default [`ComplementSimilarity`](@ref), which [`plot_clusters`](@ref) silently clips. The threshold is `1`, not "the metric is unbounded" — `Distances.CosineDist` and `Distances.CorrDist` are bounded by `2` and cross it routinely.

    That claim is scoped to this path. Handing this estimator to a [`NetworkEstimator`](@ref), [`DBHT`](@ref) or [`LoGo`](@ref) as their `de` puts the resulting distance matrix on the PMFG path, where **their own** similarity field applies rather than `sim`, and where [`assert_similarity_domain`](@ref) refuses a distance above `1` under [`ComplementSimilarity`](@ref) and a non-finite one under [`MaximumDistanceSimilarity`](@ref).

Every metric other than [`AngularDist`](@ref) and `Distances.CorrDist` is scale-sensitive, and even [`AngularDist`](@ref) is invariant to scaling an asset's feature vector but not to scaling a feature across assets. Heterogeneous features should be standardised before use. `Distances.CorrDist` is `NaN` against any constant feature vector, hence unusable with a single feature.

`Distances.Jaccard` is the general non-negative-real (Ruzicka) form, not the binary-set Jaccard, and returns values up to `2` on signed input *without erroring*. It, `Distances.BrayCurtis` and `Distances.ChiSqDist` therefore require a non-negative feature matrix, which [`assert_metric_domain`](@ref) checks in the kernel rather than at construction, because the feature matrix is not known here.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= m\\left(\\boldsymbol{z}_{i},\\, \\boldsymbol{z}_{j}\\right)\\\\
S_{i,\\,j} &= \\sigma\\left(D_{i,\\,j}\\right)\\,,
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.
  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - $(math_dict[:z_i_feature])
  - ``m``: Distance metric, `metric`.
  - ``\\sigma``: Similarity transformation, `sim`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeatureDistance(;
        metric::Distances.SemiMetric = AngularDist(),
        alg::AbstractFeatureCollapseAlgorithm = LastObservation(),
        sim::AbstractSimilarityMatrixAlgorithm = default_similarity(metric)
    ) -> FeatureDistance

Keywords correspond to the struct's fields.

## Validation

  - `sim` is defaulted from `metric` via [`default_similarity`](@ref), so the resolved value is visible on the printed object rather than hidden inside the distance kernel.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> FeatureDistance()
FeatureDistance
  metric ┼ AngularDist: AngularDist()
     alg ┼ LastObservation()
     sim ┴ AngularSimilarity()

julia> FeatureDistance(; metric = PortfolioOptimisers.Distances.CosineDist())
FeatureDistance
  metric ┼ Distances.CosineDist: Distances.CosineDist()
     alg ┼ LastObservation()
     sim ┴ ComplementSimilarity()
```

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`AngularDist`](@ref)
  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`default_similarity`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`assert_metric_domain`](@ref): the non-negativity check that the three restricted metrics take in the kernel.
  - [`DBHT`](@ref): carries a `sim` field of its own, deliberately named alike — same type, same job. When both are set DBHT's wins, because [`clusterise`](@ref) overwrites the similarity matrix immediately after [`cor_and_dist`](@ref) returns.
  - [`factory`](@ref)
"""
@propagatable @concrete struct FeatureDistance <: AbstractDistanceEstimator
    """
    $(field_dict[:fdmetric])
    """
    metric
    """
    $(field_dict[:fcalg])
    """
    @fprop alg
    """
    $(field_dict[:fdsim])
    """
    sim
    function FeatureDistance(metric::Distances.SemiMetric,
                             alg::AbstractFeatureCollapseAlgorithm,
                             sim::AbstractSimilarityMatrixAlgorithm)::FeatureDistance
        return new{typeof(metric), typeof(alg), typeof(sim)}(metric, alg, sim)
    end
end
function FeatureDistance(; metric::Distances.SemiMetric = AngularDist(),
                         alg::AbstractFeatureCollapseAlgorithm = LastObservation(),
                         sim::AbstractSimilarityMatrixAlgorithm = default_similarity(metric))::FeatureDistance
    return FeatureDistance(metric, alg, sim)
end
"""
    assert_metric_domain(metric::Distances.SemiMetric, Z::ArrNum, sym::Symbol = :Z)

Assert that `Z` lies in `metric`'s domain. The fallback is a no-op: most metrics accept any finite real input, and a blanket non-negativity check would reject signed factor loadings and the [`FeatureDistance`](@ref) default metric alike.

`Distances.Jaccard` (the Ruzicka form), `Distances.BrayCurtis` and `Distances.ChiSqDist` are the exceptions, all defined only on non-negative reals. The check matters most for `Distances.Jaccard`, which fails *silently*: it returns values up to `2` on signed input, with no error, straight into a clustering routine.

# Algorithm

 1. Select the method by the type of `metric`. The three metrics above own one method between them; every other metric reaches the `Distances.SemiMetric` method, which is a no-op and returns immediately.
 2. On that method, check `Z` for non-negativity with [`assert_nonneg`](@ref), which raises a `DomainError` naming `sym` when an entry is negative.

# Arguments

  - `metric`: Distance metric whose domain `Z` must lie in.
  - $(arg_dict[:Z])
  - `sym::Symbol = :Z`: Name that the error message gives to `Z`.

# Validation

  - Under `Distances.Jaccard`, `Distances.BrayCurtis` and `Distances.ChiSqDist`: `all(x -> x >= 0, Z)`.

# Returns

  - `nothing`.

# Related

  - [`FeatureDistance`](@ref)
  - [`assert_nonneg`](@ref)
  - [`assert_feature_matrix`](@ref)
"""
function assert_metric_domain(::Distances.SemiMetric, ::ArrNum, ::Symbol = :Z)::Nothing
    return nothing
end
function assert_metric_domain(::Union{<:Distances.Jaccard, <:Distances.BrayCurtis,
                                      <:Distances.ChiSqDist}, Z::ArrNum,
                              sym::Symbol = :Z)::Nothing
    assert_nonneg(Z, sym)
    return nothing
end
"""
    assert_feature_matrix(de::FeatureDistance, Z::ArrNum, dims::Integer)

Validate a feature matrix at the [`distance`](@ref)/[`cor_and_dist`](@ref) entry point: `dims` selects a valid axis, `Z` is non-empty, every entry is finite, and `Z` lies in the metric's domain.

Non-finite entries are rejected because no metric produces a usable distance from them — the Minkowski family gives `Inf` and the cosine family gives `NaN` — and neither can be clustered. Structurally degenerate inputs that a metric *can* handle are admitted: zero feature vectors are given a documented convention (see [`AngularDist`](@ref)), and duplicate or constant features are legitimate.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. Check that `Z` is non-empty with [`assert_nonempty`](@ref).
 3. Check that every entry of `Z` is finite with [`assert_all_finite`](@ref).
 4. Check that `Z` lies in `de.metric`'s domain with [`assert_metric_domain`](@ref).

# Arguments

  - `de`: Feature distance estimator, read for its `metric`.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - `!isempty(Z)`.
  - `all(isfinite, Z)`.
  - `Z` lies in `de.metric`'s domain (see [`assert_metric_domain`](@ref)).

# Returns

  - `nothing`.

# Related

  - [`FeatureDistance`](@ref)
  - [`assert_metric_domain`](@ref)
  - [`assert_dims`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_all_finite`](@ref)
"""
function assert_feature_matrix(de::FeatureDistance, Z::ArrNum, dims::Integer)::Nothing
    assert_dims(dims)
    assert_nonempty(Z, :Z)
    assert_all_finite(Z, :Z)
    assert_metric_domain(de.metric, Z)
    return nothing
end
"""
    zero_feature_vectors(Z::MatNum, dims::Integer)

Boolean mask of the assets whose feature vector is entirely zero, in the layout declared by `dims`.

# Algorithm

 1. Select the asset axis from `dims`: the rows of `Z` at `dims = 1`, and its columns at `dims = 2`.
 2. Test each asset's feature vector with `all(iszero, ...)`, giving one entry of the mask per asset.

# Arguments

  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - `z::Vector{Bool}`: Mask, one entry per asset, true where that asset's feature vector is entirely zero.

# Related

  - [`patch_zero_feature_vectors!`](@ref)
  - [`FeatureDistance`](@ref)
"""
function zero_feature_vectors(Z::MatNum, dims::Integer)
    return if dims == 1
        [all(iszero, view(Z, i, :)) for i in axes(Z, 1)]
    else
        [all(iszero, view(Z, :, i)) for i in axes(Z, 2)]
    end
end
"""
    patch_zero_feature_vectors!(D::MatNum, Z::MatNum, dims::Integer)

Apply the zero-feature-vector convention to `D` in place: two zero vectors are at distance `0`, a zero vector and a non-zero one at distance `1`.

Only entries the metric left as `NaN` are rewritten. A zero feature vector is structurally valid input, so construction-time validation cannot catch it, but it is undefined for the metrics normalised by a norm — the cosine family gives `NaN` against anything, and `Distances.Jaccard`/`Distances.BrayCurtis` give `NaN` between two zero vectors. It is perfectly well defined for the Minkowski family, which places it at the origin; restricting the patch to `NaN` entries fixes the former without corrupting the latter.

The convention is the one that keeps ``S = \\cos(\\pi D)`` true on every entry, so [`AngularSimilarity`](@ref) yields `+1` between two zero vectors and `-1` against a non-zero one, with a unit diagonal. `Distances.pairwise` always writes an exact zero diagonal, so self-distance needs no patching.

# Algorithm

 1. Build the zero mask `z` of `Z` with [`zero_feature_vectors`](@ref).
 2. Return `D` unchanged when no asset is masked, which is the common case.
 3. Otherwise visit every off-diagonal entry of `D` and rewrite it only when the metric left it as `NaN` and at least one of its two assets is masked: to `zero(T)` when both are masked, and to `one(T)` when exactly one is.

# Arguments

  - `D`: Distance matrix `assets × assets`, rewritten in place.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`zero_feature_vectors`](@ref)
  - [`AngularDist`](@ref)
  - [`FeatureDistance`](@ref)
"""
function patch_zero_feature_vectors!(D::MatNum, Z::MatNum, dims::Integer)
    z = zero_feature_vectors(Z, dims)
    if !any(z)
        return D
    end
    T = eltype(D)
    @inbounds for j in axes(D, 2), i in axes(D, 1)
        if i != j && isnan(D[i, j]) && (z[i] || z[j])
            D[i, j] = if z[i] && z[j]
                zero(T)
            else
                one(T)
            end
        end
    end
    return D
end
"""
    feature_distance(metric::Distances.SemiMetric, Z::MatNum, dims::Integer)

Turn a 2-D feature matrix into a distance matrix. This is the shared kernel behind every [`FeatureDistance`](@ref) entry point: the collapse algorithms differ only in the matrix they hand it, except for [`AggregateDistances`](@ref), which calls it once per observation and aggregates the results.

# Algorithm

 1. Apply `metric` to every pair of assets of `Z` with `Distances.pairwise`, along the axis `dims` names, giving `D`.
 2. Apply the zero-feature-vector convention to `D` in place with [`patch_zero_feature_vectors!`](@ref).

# Arguments

  - `metric`: Distance metric applied to the assets of `Z`.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`FeatureDistance`](@ref)
  - [`patch_zero_feature_vectors!`](@ref)
  - [`distance`](@ref)
"""
function feature_distance(metric::Distances.SemiMetric, Z::MatNum, dims::Integer)
    D = Distances.pairwise(metric, Z; dims = dims)
    return patch_zero_feature_vectors!(D, Z, dims)
end
"""
    collapse_features(alg::AbstractCollapseAlgorithm, Z::Arr3Num, w::Option{<:VecNum})

Aggregate a window of time-varying features along its leading observation axis, returning a matrix with the two trailing axes of `Z` unchanged. Used by [`AggregateFeatures`](@ref).

The element type of the result follows the aggregate, not the window: the mean and the median of a window of integers are both fractional, and a window with an even observation count has a fractional median even when every value in it is whole. Both routes therefore build their result from the values they compute.

# Algorithm

The [`MeanCollapse`](@ref) route:

 1. Reduce the leading axis of `Z` with `Statistics.mean`, weighted by `w` when `w` is not `nothing`.
 2. Drop the reduced axis.

The [`MedianCollapse`](@ref) route:

 1. For each asset `j` and each feature `k`, take the observation series `view(Z, :, j, k)`.
 2. Reduce that series with `Statistics.median`, weighted by `w` when `w` is not `nothing`, giving one entry of the result.

# Arguments

  - `alg`: Collapse algorithm, the aggregator applied along the observation axis.
  - $(arg_dict[:Z])
  - `w`: Resolved observation weights, one entry per observation, or `nothing` for an unweighted collapse.

# Returns

  - `Zc::Matrix{<:Number}`: Collapsed feature matrix, the two trailing axes of `Z` unchanged.

# Related

  - [`MeanCollapse`](@ref)
  - [`MedianCollapse`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`collapse_weights`](@ref)
"""
function collapse_features(::MeanCollapse, Z::Arr3Num, ::Nothing)
    return dropdims(Statistics.mean(Z; dims = 1); dims = 1)
end
function collapse_features(::MeanCollapse, Z::Arr3Num, w::VecNum)
    return dropdims(Statistics.mean(Z, w; dims = 1); dims = 1)
end
function collapse_features(::MedianCollapse, Z::Arr3Num, ::Nothing)
    return [Statistics.median(view(Z, :, j, k)) for j in axes(Z, 2), k in axes(Z, 3)]
end
function collapse_features(::MedianCollapse, Z::Arr3Num, w::VecNum)
    return [Statistics.median(view(Z, :, j, k), w) for j in axes(Z, 2), k in axes(Z, 3)]
end
"""
    stack_observations(Z::Arr3Num, dims::Integer)

Reshape a window of time-varying features into an `assets × (observations · features)` matrix, whose rows are the assets whichever trailing axis `dims` says they occupy.

# Algorithm

 1. Permute `Z` so the asset axis leads: `(2, 1, 3)` at `dims = 1`, and `(3, 1, 2)` at `dims = 2`. Both leave the observation axis second and the feature axis third.
 2. Reshape the permuted array to `assets × (observations · features)`, giving one long feature vector per asset.

# Arguments

  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - `Za::Matrix{<:Number}`: Stacked feature matrix, `assets × (observations · features)`.

# Related

  - [`StackObservations`](@ref)
  - [`FeatureDistance`](@ref)
"""
function stack_observations(Z::Arr3Num, dims::Integer)
    Za = dims == 1 ? permutedims(Z, (2, 1, 3)) : permutedims(Z, (3, 1, 2))
    return reshape(Za, size(Za, 1), size(Za, 2) * size(Za, 3))
end
"""
    collapse_weights(w::Option{<:ObsWeights}, Z::Arr3Num)

Resolve the observation weights of a collapse algorithm against a window of time-varying features.

`Z` is matricised to `observations × (assets · features)` first, because [`get_observation_weights`](@ref)'s documented interface is `VecNum`/`MatNum` and a raw 3-D array matches neither — a user's correct `MatNum` method would otherwise never fire. There is no caller-side `nothing` guard: [`get_observation_weights`](@ref) raises [`ObservationWeightsError`](@ref) itself when a [`DynamicAbstractWeights`](@ref) cannot resolve, so `nothing` here means only that no weights were requested (ADR 0043).

Cross-fold weighting requires a [`DynamicAbstractWeights`](@ref). It resolves against the `Z` it is handed, so it is fold-local and correct automatically. A *static* `AbstractWeights` is fixed at construction and outlives the fold: a longer one used to be read positionally by [`AggregateDistances`](@ref), giving the *oldest* weights to the *newest* observations with no bounds error, and a shorter one gave a bare `BoundsError`. The length check makes both loud.

# Algorithm

 1. Reshape `Z` to `observations × (assets · features)`, so the observation axis leads a matrix.
 2. Resolve `w` against that matrix with [`get_observation_weights`](@ref), along its first axis.
 3. Check the resolved length against the observation count of `Z`, unless the resolution gave `nothing`.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:Z])

# Validation

  - `length(w) == size(Z, 1)` once resolved. Raises a `DimensionMismatch` naming both lengths.

# Returns

  - `w::Option{<:VecNum}`: Resolved observation weights, one entry per observation, or `nothing` when no weights were requested.

# Related

  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
  - [`get_observation_weights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
"""
function collapse_weights(w::Option{<:ObsWeights}, Z::Arr3Num)
    w = get_observation_weights(w, reshape(Z, size(Z, 1), size(Z, 2) * size(Z, 3));
                                dims = 1)
    if !isnothing(w)
        @argcheck(length(w) == size(Z, 1),
                  DimensionMismatch("length(w) == size(Z, 1) must hold. Got\nlength(w) => $(length(w))\nsize(Z, 1) => $(size(Z, 1)).\nCross-fold weighting requires a DynamicAbstractWeights, which resolves against the feature window it is given."))
    end
    return w
end
"""
    feature_distance(de::FeatureDistance, Z::Arr3Num, dims::Integer)

Turn a window of time-varying features into a distance matrix, by the collapse algorithm in `de.alg`.

# Algorithm

The type of `de.alg` selects one of four methods. Each is stated on its own type, and the branch is:

 1. [`LastObservation`](@ref): hand the last slice of the observation axis to the 2-D kernel.
 2. [`StackObservations`](@ref): hand the stacked matrix from [`stack_observations`](@ref) to the 2-D kernel, along its first axis.
 3. [`AggregateFeatures`](@ref): resolve the weights with [`collapse_weights`](@ref), collapse the window with [`collapse_features`](@ref), and hand the collapsed matrix to the 2-D kernel.
 4. [`AggregateDistances`](@ref): resolve the weights with [`collapse_weights`](@ref), then accumulate one weighted distance matrix per observation and divide by the weight total.

# Arguments

  - `de`: Feature distance estimator, read for its `metric` and its `alg`.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`FeatureDistance`](@ref)
  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`collapse_weights`](@ref)
  - [`collapse_features`](@ref)
  - [`stack_observations`](@ref)
  - [`distance`](@ref)
"""
function feature_distance(de::FeatureDistance{<:Any, <:LastObservation}, Z::Arr3Num,
                          dims::Integer)
    return feature_distance(de.metric, view(Z, size(Z, 1), :, :), dims)
end
function feature_distance(de::FeatureDistance{<:Any, <:StackObservations}, Z::Arr3Num,
                          dims::Integer)
    return feature_distance(de.metric, stack_observations(Z, dims), 1)
end
function feature_distance(de::FeatureDistance{<:Any, <:AggregateFeatures}, Z::Arr3Num,
                          dims::Integer)
    w = collapse_weights(de.alg.w, Z)
    return feature_distance(de.metric, collapse_features(de.alg.alg, Z, w), dims)
end
function feature_distance(de::FeatureDistance{<:Any, <:AggregateDistances}, Z::Arr3Num,
                          dims::Integer)
    w = collapse_weights(de.alg.w, Z)
    metric = de.metric
    T = Distances.result_type(metric, eltype(Z), eltype(Z))
    N = size(Z, dims == 1 ? 2 : 3)
    D = zeros(T, N, N)
    Dt = Matrix{T}(undef, N, N)
    sw = zero(T)
    @inbounds for t in axes(Z, 1)
        Zt = view(Z, t, :, :)
        Distances.pairwise!(metric, Dt, Zt; dims = dims)
        patch_zero_feature_vectors!(Dt, Zt, dims)
        wt = isnothing(w) ? one(T) : T(w[t])
        D .+= wt .* Dt
        sw += wt
    end
    return D ./= sw
end
"""
    distance(de::FeatureDistance, Z::MatNum; dims::Int = 1, kwargs...)
    distance(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, kwargs...)

Compute the distance matrix from a feature matrix.

The 2-D method never consults `de.alg`: a static feature matrix has no observation axis to collapse, so the collapse algorithm is inert rather than an error. The 3-D method dispatches on it. Assets whose feature vector is entirely zero are given the convention documented in [`patch_zero_feature_vectors!`](@ref).

# Algorithm

 1. Validate `Z` and `dims` with [`assert_feature_matrix`](@ref).
 2. On the 2-D method, hand `de.metric` and `Z` to the kernel.
 3. On the 3-D method, hand `de` and `Z` to the collapse dispatcher, which selects the branch that `de.alg` names.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - $(val_dict[:dims])
  - `!isempty(Z)`.
  - `all(isfinite, Z)`.
  - `Z` lies in `de.metric`'s domain (see [`assert_metric_domain`](@ref)).

# Returns

  - $(ret_dict[:Ddist])

# Examples

```jldoctest
julia> Z = [1.0 0.0; 0.0 1.0; 1.0 1.0];

julia> distance(FeatureDistance(), Z)
3×3 Matrix{Float64}:
 0.0   0.5   0.25
 0.5   0.0   0.25
 0.25  0.25  0.0
```

# Related

  - [`FeatureDistance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`AbstractFeatureCollapseAlgorithm`](@ref)
"""
function distance(de::FeatureDistance, Z::MatNum; dims::Int = 1, kwargs...)
    assert_feature_matrix(de, Z, dims)
    return feature_distance(de.metric, Z, dims)
end
function distance(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, kwargs...)
    assert_feature_matrix(de, Z, dims)
    return feature_distance(de, Z, dims)
end
"""
    cor_and_dist(de::FeatureDistance, Z::MatNum; dims::Int = 1, kwargs...)
    cor_and_dist(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, kwargs...)

Compute the similarity and distance matrices from a feature matrix.

The similarity shares the distance's provenance: it is `distance_to_similarity(de.sim; D = D)`, derived from the distance matrix this call just produced, so `S` and `D` are two views of one measurement rather than two independent estimates. Deriving it from the aggregated distance is also what keeps the zero-feature-vector convention consistent under [`AggregateDistances`](@ref), since ``\\mathrm{mean}(\\cos(\\pi D_{t})) \\neq \\cos(\\pi\\,\\mathrm{mean}(D_{t}))``.

# Algorithm

 1. Compute the distance matrix `D` with [`distance`](@ref), which validates `Z` and `dims` on the way.
 2. Transform `D` with [`distance_to_similarity`](@ref) under `de.sim`, giving `S`.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - $(val_dict[:dims])
  - `!isempty(Z)`.
  - `all(isfinite, Z)`.
  - `Z` lies in `de.metric`'s domain (see [`assert_metric_domain`](@ref)).

# Returns

  - `S::Matrix{<:Number}`: Similarity matrix, `assets × assets`.
  - $(ret_dict[:Ddist])

# Examples

```jldoctest
julia> Z = [1.0 0.0; 0.0 1.0; 1.0 1.0];

julia> S, D = cor_and_dist(FeatureDistance(), Z);

julia> S
3×3 Matrix{Float64}:
 1.0          6.12323e-17  0.707107
 6.12323e-17  1.0          0.707107
 0.707107     0.707107     1.0
```

# Related

  - [`FeatureDistance`](@ref)
  - [`distance`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
"""
function cor_and_dist(de::FeatureDistance, Z::MatNum; dims::Int = 1, kwargs...)
    D = distance(de, Z; dims = dims, kwargs...)
    return distance_to_similarity(de.sim; D = D), D
end
function cor_and_dist(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, kwargs...)
    D = distance(de, Z; dims = dims, kwargs...)
    return distance_to_similarity(de.sim; D = D), D
end
"""
    assert_feature_matrix_supplied(Z::Option{<:ArrNum}, z_src::Symbol)

Assert that a feature matrix reached [`FeatureDistance`](@ref)'s three-argument entry point, and name the reason when none did.

Every way of failing to supply `Z` arrives here identically, as `Z === nothing`. `z_src` is the diagnostic that tells them apart — usually resolved by [`feature_matrix_picker`](@ref), and riding the wire beside `Z` purely so this message can be specific:

  - `:none`: nothing supplied `Z` at all. The estimator was driven from a raw returns matrix, which carries no feature matrix — the two-argument `distance(de, Z; dims)` entry point, a [`ReturnsResult`](@ref) or a prior result is needed.
  - `:neither`: a carrier was available but neither it nor the returns result holds a feature matrix. The feature matrix has not been supplied or produced.
  - `:data` / `:prior`: `z_src` selected a carrier that holds no feature matrix, while the *other* one does. This is the typo/wrong-selector case, and the message says which value to use instead.
  - `:data_only`: the call runs *before* any prior exists, so the data carrier is the only one that could have supplied a feature matrix and it holds none. It is named for the situation rather than for the caller, so any pre-prior site inherits it; [`ClusterGroups`](@ref) is the one that exists today. Sending the user to a [`FeaturePrior`](@ref) — `:neither`'s remedy — would be actively wrong here, because a prior is structurally unreachable from a selector.

Any unrecognised symbol falls through to `:neither`'s text.

# Algorithm

 1. Return immediately when `Z` is not `nothing`, which is every call that carries a feature matrix.
 2. Otherwise select the message by `z_src`, from the five cases above.
 3. Raise an [`IsNothingError`](@ref) carrying that message.

# Arguments

  - `Z`: Feature matrix that reached the three-argument entry point, or `nothing`.
  - `z_src::Symbol`: Diagnostic naming the carrier that was selected, which picks the message.

# Validation

  - `!isnothing(Z)`. Raises an [`IsNothingError`](@ref) naming the case that `z_src` selects.

# Returns

  - `nothing`.

# Related

  - [`FeatureDistance`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`ClusterGroups`](@ref)
  - [`IsNothingError`](@ref)
"""
function assert_feature_matrix_supplied(Z::Option{<:ArrNum}, z_src::Symbol)::Nothing
    if isnothing(Z)
        throw(IsNothingError(if z_src == :none
                                 "FeatureDistance requires a feature matrix `Z`, but this call supplied none. It was reached from a raw returns matrix, which carries no feature matrix: drive it from a ReturnsResult or a prior result that carries `Z`, or call `distance(de, Z; dims = dims)` directly."
                             elseif z_src == :data
                                 "FeatureDistance requires a feature matrix `Z`, but `z_src = :data` selected the returns result and it carries no `Z`. The prior result does carry one — set `z_src = :prior`."
                             elseif z_src == :prior
                                 "FeatureDistance requires a feature matrix `Z`, but `z_src = :prior` selected the prior result and it carries no `Z`. The returns result does carry one — set `z_src = :data`."
                             elseif z_src == :data_only
                                 "FeatureDistance requires a feature matrix `Z`, but the returns result carries none. This call runs before any prior exists, so only the data carrier can supply one: set `Z` on the `ReturnsResult`."
                             else
                                 "FeatureDistance requires a feature matrix `Z`, but neither the returns result nor the prior result carries one. Supply `Z` on the ReturnsResult, or use a FeaturePrior to derive it."
                             end))
    end
    return nothing
end
"""
    distance(de::FeatureDistance, ::Any, ::Any; Z::Option{<:ArrNum} = nothing,
             z_src::Symbol = :none, kwargs...)
    cor_and_dist(de::FeatureDistance, ::Any, ::Any; Z::Option{<:ArrNum} = nothing,
                 z_src::Symbol = :none, kwargs...)

Three-argument entry points, for the clustering and network estimators.

Every consumer in the clustering and network stack calls `cor_and_dist(de, ce, X; …)` or `distance(de, pl, X; …)`, passing a covariance estimator (or, in [`logo!`](@ref)'s case, a similarity matrix) and a returns matrix. [`FeatureDistance`](@ref) uses neither: it measures a feature matrix, which travels beside them on the `Z` keyword argument, resolved from a carrier by [`feature_matrix_picker`](@ref). Both positionals are therefore ignored, and typed `::Any` rather than bounded — `logo!` puts a similarity matrix where the others put a covariance estimator.

**`dims` is ignored and the kernel is called with `dims = 1`.** The ambient `dims` describes the returns matrix `X`, and a carried `Z` is canonically assets-major regardless of it. `dims` stays meaningful only at the raw-matrix entry point `distance(de, Z; dims)`.

# Algorithm

 1. Check that a feature matrix arrived, with [`assert_feature_matrix_supplied`](@ref), which names `z_src` when none did.
 2. Call the matching two-argument entry point on `Z`, at `dims = 1`.

# Arguments

  - `de`: Feature distance estimator.
  - The two positional arguments: ignored. Present so this estimator matches the signature every consumer calls.
  - `Z`: Feature matrix, carried beside the positionals and resolved by [`feature_matrix_picker`](@ref), or `nothing`.
  - `z_src::Symbol = :none`: Diagnostic naming the carrier that was selected, read only when `Z` is `nothing`.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `!isnothing(Z)`. Raises an [`IsNothingError`](@ref) naming `z_src` (see [`assert_feature_matrix_supplied`](@ref)).

# Returns

  - From [`distance`](@ref): $(ret_dict[:Ddist])
  - From [`cor_and_dist`](@ref): the tuple `(S, D)` of the similarity matrix and that same distance matrix.

# Related

  - [`FeatureDistance`](@ref)
  - [`feature_matrix_picker`](@ref)
  - [`assert_feature_matrix_supplied`](@ref)
  - [`clusterise`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function distance(de::FeatureDistance, ::Any, ::Any; Z::Option{<:ArrNum} = nothing,
                  z_src::Symbol = :none, kwargs...)
    assert_feature_matrix_supplied(Z, z_src)
    return distance(de, Z; dims = 1)
end
function cor_and_dist(de::FeatureDistance, ::Any, ::Any; Z::Option{<:ArrNum} = nothing,
                      z_src::Symbol = :none, kwargs...)
    assert_feature_matrix_supplied(Z, z_src)
    return cor_and_dist(de, Z; dims = 1)
end

export AngularDist, MeanCollapse, MedianCollapse, LastObservation, AggregateFeatures,
       AggregateDistances, StackObservations, FeatureDistance
