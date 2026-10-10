"""
$(DocStringExtensions.TYPEDEF)

Normalised angular distance metric.

Unlike `Distances.CosineDist` (``1 - \\cos``), the angular distance satisfies the triangle inequality, so it is a true metric and the hierarchies built from it are well defined. Lemma 3 of [vandongen2012](@cite) states the angle ``\\arccos`` of a cosine or of a correlation as a metric, and it shows that ``1 - \\cos`` is not one. Section 1 of [charikar2002](@cite) divides the angle by ``\\pi``. It maps ``[-1,\\,1] \\to [1,\\,0]``, so it is bounded, scale-invariant per asset, and admits signed features. Its exact similarity counterpart is [`AngularSimilarity`](@ref), which recovers the cosine from the distance alone.

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

One matrix multiplication replaces ``N^{2}`` scalar calls, and it is the faster path from three assets upward. It loses only at ``N = 2``, where the single distance it saves does not pay for the call. So there is one matrix path and nothing to tune.

!!! note "The two paths differ on the diagonal, and the matrix path is the correct one"

    ``\\arccos(1 - r) / \\pi`` is the algebraic identity of the elementwise method, not its floating-point result. Off the diagonal the two paths agree to a few units in the last place. On the diagonal they differ more: the cosine of a vector with itself rounds only to within floating-point precision of `1`, ``\\arccos`` has an infinite derivative at `1`, so that residual is amplified into a much larger error in the distance. The matrix path writes an exact zero instead.

    `Distances.pairwise` writes an exact zero diagonal, so the matrix entry points — which are the only route [`FeatureDistance`](@ref) takes — never see the residual. Call the metric directly on a pair of identical vectors and it is there. The `"AngularDist gemm path matches the elementwise method"` testset pins the two paths together, and that is why it pins them with a tolerance.

# Related

  - [`AngularSimilarity`](@ref)
  - [`FeatureDistance`](@ref)
  - [`default_similarity`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)

# References

  - $(ref_dict[:vandongen2012]) Section 4, Lemma 3.
  - $(ref_dict[:charikar2002]) Section 1.
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
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

An unweighted collapse sets every ``w_{t}`` to ``1``. The weights are non-negative and the denominator normalises them, so the aggregate is a convex combination of the window. That is what makes it a metric when it is applied to distance matrices.

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
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])
  - ``Q_{0.5}``: The ``0.5``-quantile of the window under those weights.

An unweighted collapse sets every ``w_{t}`` to ``1``.

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

Abstract supertype for the rules that name the row [`LastObservation`](@ref) reads for each asset.

A time-varying Asset Panel lists and delists assets, so an asset can be inactive at the last row of the window. A rule states which row of the window the collapse reads for each asset. It is the `alg` field of [`LastObservation`](@ref).

# Interfaces

To add a rule, subtype `AbstractLastObservationAlgorithm` and implement the two methods below.

## `collapse_rows`

  - `collapse_rows(alg::MyRule, pnl::AssetPanel) -> Union{UnitRange{Int}, Colon}`: The rows of the panel that the stack of the Feature Matrix holds for the rule.

### Arguments

  - `alg`: The concrete subtype instance.
  - `pnl`: The Asset Panel the Feature Matrix is stacked from.

### Returns

  - A range of observations, or `Colon()` for every row.

## `feature_readable`

  - `feature_readable(alg::MyRule, A::AbstractMatrix{Bool}) -> BitVector`: Whether the rule can read each asset of the window.

### Arguments

  - `alg`: The concrete subtype instance.
  - `A`: The mask of readable cells of the window, `observations × assets`: the asset is active, and every value column that the Feature Selector names holds data, observed or filled.

### Returns

  - One entry per asset, `true` where the rule reads the asset.

# Related

  - [`LastRow`](@ref)
  - [`LastActiveRow`](@ref)
  - [`LastObservation`](@ref)
"""
abstract type AbstractLastObservationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Reads every asset at the last row of the window, the default rule of [`LastObservation`](@ref).

An asset that is inactive at the last row has no value to read. Inside a fit, the entry of the optimiser drops the asset as a non-investable asset. A direct call refuses the asset and names it.

# Related

  - [`AbstractLastObservationAlgorithm`](@ref)
  - [`LastActiveRow`](@ref)
  - [`LastObservation`](@ref)
"""
struct LastRow <: AbstractLastObservationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Reads each asset at its last active row of the window, the most recent value that an investor had.

An asset that delisted before the last row is read at the last row at which it was active. An asset with no active row in the window has no value to read.

# Related

  - [`AbstractLastObservationAlgorithm`](@ref)
  - [`LastRow`](@ref)
  - [`LastObservation`](@ref)
"""
struct LastActiveRow <: AbstractLastObservationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Discards the window and measures one row of each asset's features.

The cheapest member of the family and its default, because it is the only one whose result depends on no aggregation choice. `alg` names the row of each asset. [`LastRow`](@ref) reads the last row of the window, and [`LastActiveRow`](@ref) reads the last row at which each asset is active. It is also the one member that names its rows before the stack exists: [`collapse_rows`](@ref) answers the rows that `alg` can read, so the kernel stacks those rows of an Asset Panel alone.

# Algorithm

 1. Under [`LastRow`](@ref), or when every asset is active at the last row, take the last slice of the observation axis, `view(Z, size(Z, 1), :, :)`, giving an `assets × features` matrix.
 2. Under [`LastActiveRow`](@ref), take for each asset the row of its last active observation, giving an `assets × features` matrix.
 3. Apply the metric to that matrix once.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LastObservation(;
        alg::AbstractLastObservationAlgorithm = LastRow()
    ) -> LastObservation

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LastObservation()
LastObservation
  alg ┴ LastRow()
```

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractLastObservationAlgorithm`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
  - [`StackObservations`](@ref)
  - [`collapse_rows`](@ref)
"""
@concrete struct LastObservation <: AbstractFeatureCollapseAlgorithm
    """
    $(field_dict[:loalg])
    """
    alg
    function LastObservation(alg::AbstractLastObservationAlgorithm)::LastObservation
        return new{typeof(alg)}(alg)
    end
end
function LastObservation(;
                         alg::AbstractLastObservationAlgorithm = LastRow())::LastObservation
    return LastObservation(alg)
end
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

Abstract supertype for the rules that give the distance of a pair of assets with no shared active row.

[`AggregateDistances`](@ref) and [`StackObservations`](@ref) read each pair of assets at the rows at which both assets are active. Two assets whose active rows do not meet have no such row, so the collapse has nothing to read for the pair. A rule states what the collapse does then. It is the `pair` field of the two collapses.

# Interfaces

To add a rule, subtype `AbstractEmptyPairAlgorithm` and implement the method below. A rule that drops an asset at the entry of a fit also adds a method of `drop_empty_pairs!`, whose fallback keeps every asset.

## `empty_pair_distance!`

  - `empty_pair_distance!(D::MatNum, pair::MyRule, E::AbstractVector, de::FeatureDistance, win::NamedTuple) -> MatNum`: Writes the distance of each empty pair in `E` into `D`, or refuses.

### Arguments

  - `D`: The distance matrix, `assets × assets`, written in place at each pair of `E`.
  - `pair`: The concrete subtype instance.
  - `E`: The empty pairs, `(i, j)` with `i < j`.
  - `de`: The feature distance estimator.
  - `win`: The window, a `NamedTuple` of `Z`, `dims`, `A` and `nx`.

### Returns

  - `D`, with a distance at each pair of `E`.

# Related

  - [`RefusePair`](@ref)
  - [`DropFewerRows`](@ref)
  - [`FeatureFallback`](@ref)
  - [`AggregateDistances`](@ref)
  - [`StackObservations`](@ref)
"""
abstract type AbstractEmptyPairAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Refuses a pair of assets with no shared active row, the default rule for an empty pair.

The refusal names the two assets and the two other rules.

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`DropFewerRows`](@ref)
  - [`FeatureFallback`](@ref)
"""
struct RefusePair <: AbstractEmptyPairAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Drops the asset of an empty pair that has fewer active rows, at the entry of a fit.

The entry of the optimiser finds the first pair of assets with no shared active row, drops the asset of the pair with fewer active rows, and repeats until no pair is empty. On a tie it drops the asset that comes later in the universe. A dropped asset departs as a non-investable asset: the fit announces it, gives it a zero weight, and lists it on the Non-Investable Axis.

A direct call of [`distance`](@ref) or [`clusterise`](@ref) has no entry of a fit, so it refuses an empty pair under this rule and names the remedy.

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`RefusePair`](@ref)
  - [`FeatureFallback`](@ref)
  - [`investable_reduction`](@ref)
"""
struct DropFewerRows <: AbstractEmptyPairAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures an empty pair by the features of the two assets, each aggregated over its own active rows.

For a pair of assets with no shared active row, each asset's features are collapsed over its own active rows by `alg`, under the weights `w` restricted to those rows. The distance of the pair is then the distance of the two assets as if each held its aggregated features at every row of the window. Every other pair keeps the rule of its collapse.

The rule gives one scale to the whole matrix. Under [`AggregateDistances`](@ref) the distance of the pair is the metric of the two aggregated feature vectors, the value that every row of the window then gives. Under [`StackObservations`](@ref) each asset's aggregated feature vector fills each row of the stack, so a metric of the Minkowski family gives the same distance times the rescale of the stack at one shared row.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeatureFallback(;
        w::Option{<:ObsWeights} = nothing,
        alg::AbstractCollapseAlgorithm = MeanCollapse()
    ) -> FeatureFallback

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
julia> FeatureFallback()
FeatureFallback
    w ┼ nothing
  alg ┴ MeanCollapse()
```

# Related

  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`RefusePair`](@ref)
  - [`DropFewerRows`](@ref)
  - [`AggregateFeatures`](@ref)
"""
@propagatable @concrete struct FeatureFallback <: AbstractEmptyPairAlgorithm
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:calg])
    """
    alg
    function FeatureFallback(w::Option{<:ObsWeights},
                             alg::AbstractCollapseAlgorithm)::FeatureFallback
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(w), typeof(alg)}(w, alg)
    end
end
function FeatureFallback(; w::Option{<:ObsWeights} = nothing,
                         alg::AbstractCollapseAlgorithm = MeanCollapse())::FeatureFallback
    return FeatureFallback(w, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures every observation, then aggregates the resulting distance matrices.

Produces one distance matrix per observation and combines them into a single `assets × assets` matrix. Costs `observations` metric evaluations against [`AggregateFeatures`](@ref)'s one, and accumulates into a single buffer rather than materialising the whole stack.

Only [`MeanCollapse`](@ref) is accepted: a convex combination of metrics is a metric, an entrywise median of them is not. Because the metric is applied *before* the aggregation, the zero-feature convention is applied per observation — an asset that is zero at some observations but not others is treated as zero only in the observations where it is.

A window of an Asset Panel can hold an inactive cell. Each pair of assets is then read at the rows at which both assets are active, and the weights of those rows are divided by their sum. This is available-case estimation, as a Coverage Policy fits a covariance cell. A pair with no shared active row, or with a zero weight on each of them, takes the rule in `pair`.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= \\dfrac{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i} a_{t,\\,j} D_{t,\\,i,\\,j}}{\\sum\\limits_{t=1}^{T} w_{t} a_{t,\\,i} a_{t,\\,j}}\\,,
\\end{align}
```

Where:

  - $(math_dict[:D_ij_dist])
  - ``D_{t,\\,i,\\,j}``: Distance of assets ``i`` and ``j`` at observation ``t``.
  - ``a_{t,\\,i}``: Entry of the active mask, `1` when asset ``i`` is active at observation ``t`` and `0` otherwise.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

# Algorithm

 1. Resolve `w` against `Z` with [`collapse_weights`](@ref), giving a weight vector of one entry per observation, or `nothing`.
 2. Allocate the accumulator `D` and the single per-observation buffer `Dt`, both `assets × assets`, and set the weight total `sw` to zero.
 3. For each observation `t`: measure that slice of `Z` into `Dt`; apply the zero-feature-vector convention to `Dt`; read the observation's weight `wt`, which is `one(T)` when `w` is `nothing`; add `wt .* Dt` to `D`; and add `wt` to `sw`.
 4. Divide `D` by `sw`, giving the convex combination of the per-observation distance matrices.

When the window holds an inactive cell, [`active_feature_distance`](@ref) keeps one weight total per pair in step 2, adds an entry in step 3 only at a row where both of its assets are active, divides each entry by its own total in step 4, and gives each empty pair to `pair`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AggregateDistances(;
        w::Option{<:ObsWeights} = nothing,
        alg::AbstractCollapseAlgorithm = MeanCollapse(),
        pair::AbstractEmptyPairAlgorithm = RefusePair()
    ) -> AggregateDistances

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:oow])
  - `alg` is not a [`MedianCollapse`](@ref).

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `pair`: Recursively updated via [`factory`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).
  - `pair`: Recursively indexed via [`obs_weights_view`](@ref).

# Examples

```jldoctest
julia> AggregateDistances()
AggregateDistances
     w ┼ nothing
   alg ┼ MeanCollapse()
  pair ┴ RefusePair()

julia> AggregateDistances(; alg = MedianCollapse())
ERROR: ArgumentError: alg must not be a MedianCollapse: an entrywise median of distance matrices need not satisfy the triangle inequality, so the result would not be a metric. Use MeanCollapse, or aggregate the features instead with AggregateFeatures.
[...]
```

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractCollapseAlgorithm`](@ref)
  - [`AbstractEmptyPairAlgorithm`](@ref)
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
    """
    $(field_dict[:fdpair])
    """
    @fprop pair
    function AggregateDistances(w::Option{<:ObsWeights}, alg::AbstractCollapseAlgorithm,
                                pair::AbstractEmptyPairAlgorithm)::AggregateDistances
        assert_nonempty_nonneg_finite_val(w, :w)
        @argcheck(!isa(alg, MedianCollapse),
                  ArgumentError("alg must not be a MedianCollapse: an entrywise median of distance matrices need not satisfy the triangle inequality, so the result would not be a metric. Use MeanCollapse, or aggregate the features instead with AggregateFeatures."))
        return new{typeof(w), typeof(alg), typeof(pair)}(w, alg, pair)
    end
end
function AggregateDistances(; w::Option{<:ObsWeights} = nothing,
                            alg::AbstractCollapseAlgorithm = MeanCollapse(),
                            pair::AbstractEmptyPairAlgorithm = RefusePair())::AggregateDistances
    return AggregateDistances(w, alg, pair)
end
"""
$(DocStringExtensions.TYPEDEF)

Concatenates the window into one long feature vector per asset, so nothing is averaged away.

Turns `observations × assets × features` into an `assets × (observations · features)` matrix along the feature axis, and applies the metric once. Two assets are close only when their whole trajectories agree — which is also why the result is dominated by whichever observations carry the largest magnitudes, and why heterogeneous features should be standardised before it is used.

Equals none of the other members of the family in general, but agrees with all of them when `observations == 1`.

A window of an Asset Panel can hold an inactive cell. Each pair of assets then stacks the ``n`` rows at which both assets are active, and [`stack_rescale`](@ref) rescales its distance from those rows to the ``T`` rows of the window. A metric that sums over the coordinates, such as the Minkowski family, `Distances.SqEuclidean` or `Distances.ChiSqDist`, takes a factor of ``T / n`` inside its power. A ratio metric, such as [`AngularDist`](@ref) or `Distances.CosineDist`, takes no factor. A metric with no rescale method refuses a pair that shares fewer rows than the window. A pair with no shared active row takes the rule in `pair`.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= \\left(\\dfrac{T}{n_{i,\\,j}}\\right)^{1/p} m\\left(\\boldsymbol{z}_{i,\\,\\mathcal{S}_{i,\\,j}},\\, \\boldsymbol{z}_{j,\\,\\mathcal{S}_{i,\\,j}}\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:D_ij_dist])
  - ``\\mathcal{S}_{i,\\,j}``: Rows at which both assets ``i`` and ``j`` are active.
  - ``n_{i,\\,j}``: Number of rows in ``\\mathcal{S}_{i,\\,j}``.
  - ``\\boldsymbol{z}_{i,\\,\\mathcal{S}_{i,\\,j}}``: Stacked feature vector of asset ``i`` over the rows of ``\\mathcal{S}_{i,\\,j}``.
  - ``m``: Distance metric, `metric`.
  - ``p``: Power of the metric: ``1`` for `Distances.Cityblock`, `Distances.SqEuclidean` and `Distances.ChiSqDist`, ``2`` for `Distances.Euclidean`, ``p`` for `Distances.Minkowski`, and ``\\infty`` for a ratio metric, which takes no factor.
  - $(math_dict[:T])

This is the rule that R's `stats::dist` applies to a missing coordinate. A weighted metric of the Minkowski family takes the ratio of its weight sums in place of ``T / n``.

# Algorithm

 1. Permute `Z` so the asset axis leads: `(2, 1, 3)` at `dims = 1`, and `(3, 1, 2)` at `dims = 2`.
 2. Reshape the permuted array to `assets × (observations · features)`, giving one long feature vector per asset.
 3. Apply the metric to that matrix once, along its first axis.

When the window holds an inactive cell, [`active_feature_distance`](@ref) then measures each pair that shares fewer rows than the window again, over its shared rows, rescales it with [`stack_rescale`](@ref), and gives each empty pair to `pair`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StackObservations(;
        pair::AbstractEmptyPairAlgorithm = RefusePair()
    ) -> StackObservations

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `pair`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> StackObservations()
StackObservations
  pair ┴ RefusePair()
```

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`AbstractEmptyPairAlgorithm`](@ref)
  - [`LastObservation`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
  - [`stack_rescale`](@ref)
"""
@propagatable @concrete struct StackObservations <: AbstractFeatureCollapseAlgorithm
    """
    $(field_dict[:fdpair])
    """
    @fprop pair
    function StackObservations(pair::AbstractEmptyPairAlgorithm)::StackObservations
        return new{typeof(pair)}(pair)
    end
end
function StackObservations(;
                           pair::AbstractEmptyPairAlgorithm = RefusePair())::StackObservations
    return StackObservations(pair)
end
"""
$(DocStringExtensions.TYPEDEF)

Turns a feature matrix into a distance matrix, by applying a metric to the rows of that matrix.

A feature matrix describes assets by their exposures, memberships, loadings or adjacencies rather than by their returns. This estimator is a peer of [`Distance`](@ref) and [`DistanceDistance`](@ref). Unlike them, it never reads a correlation matrix, so it works where returns are uninformative or unavailable.

The estimator accepts any `Distances.SemiMetric`, including a user-defined one, and every metric has a similarity, so no combination throws on this path. The remarks below are about the metric a caller chooses, not about this type.

!!! warning "A metric is not automatically in the similarity's domain"

    Under the default [`ComplementSimilarity`](@ref), a distance above `1` gives a negative similarity, and a distance above `2` gives a similarity below `-1`, which [`plot_clusters`](@ref) clips silently. The first threshold is `1`, not "the metric is unbounded". `Distances.CosineDist` and `Distances.CorrDist` are bounded by `2`, and they cross `1` at every negative cosine or correlation. `Distances.Euclidean` has no bound, and it crosses `2` on features of large magnitude.

    That claim is scoped to this path. Handing this estimator to a [`NetworkEstimator`](@ref), [`DBHT`](@ref) or [`LoGo`](@ref) as their `de` puts the resulting distance matrix on the PMFG path, where **their own** similarity field applies rather than `sim`, and where [`assert_similarity_domain`](@ref) refuses a distance above `1` under [`ComplementSimilarity`](@ref) and a non-finite one under [`MaximumDistanceSimilarity`](@ref).

Every metric other than [`AngularDist`](@ref) and `Distances.CorrDist` is scale-sensitive, and even [`AngularDist`](@ref) is invariant to scaling an asset's feature vector but not to scaling a feature across assets. Standardise heterogeneous features before use. `Distances.CorrDist` leaves `NaN` against a constant feature vector that is not zero, so it cannot measure a single feature, where every feature vector is constant.

`Distances.Jaccard` is the general Ruzicka form over the non-negative reals, not the binary-set Jaccard, and it returns values up to `2` on signed input without an error. It, `Distances.BrayCurtis` and `Distances.ChiSqDist` therefore require a non-negative feature matrix, which [`assert_metric_domain`](@ref) checks in the kernel rather than at construction, because the feature matrix is not known here.

## Choosing the columns

`sel` names the Panel Fields the Feature Matrix stacks, and `nothing` stacks every Panel Field's values. Without it, this estimator stacks the whole panel, which is harmless while a panel holds only features and wrong as soon as it holds anything else.

An entry of `sel` takes one of four forms, and they mix freely in one vector:

  - `"industry"` is a Panel Field name, and stands for the value columns of that Panel Field alone.
  - `"industry" => ["Tech", "Energy"]` keeps the levels or labels it names, in that order.
  - `"industry" => "Tech"` keeps one level or label. A column label takes this form.
  - `"mcap" => :observed` is the observed mask of the Panel Field, one `0`/`1` column.

There is no integer entry. Every Panel Field, level and label carries a name, so a position has nothing to index. A taxonomy is selected by the name of the categorical Panel Field it entered the panel as.

An entry that names a field, a level or a label that the panel does not hold throws when `strict` is `true`. Otherwise it warns, and the estimator drops the entry.

The order of `sel` is the column order the metric reads, so a caller decides it. [`feature_matrix`](@ref) stacks the matrix and [`feature_labels`](@ref) names its columns, one selector entry per column.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= m\\left(\\boldsymbol{z}_{i},\\, \\boldsymbol{z}_{j}\\right)\\\\
S_{i,\\,j} &= \\sigma\\left(D_{i,\\,j}\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:D_ij_dist])
  - $(math_dict[:S_ij_sim])
  - $(math_dict[:z_i_feature])
  - ``m``: Distance metric, `metric`.
  - ``\\sigma``: Similarity transformation, `sim`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FeatureDistance(;
        metric::Distances.SemiMetric = AngularDist(),
        alg::AbstractFeatureCollapseAlgorithm = LastObservation(),
        sim::AbstractSimilarityMatrixAlgorithm = default_similarity(metric),
        ape::Option{<:AbstractAssetPanelEstimator} = nothing,
        sel::Option{<:AbstractVector} = nothing,
        strict::Bool = false
    ) -> FeatureDistance

Keywords correspond to the struct's fields. `sim` defaults to [`default_similarity`](@ref) of `metric` at construction, so the printed object shows the resolved similarity, and the distance kernel resolves nothing.

## Validation

  - `sel` is checked by [`assert_feature_selector`](@ref): `nothing`, or a non-empty vector of distinct entries, each of the four admitted forms.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `alg`: Recursively updated via [`factory`](@ref).
  - `ape`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> FeatureDistance()
FeatureDistance
  metric ┼ AngularDist: AngularDist()
     alg ┼ LastObservation
         │   alg ┴ LastRow()
     sim ┼ AngularSimilarity()
     ape ┼ nothing
     sel ┼ nothing
  strict ┴ Bool: false

julia> FeatureDistance(; metric = PortfolioOptimisers.Distances.CosineDist())
FeatureDistance
  metric ┼ Distances.CosineDist: Distances.CosineDist()
     alg ┼ LastObservation
         │   alg ┴ LastRow()
     sim ┼ ComplementSimilarity()
     ape ┼ nothing
     sel ┼ nothing
  strict ┴ Bool: false
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
  - [`assert_feature_selector`](@ref): the construction check on `sel`.
  - [`select_fields`](@ref): the one resolution of `sel` against an [`AssetPanel`](@ref).
  - [`feature_matrix`](@ref): the stacking itself.
  - [`feature_labels`](@ref): one selector entry per column of the stacked matrix.
  - [`DBHT`](@ref): carries a `sim` field of its own, named alike on purpose: it has the same type and the same job. When both are set DBHT's wins, because [`clusterise`](@ref) overwrites the similarity matrix immediately after [`cor_and_dist`](@ref) returns.
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
    """
    $(field_dict[:fdape])
    """
    @fprop ape
    """
    $(field_dict[:fdsel])
    """
    sel
    """
    $(field_dict[:fdstrict])
    """
    strict
    function FeatureDistance(metric::Distances.SemiMetric,
                             alg::AbstractFeatureCollapseAlgorithm,
                             sim::AbstractSimilarityMatrixAlgorithm,
                             ape::Option{<:AbstractAssetPanelEstimator},
                             sel::Option{<:AbstractVector}, strict::Bool)::FeatureDistance
        assert_feature_selector(sel)
        return new{typeof(metric), typeof(alg), typeof(sim), typeof(ape), typeof(sel),
                   typeof(strict)}(metric, alg, sim, ape, sel, strict)
    end
end
function FeatureDistance(; metric::Distances.SemiMetric = AngularDist(),
                         alg::AbstractFeatureCollapseAlgorithm = LastObservation(),
                         sim::AbstractSimilarityMatrixAlgorithm = default_similarity(metric),
                         ape::Option{<:AbstractAssetPanelEstimator} = nothing,
                         sel::Option{<:AbstractVector} = nothing,
                         strict::Bool = false)::FeatureDistance
    return FeatureDistance(metric, alg, sim, ape, sel, strict)
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

Turn a 2-D feature matrix into a distance matrix. This is the shared kernel behind every [`FeatureDistance`](@ref) entry point, and three of the collapse algorithms differ only in the matrix they hand it. [`AggregateDistances`](@ref) does not call it. It runs the same two steps once per observation, into one buffer that it reuses, and aggregates the results.

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

`Z` is matricised to `observations × (assets · features)` first, because [`get_observation_weights`](@ref)'s documented interface is `VecNum`/`MatNum` and a raw 3-D array matches neither — a user's correct `MatNum` method would otherwise never fire. There is no caller-side `nothing` guard: [`get_observation_weights`](@ref) raises [`ObservationWeightsError`](@ref) itself when a [`DynamicAbstractWeights`](@ref) cannot resolve, so `nothing` here means only that no weights were requested.

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
    distance(de::FeatureDistance, Z::MatNum; dims::Int = 1, amsk = nothing, nx = nothing,
             kwargs...)
    distance(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, amsk = nothing, nx = nothing,
             kwargs...)

Compute the distance matrix from a feature matrix.

The 2-D method collapses nothing: a static feature matrix has no observation axis, so the collapse algorithm reads its one row. Its `amsk` is `1 × assets` and marks the assets whose every value column holds data, and the method refuses an asset that it marks `false`. The 3-D method dispatches on it. Assets whose feature vector is entirely zero are given the convention documented in [`patch_zero_feature_vectors!`](@ref).

The 3-D method reads the active mask `amsk` of the window. A collapse then reads each asset, or each pair of assets, at its own active rows, as each member of [`AbstractFeatureCollapseAlgorithm`](@ref) states. A window with no inactive cell gives the result that it gives with no mask.

# Algorithm

 1. Validate `Z` and `dims` with [`assert_feature_matrix`](@ref).
 2. On the 2-D method, check the one-row mask with [`static_window`](@ref) and [`assert_feature_readable`](@ref), and hand `de.metric` and `Z` to the kernel.
 3. On the 3-D method, check the active mask with [`active_window`](@ref), and check that the collapse can read each asset with [`assert_feature_readable`](@ref).
 4. With no inactive cell, hand `de` and `Z` to the collapse dispatcher [`feature_distance`](@ref), which selects the branch that `de.alg` names. Otherwise hand the window to [`active_feature_distance`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])
  - `amsk`: The mask of readable cells of the window, `observations × assets` on the 3-D method and `1 × assets` on the 2-D method, or `nothing` when every cell is readable.
  - `nx`: The asset names that a refusal quotes, or `nothing` to quote positions.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - $(val_dict[:dims])
  - `!isempty(Z)`.
  - `all(isfinite, Z)`.
  - `Z` lies in `de.metric`'s domain (see [`assert_metric_domain`](@ref)).
  - `amsk` is `observations × assets` on the 3-D method and `1 × assets` on the 2-D method, and the collapse can read each asset. See [`assert_feature_readable`](@ref).
  - On the 3-D method, a pair of assets with no shared active row takes the rule of the `pair` field of [`AggregateDistances`](@ref) and [`StackObservations`](@ref).

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
function distance(de::FeatureDistance, Z::MatNum; dims::Int = 1, amsk = nothing,
                  nx = nothing, kwargs...)
    assert_dims(dims)
    assert_feature_matrix(de, Z, dims)
    assert_feature_readable(de.alg, static_window(amsk, Z, dims), nx)
    return feature_distance(de.metric, Z, dims)
end
function distance(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, amsk = nothing,
                  nx = nothing, kwargs...)
    assert_dims(dims)
    assert_feature_matrix(de, Z, dims)
    A = active_window(amsk, Z, dims)
    assert_feature_readable(de.alg, A, nx)
    return if isnothing(A)
        feature_distance(de, Z, dims)
    else
        active_feature_distance(de, (; Z = Z, dims = dims, A = A, nx = nx))
    end
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
    collapse_rows(alg::AbstractFeatureCollapseAlgorithm, pnl::AssetPanel)

Name the observation rows a collapse algorithm reads, so the kernel stacks those rows alone.

A time-varying Feature Matrix is stacked from an Asset Panel and then collapsed along its observation axis, and a collapse that reads one row has no use for the others. [`LastObservation`](@ref) under [`LastRow`](@ref) reads the last row, so it names it, and the stack it is handed is a window of one observation: a lifted static Panel Field, whose values are a [`RepeatedLeading`](@ref), is then read once rather than once per observation, and the kernel's cost under the default collapse is the `assets × features` slice it measures. Under [`LastActiveRow`](@ref) it names the rows from the earliest last active row of an asset to the last row, which hold the last active row of every asset that has one. Every other member answers `Colon()`, every row. The two aggregates resolve their weights against the stacked window itself (see [`collapse_weights`](@ref)), so a window they could cut is not known before the stack exists, and [`StackObservations`](@ref) reads the whole stack by definition.

A static panel has no observation axis, so every member answers `Colon()` on one, [`LastObservation`](@ref) included.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`LastObservation`](@ref) forwards to its rule.
 2. [`LastRow`](@ref) on a time-varying panel: the last observation, `nobs:nobs`, where `nobs` is the observation count [`panel_axes`](@ref) reads.
 3. [`LastActiveRow`](@ref) on a time-varying panel: the rows that [`last_active_rows`](@ref) names on the active mask.
 4. Every other case: `Colon()`.

# Arguments

  - `alg`: The collapse algorithm, or the rule of a [`LastObservation`](@ref).
  - `pnl`: The Asset Panel the Feature Matrix is stacked from.

# Returns

  - `rows`: A range of observations, or `Colon()`. The `rows` keyword of [`feature_matrix`](@ref).

# Related

  - [`AbstractFeatureCollapseAlgorithm`](@ref)
  - [`LastObservation`](@ref)
  - [`feature_matrix`](@ref)
  - [`stacked_axes`](@ref)
  - [`panel_axes`](@ref)
"""
function collapse_rows(::AbstractFeatureCollapseAlgorithm, ::AssetPanel)
    return Colon()
end
function collapse_rows(alg::LastObservation, pnl::AssetPanel)
    return collapse_rows(alg.alg, pnl)
end
function collapse_rows(::LastRow, pnl::AssetPanel)
    ax = panel_axes(pnl)
    return length(ax) == 2 ? (ax[1]:ax[1]) : Colon()
end
function collapse_rows(::LastActiveRow, pnl::AssetPanel)
    return last_active_rows(pnl.amsk)
end
"""
    feature_matrix(de::FeatureDistance, pr, rd, X) -> AbstractArray{<:Number}

Stack the Feature Matrix a [`FeatureDistance`](@ref) measures, from the panel its `ape` slot resolves.

One site resolves the panel. Under a `nothing` producer, `asset_panel(de.ape, pr, rd, X)` returns the panel that the [`ReturnsResult`](@ref) holds, and otherwise it builds one. The panel method of [`feature_matrix`](@ref) then stacks the columns that `de.sel` names, over the observation rows that `de.alg` reads. The kernel calls this method, and [`feature_labels`](@ref) resolves the panel through the same call, so the labels that a caller rebuilds name the columns that the kernel measured.

The collapse algorithm names the rows, through [`collapse_rows`](@ref). Under [`LastObservation`](@ref), a time-varying panel stacks its last observation alone, `1 × assets × features`, which is the slice that the collapse measures. Every other collapse stacks every observation.

# Algorithm

 1. Resolve the panel with [`asset_panel`](@ref).
 2. Stack it with [`feature_matrix`](@ref), reading `de.sel` and `de.strict`, over the rows [`collapse_rows`](@ref) names for `de.alg`.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - $(arg_dict[:X_sub]) A producer reads it.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The panel holds a Panel Field, and `de.sel` resolves to at least one column. See [`select_fields`](@ref).

# Returns

  - `Z::Array`: The Feature Matrix, `assets × features` or `observations × assets × features`, where the observation count is the one that `de.alg` reads.

# Related

  - [`FeatureDistance`](@ref)
  - [`feature_labels`](@ref)
  - [`collapse_rows`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
  - [`cor_and_dist`](@ref)
"""
function feature_matrix(de::FeatureDistance, pr, rd, X)
    return first(feature_window(de, pr, rd, X))
end
"""
    feature_window(de::FeatureDistance, pr, rd, X) -> (Z, A)

Stack the Feature Matrix a [`FeatureDistance`](@ref) measures, beside the mask of the cells it can read in the rows it stacks.

It is [`feature_matrix`](@ref) with the mask added. A cell is readable where the asset is active and every value column of `de.sel` holds data: a value that the raw input carried, or that a fill policy wrote. The mask is cut to the rows that the collapse reads, so it has one row per observation of the stack. A static panel has one row, the cells of the assets. Its mask is `nothing` when every cell holds data, and a `1 × assets` matrix otherwise. A time-varying panel whose every cell is readable gives the active mask, which the kernel drops when it is all `true`.

The stack holds a zero at a cell that holds a placeholder, never the placeholder that the panel stores. The mask leaves that cell out, so the zero is never read, and the stack is the same for every placeholder.

# Algorithm

 1. Resolve the panel with [`asset_panel`](@ref), and resolve `de.sel` against it with [`select_fields`](@ref).
 2. Name the rows the collapse reads, and cut the mask of readable cells to them, with [`feature_window_mask`](@ref).
 3. Stack the panel over those rows with [`feature_stack`](@ref), with a zero at each cell that holds a placeholder.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - $(arg_dict[:X_sub]) A producer reads it.

# Returns

  - `Z::Array`: The Feature Matrix, as [`feature_matrix`](@ref) returns it.
  - `A::Option{<:AbstractMatrix{Bool}}`: The mask of readable cells of the stacked rows, `observations × assets`, or `1 × assets` for a static panel with a placeholder, or `nothing`.

# Related

  - [`feature_matrix`](@ref)
  - [`feature_window_mask`](@ref)
  - [`collapse_rows`](@ref)
  - [`distance`](@ref)
"""
function feature_window(de::FeatureDistance, pr, rd, X)
    pnl = asset_panel(de.ape, pr, rd, X)
    cols = select_fields(pnl, de.sel, de.strict)
    rows, A = feature_window_mask(de.alg, pnl, cols)
    #! The mask leaves out every placeholder, so the zero written there is never read. It
    #! keeps the stack finite for the checks of the kernel, whatever the placeholder holds.
    return feature_stack(pnl, cols; rows = rows, placeholder = 0), A
end
"""
    window_activity(amsk, rows)

Cut a mask of the cells of an Asset Panel to the rows that a Feature Matrix stacks. A static panel has no observation axis. Its mask, one entry per asset, becomes a window of one row, `1 × assets`, and `nothing` when every entry is `true`. No mask gives no window.

# Arguments

  - `amsk`: The mask, `observations × assets` or `assets`, or `nothing`.
  - $(arg_dict[:fdrows])

# Returns

  - `A::Option{<:AbstractMatrix{Bool}}`: `amsk[rows, :]`, the one-row window of a static mask, or `nothing`.

# Related

  - [`feature_window`](@ref)
  - [`collapse_rows`](@ref)
"""
function window_activity(::Nothing, ::Any)
    return nothing
end
function window_activity(amsk::AbstractMatrix{Bool}, rows)
    return amsk[rows, :]
end
function window_activity(o::AbstractVector{Bool}, ::Colon)
    return all(o) ? nothing : reshape(BitVector(o), 1, :)
end
"""
    feature_asset_names(pr, rd)

Give the asset names that a refusal of a [`FeatureDistance`](@ref) quotes: those of the returns data `rd`, or those of `pr` when a `ReturnsResult` comes in its place, and `nothing` otherwise, so that the refusal quotes positions.

# Related

  - [`feature_asset_labels`](@ref)
  - [`distance`](@ref)
"""
function feature_asset_names(::Any, rd::ReturnsResult)
    return rd.nx
end
function feature_asset_names(pr::ReturnsResult, ::Nothing)
    return pr.nx
end
function feature_asset_names(::Any, ::Any)
    return nothing
end
"""
    feature_labels(de::FeatureDistance, pr, rd, X) -> Vector

Name each column of the Feature Matrix a [`FeatureDistance`](@ref) measures.

It is the sibling of [`feature_matrix`](@ref). It resolves the panel and the selector the same way, so the two agree by construction. A label is the selector entry that selects exactly that column, so the label vector is itself a selector that rebuilds the matrix. A caller who asks *what was measured* needs exactly that.

The kernel never calls it, and no clustering or phylogeny result records the labels, because the estimator, `pr` and `rd` derive them without a distance computation. A caller who wants them calls `feature_labels(de, res.pr, rd, rd.X)` with the arguments the optimiser received.

# Algorithm

 1. Resolve the panel with [`asset_panel`](@ref).
 2. Label it with [`feature_labels`](@ref), reading `de.sel` and `de.strict`.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - $(arg_dict[:X_sub]) A producer reads it.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The panel holds a Panel Field, and `de.sel` resolves to at least one column. See [`select_fields`](@ref).

# Returns

  - `labels::Vector`: One Feature Selector entry per column of the Feature Matrix, in column order.

# Related

  - [`FeatureDistance`](@ref)
  - [`feature_matrix`](@ref)
  - [`asset_panel`](@ref)
  - [`AssetPanel`](@ref)
"""
function feature_labels(de::FeatureDistance, pr, rd, X)
    return feature_labels(asset_panel(de.ape, pr, rd, X), de.sel; strict = de.strict)
end
"""
    distance(de::FeatureDistance, ::Any, X; pr = nothing, rd = nothing, kwargs...)

Compute the distance matrix of the Feature Matrix that [`feature_matrix`](@ref) stacks from `pr` and `rd`, for the clustering and network estimators.

Every consumer in the clustering and network stack calls `cor_and_dist(de, ce, X; …)` or `distance(de, pl, X; …)`, and passes a covariance estimator and a returns matrix. [`logo!`](@ref) passes a similarity matrix in place of the covariance estimator, so the second positional is typed `::Any` rather than bounded. [`FeatureDistance`](@ref) does not read that positional. It **does** read `X`, because a producer measures it.

The prior result and the returns data come in the keyword tail as `pr` and `rd`, and [`feature_matrix`](@ref) resolves the panel from them and from `de.ape`. A forwarder that takes a prior result passes both. Preselection passes `rd` alone.

**This method ignores `dims` and calls the kernel with `dims = 1`.** The ambient `dims` describes the returns matrix `X`, and a stacked Feature Matrix is assets-major whatever `dims` says. `dims` has a meaning only at the raw-matrix entry point `distance(de, Z; dims)`.

# Algorithm

 1. Stack the Feature Matrix with [`feature_window`](@ref), which resolves the panel and cuts it to `de.sel` and to the observation rows that `de.alg` reads, beside the mask of the cells it can read in those rows: the asset is active, and every value column of `de.sel` was observed.
 2. Compute the distance matrix `D` of that stack with the two-argument method, at `dims = 1`, with that mask and the asset names of `rd`.

# Arguments

  - `de`: Feature distance estimator.
  - The second positional: ignored. It is present so that this estimator matches the signature every consumer calls.
  - $(arg_dict[:X_sub]) A producer reads it.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `kwargs...`: Additional keyword arguments (ignored), `dims` among them.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The stacked Feature Matrix passes [`assert_feature_matrix`](@ref) at `dims = 1`: it is not empty, every entry is finite, and it lies in the domain of `de.metric`.
  - The collapse can read each asset at the readable rows of the window, see [`assert_feature_readable`](@ref), and a pair with no shared readable row takes the `pair` rule of its collapse. A fit drops an unreadable asset at its entry with [`feature_readable_mask`](@ref), and this call refuses it.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`FeatureDistance`](@ref)
  - [`cor_and_dist`](@ref): the same stack, with the similarity matrix as well.
  - [`feature_matrix`](@ref)
  - [`feature_labels`](@ref)
  - [`asset_panel`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function distance(de::FeatureDistance, ::Any, X; pr = nothing, rd = nothing, kwargs...)
    Z, A = feature_window(de, pr, rd, X)
    return distance(de, Z; dims = 1, amsk = A, nx = feature_asset_names(pr, rd))
end
"""
    cor_and_dist(de::FeatureDistance, ::Any, X; pr = nothing, rd = nothing, kwargs...)

Compute the similarity and distance matrices of the Feature Matrix that [`feature_matrix`](@ref) stacks from `pr` and `rd`, for the clustering and network estimators.

This is the form that [`clusterise`](@ref) and the network estimators call. It reads its arguments as the three-argument [`distance`](@ref) method does: the second positional is ignored, `X` reaches a producer, `pr` and `rd` resolve the panel, and `dims` is ignored.

# Algorithm

 1. Stack the Feature Matrix with [`feature_window`](@ref), which resolves the panel and cuts it to `de.sel` and to the observation rows that `de.alg` reads, beside the mask of the cells it can read in those rows: the asset is active, and every value column of `de.sel` was observed.
 2. Compute the similarity matrix `S` and the distance matrix `D` of that stack with the two-argument method, at `dims = 1`, with that mask and the asset names of `rd`.

# Arguments

  - `de`: Feature distance estimator.
  - The second positional: ignored. It is present so that this estimator matches the signature every consumer calls.
  - $(arg_dict[:X_sub]) A producer reads it.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `kwargs...`: Additional keyword arguments (ignored), `dims` among them.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The stacked Feature Matrix passes [`assert_feature_matrix`](@ref) at `dims = 1`: it is not empty, every entry is finite, and it lies in the domain of `de.metric`.
  - The collapse can read each asset at the readable rows of the window, see [`assert_feature_readable`](@ref), and a pair with no shared readable row takes the `pair` rule of its collapse. A fit drops an unreadable asset at its entry with [`feature_readable_mask`](@ref), and this call refuses it.

# Returns

  - `S::Matrix{<:Number}`: Similarity matrix, `assets × assets`, derived from `D` under `de.sim`.
  - $(ret_dict[:Ddist])

# Related

  - [`FeatureDistance`](@ref)
  - [`distance`](@ref): the same stack, without the similarity matrix.
  - [`feature_matrix`](@ref)
  - [`feature_labels`](@ref)
  - [`clusterise`](@ref)
"""
function cor_and_dist(de::FeatureDistance, ::Any, X; pr = nothing, rd = nothing, kwargs...)
    Z, A = feature_window(de, pr, rd, X)
    return cor_and_dist(de, Z; dims = 1, amsk = A, nx = feature_asset_names(pr, rd))
end

export AngularDist, MeanCollapse, MedianCollapse, LastObservation, AggregateFeatures,
       AggregateDistances, StackObservations, FeatureDistance, LastRow, LastActiveRow,
       RefusePair, DropFewerRows, FeatureFallback
public AbstractLastObservationAlgorithm, AbstractEmptyPairAlgorithm, collapse_rows
