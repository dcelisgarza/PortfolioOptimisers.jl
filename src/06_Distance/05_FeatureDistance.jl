"""
$(DocStringExtensions.TYPEDEF)

Measures the angle between two feature vectors as a fraction of ``\\pi``.

The angular distance satisfies the triangle inequality, so it is a true metric, and a hierarchy built from it is well defined. `Distances.CosineDist`, which is ``1 - \\cos``, does not satisfy it. Lemma 3 of [vandongen2012](@cite) states that the angle ``\\arccos`` of a cosine or of a correlation is a metric, and it shows that ``1 - \\cos`` is not one. Section 1 of [charikar2002](@cite) divides the angle by ``\\pi``. The metric maps ``[-1,\\,1] \\to [1,\\,0]``, so it is bounded, it is invariant to a positive scale of an asset's feature vector, and it accepts signed features. Its exact similarity counterpart is [`AngularSimilarity`](@ref), which recovers the cosine from the distance alone.

A zero feature vector has no direction, so its cosine is undefined. The metric puts two zero vectors at distance `0`, because they are identical. It puts a zero vector at distance `1` from every non-zero vector, the largest distance. This convention keeps ``S = \\cos(\\pi D)`` true at every entry of the matching similarity matrix.

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

The metric has two paths, and both are part of its contract. The elementwise method measures one pair of feature vectors, and `Distances._pairwise!` measures a whole matrix.

The elementwise method, `AngularDist()(a, b)`:

 1. Promote the element types of `a` and `b` with `Float64` to get `T`.
 2. Take the norm `na` of `a` and the norm `nb` of `b`.
 3. Return `zero(T)` when both norms are zero, and `one(T)` when exactly one of them is zero. This is the convention for a zero feature vector above.
 4. Divide the dot product of `a` and `b` by `na * nb` to get the cosine.
 5. Clamp the cosine to ``[-1,\\,1]``, take its ``\\arccos``, and divide by ``\\pi``.

The matrix method, `Distances._pairwise!(::AngularDist, r, a)`, receives `a` with one column per asset, so a zero column of `a` is a zero feature vector:

 1. Pass the whole matrix to the `Distances.CosineDist` kernel. The kernel writes ``1 - \\cos`` into `r` with one BLAS `gemm` call. It divides by the norm, so a zero column of `a` leaves `NaN` in its row and in its column of `r`.
 2. Mark the zero columns of `a` in the mask `z`.
 3. Rewrite every entry of `r` in place. A diagonal entry and an entry of two zero columns become `zero(T)`. An entry of a zero column and a non-zero column becomes `one(T)`. Every other entry becomes ``\\arccos(1 - r_{i,\\,j}) / \\pi``.

One matrix multiplication replaces ``N^{2}`` scalar calls, and it is the faster path from three assets upward. At ``N = 2`` it is slower, because the one distance that it saves costs less than the call. The loss is at ``N = 2`` alone, so the metric has one matrix path and no setting that selects a path.

!!! note "The two paths differ on the diagonal, and the matrix path is the correct one"

    ``\\arccos(1 - r) / \\pi`` is the algebraic identity of the elementwise method, not its floating-point result. Off the diagonal, the two paths agree to a few units in the last place. On the diagonal, they differ more. The cosine of a vector with itself rounds to a value near `1`, not always to `1`. ``\\arccos`` has an infinite derivative at `1`, so it makes that small residual into a much larger error in the distance. The matrix path writes an exact zero instead.

    `Distances.pairwise` writes an exact zero diagonal. [`FeatureDistance`](@ref) reaches the metric through the matrix entry points alone, so it never sees the residual. A direct call of the metric on two identical vectors returns it. The testset `"AngularDist gemm path matches the elementwise method"` compares the two paths, and for this reason it compares them with a tolerance.

# Related

  - [`AngularSimilarity`](@ref)
  - [`FeatureDistance`](@ref)
  - [`default_similarity`](@ref)
  - [`Distances.jl`](https://docs.juliahub.com/General/Distances/stable/)

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
        r[i, j] = if i == j || (z[i] && z[j])
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

A collapse algorithm states how to aggregate along the observation axis of a window of time-varying features. [`AggregateFeatures`](@ref) and [`AggregateDistances`](@ref) read it. These two state what to aggregate, the features or the distances, so the two families are independent.

# Related

  - [`MeanCollapse`](@ref)
  - [`MedianCollapse`](@ref)
  - [`AbstractFeatureCollapseAlgorithm`](@ref)
"""
abstract type AbstractCollapseAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Aggregates along the observation axis with the possibly weighted arithmetic mean.

Of the two collapse algorithms, [`AggregateDistances`](@ref) accepts this one alone, because a convex combination of metrics is a metric. [`AggregateFeatures`](@ref) accepts it too. It is the default of both.

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

An unweighted collapse sets every ``w_{t}`` to ``1``. The weights are non-negative and the denominator normalises them, so the aggregate is a convex combination of the window. For this reason, the aggregate of distance matrices is a metric.

# Algorithm

 1. Reduce the first axis of `Z`, the observation axis, with `Statistics.mean`, weighted by `w` when `w` is not `nothing`.
 2. Drop the reduced axis to get an `assets × features` matrix.

# Related

  - [`AbstractCollapseAlgorithm`](@ref)
  - [`MedianCollapse`](@ref)
  - [`AggregateFeatures`](@ref)
  - [`AggregateDistances`](@ref)
"""
struct MeanCollapse <: AbstractCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Aggregates along the observation axis with the median, weighted or not, so that one outlying observation moves it little.

[`AggregateFeatures`](@ref) accepts it, because it aggregates the features and applies the metric after, so the result is a metric. [`AggregateDistances`](@ref) refuses it at construction, because an entrywise median of distance matrices need not satisfy the triangle inequality.

A quantile interpolates, so the aggregate need not be an element of the window. `Statistics.median(v, w)` is the `StatsBase` ``0.5``-quantile, not an order statistic. On the window `[0, 1, 2, 3]` under the weights `[1, 2, 3, 4]`, it returns `11/6`, which is strictly between the second and the third value. The interpolation does not always move the result off the window. The same window under the weights `[4, 3, 2, 1]` returns `1`, an element. The unweighted median of an even window is the mean of the two central values for the same reason.

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
 2. Reduce that series with `Statistics.median`, weighted by `w` when `w` is not `nothing`, to get the entry of the collapsed matrix.

# Related

  - [`AbstractCollapseAlgorithm`](@ref)
  - [`MeanCollapse`](@ref)
  - [`AggregateFeatures`](@ref)
"""
struct MedianCollapse <: AbstractCollapseAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all feature collapse algorithms.

A feature collapse algorithm reduces a window of time-varying features, `observations × assets × features`, to one `assets × assets` distance matrix. It is the `alg` field of [`FeatureDistance`](@ref). It has no effect on a 2-D feature matrix, because a static feature matrix has no observation axis to collapse. At `observations == 1`, every algorithm of the family gives the same result.

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
  - `A`: The mask of readable cells of the window, `observations × assets`. A cell is readable when the asset is active and every value column that the Feature Selector names holds data, observed or filled.

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

Reads each asset at its last readable row of the window, the most recent value that an investor had. A row is readable where the asset is active and each selected cell holds data, so a placeholder moves the read to an earlier row.

The rule reads an asset that delisted before the last row at the last row at which the asset was active. An asset with no active row in the window has no value to read.

# Related

  - [`AbstractLastObservationAlgorithm`](@ref)
  - [`LastRow`](@ref)
  - [`LastObservation`](@ref)
"""
struct LastActiveRow <: AbstractLastObservationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Discards the window and measures one row of each asset's features.

It is the default of [`FeatureDistance`](@ref). It is the cheapest member of the family, and its result depends on no choice of aggregation. `alg` names the row of each asset. [`LastRow`](@ref) reads the last row of the window. [`LastActiveRow`](@ref) reads the last row at which each asset is readable, where the asset is active and each selected cell holds data, not a placeholder. It is also the one member that names its rows before the stack exists. [`readable_window_rows`](@ref) returns the rows that `alg` can read, so the kernel stacks those rows of an Asset Panel alone.

# Algorithm

 1. Under [`LastRow`](@ref), or when every asset is readable at the last row, take the last slice of the observation axis, `view(Z, size(Z, 1), :, :)`. This is an `assets × features` matrix.
 2. Under [`LastActiveRow`](@ref), take for each asset the row of its last readable observation, to make an `assets × features` matrix.
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

`alg` aggregates each feature along the observation axis. The metric runs after the aggregation, so the result is a metric under both [`MeanCollapse`](@ref) and [`MedianCollapse`](@ref). No other member of the family accepts the median.

# Algorithm

 1. Resolve `w` against `Z` with [`collapse_weights`](@ref) to get a weight vector with one entry per observation, or `nothing`.
 2. Collapse the observation axis of `Z` with `alg` to get one `assets × features` matrix.
 3. Apply the metric to that matrix once, and apply the convention for a zero feature vector to the result.

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

The entry of the optimiser finds the first pair of assets with no shared active row, drops the asset of the pair with fewer active rows, and repeats until no pair is empty. On a tie, it drops the asset that comes later in the universe. The fit treats a dropped asset as a non-investable asset. It announces the asset, gives it a zero weight, and lists it on the Non-Investable Axis.

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

For a pair of assets with no shared active row, `alg` collapses the features of each asset over the active rows of that asset, under the weights `w` cut to those rows. The distance of the pair is then the distance of the two assets as if each asset held its aggregated features at every row of the window. Every other pair keeps the rule of its collapse.

The rule gives one scale to the whole matrix. Under [`AggregateDistances`](@ref), the distance of the pair is the metric of the two aggregated feature vectors, which is the value that each row of the window then gives. Under [`StackObservations`](@ref), the aggregated feature vector of each asset fills each row of the stack, so a metric of the Minkowski family gives the same distance, times the rescale of the stack at one shared row.

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

Measures each observation, then aggregates the distance matrices of the observations.

It makes one distance matrix per observation and combines them into one `assets × assets` matrix. It applies the metric once per observation, where [`AggregateFeatures`](@ref) applies it once. It adds each matrix into one buffer, and it does not keep the matrices of the observations.

It accepts [`MeanCollapse`](@ref) and refuses [`MedianCollapse`](@ref). A convex combination of metrics is a metric, but an entrywise median of metrics need not be one. The metric runs before the aggregation, so the convention for a zero feature vector applies at each observation. An asset whose feature vector is zero at some observations takes the convention at those observations alone.

A window of an Asset Panel can hold an inactive cell. The collapse then reads each pair of assets at the rows at which both assets are active, and it divides the weights of those rows by their sum. This is available-case estimation, the rule by which a Coverage Policy fits a covariance cell. A pair with no shared active row, or with a zero weight at each shared row, takes the rule in `pair`.

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

 1. Resolve `w` against `Z` with [`collapse_weights`](@ref) to get a weight vector with one entry per observation, or `nothing`.
 2. Allocate the accumulator `D` and the buffer `Dt` for one observation, both `assets × assets`, and set the weight total `sw` to zero.
 3. For each observation `t`, measure that slice of `Z` into `Dt`, and apply the convention for a zero feature vector to `Dt`. Then add `wt .* Dt` to `D`, and add `wt` to `sw`. The weight `wt` of the observation is `one(T)` when `w` is `nothing`.
 4. Divide `D` by `sw`. The result is the convex combination of the distance matrices of the observations.

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

Concatenates the window into one long feature vector per asset, so the collapse aggregates nothing.

It turns `observations × assets × features` into an `assets × (observations · features)` matrix, and applies the metric once. Two assets are near only when their whole trajectories agree. For the same reason, the observations with the largest magnitudes have the largest effect on the result. Standardise features of different scales before you use it.

In general, its result is different from the result of each other member of the family. At `observations == 1`, all the members give the same result.

A window of an Asset Panel can hold an inactive cell. For each pair of assets, the collapse then stacks the ``n`` rows at which both assets are active. [`stack_rescale`](@ref) rescales the distance of the pair from those rows to the ``T`` rows of the window. A metric that sums over the coordinates, such as the Minkowski family, `Distances.SqEuclidean` or `Distances.ChiSqDist`, takes a factor of ``T / n`` inside its power. A ratio metric, such as [`AngularDist`](@ref) or `Distances.CosineDist`, takes no factor. A metric with no rescale method refuses a pair that shares fewer rows than the window. A pair with no shared active row takes the rule in `pair`.

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

 1. Permute `Z` so that the asset axis is first: `(2, 1, 3)` at `dims = 1`, and `(3, 1, 2)` at `dims = 2`.
 2. Reshape the permuted array to `assets × (observations · features)`, one long feature vector per asset.
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

Applies a metric to the rows of a feature matrix to make a distance matrix.

A feature matrix describes the assets by their exposures, memberships, loadings or adjacencies, not by their returns. This estimator is a sibling of [`Distance`](@ref) and [`DistanceDistance`](@ref). These two read a correlation matrix. This estimator never reads one, and it works where the returns hold no useful data, or where no returns exist.

The estimator accepts each `Distances.SemiMetric`, and a user-defined one too. Every metric has a similarity, so no combination throws on this path. The remarks below are about the metric that a caller chooses, not about this type.

!!! warning "A metric is not automatically in the domain of the similarity"

    [`ComplementSimilarity`](@ref) is the default for each metric other than [`AngularDist`](@ref). Under it, a distance above `1` gives a negative similarity, and a distance above `2` gives a similarity below `-1`. [`plot_clusters`](@ref) clips such a similarity with no warning. The first limit is `1`, so a bounded metric can cross it too. `Distances.CosineDist` and `Distances.CorrDist` have a bound of `2`, and they cross `1` at every negative cosine or correlation. `Distances.Euclidean` has no bound, and it crosses `2` on features of large magnitude.

    This warning applies to this path alone. When this estimator is the `de` of a [`NetworkEstimator`](@ref), [`DBHT`](@ref) or [`LoGo`](@ref), its distance matrix goes to the PMFG path. There, the similarity field of that estimator applies, not `sim`. [`assert_similarity_domain`](@ref) refuses a distance above `1` under [`ComplementSimilarity`](@ref), and a non-finite distance under [`MaximumDistanceSimilarity`](@ref).

Every metric other than [`AngularDist`](@ref) and `Distances.CorrDist` changes when the feature vector of an asset changes scale. [`AngularDist`](@ref) is invariant to a positive scale of the feature vector of an asset, but it changes when one feature changes scale across the assets. Standardise features of different scales before use. `Distances.CorrDist` cannot measure a single feature. Each feature vector is then constant, and `Distances.CorrDist` leaves `NaN` against a constant feature vector that is not zero.

`Distances.Jaccard` is the general Ruzicka form over the non-negative reals, not the Jaccard index of two binary sets. On signed input it returns values up to `2`, with no error. `Distances.Jaccard`, `Distances.BrayCurtis` and `Distances.ChiSqDist` need a non-negative feature matrix. The constructor does not know the feature matrix, so [`assert_metric_domain`](@ref) checks it in the kernel.

## Selection of the columns

`sel` names the Panel Fields that the Feature Matrix stacks. `nothing` stacks the values of every Panel Field. This is correct while the panel holds features alone. When the panel holds other data too, `nothing` stacks that data into the Feature Matrix.

An entry of `sel` has one of five forms, and one vector can mix them:

  - `"industry"` is a Panel Field name. It selects the value columns of that Panel Field alone.
  - `"industry" => ["Tech", "Energy"]` keeps the levels or labels that it names, in that order.
  - `"industry" => "Tech"` keeps one level or label. A column label has this form.
  - `"exposures" => LabelGroup("momentum")` keeps each label of a tensor Panel Field whose group is `"momentum"`, in the order of the labels. See [`LabelGroup`](@ref).
  - `"mcap" => :observed` is the observed mask of the Panel Field, one `0`/`1` column.

An entry cannot be an integer. Every Panel Field, level and label has a name, so a position has nothing to index. To select a taxonomy, name the categorical Panel Field that holds it in the panel.

An entry can name a field, a level or a label that the panel does not hold. When `strict` is `true`, the entry throws. Otherwise, it warns, and the estimator drops the entry.

The order of `sel` is the order of the columns that the metric reads, so the caller sets it. [`feature_matrix`](@ref) stacks the matrix, and [`feature_labels`](@ref) names its columns, one selector entry per column.

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

Keywords correspond to the struct's fields. The constructor sets `sim` to [`default_similarity`](@ref) of `metric`, so the printed object shows the resolved similarity, and the distance kernel resolves nothing.

## Validation

  - [`assert_feature_selector`](@ref) checks `sel`. It is `nothing`, or a non-empty vector of distinct entries, each in one of the five admitted forms.

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
  - [`assert_metric_domain`](@ref): the check in the kernel that the feature matrix of the three restricted metrics is non-negative.
  - [`assert_feature_selector`](@ref): the check of `sel` at construction.
  - [`select_fields`](@ref): the one resolution of `sel` against an [`AssetPanel`](@ref).
  - [`feature_matrix`](@ref): the function that stacks the Feature Matrix.
  - [`feature_labels`](@ref): one selector entry per column of the stacked matrix.
  - [`DBHT`](@ref): has a `sim` field of its own, with the same name, the same type and the same function. When both are set, the field of `DBHT` applies, because [`clusterise`](@ref) writes over the similarity matrix after [`cor_and_dist`](@ref) returns.
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

Assert that `Z` lies in the domain of `metric`. The fallback does nothing. Most metrics accept each finite real input, and a non-negativity check on every metric would refuse signed factor loadings and the default metric of [`FeatureDistance`](@ref).

`Distances.Jaccard`, in its Ruzicka form, `Distances.BrayCurtis` and `Distances.ChiSqDist` are the exceptions. Their domain is the non-negative reals. The check is most important for `Distances.Jaccard`, because it gives no error on signed input. It returns values up to `2`, and a clustering routine reads them as distances.

# Algorithm

 1. Select the method by the type of `metric`. The three metrics above share one method. Every other metric reaches the `Distances.SemiMetric` method, which does nothing and returns.
 2. On the method of the three metrics, check that `Z` is non-negative with [`assert_nonneg`](@ref). It raises a `DomainError` that names `sym` when an entry is negative.

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

Validate a feature matrix at the entry point of [`distance`](@ref) and [`cor_and_dist`](@ref). `dims` selects a valid axis, `Z` is not empty, every entry is finite, and `Z` lies in the domain of the metric.

The checks read every cell of `Z`. On a window that holds an unreadable cell, [`distance`](@ref) gives it the copy from [`blank_unreadable`](@ref), which holds a zero at each such cell. So the checks find a fault only in the readable cells, which are the cells that the collapse reads.

The check refuses a non-finite entry, because no metric makes a usable distance from it. The Minkowski family gives `Inf`, the cosine family gives `NaN`, and a clustering cannot use either. The check accepts a degenerate input that a metric can measure. A zero feature vector has a documented convention, which [`AngularDist`](@ref) states, and a duplicate or constant feature is valid input.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. Check that `Z` is non-empty with [`assert_nonempty`](@ref).
 3. Check that every entry of `Z` is finite with [`assert_all_finite`](@ref).
 4. Check that `Z` lies in the domain of `de.metric` with [`assert_metric_domain`](@ref).

# Arguments

  - `de`: Feature distance estimator, read for its `metric`.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Validation

  - $(val_dict[:dims])
  - `!isempty(Z)`.
  - `all(isfinite, Z)`.
  - `Z` lies in the domain of `de.metric`. See [`assert_metric_domain`](@ref).

# Returns

  - `nothing`.

# Related

  - [`FeatureDistance`](@ref)
  - [`assert_metric_domain`](@ref)
  - [`assert_dims`](@ref)
  - [`assert_nonempty`](@ref)
  - [`assert_all_finite`](@ref)
  - [`blank_unreadable`](@ref)
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

Mark the assets whose feature vector is all zero, in the layout that `dims` states.

# Algorithm

 1. Select the asset axis from `dims`: the rows of `Z` at `dims = 1`, and its columns at `dims = 2`.
 2. Test the feature vector of each asset with `all(iszero, ...)`, to get one entry of the mask per asset.

# Arguments

  - $(arg_dict[:Z])
  - $(arg_dict[:dims])

# Returns

  - `z::Vector{Bool}`: Mask with one entry per asset, `true` where the feature vector of that asset is all zero.

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

Apply the convention for a zero feature vector to `D` in place. Two zero vectors are at distance `0`, and a zero vector and a non-zero vector are at distance `1`.

The function rewrites only the entries that the metric left as `NaN`. A zero feature vector is valid input, so no check at construction can find it. A metric that divides by a norm cannot measure it. The cosine family gives `NaN` against every vector, and `Distances.Jaccard` and `Distances.BrayCurtis` give `NaN` between two zero vectors. The Minkowski family puts a zero vector at the origin and measures it. Because the function rewrites only `NaN` entries, it corrects the first metrics and leaves the Minkowski family unchanged.

This convention keeps ``S = \\cos(\\pi D)`` true at every entry. With it, [`AngularSimilarity`](@ref) gives `+1` between two zero vectors and `-1` between a zero vector and a non-zero vector, with a diagonal of ones. `Distances.pairwise` always writes an exact zero diagonal, so the function does not change the diagonal.

# Algorithm

 1. Make the zero mask `z` of `Z` with [`zero_feature_vectors`](@ref).
 2. Return `D` unchanged when the mask marks no asset. Most inputs take this step.
 3. Otherwise, read each off-diagonal entry of `D`. Rewrite it only when the metric left it as `NaN` and the mask marks at least one of its two assets. Write `zero(T)` when the mask marks both, and `one(T)` when it marks exactly one.

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

Turn a 2-D feature matrix into a distance matrix. Each entry point of [`FeatureDistance`](@ref) uses this kernel, and three of the collapse algorithms differ only in the matrix that they give it. [`AggregateDistances`](@ref) does not call it. It does the same two steps once per observation, into one buffer that it uses again, and aggregates the results.

# Algorithm

 1. Apply `metric` to every pair of assets of `Z` with `Distances.pairwise`, along the axis that `dims` names, to get `D`.
 2. Apply the convention for a zero feature vector to `D` in place with [`patch_zero_feature_vectors!`](@ref).

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

Aggregate a window of time-varying features along its first axis, the observation axis. The result is a matrix with the two last axes of `Z`. [`AggregateFeatures`](@ref) calls it.

The element type of the result comes from the aggregate, not from the window. The mean and the median of a window of integers can be fractions. A window with an even count of observations can have a fractional median even when each value in it is a whole number. For this reason, both routes make their result from the values that they compute.

# Algorithm

The [`MeanCollapse`](@ref) route:

 1. Reduce the first axis of `Z` with `Statistics.mean`, weighted by `w` when `w` is not `nothing`.
 2. Drop the reduced axis.

The [`MedianCollapse`](@ref) route:

 1. For each asset `j` and each feature `k`, take the observation series `view(Z, :, j, k)`.
 2. Reduce that series with `Statistics.median`, weighted by `w` when `w` is not `nothing`, to get one entry of the result.

# Arguments

  - `alg`: Collapse algorithm, which aggregates along the observation axis.
  - $(arg_dict[:Z])
  - `w`: Resolved observation weights, one entry per observation, or `nothing` for an unweighted collapse.

# Returns

  - `Zc::Matrix{<:Number}`: Collapsed feature matrix, with the two last axes of `Z`.

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

Reshape a window of time-varying features into an `assets × (observations · features)` matrix. Its rows are the assets, on whichever of the two last axes `dims` puts them.

# Algorithm

 1. Permute `Z` so that the asset axis is first: `(2, 1, 3)` at `dims = 1`, and `(3, 1, 2)` at `dims = 2`. Both permutations put the observation axis second and the feature axis third.
 2. Reshape the permuted array to `assets × (observations · features)`, one long feature vector per asset.

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

The function first reshapes `Z` to an `observations × (assets · features)` matrix. The documented interface of [`get_observation_weights`](@ref) is `VecNum` and `MatNum`, and a 3-D array matches neither. Without the reshape, a correct `MatNum` method that a user adds never runs. The caller does not check for `nothing`. [`get_observation_weights`](@ref) raises [`ObservationWeightsError`](@ref) when a [`DynamicAbstractWeights`](@ref) cannot resolve, so `nothing` here means only that the caller asked for no weights.

Observation weights for a cross-validation need a [`DynamicAbstractWeights`](@ref). It resolves against the `Z` that it receives, so its length is the length of each fold. A static `AbstractWeights` has the length that it had at construction, in each fold. Before the length check, [`AggregateDistances`](@ref) read a longer one by position. It gave the oldest weights to the newest observations, with no bounds error. A shorter one raised a `BoundsError` with no explanation. The length check now raises a clear error for both.

# Algorithm

 1. Reshape `Z` to `observations × (assets · features)`, so that the observation axis is the first axis of a matrix.
 2. Resolve `w` against that matrix with [`get_observation_weights`](@ref), along its first axis.
 3. Check the resolved length against the count of observations of `Z`, unless the resolution gave `nothing`.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:Z])

# Validation

  - `length(w) == size(Z, 1)` after the resolution. Raises a `DimensionMismatch` that names both lengths.

# Returns

  - `w::Option{<:VecNum}`: Resolved observation weights, one entry per observation, or `nothing` when the caller asked for no weights.

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

The type of `de.alg` selects one of four methods. The docstring of each type states its method in full. In short:

 1. [`LastObservation`](@ref): pass the last slice of the observation axis to the 2-D kernel.
 2. [`StackObservations`](@ref): pass the stacked matrix from [`stack_observations`](@ref) to the 2-D kernel, along its first axis.
 3. [`AggregateFeatures`](@ref): resolve the weights with [`collapse_weights`](@ref), collapse the window with [`collapse_features`](@ref), and pass the collapsed matrix to the 2-D kernel.
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

The 2-D method collapses nothing. A static feature matrix has no observation axis, so the collapse algorithm reads its one row. Its `amsk` is `1 × assets`. The mask marks the assets whose every value column holds data, and the method refuses an asset that the mask marks `false`. The 3-D method dispatches on the collapse algorithm. An asset whose feature vector is all zero takes the convention that [`patch_zero_feature_vectors!`](@ref) states.

The 3-D method reads the mask `amsk` of the window. A collapse then reads each asset, or each pair of assets, at its own active rows, as each member of [`AbstractFeatureCollapseAlgorithm`](@ref) states. A window with no inactive cell gives the same result as no mask. The checks read the same cells as the collapse, so the value of a cell that `amsk` marks `false` changes neither the distance nor a refusal.

# Algorithm

 1. Check `dims` with [`assert_dims`](@ref).
 2. On the 2-D method, check the one-row mask with [`static_window`](@ref) and [`assert_feature_readable`](@ref). Validate `Z` with [`assert_feature_matrix`](@ref), and pass `de.metric` and `Z` to the kernel.
 3. On the 3-D method, check the mask with [`active_window`](@ref), and check that the collapse can read each asset with [`assert_feature_readable`](@ref).
 4. With no inactive cell, validate `Z` with [`assert_feature_matrix`](@ref). Pass `de` and `Z` to [`feature_distance`](@ref), which selects the method that `de.alg` names.
 5. Otherwise, put a zero at each inactive cell of a copy of `Z` with [`blank_unreadable`](@ref). Validate the copy with [`assert_feature_matrix`](@ref), and pass it to [`active_feature_distance`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])
  - `amsk`: The mask of readable cells of the window, `observations × assets` on the 3-D method and `1 × assets` on the 2-D method, or `nothing` when every cell is readable.
  - `nx`: The asset names that a refusal quotes, or `nothing` to quote positions.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - $(val_dict[:dims])
  - `amsk` is `observations × assets` on the 3-D method and `1 × assets` on the 2-D method, and the collapse can read each asset. See [`assert_feature_readable`](@ref).
  - `!isempty(Z)`.
  - Each cell of `Z` that `amsk` marks readable is finite and lies in the domain of `de.metric`. See [`assert_metric_domain`](@ref). The check reads no cell that `amsk` marks `false`.
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
    assert_feature_readable(de.alg, static_window(amsk, Z, dims), nx)
    assert_feature_matrix(de, Z, dims)
    return feature_distance(de.metric, Z, dims)
end
function distance(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, amsk = nothing,
                  nx = nothing, kwargs...)
    assert_dims(dims)
    A = active_window(amsk, Z, dims)
    assert_feature_readable(de.alg, A, nx)
    return if isnothing(A)
        assert_feature_matrix(de, Z, dims)
        feature_distance(de, Z, dims)
    else
        Zr = blank_unreadable(Z, A, dims)
        assert_feature_matrix(de, Zr, dims)
        active_feature_distance(de, (; Z = Zr, dims = dims, A = A, nx = nx))
    end
end
"""
    cor_and_dist(de::FeatureDistance, Z::MatNum; dims::Int = 1, kwargs...)
    cor_and_dist(de::FeatureDistance, Z::Arr3Num; dims::Int = 1, kwargs...)

Compute the similarity and distance matrices from a feature matrix.

The similarity is `distance_to_similarity(de.sim; D = D)`, which the call computes from the distance matrix that it returns. `S` and `D` come from one measurement, not from two independent estimates. Under [`AggregateDistances`](@ref), the similarity comes from the aggregated distance, so the convention for a zero feature vector holds in `S` too, because ``\\mathrm{mean}(\\cos(\\pi D_{t})) \\neq \\cos(\\pi\\,\\mathrm{mean}(D_{t}))``.

# Algorithm

 1. Compute the distance matrix `D` with [`distance`](@ref), which also validates `Z` and `dims`.
 2. Transform `D` with [`distance_to_similarity`](@ref) under `de.sim` to get `S`.

# Arguments

  - `de`: Feature distance estimator.
  - $(arg_dict[:Z])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - The checks of [`distance`](@ref): the mask, `!isempty(Z)`, and each readable cell of `Z` is finite and lies in the domain of `de.metric`. See [`assert_metric_domain`](@ref).

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

Name the observation rows that a collapse algorithm reads, so that the kernel stacks those rows alone.

The kernel stacks a time-varying Feature Matrix from an Asset Panel, and then collapses it along its observation axis. A collapse that reads one row does not need the other rows.

  - [`LastObservation`](@ref) under [`LastRow`](@ref) reads the last row, so it names that row, and the kernel stacks a window of one observation. The kernel then reads a lifted static Panel Field, whose values are a [`RepeatedLeading`](@ref), once and not once per observation. Under the default collapse, the work of the kernel is the `assets × features` slice that it measures.
  - Under [`LastActiveRow`](@ref), it names the rows from the earliest last active row of an asset to the last row. These rows hold the last active row of each asset that has one.
  - Every other member returns `Colon()`, which names every row. The two aggregates resolve their weights against the stacked window, as [`collapse_weights`](@ref) states, so the rows that they could cut are not known before the stack exists. [`StackObservations`](@ref) reads the whole stack by definition.

A static panel has no observation axis, so every member returns `Colon()` on it, and [`LastObservation`](@ref) does too.

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

  - `rows`: A range of observations, or `Colon()`. It is the value of the `rows` keyword of [`feature_matrix`](@ref).

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

Stack the Feature Matrix that a [`FeatureDistance`](@ref) measures, from the panel that its `ape` field resolves.

One call resolves the panel. Under a `nothing` producer, `asset_panel(de.ape, pr, rd, X)` returns the panel that the [`ReturnsResult`](@ref) holds. Otherwise, it builds one. This method returns the stack of [`feature_window`](@ref), which is the stack that the kernel measures. The stack holds the columns that `de.sel` names, over the observation rows that `de.alg` reads, with a zero at each cell that holds a placeholder. [`feature_labels`](@ref) resolves the panel through the same call, so the labels that a caller rebuilds name the columns that the kernel measured.

The panel method of [`feature_matrix`](@ref) writes `NaN` at a placeholder by default. It gives the same matrix at `placeholder = 0`, over the same rows.

The collapse algorithm names the rows through [`readable_window_rows`](@ref). Under [`LastObservation`](@ref) with [`LastRow`](@ref), a time-varying panel stacks its last observation alone, `1 × assets × features`, which is the slice that the collapse measures. With [`LastActiveRow`](@ref), it stacks the rows from the earliest last readable row of an asset to the last row. Every other collapse stacks every observation.

# Algorithm

 1. Stack the Feature Matrix and the mask of its readable cells with [`feature_window`](@ref).
 2. Return the Feature Matrix, and drop the mask.

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
  - [`feature_window`](@ref)
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

Stack the Feature Matrix that a [`FeatureDistance`](@ref) measures, with the mask of the cells that it can read in the stacked rows.

It returns the same stack as [`feature_matrix`](@ref), and the mask of that stack. A cell is readable where the asset is active and every value column of `de.sel` holds data. Data is a value that the raw input held, or a value that a fill policy wrote. The function cuts the mask to the rows that the collapse reads, so the mask has one row per observation of the stack. A static panel has one row, the cells of the assets. Its mask is `nothing` when every cell holds data, and a `1 × assets` matrix otherwise. A time-varying panel whose every cell is readable gives the active mask, and the kernel drops that mask when it is all `true`.

The stack holds a zero at a cell that holds a placeholder, not the placeholder that the panel stores. The mask leaves that cell out, so no step reads the zero, and the stack does not change with the placeholder.

# Algorithm

 1. Resolve the panel with [`asset_panel`](@ref), and resolve `de.sel` against it with [`select_fields`](@ref).
 2. Name the rows that the collapse reads, and cut the mask of readable cells to them, with [`feature_window_mask`](@ref).
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
    #! keeps the stack the same for every placeholder that the panel stores.
    return feature_stack(pnl, cols; rows = rows, placeholder = 0), A
end
"""
    window_activity(amsk, rows)

Cut a mask of the cells of an Asset Panel to the rows that a Feature Matrix stacks. A static panel has no observation axis. Its mask, one entry per asset, becomes a window of one row, `1 × assets`, or `nothing` when every entry is `true`. A `nothing` mask gives `nothing`.

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

Give the asset names that a refusal of a [`FeatureDistance`](@ref) quotes. These are the names of the returns data `rd`, or the names of `pr` when `pr` is a `ReturnsResult` and `rd` is `nothing`. Otherwise, the function returns `nothing`, and the refusal quotes positions.

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

Name each column of the Feature Matrix that a [`FeatureDistance`](@ref) measures.

It resolves the panel and the selector as [`feature_matrix`](@ref) does, so the two always agree. A label is the selector entry that selects exactly that column. So the vector of labels is a selector that rebuilds the columns of the matrix, and it tells a caller what the kernel measured.

The kernel never calls it. No clustering or phylogeny result records the labels, because the estimator, `pr` and `rd` give them without a distance computation. To get them, call `feature_labels(de, res.pr, rd, rd.X)` with the arguments that the optimiser received.

# Algorithm

 1. Resolve the panel with [`asset_panel`](@ref).
 2. Label it with [`feature_labels`](@ref), under `de.sel` and `de.strict`.

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

Each consumer in the clustering and network estimators calls `cor_and_dist(de, ce, X; …)` or `distance(de, pl, X; …)`, with a covariance estimator and a returns matrix. [`logo!`](@ref) passes a similarity matrix in place of the covariance estimator, so the type of the second positional argument is `::Any`. [`FeatureDistance`](@ref) does not read that argument. It reads `X`, because a producer measures it.

The prior result and the returns data arrive as the keywords `pr` and `rd`. [`feature_window`](@ref) resolves the panel from them and from `de.ape`. A caller that takes a prior result passes both. Preselection passes `rd` alone.

This method ignores `dims` and calls the kernel with `dims = 1`. The `dims` of the caller describes the returns matrix `X`, and the stacked Feature Matrix always has the layout of `dims = 1`. `dims` has a meaning only at the entry point for a raw matrix, `distance(de, Z; dims)`.

# Algorithm

 1. Stack the Feature Matrix with [`feature_window`](@ref), which resolves the panel and cuts it to `de.sel` and to the observation rows that `de.alg` reads. It also returns the mask of the readable cells of those rows. A cell is readable when the asset is active and every value column of `de.sel` holds data, observed or filled.
 2. Compute the distance matrix `D` of that stack with the two-argument method at `dims = 1`, with that mask and the asset names from [`feature_asset_names`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - The second positional argument: ignored. It exists so that this method matches the signature that every consumer calls.
  - $(arg_dict[:X_sub]) A producer reads it.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `kwargs...`: Additional keyword arguments (ignored), `dims` among them.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The stacked Feature Matrix passes [`assert_feature_matrix`](@ref) at `dims = 1` at its readable cells. It is not empty, and each cell that the mask marks readable is finite and lies in the domain of `de.metric`. The value of an unreadable cell causes no refusal.
  - The collapse can read each asset at the readable rows of the window. See [`assert_feature_readable`](@ref). A pair with no shared readable row takes the `pair` rule of its collapse. A fit drops an unreadable asset at its entry with [`feature_readable_mask`](@ref), and this call refuses it.

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

[`clusterise`](@ref) and the network estimators call this method. It reads its arguments as the three-argument [`distance`](@ref) method does. It ignores the second positional argument and `dims`, it passes `X` to a producer, and it resolves the panel from `pr` and `rd`.

# Algorithm

 1. Stack the Feature Matrix with [`feature_window`](@ref), which resolves the panel and cuts it to `de.sel` and to the observation rows that `de.alg` reads. It also returns the mask of the readable cells of those rows. A cell is readable when the asset is active and every value column of `de.sel` holds data, observed or filled.
 2. Compute the similarity matrix `S` and the distance matrix `D` of that stack with the two-argument method at `dims = 1`, with that mask and the asset names from [`feature_asset_names`](@ref).

# Arguments

  - `de`: Feature distance estimator.
  - The second positional argument: ignored. It exists so that this method matches the signature that every consumer calls.
  - $(arg_dict[:X_sub]) A producer reads it.
  - $(arg_dict[:pr_rr])
  - $(arg_dict[:rd])
  - `kwargs...`: Additional keyword arguments (ignored), `dims` among them.

# Validation

  - $(val_dict[:fd_panel])
  - $(val_dict[:fd_strict])
  - The stacked Feature Matrix passes [`assert_feature_matrix`](@ref) at `dims = 1` at its readable cells. It is not empty, and each cell that the mask marks readable is finite and lies in the domain of `de.metric`. The value of an unreadable cell causes no refusal.
  - The collapse can read each asset at the readable rows of the window. See [`assert_feature_readable`](@ref). A pair with no shared readable row takes the `pair` rule of its collapse. A fit drops an unreadable asset at its entry with [`feature_readable_mask`](@ref), and this call refuses it.

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
