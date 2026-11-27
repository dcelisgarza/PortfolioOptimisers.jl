"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that join the distance matrices of the parts of a [`BlendDistance`](@ref) into one.

A rule reads the weighted distance matrices of the parts and returns one distance matrix of the same size. [`QuadraticBlend`](@ref) is the default. Each rule keeps the blend a metric when each part is a metric, because each is a weighted power mean with an exponent of at least `1`.

# Interfaces

A blend rule of your own subtypes `AbstractDistanceBlendAlgorithm`, goes into the field `alg` of a [`BlendDistance`](@ref), and adds this method:

  - `blend_distances(alg::MyBlend, w::VecNum, Ds::AbstractVector{<:MatNum}) -> MatNum`: Join the distance matrices `Ds` of the parts with a positive weight, under their normalised weights `w`, into one distance matrix. `w` and `Ds` have one entry for each such part, in the order of the parts.

[`default_similarity`](@ref) of the rule gives the default `sim` of the blend. The fallback is [`ComplementSimilarity`](@ref). A rule whose inverse is not `1 - D` adds a method.

# Related

  - [`BlendDistance`](@ref)
  - [`QuadraticBlend`](@ref)
  - [`ArithmeticBlend`](@ref)
  - [`PowerBlend`](@ref)
  - [`blend_distances`](@ref)
  - [`default_similarity`](@ref)
"""
abstract type AbstractDistanceBlendAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Joins the distance matrices of the parts by the root of the weighted sum of their squares.

It is the default rule of [`BlendDistance`](@ref). On chord parts it is the chord distance of the weighted mean of their cosines or correlations, so [`ChordSimilarity`](@ref) recovers that mean exactly. [`BlendDistance`](@ref) states the identity.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= \\sqrt{\\sum_{k} w_{k} D_{k,\\,i,\\,j}^{2}}\\,,
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: The blended distance between assets ``i`` and ``j``.
  - ``D_{k,\\,i,\\,j}``: The distance between assets ``i`` and ``j`` under part ``k``.
  - ``w_{k}``: The normalised weight of part ``k``. The weights sum to `1`.

It is [`PowerBlend`](@ref) at ``p = 2``. That spelling is refused, so the setting has one name.

# Related

  - [`AbstractDistanceBlendAlgorithm`](@ref)
  - [`ArithmeticBlend`](@ref)
  - [`PowerBlend`](@ref)
  - [`BlendDistance`](@ref)
  - [`ChordSimilarity`](@ref)
"""
struct QuadraticBlend <: AbstractDistanceBlendAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Joins the distance matrices of the parts by their weighted sum.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= \\sum_{k} w_{k} D_{k,\\,i,\\,j}\\,,
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: The blended distance between assets ``i`` and ``j``.
  - ``D_{k,\\,i,\\,j}``: The distance between assets ``i`` and ``j`` under part ``k``.
  - ``w_{k}``: The normalised weight of part ``k``. The weights sum to `1`.

It is [`PowerBlend`](@ref) at ``p = 1``. That spelling is refused, so the setting has one name.

# Related

  - [`AbstractDistanceBlendAlgorithm`](@ref)
  - [`QuadraticBlend`](@ref)
  - [`PowerBlend`](@ref)
  - [`BlendDistance`](@ref)
"""
struct ArithmeticBlend <: AbstractDistanceBlendAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Joins the distance matrices of the parts by their weighted power mean, with an exponent `p`.

# Mathematical definition

```math
\\begin{align}
D_{i,\\,j} &= \\left(\\sum_{k} w_{k} D_{k,\\,i,\\,j}^{p}\\right)^{1/p}\\,,\\\\
D_{i,\\,j} &= \\max_{k \\,:\\, w_{k} > 0} D_{k,\\,i,\\,j}\\quad \\text{at } p = \\infty\\,,
\\end{align}
```

Where:

  - ``D_{i,\\,j}``: The blended distance between assets ``i`` and ``j``.
  - ``D_{k,\\,i,\\,j}``: The distance between assets ``i`` and ``j`` under part ``k``.
  - ``w_{k}``: The normalised weight of part ``k``. The weights sum to `1`.
  - ``p``: The exponent, the field `p`.

At ``p = \\infty`` the weights set only which parts enter, not how much each part counts. An exponent of at least `1` keeps the blend a metric when each part is a metric, by the Minkowski inequality.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PowerBlend(; p::Number) -> PowerBlend

The keyword `p` has no default.

## Validation

  - `1 <= p`. `NaN` is refused.
  - `p != 1`. The message names [`ArithmeticBlend`](@ref).
  - `p != 2`. The message names [`QuadraticBlend`](@ref).

# Examples

```jldoctest
julia> PowerBlend(; p = 3)
PowerBlend
  p ┴ Int64: 3
```

# Related

  - [`AbstractDistanceBlendAlgorithm`](@ref)
  - [`QuadraticBlend`](@ref)
  - [`ArithmeticBlend`](@ref)
  - [`BlendDistance`](@ref)
"""
@concrete struct PowerBlend <: AbstractDistanceBlendAlgorithm
    """
    $(field_dict[:bdp])
    """
    p
    function PowerBlend(p::Number)::PowerBlend
        @argcheck(one(p) <= p,
                  DomainError(p,
                              "`PowerBlend.p` is $p, but it must be at least `1`, so that the blend of metrics is a metric. State `p` in `(1, Inf]`, other than `2`."))
        @argcheck(!isone(p),
                  DomainError(p,
                              "`PowerBlend.p` is `1`, which is the weighted sum. State `ArithmeticBlend()`, so that the setting has one spelling."))
        @argcheck(p != 2,
                  DomainError(p,
                              "`PowerBlend.p` is `2`, which is the root of the weighted sum of squares. State `QuadraticBlend()`, so that the setting has one spelling."))
        return new{typeof(p)}(p)
    end
end
function PowerBlend(; p::Number)::PowerBlend
    return PowerBlend(p)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that change the units of the distance matrix of a part of a [`BlendDistance`](@ref) before the blend.

A part reads its distance in its own units. A part whose range is not ``[0,\\,1]`` then outweighs the other parts. A scale rule maps each part to a common range.

# Interfaces

A scale rule of your own subtypes `AbstractDistanceScaleAlgorithm`, goes into the field `scale` of a [`BlendDistance`](@ref), alone or as an entry of a vector, and adds this method:

  - `scale_distance(alg::MyScale, D::MatNum) -> MatNum`: The distance matrix `D` of one part in the new units. The method must not write to `D`, because a part can return a matrix that its caller holds.

# Related

  - [`BlendDistance`](@ref)
  - [`MaximumScale`](@ref)
  - [`scale_distance`](@ref)
"""
abstract type AbstractDistanceScaleAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides the distance matrix of a part by its largest finite entry.

The result lies in ``[0,\\,1]`` at every finite entry. A matrix whose largest finite entry is zero stays as it is. The maximum ignores a `NaN` and an infinite entry, so the `NaN` row of an asset that the part cannot read stays a `NaN` row, and an infinite entry stays infinite.

The maximum comes from the matrix of the fit. So each fold of a cross-validation and each cluster of a [`NestedClustered`](@ref) gets a scale of its own for the part. On a chord part, the scale also breaks the chord identity that [`BlendDistance`](@ref) states. Scale the unbounded parts alone, with a vector in `scale`.

# Mathematical definition

```math
\\begin{align}
\\tilde{D}_{k,\\,i,\\,j} &= \\dfrac{D_{k,\\,i,\\,j}}{\\max_{a,\\,b \\,:\\, D_{k,\\,a,\\,b} < \\infty} D_{k,\\,a,\\,b}}\\,,
\\end{align}
```

Where:

  - ``\\tilde{D}_{k,\\,i,\\,j}``: The scaled distance between assets ``i`` and ``j`` under part ``k``.
  - ``D_{k,\\,i,\\,j}``: The distance between assets ``i`` and ``j`` under part ``k``.

# Related

  - [`AbstractDistanceScaleAlgorithm`](@ref)
  - [`BlendDistance`](@ref)
  - [`scale_distance`](@ref)
"""
struct MaximumScale <: AbstractDistanceScaleAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Blends the distance matrices of several distance estimators into one Distance Matrix.

Each part computes its own distance matrix from the same covariance estimator, data, prior result and returns data. Each part keeps its own panel, metric, collapse and rule for the assets that it can read. A scale rule can change the units of each part. The blend rule then joins the parts under their weights. So the estimator joins a distance of the returns with a [`FeatureDistance`](@ref) of a taxonomy, or two feature distances over different Asset Panels, with no merge of the panels.

A part is any distance estimator: a [`Distance`](@ref), a [`DistanceDistance`](@ref), a [`FeatureDistance`](@ref), or another `BlendDistance`. A blend of one part gives the distance of that part.

!!! warning "A part whose range is not [0, 1] outweighs the other parts"

    With `scale = nothing`, the blend reads each part in its own units. A [`FeatureDistance`](@ref) under `Distances.Euclidean` or another unbounded metric, a [`Distance`](@ref) under [`LogDistance`](@ref), and a [`DistanceDistance`](@ref) can give distances far above `1`. Such a part then sets the blend at any weight. The bounded forms are the canonical [`Distance`](@ref) of a correlation, [`ChordDist`](@ref) and [`AngularDist`](@ref). Give an unbounded part a scale rule, such as [`MaximumScale`](@ref), with a vector in `scale` that keeps `nothing` for the bounded parts.

!!! note "What a blend gives out of sample"

    A walk-forward measurement compared blends of the canonical [`Distance`](@ref) and a taxonomy [`FeatureDistance`](@ref) with each part alone. The blends were [`QuadraticBlend`](@ref) at the weights `(0.25, 0.75)`, `(0.5, 0.5)` and `(0.75, 0.25)`, and [`ArithmeticBlend`](@ref) at `(0.5, 0.5)`. The data were 20 stocks of the S&P 500 over 1990 to 2022 and 36 stocks of the NYSE (O) set over 1962 to 1984. The optimisers were [`HierarchicalRiskParity`](@ref), [`HierarchicalEqualRiskContribution`](@ref), [`SchurComplementHierarchicalRiskParity`](@ref) and [`NestedClustered`](@ref). No out-of-sample gain over its parts on these data sets. No blend gave a lower risk or a higher Sharpe ratio than its best part, and the rule of the blend changed no result.

    A second measurement added [`WalkDistance`](@ref) as a part, over the customer and the supplier graphs of the yearly input-output tables of the United States. The data were 49 industry portfolios over 1970 to 2026 and the two sets above, with the same optimisers. The blends joined the walk with the canonical distance at the same weights, with the taxonomy, and with both. No blend with the walk lowered the volatility by 2 %, or raised the Sharpe ratio by 0.1, over its best part for two of the four optimisers on any data set. The blend of the walk and the taxonomy cut the volatility of [`HierarchicalEqualRiskContribution`](@ref) by 2.5 % and 3.8 % against its best part on two sets, and raised it by 10 % on the third.

    The weight of a part is not its share of the structure. A part with a larger spread of values, such as a taxonomy, set the tree at every weight from `0.25` to `0.75`. The blend kept the tree, the leaf order and the cut of neither part, and the cut followed the taxonomy.

# The chord identity

On chord parts, [`QuadraticBlend`](@ref) gives the numbers of one Feature Matrix that stacks the parts. Let each part be the chord distance ``D_{k} = \\sqrt{(1 - c_{k}) / 2}`` of a cosine or a correlation ``c_{k}``. Then

```math
\\begin{align}
D_{i,\\,j}^{2} &= \\sum_{k} w_{k} \\dfrac{1 - c_{k,\\,i,\\,j}}{2} = \\dfrac{1 - \\bar{c}_{i,\\,j}}{2}\\,, & \\bar{c}_{i,\\,j} &= \\sum_{k} w_{k} c_{k,\\,i,\\,j}\\,,
\\end{align}
```

because the normalised weights sum to `1`. So ``D`` is the chord distance of the weighted mean ``\\bar{c}``, and the default [`ChordSimilarity`](@ref) returns ``\\bar{c}`` exactly. ``\\bar{c}`` is also the cosine of the stacked vectors ``(\\sqrt{w_{1}}\\,\\boldsymbol{u}_{1},\\, \\sqrt{w_{2}}\\,\\boldsymbol{u}_{2},\\, \\ldots)`` of the unit-scaled feature vectors ``\\boldsymbol{u}_{k}`` of the parts. The canonical [`Distance`](@ref) of a correlation and [`ChordDist`](@ref) are chord parts. A scale rule or another blend rule breaks the identity. Another blend rule then takes [`ComplementSimilarity`](@ref) by default, and a scale rule keeps the default of the parts, which is not exact.

The same identity holds for orthant chord parts, ``D_{k} = \\sqrt{1 - c_{k}}``, such as [`WalkDistance`](@ref): ``D^{2} = 1 - \\bar{c}``, and [`OrthantChordSimilarity`](@ref) returns ``\\bar{c}``. So a blend whose parts with a positive weight all hold [`OrthantChordSimilarity`](@ref) takes it by default. A blend of a chord part and an orthant chord part has no exact inverse, and it takes [`ChordSimilarity`](@ref). State `sim` for such a blend.

# Mathematical definition

```math
\\begin{align}
D &= \\beta\\left(\\left\\{w_{k},\\, s_{k}(D_{k})\\right\\}_{k \\,:\\, w_{k} > 0}\\right)\\\\
S &= \\sigma\\left(D\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:D_mat_dist])
  - ``S``: The similarity matrix.
  - ``D_{k}``: The distance matrix of part ``k``.
  - ``s_{k}``: The scale rule of part ``k``, the identity when it is `nothing`.
  - ``w_{k}``: The normalised weight of part ``k``. The weights sum to `1`.
  - ``\\beta``: The blend rule, `alg`.
  - ``\\sigma``: The similarity transformation, `sim`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BlendDistance(;
        des::AbstractVector{<:AbstractDistanceEstimator},
        w::Option{<:VecNum} = nothing,
        alg::AbstractDistanceBlendAlgorithm = QuadraticBlend(),
        scale::Union{Nothing, AbstractDistanceScaleAlgorithm,
                     AbstractVector{<:Option{<:AbstractDistanceScaleAlgorithm}}} = nothing,
        sim::AbstractSimilarityMatrixAlgorithm = blend_similarity(alg, des, w)
    ) -> BlendDistance

Keywords correspond to the struct's fields. The keyword `des` has no default. The constructor sets `sim` to [`blend_similarity`](@ref) of `alg`, `des` and `w`, so the printed object shows the resolved similarity. Under [`QuadraticBlend`](@ref), parts with a positive weight that hold one `sim` of the chord family give that `sim`, so a blend of [`WalkDistance`](@ref) parts takes [`OrthantChordSimilarity`](@ref). Every other blend takes the similarity of its rule. The field `w` keeps the weights as the caller states them, and the kernel divides them by their sum.

## Validation

  - `!isempty(des)`. Raises an [`IsEmptyError`](@ref).
  - If `w` is not `nothing`: `!isempty(w)`, `length(w) == length(des)`, every entry is finite and non-negative, and at least one entry is positive.
  - If `scale` is a vector: `length(scale) == length(des)`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `des`: Recursively updated through [`factory`](@ref), part by part.

# Examples

```jldoctest
julia> BlendDistance(; des = [Distance(; alg = CanonicalDistance()), FeatureDistance()])
BlendDistance
    des ┼ 2-element Vector{PortfolioOptimisers.AbstractDistanceEstimator}
        │ Distance ⋯
        │ FeatureDistance ⋯
      w ┼ nothing
    alg ┼ QuadraticBlend()
  scale ┼ nothing
    sim ┴ ChordSimilarity()
```

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`AbstractDistanceBlendAlgorithm`](@ref)
  - [`AbstractDistanceScaleAlgorithm`](@ref)
  - [`Distance`](@ref)
  - [`FeatureDistance`](@ref)
  - [`default_similarity`](@ref)
  - [`blend_similarity`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`feature_readable_mask`](@ref): composes the mask of each part with a positive weight.
  - [`factory`](@ref)
"""
@propagatable @concrete struct BlendDistance <: AbstractDistanceEstimator
    """
    $(field_dict[:bddes])
    """
    @fprop des
    """
    $(field_dict[:bdw])
    """
    w
    """
    $(field_dict[:bdalg])
    """
    alg
    """
    $(field_dict[:bdscale])
    """
    scale
    """
    $(field_dict[:bdsim])
    """
    sim
    function BlendDistance(des::AbstractVector{<:AbstractDistanceEstimator},
                           w::Option{<:VecNum}, alg::AbstractDistanceBlendAlgorithm,
                           scale::Union{Nothing, AbstractDistanceScaleAlgorithm,
                                        AbstractVector{<:Option{<:AbstractDistanceScaleAlgorithm}}},
                           sim::AbstractSimilarityMatrixAlgorithm)::BlendDistance
        @argcheck(!isempty(des), IsEmptyError("`BlendDistance.des` cannot be empty"))
        if !isnothing(w)
            @argcheck(!isempty(w), IsEmptyError("`BlendDistance.w` cannot be empty"))
            @argcheck(length(w) == length(des),
                      DimensionMismatch("`BlendDistance.w` has $(length(w)) entries, but `des` has $(length(des)) parts. State one weight for each part."))
            @argcheck(all(x -> isfinite(x) && zero(x) <= x, w),
                      DomainError(w,
                                  "every entry of `BlendDistance.w` must be finite and non-negative"))
            @argcheck(any(x -> zero(x) < x, w),
                      DomainError(w,
                                  "`BlendDistance.w` has no positive entry, so no part enters the blend. State a positive weight for at least one part."))
        end
        if isa(scale, AbstractVector)
            @argcheck(length(scale) == length(des),
                      DimensionMismatch("`BlendDistance.scale` has $(length(scale)) entries, but `des` has $(length(des)) parts. State one rule or `nothing` for each part, or one rule for every part."))
        end
        return new{typeof(des), typeof(w), typeof(alg), typeof(scale), typeof(sim)}(des, w,
                                                                                    alg,
                                                                                    scale,
                                                                                    sim)
    end
end
function BlendDistance(; des::AbstractVector{<:AbstractDistanceEstimator},
                       w::Option{<:VecNum} = nothing,
                       alg::AbstractDistanceBlendAlgorithm = QuadraticBlend(),
                       scale::Union{Nothing, AbstractDistanceScaleAlgorithm,
                                    AbstractVector{<:Option{<:AbstractDistanceScaleAlgorithm}}} = nothing,
                       sim::AbstractSimilarityMatrixAlgorithm = blend_similarity(alg, des,
                                                                                 w))::BlendDistance
    return BlendDistance(des, w, alg, scale, sim)
end
function default_similarity(::AbstractDistanceBlendAlgorithm)::ComplementSimilarity
    return ComplementSimilarity()
end
function default_similarity(::QuadraticBlend)::ChordSimilarity
    return ChordSimilarity()
end
"""
    blend_similarity(alg::AbstractDistanceBlendAlgorithm, des::AbstractVector, w)
    blend_similarity(alg::QuadraticBlend, des::AbstractVector, w)

Select the default `sim` of a [`BlendDistance`](@ref) from its blend rule `alg`, its parts `des` and its weights `w`.

A rule alone sets the similarity of most blends, through [`default_similarity`](@ref) of the rule, which a rule of your own extends. A quadratic blend of parts of one chord family has an exact inverse that the parts set, so this function reads the parts under [`QuadraticBlend`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. The fallback: return [`default_similarity`](@ref) of `alg`.
 2. [`QuadraticBlend`](@ref): take the parts with a positive weight, every part when `w` is `nothing`. When each of them is a [`FeatureDistance`](@ref) or a [`BlendDistance`](@ref), and all of them hold one `sim` that is [`ChordSimilarity`](@ref) or [`OrthantChordSimilarity`](@ref), return that `sim`. Otherwise return [`default_similarity`](@ref) of `alg`, [`ChordSimilarity`](@ref).

Each of the two similarities of step 2 is an inverse of the form ``S = 1 - a D^{2}``. The quadratic blend of such parts is the same form of the weighted mean of their similarities, so the `sim` of the parts recovers that mean exactly. So a blend of [`WalkDistance`](@ref) parts takes [`OrthantChordSimilarity`](@ref). A blend of a chord part and an orthant chord part has no exact inverse, so state its `sim`.

# Arguments

  - `alg`: The blend rule.
  - `des`: The parts of the blend.
  - `w`: The weights of the parts, or `nothing` for equal weights. A `w` of a wrong length gives no error here, so the constructor refuses it.

# Returns

  - `sim::AbstractSimilarityMatrixAlgorithm`: The default similarity of the blend.

# Related

  - [`BlendDistance`](@ref)
  - [`default_similarity`](@ref)
  - [`QuadraticBlend`](@ref)
  - [`OrthantChordSimilarity`](@ref)
"""
function blend_similarity(alg::AbstractDistanceBlendAlgorithm, ::AbstractVector, ::Any)
    return default_similarity(alg)
end
function blend_similarity(alg::QuadraticBlend, des::AbstractVector, w)
    # The parts that enter. `zip` stops at the shorter of the two, so a `w` of a wrong length
    # reaches the `DimensionMismatch` of the constructor and not a `BoundsError` here.
    ps = isnothing(w) ? des : [d for (d, x) in zip(des, w) if zero(x) < x]
    sims = unique(isa(d, Union{FeatureDistance, BlendDistance}) ? d.sim : nothing
                  for d in ps)
    sim = isone(length(sims)) ? only(sims) : nothing
    return if isa(sim, Union{ChordSimilarity, OrthantChordSimilarity})
        sim
    else
        default_similarity(alg)
    end
end
"""
    positive_parts(de::BlendDistance)

Give the indices of the parts of `de` that enter the blend, the parts with a positive weight.

A part with weight zero adds nothing to the blend. So the blend computes no distance for it, and it drops no asset for it.

# Algorithm

 1. When `de.w` is `nothing`, every part enters. Return the indices of `de.des`.
 2. Otherwise return the indices of the positive entries of `de.w`.

# Arguments

  - `de`: The blend distance estimator.

# Returns

  - `ks::AbstractVector{<:Integer}`: The indices of the parts that enter, in the order of the parts.

# Related

  - [`BlendDistance`](@ref)
  - [`blend_weights`](@ref)
"""
function positive_parts(de::BlendDistance)
    return isnothing(de.w) ? eachindex(de.des) : findall(x -> zero(x) < x, de.w)
end
"""
    blend_weights(w::Nothing, n::Integer, T::Type)
    blend_weights(w::VecNum, n::Integer, T::Type)

Give the normalised weights of the `n` parts of a [`BlendDistance`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `w` is `nothing`: return `n` equal weights of `1 / n`, of the element type `T` of the distance matrices of the parts. So the default weights keep a `Float32` blend in `Float32`.
 2. `w` is a vector: return `w` divided by its sum, as a new vector. The weights of the caller set their own type, and the method does not read `T`. The function does not write to `w`.

# Arguments

  - `w`: The field `w` of the blend.
  - `n`: The number of parts.
  - `T`: The element type of the distance matrices of the parts.

# Returns

  - `w::VecNum`: One weight for each part. The weights sum to `1`.

# Related

  - [`BlendDistance`](@ref)
  - [`positive_parts`](@ref)
"""
function blend_weights(::Nothing, n::Integer, T::Type)
    return fill(inv(T(n)), n)
end
function blend_weights(w::VecNum, ::Integer, ::Type)
    return w ./ sum(w)
end
"""
    part_scale(scale::AbstractVector, k::Integer)
    part_scale(scale::Option{<:AbstractDistanceScaleAlgorithm}, k::Integer)

Give the scale rule of part `k` of a [`BlendDistance`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `scale` is a vector: return its entry `k`.
 2. `scale` is one rule or `nothing`: return it, because it applies to every part.

# Arguments

  - `scale`: The field `scale` of the blend.
  - `k`: The index of the part.

# Returns

  - `alg::Option{<:AbstractDistanceScaleAlgorithm}`: The scale rule of the part, or `nothing`.

# Related

  - [`BlendDistance`](@ref)
  - [`scale_distance`](@ref)
"""
function part_scale(scale::AbstractVector, k::Integer)
    return scale[k]
end
function part_scale(scale::Option{<:AbstractDistanceScaleAlgorithm}, ::Integer)
    return scale
end
"""
    scale_distance(alg::Nothing, D::MatNum)
    scale_distance(alg::MaximumScale, D::MatNum)

Change the units of the distance matrix `D` of one part of a [`BlendDistance`](@ref) under the scale rule `alg`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `alg` is `nothing`: return `D` as it is.
 2. `alg` is a [`MaximumScale`](@ref): find the largest finite entry `m` of `D`. Return `D` as it is when `m` is zero. Otherwise return `D ./ m`, a new matrix.

# Arguments

  - `alg`: The scale rule of the part, or `nothing`.
  - `D`: The distance matrix of the part.

# Returns

  - `D::MatNum`: The distance matrix in the new units. The function does not write to the argument.

# Related

  - [`AbstractDistanceScaleAlgorithm`](@ref)
  - [`MaximumScale`](@ref)
  - [`BlendDistance`](@ref)
"""
function scale_distance(::Nothing, D::MatNum)
    return D
end
function scale_distance(::MaximumScale, D::MatNum)
    m = maximum(x -> isfinite(x) ? x : zero(x), D)
    return iszero(m) ? D : D ./ m
end
"""
    blend_distances(alg::QuadraticBlend, w::VecNum, Ds::AbstractVector{<:MatNum})
    blend_distances(alg::ArithmeticBlend, w::VecNum, Ds::AbstractVector{<:MatNum})
    blend_distances(alg::PowerBlend, w::VecNum, Ds::AbstractVector{<:MatNum})

Join the distance matrices `Ds` of the parts with a positive weight into one distance matrix, under the blend rule `alg` and the normalised weights `w`.

The element type of the result is the promotion of the element types of `w` and of every matrix of `Ds`. A `NaN` entry of a part gives a `NaN` entry of the blend, so a fit can still drop an asset that a part cannot read.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`QuadraticBlend`](@ref): sum `w[k] .* Ds[k] .^ 2` over the parts, and take the root of each entry.
 2. [`ArithmeticBlend`](@ref): sum `w[k] .* Ds[k]` over the parts.
 3. [`PowerBlend`](@ref) with an infinite `p`: take the largest entry over the parts, entry by entry. The weights do not enter.
 4. [`PowerBlend`](@ref) with a finite `p`: sum `w[k] .* Ds[k] .^ p` over the parts, and raise each entry to `1 / p`.

# Arguments

  - `alg`: The blend rule.
  - `w`: The normalised weights of the parts in `Ds`, one for each matrix.
  - `Ds`: The distance matrices of the parts with a positive weight, after their scale rules.

# Returns

  - `D::Matrix{<:Number}`: The blended distance matrix.

# Related

  - [`AbstractDistanceBlendAlgorithm`](@ref)
  - [`BlendDistance`](@ref)
"""
function blend_distances(::QuadraticBlend, w::VecNum, Ds::AbstractVector{<:MatNum})
    return sqrt.(sum(w[k] .* Ds[k] .^ 2 for k in eachindex(w, Ds)))
end
function blend_distances(::ArithmeticBlend, w::VecNum, Ds::AbstractVector{<:MatNum})
    return sum(w[k] .* Ds[k] for k in eachindex(w, Ds))
end
function blend_distances(alg::PowerBlend, w::VecNum, Ds::AbstractVector{<:MatNum})
    p = alg.p
    if isinf(p)
        D = Ds[firstindex(Ds)]
        for k in Iterators.drop(eachindex(Ds), 1)
            D = max.(D, Ds[k])
        end
        return D
    end
    S = sum(w[k] .* Ds[k] .^ p for k in eachindex(w, Ds))
    # The root takes the type of the sum, so an integer `p` keeps a `Float32` blend in
    # `Float32`: `inv(3)` is a `Float64`.
    return S .^ inv(oftype(one(eltype(S)), p))
end
"""
    reads_returns_matrix(de::AbstractDistanceEstimator) -> Bool

Answer whether the distance estimator `de` reads the returns matrix that its consumer passes.

[`plain_graph_reader`](@ref) asks it. A [`FeatureDistance`](@ref) reads the features of an Asset Panel and not the returns matrix, so its consumer is not a plain graph reader, whatever its covariance estimator.

# Algorithm

The method that Julia selects is the algorithm.

 1. A [`FeatureDistance`](@ref) answers `false`.
 2. A [`BlendDistance`](@ref) answers `true` when one of its parts with a positive weight answers `true`. A nested blend answers through its own parts.
 3. Every other distance estimator answers `true`.

# Arguments

  - `de`: The distance estimator.

# Returns

  - `flag::Bool`: `true` when `de` reads the returns matrix.

# Related

  - [`plain_graph_reader`](@ref)
  - [`FeatureDistance`](@ref)
  - [`BlendDistance`](@ref)
"""
function reads_returns_matrix(::AbstractDistanceEstimator)
    return true
end
function reads_returns_matrix(::FeatureDistance)
    return false
end
function reads_returns_matrix(de::BlendDistance)
    return any(k -> reads_returns_matrix(de.des[k]), positive_parts(de))
end
function reads_sample_values(de::BlendDistance)
    return any(k -> reads_sample_values(de.des[k]), positive_parts(de))
end
function feature_readable_mask(de::BlendDistance, imsk::Option{<:BitVector}, rd)
    for k in positive_parts(de)
        imsk = feature_readable_mask(de.des[k], imsk, rd)
    end
    return imsk
end
"""
    distance(de::BlendDistance, ce, X; kwargs...)

Compute the blended distance matrix of the parts of `de`.

Each consumer calls `distance(de, ce, X; …)` or `cor_and_dist(de, ce, X; …)` with a covariance estimator and a returns matrix, and [`logo!`](@ref) passes a similarity matrix in place of the covariance estimator. The method gives each part the same positional arguments and keywords. A [`FeatureDistance`](@ref) part reads `pr` and `rd` from the keywords, and a [`Distance`](@ref) part reads `ce` and `X`.

# Algorithm

 1. Take the parts with a positive weight with [`positive_parts`](@ref). A part with weight zero computes nothing.
 2. For each such part `k`, compute its distance matrix with `distance(de.des[k], ce, X; kwargs...)`, and change its units with [`scale_distance`](@ref) under the scale rule of the part, [`part_scale`](@ref).
 3. Normalise the weights with [`blend_weights`](@ref), and keep the weights of the parts of step 1.
 4. Join the matrices of step 2 with [`blend_distances`](@ref) under `de.alg`.

# Arguments

  - `de`: The blend distance estimator.
  - `ce`: The covariance estimator, or a matrix, that each part reads.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`, or `nothing` on the route of an incremental fit.
  - `kwargs...`: Keyword arguments for each part, `dims`, `pr` and `rd` among them.

# Validation

  - Each part refuses its own arguments. A [`FeatureDistance`](@ref) part refuses an asset that it cannot read, as [`feature_readable_mask`](@ref) states.
  - The parts give matrices of one size. Raises a `DimensionMismatch` otherwise.

# Returns

  - $(ret_dict[:Ddist])

# Related

  - [`BlendDistance`](@ref)
  - [`cor_and_dist`](@ref): the same matrix, with the similarity matrix as well.
  - [`blend_distances`](@ref)
  - [`scale_distance`](@ref)
"""
function distance(de::BlendDistance, ce, X; kwargs...)
    ks = positive_parts(de)
    Ds = [scale_distance(part_scale(de.scale, k), distance(de.des[k], ce, X; kwargs...))
          for k in ks]
    T = mapreduce(eltype, promote_type, Ds)
    return blend_distances(de.alg, blend_weights(de.w, length(de.des), T)[ks], Ds)
end
"""
    cor_and_dist(de::BlendDistance, ce, X; kwargs...)

Compute the similarity matrix and the blended distance matrix of the parts of `de`.

[`clusterise`](@ref) and the network estimators call this method. It reads its arguments as the [`distance`](@ref) method of the blend does. The similarity matrix always follows the blended distance matrix, also under a scale rule or a power. It is not a blend of the similarity matrices of the parts, because a correlation, the complement of a distance and a similarity of another form cannot be compared.

# Algorithm

 1. Compute the blended distance matrix `D` with [`distance`](@ref).
 2. Compute the similarity matrix `S` from `D` with [`distance_to_similarity`](@ref) under `de.sim`.

# Arguments

  - `de`: The blend distance estimator.
  - `ce`: The covariance estimator, or a matrix, that each part reads.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - `kwargs...`: Keyword arguments for each part, `dims`, `pr` and `rd` among them.

# Validation

  - As in the [`distance`](@ref) method of the blend.

# Returns

  - `S::Matrix{<:Number}`: Similarity matrix, `assets × assets`, derived from `D` under `de.sim`.
  - $(ret_dict[:Ddist])

# Related

  - [`BlendDistance`](@ref)
  - [`distance`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`clusterise`](@ref)
"""
function cor_and_dist(de::BlendDistance, ce, X; kwargs...)
    D = distance(de, ce, X; kwargs...)
    return distance_to_similarity(de.sim; D = D), D
end

export BlendDistance, QuadraticBlend, ArithmeticBlend, PowerBlend, MaximumScale
public AbstractDistanceBlendAlgorithm, AbstractDistanceScaleAlgorithm, blend_distances,
       scale_distance
