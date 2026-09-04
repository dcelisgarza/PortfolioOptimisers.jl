"""
    assert_composite_weights(weights::Nothing, n::Integer) -> nothing
    assert_composite_weights(weights::VecNum, n::Integer) -> nothing

Check the fixed descriptor weights of a [`CompositeExposure`](@ref) against the number of Descriptors.

The weights are an input and not an estimate, so they are checked once in the constructor. They are a convex combination: a negative weight would subtract a Descriptor from the composite, and a sum other than one would rescale the composite against the other factors of the same fit.

# Arguments

  - `weights`: The fixed descriptor weights, or `nothing` for equal weights.
  - `n`: The number of Descriptors.

# Validation

  - `length(weights) == n`. Raises a `DimensionMismatch`.
  - Every weight is finite. Raises an [`IsNonFiniteError`](@ref).
  - Every weight is non-negative. Raises a `DomainError`.
  - `sum(weights) ≈ 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`CompositeExposure`](@ref)
  - [`composite_weights`](@ref)
"""
function assert_composite_weights(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_composite_weights(weights::VecNum, n::Integer)::Nothing
    @argcheck(length(weights) == n,
              DimensionMismatch("the descriptor weights are positional, so there is one per Descriptor, got length(weights) = $(length(weights)) and $n Descriptor(s)"))
    assert_all_finite(weights, :weights)
    assert_nonneg(weights, :weights)
    s = sum(weights)
    @argcheck(isapprox(s, one(s)),
              DomainError(s,
                          "the descriptor weights are a convex combination, so they must sum to one, got sum(weights) = $s"))
    return nothing
end
"""
    composite_weights(weights::Nothing, n::Integer) -> VecNum
    composite_weights(weights::VecNum, n::Integer) -> VecNum

Return the descriptor weights a [`CompositeExposure`](@ref) combines its Descriptors under.

`nothing` is the equal-weight composite, which is the one weighting a caller need not write out.

# Arguments

  - `weights`: The fixed descriptor weights, or `nothing`.
  - `n`: The number of Descriptors.

# Returns

  - `w::VecNum`: `fill(1 / n, n)` when `weights` is `nothing`, `weights` itself otherwise.

# Related

  - [`CompositeExposure`](@ref)
  - [`assert_composite_weights`](@ref)
"""
function composite_weights(::Nothing, n::Integer)::VecNum
    return fill(inv(float(n)), n)
end
function composite_weights(weights::VecNum, ::Integer)::VecNum
    return weights
end
"""
$(DocStringExtensions.TYPEDEF)

A Factor Exposure that is a fixed weighted combination of Descriptors.

The member computes each Descriptor, transforms each one cross-sectionally, and combines the scores under fixed weights. The combination is finite-aware: an asset whose Descriptor is unavailable, a gross margin for a firm that reports no cost of revenue, takes its composite from the Descriptors that remain, and the weights renormalise over them.

# Mathematical definition

```math
\\begin{align}
c_{t,j} &= \\frac{\\sum_{i \\in \\mathcal{V}_{t,j}} w_{i} \\, s_{i,t,j}}{\\sum_{i \\in \\mathcal{V}_{t,j}} w_{i}}\\,, &
\\mathcal{V}_{t,j} &= \\left\\{i : s_{i,t,j} \\text{ is finite}\\right\\}\\,.
\\end{align}
```

Where:

  - ``s_{i,t,j}``: Score of Descriptor ``i`` for asset ``j`` at observation ``t``, after the two cross-sectional transforms.
  - ``w_{i}``: Fixed weight of Descriptor ``i``.
  - ``\\mathcal{V}_{t,j}``: Descriptors with a finite score for that asset and observation.
  - ``c_{t,j}``: Composite score.

The composite is `NaN` where the surviving weight ``\\sum_{i \\in \\mathcal{V}_{t,j}} w_{i}`` is zero, and where it is below `min_coverage`. The threshold is on weight and not on count: under the weights `[0.8, 0.2]` a cell carrying only the first Descriptor keeps a surviving weight of `0.8`, so `min_coverage = 0.5` admits it even though one Descriptor of two is missing.

When there is more than one Descriptor and `scoring` is not `nothing`, the composite is scored once more, so assets whose composites rest on different Descriptors are on one scale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CompositeExposure(;
        descriptors::AbstractVector{<:AbstractDescriptorEstimator},
        weights::Option{<:VecNum} = nothing,
        min_coverage::Real = 0.0,
        outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
        scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
        group::Option{<:AbstractString} = nothing,
        bw::AbstractString = "benchmark_weights",
        family::AbstractString = "style"
    ) -> CompositeExposure

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(descriptors)`.
  - The rules of [`assert_composite_weights`](@ref).
  - `isfinite(min_coverage)` and `0 <= min_coverage <= 1`.
  - `!isempty(group)`, `!isempty(bw)` and `!isempty(family)`.

# Examples

```jldoctest
julia> CompositeExposure(; descriptors = [Passthrough(; field = \"a\")], outlier = nothing,
                         scoring = nothing)
CompositeExposure
   descriptors ┼ 1-element Vector{Passthrough}
               │ Passthrough ⋯
       weights ┼ nothing
  min_coverage ┼ Float64: 0.0
       outlier ┼ nothing
       scoring ┼ nothing
         group ┼ nothing
            bw ┼ String: \"benchmark_weights\"
        family ┴ String: \"style\"
```

# Related

  - [`AbstractExposureEstimator`](@ref)
  - [`factor_exposure`](@ref)
  - [`DerivedExposure`](@ref)
  - [`AbstractDescriptorEstimator`](@ref)
  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalWinsoriser`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
"""
@concrete struct CompositeExposure <: AbstractExposureEstimator
    """
    Descriptor Estimators the composite combines, in the order the weights are written in.
    """
    descriptors
    """
    Fixed non-negative descriptor weights summing to one, or `nothing` for equal weights.
    """
    weights
    """
    Smallest surviving descriptor weight a cell may carry. A cell below it is `NaN` rather than a composite of too few Descriptors.
    """
    min_coverage
    """
    Cross-sectional transform applied to each Descriptor before it is scored, or `nothing` to skip the step.
    """
    outlier
    """
    Cross-sectional transform applied to each Descriptor after the outlier step, and once more to the composite, or `nothing` to skip the step.
    """
    scoring
    """
    Name of the categorical Panel Field the transforms are applied within, or `nothing` to transform each observation as one cross-section.
    """
    group
    """
    Name of the numeric Panel Field holding the benchmark weights the transforms are weighted by.
    """
    bw
    """
    Label of the Factor Family the factor belongs to.
    """
    family
    function CompositeExposure(descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                               weights::Option{<:VecNum}, min_coverage::Real,
                               outlier::Option{<:AbstractCrossSectionalTransform},
                               scoring::Option{<:AbstractCrossSectionalTransform},
                               group::Option{<:AbstractString}, bw::AbstractString,
                               family::AbstractString)
        @argcheck(!isempty(descriptors),
                  IsEmptyError("a composite Factor Exposure combines Descriptors, so it needs at least one"))
        assert_composite_weights(weights, length(descriptors))
        assert_finite(min_coverage, :min_coverage)
        assert_closed_unit_interval(min_coverage, :min_coverage)
        if !isnothing(group)
            assert_panel_terms(group, :group)
        end
        assert_panel_terms(bw, :bw)
        assert_exposure_family(family)
        return new{typeof(descriptors), typeof(weights), typeof(min_coverage),
                   typeof(outlier), typeof(scoring), typeof(group), typeof(bw),
                   typeof(family)}(descriptors, weights, min_coverage, outlier, scoring,
                                   group, bw, family)
    end
end
function CompositeExposure(; descriptors::AbstractVector{<:AbstractDescriptorEstimator},
                           weights::Option{<:VecNum} = nothing, min_coverage::Real = 0.0,
                           outlier::Option{<:AbstractCrossSectionalTransform} = CrossSectionalWinsoriser(),
                           scoring::Option{<:AbstractCrossSectionalTransform} = CrossSectionalStandardiser(),
                           group::Option{<:AbstractString} = nothing,
                           bw::AbstractString = "benchmark_weights",
                           family::AbstractString = "style")::CompositeExposure
    return CompositeExposure(descriptors, weights, min_coverage, outlier, scoring, group,
                             bw, family)
end
"""
    composite_score(de::AbstractDescriptorEstimator, rd::ReturnsResult,
                    outlier::Option{<:AbstractCrossSectionalTransform},
                    scoring::Option{<:AbstractCrossSectionalTransform}, w::MatNum,
                    groups::Option{<:AbstractMatrix{<:Integer}}) -> MatNum

Compute one Descriptor and apply the two cross-sectional transform slots to it.

Both slots run before the Descriptors are combined, so a Descriptor measured in a currency and a Descriptor measured as a ratio reach the composite on one scale.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd])
  - `outlier`: The outlier transform, or `nothing`.
  - `scoring`: The scoring transform, or `nothing`.
  - `w`: Benchmark weight matrix `observations × assets`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.

# Returns

  - `S::MatNum`: The score, `observations × assets`.

# Related

  - [`CompositeExposure`](@ref)
  - [`descriptor`](@ref)
  - [`exposure_transform`](@ref)
"""
function composite_score(de::AbstractDescriptorEstimator, rd::ReturnsResult,
                         outlier::Option{<:AbstractCrossSectionalTransform},
                         scoring::Option{<:AbstractCrossSectionalTransform}, w::MatNum,
                         groups::Option{<:AbstractMatrix{<:Integer}})::MatNum
    S = exposure_transform(outlier, descriptor(de, rd), w, groups)
    return exposure_transform(scoring, S, w, groups)
end
"""
    composite_accumulate!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                          S::MatNum, wk::Real) -> nothing

Add one weighted Descriptor score into the numerator and the surviving weight of a composite, in place.

A non-finite score contributes to neither, which is what makes the combination finite-aware: the weight of a Descriptor that is missing on a cell never reaches the denominator of that cell.

# Arguments

  - `num`: Weighted score sum, `observations × assets`, changed in place.
  - `den`: Surviving weight sum, `observations × assets`, changed in place.
  - `S`: The Descriptor score, `observations × assets`.
  - `wk`: The Descriptor's weight.

# Returns

  - `nothing`. `num` and `den` carry the accumulated sums.

# Related

  - [`CompositeExposure`](@ref)
  - [`composite_finalise!`](@ref)
"""
function composite_accumulate!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                               S::MatNum, wk::Real)::Nothing
    for k in CartesianIndices(num)
        s = S[k]
        if isfinite(s)
            num[k] += wk * s
            den[k] += wk
        end
    end
    return nothing
end
"""
    composite_finalise!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                        mc::Real) -> nothing

Divide the weighted score sum of a composite by its surviving weight, and apply the coverage threshold, in place.

# Algorithm

 1. Write `NaN` into every cell whose surviving weight is zero, because no Descriptor reached it.
 2. Write `NaN` into every cell whose surviving weight is below `mc`.
 3. Divide every other cell by its surviving weight.

# Arguments

  - `num`: Weighted score sum, `observations × assets`, changed in place.
  - `den`: Surviving weight sum, `observations × assets`.
  - `mc`: Smallest surviving weight a cell may carry.

# Returns

  - `nothing`. `num` carries the composite.

# Related

  - [`CompositeExposure`](@ref)
  - [`composite_accumulate!`](@ref)
"""
function composite_finalise!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                             mc::Real)::Nothing
    Tf = eltype(num)
    for k in CartesianIndices(num)
        d = den[k]
        num[k] = d > zero(d) && d >= mc ? num[k] / d : Tf(NaN)
    end
    return nothing
end
"""
    factor_exposure(xe::CompositeExposure, rd::ReturnsResult) -> Matrix{<:Real}

Compute the Factor Exposure of a fixed weighted combination of Descriptors.

# Algorithm

 1. Read the benchmark weights and the group labels off the carrier.
 2. Compute each Descriptor, and apply the outlier slot and then the scoring slot to it.
 3. Accumulate the finite-aware weighted sum and the surviving weight of every cell.
 4. Divide, and write `NaN` where the surviving weight is zero or below `min_coverage`.
 5. Score the composite once more when there is more than one Descriptor and `scoring` is not `nothing`.

# Arguments

  - `xe`: Composite Exposure Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`panel_field_values`](@ref) for the benchmark weights and for every Panel Field the Descriptors name.
  - The rules of [`cross_sectional_transform`](@ref) for both transform slots.

# Returns

  - `L::Matrix{<:Real}`: The Factor Exposure, `observations × assets`.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"a\", vals = [1.0 2.0; 3.0 4.0]),
                          NumericPanelInput(; name = \"b\", vals = [5.0 6.0; 7.0 8.0]),
                          NumericPanelInput(; name = \"benchmark_weights\",
                                            vals = [1.0 1.0; 1.0 1.0])]; amsk = trues(2, 2),
                         emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), res...);

julia> xe = CompositeExposure(;
                              descriptors = [Passthrough(; field = \"a\"),
                                             Passthrough(; field = \"b\")], weights = [0.25, 0.75],
                              outlier = nothing, scoring = nothing);

julia> factor_exposure(xe, rd)
2×2 Matrix{Float64}:
 4.0  5.0
 6.0  7.0
```

# Related

  - [`CompositeExposure`](@ref)
  - [`AbstractExposureEstimator`](@ref)
  - [`composite_score`](@ref)
  - [`composite_accumulate!`](@ref)
  - [`composite_finalise!`](@ref)
"""
function factor_exposure(xe::CompositeExposure, rd::ReturnsResult)::Matrix{<:Real}
    w = exposure_benchmark_weights(rd, xe.bw)
    groups = exposure_group_labels(rd, xe.group)
    des = xe.descriptors
    wv = composite_weights(xe.weights, length(des))
    S = composite_score(des[1], rd, xe.outlier, xe.scoring, w, groups)
    Tf = float(promote_type(eltype(S), eltype(wv)))
    num = zeros(Tf, size(S))
    den = zeros(Tf, size(S))
    composite_accumulate!(num, den, S, wv[1])
    for k in 2:length(des)
        composite_accumulate!(num, den,
                              composite_score(des[k], rd, xe.outlier, xe.scoring, w,
                                              groups), wv[k])
    end
    composite_finalise!(num, den, xe.min_coverage)
    return length(des) > 1 ? exposure_transform(xe.scoring, num, w, groups) : num
end

export CompositeExposure
