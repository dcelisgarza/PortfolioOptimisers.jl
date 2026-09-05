"""
    assert_signed_composite_weights(weights::Nothing, n::Integer) -> nothing
    assert_signed_composite_weights(weights::VecNum, n::Integer) -> nothing

Check the signed Descriptor weights of a [`FixedWeightedReturnForecast`](@ref).

The weights of a Return Forecast are signed, because a Descriptor may forecast a return that falls, so they are not the convex weights of a [`CompositeExposure`](@ref). They are normalised by their absolute sum when the forecast is computed, and that sum is what must be positive: a set of weights that cancels states no forecast.

# Arguments

  - `weights`: The signed weights, or `nothing` for equal weights.
  - `n`: Number of Descriptors.

# Validation

  - `length(weights) == n`. Raises a `DimensionMismatch`.
  - Every entry of `weights` is finite. Raises a `DomainError`.
  - `sum(abs, weights) > 0`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`FixedWeightedReturnForecast`](@ref)
  - [`signed_composite_weights`](@ref)
  - [`assert_composite_weights`](@ref)
"""
function assert_signed_composite_weights(::Nothing, ::Integer)::Nothing
    return nothing
end
function assert_signed_composite_weights(weights::VecNum, n::Integer)::Nothing
    @argcheck(length(weights) == n,
              DimensionMismatch("the descriptor weights are positional, so there is one per Descriptor, got length(weights) = $(length(weights)) and $n Descriptor(s)"))
    assert_all_finite(weights, :weights)
    s = sum(abs, weights)
    @argcheck(s > zero(s),
              DomainError(s,
                          "the descriptor weights are normalised by their absolute sum, so that sum must be positive, got sum(abs, weights) = $s"))
    return nothing
end
"""
    signed_composite_weights(weights::Nothing, n::Integer) -> VecNum
    signed_composite_weights(weights::VecNum, n::Integer) -> VecNum

Return the signed Descriptor weights a [`FixedWeightedReturnForecast`](@ref) combines its scores under, normalised by their absolute sum.

`nothing` is the equal-weight forecast, which is the one weighting a caller need not write out. The normalisation puts the composite on one scale whatever the caller wrote, so the forecast scale means the same thing across estimators.

# Arguments

  - `weights`: The signed weights, or `nothing` for equal weights.
  - `n`: Number of Descriptors.

# Returns

  - `wv::VecNum`: The normalised signed weights, whose absolute values sum to one.

# Examples

```jldoctest
julia> PortfolioOptimisers.signed_composite_weights([2.0, -2.0], 2)
2-element Vector{Float64}:
  0.5
 -0.5
```

# Related

  - [`FixedWeightedReturnForecast`](@ref)
  - [`assert_signed_composite_weights`](@ref)
  - [`composite_weights`](@ref)
"""
function signed_composite_weights(::Nothing, n::Integer)::VecNum
    return fill(inv(float(n)), n)
end
function signed_composite_weights(weights::VecNum, ::Integer)::VecNum
    return weights ./ sum(abs, weights)
end
"""
    signed_composite_accumulate!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                                 S::Arr3Num, wv::VecNum) -> nothing

Accumulate the signed weighted Descriptor scores and the surviving absolute weight of a composite, in place.

It is [`composite_accumulate!`](@ref) with two changes, and it reads the whole score array rather than one slice of it: the weights are signed, so the denominator takes the **absolute** weight while the numerator takes the signed product; and the Descriptor axis is the third axis of one array, so no slice of it is formed.

The accumulation is finite-aware. A Descriptor that is not finite on a cell contributes to neither sum there, which is what renormalises the composite of that cell over the Descriptors that remain.

# Arguments

  - `num`: Weighted score sum, `observations × assets`, changed in place.
  - `den`: Surviving absolute weight sum, `observations × assets`, changed in place.
  - `S`: The Descriptor scores, `observations × assets × descriptors`.
  - `wv`: The signed Descriptor weights, one per Descriptor.

# Returns

  - `nothing`. `num` and `den` carry the accumulated sums.

# Related

  - [`FixedWeightedReturnForecast`](@ref)
  - [`signed_composite_weights`](@ref)
  - [`composite_accumulate!`](@ref)
  - [`composite_finalise!`](@ref)
"""
function signed_composite_accumulate!(num::AbstractMatrix{<:Real},
                                      den::AbstractMatrix{<:Real}, S::Arr3Num,
                                      wv::VecNum)::Nothing
    for k in axes(S, 3)
        w = wv[k]
        a = abs(w)
        for idx in CartesianIndices(num)
            s = S[idx, k]
            if isfinite(s)
                num[idx] += w * s
                den[idx] += a
            end
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast that is a fixed signed combination of Descriptor scores.

The member turns its Descriptors into scores with the recipe in `scores`, combines them under fixed signed weights, and multiplies the composite by a fixed scale. Neither the weights nor the scale is estimated from realised returns: both are the caller's, which is what separates this member from the fitted ones.

The combination is finite-aware, as [`CompositeExposure`](@ref)'s is: an asset whose score is unavailable takes its composite from the Descriptors that remain, and the weights renormalise over their absolute values. `min_coverage` is the smallest share of the absolute weight a cell may be built from.

# Mathematical definition

```math
\\begin{align}
z_{t,j} &= \\frac{\\sum_{k \\in \\mathcal{V}_{t,j}} w_{k} \\, s_{k,t,j}}{\\sum_{k \\in \\mathcal{V}_{t,j}} \\lvert w_{k} \\rvert}\\,, &
\\mathcal{V}_{t,j} &= \\left\\{k : s_{k,t,j} \\text{ is finite}\\right\\}\\,.
\\end{align}
```

Where:

  - ``s_{k,t,j}``: score of Descriptor ``k`` for asset ``j`` at observation ``t``.
  - ``w_{k}``: signed weight of Descriptor ``k``, normalised so that ``\\sum_{k} \\lvert w_{k} \\rvert = 1``.
  - ``\\mathcal{V}_{t,j}``: Descriptors whose score is finite for that asset and observation.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`FixedWeightedReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`DescriptorScores`](@ref)
  - [`AbstractForecastUnit`](@ref)
  - [`CompositeExposure`](@ref)
"""
@concrete struct FixedWeightedReturnForecast <: AbstractReturnForecastEstimator
    """
    The recipe that turns the Descriptors into cross-sectional scores.
    """
    scores
    """
    Multiplicative scale of the composite score, in the Forecast Unit `unit`. It is the forecast one unit of composite score is worth.
    """
    scale
    """
    Signed Descriptor weights, in the order the Descriptors are written in, or `nothing` for equal weights. They are normalised by their absolute sum.
    """
    weights
    """
    Smallest share of the absolute Descriptor weight a cell may carry. A cell below it is `NaN` rather than a composite of too few Descriptors.
    """
    min_coverage
    """
    The Forecast Unit the composite score is read in, before the member converts it to return units.
    """
    unit
    function FixedWeightedReturnForecast(scores::DescriptorScores, scale::Real,
                                         weights::Option{<:VecNum}, min_coverage::Real,
                                         unit::AbstractForecastUnit)
        assert_finite(scale, :scale)
        assert_gt0(scale, :scale)
        assert_signed_composite_weights(weights, length(scores.descriptors))
        assert_finite(min_coverage, :min_coverage)
        assert_closed_unit_interval(min_coverage, :min_coverage)
        return new{typeof(scores), typeof(scale), typeof(weights), typeof(min_coverage),
                   typeof(unit)}(scores, scale, weights, min_coverage, unit)
    end
end
function FixedWeightedReturnForecast(; scores::DescriptorScores, scale::Real,
                                     weights::Option{<:VecNum} = nothing,
                                     min_coverage::Real = 0.0,
                                     unit::AbstractForecastUnit = IdiosyncraticReturnUnit())::FixedWeightedReturnForecast
    return FixedWeightedReturnForecast(scores, scale, weights, min_coverage, unit)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`FixedWeightedReturnForecast`](@ref).

Beside the two reads [`AbstractReturnForecastResult`](@ref) states, it carries the normalised signed weights the composite was built under, so a reader sees the weighting the forecast came from without renormalising the estimator's own.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`return_forecast`](@ref)
"""
@concrete struct FixedWeightedReturnForecastResult <: AbstractReturnForecastResult
    """
    $(field_dict[:rf_mu])
    """
    mu
    """
    $(field_dict[:rf_hist])
    """
    hist
    """
    Normalised signed Descriptor weights the composite was built under, whose absolute values sum to one.
    """
    weights
    function FixedWeightedReturnForecastResult(mu::VecNum, hist::MatNum, weights::VecNum)
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        @argcheck(!isempty(hist), IsEmptyError("hist cannot be empty"))
        @argcheck(!isempty(weights), IsEmptyError("weights cannot be empty"))
        @argcheck(length(mu) == size(hist, 2),
                  DimensionMismatch("mu ($(length(mu))) must match hist ($(size(hist, 2)) columns)"))
        return new{typeof(mu), typeof(hist), typeof(weights)}(mu, hist, weights)
    end
end
function FixedWeightedReturnForecastResult(; mu::VecNum, hist::MatNum,
                                           weights::VecNum)::FixedWeightedReturnForecastResult
    return FixedWeightedReturnForecastResult(mu, hist, weights)
end
"""
    return_forecast(rfe::FixedWeightedReturnForecast, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> FixedWeightedReturnForecastResult

Compute the Return Forecast of a fixed signed combination of Descriptor scores.

# Algorithm

 1. Compute the Descriptor scores through [`descriptor_scores`](@ref).
 2. Normalise the signed weights by their absolute sum.
 3. Accumulate the finite-aware signed weighted sum and the surviving absolute weight of every cell over the Descriptor axis.
 4. Divide, and write `NaN` where the surviving absolute weight is zero or below `min_coverage`.
 5. Score the composite once more when there is more than one Descriptor and the recipe's scoring slot is set.
 6. Multiply by `scale`, and convert the whole history from the Forecast Unit to return units.
 7. Read `mu` off the last observation of that history.

# Arguments

  - `rfe`: Fixed weighted Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. Its exposure history is read only under a Neutralisation, and its idiosyncratic variance history only under [`IdiosyncraticSharpeUnit`](@ref).

# Validation

  - The rules of [`descriptor_scores`](@ref) and of [`forecast_return_units`](@ref).

# Returns

  - `rf::FixedWeightedReturnForecastResult`: The fitted forecast, its history and the normalised weights.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"a\", vals = [1.0 2.0; 3.0 4.0]),
                          NumericPanelInput(; name = \"b\", vals = [5.0 6.0; 7.0 8.0])];
                         amsk = trues(2, 2), emsk = trues(2, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2), pnl = pnl);

julia> csfm = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = [0.0, 0.0]);

julia> ds = DescriptorScores(;
                             descriptors = [Passthrough(; field = \"a\"),
                                            Passthrough(; field = \"b\")], outlier = nothing,
                             scoring = nothing);

julia> rf = return_forecast(FixedWeightedReturnForecast(; scores = ds, scale = 1.0,
                                                        weights = [1.0, -1.0]), rd, csfm);

julia> rf.hist
2×2 Matrix{Float64}:
 -2.0  -2.0
 -2.0  -2.0
```

# Related

  - [`FixedWeightedReturnForecast`](@ref)
  - [`FixedWeightedReturnForecastResult`](@ref)
  - [`descriptor_scores`](@ref)
  - [`signed_composite_accumulate!`](@ref)
  - [`composite_finalise!`](@ref)
  - [`forecast_return_units`](@ref)
"""
function return_forecast(rfe::FixedWeightedReturnForecast, rd::ReturnsResult,
                         csfm::CrossSectionalFactorModel)::FixedWeightedReturnForecastResult
    ds = rfe.scores
    S = descriptor_scores(ds, rd, csfm)
    K = size(S, 3)
    wv = signed_composite_weights(rfe.weights, K)
    Tf = float(promote_type(eltype(S), eltype(wv)))
    num = zeros(Tf, size(S, 1), size(S, 2))
    den = zeros(Tf, size(S, 1), size(S, 2))
    signed_composite_accumulate!(num, den, S, wv)
    composite_finalise!(num, den, rfe.min_coverage)
    Z = if K > 1
        exposure_transform(ds.scoring, num, return_forecast_weights(rd),
                           exposure_group_labels(rd, ds.group))
    else
        num
    end
    hist = forecast_return_units(rfe.unit, rfe.scale * Z, csfm.vs)
    return FixedWeightedReturnForecastResult(; mu = hist[end, :], hist = hist, weights = wv)
end

export FixedWeightedReturnForecast, FixedWeightedReturnForecastResult
