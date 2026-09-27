"""
    assert_signed_composite_weights(weights::Nothing, n::Integer) -> nothing
    assert_signed_composite_weights(weights::VecNum, n::Integer) -> nothing

Check the signed Descriptor weights of a [`FixedWeightedReturnForecast`](@ref).

The weights of a Return Forecast are signed, because a Descriptor can forecast a falling return. They are therefore not the convex weights of a [`CompositeExposure`](@ref). [`signed_composite_weights`](@ref) divides them by their absolute sum, so that sum must be positive. Only weights that are all zero fail, and they give no forecast.

# Arguments

  - `weights`: The signed weights, or `nothing` for equal weights.
  - `n`: Number of Descriptors.

# Validation

  - `length(weights) == n`. Raises a `DimensionMismatch`.
  - Every entry of `weights` is finite. Raises an [`IsNonFiniteError`](@ref).
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

Return the signed Descriptor weights of a [`FixedWeightedReturnForecast`](@ref), divided by their absolute sum.

`nothing` gives the equal weights ``1 / n``, so a caller need not write them out. The division puts the composite on one scale whatever weights the caller writes, so one value of `scale` gives a forecast of the same strength under any weights.

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
    return fill(inv(n), n)
end
function signed_composite_weights(weights::VecNum, ::Integer)::VecNum
    return weights ./ sum(abs, weights)
end
"""
    signed_composite_accumulate!(num::AbstractMatrix{<:Real}, den::AbstractMatrix{<:Real},
                                 S::Arr3Num, wv::VecNum) -> nothing

Accumulate the signed weighted Descriptor scores and the surviving absolute weight of a composite, in place.

It is [`composite_accumulate!`](@ref) with two changes. The weights are signed, so the numerator takes the signed product and the denominator takes the absolute weight. It also reads the Descriptors as the third axis of one score array, so it forms no slice of that array.

A Descriptor whose score is not finite on a cell adds to neither sum there. The composite of that cell thus renormalises over the Descriptors that remain.

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

The member turns its Descriptors into scores with the recipe in `scores`, combines them under fixed signed weights, and multiplies the composite by a fixed scale. The caller sets the weights and the scale, and the member estimates neither from realised returns. That separates it from the fitted members.

The combination is finite-aware, as the one of [`CompositeExposure`](@ref) is. An asset that lacks a score takes its composite from the Descriptors that remain, and the weights renormalise over their absolute values.

# Mathematical definition

```math
\\begin{align}
w_{k} &= \\frac{\\omega_{k}}{\\sum_{l = 1}^{K} \\lvert \\omega_{l} \\rvert}\\,, \\\\
W_{ti} &= \\sum_{k \\in \\mathcal{V}_{ti}} \\lvert w_{k} \\rvert\\,, \\\\
z_{ti} &= \\frac{1}{W_{ti}} \\sum_{k \\in \\mathcal{V}_{ti}} w_{k} \\, s_{tik}\\,, \\\\
\\tilde{\\boldsymbol{z}}_{t} &= \\begin{cases} \\phi\\left(\\boldsymbol{z}_{t}\\right) & K > 1\\,, \\\\ \\boldsymbol{z}_{t} & K = 1\\,, \\end{cases} \\\\
\\alpha_{ti} &= \\gamma \\, g_{ti} \\, \\tilde{z}_{ti}\\,.
\\end{align}
```

Where:

  - ``\\omega_{k}``: Signed weight of Descriptor ``k`` in `weights`, or ``1`` for every Descriptor under `weights = nothing`.
  - ``K``: Number of Descriptors.
  - ``w_{k}``: Normalised signed weight of Descriptor ``k``. The absolute values of the normalised weights sum to one.
  - ``s_{tik}``: Score of Descriptor ``k`` for asset ``i`` at observation ``t``, after the outlier and the scoring transforms of the recipe.
  - ``\\mathcal{V}_{ti}``: The Descriptors whose score ``s_{tik}`` is finite.
  - ``W_{ti}``: Surviving absolute weight of asset ``i`` at observation ``t``, the share of the absolute weight that reaches the cell. It lies in ``[0, 1]``.
  - ``z_{ti}``: Composite score of asset ``i`` at observation ``t``. It is `NaN` where ``W_{ti} = 0`` and where ``W_{ti}`` is below `min_coverage`. The threshold is on weight and not on count. Under the weights `[0.8, -0.2]`, a cell with only the first Descriptor keeps ``W_{ti} = 0.8``, so `min_coverage = 0.5` admits it.
  - ``\\boldsymbol{z}_{t}``: The composite scores of observation ``t``, one per asset.
  - ``\\phi``: The cross-sectional transform in the scoring slot of the recipe, or the identity when that slot is `nothing`. It puts the composites of assets that use different Descriptors on one scale.
  - ``\\tilde{z}_{ti}``: Rescored composite of asset ``i`` at observation ``t``.
  - $(math_dict[:g_ti_unit])
  - $(math_dict[:v_ti_idio])
  - $(math_dict[:gamma_rf_scale])
  - $(math_dict[:alpha_ti_fc]) The member publishes the row of the latest observation ``T``.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    FixedWeightedReturnForecast(; scores::DescriptorScores, scale::Real,
                                weights::Option{<:VecNum} = nothing,
                                min_coverage::Real = 0.0,
                                unit::AbstractForecastUnit = IdiosyncraticReturnUnit()) -> FixedWeightedReturnForecast

Keywords correspond to the struct's fields.

## Validation

  - `scale > 0` and is finite.
  - The rules of [`assert_signed_composite_weights`](@ref), with one weight for each Descriptor of `scores`.
  - `isfinite(min_coverage)` and `0 <= min_coverage <= 1`.

# Examples

```jldoctest
julia> ds = DescriptorScores(; descriptors = [Passthrough(; field = \"a\")]);

julia> FixedWeightedReturnForecast(; scores = ds, scale = 0.02)
FixedWeightedReturnForecast
        scores ┼ DescriptorScores
               │   descriptors ┼ 1-element Vector{Passthrough}
               │               │ Passthrough ⋯
               │    neutralise ┼ nothing
               │           cre ┼ CrossSectionalLinearRegression
               │               │         alg ┼ PseudoInverseFallback()
               │               │   intercept ┴ Bool: false
               │       outlier ┼ CrossSectionalWinsoriser
               │               │    low ┼ Float64: 0.01
               │               │   high ┴ Float64: 0.99
               │       scoring ┼ CrossSectionalStandardiser
               │               │   min_group_size ┼ Int64: 8
               │               │             atol ┴ Float64: 1.0e-12
               │         group ┴ nothing
         scale ┼ Float64: 0.02
       weights ┼ nothing
  min_coverage ┼ Float64: 0.0
          unit ┴ IdiosyncraticReturnUnit()
```

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
    $(field_dict[:rf_scores])
    """
    scores
    """
    Multiplicative scale of the composite score, in the Forecast Unit `unit`. It is the forecast that one unit of composite score is worth.
    """
    scale
    """
    Signed Descriptor weights, in the order of the Descriptors in `scores`, or `nothing` for equal weights. The member divides them by their absolute sum.
    """
    weights
    """
    Smallest share of the absolute Descriptor weight a cell may carry. A cell below it is `NaN` rather than a composite of too few Descriptors.
    """
    min_coverage
    """
    $(field_dict[:rf_unit])
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

It carries the two fields that [`AbstractReturnForecastResult`](@ref) requires, and also the normalised signed weights of the composite. A reader thus sees the weights of the forecast and need not normalise the weights of the estimator again.

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
    Normalised signed Descriptor weights of the composite. Their absolute values sum to one.
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

 1. Compute the Descriptor scores over the whole carrier through [`descriptor_scores`](@ref), and cut them to the block's rows.
 2. Normalise the signed weights by their absolute sum.
 3. Accumulate the finite-aware signed weighted sum and the surviving absolute weight of every cell over the Descriptor axis.
 4. Divide, and write `NaN` where the surviving absolute weight is zero or below `min_coverage`.
 5. Score the composite once more when there is more than one Descriptor and the recipe's scoring slot is set.
 6. Multiply by `scale`, and convert the whole history from the Forecast Unit to return units.
 7. Read `mu` off the last observation of that history.

# Arguments

  - `rfe`: Fixed weighted Return Forecast Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `csfm`: The fitted factor-model block. Its histories set the rows of the block. The member reads its exposure history only under a Neutralisation, and its idiosyncratic variance history only under [`IdiosyncraticSharpeUnit`](@ref).

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
  - [`return_forecast_cut`](@ref)
  - [`signed_composite_accumulate!`](@ref)
  - [`composite_finalise!`](@ref)
  - [`forecast_return_units`](@ref)
"""
function return_forecast(rfe::FixedWeightedReturnForecast, rd::ReturnsResult,
                         csfm::CrossSectionalFactorModel)::FixedWeightedReturnForecastResult
    ds = rfe.scores
    (; S, rows) = descriptor_scores(ds, rd, csfm)
    Sb = return_forecast_cut(S, rows)
    K = size(Sb, 3)
    wv = signed_composite_weights(rfe.weights, K)
    Tf = promote_type(eltype(Sb), eltype(wv))
    num = zeros(Tf, size(Sb, 1), size(Sb, 2))
    den = zeros(Tf, size(Sb, 1), size(Sb, 2))
    signed_composite_accumulate!(num, den, Sb, wv)
    composite_finalise!(num, den, rfe.min_coverage)
    Z = if K > 1
        exposure_transform(ds.scoring, num,
                           return_forecast_cut(return_forecast_weights(rd), rows),
                           return_forecast_cut(exposure_group_labels(rd, ds.group), rows))
    else
        num
    end
    hist = forecast_return_units(rfe.unit, rfe.scale * Z, csfm.vs)
    return FixedWeightedReturnForecastResult(; mu = hist[end, :], hist = hist, weights = wv)
end

"""
    port_opt_view(rf::FixedWeightedReturnForecastResult, i, args...)

Return a view of a [`FixedWeightedReturnForecastResult`](@ref), selecting only the assets indexed by `i`.

The view cuts `mu` on its one axis and `hist` on its second axis, the asset axis. The Descriptor weights have one entry per Descriptor and not per asset, so the view keeps them unchanged.

# Arguments

  - `rf`: A fixed weighted Return Forecast result.
  - `i`: Indices of the assets to select.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `rf::FixedWeightedReturnForecastResult`: A new result whose per-asset fields are restricted to the selected assets.

# Related

  - [`FixedWeightedReturnForecastResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function port_opt_view(rf::FixedWeightedReturnForecastResult, i,
                       args...)::FixedWeightedReturnForecastResult
    return FixedWeightedReturnForecastResult(; mu = view(rf.mu, i),
                                             hist = view(rf.hist, :, i),
                                             weights = rf.weights)
end

export FixedWeightedReturnForecast, FixedWeightedReturnForecastResult
