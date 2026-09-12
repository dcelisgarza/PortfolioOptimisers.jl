"""
    forecast_calibration_pairs(alpha::MatNum, y::MatNum, u::MatNum,
                               dates::AbstractVector{<:Integer})

Pool the scorable pairs of an evaluation into three vectors.

Every statistic of this file is taken over the pairs of a forecast and its forward target rather than over the cross-sections that carry them, so the pooling is done once, here, and the three verbs above read vectors. A pair enters where both the forecast and the target are finite, which is the same cross-section [`forecast_cross_section!`](@ref) marks, and the pairs of every evaluation date are concatenated in date order.

The weight of a pair is carried beside it rather than applied to it, because the three readings do not agree on it: [`forecast_calibration_slope`](@ref) is taken under the weights, and [`forecast_calibration_curve`](@ref) and the pooled moments read every pair alike. A pair whose weight is not finite is carried at zero, so it shapes the curve and the moments and takes no part in the slope.

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`.
  - `u`: Cross-sectional weight history `observations × assets`, from [`forecast_ic_weights`](@ref).
  - `dates`: Row indices of the observations to pool.

# Returns

  - `pairs::Tuple`: `(a, b, q)`, three vectors of the same length.

      + `a::VecNum`: The forecast of each pair.
      + `b::VecNum`: The target of each pair.
      + `q::VecNum`: The weight of each pair, zero where `u` is not finite.

# Related

  - [`forecast_calibration`](@ref)
  - [`forecast_calibration_slope`](@ref)
  - [`forecast_calibration_curve`](@ref)
  - [`forecast_ic_weights`](@ref)
"""
function forecast_calibration_pairs(alpha::MatNum, y::MatNum, u::MatNum,
                                    dates::AbstractVector{<:Integer})
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)), real(eltype(u)))
    N = size(alpha, 2)
    a = Vector{Tf}(undef, 0)
    b = Vector{Tf}(undef, 0)
    q = Vector{Tf}(undef, 0)
    for t in dates, i in 1:N
        if isfinite(alpha[t, i]) && isfinite(y[t, i])
            push!(a, alpha[t, i])
            push!(b, y[t, i])
            push!(q, isfinite(u[t, i]) ? u[t, i] : zero(Tf))
        end
    end
    return a, b, q
end
"""
    forecast_calibration_slope(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                               q::AbstractVector{<:Real}) -> Real

Return the scale multiplier that maps a forecast onto its realised target.

It is the slope of a weighted regression of the target on the forecast with **no intercept**, which is what makes it a statement about scale alone: the line is pinned through the origin, so a forecast that is right on average about the level of the cross-section and wrong about its spread is not rescued by a free constant. A slope of `1` says the forecast is already in target units, a slope above `1` says its magnitude is too small, and a slope below `1` says it is too large.

The intercept is refused rather than fitted because a Return Forecast is a cross-sectional statement. The cross-sectional mean of the target is what the factor model is for, and the forecast is answerable for what is left; a fitted intercept would absorb that mean and report a scale measured against a level the forecast never claimed.

# Mathematical definition

```math
\\mathrm{slope} = \\dfrac{\\sum_{k} q_{k} \\, a_{k} \\, b_{k}}{\\sum_{k} q_{k} \\, a_{k}^{2}}
```

Where:

  - ``a_{k}``: Forecast of pair ``k``.
  - ``b_{k}``: Target of pair ``k``.
  - ``q_{k}``: Weight of pair ``k``.

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref).
  - `b`: The target of each pair.
  - `q`: The weight of each pair.

# Returns

  - `slope::Real`: The scale multiplier, or `NaN` when no pair carries a positive weighted square, which is the case of an empty pooling and the case of a forecast that is identically zero.

# Related

  - [`forecast_calibration`](@ref)
  - [`forecast_calibration_pairs`](@ref)
  - [`forecast_calibration_curve`](@ref)
"""
function forecast_calibration_slope(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                                    q::AbstractVector{<:Real})
    Tf = promote_type(real(eltype(a)), real(eltype(b)), real(eltype(q)))
    num = zero(Tf)
    den = zero(Tf)
    for k in eachindex(a)
        num += q[k] * a[k] * b[k]
        den += q[k] * a[k]^2
    end
    return den > zero(Tf) ? num / den : Tf(NaN)
end
"""
    forecast_calibration_edges(a::AbstractVector{<:Real}, bins::Integer) -> VecNum

Return the distinct quantile edges a pooled forecast is cut at.

The edges are the `bins + 1` evenly spaced quantiles of the pooling, taken by linear interpolation between the two order statistics each probability falls between — the rule `Statistics.quantile` applies by default, written out here so the cut reads one sorted copy rather than sorting once per probability.

They are answered through `unique`, so a tie that spans an edge leaves one edge rather than two and the bins it would have split collapse into one. A forecast that carries a single distinct value therefore answers a single edge, which [`forecast_calibration_curve`](@ref) reads as its one-bin case.

# Mathematical definition

```math
e_{i} = x_{\\left\\lfloor h_{i} \\right\\rfloor} + \\left( h_{i} - \\left\\lfloor h_{i} \\right\\rfloor \\right) \\left( x_{\\min \\left( \\left\\lfloor h_{i} \\right\\rfloor + 1,\\, n \\right)} - x_{\\left\\lfloor h_{i} \\right\\rfloor} \\right)\\,, \\qquad h_{i} = \\left( n - 1 \\right) \\dfrac{i}{B} + 1\\,.
```

Where:

  - ``x``: The pooled forecast, sorted.
  - ``n``: Its length.
  - ``B``: Number of bins.
  - ``i``: The edge, from ``0`` to ``B``.

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref). It must carry no entry that is not finite.
  - `bins`: Number of quantile bins the edges cut.

# Returns

  - `edges::VecNum`: The distinct edges, in increasing order. It carries at most `bins + 1` of them, and at least one.

# Related

  - [`forecast_calibration_curve`](@ref)
  - [`forecast_calibration_pairs`](@ref)
  - [`forecast_calibration`](@ref)
"""
function forecast_calibration_edges(a::AbstractVector{<:Real}, bins::Integer)
    Tf = real(eltype(a))
    x = sort(a)
    n = length(x)
    e = Vector{Tf}(undef, bins + 1)
    for i in 0:bins
        h = (n - 1) * (i / bins) + 1
        lo = floor(Int, h)
        hi = min(lo + 1, n)
        e[i + 1] = x[lo] + (h - lo) * (x[hi] - x[lo])
    end
    return unique(e)
end
"""
    forecast_calibration_curve(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                               bins::Integer) -> NamedTuple

Return the mean forecast and the mean realised target of each quantile bin of a pooling.

The slope states the scale of a forecast under one number and assumes it holds everywhere; the curve states it bin by bin and therefore shows where it does not. A forecast whose ordering is right and whose scale bends — flat in the middle of the cross-section and steep in the tails, which is the usual shape — reports a plausible slope and a curve that says otherwise.

The bins are quantile bins of the pooled forecast, so each carries roughly the same number of pairs rather than the same width, and the mean forecast of the bins is non-decreasing by construction. A tie that spans a bin edge collapses the two bins into one, because the edges are taken through `unique`, so a forecast that carries fewer distinct values than `bins` reports fewer bins rather than empty ones. A bin that no pair falls in is dropped, so `bin` is the index a bin carries among the edges and not its position in the answer.

# Algorithm

 1. Cut `a` at the `bins + 1` evenly spaced quantiles with [`forecast_calibration_edges`](@ref). Fewer than two distinct edges leave one bin, which every pair falls in.
 2. Place each pair in the bin its forecast falls in, counting the interior edges it stands at or above. The interval is closed on the left.
 3. Average the forecast and the target of each non-empty bin, and count its pairs.

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref).
  - `b`: The target of each pair.
  - `bins`: Number of quantile bins to cut the forecast at.

# Validation

  - `bins >= 1`. Raises a `DomainError`.

# Returns

  - `curve::NamedTuple`: `(; bin, mean_alpha, mean_y, count)`, four vectors of the same length, one entry per non-empty bin, in increasing order of the forecast. An empty pooling gives four empty vectors.

      + `bin::Vector{Int}`: Index of the bin among the edges.
      + `mean_alpha::VecNum`: Mean forecast of the bin.
      + `mean_y::VecNum`: Mean realised target of the bin.
      + `count::Vector{Int}`: Number of pairs in the bin.

# Related

  - [`forecast_calibration`](@ref)
  - [`forecast_calibration_edges`](@ref)
  - [`forecast_calibration_pairs`](@ref)
  - [`forecast_calibration_slope`](@ref)
"""
function forecast_calibration_curve(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                                    bins::Integer)
    @argcheck(bins >= one(bins), DomainError(bins, "bins must be >= 1"))
    Tf = promote_type(real(eltype(a)), real(eltype(b)))
    if isempty(a)
        return (; bin = Vector{Int}(undef, 0), mean_alpha = Vector{Tf}(undef, 0),
                mean_y = Vector{Tf}(undef, 0), count = Vector{Int}(undef, 0))
    end
    edges = forecast_calibration_edges(a, bins)
    nb = max(length(edges) - 1, 1)
    sa = zeros(Tf, nb)
    sy = zeros(Tf, nb)
    n = zeros(Int, nb)
    for k in eachindex(a)
        j = 1
        for e in 2:(length(edges) - 1)
            j += edges[e] <= a[k]
        end
        sa[j] += a[k]
        sy[j] += b[k]
        n[j] += 1
    end
    keep = findall(x -> !iszero(x), n)
    return (; bin = keep, mean_alpha = sa[keep] ./ n[keep], mean_y = sy[keep] ./ n[keep],
            count = n[keep])
end
"""
    forecast_pooled_moments(x::AbstractVector{<:Real}) -> NamedTuple

Return the mean and the standard deviation of a pooled vector.

The two figures are pooled over pairs rather than taken per date and averaged, so they describe the whole sample the calibration was measured on. They are what puts the slope in context: a slope of `2` on a forecast whose standard deviation is a tenth of the target's is the same statement twice, and the two figures are what let a reader see that.

# Arguments

  - `x`: The pooled vector.

# Returns

  - `moments::NamedTuple`: `(; mean, std)`. The mean is `NaN` on an empty vector, and the standard deviation, which removes one degree of freedom, is `NaN` on a vector shorter than two.

# Related

  - [`forecast_calibration`](@ref)
  - [`forecast_calibration_pairs`](@ref)
  - [`forecast_series_summary`](@ref)
"""
function forecast_pooled_moments(x::AbstractVector{<:Real})
    Tf = real(eltype(x))
    n = length(x)
    m = iszero(n) ? Tf(NaN) : sum(x) / n
    v = n > 1 ? Statistics.std(x; mean = m, corrected = true) : Tf(NaN)
    return (; mean = m, std = v)
end
"""
    forecast_calibration(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing;
                         bins::Integer = 10) -> NamedTuple
    forecast_calibration(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                         weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                         bins::Integer = 10) -> NamedTuple

Score whether the magnitude of a Return Forecast is right, and not only its ordering.

This is the third reading of a forecast, and the only one that asks anything of its scale. [`forecast_ic`](@ref) correlates, so it is invariant to a rescaling of the forecast; [`forecast_portfolio`](@ref) rescales every book to 200 % gross, so it is invariant by construction. A forecast that scores well under both can still state a return of five per cent where one is earned, and an optimiser that reads it as a mean will size the position on the five. The slope is the number that catches that, and the curve is where it bends.

# The word calibration is used three other ways in this library

The three are different objects, and only this one is out of sample.

  - An [`AbstractCalibrationAlgorithm`](@ref) computes the radius of an uncertainty set, or another number a slot would otherwise state, from the Prior. ADR 0095 owns it, and it has nothing to do with a forecast.
  - [`idio_calibration`](@ref) states whether a fitted factor model predicted the size of its own idiosyncratic returns, in sample, by the standard deviation of the standardised residuals.
  - `TargetReturnForecast`'s `calib` field is a member's **own** calibration coefficient, one exponentially weighted scalar regression fitted at fit time to put its transformed predictions back into return units. That one acts on the forecast; this verb measures, out of sample, whether the result landed in the right units. A member whose `calib` did its work reports a slope near `1` here, and a member that never calibrated reports whatever scale its predictions happen to carry.

# The threshold does not apply here

Every other statistic of an evaluation is a statistic of a cross-section, so `fe.min_count` refuses one that carries too few assets. This one is not: the slope, the curve and the moments pool the pairs of every evaluation date and read them as one sample, so a thin cross-section contributes few pairs rather than an unreliable number. There is no cross-sectional count to threshold, and this verb takes no `min_count`.

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref).
 2. Pool the scorable pairs with [`forecast_calibration_pairs`](@ref).
 3. Take the slope with [`forecast_calibration_slope`](@ref) and the curve with [`forecast_calibration_curve`](@ref).
 4. Take the pooled moments of the forecast and of the target with [`forecast_pooled_moments`](@ref).

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The curve and the moments read no weights.
  - `csfm`: The fitted factor-model block the evaluation was built on. It supplies the weight history the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history the slope is taken under, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default reads equal weights.
  - `bins`: Number of quantile bins the curve cuts the forecast at.

# Validation

  - The rules of [`forecast_ic_weights`](@ref), of [`forecast_calibration_curve`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `calibration::NamedTuple`: `(; slope, curve, mean_alpha, std_alpha, mean_y, std_y, n_bins)`.

      + `slope::Real`: The scale multiplier, from [`forecast_calibration_slope`](@ref).
      + `curve::NamedTuple`: `(; bin, mean_alpha, mean_y, count)`, from [`forecast_calibration_curve`](@ref).
      + `mean_alpha::Real`: Mean forecast over the pooled pairs.
      + `std_alpha::Real`: Its standard deviation.
      + `mean_y::Real`: Mean realised target over the same pairs.
      + `std_y::Real`: Its standard deviation.
      + `n_bins::Int`: Number of non-empty bins the curve reports.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 3.0 4.0; 2.0 4.0 6.0 8.0; 4.0 3.0 2.0 1.0];

julia> y = PortfolioOptimisers.forward_mean_returns(alpha, 1, 1);

julia> c = forecast_calibration(forecast_evaluation(alpha, y); bins = 2);

julia> (c.slope, c.n_bins, c.curve.count)
(0.6666666666666666, 2, [4, 4])
```

# Related

  - [`forecast_calibration_pairs`](@ref)
  - [`forecast_calibration_slope`](@ref)
  - [`forecast_calibration_edges`](@ref)
  - [`forecast_calibration_curve`](@ref)
  - [`forecast_pooled_moments`](@ref)
  - [`forecast_ic`](@ref)
  - [`forecast_portfolio`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`idio_calibration`](@ref)
"""
function forecast_calibration(fe::ForecastEvaluationResult, w::Option{<:MatNum} = nothing;
                              bins::Integer = 10)
    alpha::MatNum = fe.alpha
    y::MatNum = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    a, b, q = forecast_calibration_pairs(alpha, y, forecast_ic_weights(alpha, w), dates)
    curve = forecast_calibration_curve(a, b, bins)
    ma = forecast_pooled_moments(a)
    my = forecast_pooled_moments(b)
    return (; slope = forecast_calibration_slope(a, b, q), curve = curve,
            mean_alpha = ma.mean, std_alpha = ma.std, mean_y = my.mean, std_y = my.std,
            n_bins = length(curve.bin))
end
function forecast_calibration(fe::ForecastEvaluationResult, csfm::CrossSectionalFactorModel;
                              weighting::AbstractOrthogonalityMetric = IdentityMetric(),
                              bins::Integer = 10)
    return forecast_calibration(fe, cs_diagnostic_weights(weighting, csfm); bins = bins)
end

export forecast_calibration
