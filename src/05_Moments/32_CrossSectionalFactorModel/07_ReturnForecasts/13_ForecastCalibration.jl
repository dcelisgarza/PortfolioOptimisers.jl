"""
    forecast_calibration_pairs(alpha::MatNum, y::MatNum, u::MatNum,
                               dates::AbstractVector{<:Integer})

Pool the scorable pairs of an evaluation into three vectors.

Every statistic of this file reads the pairs of a forecast and its forward target, not the cross-sections that hold them. This function pools the pairs once, and the verbs that follow read vectors. A pair enters where both the forecast and the target are finite, which is the cross-section that [`forecast_cross_section!`](@ref) marks. The pairs follow the order of `dates`, and the pairs of one date follow the order of the assets.

The weight of a pair stays beside the pair and does not scale it, because the readings use it differently. [`forecast_calibration_slope`](@ref) reads the weights, and [`forecast_calibration_curve`](@ref) and the pooled moments read every pair alike. A pair whose weight is not finite gets a weight of zero, so it enters the curve and the moments and takes no part in the slope.

# Mathematical definition

```math
\\begin{align}
\\mathcal{P} &= \\left\\{ \\left( t_{j},\\, i \\right) : \\alpha_{t_{j} i} \\text{ and } y_{t_{j} i} \\text{ are finite} \\right\\}\\,,\\\\
\\left( a_{k},\\, b_{k},\\, q_{k} \\right) &= \\begin{cases}
\\left( \\alpha_{t_{j} i},\\, y_{t_{j} i},\\, u_{t_{j} i} \\right) & u_{t_{j} i} \\text{ is finite}\\,,\\\\
\\left( \\alpha_{t_{j} i},\\, y_{t_{j} i},\\, 0 \\right) & \\text{otherwise}\\,,
\\end{cases}
\\end{align}
```

Where:

  - ``\\mathcal{P}``: The scorable pairs, ordered by ``j`` and then by ``i``. Pair ``k`` is the ``k``-th member ``\\left( t_{j},\\, i \\right)``.
  - $(math_dict[:alpha_ti_fc])
  - $(math_dict[:y_ti_fwd])
  - $(math_dict[:u_ti_cs])
  - $(math_dict[:t_j_eval])
  - $(math_dict[:a_k_pair])
  - $(math_dict[:b_k_pair])
  - $(math_dict[:q_k_pair])

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

The slope comes from a weighted regression of the target on the forecast with **no intercept**. The line passes through the origin, so the slope states the scale and nothing else. A free constant cannot correct a forecast that has the right level for the cross-section and the wrong spread. A slope of `1` says that the forecast is in target units. A slope above `1` says that its magnitude is too small, a slope between `0` and `1` says that its magnitude is too large, and a negative slope says that its sign is wrong.

The verb fits no intercept because a Return Forecast is a cross-sectional statement. The factor model accounts for the cross-sectional mean of the target, and the forecast answers for the rest. A fitted intercept would absorb that mean, and the slope would then measure the scale against a level that the forecast never stated.

# Mathematical definition

```math
\\mathrm{slope} = \\dfrac{\\sum_{k} q_{k} \\, a_{k} \\, b_{k}}{\\sum_{k} q_{k} \\, a_{k}^{2}}
```

Where:

  - $(math_dict[:a_k_pair])
  - $(math_dict[:b_k_pair])
  - $(math_dict[:q_k_pair])

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref).
  - `b`: The target of each pair.
  - `q`: The weight of each pair.

# Returns

  - `slope::Real`: The scale multiplier. It is `NaN` when no pair has a positive weighted square. An empty pooling, a forecast that is zero at every pair, and a weight of zero at every pair all give `NaN`.

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

Return the distinct quantile edges at which the curve cuts a pooled forecast.

The edges are the `bins + 1` evenly spaced quantiles of the pooling. Each edge interpolates linearly between the two order statistics that its probability falls between, which is the default rule of `Statistics.quantile`. This function writes the rule out so that the cut sorts one copy of the forecast, not one copy per probability.

The index ``\\lfloor h_{i} \\rfloor`` and the fraction ``h_{i} - \\lfloor h_{i} \\rfloor`` come from integer arithmetic. So an edge that falls on an order statistic equals that order statistic exactly, and a pair at that value falls in the bin above the edge. The edges agree with `Statistics.quantile` to rounding, not bit for bit.

The function takes the edges through `unique`. A tie that spans an edge therefore leaves one edge and not two, and the bins it would split become one bin. A forecast with one distinct value gives one edge, which [`forecast_calibration_curve`](@ref) reads as its one-bin case.

# Mathematical definition

```math
e_{i} = x_{\\left\\lfloor h_{i} \\right\\rfloor} + \\left( h_{i} - \\left\\lfloor h_{i} \\right\\rfloor \\right) \\left( x_{\\min \\left( \\left\\lfloor h_{i} \\right\\rfloor + 1,\\, n \\right)} - x_{\\left\\lfloor h_{i} \\right\\rfloor} \\right)\\,, \\qquad h_{i} = \\left( n - 1 \\right) \\dfrac{i}{B} + 1\\,.
```

Where:

  - ``e_{i}``: Edge ``i``, before `unique` removes the duplicates.
  - ``x``: The pooled forecast, sorted in increasing order.
  - $(math_dict[:n_pool])
  - ``B``: Number of bins.
  - ``i``: Index of the edge, from ``0`` to ``B``.

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref). It must not be empty, and every entry must be finite.
  - `bins`: Number of quantile bins that the edges cut. It must be at least `1`, which [`forecast_calibration_curve`](@ref) checks.

# Returns

  - `edges::VecNum`: The distinct edges, in increasing order. There are at most `bins + 1` of them, and at least one.

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
        lo, r = divrem((n - 1) * i, bins)
        lo += 1
        hi = min(lo + 1, n)
        e[i + 1] = x[lo] + (Tf(r) / bins) * (x[hi] - x[lo])
    end
    return unique(e)
end
"""
    forecast_calibration_curve(a::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                               bins::Integer) -> NamedTuple

Return the mean forecast and the mean realised target of each quantile bin of a pooling.

The slope states the scale of a forecast in one number, and it assumes that the scale holds everywhere. The curve states the scale bin by bin, so it shows where the scale changes. A common case is a forecast with the right order and a scale that bends, flat in the middle of the cross-section and steep in the tails. That forecast gets a plausible slope and a curve that shows the bend.

The bins are quantile bins of the pooled forecast. Each bin holds about the same number of pairs, not the same width, and the mean forecast rises from each bin to the next. The edges go through `unique`, so a tie that spans an edge joins the two bins on each side of it. A forecast with fewer distinct values than `bins` therefore gives fewer bins, not empty ones. The verb drops a bin that holds no pair, so `bin` is the index of a bin among the edges and not its position in the answer.

# Algorithm

 1. Cut `a` at the `bins + 1` evenly spaced quantiles with [`forecast_calibration_edges`](@ref), giving the edges. Fewer than two distinct edges give one bin, which holds every pair.
 2. Put each pair in the bin of its forecast, giving its bin index. The index is one plus the number of interior edges at or below the forecast, so each bin is closed on the left and the last bin is closed on both sides.
 3. Average the forecast and the target over each non-empty bin, giving `mean_alpha` and `mean_y`, and count its pairs, giving `count`.

# Arguments

  - `a`: The forecast of each pair, from [`forecast_calibration_pairs`](@ref).
  - `b`: The target of each pair.
  - `bins`: Number of quantile bins at which to cut the forecast.

# Validation

  - `bins >= 1`. Raises a `DomainError`.

# Returns

  - `curve::NamedTuple`: `(; bin, mean_alpha, mean_y, count)`, four vectors of the same length. There is one entry per non-empty bin, in increasing order of the forecast. An empty pooling gives four empty vectors.

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

The two figures pool the pairs of every date. They do not average figures taken per date, so they describe the whole sample that the calibration reads. They put the slope in context. When both means are near zero, the slope is about the correlation of the pairs times the ratio of the standard deviation of the target to that of the forecast. The two standard deviations separate that ratio from the correlation.

# Mathematical definition

```math
\\begin{align}
\\bar{v} &= \\frac{1}{n} \\sum_{k = 1}^{n} v_{k}\\,,\\\\
s_{v} &= \\sqrt{\\frac{1}{n - 1} \\sum_{k = 1}^{n} \\left( v_{k} - \\bar{v} \\right)^{2}}\\,.
\\end{align}
```

Where:

  - ``v_{k}``: Entry ``k`` of the pooled vector, which is the forecast or the target of the ``k``-th pair.
  - ``\\bar{v}``: The mean of the pooled vector.
  - ``s_{v}``: The standard deviation of the pooled vector.
  - $(math_dict[:n_pool])

# Arguments

  - `x`: The pooled vector.

# Returns

  - `moments::NamedTuple`: `(; mean, std)`. The mean is `NaN` for an empty vector. The standard deviation divides by ``n - 1``, and it is `NaN` for a vector with fewer than two entries.

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

Score whether the magnitude of a Return Forecast is right, and not only its order.

This is the third reading of a forecast, and the only one that tests its scale. [`forecast_ic`](@ref) correlates, so a positive rescaling of the forecast does not move it. [`forecast_portfolio`](@ref) rescales every book to 200 % gross, so a positive rescaling does not move it either. A forecast can score well on both and still state a return of five per cent where the asset earns one per cent. An optimiser that reads the forecast as a mean then sizes the position on the five. The slope finds that error, and the curve shows where the scale bends.

# Three other uses of the word calibration

The three are different objects, and only this verb reads out of sample.

  - An [`AbstractCalibrationAlgorithm`](@ref) computes a quantity that a slot would otherwise state, such as the radius of an uncertainty set, from the data a prior result carries. It has nothing to do with a forecast.
  - [`idio_calibration`](@ref) states whether a fitted factor model predicted the size of its own idiosyncratic returns. It reads in sample, through the standard deviation of the standardised residuals.
  - The `calib` field of a [`TargetReturnForecastResult`](@ref) is the member's **own** calibration coefficient. The member fits it at fit time, as one exponentially weighted scalar regression, to put its transformed predictions back into return units. That coefficient acts on the forecast, and this verb measures out of sample whether the result is in the right units. A member whose `calib` did its work gets a slope near `1` here. A member that did not calibrate gets whatever scale its predictions carry.

# The threshold does not apply here

Every other statistic of an evaluation is a statistic of a cross-section. Each of them except [`forecast_coverage`](@ref), which explains why a date has no statistic, refuses a cross-section with fewer than `fe.min_count` assets. This verb pools the pairs of every evaluation date and reads them as one sample, so a thin cross-section adds few pairs and not an unreliable number. There is no cross-sectional count to compare with a threshold, and this verb takes no `min_count`.

# Algorithm

 1. Resolve the weight history with [`forecast_ic_weights`](@ref), giving `u`.
 2. Pool the scorable pairs with [`forecast_calibration_pairs`](@ref), giving `a`, `b` and `q`.
 3. Take the slope with [`forecast_calibration_slope`](@ref), and the curve with [`forecast_calibration_curve`](@ref).
 4. Take the pooled moments of `a` and of `b` with [`forecast_pooled_moments`](@ref).

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `w`: Cross-sectional weight history `observations × assets`, on the axis of `fe.alpha`, or `nothing` for equal weights. The curve and the moments read no weights.
  - `csfm`: The fitted factor-model block of the evaluation. It supplies the weight history that the metric names.
  - `weighting`: A member of [`AbstractOrthogonalityMetric`](@ref). It names the weight history of the slope, and [`cs_diagnostic_weights`](@ref) resolves it over the whole observation axis. The default reads equal weights.
  - `bins`: Number of quantile bins at which the curve cuts the forecast.

# Validation

  - The rules of [`forecast_ic_weights`](@ref), of [`forecast_calibration_curve`](@ref) and, for the block method, of [`cs_diagnostic_weights`](@ref).

# Returns

  - `calibration::NamedTuple`: `(; slope, curve, mean_alpha, std_alpha, mean_y, std_y, n_bins)`.

      + `slope::Real`: The scale multiplier, from [`forecast_calibration_slope`](@ref).
      + `curve::NamedTuple`: `(; bin, mean_alpha, mean_y, count)`, from [`forecast_calibration_curve`](@ref).
      + `mean_alpha::Real`: Mean forecast over the pooled pairs.
      + `std_alpha::Real`: Standard deviation of the forecast over the pooled pairs.
      + `mean_y::Real`: Mean realised target over the same pairs.
      + `std_y::Real`: Standard deviation of the target over the same pairs.
      + `n_bins::Int`: Number of non-empty bins in the curve.

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
