"""
    expand_positive_weight_rows(qr::VecNum, rows::Base.OneTo, T::Integer) -> VecNum
    expand_positive_weight_rows(qr::VecNum, rows::AbstractVector{<:Integer}, T::Integer)
        -> Vector

Expand a vector with one entry for each observation of positive weight to one entry for each observation.

A programme of [`ERM`](@ref), [`RRM`](@ref) or [`PRM`](@ref) has one row for each index of [`positive_weight_rows`](@ref). The Worst-Case Distribution that [`ERM_worst_case`](@ref), [`RRM_worst_case`](@ref) and [`PRM_worst_case`](@ref) read off those rows has the same length. An observation of weight zero has no probability, so its entry is zero.

# Arguments

  - `qr`: Vector with one entry for each index of `rows`.
  - `rows`: Indices from [`positive_weight_rows`](@ref).
  - `T`: Number of observations.

# Returns

  - `qr` itself when `rows` holds every observation. Otherwise, a new vector of length `T` that holds `qr` at `rows` and zero elsewhere.

# Related

  - [`positive_weight_rows`](@ref)
  - [`worst_loss_distribution`](@ref)
"""
function expand_positive_weight_rows(qr::VecNum, ::Base.OneTo, ::Integer)
    return qr
end
function expand_positive_weight_rows(qr::VecNum, rows::AbstractVector{<:Integer},
                                     T::Integer)
    q = zeros(eltype(qr), T)
    q[rows] = qr
    return q
end
"""
    worst_loss_distribution(x::VecNum, ::Nothing) -> Vector
    worst_loss_distribution(x::VecNum, w::VecNum) -> Vector

Return the Worst-Case Distribution of a measure whose value is the largest loss among the observations of positive weight.

The distribution puts all its mass on the observations at the loss of [`worst_positive_weight_loss`](@ref), in proportion to their weights. So its expected loss is that largest loss. [`ERM_worst_case`](@ref) reads it at `alpha = 0`, and [`PRM_worst_case`](@ref) reads it under the condition of [`PowerNormValueatRisk`](@ref). At a tie of the largest loss, the gradient of the measure is not defined, and the distribution gives the subgradient that shares the mass by weight.

# Arguments

  - `x`: Return or drawdown series. A loss is a negative entry. Not modified.
  - `w`: Resolved observation weights from [`checked_observation_weights`](@ref), or `nothing`.

# Returns

  - `q::Vector`: The distribution, one entry for each observation. Its entries sum to one.

# Related

  - [`worst_positive_weight_loss`](@ref)
  - [`expand_positive_weight_rows`](@ref)
"""
function worst_loss_distribution(x::VecNum, ::Nothing)
    l = worst_positive_weight_loss(x, nothing)
    p = map(v -> ifelse(-v == l, one(v), zero(v)), x)
    return p / sum(p)
end
function worst_loss_distribution(x::VecNum, w::VecNum)
    l = worst_positive_weight_loss(x, w)
    p = zero.(w)
    for t in positive_weight_rows(w, length(w))
        p[t] = ifelse(-x[t] == l, w[t], p[t])
    end
    return p / sum(p)
end
"""
    element_worst_case_distributions(ps::Nothing, r::AbstractBaseRiskMeasure, w::VecNum,
                                     X::MatNum, fees::Option{<:Fees})
    element_worst_case_distributions(ps::VecNum, rs::VecBaseRM, w::VecNum, X::MatNum,
                                     fees::Option{<:Fees}) -> Vector

Compute the Worst-Case Distribution of one measure, or of each element of a vector of measures, at the unperturbed weights `w`.

[`risk_contribution`](@ref) takes it once, before its finite difference, and [`adjusted_risk`](@ref) reads it at each perturbed weight vector. An element whose chain-rule weight is zero adds nothing to the risk, so the function does not solve its programme, and gives it `nothing`.

# Arguments

  - `ps`: Chain-rule weights of a vector of measures, or `nothing` for a single measure.
  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Returns

  - The value of [`worst_case_distribution`](@ref) for a single measure, and a vector of them for a vector of measures.

# Related

  - [`worst_case_distribution`](@ref)
  - [`adjusted_risk`](@ref)
  - [`scalariser_element_weights`](@ref)
"""
function element_worst_case_distributions(::Nothing, r::AbstractBaseRiskMeasure, w::VecNum,
                                          X::MatNum, fees::Option{<:Fees})
    return worst_case_distribution(r, w, X, fees)
end
function element_worst_case_distributions(ps::VecNum, rs::VecBaseRM, w::VecNum, X::MatNum,
                                          fees::Option{<:Fees})
    return map(ps, rs) do p, r
        return iszero(p) ? nothing : worst_case_distribution(r, w, X, fees)
    end
end
"""
    worst_case_distribution(r::AbstractBaseRiskMeasure, x::VecNum)
    worst_case_distribution(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                            fees::Option{<:Fees})

Compute the Worst-Case Distribution of a measure whose value is a solve, or return `nothing` for every other measure.

The measures of the entropic, the relativistic and the power norm families are each the largest expected loss over a set of distributions. Each solves a conic programme for its value, and the Worst-Case Distribution is the distribution in the set that reaches it. By the envelope theorem, the gradient of the measure with respect to the weights equals the gradient of the expected loss under that distribution, with the distribution held fixed. [`linearised_risk`](@ref) evaluates that expected loss, and [`risk_contribution`](@ref) and [`risk_gradient`](@ref) take its finite difference. A finite difference of the measure itself differences two solves, and the noise of the solver is as large as the change that the step makes.

The method on a return series is the one that a measure implements. Each measure of the three families reads its distribution with [`ERM_worst_case`](@ref), [`RRM_worst_case`](@ref) or [`PRM_worst_case`](@ref), on the series that its functor reads. A range measure returns the pair of the distributions of its loss tail and its gain tail, through [`worst_case_pair`](@ref). The method on the weights checks the slots of `r` as [`expected_risk`](@ref) does, and calls the method on the series for a measure of the kind [`NetReturnsInput`](@ref).

A composite that wraps other measures returns a tuple of their distributions, from the method on the weights. These composites are a vector of measures, [`RiskRatio`](@ref), [`NonOptimisationRiskRatio`](@ref), [`MeanReturnRiskRatio`](@ref) and [`RiskTrackingRiskMeasure`](@ref). The tuple of the dependent tracking measure also holds the risk of the benchmark, which does not move with the weights. [`linearised_risk`](@ref) applies the formula of the composite to the linearised values of the measures that it wraps.

# Arguments

  - $(arg_dict[:r])
  - `x::VecNum`: Net return series of the portfolio.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Validation

  - The method on the weights raises through [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref), as [`expected_risk`](@ref) does.

# Returns

  - `q::VecNum`: The Worst-Case Distribution, one entry for each observation.
  - `(q_loss, q_gain)`: The distributions of the two tails of a range measure.
  - `Tuple`: The distributions of the measures that a composite wraps.
  - `Vector`: The distribution of each element of a vector of measures.
  - `nothing`: For a measure whose value is not a solve, and when no solver solves the programme or gives its dual.

# Related

  - [`worst_case_risk`](@ref)
  - [`linearised_risk`](@ref)
  - [`element_worst_case_distributions`](@ref)
  - [`ERM_worst_case`](@ref)
  - [`RRM_worst_case`](@ref)
  - [`PRM_worst_case`](@ref)
"""
function worst_case_distribution(::AbstractBaseRiskMeasure, ::VecNum)
    return nothing
end
function worst_case_distribution(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                                 fees::Option{<:Fees})
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return worst_case_distribution(risk_input_kind(r), r, w, X, fees)
end
function worst_case_distribution(::RiskInputKind, ::AbstractBaseRiskMeasure, ::VecNum,
                                 ::MatNum, ::Option{<:Fees})
    return nothing
end
function worst_case_distribution(::NetReturnsInput, r::AbstractBaseRiskMeasure, w::VecNum,
                                 X::MatNum, fees::Option{<:Fees})
    return worst_case_distribution(r, calc_net_returns(w, X, fees))
end
"""
    worst_case_pair(q_loss::VecNum, q_gain::VecNum) -> Tuple
    worst_case_pair(q_loss, q_gain) -> Nothing

Pair the Worst-Case Distributions of the loss tail and the gain tail of a range measure.

A range measure is the sum of a measure of the losses and a measure of the gains. When the programme of either tail gives no distribution, the pair is `nothing`, so that [`linearised_risk`](@ref) evaluates the measure itself.

# Arguments

  - `q_loss`: Worst-Case Distribution of the loss tail, or `nothing`.
  - `q_gain`: Worst-Case Distribution of the gain tail, read on the negated series, or `nothing`.

# Returns

  - `(q_loss, q_gain)` when both are vectors. Otherwise, `nothing`.

# Related

  - [`worst_case_distribution`](@ref)
  - [`worst_case_risk`](@ref)
"""
function worst_case_pair(q_loss::VecNum, q_gain::VecNum)
    return (q_loss, q_gain)
end
function worst_case_pair(::Any, ::Any)
    return nothing
end
"""
    worst_case_risk(r::AbstractBaseRiskMeasure, x::VecNum, q::VecNum) -> Number
    worst_case_risk(r::AbstractBaseRiskMeasure, x::VecNum, q::Tuple) -> Number

Evaluate the expected loss of the series that `r` reads from the net returns `x`, under the fixed Worst-Case Distribution `q`.

At the series that gave `q`, the value equals the value of `r`, to the tolerance of the solver. Its gradient with respect to `x` is the gradient of `r`. A measure on the returns reads `x` itself. A drawdown measure implements a method that reads its drawdown series instead. A range measure carries a pair, and the gain tail is the loss of the negated series.

# Mathematical definition

```math
\\begin{align}
\\tilde{R}(\\boldsymbol{x}) &= -\\boldsymbol{q}^\\intercal \\boldsymbol{y}(\\boldsymbol{x})\\,, \\\\
\\tilde{R}_{\\mathrm{range}}(\\boldsymbol{x}) &= -\\boldsymbol{q}_{L}^\\intercal \\boldsymbol{x} + \\boldsymbol{q}_{G}^\\intercal \\boldsymbol{x}\\,.
\\end{align}
```

Where:

  - ``\\tilde{R}``: Expected loss under the fixed Worst-Case Distribution.
  - ``\\boldsymbol{q}``: Worst-Case Distribution, one entry for each observation.
  - ``\\boldsymbol{y}(\\boldsymbol{x})``: Series that the measure reads: the returns, the absolute drawdowns or the relative drawdowns.
  - ``\\boldsymbol{q}_{L}``, ``\\boldsymbol{q}_{G}``: Worst-Case Distributions of the loss tail and of the gain tail of a range measure.
  - $(math_dict[:xret])

# Arguments

  - $(arg_dict[:r])
  - `x::VecNum`: Net return series of the portfolio.
  - `q`: Worst-Case Distribution, or the pair of a range measure.

# Returns

  - `rk::Number`: The expected loss under `q`.

# Related

  - [`worst_case_distribution`](@ref)
  - [`linearised_risk`](@ref)
"""
function worst_case_risk(::AbstractBaseRiskMeasure, x::VecNum, q::VecNum)
    return -LinearAlgebra.dot(q, x)
end
function worst_case_risk(::AbstractBaseRiskMeasure, x::VecNum, q::Tuple)
    return LinearAlgebra.dot(q[2], x) - LinearAlgebra.dot(q[1], x)
end
"""
    linearised_risk(q::Nothing, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                    fees::Option{<:Fees}; kwargs...)
    linearised_risk(q, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                    fees::Option{<:Fees}; kwargs...)

Evaluate the risk of `r` at `w`, or its expected loss under the fixed Worst-Case Distribution `q`, for a finite difference.

Without a distribution, the function returns [`expected_risk`](@ref). With one, it returns [`worst_case_risk`](@ref) of the net returns at `w`. Both have the same gradient at the weights that gave `q`, but the second differences no solve.

A composite, whose `q` is a tuple from [`worst_case_distribution`](@ref), applies its own formula to the linearised values of the measures that it wraps, as its [`expected_risk`](@ref) method does to their values. A vector of measures scalarises them with `sca`, and the tracking measure with independent variables reads the series of [`difference_risk`](@ref) through [`difference_linearised_risk`](@ref).

# Arguments

  - `q`: Worst-Case Distribution from [`worst_case_distribution`](@ref), or `nothing`.
  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Returns

  - `rk::Number`: The risk, or the expected loss under `q`.

# Related

  - [`worst_case_distribution`](@ref)
  - [`worst_case_risk`](@ref)
  - [`adjusted_risk`](@ref)
  - [`measure_gradient`](@ref)
"""
function linearised_risk(::Nothing, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; kwargs...)
    return expected_risk(r, w, X, fees; kwargs...)
end
function linearised_risk(q, r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; kwargs...)
    return worst_case_risk(r, calc_net_returns(w, X, fees), q)
end
# A composite carries the distributions of the measures that it wraps, in a tuple, and applies
# its own formula to their linearised values, as its `expected_risk` method does to their
# values. The tuple is never `nothing`, so these methods never meet the plain route, and an
# inner measure that is not a solve takes `expected_risk` through its own `nothing`.
function worst_case_distribution(rs::VecBaseRM, w::VecNum, X::MatNum, fees::Option{<:Fees})
    return map(r -> worst_case_distribution(r, w, X, fees), rs)
end
function linearised_risk(qs::AbstractVector, rs::VecBaseRM, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; sca::Scalariser = SumScalariser(), kwargs...)
    return scalarise(sca, zip(qs, rs)) do (q, r)
        return linearised_risk(q, r, w, X, fees; kwargs...) * r.settings.scale
    end
end
function worst_case_distribution(r::RkRatioRM, w::VecNum, X::MatNum, fees::Option{<:Fees})
    return (worst_case_distribution(r.r1, w, X, fees),
            worst_case_distribution(r.r2, w, X, fees))
end
function linearised_risk(q::Tuple, r::RiskRatio, w::VecNum, X::MatNum, fees::Option{<:Fees};
                         kwargs...)
    return linearised_risk(q[1], r.r1, w, X, fees; kwargs...) /
           linearised_risk(q[2], r.r2, w, X, fees; kwargs...)
end
function linearised_risk(q::Tuple, r::NonOptimisationRiskRatio, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; kwargs...)
    return linearised_risk(q[1], r.r1, w, X, fees; kwargs..., sca = r.sca1) /
           linearised_risk(q[2], r.r2, w, X, fees; kwargs..., sca = r.sca2)
end
function worst_case_distribution(r::MeanReturnRiskRatio, w::VecNum, X::MatNum,
                                 fees::Option{<:Fees})
    return (worst_case_distribution(r.rk, w, X, fees),)
end
function linearised_risk(q::Tuple, r::MeanReturnRiskRatio, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; kwargs...)
    return (expected_risk(r.rt, w, X, fees; kwargs...) - r.rf) /
           linearised_risk(q[1], r.rk, w, X, fees; kwargs..., sca = r.sca)
end
function worst_case_distribution(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                            <:IndependentVariableTracking},
                                 w::VecNum, X::MatNum, fees::Option{<:Fees})
    return (difference_worst_case(r.r, w - r.tr.w, w, X, fees),)
end
function linearised_risk(q::Tuple,
                         r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                    <:IndependentVariableTracking},
                         w::VecNum, X::MatNum, fees::Option{<:Fees}; kwargs...)
    return difference_linearised_risk(q[1], r.r, w - r.tr.w, w, X, fees)
end
# The risk of the benchmark does not move with `w`, so the tuple carries its value, and the
# finite difference does not solve it again.
function worst_case_distribution(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                            <:DependentVariableTracking},
                                 w::VecNum, X::MatNum, fees::Option{<:Fees})
    return (worst_case_distribution(r.r, w, X, fees), expected_risk(r.r, r.tr.w, X, fees))
end
function linearised_risk(q::Tuple,
                         r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                    <:DependentVariableTracking}, w::VecNum,
                         X::MatNum, fees::Option{<:Fees}; kwargs...)
    return abs(linearised_risk(q[1], r.r, w, X, fees) - q[2])
end
"""
    difference_worst_case(r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                          fees::Option{<:Fees})

Compute the Worst-Case Distribution of `r` on the active weights `wd`, on the series that [`difference_risk`](@ref) reads.

[`RiskTrackingRiskMeasure`](@ref) with [`IndependentVariableTracking`](@ref) measures the risk of the active weights, and charges the fees on the portfolio weights `w`. Without fees, the series is the net return series of `wd`, so the function reads the distribution on the weights. With fees, it reads the distribution of a measure of the kind [`NetReturnsInput`](@ref) on the series that [`difference_risk`](@ref) builds, and every other kind gets `nothing`.

# Arguments

  - $(arg_dict[:r])
  - `wd::VecNum`: Active weights, the portfolio weights minus the benchmark weights.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Validation

  - With fees, the function raises through [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref), as the functor of the tracking measure does.

# Returns

  - The value of [`worst_case_distribution`](@ref) on the series of the active weights.

# Related

  - [`difference_risk`](@ref)
  - [`difference_linearised_risk`](@ref)
  - [`worst_case_distribution`](@ref)
"""
function difference_worst_case(r::AbstractBaseRiskMeasure, wd::VecNum, ::VecNum, X::MatNum,
                               ::Nothing)
    return worst_case_distribution(r, wd, X, nothing)
end
function difference_worst_case(r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                               fees::Fees)
    assert_resolved_slots(r)
    assert_calibrated_slots(r)
    return difference_worst_case(risk_input_kind(r), r, wd, w, X, fees)
end
function difference_worst_case(::RiskInputKind, ::AbstractBaseRiskMeasure, ::VecNum,
                               ::VecNum, ::MatNum, ::Fees)
    return nothing
end
function difference_worst_case(::NetReturnsInput, r::AbstractBaseRiskMeasure, wd::VecNum,
                               w::VecNum, X::MatNum, fees::Fees)
    return worst_case_distribution(r, charge_fees(X * wd, w, fees))
end
"""
    difference_linearised_risk(q, r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum,
                               X::MatNum, fees::Option{<:Fees})

Evaluate the risk of `r` on the active weights `wd`, or its expected loss under the fixed Worst-Case Distribution `q`, for a finite difference.

This is [`linearised_risk`](@ref) on the series that [`difference_risk`](@ref) reads. Without a distribution, the function returns [`difference_risk`](@ref) itself.

# Arguments

  - `q`: Worst-Case Distribution from [`difference_worst_case`](@ref), or `nothing`.
  - $(arg_dict[:r])
  - `wd::VecNum`: Active weights, the portfolio weights minus the benchmark weights.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Returns

  - `rk::Number`: The risk of the active weights, or its expected loss under `q`.

# Related

  - [`difference_risk`](@ref)
  - [`difference_worst_case`](@ref)
  - [`linearised_risk`](@ref)
"""
function difference_linearised_risk(q, r::AbstractBaseRiskMeasure, wd::VecNum, ::VecNum,
                                    X::MatNum, ::Nothing)
    return linearised_risk(q, r, wd, X, nothing)
end
function difference_linearised_risk(::Nothing, r::AbstractBaseRiskMeasure, wd::VecNum,
                                    w::VecNum, X::MatNum, fees::Fees)
    return difference_risk(r, wd, w, X, fees)
end
function difference_linearised_risk(q, r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum,
                                    X::MatNum, fees::Fees)
    return worst_case_risk(r, charge_fees(X * wd, w, fees), q)
end
