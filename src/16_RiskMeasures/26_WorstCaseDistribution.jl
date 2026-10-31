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

The distribution puts all its mass on the observations at the loss of [`worst_positive_weight_loss`](@ref), in proportion to their weights. [`ERM_worst_case`](@ref) reads it at `alpha = 0`, and [`PRM_worst_case`](@ref) reads it under the condition of [`PowerNormValueatRisk`](@ref).

# Mathematical definition

```math
\\begin{align}
L^{*} &= \\underset{t \\in \\mathcal{I}}{\\max} \\; (-x_t)\\,, \\\\
q_t &= \\frac{w_t \\, \\mathbb{1}\\{-x_t = L^{*}\\}}{\\sum_{s \\in \\mathcal{I}} w_s \\, \\mathbb{1}\\{-x_s = L^{*}\\}}\\,, \\quad t \\in \\mathcal{I}\\,, \\\\
q_t &= 0\\,, \\quad t \\notin \\mathcal{I}\\,.
\\end{align}
```

Where:

  - ``L^{*}``: Largest loss among the observations of positive weight.
  - $(math_dict[:q_wcd])
  - $(math_dict[:xret])
  - $(math_dict[:w_t_obs]) Without observation weights, every ``w_t`` is one.
  - $(math_dict[:I_pos_obs])
  - ``\\mathbb{1}\\{\\cdot\\}``: Indicator, one when the condition holds and zero otherwise.

The expected loss under ``\\boldsymbol{q}`` is ``L^{*}``, so ``\\boldsymbol{q}`` reaches the largest loss. When two or more observations tie at ``L^{*}``, every distribution on the tied observations reaches it, and the measure has no gradient. By Theorem 17 of [delbaen2000](@cite), each such distribution gives a subgradient of a coherent measure. The function gives the subgradient that shares the mass by weight.

# Arguments

  - `x`: Return or drawdown series. A loss is a negative entry. The function does not change it.
  - `w`: Resolved observation weights from [`checked_observation_weights`](@ref), or `nothing`.

# Returns

  - `q::Vector`: The distribution, one entry for each observation. Its entries sum to one.

# Related

  - [`worst_positive_weight_loss`](@ref)
  - [`expand_positive_weight_rows`](@ref)

# References

  - $(ref_dict[:delbaen2000]) Section 8, Theorem 17.
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

The measures of the entropic, the relativistic and the power norm families solve a conic programme for their value. [`linearised_risk`](@ref) evaluates the expected loss under the distribution that this function returns, and [`risk_contribution`](@ref) and [`risk_gradient`](@ref) take the finite difference of that expected loss. A finite difference of the measure itself differences two solves, and the noise of the solver is as large as the change that the step makes.

The method on a return series is the one that a measure implements. Each measure of the three families reads its distribution with [`ERM_worst_case`](@ref), [`RRM_worst_case`](@ref) or [`PRM_worst_case`](@ref), on the series that its functor reads. A range measure returns the pair of the distributions of its loss tail and its gain tail, through [`worst_case_pair`](@ref).

A composite that wraps other measures returns a tuple of their distributions. These composites are a vector of measures, [`RiskRatio`](@ref), [`NonOptimisationRiskRatio`](@ref), [`MeanReturnRiskRatio`](@ref), [`RiskTrackingRiskMeasure`](@ref) and [`GenericValueatRiskRange`](@ref). The tuple of the dependent tracking measure also holds the risk of the benchmark, which does not move with the weights. A tail of a [`GenericValueatRiskRange`](@ref) whose value is not a solve has `nothing` in the tuple. [`linearised_risk`](@ref) applies the formula of the composite to the linearised values of the measures that it wraps.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{x} &= \\mathbf{X} \\boldsymbol{w} - F(\\boldsymbol{w})\\,, \\\\
R(\\boldsymbol{w}) &= \\underset{\\boldsymbol{q} \\in \\mathcal{Q}}{\\max} \\; -\\boldsymbol{q}^\\intercal \\boldsymbol{y}(\\boldsymbol{x})\\,, \\\\
\\nabla_{\\boldsymbol{w}} R(\\boldsymbol{w}) &= -\\mathbf{J}^\\intercal \\boldsymbol{q}^{*}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:X_returns])
  - $(math_dict[:w_port])
  - $(math_dict[:F_fee_series])
  - $(math_dict[:R_w])
  - ``\\mathcal{Q}``: Set of distributions of the measure. It is a Kullback-Leibler ball for the entropic family and a Kaniadakis ball for the relativistic family.
  - $(math_dict[:q_wcd])
  - ``\\boldsymbol{q}^{*}``: Maximiser of the expected loss over ``\\mathcal{Q}``, the Worst-Case Distribution that the function returns.
  - $(math_dict[:y_wcd])
  - ``\\mathbf{J}``: Jacobian of ``\\boldsymbol{y}(\\boldsymbol{x})`` with respect to ``\\boldsymbol{w}``, ``T \\times N``. For a measure of the returns without fees, it is ``\\mathbf{X}``.

The measures of the three families are coherent. Proposition 4.1 of [artzner1999](@cite) states that a coherent measure is the largest expected loss over a set of probability measures. That is the second line. Proposition 5 of [delbaen2000](@cite) applies when the maximiser is unique. It states that the derivative of a coherent measure in a direction is then the expected loss of that direction under the maximiser. The chain rule through ``\\boldsymbol{y}`` and ``\\boldsymbol{x}`` then gives the third line, where ``\\boldsymbol{y}`` is differentiable. When the maximiser is not unique, the measure of ``\\boldsymbol{y}`` has no gradient. Theorem 17 of [delbaen2000](@cite) states that each maximiser then gives a subgradient.

# Algorithm

The method on the weights does these steps:

 1. Check the slots of `r` with [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref), as [`expected_risk`](@ref) does.
 2. Read the kind of `r` with [`risk_input_kind`](@ref). For a kind other than [`NetReturnsInput`](@ref), return `nothing`.
 3. Compute the net returns `x` of `w` with [`calc_net_returns`](@ref).
 4. Return the method of `r` on `x`.

A composite replaces these steps with its own method on the weights, which calls this function on each measure that it wraps.

# Arguments

  - $(arg_dict[:r])
  - `x::VecNum`: Net return series of the portfolio.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.

# Validation

  - The method on the weights raises through [`assert_resolved_slots`](@ref) and [`assert_calibrated_slots`](@ref), as [`expected_risk`](@ref) does.

# Returns

  - `q::VecNum`: The Worst-Case Distribution, one entry for each observation. When the maximiser is not unique, it is the maximiser that the solver finds.
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

# References

  - $(ref_dict[:artzner1999]) Section 4.1, Proposition 4.1.
  - $(ref_dict[:delbaen2000]) Section 8, Theorem 17 and Proposition 5.
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

A measure on the returns reads `x` itself. A drawdown measure implements a method that reads its drawdown series instead. A range measure carries a pair, and the gain tail is the loss of the negated series. [`GenericValueatRiskRange`](@ref) implements a method that adds its two tails through [`tail_worst_case_risk`](@ref).

# Mathematical definition

```math
\\begin{align}
\\tilde{R}(\\boldsymbol{x}) &= -\\boldsymbol{q}^\\intercal \\boldsymbol{y}(\\boldsymbol{x})\\,, \\\\
\\tilde{R}_{\\mathrm{range}}(\\boldsymbol{x}) &= -\\boldsymbol{q}_{L}^\\intercal \\boldsymbol{x} + \\boldsymbol{q}_{G}^\\intercal \\boldsymbol{x}\\,.
\\end{align}
```

Where:

  - $(math_dict[:Rtilde_wcd])
  - $(math_dict[:q_wcd])
  - $(math_dict[:y_wcd])
  - ``\\boldsymbol{q}_{L}``, ``\\boldsymbol{q}_{G}``: Worst-Case Distributions of the loss tail and of the gain tail of a range measure.
  - $(math_dict[:xret])

At the series that gave ``\\boldsymbol{q}``, ``\\tilde{R}`` equals the value of the measure, to the tolerance of the solver. Its gradient with respect to ``\\boldsymbol{x}`` is the gradient of the measure, as [`worst_case_distribution`](@ref) states.

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
    linearised_risk(qs::AbstractVector, rs::VecBaseRM, w::VecNum, X::MatNum,
                    fees::Option{<:Fees}; sca::Scalariser = SumScalariser(), kwargs...)

Evaluate the risk of `r` at `w`, or its expected loss under the fixed Worst-Case Distribution `q`, for a finite difference.

Without a distribution, the function returns [`expected_risk`](@ref). With one, it returns [`worst_case_risk`](@ref) of the net returns at `w`. Both have the same gradient at the weights that gave `q`, but the second differences no solve.

A composite, whose `q` is a tuple from [`worst_case_distribution`](@ref), applies its own formula to the linearised values of the measures that it wraps, as its [`expected_risk`](@ref) method does to their values. A vector of measures scalarises them with `sca`, and the tracking measure with independent variables reads the series of [`difference_risk`](@ref) through [`difference_linearised_risk`](@ref).

# Mathematical definition

```math
\\begin{align}
\\tilde{R}(\\boldsymbol{w}) &= -\\boldsymbol{q}^\\intercal \\boldsymbol{y}(\\boldsymbol{x}(\\boldsymbol{w}))\\,, \\\\
\\tilde{R}(\\boldsymbol{w}_0) &= R(\\boldsymbol{w}_0)\\,, \\\\
\\nabla_{\\boldsymbol{w}} \\tilde{R}(\\boldsymbol{w}_0) &= \\nabla_{\\boldsymbol{w}} R(\\boldsymbol{w}_0)\\,.
\\end{align}
```

The composites apply their formulas to the linearised values of the measures that they wrap:

```math
\\begin{align}
\\tilde{R}_{\\mathrm{vec}}(\\boldsymbol{w}) &= S\\left(s_1 \\tilde{R}_1(\\boldsymbol{w}), \\ldots, s_K \\tilde{R}_K(\\boldsymbol{w})\\right)\\,, \\\\
\\tilde{R}_{\\mathrm{ratio}}(\\boldsymbol{w}) &= \\frac{\\tilde{R}_1(\\boldsymbol{w})}{\\tilde{R}_2(\\boldsymbol{w})}\\,, \\\\
\\tilde{R}_{\\mathrm{mean}}(\\boldsymbol{w}) &= \\frac{m(\\boldsymbol{w}) - r_f}{\\tilde{R}_1(\\boldsymbol{w})}\\,, \\\\
\\tilde{R}_{\\mathrm{dep}}(\\boldsymbol{w}) &= \\left| \\tilde{R}_1(\\boldsymbol{w}) - R_1(\\boldsymbol{w}_b) \\right|\\,.
\\end{align}
```

Where:

  - $(math_dict[:Rtilde_wcd])
  - $(math_dict[:q_wcd])
  - $(math_dict[:y_wcd])
  - ``\\boldsymbol{x}(\\boldsymbol{w})``: Net return series of the portfolio at ``\\boldsymbol{w}``.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_0``: Portfolio weights at which the solve gave ``\\boldsymbol{q}``.
  - $(math_dict[:R_w])
  - ``\\tilde{R}_k``, ``R_k``: Linearised risk and risk of the ``k``-th measure that the composite wraps. Without a distribution, ``\\tilde{R}_k = R_k``.
  - $(math_dict[:S_sca_vec])
  - $(math_dict[:s_k_scale])
  - ``K``: Number of measures in the vector.
  - ``m(\\boldsymbol{w})``: Mean net return of the portfolio, the value of the return measure of [`MeanReturnRiskRatio`](@ref).
  - ``r_f``: Risk-free rate, the `rf` field of [`MeanReturnRiskRatio`](@ref).
  - $(math_dict[:w_b_track])

The first line is linear in ``\\boldsymbol{y}``, so its finite difference holds no solver noise. The second and the third lines hold to the tolerance of the solver when ``\\boldsymbol{q}`` is the unique maximiser, as [`worst_case_distribution`](@ref) states. By the chain rule, each composite formula then has the gradient of the composite at ``\\boldsymbol{w}_0``. The ratios cover [`RiskRatio`](@ref) and [`NonOptimisationRiskRatio`](@ref), and ``\\tilde{R}_{\\mathrm{dep}}`` is [`RiskTrackingRiskMeasure`](@ref) with [`DependentVariableTracking`](@ref). With [`IndependentVariableTracking`](@ref), the tracking measure is ``\\tilde{R}_1`` of the active weights.

# Arguments

  - `q`: Worst-Case Distribution from [`worst_case_distribution`](@ref), or `nothing`.
  - `qs`: Worst-Case Distribution of each measure of `rs`, or `nothing` for a measure without one.
  - $(arg_dict[:r])
  - `rs`: Vector of risk measures.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees`: Optional [`Fees`](@ref) structure.
  - $(arg_dict[:sca])

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
# Each tail of a generic range is a measure of its own, so the tuple carries the distribution of
# each tail, and a tail whose value is not a solve keeps its own value through its `nothing`.
function worst_case_distribution(r::GenericValueatRiskRange, x::VecNum)
    return (worst_case_distribution(r.loss, x), worst_case_distribution(r.gain, -x))
end
function worst_case_risk(r::GenericValueatRiskRange, x::VecNum, q::Tuple)
    return tail_worst_case_risk(q[1], r.loss, x) + tail_worst_case_risk(q[2], r.gain, -x)
end
"""
    tail_worst_case_risk(q::Nothing, r::AbstractBaseRiskMeasure, x::VecNum) -> Number
    tail_worst_case_risk(q::VecNum, r::AbstractBaseRiskMeasure, x::VecNum) -> Number

Evaluate one tail of a [`GenericValueatRiskRange`](@ref) on the series `x`, with or without the Worst-Case Distribution of the tail.

Each tail of the range is a measure of its own, so a tail whose value is not a solve keeps its own value. [`worst_case_risk`](@ref) of the range adds the loss tail on the net returns and the gain tail on the negated net returns, as the functor of the range does.

# Arguments

  - `q`: Worst-Case Distribution of the tail from [`worst_case_distribution`](@ref), or `nothing`.
  - `r`: Tail measure, a member of [`ValueatRiskRMs`](@ref).
  - `x::VecNum`: Series of the tail. It is the net returns for the loss tail and the negated net returns for the gain tail.

# Returns

  - `rk::Number`: The value of `r` on `x` without `q`, and [`worst_case_risk`](@ref) of `r` on `x` under `q` with it.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`worst_case_risk`](@ref)
  - [`worst_case_distribution`](@ref)
"""
function tail_worst_case_risk(::Nothing, r::AbstractBaseRiskMeasure, x::VecNum)
    return r(x)
end
function tail_worst_case_risk(q::VecNum, r::AbstractBaseRiskMeasure, x::VecNum)
    return worst_case_risk(r, x, q)
end
"""
    difference_worst_case(r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum, X::MatNum,
                          fees::Option{<:Fees})

Compute the Worst-Case Distribution of `r` on the active weights `wd`, on the series that [`difference_risk`](@ref) reads.

[`RiskTrackingRiskMeasure`](@ref) with [`IndependentVariableTracking`](@ref) measures the risk of the active weights, and charges the fees on the portfolio weights `w`. Without fees, the series is the net return series of `wd`, so the function reads the distribution on the weights. With fees, it reads the distribution of a measure of the kind [`NetReturnsInput`](@ref) on the series that [`difference_risk`](@ref) builds, and it returns `nothing` for every other kind.

With fees, [`difference_risk`](@ref) decomposes each ratio and a nested [`RiskTrackingRiskMeasure`](@ref) into the measures that they wrap. So the function returns a tuple of the distributions of those measures, as [`worst_case_distribution`](@ref) does without fees. The tuple of the dependent tracking measure also holds the risk of its benchmark. The return side of [`MeanReturnRiskRatio`](@ref) and of [`ExpectedReturnRiskRatio`](@ref) is linear, so their tuple holds the distribution of the risk side alone. A vector side gives a vector of the distributions of its measures.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}_d &= \\boldsymbol{w} - \\boldsymbol{w}_b\\,, \\\\
\\boldsymbol{x}_d &= \\mathbf{X} \\boldsymbol{w}_d - F(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}_d``: Active weights vector ``N \\times 1``.
  - $(math_dict[:w_port])
  - $(math_dict[:w_b_track])
  - ``\\boldsymbol{x}_d``: Series of the active weights ``T \\times 1``, the series that [`difference_risk`](@ref) measures.
  - $(math_dict[:X_returns])
  - $(math_dict[:F_fee_series])

The function returns the Worst-Case Distribution of `r` on ``\\boldsymbol{x}_d``.

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
  - `Tuple`: With fees, the distributions of the measures that a ratio or a nested tracking measure wraps.

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

This is [`linearised_risk`](@ref) on the series that [`difference_risk`](@ref) reads. Without a distribution, the function returns [`difference_risk`](@ref) itself. With one, it returns [`worst_case_risk`](@ref) of `r` on the series of the active weights that [`difference_worst_case`](@ref) states. With fees, a ratio and a nested tracking measure apply their formulas to the linearised values of the measures that they wrap, as [`difference_risk`](@ref) does to their values. A vector side scalarises the linearised values of its measures with the scalariser of the ratio.

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
                                    X::MatNum, ::Nothing; kwargs...)
    return linearised_risk(q, r, wd, X, nothing; kwargs...)
end
function difference_linearised_risk(::Nothing, r::AbstractBaseRiskMeasure, wd::VecNum,
                                    w::VecNum, X::MatNum, fees::Fees; kwargs...)
    return difference_risk(r, wd, w, X, fees; kwargs...)
end
function difference_linearised_risk(q, r::AbstractBaseRiskMeasure, wd::VecNum, w::VecNum,
                                    X::MatNum, fees::Fees; kwargs...)
    return worst_case_risk(r, charge_fees(X * wd, w, fees), q)
end
# With fees, a composite that `difference_risk` decomposes carries the distributions of the
# measures that it wraps, as `worst_case_distribution` does without fees. A composite declares
# no input kind, so the generic method cannot read one for it. A vector side carries one
# distribution per measure and scalarises their linearised values, as `linearised_risk` does.
function difference_worst_case(rs::VecBaseRM, wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return map(r -> difference_worst_case(r, wd, w, X, fees), rs)
end
function difference_linearised_risk(qs::AbstractVector, rs::VecBaseRM, wd::VecNum,
                                    w::VecNum, X::MatNum, fees::Fees;
                                    sca::Scalariser = SumScalariser())
    return scalarise(sca, zip(qs, rs)) do (q, r)
        return difference_linearised_risk(q, r, wd, w, X, fees) * r.settings.scale
    end
end
function difference_worst_case(r::RkRatioRM, wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return (difference_worst_case(r.r1, wd, w, X, fees),
            difference_worst_case(r.r2, wd, w, X, fees))
end
function difference_linearised_risk(q::Tuple, r::RiskRatio, wd::VecNum, w::VecNum,
                                    X::MatNum, fees::Fees; kwargs...)
    return difference_linearised_risk(q[1], r.r1, wd, w, X, fees) /
           difference_linearised_risk(q[2], r.r2, wd, w, X, fees)
end
function difference_linearised_risk(q::Tuple, r::NonOptimisationRiskRatio, wd::VecNum,
                                    w::VecNum, X::MatNum, fees::Fees; kwargs...)
    return difference_linearised_risk(q[1], r.r1, wd, w, X, fees; kwargs..., sca = r.sca1) /
           difference_linearised_risk(q[2], r.r2, wd, w, X, fees; kwargs..., sca = r.sca2)
end
# The return of `MeanReturnRiskRatio` is linear in the series, so its value needs no
# distribution, as in `linearised_risk`.
function difference_worst_case(r::MeanReturnRiskRatio, wd::VecNum, w::VecNum, X::MatNum,
                               fees::Fees)
    return (difference_worst_case(r.rk, wd, w, X, fees),)
end
function difference_linearised_risk(q::Tuple, r::MeanReturnRiskRatio, wd::VecNum, w::VecNum,
                                    X::MatNum, fees::Fees; kwargs...)
    return (difference_risk(r.rt, wd, w, X, fees) - r.rf) /
           difference_linearised_risk(q[1], r.rk, wd, w, X, fees; kwargs..., sca = r.sca)
end
function difference_worst_case(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:IndependentVariableTracking},
                               wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return (difference_worst_case(r.r, wd - r.tr.w, w, X, fees),)
end
function difference_linearised_risk(q::Tuple,
                                    r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                    wd::VecNum, w::VecNum, X::MatNum, fees::Fees; kwargs...)
    return difference_linearised_risk(q[1], r.r, wd - r.tr.w, w, X, fees)
end
function difference_worst_case(r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:DependentVariableTracking},
                               wd::VecNum, w::VecNum, X::MatNum, fees::Fees)
    return (difference_worst_case(r.r, wd, w, X, fees), expected_risk(r.r, r.tr.w, X, fees))
end
function difference_linearised_risk(q::Tuple,
                                    r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                               <:DependentVariableTracking},
                                    wd::VecNum, w::VecNum, X::MatNum, fees::Fees; kwargs...)
    return abs(difference_linearised_risk(q[1], r.r, wd, w, X, fees) - q[2])
end
