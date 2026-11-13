"""
    expected_return(ret::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    expected_return(ret::NoReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; kwargs...)
    expected_return(ret::JRE_VecJRE, w::VecVecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)

Compute the expected return of a portfolio with a return estimator.

`expected_return` reads the portfolio weights, a prior result and optional transaction fees. `fees` is a **positional** argument after `pr` on every method.

The prior route reduces to the Investable Mask. A prior result holds every asset of the universe, and an asset that it cannot estimate carries `NaN` in `mu` and down its column of `pr.X`. On the whole universe, `dot(w, mu)` and `X * w` are `NaN` at every weight, a zero weight included. So each method reduces the prior, the weights and the fees once at its entry, through [`investable_reduction`](@ref). An optimiser does the same reduction at its entry. [`NoReturn`](@ref) gives zero at every weight, so it reads nothing and reduces nothing.

The reduction changes nothing on a prior that is already reduced. The moments of a reduced prior are finite everywhere, so [`investable_mask`](@ref) gives `nothing`. Thus a vector of terms, a population and a composite such as [`expected_ratio`](@ref) each give one diagnostic only.

Each method is the scalar twin of the `ret` expression that `set_return_constraints!` builds for the same estimator, so the two sides charge the same fee. For a [`NoReturn`](@ref), the model builds a zero expression and charges no fee on it. `settings.fee` has no effect on that term, and the scalar twin charges no fee either.

The fourth method takes a vector of weight vectors. It applies one of the first three methods to each weight vector, and returns one expected return for each.

# Mathematical definition

```math
\\begin{align}
R_{\\mathrm{a}}(\\boldsymbol{w}) &= \\boldsymbol{w}^{\\intercal} \\boldsymbol{\\mu} - F(\\boldsymbol{w})\\\\
R_{\\mathrm{l}}(\\boldsymbol{w}) &= \\dfrac{\\sum\\limits_{t=1}^{T} v_{t} \\ln\\left(1 + \\boldsymbol{x}_{t}^{\\intercal} \\boldsymbol{w}\\right)}{\\sum\\limits_{t=1}^{T} v_{t}} - F(\\boldsymbol{w})\\\\
R_{\\mathrm{n}}(\\boldsymbol{w}) &= 0\\,.
\\end{align}
```

Where:

  - ``R_{\\mathrm{a}}(\\boldsymbol{w})``: Expected return of an [`ArithmeticReturn`](@ref).
  - ``R_{\\mathrm{l}}(\\boldsymbol{w})``: Expected return of a [`LogarithmicReturn`](@ref), the mean log-growth rate of the portfolio.
  - ``R_{\\mathrm{n}}(\\boldsymbol{w})``: Expected return of a [`NoReturn`](@ref). It has no fee term, so it is zero for every value of `fees`.
  - $(math_dict[:w_port])
  - $(math_dict[:mu_er])
  - $(math_dict[:x_t_obs])
  - $(math_dict[:T])
  - ``v_{t}``: Observation weight of observation ``t``, the ``t``-th entry of the `w` field of a [`LogarithmicReturn`](@ref), or of the observation weights `pr.w` of the prior when that field is `nothing`. Every weight is one when both are `nothing`, and the quotient is then the unweighted mean.
  - ``F(\\boldsymbol{w})``: Fee term of [`term_fees`](@ref). It is [`calc_fees`](@ref) of `w` and `fees` when the term's `settings.fee` is `true`, and zero otherwise.

# Arguments

  - `ret`: Return estimator, or a vector of them on the fourth method.
  - `w`: Portfolio weights, or a vector of portfolio weight vectors on the fourth method.
  - `pr`: Prior result. It resolves a Deferred Quantity of `ret`, through [`factory`](@ref). It gives `mu` to an [`ArithmeticReturn`](@ref) whose own `mu` is `nothing`, and its observation weights to a [`LogarithmicReturn`](@ref) whose own `w` is `nothing`.
  - `fees`: Optional fees.
  - $(arg_dict[:strict_inv]) A [`NoReturn`](@ref) reads no weights, so `strict` has no effect on it.
  - `kwargs...`: Keyword arguments that the methods do not read.

# Returns

  - `rt::Number`: Expected portfolio return, net of fees, on the first three methods.
  - `rt::VecNum`: One expected portfolio return for each weight vector, on the fourth method.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`NoReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`VecNum`](@ref)
  - [`VecVecNum`](@ref)
  - [`Num_VecNum`](@ref)
  - [`Option`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_risk_ret_sric`](@ref)
  - [`term_fees`](@ref): It charges ``F(\\boldsymbol{w})``, and it holds the rule that decides when to charge it.
  - [`calc_fees`](@ref)
"""
function expected_return(r::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask. A non-investable asset carries `NaN` in `mu`, so
    # `dot(w, mu)` is `NaN` at any weight, the optimiser's own zero included. The reduction
    # is a no-op on a prior that is already reduced, which is what keeps a composite to one
    # diagnostic.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    # The scalar twin of the `ret` expression `set_return_constraints!` builds, so it takes
    # the same ladder: the `factory` that the model calls resolves a Deferred Quantity and
    # fills the slots from the prior.
    return resolved_expected_return(factory(r, pr), w, w, pr.X, fees)
end
function expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask, and take the ladder of the model, as the arithmetic twin
    # above does. The ladder falls back to the observation weights of the prior.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    return resolved_expected_return(factory(ret, pr), w, w, pr.X, fees)
end
function expected_return(::NoReturn, w::VecNum, ::AbstractPriorResult,
                         ::Option{<:Fees} = nothing; kwargs...)
    # The scalar twin of a zero expression. No charge is applied, exactly as
    # `set_return_constraints!` applies none: `settings.fee` is inert on this term.
    return zero(eltype(w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Charge the fees to a return term when the `settings.fee` of the term is `true`.

The scalar twin reads the **fee** flag only. When `settings.mic` is `true`, the model
expression also subtracts a market impact cost, through `add_market_impact_cost!`. No method
of [`expected_return`](@ref) charges that cost, for one term or for a vector of terms. So the
model expression and its scalar twin differ by the market impact cost, and this function
does not change that difference.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: Optional fees.
  - `T`: Number of observations of the fit. The function spreads the one-off fees over them.
  - `fee`: The `settings.fee` flag of the term.

# Returns

  - `f::Number`: The charge for one period. It is [`calc_periodic_fees`](@ref) plus [`calc_one_off_fees`](@ref) divided by `T` when `fee` is `true`. It is a zero of the element type of `w` when `fee` is `false` or `fees` is `nothing`.

# Related

  - [`expected_return`](@ref)
  - [`JuMPReturnsSettings`](@ref)
  - [`calc_fees`](@ref)
"""
function term_fees(w::VecNum, ::Nothing, ::Number, ::Bool)
    return zero(eltype(w))
end
function term_fees(w::VecNum, fees::Fees, T::Number, fee::Bool)
    if !fee
        return zero(eltype(w))
    end
    # An expected return is a per period number, so the one-off terms are always spread over
    # the observation count of the fit, whatever clock `fees.fa` names for a realised series.
    return calc_periodic_fees(w, fees) + calc_one_off_fees(w, fees) / T
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sum the value-level returns of many return terms, each at its own scale.

This method is the scalar twin of the `ret` expression of the model, which is the same
weighted sum over the same terms. The method skips a term whose `settings.rte` is `false`, and
the model skips it too.

The barrier of [`NearOptimalCentering`](@ref) reads the aggregate return of all the terms as
one number, and this method gives that number. [`expected_ratio`](@ref) and the other
value-level functions of this file call this method when `ret` is a vector.

# Mathematical definition

```math
\\begin{align}
R(\\boldsymbol{w}) &= \\sum_{j \\,:\\, e_{j}} s_{j} \\, R_{j}(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``R(\\boldsymbol{w})``: Aggregate expected return of the whole vector of terms.
  - $(math_dict[:w_port])
  - ``R_{j}(\\boldsymbol{w})``: Expected return of the ``j``-th term, net of its own fee.
  - ``s_{j}``: `settings.scale` of the ``j``-th term.
  - ``e_{j}``: `settings.rte` of the ``j``-th term. The sum runs over the terms whose flag is `true`.

# Algorithm

 1. Reduce the prior, the weights and the fees to the Investable Mask once, with [`investable_reduction`](@ref). A held asset outside the mask gives one diagnostic, and not one for each term.
 2. Resolve each term against the prior with [`factory`](@ref), as the model does.
 3. Return [`resolved_expected_return`](@ref) of the resolved terms. It skips each term whose `settings.rte` is `false`, and it sums the `settings.scale` of each other term times its return. The method for one term reads the same function, so the two give the same value.

# Arguments

  - `ret`: Vector of return estimators.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional fees.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that the method does not read.

# Returns

  - `rt::Number`: Aggregate expected portfolio return of the terms whose `settings.rte` is `true`.

# Related

  - [`expected_return`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function expected_return(ret::VecJRE, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # The vector reduces once, so a held non-investable asset is named once and not once per
    # term. The sum over the terms is the one of the risk path, so the two cannot drift.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    return resolved_expected_return(factory(ret, pr), w, w, pr.X, fees)
end
function expected_return(ret::JRE_VecJRE, w::VecVecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # The population reduces once, on the mask any member holds an asset under, so the
    # diagnostic is raised once rather than once per member.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    return [expected_return(ret, wi, pr, fees; kwargs...) for wi in w]
end
"""
    expected_ratio(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                   pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                   rf::Number = 0, sca::Scalariser = SumScalariser(),
                   strict::Bool = false, kwargs...)

Compute the ratio of the expected return of a portfolio to its expected risk.

`expected_ratio` divides the expected portfolio return, net of the fees and of the risk-free rate, by the expected portfolio risk. The risk measure `r` gives the risk, and the return estimator `ret` gives the return.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ratio}(\\boldsymbol{w}) &= \\dfrac{R(\\boldsymbol{w}) - r_{f}}{\\mathrm{sca}\\left(\\left\\{c_{i}\\, \\rho_{i}(\\boldsymbol{w})\\right\\}\\right)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ratio}(\\boldsymbol{w})``: Risk-adjusted return ratio.
  - $(math_dict[:w_port])
  - ``R(\\boldsymbol{w})``: Aggregate expected return of `ret`, the sum of its terms at their own `settings.scale` weights, each net of `fees`.
  - ``r_{f}``: Risk-free rate, the `rf` keyword.
  - ``\\rho_{i}``: The ``i``-th risk measure in `r`.
  - ``c_{i}``: `settings.scale` of the ``i``-th risk measure.
  - ``\\mathrm{sca}``: Scalariser held in `sca`, which reduces the risk axis to one number.

# Multiplicity

Each axis takes one item or a vector of items, and the ratio divides one **aggregate** by the other. `sca` scalarises the risk axis into one number. The return axis sums at the `settings.scale` weights of its terms. Then the function divides the two aggregates.

Only the risk axis has a scalariser. The return axis sums and never scalarises.

# Resolution

Both axes receive `pr` itself, so both apply the same precedence rule. A stated slot comes first, then the estimator of the slot, then the field of the prior. The risk axis goes through the prior route of [`expected_risk`](@ref). That route resolves the measure through [`resolve_risk_inputs`](@ref) once for each call. It fits a Deferred Quantity against `pr`, and an unstated slot takes the field of the prior.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser that reduces a vector `r` to one number. It has no effect on a single measure.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that go to [`expected_risk`](@ref) and to [`expected_return`](@ref).

# Returns

  - `ratio::Number`: Risk-adjusted return ratio.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_ratio(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                        pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                        rf::Number = 0, sca::Scalariser = SumScalariser(),
                        strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask here, at the outermost entry the caller reached, so a
    # held non-investable asset is named once. Both children reduce again on the reduced
    # prior, where every moment is finite and the mask is `nothing`, so neither repeats it.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    # Both axes are handed the prior itself, never `pr.X`. The risk axis then enters
    # `expected_risk`'s own prior route, which resolves the measure through
    # `resolve_risk_inputs` exactly once. Unwrapping the matrix here dropped the prior
    # fallback for an unstated slot and refused a Deferred Quantity that this argument list
    # was already holding the prior for.
    rk = expected_risk(sca, r, w, pr, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return (rt - rf) / rk
end
"""
    expected_risk_ret_ratio(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                            pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                            rf::Number = 0, sca::Scalariser = SumScalariser(),
                            strict::Bool = false, kwargs...)

Compute the expected risk, the expected return and the risk-adjusted return ratio of a portfolio.

`expected_risk_ret_ratio` returns the three numbers as a tuple. The risk measure `r` gives the risk, and the return estimator `ret` gives the return.

# Mathematical definition

```math
\\begin{align}
\\mathrm{risk}(\\boldsymbol{w}) &= \\mathrm{sca}\\left(\\left\\{c_{i}\\, \\rho_{i}(\\boldsymbol{w})\\right\\}\\right)\\\\
\\mathrm{return}(\\boldsymbol{w}) &= R(\\boldsymbol{w})\\\\
\\mathrm{ratio}(\\boldsymbol{w}) &= \\dfrac{\\mathrm{return}(\\boldsymbol{w}) - r_{f}}{\\mathrm{risk}(\\boldsymbol{w})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{risk}(\\boldsymbol{w})``, ``\\mathrm{return}(\\boldsymbol{w})``, ``\\mathrm{ratio}(\\boldsymbol{w})``: The three entries of the returned tuple, in that order.
  - $(math_dict[:w_port])
  - ``R(\\boldsymbol{w})``: Aggregate expected return of `ret`, the sum of its terms at their own `settings.scale` weights, each net of `fees`.
  - ``r_{f}``: Risk-free rate, the `rf` keyword.
  - ``\\rho_{i}``: The ``i``-th risk measure in `r`.
  - ``c_{i}``: `settings.scale` of the ``i``-th risk measure.
  - ``\\mathrm{sca}``: Scalariser held in `sca`, which reduces the risk axis to one number.

# Multiplicity

The rule is the same as for [`expected_ratio`](@ref). Each axis takes one item or a vector of items, and the ratio divides one **aggregate** by the other. `sca` scalarises the risk axis into one number, and the return axis sums at the `settings.scale` weights of its terms. The returned `risk` and `return` are those two aggregates. They are never a breakdown for each item.

Only the risk axis has a scalariser. The return axis sums and never scalarises.

# The figures of an optimisation result

Use this function to get the figures of an optimisation result. A result holds the measure and the scalariser of its solve, so the call can read them from the result:

```julia
rk, rt, rr = expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca, rf = rf)
```

A result stores `res.r` and `res.ret` **resolved**. So this call gives the aggregates that the optimisation optimised, for one item or for many items on each axis. A caller can also name the measures by hand. Then the figure is equal to that of the optimisation only when the caller names the same measures **and** the same scalariser. The same holds for `fees` and `rf`.

A [`NonRiskJuMPOptimisationResult`](@ref) has no `r`, because its solve resolved no measure. With such a result, the caller must name a measure.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser that reduces a vector `r` to one number. It has no effect on a single measure.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that go to [`expected_risk`](@ref) and to [`expected_return`](@ref).

# Returns

  - `risk::Number`: Expected portfolio risk.
  - `return::Number`: Expected portfolio return.
  - `ratio::Number`: Risk-adjusted return ratio.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
  - [`expected_sric`](@ref)
"""
function expected_risk_ret_ratio(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                                 pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                                 rf::Number = 0, sca::Scalariser = SumScalariser(),
                                 strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask once here — see the note in `expected_ratio`.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    # The prior, not `pr.X` — see the note in `expected_ratio`.
    rk = expected_risk(sca, r, w, pr, fees; kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the estimation-error penalty that changes a ratio into a SRIC.

The SRIC functions subtract the penalty once from the aggregate ratio, and never from each
item. The penalty is the only difference between [`expected_sric`](@ref) and
[`expected_ratio`](@ref), and between [`expected_risk_ret_sric`](@ref) and
[`expected_risk_ret_ratio`](@ref). So both SRIC functions call this function, and they do not
state the ratio again.

# Mathematical definition

```math
\\begin{align}
P(\\mathrm{sr}) &= \\dfrac{N}{T \\, \\mathrm{sr}}\\,.
\\end{align}
```

Where:

  - ``P(\\mathrm{sr})``: Estimation-error penalty of the ratio ``\\mathrm{sr}``.
  - ``\\mathrm{sr}``: Aggregate risk-adjusted return ratio to penalise.
  - $(math_dict[:T])
  - $(math_dict[:N])

# Arguments

  - `sr`: Aggregate risk-adjusted return ratio.
  - `pr`: Prior result. Its returns matrix gives ``T`` and ``N``.

# Returns

  - `p::Number`: The estimation-error penalty.

# Related

  - [`expected_sric`](@ref)
  - [`expected_risk_ret_sric`](@ref)
  - [`AbstractPriorResult`](@ref)
"""
function sric_penalty(sr::Number, pr::AbstractPriorResult)
    T, N = size(pr.X)
    return N / (T * sr)
end
"""
    expected_sric(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                  pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                  rf::Number = 0, sca::Scalariser = SumScalariser(),
                  strict::Bool = false, kwargs...)

Compute the Sharpe ratio information criterion (SRIC) of a portfolio.

The SRIC corrects the risk-adjusted return ratio for the estimation error of an optimisation, which makes the in-sample ratio too large. `expected_sric` subtracts from the ratio a penalty that reads the number of assets and the number of observations.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SRIC}(\\boldsymbol{w}) &= \\mathrm{ratio}(\\boldsymbol{w}) - P\\left(\\mathrm{ratio}(\\boldsymbol{w})\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{SRIC}(\\boldsymbol{w})``: Sharpe ratio information criterion.
  - $(math_dict[:w_port])
  - ``\\mathrm{ratio}(\\boldsymbol{w})``: Risk-adjusted return ratio of [`expected_ratio`](@ref), which states its own closed form.
  - ``P``: Estimation-error penalty of [`sric_penalty`](@ref), which states its own closed form.

# Multiplicity

The function penalises the ratio of [`expected_ratio`](@ref), so the rule is the same as for that function. The ratio divides one **aggregate** by the other. `sca` scalarises the risk axis, and the return axis sums at the `settings.scale` weights of its terms. The function subtracts the penalty once from the scalar ratio, and not from each item.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser that reduces a vector `r` to one number. It has no effect on a single measure.
  - $(arg_dict[:strict_inv]) The reduction also sets the number of assets of the penalty to the number of assets that the prior can estimate.
  - `kwargs...`: Keyword arguments that go to [`expected_ratio`](@ref).

# Returns

  - `sric::Number`: Sharpe ratio information criterion.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
  - [`expected_risk_ret_sric`](@ref)
"""
function expected_sric(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                       pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                       rf::Number = 0, sca::Scalariser = SumScalariser(),
                       strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask once here, so the penalty's asset count is the count
    # of assets the prior could estimate — see the note in `expected_ratio`.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    sr = expected_ratio(r, ret, w, pr, fees; rf = rf, sca = sca, kwargs...)
    return sr - sric_penalty(sr, pr)
end
"""
    expected_risk_ret_sric(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                           pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                           rf::Number = 0, sca::Scalariser = SumScalariser(),
                           strict::Bool = false, kwargs...)

Compute the expected risk, the expected return and the SRIC of a portfolio.

`expected_risk_ret_sric` returns the three numbers as a tuple. The Sharpe ratio information criterion (SRIC) corrects the risk-adjusted return ratio for estimation error, as [`expected_sric`](@ref) states.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SRIC}(\\boldsymbol{w}) &= \\mathrm{ratio}(\\boldsymbol{w}) - P\\left(\\mathrm{ratio}(\\boldsymbol{w})\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{SRIC}(\\boldsymbol{w})``: Sharpe ratio information criterion, the third entry of the returned tuple. The first two entries are those of [`expected_risk_ret_ratio`](@ref), which states their closed forms.
  - $(math_dict[:w_port])
  - ``\\mathrm{ratio}(\\boldsymbol{w})``: Risk-adjusted return ratio of [`expected_risk_ret_ratio`](@ref).
  - ``P``: Estimation-error penalty of [`sric_penalty`](@ref), which states its own closed form.

# Multiplicity

The tuple is the tuple of [`expected_risk_ret_ratio`](@ref), with the penalty subtracted from its third entry. So the rule is the same as for that function, and the ratio divides one **aggregate** by the other. The returned `risk` and `return` are the two aggregates, and never a breakdown for each item. The function subtracts the penalty once from the scalar ratio.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result. Its returns matrix `X` gives the two counts of the penalty.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate. It defaults to `0`.
  - `sca`: Scalariser that reduces a vector `r` to one number. It has no effect on a single measure.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that go to [`expected_risk_ret_ratio`](@ref).

# Returns

  - `risk::Number`: Expected portfolio risk.
  - `return::Number`: Expected portfolio return.
  - `sric::Number`: Sharpe ratio information criterion.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_sric`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk_ret_ratio`](@ref)
"""
function expected_risk_ret_sric(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                                pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                                rf::Number = 0, sca::Scalariser = SumScalariser(),
                                strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask once here — see the note in `expected_sric`.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    rk, rt, sr = expected_risk_ret_ratio(r, ret, w, pr, fees; rf = rf, sca = sca, kwargs...)
    return rk, rt, sr - sric_penalty(sr, pr)
end
"""
$(DocStringExtensions.TYPEDEF)

Risk measure whose value is the expected portfolio return of a prior result.

`ExpectedReturn` is a [`NonOptimisationRiskMeasure`](@ref), so an optimiser cannot use it as an objective or as a constraint. Use it where a function takes a risk measure and the figure must be the expected return. For example, it is the default `y` axis of [`plot_measures`](@ref) and of [`plot_efficient_frontier`](@ref). A larger value is a better value, as [`PerfRM`](@ref) states.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ER}(\\boldsymbol{w}) &= \\sum_{j \\,:\\, e_{j}} s_j\\, R_j(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ER}(\\boldsymbol{w})``: Expected portfolio return, the value of the measure.
  - $(math_dict[:w_port])
  - ``R_j``: Expected return of the ``j``-th term in `rt`, net of `fees`.
  - ``s_j``: `settings.scale` of the ``j``-th term. A single `rt` has one term.
  - ``e_j``: `settings.rte` of the ``j``-th term. The sum runs over the terms whose flag is `true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectedReturn(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        rt::JRE_VecJRE = ArithmeticReturn()
    ) -> ExpectedReturn

Keywords correspond to the struct's fields.

## Multiplicity

`rt` takes one return term or a vector of them, so `ExpectedReturn(; rt = res.ret)` works after a solve with many terms. The measure sums a vector at the `settings.scale` weights of its terms, and it skips a term whose `settings.rte` is `false`.

The type has **no** scalariser. The return axis sums and never scalarises.

## Validation

  - If `rt` is a vector: `!isempty(rt)`.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `rt`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> ExpectedReturn()
ExpectedReturn
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
        rt ┼ ArithmeticReturn
           │   settings ┼ JuMPReturnsSettings
           │            │   scale ┼ Float64: 1.0
           │            │      lb ┼ nothing
           │            │     rte ┼ Bool: true
           │            │     fee ┼ Bool: true
           │            │     mic ┴ Bool: true
           │        ucs ┼ nothing
           │         mu ┼ nothing
           │   mtx_sqrt ┴ EigenFallbackSquareRoot()
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`expected_return`](@ref)
  - [`expected_risk`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ExpectedReturn <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:rt])
    """
    @fprop rt
    function ExpectedReturn(settings::HierarchicalRiskMeasureSettings, rt::JRE_VecJRE)
        if isa(rt, VecJRE)
            @argcheck(!isempty(rt), IsEmptyError("rt cannot be empty"))
        end
        return new{typeof(settings), typeof(rt)}(settings, rt)
    end
end
function ExpectedReturn(;
                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                        rt::JRE_VecJRE = ArithmeticReturn())::ExpectedReturn
    return ExpectedReturn(settings, rt)
end
# Deferrable slots — see `deferred_slots`. The return term carries its own `mu`, so both the
# check and the derived recursion in `resolve_deferred_quantities` reach it through `rt`.
deferred_slots(r::ExpectedReturn) = (; rt = r.rt)
# Expected-risk input kind — see `risk_input_kind`. The `factory` of the risk path fills the
# moments of the return term from the prior, so the resolved measure reads the weights, the
# returns matrix and the fees alone, and every composite of that path reaches it.
risk_input_kind(::ExpectedReturn) = WeightsReturnsFeesInput()
function (r::ExpectedReturn)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return resolved_expected_return(r.rt, w, w, X, fees)
end
function difference_risk(r::ExpectedReturn, wd::VecNum, w::VecNum, X::MatNum, fees::Fees;
                         kwargs...)
    return resolved_expected_return(r.rt, wd, w, X, fees)
end
"""
    expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; kwargs...)

Compute the value of an [`ExpectedReturn`](@ref), the expected return of a portfolio.

The method calls [`expected_return`](@ref) on the `rt` field of the measure. [`ExpectedReturn`](@ref) states the closed form.

The measure also has a route on a returns matrix. It declares [`WeightsReturnsFeesInput`](@ref), and its functor `r(w, X, fees)` calls [`resolved_expected_return`](@ref) on `rt`. That route needs the `mu` of an `rt` that [`factory`](@ref) resolved, and the prior route of [`risk_contribution`](@ref), [`risk_gradient`](@ref) and of each composite measure gives it. Both routes read one ladder, so they give the same value.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) that holds a return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Keyword arguments that go to [`expected_return`](@ref), `strict` included.

# Returns

  - `risk::Number`: Expected portfolio return, net of `fees` when they are given.

# Related

  - [`ExpectedReturn`](@ref)
  - [`expected_return`](@ref)
  - [`expected_ratio`](@ref)
"""
function expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return expected_return(r.rt, w, pr, fees; kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Risk measure whose value is the ratio of the expected return of a portfolio to its expected risk.

`ExpectedReturnRiskRatio` holds a return estimator, a risk measure and a risk-free rate. Its value is the risk-adjusted return ratio of [`expected_ratio`](@ref), for example the Sharpe ratio when `rk` is a [`StandardDeviation`](@ref). It is a [`NonOptimisationRiskMeasure`](@ref), so an optimiser cannot use it as an objective or as a constraint. A larger value is a better value, as [`PerfRM`](@ref) states.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ERRR}(\\boldsymbol{w}) &= \\frac{\\sum_{j \\,:\\, e_{j}} s_j\\, R_j(\\boldsymbol{w}) - r_f}{\\mathrm{sca}\\left(\\left\\{c_i\\, \\rho_i(\\boldsymbol{w})\\right\\}\\right)}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ERRR}(\\boldsymbol{w})``: Expected return to risk ratio.
  - $(math_dict[:w_port])
  - ``R_j``: Expected return of the ``j``-th term in `rt`, net of `fees`.
  - ``s_j``: `settings.scale` of the ``j``-th term of `rt`.
  - ``e_j``: `settings.rte` of the ``j``-th term of `rt`. The sum runs over the terms whose flag is `true`.
  - ``r_f``: Risk-free rate, the `rf` field.
  - ``\\rho_i``: The ``i``-th risk measure in `rk`.
  - ``c_i``: `settings.scale` of the ``i``-th risk measure.
  - ``\\mathrm{sca}``: Scalariser held in `sca`, which reduces the risk axis to one number.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpectedReturnRiskRatio(;
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        rt::JRE_VecJRE = ArithmeticReturn(),
        rk::BaseRM_VecBaseRM = Variance(),
        sca::Scalariser = SumScalariser(),
        rf::Number = 0.0,
    ) -> ExpectedReturnRiskRatio

Keywords correspond to the struct's fields.

## Multiplicity

Each axis takes one item or a vector of items, and only the **risk** axis has a scalariser. `rt` sums its terms at their `settings.scale` weights, and it skips a term whose `settings.rte` is `false`. `sca` scalarises `rk` into one number. Then the ratio divides one aggregate by the other. The return axis sums and never scalarises.

The field takes precedence over a `sca` keyword of the caller. A `sca` that the caller gives at the call site goes no further than this type. So the figure of an `ExpectedReturnRiskRatio` is always the figure that the type names.

## Validation

  - If `rt` is a vector: `!isempty(rt)`.
  - If `rk` is a vector: `!isempty(rk)`.
  - $(val_dict[:rf])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `rt`: Recursively updated through [`factory`](@ref).
  - `rk`: Recursively updated through [`factory`](@ref).

# Examples

```jldoctest
julia> ExpectedReturnRiskRatio()
ExpectedReturnRiskRatio
  settings ┼ HierarchicalRiskMeasureSettings
           │   scale ┴ Float64: 1.0
        rt ┼ ArithmeticReturn
           │   settings ┼ JuMPReturnsSettings
           │            │   scale ┼ Float64: 1.0
           │            │      lb ┼ nothing
           │            │     rte ┼ Bool: true
           │            │     fee ┼ Bool: true
           │            │     mic ┴ Bool: true
           │        ucs ┼ nothing
           │         mu ┼ nothing
           │   mtx_sqrt ┴ EigenFallbackSquareRoot()
        rk ┼ Variance
           │    settings ┼ RiskMeasureSettings
           │             │   scale ┼ Float64: 1.0
           │             │      ub ┼ nothing
           │             │     rke ┴ Bool: true
           │       sigma ┼ nothing
           │        chol ┼ nothing
           │          rc ┼ nothing
           │         alg ┼ SquaredSOCRiskExpr()
           │    mtx_sqrt ┼ EigenFallbackSquareRoot()
           │   ratio_alg ┼ SquareRootRatio()
           │      corner ┴ nothing
       sca ┼ SumScalariser()
        rf ┴ Float64: 0.0
```

# Related

  - [`JuMPReturnsEstimator`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_risk`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct ExpectedReturnRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:rt])
    """
    @fprop rt
    """
    $(field_dict[:rk])
    """
    @fprop rk
    """
    $(field_dict[:sca_rk])
    """
    sca
    """
    $(field_dict[:rf])
    """
    rf
    function ExpectedReturnRiskRatio(settings::HierarchicalRiskMeasureSettings,
                                     rt::JRE_VecJRE, rk::BaseRM_VecBaseRM, sca::Scalariser,
                                     rf::Number)
        if isa(rt, VecJRE)
            @argcheck(!isempty(rt), IsEmptyError("rt cannot be empty"))
        end
        if isa(rk, VecBaseRM)
            @argcheck(!isempty(rk), IsEmptyError("rk cannot be empty"))
        end
        @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
        return new{typeof(settings), typeof(rt), typeof(rk), typeof(sca), typeof(rf)}(settings,
                                                                                      rt,
                                                                                      rk,
                                                                                      sca,
                                                                                      rf)
    end
end
function ExpectedReturnRiskRatio(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 rt::JRE_VecJRE = ArithmeticReturn(),
                                 rk::BaseRM_VecBaseRM = Variance(),
                                 sca::Scalariser = SumScalariser(),
                                 rf::Number = 0.0)::ExpectedReturnRiskRatio
    return ExpectedReturnRiskRatio(settings, rt, rk, sca, rf)
end
# Deferrable slots — see `deferred_slots`. Each axis is one item or a vector of them, and the
# derived recursion resolves a vector element by element.
deferred_slots(r::ExpectedReturnRiskRatio) = (; rt = r.rt, rk = r.rk)
"""
    reader_frames(r::Union{ExpectedReturn, ExpectedReturnRiskRatio}, path::AbstractString,
                  refit::Bool) -> ReturnsFrames

Return the frames of the readers in a risk measure that holds a return term, for the frame walk of a head.

The return term `rt` can hold an uncertainty set with a prior of its own, as [`ArithmeticReturn`](@ref) states. The risk side `rk` of an [`ExpectedReturnRiskRatio`](@ref) can hold a risk measure with a reader. Each of those readers belongs to the group of the head that holds the measure. So the method reads every field of `r` with [`field_frames`](@ref). A field that holds no reader adds no frame.

# Arguments

  - `r`: The risk measure.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])

# Returns

  - `fr::ReturnsFrames`: The frames of the fields of `r`.

# Related

  - [`reader_frames`](@ref)
  - [`field_frames`](@ref)
  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
"""
function reader_frames(r::Union{ExpectedReturn, ExpectedReturnRiskRatio},
                       path::AbstractString, refit::Bool)
    return field_frames(r, path, refit)
end
"""
    expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; kwargs...)

Compute the value of an [`ExpectedReturnRiskRatio`](@ref), the risk-adjusted return ratio of a portfolio.

The method calls [`expected_ratio`](@ref) with the fields of the measure. [`ExpectedReturnRiskRatio`](@ref) states the closed form.

The call puts `kwargs...` first and the `sca` and `rf` fields of the measure after it. So the fields take precedence over a `sca` or an `rf` that the caller gives at the call site. The figure of an [`ExpectedReturnRiskRatio`](@ref) is always the figure that the type names.

The measure is also a composite on a returns matrix, as [`MeanReturnRiskRatio`](@ref) is. Its methods of [`expected_risk`](@ref), [`worst_case_distribution`](@ref) and [`linearised_risk`](@ref) on `X::MatNum` divide [`resolved_expected_return`](@ref) of `rt`, minus `rf`, by the risk of `rk`. They need an `rt` that [`factory`](@ref) resolved, and the prior route of [`risk_contribution`](@ref), [`risk_gradient`](@ref) and of each composite measure gives it. A weight path, `w::MatNum`, throws an `ArgumentError`.

# Arguments

  - `r`: [`ExpectedReturnRiskRatio`](@ref) that holds a return estimator, a risk measure and a risk-free rate.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Keyword arguments that go to [`expected_ratio`](@ref), `strict` included. A `sca` or an `rf` among them has no effect.

# Returns

  - `risk::Number`: Risk-adjusted return ratio.

# Related

  - [`ExpectedReturnRiskRatio`](@ref)
  - [`VecNum`](@ref)
  - [`AbstractPriorResult`](@ref)
  - [`Option`](@ref)
  - [`Fees`](@ref)
  - [`expected_ratio`](@ref)
  - [`expected_return`](@ref)
"""
function expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
    # `kwargs...` is splatted **first** and the fields pinned last, so a `sca` supplied at the
    # call site loses to the one this wrapper holds. Without the order a caller's keyword would
    # flow through `expected_ratio` into the inner vector call and silently win.
    return expected_ratio(r.rk, r.rt, w, pr, fees; kwargs..., rf = r.rf, sca = r.sca)
end
function expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, X::MatNum,
                       fees::Option{<:Fees} = nothing; kwargs...)
    return (resolved_expected_return(r.rt, w, w, X, fees) - r.rf) /
           expected_risk(r.sca, r.rk, w, X, fees; kwargs...)
end
function expected_risk(r::ExpectedReturnRiskRatio, w::MatNum, ::MatNum,
                       ::Option{<:Fees} = nothing; kwargs...)
    return throw(ArgumentError("`$(nameof(typeof(r)))` reads the expected return of one cross-section of weights. A `w::MatNum` is a weight path, one row of weights per observation, which is a different quantity.\nScore this measure against the target weight vector, `w::VecNum`, which is the first row of the path.\nGot\nsize(w) => $(size(w))"))
end
# The composite carries the distribution of its risk measure, as `MeanReturnRiskRatio` does.
function worst_case_distribution(r::ExpectedReturnRiskRatio, w::VecNum, X::MatNum,
                                 fees::Option{<:Fees})
    return (worst_case_distribution(r.rk, w, X, fees),)
end
function linearised_risk(q::Tuple, r::ExpectedReturnRiskRatio, w::VecNum, X::MatNum,
                         fees::Option{<:Fees}; kwargs...)
    return (resolved_expected_return(r.rt, w, w, X, fees) - r.rf) /
           linearised_risk(r.sca, q[1], r.rk, w, X, fees; kwargs...)
end
# The active weights of an independent tracking measure read the expected return of `wd` minus
# the fee of `w`, as `difference_risk` of `ExpectedReturn` does.
function difference_risk(r::ExpectedReturnRiskRatio, wd::VecNum, w::VecNum, X::MatNum,
                         fees::Fees; kwargs...)
    return (resolved_expected_return(r.rt, wd, w, X, fees) - r.rf) /
           difference_risk(r.sca, r.rk, wd, w, X, fees)
end
function difference_worst_case(r::ExpectedReturnRiskRatio, wd::VecNum, w::VecNum, X::MatNum,
                               fees::Fees)
    return (difference_worst_case(r.rk, wd, w, X, fees),)
end
function difference_linearised_risk(q::Tuple, r::ExpectedReturnRiskRatio, wd::VecNum,
                                    w::VecNum, X::MatNum, fees::Fees; kwargs...)
    return (resolved_expected_return(r.rt, wd, w, X, fees) - r.rf) /
           difference_linearised_risk(r.sca, q[1], r.rk, wd, w, X, fees)
end
function needs_previous_weights(r::ExpectedReturnRiskRatio)
    return needs_previous_weights(r.rk)
end
"""
    homogeneity_degree(rt::JuMPReturnsEstimator, fees::Option{<:Fees} = nothing) -> Option{<:Integer}
    homogeneity_degree(rt::VecJRE, fees::Option{<:Fees} = nothing) -> Option{<:Integer}
    homogeneity_degree(r::ExpectedReturn, fees::Option{<:Fees} = nothing,
                       T::Option{<:Integer} = nothing) -> Option{<:Integer}
    homogeneity_degree(r::ExpectedReturnRiskRatio, fees::Option{<:Fees} = nothing,
                       T::Option{<:Integer} = nothing) -> Option{<:Integer}

Return the degree of homogeneity of a return estimator in the portfolio weights, under the fees `fees`, and of the two measures that read one.

The arithmetic return ``\\boldsymbol{\\mu}^{\\intercal} \\boldsymbol{w}`` has degree 1. The logarithmic return ``\\mathbb{E}[\\log(1 + \\boldsymbol{x}^{\\intercal} \\boldsymbol{w})]`` is not homogeneous, so it has no degree. A vector of return estimators sums its terms whose `settings.rte` is `true`, so it has no degree when one of those terms has none.

A term whose `settings.fee` is `true` subtracts the fees, through [`term_fees`](@ref). A fee that does not scale with the weights, from [`has_fixed_fee`](@ref), then takes the degree from the term. A constant shift moves the return, so no such fee cancels. [`NoReturn`](@ref) charges no fee.

# Algorithm

 1. Return `nothing` for [`LogarithmicReturn`](@ref), and `1` for [`NoReturn`](@ref).
 2. For every other return estimator, return `nothing` when `settings.fee` is `true` and [`has_fixed_fee`](@ref) finds a fee. Otherwise return `1`.
 3. For a vector, return `nothing` when a term whose `settings.rte` is `true` has no degree. Otherwise return `1`.
 4. For [`ExpectedReturn`](@ref), return the degree of `rt`.
 5. For [`ExpectedReturnRiskRatio`](@ref), return `nothing` when `rf` is not zero. Otherwise subtract the degree of `rk`, from [`scalarised_degree`](@ref), from the degree of `rt` with [`degree_difference`](@ref).

# Arguments

  - `rt`: Return estimator, or a vector of them.
  - $(arg_dict[:r])
  - `fees`: The fees that the return pays, or `nothing`.
  - `T`: The number of observations of the net returns, or `nothing` when it is not known. A return estimator does not read it, and the risk measure `rk` of a ratio does.

# Returns

  - `d::Option{<:Integer}`: The degree of homogeneity, or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.homogeneity_degree(ExpectedReturn(; rt = LogarithmicReturn()))

julia> PortfolioOptimisers.homogeneity_degree(ExpectedReturn(), Fees(; fl = 0.001))

julia> PortfolioOptimisers.homogeneity_degree(ExpectedReturnRiskRatio())
-1

julia> PortfolioOptimisers.homogeneity_degree(ExpectedReturnRiskRatio(; rk = StandardDeviation()))
0
```

# Related

  - [`homogeneity_degree`](@ref)
  - [`has_fixed_fee`](@ref)
  - [`expected_return`](@ref)
  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
"""
function homogeneity_degree(rt::JuMPReturnsEstimator, fees::Option{<:Fees} = nothing)
    return rt.settings.fee && has_fixed_fee(fees, false) ? nothing : 1
end
function homogeneity_degree(::NoReturn, ::Option{<:Fees} = nothing)
    return 1
end
function homogeneity_degree(::LogarithmicReturn, ::Option{<:Fees} = nothing)
    return nothing
end
function homogeneity_degree(rt::VecJRE, fees::Option{<:Fees} = nothing)
    unbounded = any(r -> r.settings.rte && isnothing(homogeneity_degree(r, fees)), rt)
    return unbounded ? nothing : 1
end
function homogeneity_degree(r::ExpectedReturn, fees::Option{<:Fees} = nothing,
                            ::Option{<:Integer} = nothing)
    return homogeneity_degree(r.rt, fees)
end
function homogeneity_degree(r::ExpectedReturnRiskRatio, fees::Option{<:Fees} = nothing,
                            T::Option{<:Integer} = nothing)
    if !iszero(r.rf)
        return nothing
    end
    return degree_difference(homogeneity_degree(r.rt, fees),
                             scalarised_degree(r.rk, r.sca, fees, T))
end
"""
    const PerfRM = Union{MeanReturn, MeanReturnRiskRatio, ExpectedReturn,
                         ExpectedReturnRiskRatio}

Union of the risk measures whose value is a performance figure, a return or a ratio of return to risk.

Each member gives a **performance** figure and not a loss, so a larger value is a better value. [`bigger_is_better`](@ref) dispatches on this alias and gives `true` for the four members in one method. That method is the only reason for the group.

# Related

  - [`MeanReturn`](@ref): It reads a realised return series.
  - [`MeanReturnRiskRatio`](@ref): It reads a realised return series.
  - [`ExpectedReturn`](@ref): It reads a prior result, so it is also a [`PrRM`](@ref).
  - [`ExpectedReturnRiskRatio`](@ref): It reads a prior result, so it is also a [`PrRM`](@ref).
  - [`PrRM`](@ref): The two members of this group that read a prior result.
  - [`bigger_is_better`](@ref): The method that dispatches on this alias.
"""
const PerfRM = Union{MeanReturn, MeanReturnRiskRatio, ExpectedReturn,
                     ExpectedReturnRiskRatio}
"""
    const PrRM = Union{ExpectedReturn, ExpectedReturnRiskRatio}

Union of the return risk measures that read a prior result, and that refuse a [`PredictionResult`](@ref). On a prediction result, use [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) in their place.

Both members read the **prior result** through their return estimator, which a return series does not hold: an [`ArithmeticReturn`](@ref) reads the expected returns `mu`, and a [`LogarithmicReturn`](@ref) reads the observation weights. The [`factory`](@ref) of the risk path fills them into the return estimator of the measure, so the resolved measure reads the weights, the returns matrix and the fees. [`ExpectedReturn`](@ref) declares [`WeightsReturnsFeesInput`](@ref), and [`ExpectedReturnRiskRatio`](@ref) is a composite with methods of its own, as [`MeanReturnRiskRatio`](@ref) is. Thus [`risk_contribution`](@ref), [`risk_gradient`](@ref) and every composite of the risk path take both members. Each method that dispatches on the alias follows from the need for a prior:

  - The methods for a prediction result refuse the call.
  - [`supports_precomputed_returns`](@ref) gives `false`.

# Related

  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`PerfRM`](@ref): The larger group that both members also belong to.
  - [`prrm_prediction_message`](@ref): It builds the message of the refusal of the prediction-result methods.
  - [`resolved_expected_return`](@ref): The return of a resolved estimator on the returns matrix.
"""
const PrRM = Union{ExpectedReturn, ExpectedReturnRiskRatio}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`, because a [`PrRM`](@ref) does not support precomputed returns.

Both members read the portfolio weights and the prior through their return estimator: an [`ArithmeticReturn`](@ref) multiplies the expected returns of the prior by the weights, and a [`LogarithmicReturn`](@ref) weights each observation by the observation weights of the prior and charges the fee of the weights once, after the mean. A bare series of net returns holds neither the prior nor the weights. So neither type defines `r(x::VecNum)`, and [`expected_risk_from_returns`](@ref) refuses both with a message that names the cause.

To score a bare return series, use [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref). [`prrm_prediction_message`](@ref) names the same replacement for a prediction result.

# Related

  - [`PrRM`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`MeanReturn`](@ref): The [`NetReturnsInput`](@ref) measure that scores a bare series.
  - [`expected_risk_from_returns`](@ref): The entry of the contract that this predicate controls.
"""
supports_precomputed_returns(::PrRM)::Bool = false
function bigger_is_better(::PerfRM)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the message of the error that a [`PrRM`](@ref) raises on a prediction result.

The refusal is about **prior returns and realised returns**, and the number of return terms does not change it. A [`PrRM`](@ref) reads a prior result, and a prediction result holds a realised series.

The replacement that the message names does change with the number of terms. [`MeanReturn`](@ref) holds **no** return estimator, so it cannot hold the many terms of a vector `rt`. When `rt` is a vector, the message says so. It does not tell the caller that the replacement gives the same figure, because the replacement loses the terms.

# Algorithm

 1. Choose the replacement measure `alt` from the type of `r`. It is [`MeanReturn`](@ref) for an [`ExpectedReturn`](@ref), and [`MeanReturnRiskRatio`](@ref) for an [`ExpectedReturnRiskRatio`](@ref).
 2. Write the first sentence of `msg`. It names `r`, the wrapper type of `pred` and `alt`.
 3. When the `rt` of `r` is a vector, add the second sentence. It names the number of terms in `rt` and the two routes that keep them.

# Arguments

  - `r`: The measure that refuses the call.
  - `pred`: The prediction result that the caller gave.

# Returns

  - `msg::String`: The message of the `ArgumentError` that the four methods for a prediction result raise.

# Related

  - [`PrRM`](@ref)
  - [`MeanReturn`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`expected_risk`](@ref): The four methods that raise this message.
"""
function prrm_prediction_message(r::PrRM, pred)
    alt = isa(r, ExpectedReturn) ? MeanReturn : MeanReturnRiskRatio
    msg = "risk measure $r is incompatible for computing the `expected_risk` of a $(Base.typename(typeof(pred)).wrapper), please use $(alt) instead"
    if isa(r.rt, VecJRE)
        msg *= "\n`rt` holds $(length(r.rt)) return terms, and `$(alt)` carries no return estimator at all, so the swap would collapse them to one. Scalarise the terms yourself, or evaluate this measure against a prior result rather than a prediction result."
    end
    return msg
end
"""
    expected_risk(::Scalariser, r::PrRM,
                  pred::PredictionResult{<:Any,
                                         <:PredictionReturnsResult{<:Any, <:VecNum}};
                  kwargs...)
    expected_risk(::Scalariser, r::PrRM,
                  pred::PredictionResult{<:Any,
                                         <:PredictionReturnsResult{<:Any, <:VecVecNum}};
                  kwargs...)
    expected_risk(::Scalariser, r::PrRM, pred::MultiPeriodPredictionResult; kwargs...)
    expected_risk(::Scalariser, r::PrRM, pred::PopulationPredictionResult; kwargs...)

Refuse to compute a risk measure that reads a prior on a prediction result.

A [`PrRM`](@ref) reads a **prior** result. It multiplies the expected returns of the prior by the weights. A prediction result holds a **realised** return series and no expected returns, so the two measures have nothing to read. Each of the four methods raises an error, and none gives a figure from the wrong input.

The four methods are positional forms, which take the scalariser first and ignore it. The keyword form `expected_risk(r, pred; kwargs...)` of a prediction calls them, so both forms raise the error.

The caller has two routes:

  - To measure the realised series, use [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) as `alt` in place of the measure, and call `expected_risk(alt, pred)`.
  - To measure the prior, call `expected_risk(r, w, pr)` with the weights and the prior result. The `expected_risk` methods for an [`ExpectedReturn`](@ref) and for an [`ExpectedReturnRiskRatio`](@ref) document that route.

# Arguments

  - `::Scalariser`: The scalariser of the call, which none of the four methods reads.
  - `r`: [`ExpectedReturn`](@ref) or [`ExpectedReturnRiskRatio`](@ref).
  - `pred`: The prediction result that the caller gave.
  - `kwargs...`: Keyword arguments that none of the four methods reads.

# Validation

  - Every method raises an `ArgumentError`, with the message that [`prrm_prediction_message`](@ref) builds. No method returns for any input.

# Related

  - [`PrRM`](@ref)
  - [`prrm_prediction_message`](@ref): It builds the message of the error.
  - [`MeanReturn`](@ref): The replacement for an [`ExpectedReturn`](@ref).
  - [`MeanReturnRiskRatio`](@ref): The replacement for an [`ExpectedReturnRiskRatio`](@ref).
  - [`PredictionResult`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
  - [`PopulationPredictionResult`](@ref)
"""
function expected_risk(::Scalariser, r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecNum}};
                       kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(::Scalariser, r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecVecNum}};
                       kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(::Scalariser, r::PrRM, pred::MultiPeriodPredictionResult; kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(::Scalariser, r::PrRM, pred::PopulationPredictionResult; kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, ExpectedReturn, ExpectedReturnRiskRatio
