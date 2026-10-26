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
  - ``v_{t}``: Observation weight of observation ``t``, the ``t``-th entry of the `w` field of a [`LogarithmicReturn`](@ref). Every weight is one when that field is `nothing`, and the quotient is then the unweighted mean.
  - ``F(\\boldsymbol{w})``: Fee term of [`term_fees`](@ref). It is [`calc_fees`](@ref) of `w` and `fees` when the term's `settings.fee` is `true`, and zero otherwise.

# Arguments

  - `ret`: Return estimator, or a vector of them on the fourth method.
  - `w`: Portfolio weights, or a vector of portfolio weight vectors on the fourth method.
  - `pr`: Prior result. It resolves a Deferred Quantity of `ret`. It gives `mu` to an [`ArithmeticReturn`](@ref) whose own `mu` is `nothing`.
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
    # The scalar twin of the `ret` expression `set_return_constraints!` builds, so it uses
    # the same ladder. The prior is in hand, so a Deferred Quantity resolves here too.
    r = resolve_deferred_quantities(r, pr)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return LinearAlgebra.dot(w, mu) - term_fees(w, fees, size(pr.X, 1), r.settings.fee)
end
function expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # Reduce to the Investable Mask — see the note on the arithmetic twin above.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    rw = ret.w
    X = pr.X
    kret = if isnothing(rw)
        Statistics.mean(log1p, X * w)
    else
        Statistics.mean(log1p.(X * w), rw)
    end
    return kret - term_fees(w, fees, size(X, 1), ret.settings.fee)
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
 2. Set the accumulator `rt` to a zero of the element type of `w`.
 3. Take the next term `ret_i` of `ret`. Skip it when its `settings.rte` is `false`.
 4. Compute the value-level return of `ret_i` with [`expected_return`](@ref), multiply it by the `settings.scale` of the term, and add it to `rt`.
 5. Repeat steps 3 and 4 for the other terms, then return `rt`.

# Arguments

  - `ret`: Vector of return estimators.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional fees.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that go to [`expected_return`](@ref) of each term.

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
    # term. Every term reduces again on the reduced prior, where the mask is `nothing`.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    rt = zero(eltype(w))
    for ret_i in ret
        if !ret_i.settings.rte
            continue
        end
        rt += ret_i.settings.scale * expected_return(ret_i, w, pr, fees; kwargs...)
    end
    return rt
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
    rk = expected_risk(r, w, pr, fees; sca = sca, kwargs...)
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
    rk = expected_risk(r, w, pr, fees; sca = sca, kwargs...)
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
"""
    expected_risk(r::ExpectedReturn, w::VecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; kwargs...)

Compute the value of an [`ExpectedReturn`](@ref), the expected return of a portfolio.

The method calls [`expected_return`](@ref) on the `rt` field of the measure. [`ExpectedReturn`](@ref) states the closed form.

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
function needs_previous_weights(r::ExpectedReturnRiskRatio)
    return needs_previous_weights(r.rk)
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

Both members read the **prior result** itself and not a returns matrix, so neither declares a [`risk_input_kind`](@ref). Each method that dispatches on the alias follows from that property:

  - The method for a vector of weight vectors resolves the measure once and keeps the prior.
  - The methods for a prediction result refuse the call.
  - [`supports_precomputed_returns`](@ref) gives `false`.

# Related

  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`PerfRM`](@ref): The larger group that both members also belong to.
  - [`prrm_prediction_message`](@ref): It builds the message of the refusal of the prediction-result methods.
  - [`risk_input_kind`](@ref): The declaration that neither member makes.
"""
const PrRM = Union{ExpectedReturn, ExpectedReturnRiskRatio}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`, because a [`PrRM`](@ref) does not support precomputed returns.

Both members read the **prior result**, and they multiply the expected returns of the prior by the portfolio weights. A bare series of net returns holds neither the expected returns nor the weights. So neither type defines `r(x::VecNum)`. This method gives the answer before the call gets to the default of [`risk_input_kind`](@ref). That default raises an error, and to a caller the error looks like an internal defect, not like a fact about the chosen measure.

To score a bare return series, use [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref). [`prrm_prediction_message`](@ref) names the same replacement for a prediction result.

# Related

  - [`PrRM`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`risk_input_kind`](@ref): The declaration that neither member makes.
  - [`MeanReturn`](@ref): The [`NetReturnsInput`](@ref) measure that scores a bare series.
  - [`expected_risk_from_returns`](@ref): The entry of the contract that this predicate controls.
"""
supports_precomputed_returns(::PrRM)::Bool = false
"""
    expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)

Compute a risk measure that reads a prior on each weight vector of `w`, for example on the
points of a [`Frontier`](@ref).

A [`PrRM`](@ref) reads the prior result itself and not a returns matrix. For that reason it
declares no [`risk_input_kind`](@ref). The generic `VecVecNum` method resolves the measure and
then gives `pr.X` to the loop. That is correct for a measure whose functor takes the matrix,
but the matrix does not hold the moments of the prior, such as `mu`, that this family reads. So this method
resolves the measure once, and it keeps the prior for each weight vector.

# Algorithm

 1. Reduce the prior, the weights and the fees to the Investable Mask once, with [`investable_reduction`](@ref). A held asset outside the mask gives one diagnostic, and not one for each weight vector.
 2. Resolve the measure against the prior once, with [`factory`](@ref). The slots of the resolved measure are filled.
 3. Compute the resolved measure on each weight vector `wi` of `w`. The input is the prior itself, and not `pr.X`.
 4. Collect one expected risk for each weight vector, in the order of `w`.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) or [`ExpectedReturnRiskRatio`](@ref).
  - `w`: Vector of portfolio weight vectors.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - $(arg_dict[:strict_inv])
  - `kwargs...`: Keyword arguments that go to the method for one weight vector.

# Returns

  - `risk::VecNum`: One expected risk for each weight vector.

# Related

  - [`PrRM`](@ref)
  - [`expected_risk`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`factory`](@ref)
"""
function expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; strict::Bool = false, kwargs...)
    # The population reduces once, so the diagnostic is raised once rather than once per
    # member. Every member reduces again on the reduced prior, where the mask is `nothing`.
    _, pr, w, fees = investable_reduction(pr, w, fees, strict)
    r = factory(r, pr)
    return [expected_risk(r, wi, pr, fees; kwargs...) for wi in w]
end
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
    expected_risk(r::PrRM,
                  pred::PredictionResult{<:Any,
                                         <:PredictionReturnsResult{<:Any, <:VecNum}};
                  kwargs...)
    expected_risk(r::PrRM,
                  pred::PredictionResult{<:Any,
                                         <:PredictionReturnsResult{<:Any, <:VecVecNum}};
                  kwargs...)
    expected_risk(r::PrRM, pred::MultiPeriodPredictionResult; kwargs...)
    expected_risk(r::PrRM, pred::PopulationPredictionResult; kwargs...)

Refuse to compute a risk measure that reads a prior on a prediction result.

A [`PrRM`](@ref) reads a **prior** result. It multiplies the expected returns of the prior by the weights. A prediction result holds a **realised** return series and no expected returns, so the two measures have nothing to read. Each of the four methods raises an error, and none gives a figure from the wrong input.

The caller has two routes:

  - To measure the realised series, use [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) as `alt` in place of the measure, and call `expected_risk(alt, pred)`.
  - To measure the prior, call `expected_risk(r, w, pr)` with the weights and the prior result. The `expected_risk` methods for an [`ExpectedReturn`](@ref) and for an [`ExpectedReturnRiskRatio`](@ref) document that route.

# Arguments

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
function expected_risk(r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecNum}};
                       kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(r::PrRM,
                       pred::PredictionResult{<:Any,
                                              <:PredictionReturnsResult{<:Any, <:VecVecNum}};
                       kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(r::PrRM, pred::MultiPeriodPredictionResult; kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
function expected_risk(r::PrRM, pred::PopulationPredictionResult; kwargs...)
    return throw(ArgumentError(prrm_prediction_message(r, pred)))
end
"""
    brinson_attribution(X::TimeArray, w::VecNum, wb::VecNum,
                        asset_classes::DataFrame, col, date0 = nothing, date1 = nothing)

Compute the Brinson performance attribution of each asset class [brinson_attribution](@cite).

`brinson_attribution` returns a DataFrame with the Brinson attribution of each asset class. It splits the total excess return of the class into asset allocation, security selection and interaction. The function reads the asset weights, the benchmark weights and the class of each asset over the date range.

`date0` and `date1` are **positional** arguments after `col`. The filter applies only when the call gives both. A call that gives only one of them attributes the whole of `X`.

`X` holds **prices**, not returns. The period return of an asset is the ratio of its last value in the range to its first value, minus one. This ratio is a return only when the entries are prices.

The function attributes **one** evaluation period, between the two dates, and it reads no weight path. `w` and `wb` stay the same over that period. So the function attributes a fund whose holdings drifted in the period at its weights at the start. A multi-period attribution that follows the drift needs a rule that links the effects of the single periods into a cumulative effect. This library has no such rule, so it has no such method. Attribute each period alone, and link the results outside the library.

Both class returns divide by the weight of the class, and no guard stops a division by zero. A class with zero portfolio weight makes its ``r_{i}`` a `NaN`. A class with zero benchmark weight makes its ``r_{i}^{b}`` a `NaN`. The two cases spread differently, because ``\\mathrm{AA}_{i}`` reads only ``r_{i}^{b}``. A zero portfolio weight keeps ``\\mathrm{AA}_{i}`` finite, and it makes the other three rows of that class a `NaN`. A zero benchmark weight makes all four rows a `NaN`.

The `Total` column is a row sum, so its entry of a row is a `NaN` when one of the class entries of the row is a `NaN`. The columns of the other classes stay finite, because each column reads only its own class and ``r^{b}``. Do not remove the rows of the empty class from `asset_classes` alone. Then `asset_classes` and `w` have different lengths, and the function raises a `DimensionMismatch`.

# Mathematical definition

```math
\\begin{align}
r_{j} &= \\dfrac{P_{j,\\,1}}{P_{j,\\,0}} - 1\\\\
r^{b} &= \\sum_{j} w_{j}^{b}\\, r_{j}\\\\
r_{i} &= \\dfrac{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}\\, r_{j}}{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}}\\\\
r_{i}^{b} &= \\dfrac{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}\\, r_{j}}{\\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}}\\\\
\\mathrm{AA}_{i} &= \\left(w_{i} - w_{i}^{b}\\right) \\left(r_{i}^{b} - r^{b}\\right)\\\\
\\mathrm{SS}_{i} &= w_{i}^{b} \\left(r_{i} - r_{i}^{b}\\right)\\\\
\\mathrm{I}_{i} &= \\left(w_{i} - w_{i}^{b}\\right) \\left(r_{i} - r_{i}^{b}\\right)\\\\
\\mathrm{TER}_{i} &= \\mathrm{AA}_{i} + \\mathrm{SS}_{i} + \\mathrm{I}_{i}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{C}_{i}``: The ``i``-th asset class, the set of assets whose `col` entry is that class.
  - ``P_{j,\\,0}``, ``P_{j,\\,1}``: Price of asset ``j`` at the first and at the last observation of the range.
  - ``r_{j}``: Return of asset ``j`` over the range.
  - ``w_{j}``, ``w_{j}^{b}``: Portfolio and benchmark weight of asset ``j``.
  - ``w_{i} = \\sum_{j \\in \\mathcal{C}_{i}} w_{j}``, ``w_{i}^{b} = \\sum_{j \\in \\mathcal{C}_{i}} w_{j}^{b}``: Portfolio and benchmark weight of the class.
  - ``r_{i}``, ``r_{i}^{b}``: Portfolio and benchmark return of the class. Each is the weighted sum of the returns of its members, divided by the weight of the class. So a class return is a weighted average and not a weighted sum.
  - ``r^{b}``: Benchmark return of the whole universe.
  - ``\\mathrm{AA}_{i}``: Asset allocation of the class, the part of the excess return that comes from the weight of the class.
  - ``\\mathrm{SS}_{i}``: Security selection of the class, the part that comes from the choice of assets in the class.
  - ``\\mathrm{I}_{i}``: Interaction of the class, the product of the two differences.
  - ``\\mathrm{TER}_{i}``: Total excess return of the class, the sum of the three.

# Algorithm

 1. Filter the date range. When the call gives `date0` and `date1`, find the first and the last observation of `X` in the range. These are `idx1` and `idx2`. Otherwise take the whole of `X`, so `idx1` is `1` and `idx2` is `length(X)`.
 2. Make `ret`, the period return of each asset, from the values at `idx1` and `idx2`.
 3. Take the dot product of `ret` and `wb`. This is `ret_b`, the benchmark return of the whole universe.
 4. Read the class of each asset from column `col` of `asset_classes`. Build `sets_mat`, the membership matrix. Its entry is `true` when the asset belongs to the class.
 5. Take the next class. Sum the two weight vectors over its members, to get `w_i` and `wb_i`. Divide the two weighted sums of the returns of its members by those weights, to get `ret_i` and `ret_b_i`.
 6. Compute `AA_i`, `SS_i`, `I_i` and `TER_i` of that class. Write the four numbers as the column of the class in `df`.
 7. Repeat steps 5 and 6 for the other classes.
 8. Add the `Total` column, the sum of the class columns for each row.

# Arguments

  - `X`: TimeArray of asset prices.
  - `w`: Vector of portfolio weights.
  - `wb`: Vector of benchmark weights.
  - `asset_classes`: DataFrame that gives the class of each asset.
  - `col`: Column of `asset_classes` that holds the class of each asset.
  - `date0`: Optional start date of the attribution period.
  - `date1`: Optional end date of the attribution period.

# Returns

  - `df::DataFrame`: DataFrame with one row each for asset allocation, security selection, interaction and the total effect. It has one column for each asset class, and a `Total` column.

# Related

  - [`VecNum`](@ref)
  - [`performance_summary`](@ref): It summarises one realised return series. This function splits one period across the classes.
  - [`calc_net_returns`](@ref): It gives the net return series of a portfolio, and it charges fees. This function charges no fees.
  - [`AssetSetsMatrixEstimator`](@ref): It builds the group membership matrix of the library. Step 4 builds its own matrix from a DataFrame column.

# References

  - $(ref_dict[:brinson_attribution]) Pages 73-76. $(ref_dict[:no_original_source])
"""
function brinson_attribution(X::TimeSeries.TimeArray, w::VecNum, wb::VecNum,
                             asset_classes::DataFrames.DataFrame, col, date0 = nothing,
                             date1 = nothing)
    # Efficient filtering of date range
    idx1, idx2 = if !isnothing(date0) && !isnothing(date1)
        timestamps = TimeSeries.timestamp(X)
        idx = (Dates.DateTime(date0) .<= timestamps) .&
              (timestamps .<= Dates.DateTime(date1))
        findfirst(idx), findlast(idx)
    else
        1, length(X)
    end

    ret = vec(values(X[idx2]) ./ values(X[idx1]) .- 1)
    ret_b = LinearAlgebra.dot(ret, wb)

    classes = asset_classes[!, col]
    unique_classes = unique(classes)

    df = DataFrames.DataFrame(;
                              index = ["Asset Allocation", "Security Selection",
                                       "Interaction", "Total Excess Return"])

    # Precompute class membership matrix for efficiency
    sets_mat = [class_j == class_i for class_j in classes, class_i in unique_classes]

    for (i, class_i) in enumerate(unique_classes)
        sets_i = view(sets_mat, :, i)

        w_i = LinearAlgebra.dot(sets_i, w)
        wb_i = LinearAlgebra.dot(sets_i, wb)

        ret_i = LinearAlgebra.dot(ret .* sets_i, w) / w_i
        ret_b_i = LinearAlgebra.dot(ret .* sets_i, wb) / wb_i

        w_diff_i = w_i - wb_i
        ret_diff_i = ret_i - ret_b_i

        AA_i = w_diff_i * (ret_b_i - ret_b)
        SS_i = wb_i * ret_diff_i
        I_i = w_diff_i * ret_diff_i
        TER_i = AA_i + SS_i + I_i

        df[!, class_i] = [AA_i, SS_i, I_i, TER_i]
    end

    df[!, "Total"] = sum(eachcol(df[!, 2:end]))

    return df
end

"""
$(DocStringExtensions.TYPEDEF)

The headline performance statistics of a realised return series.

[`performance_summary`](@ref) returns a `PerformanceSummaryResult`. It holds the eleven numbers that [`plot_performance_summary`](@ref) draws, the standard error of the Sharpe ratio, and the four inputs of the statistics. A caller can put a summary in a table, compare it across runs, test it, or give it to a search, with no plotting package installed.

The last four numbers read inputs that a bare return series does not hold. Each is `NaN` when its input is absent. The three excess statistics need a `benchmark` series. A prediction result reads that series from its own rows when the returns data held a `B`. The turnover needs the weight path of a multi-period prediction result.

# Sign convention

`max_drawdown` and `cvar` are in **return space**, so both are not positive for a portfolio that has a loss. The [`ConditionalValueatRisk`](@ref) risk measure has the opposite sign, because it gives a loss as a positive number. Thus `ConditionalValueatRisk(; alpha = alpha)(ret) == -summary.cvar`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PerformanceSummaryResult(
        n_periods, ppy, alpha, compound,
        ann_return, ann_volatility, sharpe, sharpe_stderr,
        sortino, calmar, max_drawdown, cvar,
        excess_ret, tracking_error, information_ratio, turnover
    ) -> PerformanceSummaryResult

The arguments are the fields of the struct, in the order of the declaration. The type is a
Result, so [`performance_summary`](@ref) builds it and a caller reads it. The type has no
keyword constructor, and it does no validation of its own. The first four fields are the
inputs of the statistics in the fields after them.

# Related

  - [`performance_summary`](@ref)
  - [`plot_performance_summary`](@ref)
  - [`ConditionalValueatRisk`](@ref)
"""
@concrete struct PerformanceSummaryResult <: AbstractResult
    """
    $(field_dict[:ps_n_periods])
    """
    n_periods
    """
    $(field_dict[:ps_ppy])
    """
    ppy
    """
    $(field_dict[:ps_alpha])
    """
    alpha
    """
    $(field_dict[:ps_compound])
    """
    compound
    """
    $(field_dict[:ps_ann_return])
    """
    ann_return
    """
    $(field_dict[:ps_ann_volatility])
    """
    ann_volatility
    """
    $(field_dict[:ps_sharpe])
    """
    sharpe
    """
    $(field_dict[:ps_sharpe_stderr])
    """
    sharpe_stderr
    """
    $(field_dict[:ps_sortino])
    """
    sortino
    """
    $(field_dict[:ps_calmar])
    """
    calmar
    """
    $(field_dict[:ps_max_drawdown])
    """
    max_drawdown
    """
    $(field_dict[:ps_cvar])
    """
    cvar
    """
    `excess_ret`: Annualised arithmetic mean of `ret - benchmark`. It is `NaN` when no benchmark is given.
    """
    excess_ret
    """
    `tracking_error`: Annualised sample standard deviation of `ret - benchmark`. It is `NaN` when no benchmark is given.
    """
    tracking_error
    """
    `information_ratio`: `excess_ret / tracking_error`. It is `NaN` when no benchmark is given, or when the tracking error is zero or `NaN`.
    """
    information_ratio
    """
    `turnover`: Mean turnover for each rebalance of the held path of a multi-period prediction result. One rebalance trades `sum(abs, w_next - w_held)`, from the weights that a fold held after its last observation to the target of the next fold. It is `NaN` for a bare series, for a single fold and for a multi-period result of one fold, because none of them records a rebalance.
    """
    turnover
end
"""
    performance_summary(ret::VecNum; ppy::Number = 1, alpha::Number = 0.05,
                        compound::Bool = false,
                        benchmark::Option{<:VecNum} = nothing) -> PerformanceSummaryResult
    performance_summary(w::ArrNum, X::MatNum, fees::Option{<:Fees} = nothing;
                        ppy, alpha, compound, benchmark) -> PerformanceSummaryResult
    performance_summary(w::ArrNum, rd::ReturnsResult, fees::Option{<:Fees} = nothing;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(res::OptimisationResult, rd::ReturnsResult;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(pred::PredRes_MultiPredRes;
                        benchmark::Option{<:VecNum} = regret_benchmark(pred),
                        kwargs...) -> PerformanceSummaryResult

Summarise a realised return series as a [`PerformanceSummaryResult`](@ref).

The methods that take weights and returns first make the net returns with [`calc_net_returns`](@ref). So the summary of a portfolio includes its fees.

Every method takes a `benchmark`, a second return series over the same periods, and gives the three excess statistics against it. When no benchmark is given, the three are `NaN`. On the prediction-result method, `benchmark` defaults to the benchmark of the rows that the prediction scored, which [`regret_benchmark`](@ref) reads. So a prediction over returns data with a `B` gives the three statistics with no keyword, and `benchmark = nothing` turns them off. The other methods hold no prediction, and they read no `B`.

Each fold collapsed a benchmark matrix with its own weights, as [`collapse_benchmark`](@ref) states. Against such a benchmark, `excess_ret` is the return of ``(\\boldsymbol{X} - \\boldsymbol{B}) \\boldsymbol{w}``. That is the active return against the same weights in the benchmark assets, and not the excess over one index.

The prediction-result method also gives the **turnover** of the held path, the mean trade at a rebalance of a multi-period result. At `test_size = 1`, a rebalance is a period. A bare series, a single fold and a result of one fold record no rebalance, so their turnover is `NaN`.

**The series `ret` must be finite, as the Precomputed-returns contract states.** No method checks that the series is finite, because every internal caller gives a finite series. The prediction methods read the output of [`calc_net_returns`](@ref) for the fold, and a check of a long series costs time on every call. One entry that is not finite makes the mean, the volatility and every ratio not finite. The tail figure then gives a **finite wrong number** and not a `NaN`, because `partialsort` puts a `NaN` after every real number. A caller with a gapped series must first remove the gaps with `x[isfinite.(x)]`. A caller with a gapped panel must score it through [`predict(res::NonFiniteAllocationOptimisationResult, rd::ReturnsResult)`](@ref), which filters the Held Gaps once.

The standard error of the Sharpe ratio reads every observation as independent. So on a series scored under a Weight Drift, it is less than the true standard error, as the `# Mathematical definition` states. A long-run variance estimator corrects for serial dependence, but it needs a bandwidth, and no bandwidth is correct for every sample. The library does not build one. It applies no guard and no threshold, and it gives the figure with no change.

# Mathematical definition

The five headline statistics:

```math
\\begin{align}
\\mathrm{ann\\_return} &= m p\\\\
\\mathrm{ann\\_volatility} &= s \\sqrt{p}\\\\
\\mathrm{sharpe} &= \\dfrac{\\mathrm{ann\\_return}}{\\mathrm{ann\\_volatility}}\\\\
\\mathrm{sortino} &= \\dfrac{\\mathrm{ann\\_return}}{\\sqrt{\\dfrac{p}{T} \\sum\\limits_{t=1}^{T} \\min(x_{t}, 0)^{2}}}\\\\
\\mathrm{calmar} &= \\dfrac{\\mathrm{ann\\_return}}{\\left\\lvert \\min_{t} d_{t} \\right\\rvert}\\,.
\\end{align}
```

The three excess statistics against a benchmark, and the turnover of the held path of a multi-period result:

```math
\\begin{align}
\\mathrm{excess\\_ret} &= m_e p\\\\
\\mathrm{tracking\\_error} &= s_e \\sqrt{p}\\\\
\\mathrm{information\\_ratio} &= \\dfrac{\\mathrm{excess\\_ret}}{\\mathrm{tracking\\_error}}\\\\
\\mathrm{turnover} &= \\dfrac{1}{F - 1} \\sum_{f=2}^{F} \\left\\lVert \\boldsymbol{w}_f - \\hat{\\boldsymbol{w}}_{f-1} \\right\\rVert_1\\,.
\\end{align}
```

The standard error of the Sharpe ratio corrects for the skewness and the excess kurtosis of the returns. Its variance is the asymptotic variance of Equation 2 of [mertens2002](@cite), divided by ``T - 1`` as in Section 2.5 of [sharpe_stderr](@cite), and annualised:

```math
\\begin{align}
\\mathrm{sharpe\\_stderr} &= \\sqrt{\\dfrac{p}{T - 1} \\left(1 - g_{1} \\dfrac{m}{s} + \\dfrac{g_{2} + 2}{4} \\dfrac{m^{2}}{s^{2}}\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xret])
  - $(math_dict[:b_norm_err])
  - $(math_dict[:T])
  - ``p``: Number of periods per year, `ppy`.
  - ``m``, ``s``: Sample mean and corrected sample standard deviation of ``\\boldsymbol{x}``.
  - ``\\boldsymbol{d}``: Drawdown path of the cumulative wealth of ``\\boldsymbol{x}``, compounded or not as `compound` states, with ``d_{t} \\leq 0``.
  - ``m_e``, ``s_e``: Sample mean and corrected sample standard deviation of the excess series ``\\boldsymbol{e} = \\boldsymbol{x} - \\boldsymbol{b}``.
  - ``F``: Number of folds of the multi-period result.
  - ``\\boldsymbol{w}_f``: Weights at the start of fold ``f``, its target on a solved fold.
  - ``\\hat{\\boldsymbol{w}}_f``: Weights of fold ``f`` after its last observation. They are the drifted holding when the fold has a Held Weights record, and ``\\boldsymbol{w}_f`` otherwise.
  - ``g_{1}``, ``g_{2}``: Sample skewness and sample excess kurtosis of ``\\boldsymbol{x}``.

The turnover puts every weight vector on the union of the universes of the folds, by asset name. A weight is zero at an asset that its fold does not hold. Each term of the sum is the trade at one rebalance. [`calc_turnover`](@ref) reads that quantity at one step, and [`Turnover`](@ref) bounds it. The drift in a fold is not a trade, so it enters the turnover only through ``\\hat{\\boldsymbol{w}}_{f-1}``.

The downside deviation of the Sortino ratio divides by ``T``, and not by the number of periods with a loss. This is the convention of Sortino and Price. With it, the ratios of two series with different frequencies of loss stay comparable.

The kurtosis term is ``(g_{2} + 2)/4``. Section 2.5 of [sharpe_stderr](@cite) states it as ``(\\gamma_{4} - 1)/4`` on the **raw** fourth standardised moment ``\\gamma_{4}``, and `StatsBase.kurtosis` gives the **excess** moment ``g_{2} = \\gamma_{4} - 3``. The two forms are equal. Only this form becomes the naive ``\\sqrt{(1 + \\mathrm{SR}^{2}/2)/T}`` on a normal series. Over 200,000 samples of 250 normal returns at a Sharpe ratio per period of 0.5, the sample standard deviation of the Sharpe ratio is **0.06728323**. This expression at the population moments gives **0.06721661**, and the same expression without the ``+2`` gives **0.06337243**.

A return series that is not normal makes a Sharpe ratio less precise than the naive expression shows, and a negative skew makes it less precise again. A real portfolio has such a series, so the summary gives the standard error with the ratio.

The expression corrects for the third and fourth moments and **not** for serial dependence, so it reads every observation as independent of the others. A series scored under a Weight Drift is serially dependent through the weights that it held, because a position that grew has a larger weight in the next observation. So on such a series, this figure is less than the true standard error.

# Algorithm

Every method calls [`summarise_returns`](@ref), which does the steps. A method that takes weights first makes the net returns with [`calc_net_returns`](@ref). The method that takes a prediction result reads the first series when it holds many series. It sets the default benchmark to the first series of its own `B`, and it reads the turnover of its held path.

 1. Check `alpha` and `ppy`, as `# Validation` states.
 2. Take `T`, the number of periods. Take `m`, the sample mean of `ret`, and clamp it to the least and the greatest entry of `ret`. Take `s`, the corrected sample standard deviation of `ret` about `m`. The clamp changes nothing in exact arithmetic. The rounded mean of a constant `ret` can differ from its common value, and the clamp makes `m` equal to that value, so `s` is exactly zero.
 3. Annualise `m` and `s`, to get `ann_ret` and `ann_vol`.
 4. Divide `ann_ret` by `ann_vol`, to get `sharpe`. When `ann_vol` is not positive, `sharpe` is `NaN`.
 5. Set every positive entry of `ret` to zero. Square the entries, average them over all `T` periods, annualise, and take the square root, to get the downside deviation `ddev`. Divide `ann_ret` by `ddev`, to get `sortino`. When `ddev` is not positive, `sortino` is `NaN`.
 6. Build the cumulative wealth series with [`cumulative_returns`](@ref), and take its drawdown path with [`drawdowns`](@ref). Read `max_dd`, the minimum of that path. Divide `ann_ret` by the absolute value of `max_dd`, to get `calmar`. When `max_dd` is not negative, `calmar` is `NaN`.
 7. Compute [`ConditionalValueatRisk`](@ref) at `alpha` on `ret`, and change its sign, to get `cvar_val` in return space.
 8. Make the Sharpe ratio per period `sr_p`. Correct its variance `var_sr` for the sample skewness and the sample excess kurtosis of `ret`. Annualise the square root, to get `sharpe_se`. When `var_sr` is not positive, `sharpe_se` is `NaN`.
 9. Compute the three excess statistics with [`excess_statistics`](@ref). With a `benchmark`, the function checks its length and takes `e = ret - benchmark`. It annualises the mean of `e`, clamped as in step 2, and the standard deviation of `e` about that mean, to get `excess_ret` and `tracking_error`. It divides them, to get `information_ratio`, which is `NaN` when `tracking_error` is not positive. With no benchmark, all three are `NaN`.
10. Take the turnover that the prediction-result method computed with [`held_path_turnover`](@ref). A `nothing` from that function, and every other method, gives a `NaN`.
11. Put the four inputs and the twelve statistics into a [`PerformanceSummaryResult`](@ref).

# Arguments

  - `ret`: Vector of periodic portfolio returns.
  - `w`: Portfolio weights.
  - $(arg_dict[:X])
  - `fees`: Optional transaction fees.
  - `rd`: [`ReturnsResult`](@ref) that holds the asset returns.
  - `res`: An [`OptimisationResult`](@ref). [`result_investable_view`](@ref) puts its weights and fees, and the `rd` of the caller, on the investable universe of the result.
  - $(arg_dict[:ps_ppy]) It defaults to `1`, which gives the statistics per period at any frequency of the data.
  - $(arg_dict[:ps_alpha])
  - $(arg_dict[:ps_compound])
  - `benchmark`: Benchmark return series over the same periods as `ret`, or `nothing` for no excess statistics. It defaults to `nothing`, and on a prediction result to the benchmark of its rows.

# Validation

  - `0 < alpha < 1`.
  - `ppy > 0`.
  - `length(benchmark) == length(ret)` when a benchmark is given. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - `summary::PerformanceSummaryResult`: The computed summary.

# Related

  - [`PerformanceSummaryResult`](@ref)
  - [`plot_performance_summary`](@ref)
  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
  - [`drawdowns`](@ref)
  - [`calc_turnover`](@ref)
  - [`held_path_turnover`](@ref)
  - [`log_wealth_regret`](@ref)

# References

  - $(ref_dict[:mertens2002]) Section 1, Equation 2.
  - $(ref_dict[:sharpe_stderr]) Section 2.5.
"""
function performance_summary(ret::VecNum; kwargs...)::PerformanceSummaryResult
    return summarise_returns(ret, nothing; kwargs...)
end
"""
    excess_statistics(ret::VecNum, benchmark::Nothing, ann::Number, m::Number)
    excess_statistics(ret::VecNum, benchmark::VecNum, ann::Number, m::Number)

Compute the three excess statistics of [`performance_summary`](@ref) against a benchmark series, or give three `NaN` values when no benchmark is given.

The closed forms are the `excess_ret`, `tracking_error` and `information_ratio` rows of the `# Mathematical definition` of [`performance_summary`](@ref).

# Arguments

  - `ret`: Periodic portfolio returns.
  - `benchmark`: Benchmark returns over the same periods, or `nothing`.
  - `ann`: Annualisation factor.
  - `m`: Sample mean of `ret`. With no benchmark, each `NaN` has the number type that its statistic has with a benchmark. The function derives that type from `m` and `ann`.

# Validation

  - `length(benchmark) == length(ret)`. Otherwise the function raises a `DimensionMismatch`.

# Returns

  - `(excess_ret, tracking_error, information_ratio)`: Annualised mean and standard deviation of `ret - benchmark`, and their ratio. The ratio is `NaN` when the tracking error is not positive, that is when the tracking error is zero or `NaN`.

# Related

  - [`performance_summary`](@ref)
  - [`PerformanceSummaryResult`](@ref)
"""
function excess_statistics(::VecNum, ::Nothing, ann::Number, m::Number)
    # Each `NaN` takes the number type the statistic takes with a benchmark.
    excess_ret = oftype(m * ann, NaN)
    tracking_error = oftype(m * sqrt(ann), NaN)
    return excess_ret, tracking_error, oftype(one(excess_ret) / one(tracking_error), NaN)
end
function excess_statistics(ret::VecNum, benchmark::VecNum, ann::Number, ::Number)
    @argcheck(length(benchmark) == length(ret),
              DimensionMismatch("`length(benchmark) == length(ret)` must hold.\nlength(benchmark) => $(length(benchmark))\nlength(ret) => $(length(ret))"))
    e = ret .- benchmark
    # The clamp moves nothing in exact arithmetic. On a constant excess series it makes the
    # mean the common value, so the tracking error is exactly zero and the guard fires.
    me = clamp(mean(e), extrema(e)...)
    excess_ret = me * ann
    tracking_error = std(e; mean = me) * sqrt(ann)
    information_ratio = if tracking_error > zero(tracking_error)
        excess_ret / tracking_error
    else
        oftype(one(excess_ret) / one(tracking_error), NaN)
    end
    return excess_ret, tracking_error, information_ratio
end
"""
    held_path_turnover(pred::PredictionResult)
    held_path_turnover(pred::MultiPeriodPredictionResult)

Compute the mean turnover for each rebalance of the held path of a prediction result, as [`performance_summary`](@ref) gives it.

A fold trades from the weights that the previous fold held after its last observation, and not from the target of that fold. Under a Held Weights record the two are different, so the turnover of a drifted path is not the turnover of its targets. The closed form is the `turnover` row of the `# Mathematical definition` of [`performance_summary`](@ref).

# Algorithm

 1. Stack the weights at the start of every fold, to get `W0` with one row for each fold. [`fold_target`](@ref) reads the weights, and [`stacked_fold_weights`](@ref) puts them on the union of the universes of the folds, by asset name.
 2. Stack the weights of every fold after its last observation on the same axis, to get `W1`. [`fold_held`](@ref) reads these weights.
 3. Take `F`, the number of folds. When `F < 2`, the result records no rebalance, and the answer is `nothing`.
 4. Average the trade from row `f - 1` of `W1` to row `f` of `W0` over the `F - 1` rebalances, to get the turnover.

# Arguments

  - `pred`: The prediction result.

# Returns

  - `tn::Option{<:Number}`: The mean turnover for each rebalance. It is `nothing` when the result records no rebalance, that is for a single prediction result and for a multi-period result of one fold. The summary writes `nothing` as `NaN`.

# Related

  - [`performance_summary`](@ref)
  - [`stacked_fold_weights`](@ref)
  - [`calc_turnover`](@ref)
  - [`Turnover`](@ref)
"""
function held_path_turnover(::PredictionResult)
    return nothing
end
function held_path_turnover(pred::MultiPeriodPredictionResult)
    W0 = stacked_fold_weights(pred, fold_target)
    W1 = stacked_fold_weights(pred, fold_held)
    F = size(W0, 1)
    if F < 2
        return nothing
    end
    return sum(sum(abs, view(W0, f, :) - view(W1, f - 1, :)) for f in 2:F) / (F - 1)
end
"""
    summarise_returns(ret::VecNum, turnover::Option{<:Number}; ppy::Number = 1,
                      alpha::Number = 0.05, compound::Bool = false,
                      benchmark::Option{<:VecNum} = nothing) -> PerformanceSummaryResult

Compute a [`PerformanceSummaryResult`](@ref) from a series, a turnover and an optional benchmark.

This function is the one site that computes a [`PerformanceSummaryResult`](@ref). It does the steps of [`performance_summary`](@ref). The turnover is a number that the caller read from a weight path. Every `performance_summary` method calls this function. The methods for a bare series give `nothing` for the turnover, and this function writes it as `NaN`. The prediction-result method gives the value of [`held_path_turnover`](@ref).

# Arguments

  - `ret`: Periodic portfolio returns.
  - `turnover`: Mean turnover for each rebalance, or `nothing` when the caller holds no weight path.
  - `ppy`, `alpha`, `compound`, `benchmark`: As [`performance_summary`](@ref) states them.

# Validation

  - `0 < alpha < 1` and `ppy > 0`. Otherwise the function raises a `DomainError`.
  - `length(benchmark) == length(ret)` when a benchmark is given. [`excess_statistics`](@ref) checks it, and otherwise raises a `DimensionMismatch`.

# Returns

  - `summary::PerformanceSummaryResult`: The computed summary.

# Related

  - [`performance_summary`](@ref)
  - [`excess_statistics`](@ref)
  - [`held_path_turnover`](@ref)
"""
function summarise_returns(ret::VecNum, turnover::Option{<:Number}; ppy::Number = 1,
                           alpha::Number = 0.05, compound::Bool = false,
                           benchmark::Option{<:VecNum} = nothing)::PerformanceSummaryResult
    assert_unit_interval(alpha, :alpha)
    @argcheck(ppy > zero(ppy), DomainError(ppy, "ppy must be positive"))
    T = length(ret)
    ann = ppy
    # The clamp moves nothing in exact arithmetic. On a constant series it makes the mean the
    # common value, so `s` is exactly zero and every guard below fires.
    m = clamp(mean(ret), extrema(ret)...)
    s = std(ret; mean = m)
    ann_ret = m * ann
    ann_vol = s * sqrt(ann)
    # A guarded statistic's `NaN` takes the number type its unguarded branch lands in, read
    # off `one` so no division by zero runs, and never the `Float64` of a bare literal.
    sharpe = if ann_vol > zero(ann_vol)
        ann_ret / ann_vol
    else
        oftype(one(ann_ret) / one(ann_vol), NaN)
    end
    neg_ret = min.(ret, zero(eltype(ret)))
    ddev = sqrt(mean(abs2, neg_ret) * ann)
    sortino = ddev > zero(ddev) ? ann_ret / ddev : oftype(one(ann_ret) / one(ddev), NaN)
    cret = cumulative_returns(ret, compound)
    dd_series = drawdowns(cret, compound; cX = true)
    max_dd = minimum(dd_series)
    calmar = if max_dd < zero(max_dd)
        ann_ret / abs(max_dd)
    else
        oftype(one(ann_ret) / one(max_dd), NaN)
    end
    cvar_val = -ConditionalValueatRisk(; alpha = alpha)(ret)
    # The per-period Sharpe ratio, corrected by the third and fourth standardised
    # moments. `T - 1` matches the corrected `std` used for `sharpe` itself.
    sr_p = s > zero(s) ? m / s : oftype(one(m) / one(s), NaN)
    var_sr = sharpe_ratio_variance_factor(sr_p, StatsBase.skewness(ret),
                                          StatsBase.kurtosis(ret)) / (T - 1)
    sharpe_se = if var_sr > zero(var_sr)
        sqrt(var_sr * ann)
    else
        oftype(sqrt(one(var_sr) * one(ann)), NaN)
    end
    excess_ret, tracking_error, information_ratio = excess_statistics(ret, benchmark, ann,
                                                                      m)
    return PerformanceSummaryResult(T, ann, alpha, compound, ann_ret, ann_vol, sharpe,
                                    sharpe_se, sortino, calmar, max_dd, cvar_val,
                                    excess_ret, tracking_error, information_ratio,
                                    something(turnover, oftype(ann_ret, NaN)))
end
function performance_summary(w::ArrNum, X::MatNum, fees::Option{<:Fees} = nothing;
                             kwargs...)::PerformanceSummaryResult
    return performance_summary(calc_net_returns(w, X, fees); kwargs...)
end
function performance_summary(w::ArrNum, rd::ReturnsResult, fees::Option{<:Fees} = nothing;
                             kwargs...)::PerformanceSummaryResult
    return performance_summary(w, rd.X, fees; kwargs...)
end
function performance_summary(res::OptimisationResult, rd::ReturnsResult;
                             kwargs...)::PerformanceSummaryResult
    _, w, X, fees = result_investable_view(res, rd.X)
    return performance_summary(w, X, fees; kwargs...)
end
function performance_summary(pred::PredRes_MultiPredRes;
                             benchmark::Option{<:VecNum} = regret_benchmark(pred),
                             kwargs...)::PerformanceSummaryResult
    ret, _ = regret_series(pred)
    return summarise_returns(ret, held_path_turnover(pred); benchmark = benchmark,
                             kwargs...)
end
"""
    sharpe_ratio_variance_factor(sr::Number, g1::Number, g2::Number)

Return the factor by which the skewness and the kurtosis of a return series scale the asymptotic variance of its Sharpe ratio.

The factor is the asymptotic variance of the sample Sharpe ratio for independent returns of any distribution, Equation 2 of [mertens2002](@cite). Equation 8 of [sharpe_stderr](@cite) restates it, and Section 2.5 of [sharpe_stderr](@cite) divides it by ``T - 1`` for the variance of the sample Sharpe ratio. [`performance_summary`](@ref) reads it for the standard error of the Sharpe ratio, and [`sharpe_ratio_moments`](@ref) reads it for the Probabilistic Sharpe Ratio.

# Mathematical definition

```math
\\begin{align}
v &= 1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}\\,.
\\end{align}
```

Where:

  - ``v``: Variance factor of the Sharpe ratio.
  - $(math_dict[:sr_hat])
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:gamma4_hat])

The factor is ``1 + \\widehat{\\mathrm{SR}}^{2} / 2`` for a normal distribution. The kurtosis of a distribution is at least the square of its skewness plus one, so ``v \\geq (1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} / 2)^{2} \\geq 0``. The factor is zero only for a distribution on two points.

# Arguments

  - `sr`: Sample Sharpe ratio per period.
  - `g1`: Sample skewness.
  - `g2`: Sample excess kurtosis ``\\hat{\\gamma}_{4} - 3``, the moment that `StatsBase.kurtosis` returns. So the coefficient of the square is ``(g_{2} + 2) / 4``.

# Returns

  - `v::Number`: The variance factor.

# Related

  - [`sharpe_ratio_moments`](@ref)
  - [`performance_summary`](@ref)
  - [`probabilistic_sharpe_ratio`](@ref)

# References

  - $(ref_dict[:mertens2002]) Section 1, Equation 2.
  - $(ref_dict[:sharpe_stderr]) Equation 8 and Section 2.5.
"""
function sharpe_ratio_variance_factor(sr::Number, g1::Number, g2::Number)
    # `g2` is the EXCESS kurtosis, and the coefficient is `(gamma_4 - 1) / 4` on the raw
    # one, so the `+ 2` converts between them. Without it a normal series loses the whole
    # `SR^2 / 2` term of the naive expression.
    return one(sr) - g1 * sr + (g2 + 2) / 4 * sr^2
end
"""
    sharpe_ratio_moments(x::VecNum, rf::Number)

Return the number of observations, the sample Sharpe ratio and its variance factor for a return series without its `NaN` entries.

[`probabilistic_sharpe_ratio`](@ref) and [`minimum_track_record_length`](@ref) read the three numbers. The closed forms are those of the `# Mathematical definition` of [`probabilistic_sharpe_ratio`](@ref) and of [`sharpe_ratio_variance_factor`](@ref).

# Algorithm

 1. Drop the `NaN` entries of `x`.
 2. Take `T`, the number of entries that remain. When `T < 3`, the Sharpe ratio and the factor are `NaN`. Two observations always have a skewness of zero and a kurtosis of one, so they give no data about the shape of the distribution.
 3. Take `m`, the sample mean of `x`, clamped to the least and the greatest entry, and `s`, the corrected sample standard deviation about `m`. The clamp moves nothing in exact arithmetic. On a constant series it makes `s` exactly zero. When `s` is not positive, the Sharpe ratio and the factor are `NaN`.
 4. Divide the excess `m - rf` by `s`, to get `sr`.
 5. Read the sample skewness and the sample excess kurtosis of `x` about `m`. Give them with `sr` to [`sharpe_ratio_variance_factor`](@ref), to get `v`. When `v` is not positive, it is `NaN`.

# Arguments

  - `x`: Periodic return series.
  - $(arg_dict[:rf])

# Returns

  - `(T, sr, v)`: The number of entries of `x` that are not `NaN`, the sample Sharpe ratio per period, and its variance factor.

# Related

  - [`sharpe_ratio_variance_factor`](@ref)
  - [`probabilistic_sharpe_ratio`](@ref)
  - [`minimum_track_record_length`](@ref)
"""
function sharpe_ratio_moments(x::VecNum, rf::Number)
    if any(isnan, x)
        x = filter(!isnan, x)
    end
    T = length(x)
    # The `NaN` takes the number type of the Sharpe ratio, which the square root of the
    # standard deviation widens.
    o = one(eltype(x))
    nan = oftype((o - rf) / sqrt(o), NaN)
    if T < 3
        return T, nan, nan
    end
    # The clamp moves nothing in exact arithmetic. On a constant series it makes the mean the
    # common value, so `s` is exactly zero.
    m = clamp(mean(x), extrema(x)...)
    s = std(x; mean = m)
    if !(s > zero(s))
        return T, nan, nan
    end
    sr = (m - rf) / s
    v = sharpe_ratio_variance_factor(sr, StatsBase.skewness(x, m), StatsBase.kurtosis(x, m))
    return T, sr, v > zero(v) ? v : oftype(v, NaN)
end
"""
    probabilistic_sharpe_ratio(x::VecNum; bsr::Number = 0, rf::Number = 0)
    probabilistic_sharpe_ratio(X::MatNum; bsr::Number = 0, rf::Number = 0)

Compute the Probabilistic Sharpe Ratio, the probability that the true Sharpe ratio of a return series is greater than a benchmark.

The Probabilistic Sharpe Ratio of [sharpe_stderr](@cite) corrects the sample Sharpe ratio for the length of the series and for the skewness and the kurtosis of its returns. Fat tails make a Sharpe ratio that is not zero less precise, and negative skewness makes a positive Sharpe ratio less precise. When the Sharpe ratio is above the benchmark, a less precise Sharpe ratio has a smaller probability. Every quantity is per period, so `x`, `bsr` and `rf` must have the same frequency, and the function annualises none of them. The matrix method computes one probability for each column.

# Mathematical definition

```math
\\begin{align}
\\mathrm{PSR}\\left(\\mathrm{SR}^{*}\\right) &= \\Phi\\left(\\dfrac{\\left(\\widehat{\\mathrm{SR}} - \\mathrm{SR}^{*}\\right) \\sqrt{T - 1}}{\\sqrt{1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}}}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PSR}``: Probabilistic Sharpe Ratio.
  - $(math_dict[:sr_star])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:sr_hat])
  - $(math_dict[:T])
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:gamma4_hat])
  - $(math_dict[:xret])

The denominator is the square root of the variance factor of [`sharpe_ratio_variance_factor`](@ref). A Sharpe ratio equal to the benchmark gives a probability of ``1/2``.

# Algorithm

 1. Check `bsr` and `rf`, as `# Validation` states.
 2. Take `T`, `sr` and `v`, the number of observations, the sample Sharpe ratio and its variance factor, through [`sharpe_ratio_moments`](@ref). It drops the `NaN` entries of `x`.
 3. Multiply the excess `sr - bsr` by the square root of `T - 1`, and divide it by the square root of `v`, to get `z`.
 4. Evaluate the standard normal distribution function at `z`. A `NaN` from step 2 gives a `NaN`.

# Arguments

  - $(arg_dict[:sr_x])
  - $(arg_dict[:bsr]) It defaults to `0`.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.

# Validation

  - $(val_dict[:bsr])
  - $(val_dict[:rf])

# Returns

  - `psr::Number`: The probability, for a vector `x`. It is `NaN` when `x` has fewer than three entries that are not `NaN`, when its standard deviation is zero, or when its variance factor is not positive.
  - `psr::Vector`: One probability for each column, for a matrix `X`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(probabilistic_sharpe_ratio(x); digits = 6)
0.920149

julia> round(probabilistic_sharpe_ratio(x; bsr = 0.2); digits = 6)
0.818392
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`minimum_track_record_length`](@ref)
  - [`sharpe_ratio_moments`](@ref)
  - [`performance_summary`](@ref): The standard error of its Sharpe ratio reads the same variance factor.

# References

  - $(ref_dict[:sharpe_stderr]) Equation 11.
"""
function probabilistic_sharpe_ratio(x::VecNum; bsr::Number = 0, rf::Number = 0)
    @argcheck(isfinite(bsr), IsNonFiniteError("bsr must be finite, got $bsr"))
    @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
    T, sr, v = sharpe_ratio_moments(x, rf)
    # The count enters in the type of the Sharpe ratio, so a `Float32` series stays `Float32`.
    z = (sr - bsr) * sqrt(max(T - 1, 0) * one(sr)) / sqrt(v)
    return Distributions.cdf(Distributions.Normal(zero(z), one(z)), z)
end
function probabilistic_sharpe_ratio(X::MatNum; kwargs...)
    # The `NaN` of an empty column has the type of every column's value, so the vector is
    # typed even when inference cannot see that type.
    T = typeof(probabilistic_sharpe_ratio(similar(X, 0); kwargs...))
    return T[probabilistic_sharpe_ratio(x; kwargs...) for x in eachcol(X)]
end
"""
    minimum_track_record_length(x::VecNum; bsr::Number = 0, rf::Number = 0,
                                alpha::Number = 0.05)
    minimum_track_record_length(X::MatNum; bsr::Number = 0, rf::Number = 0,
                                alpha::Number = 0.05)

Compute the Minimum Track Record Length, the number of observations that a return series needs for its Sharpe ratio to exceed a benchmark at a confidence level.

The Minimum Track Record Length of [sharpe_stderr](@cite) is the least number of observations at which the [`probabilistic_sharpe_ratio`](@ref) of the series reaches `1 - alpha`, for the sample moments of the series. It is a number of observations, not a number of years. The matrix method computes one length for each column.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MinTRL} &= 1 + \\left(1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}\\right) \\left(\\dfrac{\\Phi^{-1}(1 - \\alpha)}{\\widehat{\\mathrm{SR}} - \\mathrm{SR}^{*}}\\right)^{2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MinTRL}``: Minimum Track Record Length, a number of observations.
  - ``\\alpha``: Significance level, so ``1 - \\alpha`` is the confidence level.
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:sr_hat])
  - $(math_dict[:gamma4_hat])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:sr_star])
  - $(math_dict[:xret])

The length solves ``\\mathrm{PSR}(\\mathrm{SR}^{*}) = 1 - \\alpha`` for ``T``. When ``\\widehat{\\mathrm{SR}} \\leq \\mathrm{SR}^{*}``, the Probabilistic Sharpe Ratio is at most ``1/2`` for every ``T``, so no length is enough and the length is ``+\\infty``.

# Algorithm

 1. Check `alpha`, `bsr` and `rf`, as `# Validation` states.
 2. Take `sr` and `v`, the sample Sharpe ratio and its variance factor, through [`sharpe_ratio_moments`](@ref). It drops the `NaN` entries of `x`.
 3. Take the excess `d = sr - bsr`, and `za`, the quantile of the standard normal distribution at `1 - alpha`.
 4. Add one to `v` times the square of `za / d`, to get `n`.
 5. When `d` is not positive and `n` is not `NaN`, replace `n` with `Inf`.

# Arguments

  - $(arg_dict[:sr_x])
  - $(arg_dict[:bsr]) It defaults to `0`.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.
  - `alpha`: Significance level. It defaults to `0.05`, a confidence level of 95 %.

# Validation

  - `0 < alpha < 1/2`. The quantile ``\\Phi^{-1}(1 - \\alpha)`` is then positive, which the closed form needs.
  - $(val_dict[:bsr])
  - $(val_dict[:rf])

# Returns

  - `n::Number`: The length, for a vector `x`. It is `Inf` when the sample Sharpe ratio does not exceed `bsr`. It is `NaN` in the cases where [`probabilistic_sharpe_ratio`](@ref) is `NaN`.
  - `n::Vector`: One length for each column, for a matrix `X`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(minimum_track_record_length(x); digits = 6)
13.316333

julia> minimum_track_record_length(x; bsr = 1.0)
Inf
```

# Related

  - [`probabilistic_sharpe_ratio`](@ref)
  - [`sharpe_ratio_moments`](@ref)

# References

  - $(ref_dict[:sharpe_stderr]) Equation 13.
"""
function minimum_track_record_length(x::VecNum; bsr::Number = 0, rf::Number = 0,
                                     alpha::Number = 0.05)
    @argcheck(zero(alpha) < alpha < one(alpha) / 2,
              DomainError(alpha, "alpha must satisfy 0 < alpha < 1/2"))
    @argcheck(isfinite(bsr), IsNonFiniteError("bsr must be finite, got $bsr"))
    @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
    _, sr, v = sharpe_ratio_moments(x, rf)
    d = sr - bsr
    za = Distributions.quantile(Distributions.Normal(zero(d), one(d)), one(alpha) - alpha)
    n = one(d) + v * (za / d)^2
    # A Sharpe ratio at or below the benchmark never clears it, however long the record.
    return isnan(n) || d > zero(d) ? n : oftype(n, Inf)
end
function minimum_track_record_length(X::MatNum; kwargs...)
    # As the matrix method of `probabilistic_sharpe_ratio`.
    T = typeof(minimum_track_record_length(similar(X, 0); kwargs...))
    return T[minimum_track_record_length(x; kwargs...) for x in eachcol(X)]
end
"""
    expected_max_sharpe_ratio(N::Number, sigma::Number, mu::Number = 0)

Approximate the expected maximum of the Sharpe ratios of `N` independent trials.

The best of several trials has a Sharpe ratio that the selection itself inflates. The approximation of [bailey2014deflated](@cite) reads the trial Sharpe ratios as independent draws of one normal distribution. [`deflated_sharpe_ratio`](@ref) uses it at `mu = 0` as the benchmark that the selected trial must exceed. Trials with similar parameters have correlated Sharpe ratios, and then [`implied_independent_trials`](@ref) gives a smaller `N`.

# Mathematical definition

```math
\\begin{align}
\\mathbb{E}\\left[\\max_{n} \\widehat{\\mathrm{SR}}_{n}\\right] &\\approx \\mu + \\sigma \\max\\left(0,\\, (1 - \\gamma) \\Phi^{-1}\\left(1 - \\dfrac{1}{N}\\right) + \\gamma \\Phi^{-1}\\left(1 - \\dfrac{1}{N e}\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{SR}}_{n}``: Sharpe ratio of trial ``n``.
  - ``\\mu``: Mean of the Sharpe ratios of the trials.
  - $(math_dict[:sigma_sr_trials])
  - $(math_dict[:gamma_em])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:n_trials]) It can be fractional, and it is at least one.
  - ``e``: Euler's number.

The maximum of ``N \\geq 1`` draws is at least one draw, so its expectation is at least ``\\mu``. The floor at zero keeps that bound. The source states the approximation for ``N \\gg 1`` and has no floor. Without the floor, the approximation falls below ``\\mu`` for ``N < 1.2836``, and it is ``-\\infty`` at ``N = 1``. For one trial, or for trials with no dispersion, the expected maximum is ``\\mu``.

# Arguments

  - `N`: Number of independent trials. A fractional value is an effective number of trials.
  - `sigma`: Standard deviation of the Sharpe ratios of the trials.
  - `mu`: Mean of the Sharpe ratios of the trials.

# Validation

  - `isfinite(N)` and `N >= 1`.
  - `isfinite(sigma)` and `sigma >= 0`.
  - `isfinite(mu)`.

# Returns

  - `sr::Number`: The approximate expected maximum Sharpe ratio, in the frequency of `sigma` and `mu`.

# Examples

```jldoctest
julia> round(expected_max_sharpe_ratio(100, 1.0); digits = 6)
2.530603

julia> expected_max_sharpe_ratio(1, 0.5, 0.2)
0.2
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`implied_independent_trials`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equation 1.
"""
function expected_max_sharpe_ratio(N::Number, sigma::Number, mu::Number = 0)
    @argcheck(isfinite(N) && N >= one(N),
              DomainError(N, "N must be finite and at least one"))
    @argcheck(isfinite(sigma) && sigma >= zero(sigma),
              DomainError(sigma, "sigma must be finite and non-negative"))
    @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
    # The probabilities take the type of `N` and `sigma` together, so an integer count does
    # not widen a `Float32` dispersion.
    n = N * one(sigma)
    p1 = one(n) - inv(n)
    p2 = one(n) - inv(n * MathConstants.e)
    d = Distributions.Normal(zero(p1), one(p1))
    eg = MathConstants.eulergamma
    q1 = Distributions.quantile(d, p1)
    z = (one(q1) - eg) * q1 + eg * Distributions.quantile(d, p2)
    # The maximum of `N >= 1` draws is never below the mean of one draw, but the
    # approximation is, for `N < 1.2836`, and it is `-Inf` at `N = 1`.
    return mu + sigma * max(z, zero(z))
end
"""
    deflated_sharpe_ratio(x::VecNum_MatNum, srs::VecNum; rf::Number = 0,
                          ntrials::Option{<:Number} = nothing)

Compute the Deflated Sharpe Ratio, the probability that the true Sharpe ratio of the best of several trials is greater than zero.

The Deflated Sharpe Ratio of [bailey2014deflated](@cite) is the [`probabilistic_sharpe_ratio`](@ref) of the selected trial against the [`expected_max_sharpe_ratio`](@ref) of the search. So it corrects for the number and the dispersion of the trials, and for the skewness and the kurtosis of the selected returns. The selected trial meets a confidence level of 95 % when the ratio is at least `0.95`. Every quantity is per period, so `srs` must have the frequency of `x` and the same risk-free rate.

# Mathematical definition

```math
\\begin{align}
\\mathrm{DSR} &= \\mathrm{PSR}\\left(\\mathrm{SR}_{0}\\right)\\,,\\\\
\\mathrm{SR}_{0} &= \\sigma \\max\\left(0,\\, (1 - \\gamma) \\Phi^{-1}\\left(1 - \\dfrac{1}{N}\\right) + \\gamma \\Phi^{-1}\\left(1 - \\dfrac{1}{N e}\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DSR}``: Deflated Sharpe Ratio.
  - ``\\mathrm{PSR}``: Probabilistic Sharpe Ratio of the selected trial, as [`probabilistic_sharpe_ratio`](@ref) states it.
  - ``\\mathrm{SR}_{0}``: Expected maximum Sharpe ratio of ``N`` trials whose true Sharpe ratios are zero, as [`expected_max_sharpe_ratio`](@ref) states it.
  - $(math_dict[:sigma_sr_trials]) It is the corrected sample standard deviation of `srs`.
  - $(math_dict[:gamma_em])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:n_trials])
  - ``e``: Euler's number.

# Algorithm

 1. Check `srs`, as `# Validation` states, and drop its `NaN` entries.
 2. Take `M`, the number of entries that remain, and `sigma`, their corrected sample standard deviation. When `M = 1`, `sigma` is zero.
 3. Take the number of trials `ntrials`, or `M` when it is `nothing`, and compute the expected maximum Sharpe ratio at a zero mean with [`expected_max_sharpe_ratio`](@ref), to get `bsr`.
 4. Compute the [`probabilistic_sharpe_ratio`](@ref) of `x` against `bsr`.

# Arguments

  - $(arg_dict[:sr_x]) It holds the selected trial. Every column of a matrix meets the same benchmark.
  - `srs`: Sharpe ratios per period of every trial of the search, the selected trial included. Their number and their dispersion set the benchmark.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.
  - `ntrials`: Number of independent trials. It defaults to `nothing`, which counts every entry of `srs` that is not `NaN` as an independent trial. Correlated trials are fewer independent trials, and [`implied_independent_trials`](@ref) estimates their number.

# Validation

  - No entry of `srs` is infinite.
  - At least one entry of `srs` is not `NaN`.
  - The checks of [`expected_max_sharpe_ratio`](@ref) on `ntrials`, and the checks of [`probabilistic_sharpe_ratio`](@ref) on `rf`.

# Returns

  - `dsr::Number`: The probability, for a vector `x`. It is `NaN` in the cases where [`probabilistic_sharpe_ratio`](@ref) is `NaN`.
  - `dsr::Vector`: One probability for each column, for a matrix `x`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(deflated_sharpe_ratio(x, [0.1, 0.3, 0.5, 0.2, 0.4]); digits = 6)
0.82579
```

# Related

  - [`probabilistic_sharpe_ratio`](@ref)
  - [`expected_max_sharpe_ratio`](@ref)
  - [`implied_independent_trials`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equation 2.
"""
function deflated_sharpe_ratio(x::VecNum_MatNum, srs::VecNum; rf::Number = 0,
                               ntrials::Option{<:Number} = nothing)
    @argcheck(!any(isinf, srs), IsNonFiniteError("srs must not hold an infinite entry"))
    srs = filter(!isnan, srs)
    @argcheck(!isempty(srs),
              IsEmptyError("srs must hold at least one entry that is not NaN"))
    M = length(srs)
    s = std(srs)
    sigma = M > 1 ? s : zero(s)
    bsr = expected_max_sharpe_ratio(something(ntrials, M), sigma)
    return probabilistic_sharpe_ratio(x; bsr = bsr, rf = rf)
end
"""
    implied_independent_trials(M::Integer, rho::Number)
    implied_independent_trials(C::MatNum)

Estimate the number of independent trials among `M` trials whose returns have the mean correlation `rho`.

The estimate of [bailey2014deflated](@cite) interpolates between one trial, when every pair of trials is perfectly correlated, and `M` trials, when no pair is correlated. Give the result as `ntrials` to [`deflated_sharpe_ratio`](@ref). The matrix method reads `M` and `rho` from the correlation matrix `C` of the returns of the trials.

# Mathematical definition

```math
\\begin{align}
\\hat{\\rho} &= \\dfrac{2}{M (M - 1)} \\sum_{i=1}^{M} \\sum_{j=i+1}^{M} \\rho_{ij}\\,,\\\\
\\hat{N} &= \\hat{\\rho} + (1 - \\hat{\\rho}) M\\,.
\\end{align}
```

Where:

  - ``\\hat{N}``: Implied number of independent trials.
  - ``\\hat{\\rho}``: Mean correlation between the returns of two different trials.
  - ``M``: Number of trials.
  - ``\\rho_{ij}``: Correlation between the returns of trials ``i`` and ``j``, entry ``(i, j)`` of ``\\mathbf{C}``.
  - ``\\mathbf{C}``: Correlation matrix of the returns of the trials, ``M \\times M``.

The source takes a correlation matrix that is positive definite, so ``\\boldsymbol{1}^{\\intercal} \\mathbf{C} \\boldsymbol{1} = M + M (M - 1) \\hat{\\rho} > 0`` and ``\\hat{\\rho} > -1/(M - 1)``. A singular correlation matrix can reach the bound, and the function refuses it. The estimate is ``1`` at ``\\hat{\\rho} = 1`` and ``M`` at ``\\hat{\\rho} = 0``.

# Arguments

  - `M`: Number of trials.
  - `rho`: Mean correlation between the returns of two different trials.
  - `C`: Correlation matrix of the returns of the trials. The diagonal does not enter the mean.

# Validation

  - `M >= 1`.
  - `rho <= 1`, and `rho > -1/(M - 1)` when `M > 1`.
  - `C` is square.

# Returns

  - `n::Number`: The implied number of independent trials.

# Examples

```jldoctest
julia> implied_independent_trials(10, 0.25)
7.75
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`expected_max_sharpe_ratio`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equations 8 and 9.
"""
function implied_independent_trials(M::Integer, rho::Number)
    @argcheck(M >= 1, DomainError(M, "M must be at least one"))
    @argcheck(rho <= one(rho) && (isone(M) || rho > -inv(M - 1)),
              DomainError(rho, "rho must satisfy -1/(M - 1) < rho <= 1"))
    return rho + (one(rho) - rho) * M
end
function implied_independent_trials(C::MatNum)
    @argcheck(size(C, 1) == size(C, 2),
              DimensionMismatch("C must be square, got size $(size(C))"))
    M = size(C, 1)
    # One trial has no pair, and its mean correlation is zero.
    rho = (sum(C) - sum(LinearAlgebra.diag(C))) / max(M * (M - 1), 1)
    return implied_independent_trials(M, rho)
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, ExpectedReturn, ExpectedReturnRiskRatio, brinson_attribution,
       PerformanceSummaryResult, performance_summary, probabilistic_sharpe_ratio,
       minimum_track_record_length, expected_max_sharpe_ratio, deflated_sharpe_ratio,
       implied_independent_trials
