"""
    expected_return(ret::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; kwargs...)
    expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; kwargs...)
    expected_return(ret::NoReturn, w::VecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; kwargs...)
    expected_return(ret::JRE_VecJRE, w::VecVecNum, pr::AbstractPriorResult,
                    fees::Option{<:Fees} = nothing; kwargs...)

Compute the expected portfolio return using the specified return estimator.

`expected_return` computes the expected return for a portfolio given its weights, a prior result, and optional transaction fees. `fees` is **positional** and follows `pr`, on every method.

Each method is the scalar twin of the `ret` expression that `set_return_constraints!` builds for the same estimator, so the two sides charge the same fee. [`NoReturn`](@ref) builds a zero expression and the model charges it nothing, so `settings.fee` is inert on that term and the scalar twin charges nothing either.

The fourth method takes a vector of weight vectors, applies one of the first three to each of them, and returns one expected return per weight vector.

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
  - ``R_{\\mathrm{n}}(\\boldsymbol{w})``: Expected return of a [`NoReturn`](@ref). It carries no fee term, so no `fees` argument moves it away from zero.
  - $(math_dict[:w_port])
  - $(math_dict[:mu_er])
  - $(math_dict[:x_t_obs])
  - $(math_dict[:T])
  - ``v_{t}``: Observation weight of observation ``t``, the ``t``-th entry of the `w` field of a [`LogarithmicReturn`](@ref). Every weight is one when that field is `nothing`, and the quotient is then the unweighted mean.
  - ``F(\\boldsymbol{w})``: Fee term of [`term_fees`](@ref). It is [`calc_fees`](@ref) of `w` and `fees` when the term's `settings.fee` is `true`, and zero otherwise.

# Arguments

  - `ret`: Return estimator, or a vector of them on the fourth method.
  - `w`: Portfolio weights, or a vector of portfolio weight vectors on the fourth method.
  - `pr`: Prior result. It resolves a **Deferred Quantity** of `ret`, and it supplies `mu` to an [`ArithmeticReturn`](@ref) whose own `mu` is `nothing`.
  - `fees`: Optional fees.
  - `kwargs...`: Additional keyword arguments passed to underlying routines.

# Returns

  - `rt::Number`: Expected portfolio return, net of fees, on the first three methods.
  - `rt::VecNum`: One expected portfolio return per weight vector, on the fourth method.

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
  - [`term_fees`](@ref): Charges ``F(\\boldsymbol{w})``, and holds the rule that decides whether it is charged.
  - [`calc_fees`](@ref)
"""
function expected_return(r::ArithmeticReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
    # The scalar twin of the `ret` expression `set_return_constraints!` builds, so it uses
    # the same ladder. The prior is in hand, so a Deferred Quantity resolves here too.
    r = resolve_deferred_quantities(r, pr)
    mu = nothing_scalar_array_selector(r.mu, pr.mu)
    return LinearAlgebra.dot(w, mu) - term_fees(w, fees, r.settings.fee)
end
function expected_return(ret::LogarithmicReturn, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
    rw = ret.w
    X = pr.X
    kret = if isnothing(rw)
        Statistics.mean(log1p.(X * w))
    else
        Statistics.mean(log1p.(X * w), rw)
    end
    return kret - term_fees(w, fees, ret.settings.fee)
end
function expected_return(::NoReturn, w::VecNum, ::AbstractPriorResult,
                         ::Option{<:Fees} = nothing; kwargs...)
    # The scalar twin of a zero expression. No charge is applied, exactly as
    # `set_return_constraints!` applies none: `settings.fee` is inert on this term.
    return zero(eltype(w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Charge the fees to a return term only when that term's `settings.fee` says so.

The scalar twin follows the **fee** flag alone. Market impact is absent from `expected_return`
on either side of the multiplicity, so this preserves a pre-existing divergence between the
model expression and its scalar twin rather than widening one.

# Arguments

  - `w`: Portfolio weights.
  - `fees`: Optional fees.
  - `fee`: The term's `settings.fee` flag.

# Returns

  - `f::Number`: [`calc_fees`](@ref) of `w` and `fees` when `fee` is `true`, and a zero of the element type of `w` otherwise.

# Related

  - [`expected_return`](@ref)
  - [`JuMPReturnsSettings`](@ref)
  - [`calc_fees`](@ref)
"""
function term_fees(w::VecNum, fees::Option{<:Fees}, fee::Bool)
    return fee ? calc_fees(w, fees) : zero(eltype(w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sum the value-level returns of several return terms, each at its own scale.

The scalar twin of the model's `ret` expression, which is the same weighted sum over the same
terms. A term whose `settings.rte` is `false` is skipped, exactly as it is skipped in the
model.

This is the one carve-out from the rule that the value-level `expected_*` family stays
singular: [`NearOptimalCentering`](@ref)'s barrier needs the aggregate scalar, and this
function declares itself the scalar twin of the `ret` expression. The rest of the family, and
the whole risk side, still take one measure and one term.

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

 1. Set the accumulator `rt` to a zero of the element type of `w`.
 2. Take the next term `ret_i` of `ret`. Skip it when its `settings.rte` is `false`.
 3. Compute the value-level return of `ret_i` with [`expected_return`](@ref), scale it by the term's `settings.scale`, and add it to `rt`.
 4. Repeat steps 2 and 3 over the remaining terms, then return `rt`.

# Arguments

  - `ret`: Vector of return estimators.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional fees.
  - `kwargs...`: Additional keyword arguments passed to underlying routines.

# Returns

  - `rt::Number`: Aggregate expected portfolio return of the terms whose `settings.rte` is `true`.

# Related

  - [`expected_return`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function expected_return(ret::VecJRE, w::VecNum, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing; kwargs...)
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
                         fees::Option{<:Fees} = nothing; kwargs...)
    return [expected_return(ret, wi, pr, fees; kwargs...) for wi in w]
end
"""
    expected_ratio(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                   pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                   rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)

Compute the expected risk-adjusted return ratio for a portfolio.

`expected_ratio` computes the ratio of expected portfolio return (net of fees and risk-free rate) to expected portfolio risk, using the specified risk measure and return estimator.

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

Both axes take one item or several, and the ratio is **aggregate over aggregate**: the risk axis is scalarised by `sca` into one number, the return axis is summed at its terms' own `settings.scale` weights, and the two aggregates are divided.

Only the risk axis carries a scalariser. The return axis sums and never scalarises.

# Resolution

Both axes are handed `pr` itself, so both apply the same precedence rule: a stated slot wins, then the slot's own estimator, then the prior's field. The risk axis enters [`expected_risk`](@ref)'s prior route, which resolves the measure through [`resolve_risk_inputs`](@ref) once per call — a **Deferred Quantity** is fitted against `pr`, and an unstated slot takes the prior's field.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser combining a vector `r` into one number. Inert on a single measure.
  - `kwargs...`: Additional keyword arguments.

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
                        rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)
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
                            rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)

Compute expected risk, expected return, and risk-adjusted return ratio for a portfolio.

`expected_risk_ret_ratio` returns a tuple containing the expected portfolio risk, expected portfolio return, and the risk-adjusted return ratio, using the specified risk measure and return estimator.

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

Both axes take one item or several, and the ratio is **aggregate over aggregate**, exactly as in [`expected_ratio`](@ref): the risk axis is scalarised by `sca` into one number, the return axis is summed at its terms' own `settings.scale` weights, and the two aggregates are divided. The returned `risk` and `return` are those two aggregates, never a per-element breakdown.

Only the risk axis carries a scalariser. The return axis sums and never scalarises.

# Reporting a result

This is the documented route out of an optimisation result. A result carries the measure and the scalariser it ran under, so it can name them itself:

```julia
rk, rt, rr = expected_risk_ret_ratio(res.r, res.ret, res.w, res.pr; sca = res.sca, rf = rf)
```

`res.r` and `res.ret` are stored **resolved**, so this route reports the aggregates the optimisation actually optimised, whether either axis held one item or several. Naming the measures by hand instead is still supported, but the figure then matches the optimisation only if the caller names the same measures **and** the same scalariser; `fees` and `rf` carry the same responsibility.

A result with no `r` — one on the [`NonRiskJuMPOptimisationResult`](@ref) half — resolved no measure, so there is nothing to pass back and the caller must name one.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser combining a vector `r` into one number. Inert on a single measure.
  - `kwargs...`: Additional keyword arguments.

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
                                 kwargs...)
    # The prior, not `pr.X` — see the note in `expected_ratio`.
    rk = expected_risk(r, w, pr, fees; sca = sca, kwargs...)
    rt = expected_return(ret, w, pr, fees; kwargs...)
    return rk, rt, (rt - rf) / rk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the estimation-error penalty that turns a ratio into a SRIC.

The penalty is applied once to the aggregate ratio, never per element. This is the only
difference between [`expected_sric`](@ref) and [`expected_ratio`](@ref), and between
[`expected_risk_ret_sric`](@ref) and [`expected_risk_ret_ratio`](@ref), so both SRIC
functions delegate here rather than restate the ratio.

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
  - `pr`: Prior result. Its returns matrix supplies ``T`` and ``N``.

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
                  rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)

Compute the risk-adjusted ratio information criterion (SRIC) for a portfolio.

`expected_sric` computes the SRIC, which adjusts the risk-adjusted return ratio for estimation error, penalizing overfitting in portfolio optimization. The SRIC is computed as the risk-adjusted return ratio minus a penalty term based on the number of assets and sample size.

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

The ratio it penalises is [`expected_ratio`](@ref)'s, so it inherits the same rule: **aggregate over aggregate**. The risk axis is scalarised by `sca`, the return axis is summed at its terms' own `settings.scale` weights, and the penalty is applied once to the resulting scalar — not per element.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate.
  - `sca`: Scalariser combining a vector `r` into one number. Inert on a single measure.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `sric::Number`: Sharpe Ratio Information Criterion.

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
                       rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)
    sr = expected_ratio(r, ret, w, pr, fees; rf = rf, sca = sca, kwargs...)
    return sr - sric_penalty(sr, pr)
end
"""
    expected_risk_ret_sric(r::BaseRM_VecBaseRM, ret::JRE_VecJRE, w::VecNum,
                           pr::AbstractPriorResult, fees::Option{<:Fees} = nothing;
                           rf::Number = 0, sca::Scalariser = SumScalariser(), kwargs...)

Compute expected risk, expected return, and SRIC for a portfolio.

`expected_risk_ret_sric` returns a tuple containing the expected portfolio risk, expected portfolio return, and the Sharpe Ratio Information Criterion (SRIC), which adjusts the risk-adjusted return ratio for estimation error.

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

The tuple is [`expected_risk_ret_ratio`](@ref)'s with the penalty applied to its third element, so it inherits the same rule: **aggregate over aggregate**. The returned `risk` and `return` are the two aggregates, never a per-element breakdown, and the penalty is applied once to the resulting scalar.

# Arguments

  - `r`: Risk measure, or a vector of them.
  - `ret`: Return estimator, or a vector of them.
  - `w`: Portfolio weights.
  - `pr`: Prior result (must contain asset return matrix `X`).
  - `fees`: Optional transaction fees.
  - `rf`: Risk-free rate (default: 0).
  - `sca`: Scalariser combining a vector `r` into one number. Inert on a single measure.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Number`: Expected portfolio risk.
  - `return::Number`: Expected portfolio return.
  - `sric::Number`: Sharpe Ratio Information Criterion.

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
                                kwargs...)
    rk, rt, sr = expected_risk_ret_ratio(r, ret, w, pr, fees; rf = rf, sca = sca, kwargs...)
    return rk, rt, sr - sric_penalty(sr, pr)
end
"""
$(DocStringExtensions.TYPEDEF)

Return-based risk measure.

`ExpectedReturn` is a risk measure that uses the expected portfolio return as its risk metric. This is useful for algorithms or analyses where the risk is defined as the expected return, used in portfolio performance analysis.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ER}(\\boldsymbol{w}) &= \\sum_{j \\,:\\, e_{j}} s_j\\, R_j(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{ER}(\\boldsymbol{w})``: Expected portfolio return, used here as the risk metric.
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

`rt` takes one return term or a vector of them, so `ExpectedReturn(; rt = res.ret)` works after a solve against several terms. A vector is summed at its terms' `settings.scale` weights, skipping any term whose `settings.rte` is `false`.

The type gains **no** scalariser: the return axis sums and never scalarises.

## Validation

  - If `rt` is a vector: `!isempty(rt)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `rt`: Recursively updated via [`factory`](@ref).

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
           │         mu ┴ nothing
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

Compute the expected risk for a portfolio using a return-based risk measure.

`expected_risk` returns the expected portfolio return as the risk metric, using the specified return estimator in the [`ExpectedReturn`](@ref). This is useful for algorithms where risk is defined as expected return. The closed form is the one [`ExpectedReturn`](@ref) states, and the method delegates to [`expected_return`](@ref) on the measure's `rt` field.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) containing a return estimator.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::Number`: Expected portfolio return (net of fees if provided).

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

Ratio-based risk measure.

`ExpectedReturnRiskRatio` is a risk measure that computes the risk-adjusted return ratio, such as the Sharpe ratio, for a portfolio. It combines a return estimator, a risk measure, and a risk-free rate to produce a ratio metric, used in portfolio performance analysis.

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

Both axes widen, and only the **risk** axis carries a scalariser. `rt` sums its terms at their own `settings.scale` weights, skipping any term whose `settings.rte` is `false`; `rk` is scalarised into one number by `sca`. The ratio is then aggregate over aggregate. The return axis sums and never scalarises.

The field beats a caller's `sca` keyword. A `sca` passed at the call site flows no further than this type, so a figure reported from an `ExpectedReturnRiskRatio` is always the one the type names.

## Validation

  - If `rt` is a vector: `!isempty(rt)`.
  - If `rk` is a vector: `!isempty(rk)`.
  - $(val_dict[:rf])

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `rt`: Recursively updated via [`factory`](@ref).
  - `rk`: Recursively updated via [`factory`](@ref).

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
           │         mu ┴ nothing
        rk ┼ Variance
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: true
           │      sigma ┼ nothing
           │       chol ┼ nothing
           │         rc ┼ nothing
           │        alg ┴ SquaredSOCRiskExpr()
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
    expected_risk(r::ExpectedReturnRiskRatio, w::VecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; kwargs...)

Compute the expected risk for a portfolio using a ratio-based risk measure.

`expected_risk` returns the risk-adjusted return ratio (e.g., Sharpe ratio) for the portfolio, using the specified return estimator, risk measure, and risk-free rate in the [`ExpectedReturnRiskRatio`](@ref). The closed form is the one [`ExpectedReturnRiskRatio`](@ref) states, and the method delegates to [`expected_ratio`](@ref) on the measure's fields.

The measure's own `sca` and `rf` are pinned after `kwargs...` is splatted, so a `sca` or an `rf` supplied at the call site loses to the field. A figure reported from an [`ExpectedReturnRiskRatio`](@ref) is therefore always the one the type names.

# Arguments

  - `r`: [`ExpectedReturnRiskRatio`](@ref) containing a return estimator, risk measure, and risk-free rate.
  - `w`: Portfolio weights.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

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
    const PerfRM = Union{<:MeanReturn, <:MeanReturnRiskRatio, <:ExpectedReturn,
                         <:ExpectedReturnRiskRatio}

Union of performance risk measures used to compute portfolio performance metrics (returns and return/risk ratios).

The group exists because every one of its members reports a **performance** figure rather than a loss, so a larger value is a better one. [`bigger_is_better`](@ref) dispatches on this alias and answers `true` for all four in one method, which is the whole reason the four are grouped.

# Related

  - [`MeanReturn`](@ref): reads a realised return series.
  - [`MeanReturnRiskRatio`](@ref): reads a realised return series.
  - [`ExpectedReturn`](@ref): reads a prior result, so it is also a [`PrRM`](@ref).
  - [`ExpectedReturnRiskRatio`](@ref): reads a prior result, so it is also a [`PrRM`](@ref).
  - [`PrRM`](@ref): the two members of this group that read a prior result.
  - [`bigger_is_better`](@ref): the method that dispatches on this alias.
"""
const PerfRM = Union{<:MeanReturn, <:MeanReturnRiskRatio, <:ExpectedReturn,
                     <:ExpectedReturnRiskRatio}
"""
    const PrRM = Union{<:ExpectedReturn, <:ExpectedReturnRiskRatio}

Union of prior-based return risk measures that are incompatible with [`PredictionResult`](@ref) inputs and require the use of [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) instead.

The group exists because both members read the **prior result** itself rather than a returns matrix, so neither declares a [`risk_input_kind`](@ref). That one property drives both methods that dispatch on the alias: the vector-of-weights route resolves the measure once and keeps the prior in hand, and the prediction-result routes refuse the call.

# Related

  - [`ExpectedReturn`](@ref)
  - [`ExpectedReturnRiskRatio`](@ref)
  - [`PerfRM`](@ref): the wider group both members also belong to.
  - [`prrm_prediction_message`](@ref): builds the refusal the prediction-result methods raise.
  - [`risk_input_kind`](@ref): the declaration neither member makes.
"""
const PrRM = Union{<:ExpectedReturn, <:ExpectedReturnRiskRatio}
"""
    expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult,
                  fees::Option{<:Fees} = nothing; kwargs...)

Evaluate a prior-reading risk measure over a vector of weight vectors, such as the points of a
[`Frontier`](@ref).

A [`PrRM`](@ref) reads the prior result itself and not a returns matrix — it declares no
[`risk_input_kind`](@ref) for that reason. The generic `VecVecNum` method resolves the measure and
then hands the loop `pr.X`, which is right for a measure whose kernel takes the matrix and strips
what this family needs. So the family resolves once here and maps with the prior still in hand.

# Algorithm

 1. Resolve the measure against the prior once, with [`factory`](@ref), giving a measure whose slots are filled.
 2. Evaluate the resolved measure on each weight vector `wi` of `w`, with the prior itself as the input rather than `pr.X`.
 3. Collect one expected risk per weight vector, in the order of `w`.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) or [`ExpectedReturnRiskRatio`](@ref).
  - `w`: Vector of portfolio weight vectors.
  - `pr`: Prior result.
  - `fees`: Optional transaction fees.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `risk::VecNum`: One expected risk per weight vector.

# Related

  - [`PrRM`](@ref)
  - [`expected_risk`](@ref)
  - [`resolve_risk_inputs`](@ref)
  - [`factory`](@ref)
"""
function expected_risk(r::PrRM, w::VecVecNum, pr::AbstractPriorResult,
                       fees::Option{<:Fees} = nothing; kwargs...)
    r = factory(r, pr)
    return [expected_risk(r, wi, pr, fees; kwargs...) for wi in w]
end
function bigger_is_better(::PerfRM)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the refusal a [`PrRM`](@ref) raises on a prediction result.

The refusal itself is about **prior versus realised returns**, and multiplicity does not touch it: a [`PrRM`](@ref) reads a prior result, and a prediction result carries a realised series instead.

The suggested replacement does depend on multiplicity. [`MeanReturn`](@ref) carries **no** return estimator, so it cannot hold the several terms a widened `rt` holds. When `rt` is a vector the message says so, rather than telling the caller that a lossy substitution is equivalent.

# Algorithm

 1. Choose the replacement measure `alt` from the kind of `r`: [`MeanReturn`](@ref) for an [`ExpectedReturn`](@ref), and [`MeanReturnRiskRatio`](@ref) for an [`ExpectedReturnRiskRatio`](@ref).
 2. Write the first sentence of `msg`, naming `r`, the wrapper type of `pred` and `alt`.
 3. When the `rt` of `r` is a vector, append the second sentence, naming the number of terms it holds and the two routes that keep them.

# Arguments

  - `r`: The measure that refuses the call.
  - `pred`: The prediction result the caller supplied.

# Returns

  - `msg::String`: The message of the `ArgumentError` the refusing methods raise.

# Related

  - [`PrRM`](@ref)
  - [`MeanReturn`](@ref)
  - [`MeanReturnRiskRatio`](@ref)
  - [`expected_risk`](@ref): the four methods that raise this message.
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

Refuse to evaluate a prior-reading risk measure against a prediction result.

A [`PrRM`](@ref) reads a **prior** result: it takes the expected returns the prior states and contracts them with the weights. A prediction result carries a **realised** return series instead, and it states no expected returns, so there is nothing for these two measures to read. The four methods exist to say so, and each of them raises rather than returning a figure computed from the wrong input.

The caller has two routes. To measure the realised series, name [`MeanReturn`](@ref) or [`MeanReturnRiskRatio`](@ref) in place of the measure and call `expected_risk(alt, pred)`. To measure the prior instead, call `expected_risk(r, w, pr)` with the prior result and the weights, which is the route the first two methods of this file document.

# Arguments

  - `r`: [`ExpectedReturn`](@ref) or [`ExpectedReturnRiskRatio`](@ref).
  - `pred`: The prediction result the caller supplied.
  - `kwargs...`: Additional keyword arguments. They are read by none of the four methods.

# Validation

  - Every method raises `ArgumentError`, with the message [`prrm_prediction_message`](@ref) builds. There is no input on which any of them returns.

# Related

  - [`PrRM`](@ref)
  - [`prrm_prediction_message`](@ref): builds the message of the raise.
  - [`MeanReturn`](@ref): the replacement for an [`ExpectedReturn`](@ref).
  - [`MeanReturnRiskRatio`](@ref): the replacement for an [`ExpectedReturnRiskRatio`](@ref).
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

Compute Brinson performance attribution aggregated per asset class [brinson_attribution](@cite).

`brinson_attribution` generates a DataFrame summarizing the Brinson performance attribution, decomposing total excess return into asset allocation, security selection, interaction, and total effect for each asset class. The calculation is performed over the specified date range, using the provided asset weights, benchmark weights, and asset class assignments.

`date0` and `date1` are **positional**, and they follow `col`. Both must be given for the filter to apply: a call that names only one of them attributes the whole of `X`.

`X` holds **prices**, not returns. The period return of an asset is the ratio of its last value in the range to its first value, less one, which is a return only when the entries are prices.

Both class returns divide by the class weight, and neither division is guarded. A class holding zero portfolio weight makes its ``r_{i}`` a `NaN`, and a class holding zero benchmark weight makes its ``r_{i}^{b}`` a `NaN`. The two spread differently, because ``\\mathrm{AA}_{i}`` reads only ``r_{i}^{b}``. A zero portfolio weight leaves ``\\mathrm{AA}_{i}`` finite and makes the other three rows of that class a `NaN`; a zero benchmark weight makes all four a `NaN`. The `Total` column is a row sum, so a row is a `NaN` exactly when one of its class entries is. Drop the empty class from `asset_classes` to remove it.

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
  - ``r_{i}``, ``r_{i}^{b}``: Portfolio and benchmark return of the class. Each is the average of its members' returns, normalised by the weight of the class, so a class return is a weighted average and not a weighted sum.
  - ``r^{b}``: Benchmark return of the whole universe.
  - ``\\mathrm{AA}_{i}``: Asset allocation of the class, the part of the excess return that the class weight decision earned.
  - ``\\mathrm{SS}_{i}``: Security selection of the class, the part that the choice of assets inside the class earned.
  - ``\\mathrm{I}_{i}``: Interaction of the class, the product of the two differences.
  - ``\\mathrm{TER}_{i}``: Total excess return of the class, the sum of the three.

# Algorithm

 1. Filter the date range. When `date0` and `date1` are both given, find the first and the last observation of `X` inside the range, giving `idx1` and `idx2`. Otherwise take the whole of `X`, so `idx1` is `1` and `idx2` is `length(X)`.
 2. Form `ret`, the period return of each asset, from the values at `idx1` and `idx2`.
 3. Contract `ret` with `wb`, giving `ret_b`, the benchmark return of the whole universe.
 4. Read the class of each asset from column `col` of `asset_classes`, and build `sets_mat`, the membership matrix whose entry is `true` when the asset belongs to the class.
 5. Take the next class. Sum the two weight vectors over its members, giving `w_i` and `wb_i`, and form its two weight-normalised returns `ret_i` and `ret_b_i`.
 6. Compute `AA_i`, `SS_i`, `I_i` and `TER_i` of that class, and write the four numbers as its column of `df`.
 7. Repeat steps 5 and 6 over the remaining classes.
 8. Append the `Total` column, the row-wise sum over the class columns.

# Arguments

  - `X`: TimeArray of asset prices.
  - `w`: Vector of portfolio weights.
  - `wb`: Vector of benchmark weights.
  - `asset_classes`: DataFrame containing asset class assignments for each asset.
  - `col`: Column in `asset_classes` specifying the class for each asset.
  - `date0`: (Optional) Start date for the attribution period.
  - `date1`: (Optional) End date for the attribution period.

# Returns

  - `df::DataFrame`: DataFrame with rows for asset allocation, security selection, interaction, and total effect, and columns for each asset class and the total.

# Related

  - [`VecNum`](@ref)
  - [`performance_summary`](@ref): summarises one realised return series, where this function splits one period across the classes.
  - [`calc_net_returns`](@ref): the net return series of a portfolio, which charges fees. This function charges none.
  - [`AssetSetsMatrixEstimator`](@ref): the library's own group-membership matrix. Step 4 builds its own from a DataFrame column instead.

# References

  - $(ref_dict[:brinson_attribution])
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

`PerformanceSummaryResult` is what [`performance_summary`](@ref) returns. It carries the seven numbers [`plot_performance_summary`](@ref) draws, the standard error of the Sharpe ratio, and the four inputs that produced them, so a summary can be tabulated, compared across runs, asserted on in a test, or fed to a search without a plotting package installed.

# Sign convention

`max_drawdown` and `cvar` are reported in **return space**, so both are non-positive for a portfolio that ever loses. This is the opposite sign to the [`ConditionalValueatRisk`](@ref) risk measure, which reports a positive loss: `ConditionalValueatRisk(; alpha = alpha)(ret) == -summary.cvar`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PerformanceSummaryResult(
        n_periods, periods_per_year, alpha, compound,
        ann_return, ann_volatility, sharpe, sharpe_stderr,
        sortino, calmar, max_drawdown, cvar
    ) -> PerformanceSummaryResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a
Result, so [`performance_summary`](@ref) builds it and a caller reads it; there is no keyword
constructor, and the type validates nothing of its own. The four inputs it carries first are
the arguments that produced the statistics after them.

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
    periods_per_year
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
end
"""
    performance_summary(ret::VecNum; periods_per_year::Number = 252, alpha::Number = 0.05,
                        compound::Bool = false) -> PerformanceSummaryResult
    performance_summary(w::ArrNum, X::MatNum, fees::Option{<:Fees} = nothing;
                        periods_per_year, alpha, compound) -> PerformanceSummaryResult
    performance_summary(w::ArrNum, rd::ReturnsResult, fees::Option{<:Fees} = nothing;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(res::OptimisationResult, rd::ReturnsResult;
                        kwargs...) -> PerformanceSummaryResult
    performance_summary(pred::PredictionResult; kwargs...) -> PerformanceSummaryResult
    performance_summary(pred::MultiPeriodPredictionResult;
                        kwargs...) -> PerformanceSummaryResult

Summarise a realised return series as a [`PerformanceSummaryResult`](@ref).

The weight-and-returns methods net the returns through [`calc_net_returns`](@ref) first, so a summary of a portfolio accounts for its fees.

# Mathematical definition

Let ``m`` and ``s`` be the sample mean and the sample standard deviation of the periodic returns ``\\boldsymbol{r}``, let ``p`` be `periods_per_year`, and let ``\\boldsymbol{d}`` be the drawdown path of the cumulative wealth series.

```math
\\begin{align}
\\mathrm{ann\\_return} &= m p\\\\
\\mathrm{ann\\_volatility} &= s \\sqrt{p}\\\\
\\mathrm{sharpe} &= \\dfrac{\\mathrm{ann\\_return}}{\\mathrm{ann\\_volatility}}\\\\
\\mathrm{sortino} &= \\dfrac{\\mathrm{ann\\_return}}{\\sqrt{p \\, \\mathbb{E}\\left[\\min(\\boldsymbol{r}, 0)^{2}\\right]}}\\\\
\\mathrm{calmar} &= \\dfrac{\\mathrm{ann\\_return}}{\\left\\lvert \\min_{t} d_{t} \\right\\rvert}\\,.
\\end{align}
```

The downside deviation of the Sortino ratio divides by ``T`` rather than by the count of losing periods, which is the Sortino and Price convention and the one that keeps the ratio comparable across series with different loss frequencies.

The standard error of the Sharpe ratio is the Bailey and Lopez de Prado [sharpe_stderr](@cite) expression, which corrects for the skewness ``g_{1}`` and the excess kurtosis ``g_{2}`` of the returns:

```math
\\begin{align}
\\mathrm{sharpe\\_stderr} &= \\sqrt{\\dfrac{p}{T - 1} \\left(1 - g_{1} \\dfrac{m}{s} + \\dfrac{g_{2} + 2}{4} \\dfrac{m^{2}}{s^{2}}\\right)}\\,.
\\end{align}
```

The kurtosis term is ``(g_{2} + 2)/4`` because the source states it as ``(\\gamma_{2} - 1)/4`` on the **raw** fourth standardised moment ``\\gamma_{2}``, and `StatsBase.kurtosis` returns the **excess** moment ``g_{2} = \\gamma_{2} - 3``. The two forms agree, and only this one reduces to the naive ``\\sqrt{(1 + \\mathrm{SR}^{2}/2)/T}`` on a normal series: over 200,000 samples of 250 normal returns at a per-period Sharpe ratio of 0.5, the sample standard deviation of the Sharpe ratio is **0.06728323**, against **0.06721661** from this expression evaluated at the population moments and **0.06337243** from the same expression without the ``+2``.

A non-normal return series makes a Sharpe ratio less precise than the naive expression suggests, and negative skew makes it worse. That is the case that matters for a real portfolio, which is why the standard error ships beside the ratio.

# Algorithm

The five other methods reduce to the `ret::VecNum` method, which runs the steps. A method that takes weights nets the returns through [`calc_net_returns`](@ref) first, and a method that takes a prediction result takes the first series when it carries several.

 1. Check `alpha` and `periods_per_year`, as `# Validation` states.
 2. Take `T`, the number of periods, and `m` and `s`, the sample mean and the corrected sample standard deviation of `ret`.
 3. Annualise the two, giving `ann_ret` and `ann_vol`.
 4. Divide, giving `sharpe`. A non-positive `ann_vol` gives a `NaN` in its place.
 5. Clip every positive entry of `ret` to zero, square, average over all `T` periods and annualise, giving the downside deviation `ddev`. Divide, giving `sortino`. A non-positive `ddev` gives a `NaN`.
 6. Build the cumulative wealth series with [`cumulative_returns`](@ref), take its drawdown path with [`drawdowns`](@ref), and read `max_dd`, the minimum of that path. Divide, giving `calmar`. A non-negative `max_dd` gives a `NaN`.
 7. Evaluate [`ConditionalValueatRisk`](@ref) at `alpha` on `ret` and negate it, giving `cvar_val` in return space.
 8. Form the per-period Sharpe ratio `sr_p`, correct its variance `var_sr` by the sample skewness and the sample excess kurtosis of `ret`, and annualise the square root, giving `sharpe_se`. A non-positive `var_sr` gives a `NaN`.
 9. Collect the four inputs and the eight statistics into a [`PerformanceSummaryResult`](@ref).

# Arguments

  - `ret`: Periodic portfolio returns vector.
  - `w`: Portfolio weights.
  - $(arg_dict[:X])
  - `fees`: Optional transaction fees.
  - `rd`: [`ReturnsResult`](@ref) carrying the asset returns.
  - `res`: An [`OptimisationResult`](@ref), whose fees are extracted with [`extract_fees`](@ref).
  - $(arg_dict[:ps_ppy])
  - $(arg_dict[:ps_alpha])
  - $(arg_dict[:ps_compound])

# Validation

  - `0 < alpha < 1`.
  - `periods_per_year > 0`.

# Returns

  - `summary::PerformanceSummaryResult`: The computed summary.

# Related

  - [`PerformanceSummaryResult`](@ref)
  - [`plot_performance_summary`](@ref)
  - [`calc_net_returns`](@ref)
  - [`cumulative_returns`](@ref)
  - [`drawdowns`](@ref)

# References

  - $(ref_dict[:sharpe_stderr])
"""
function performance_summary(ret::VecNum; periods_per_year::Number = 252,
                             alpha::Number = 0.05,
                             compound::Bool = false)::PerformanceSummaryResult
    assert_unit_interval(alpha, :alpha)
    @argcheck(periods_per_year > zero(periods_per_year),
              DomainError(periods_per_year, "periods_per_year must be positive"))
    T = length(ret)
    ann = periods_per_year
    m = mean(ret)
    s = std(ret)
    ann_ret = m * ann
    ann_vol = s * sqrt(ann)
    sharpe = ann_vol > zero(ann_vol) ? ann_ret / ann_vol : NaN
    neg_ret = min.(ret, zero(eltype(ret)))
    ddev = sqrt(mean(neg_ret .^ 2) * ann)
    sortino = ddev > zero(ddev) ? ann_ret / ddev : NaN
    cret = cumulative_returns(ret, compound)
    dd_series = drawdowns(cret, compound; cX = true)
    max_dd = minimum(dd_series)
    calmar = max_dd < zero(max_dd) ? ann_ret / abs(max_dd) : NaN
    cvar_val = -ConditionalValueatRisk(; alpha = alpha)(ret)
    # The per-period Sharpe ratio, corrected by the third and fourth standardised
    # moments. `T - 1` matches the corrected `std` used for `sharpe` itself.
    sr_p = s > zero(s) ? m / s : NaN
    # `StatsBase.kurtosis` is the EXCESS moment, and the source's coefficient is
    # `(gamma_2 - 1) / 4` on the raw one, so the `+ 2` converts between them. Without it a
    # normal series loses the whole `SR^2 / 2` term the naive expression carries.
    var_sr = (one(sr_p) - StatsBase.skewness(ret) * sr_p +
              (StatsBase.kurtosis(ret) + 2) / 4 * sr_p^2) / (T - 1)
    sharpe_se = var_sr > zero(var_sr) ? sqrt(var_sr * ann) : NaN
    return PerformanceSummaryResult(T, ann, alpha, compound, ann_ret, ann_vol, sharpe,
                                    sharpe_se, sortino, calmar, max_dd, cvar_val)
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
    return performance_summary(res.w, rd.X, extract_fees(res, nothing); kwargs...)
end
function performance_summary(pred::PredictionResult; kwargs...)::PerformanceSummaryResult
    rd = pred.rd
    ret = isa(rd.X, VecVecNum) ? first(rd.X) : rd.X
    return performance_summary(ret; kwargs...)
end
function performance_summary(pred::MultiPeriodPredictionResult;
                             kwargs...)::PerformanceSummaryResult
    mrd = pred.mrd
    ret = isa(mrd.X, VecVecNum) ? first(mrd.X) : mrd.X
    return performance_summary(ret; kwargs...)
end

export expected_return, expected_ratio, expected_risk_ret_ratio, expected_sric,
       expected_risk_ret_sric, ExpectedReturn, ExpectedReturnRiskRatio, brinson_attribution,
       PerformanceSummaryResult, performance_summary
