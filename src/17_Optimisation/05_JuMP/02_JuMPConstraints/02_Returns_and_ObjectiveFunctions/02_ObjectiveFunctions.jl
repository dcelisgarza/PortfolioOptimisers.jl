"""
$(DocStringExtensions.TYPEDEF)

Objective function that minimises portfolio risk.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min}\\; R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:R_w])

The objective reads no return expression, so it admits a [`NoReturn`](@ref) term and a zero
`:ret`. It also states no floor on the return, so a term's `settings.lb` is the only thing
that keeps the solution off a portfolio whose expected return is negative.

# Related

  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`JuMPReturnsSettings`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.1, equation 8.7.
  - $(ref_dict[:markowitz1952])
"""
struct MinimumRisk <: ObjectiveFunction end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises risk-adjusted utility.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - l\\, R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``l``: Risk-aversion coefficient.
  - $(math_dict[:R_w])

The risk carries the coefficient **whole**: the model builds `ret - l * risk`, with no factor
of one half. A caller porting a ``\\tfrac{\\lambda}{2}`` convention halves its own ``\\lambda``. At
`l = 3.0` on a five-asset sample the model's own objective is **0.0015922030484060072**, which
is `ret - 3.0 * risk` exactly; the halved form would give **0.0018692372364207165**.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumUtility(; l::Number = 2) -> MaximumUtility

Keywords correspond to the struct's fields.

## Validation

  - `l >= 0`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.3, equation 8.12.
  - $(ref_dict[:markowitz1952])
"""
@concrete struct MaximumUtility <: ObjectiveFunction
    """
    $(field_dict[:l])
    """
    l
    function MaximumUtility(l::Number)
        @argcheck(l >= zero(l), DomainError(l, "l must be >= 0"))
        return new{typeof(l)}(l)
    end
end
function MaximumUtility(; l::Number = 2)
    return MaximumUtility(l)
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises the risk-adjusted Sharpe-type ratio.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f}{R(\\boldsymbol{w})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``r_f``: Risk-free rate.
  - $(math_dict[:R_w])

The quotient is not convex, so the model solves the equivalent fractional programme instead.
It homogenises the whole feasible set by a scalar ``k \\geq 0``, optimises in
``\\boldsymbol{y} = k \\boldsymbol{w}``, and de-homogenises with
``\\boldsymbol{w} = \\boldsymbol{y} / k``. One of two normalisations pins ``k``:

  - **Return form.** ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{y} - r_f k = \\mathrm{ohf}``, and the model minimises the risk. This is the branch [`set_max_ratio_return_constraints!`](@ref) registers as `sr_ret`.
  - **Risk form.** ``R(\\boldsymbol{y}) \\leq \\mathrm{ohf}``, and the model maximises ``\\mathrm{ret} - r_f k``. This is the branch registered as `sr_risk`.

The return form fails whenever the portfolio's expected return cannot exceed ``r_f``, so the
risk form takes over there, and wherever a term raises a cone the return form cannot carry.
[`set_max_ratio_return_constraints!`](@ref) states the exact test.

## The scale floor

Homogenisation carries a degenerate ray. Every constraint the model writes in ``\\boldsymbol{y}``
is homogeneous — ``\\mathbf{A}\\boldsymbol{y} - k \\boldsymbol{b} \\leq \\boldsymbol{0}``, the
budget, the weight bounds — so all of them hold at ``\\boldsymbol{y} = \\boldsymbol{0}``,
``k = 0``, and only the normalisation keeps the solver off that point. The return form's
normalisation is an equality on a non-zero right-hand side, so it excludes the ray outright.
The risk form's is an inequality, and it does not: when no feasible portfolio's return
expression can beat ``r_f`` — which a mean uncertainty set of a large enough radius
guarantees — the objective ``\\mathrm{ret} - r_f k`` is non-positive along every ray, its
supremum is zero at the origin, and the solver answers with ``k`` at the size of its own
feasibility tolerance. Each constraint then holds to about that tolerance rather than on its
own terms, and the recovered ``\\boldsymbol{w} = \\boldsymbol{y} / k`` can break the bound the
caller wrote while the result still reports success.

``k \\geq k_{\\min}`` closes the ray, and [`set_maximum_ratio_scale_floor!`](@ref) writes it
on the risk form — the branch that needs it. The floor binds **only** on a model that has no
tangency portfolio to find, because a feasible ray with a positive ratio is scaled by the
normalisation alone; so a `k` that comes back at the floor is the signal that the objective
never rose above zero, and the weights beside it maximise the return expression at that
scale rather than the ratio. `kmin` is caller-visible for that reason, and a `kmin` the
caller names is written on either branch.

The ratio is taken at the **aggregate** level: its numerator is the model's single `ret`
expression, whatever number of terms built it. `rf` is therefore a single rate on that
aggregate, and a term that is not in return units belongs out of the numerator through
`settings.rte = false`, not through a per-term rate.

## The degree of the risk

The quotient above holds for a risk expression of degree one in ``(\\boldsymbol{y}, k)``. The
model builds the risk on ``\\boldsymbol{y}``, so a risk expression of degree ``d`` equals
``k^{d} R(\\boldsymbol{w})``, and both normalisations reach the maximiser of

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f}{R(\\boldsymbol{w})^{1/d}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\boldsymbol{y}``: The homogenised weights, ``\\boldsymbol{y} = k \\boldsymbol{w}``.
  - $(math_dict[:k_budget])
  - ``r_f``: Risk-free rate.
  - $(math_dict[:R_w])
  - $(math_dict[:d_homog]) Here it is the degree of the expression that the model builds.
  - ``\\mathbf{\\Sigma}``: The covariance matrix of the variance.
  - ``\\sigma(\\boldsymbol{w})``: The portfolio's standard deviation, ``\\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}``.
  - ``\\mathbf{W}``: The symmetric lifted matrix of [`set_sdp_constraints!`](@ref).

The return form fixes ``k = \\mathrm{ohf} / (\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f)``
and minimises ``\\mathrm{ohf}^{d} R(\\boldsymbol{w}) / (\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f)^{d}``.
The risk form meets its bound at ``k = (\\mathrm{ohf} / R(\\boldsymbol{w}))^{1/d}`` and maximises
``k (\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f)``. Neither maximiser depends on `ohf`.

The degree belongs to the formulation, not to the measure, and [`Variance`](@ref) has two:

  - A measure of degree one, such as [`StandardDeviation`](@ref) or [`ConditionalValueatRisk`](@ref), gives the quotient of the first equation.
  - [`Variance`](@ref) in the [`SquaredSOCRiskExpr`](@ref) or [`QuadRiskExpr`](@ref) formulation builds ``\\boldsymbol{y}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{y} = k^{2} \\sigma^{2}(\\boldsymbol{w})``, of degree two. The model maximises ``(\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f) / \\sigma(\\boldsymbol{w})``, the Sharpe ratio.
  - [`Variance`](@ref) in the semidefinite formulation builds ``\\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W})`` with ``\\mathbf{W} \\succeq \\boldsymbol{y}\\boldsymbol{y}^\\intercal / k``. The trace is at least ``k \\sigma^{2}(\\boldsymbol{w})``, with equality at ``\\mathbf{W} = \\boldsymbol{y}\\boldsymbol{y}^\\intercal / k``, so it has degree one. The model maximises ``(\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f) / \\sigma^{2}(\\boldsymbol{w})``, the excess return per unit of variance. A variance takes this formulation when it holds risk-contribution rows `rc`, when the constraints hold a [`SemiDefinitePhylogeny`](@ref), and always under [`FactorRiskContribution`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing,
                 kmin::Option{<:Number} = nothing) -> MaximumRatio

Keywords correspond to the struct's fields.

## Validation

  - If `ohf` is provided: `ohf > 0`.
  - If `kmin` is provided: `kmin > 0`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`set_max_ratio_return_constraints!`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
  - [`set_maximum_ratio_scale_floor!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.4, equations 8.13 to 8.16.
  - $(ref_dict[:sharpe1964])
  - $(ref_dict[:schaibleibaraki1983])
  - $(ref_dict[:charnescooper1962])
"""
@concrete struct MaximumRatio <: ObjectiveFunction
    """
    $(field_dict[:rf])
    """
    rf
    """
    $(field_dict[:ohf])
    """
    ohf
    """
    $(field_dict[:kmin])
    """
    kmin
    function MaximumRatio(rf::Number, ohf::Option{<:Number}, kmin::Option{<:Number})
        if !isnothing(ohf)
            @argcheck(ohf > zero(ohf), DomainError(ohf, "ohf must be > 0"))
        end
        if !isnothing(kmin)
            @argcheck(kmin > zero(kmin), DomainError(kmin, "kmin must be > 0"))
        end
        return new{typeof(rf), typeof(ohf), typeof(kmin)}(rf, ohf, kmin)
    end
end
function MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing,
                      kmin::Option{<:Number} = nothing)
    return MaximumRatio(rf, ohf, kmin)
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises the model's return expression.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret}(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\mathrm{ret}(\\boldsymbol{w})``: The model's single scalar return expression, the weighted sum over the return terms.

The objective reads the aggregate expression, not ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``:
a [`LogarithmicReturn`](@ref) term contributes a mean logarithmic return, and several terms
contribute their weighted sum. It states no ceiling on the risk, so a risk measure's
`settings.ub` is the only thing that keeps the solution off the single highest-return asset.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`MaximumElementReturn`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.2, equation 8.10.
  - $(ref_dict[:markowitz1952])
"""
struct MaximumReturn <: ObjectiveFunction end
"""
$(DocStringExtensions.TYPEDEF)

Internal objective that maximises the expression of **one** return term.

Used only by the return-frontier corner solves. With *k* terms the span of term *i* has to be
read off a portfolio that maximised term *i* alone: reading it off the aggregate
maximum-return corner makes it an artefact of the other terms' `scale`, and can leave
`rt_min > rt_max`, so the sweep range descends.

Not part of the user-facing API.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumElementReturn(i::Integer) -> MaximumElementReturn

The argument corresponds to the struct's field. The type takes a positional argument alone,
because it carries one field and no caller outside the frontier builds it.

## Validation

  - `i > 0`. The other half of the domain, `i <= length(ret)`, is checked at model build by
    [`assert_no_return_objective_compatibility`](@ref), which is the first site that sees the
    return terms.

# Related

  - [`MaximumReturn`](@ref)
  - [`compute_ret_lbs`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
"""
@concrete struct MaximumElementReturn <: ObjectiveFunction
    """
    $(field_dict[:i_ret_term])
    """
    i
    function MaximumElementReturn(i::Integer)
        @argcheck(i > zero(i), DomainError(i, "i must be > 0"))
        return new{typeof(i)}(i)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the model's `:ret` expression is identically zero.

The degeneracy guard tests the **state** of the expression, not the **type** of the term.
Two routes reach that state — a [`NoReturn`](@ref) term, and a term taken out of the
sum by `settings.rte = false` — and this predicate is one **fused** test over both.

The fusion is load-bearing. `:ret` is the weighted sum of the terms, so it is zero exactly
when *every* term is out of it, by either route. Testing the routes separately and composing
them is wrong:

```julia
r = [NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false))]
```

Every term here is out of `:ret`, yet `all(isa NoReturn) || all(!rte)` returns `false`. Only
`all(isa NoReturn || !rte)` sees it. The risk axis composes instead, because its two halves
carry different quantifiers; see [`zero_risk_expression_flag`](@ref).

One real term beside a `NoReturn` leaves `:ret` non-zero, so `[ArithmeticReturn(), NoReturn()]` solves correctly under every objective and is not refused. An empty vector is
refused separately by [`set_return_constraints!`](@ref).

# Related

  - [`NoReturn`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
  - [`assert_return_term_required`](@ref)
  - [`zero_risk_expression_flag`](@ref)
"""
function zero_return_expression_flag(r)::Bool
    return if isa(r, AbstractVector)
        !isempty(r) && all(x -> isa(x, NoReturn) || !x.settings.rte, r)
    else
        isa(r, NoReturn) || !r.settings.rte
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that an objective which reads the return expression is given a non-zero one.

Dispatches on the objective. Three of them read `:ret` and are refused when it is degenerate;
every other objective takes the no-op fallback, because a zero `:ret` is legitimate under
[`MinimumRisk`](@ref) and [`MaximumUtility`](@ref).

| Objective                      | Refused when                     |
|:------------------------------ |:-------------------------------- |
| [`MaximumReturn`](@ref)        | every term is out of `:ret`      |
| [`MaximumRatio`](@ref)         | every term is out of `:ret`      |
| [`MaximumElementReturn`](@ref) | term `i` is a [`NoReturn`](@ref) |

The first two use [`zero_return_expression_flag`](@ref), so both routes to a zero expression
are covered. [`MaximumElementReturn`](@ref) is **per index** and ignores
`settings.rte` entirely: it maximises `ret_i` directly, which the builder registers whatever
the flag says, so a `false` flag removes that term from the *sum* without touching the
objective. Only the sentinel type makes `ret_i` itself zero.

[`MaximumElementReturn`](@ref)'s method **range-checks `i` first**, because the `NoReturn`
test indexes `ret[i]`. The refusal is a `DomainError`, matching the constructor's own
`i > zero(i)` guard: the two halves of one domain, spelled alike. The constructor cannot
check the upper half, because it never sees `ret`. A [`TimeDependent`](@ref) schedule is
checked **per fold**, by construction — a callable schedule has no value until its fold
exists, so an eager scan could cover vector schedules only.

Called from [`set_return_constraints!`](@ref), the shared model-build hook every JuMP
optimiser reaches, **not** from a constructor. The hook is the only site that sees all three
objective-carrying optimisers: [`MeanRisk`](@ref), [`FactorRiskContribution`](@ref) — which
carries its own `obj` and had no return-side guard at all — and
[`NearOptimalCentering`](@ref). The cost is that a refusal now arrives at `optimise` time
rather than at construction time.

The seam is also where a [`TimeDependent`](@ref) schedule is already resolved, so neither side
needs a schedule skip here.

# Related

  - [`NoReturn`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_return_term_required`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function assert_no_return_objective_compatibility(ret, ::ObjectiveFunction)::Nothing
    return nothing
end
function assert_no_return_objective_compatibility(ret, ::MaximumReturn)::Nothing
    @argcheck(!zero_return_expression_flag(ret),
              ArgumentError("MaximumReturn needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. The objective would be identically zero, so every feasible portfolio is optimal and the solver would return an arbitrary one while reporting success. Use obj = MinimumRisk() or obj = MaximumUtility(), or give a return term that is in the expression."))
    return nothing
end
function assert_no_return_objective_compatibility(ret, ::MaximumRatio)::Nothing
    @argcheck(!zero_return_expression_flag(ret),
              ArgumentError("MaximumRatio needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. The ratio's homogenisation variable `k` collapses to zero when `rf > 0`, and the problem returns an arbitrary feasible point when `rf = 0`. Use obj = MinimumRisk() or obj = MaximumUtility(), or give a return term that is in the expression."))
    return nothing
end
function assert_no_return_objective_compatibility(ret, obj::MaximumElementReturn)::Nothing
    i = obj.i
    rets = isa(ret, AbstractVector) ? ret : (ret,)
    @argcheck(i <= length(rets),
              DomainError(i,
                          "i must be <= the number of return terms; ret has $(length(rets)) $(length(rets) == 1 ? "term" : "terms")"))
    @argcheck(!isa(rets[i], NoReturn),
              ArgumentError("MaximumElementReturn($i) needs a non-zero return term at index $i, and that term is a `NoReturn`, whose expression is identically zero. The objective would be identically zero, so every feasible portfolio is optimal and the solver would return an arbitrary one while reporting success. Name a different term, or give a real return term at index $i. `settings.rte` is not consulted here: it removes a term from the summed `:ret` expression, while this objective reads `ret_$i` directly."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that `ret` gives a non-zero return expression, for optimisers built around one.

A zero `:ret` is coherent in the three optimisers that never read it, and in
[`MeanRisk`](@ref) under an objective that never reads it. [`NearOptimalCentering`](@ref) is
the one optimiser that neither describes: its logarithmic barrier constrains
`exp(log_ret) <= ret - rt`, so with a zero return expression the model is **infeasible**
rather than degenerate, and without this check the failure arrives as a solver
`OptimisationFailure` naming nothing.

The criterion is [`zero_return_expression_flag`](@ref), so both routes to a zero expression
are covered. This guard stays at the **constructor**, unlike the objective-level
[`assert_no_return_objective_compatibility`](@ref), and the split is principled: this one asks
whether the *formulation* needs a return term, which is knowable from the estimator alone,
while the other asks whether the *objective* does, and objective and term first meet at model
build.

`T` names the calling optimiser, for the error message. [`TimeDependent`](@ref) schedules are
skipped here and reached instead through [`assert_time_dependent_substitution`](@ref), which
re-runs the host's own constructor on each resolved entry.

# Related

  - [`NoReturn`](@ref)
  - [`zero_return_expression_flag`](@ref)
  - [`assert_no_return_objective_compatibility`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assert_return_term_required(ret, T::Symbol)::Nothing
    if isa(ret, TimeDependent) || !zero_return_expression_flag(ret)
        return nothing
    end
    return throw(ArgumentError("$T needs a non-zero return expression, and every return term is out of it: each term is either a `NoReturn` or carries `settings.rte = false`. $T's logarithmic barrier constrains exp(log_ret) <= ret - rt, and with a zero return expression the reference return rt and the model's return expression are both zero, so the constraint reads exp(log_ret) <= 0, which no real log_ret satisfies. The model is infeasible, not merely degenerate. Give a return term that is in the expression. A zero return expression is for the optimisers whose formulation never reads it — RiskBudgeting, RelaxedRiskBudgeting and FactorRiskContribution — and for MeanRisk under MinimumRisk or MaximumUtility."))
end
"""
    set_maximum_ratio_factor_variables!(model, obj)

Register the homogenisation variable `k` for the maximum ratio objective.

This is the head-level producer of `k`. Every head that shapes `w` from an objective calls
it exactly once, so the two spellings of `k` live here rather than at each head: `k >= 0`
under [`MaximumRatio`](@ref), and the literal `1` under every other objective. The heads
whose formulation is fixed pass their own fixed objective ([`MinimumRisk`](@ref)) and take
the second branch. [`RiskBudgeting`](@ref) is the one head that does not come through here
— see [`get_k`](@ref).

This runs **before** the model is assembled, because [`set_weight_constraints!`](@ref) reads
`k` immediately afterwards. The other half of the old bundle — the normalisation factor `ohf`
— is sized from the *resolved* return characteristic, which does not exist until the return
builders have run, so [`set_maximum_ratio_normalisation!`](@ref) registers it later.

The second method takes exactly one objective, not `args...`. A variadic fallback used to
absorb a wrong-arity call silently and register `k = 1` under a `MaximumRatio` objective, so
the ratio branch went untested while the call still looked correct.

# Arguments

  - `model`: JuMP optimisation model.
  - `obj`: Objective function (e.g., [`MaximumRatio`](@ref)).

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`get_k`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
function set_maximum_ratio_factor_variables!(model::JuMP.Model, obj::MaximumRatio)
    JuMP.@variable(model, k >= 0)
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, obj)
    JuMP.@expression(model, k, 1)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Register the ratio problem's normalisation factor `ohf`.

Sized from the resolved aggregate characteristic ``\\sum_{i:\\,\\mathrm{rte}} s_i \\boldsymbol{\\mu}_i`` when there is one, and from the prior's own vector when no term carries
a per-asset quantity. Reading it off the resolved aggregate corrects a pre-existing mismatch:
the factor used to come from `pr.mu` alone, so it ignored a term's own `mu` and a set's
carried centre. The change is numerical, not semantic — any `ohf > 0` recovers the same
`w / k`.

# Arguments

  - $(arg_dict[:model])
  - `obj::MaximumRatio`: The ratio objective.
  - `mu`: The resolved aggregate characteristic, or `nothing`.
  - `pr`: Prior result, the fallback when `mu` is `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_maximum_ratio_factor_variables!`](@ref)
  - [`set_max_ratio_return_constraints!`](@ref)
  - [`set_maximum_ratio_scale_floor!`](@ref)
"""
function set_maximum_ratio_normalisation!(model::JuMP.Model, obj::MaximumRatio,
                                          mu::Option{<:Num_VecNum}, pr::AbstractPriorResult)
    ohf = if isnothing(obj.ohf)
        mu = isnothing(mu) ? pr.mu : mu
        min(1e3, max(1e-3, Statistics.mean(abs, mu)))
    else
        @argcheck(obj.ohf > zero(obj.ohf), DomainError(obj.ohf, "obj.ohf must be > 0"))
        obj.ohf
    end
    JuMP.@expression(model, ohf, ohf)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Close the ratio problem's degenerate ray by tightening `k`'s own lower bound to ``k_{\\min}``.

Every constraint the homogenised model writes is homogeneous in ``(\\boldsymbol{y}, k)``, so
each of them holds at the origin. [`MaximumRatio`](@ref) states what that costs. This closes
that ray.

The floor is written as a **variable bound** — [`set_maximum_ratio_factor_variables!`](@ref)
already declares `k >= 0`, and this tightens that same bound rather than adding a row. A row
would carry a dual and change the interior point the solver lands on even where it cannot
bind, and a `MaximumRatio` answer feeds allocators and stacked optimisers that are pinned far
tighter than that perturbation.

**Only the risk form gets the derived floor**, for the same reason. The return form's
normalisation is an equality on a non-zero right-hand side, so under finite weight bounds it
excludes the ray already and a floor there is inert by construction — and inert is not free.
A `kmin` the caller names is written on either branch, because a caller who asks for a floor
is asking for this bound.

Sized from the resolved aggregate characteristic when `obj.kmin` is `nothing`:

```math
k_{\\min} = \\frac{10^{-4}\\,\\mathrm{ohf}}{\\max\\left(\\mathrm{ohf},\\, \\max_i \\mu_i - r_f\\right)}\\,.
```

``\\mathrm{ohf} / (\\max_i \\mu_i - r_f)`` is the return form's own floor: no long-only fully
invested portfolio earns more than its best asset, so no such model can pin `k` below it. The
risk form's scale is ``\\mathrm{ohf}^{1/d} / R(\\boldsymbol{w})`` for a risk measure
homogeneous of degree ``d``, which that expression does not bound — measured over the risk
measures the library ships, it lands between one eighth and seventeen times it.

The ``10^{-4}`` is the margin, and three measurements fix it. It has to sit **below** the
smallest scale a well-posed model pins, which is `MaximumDrawdown`'s; **above** the
feasibility tolerance the collapsed ray answers on, or the recovered
``\\boldsymbol{w} = \\boldsymbol{y} / k`` still carries a useless residual; and below the
point where a slack bound spoils the conditioning of the widest model that reaches here —
`ExactOrderedWeightsArray` under a [`LogarithmicReturn`](@ref) stops converging at
``10^{-3}`` though its own scale is three orders above the bound. ``10^{-4}`` is the value
that clears all three. It buys the middle one the least: the recovered weights meet a bound
to about ``10^{-5}`` rather than to ``10^{-7}``, against the breach of the bound outright
that the collapsed ray returned.

The denominator's ``\\max`` keeps a universe whose characteristic is smaller than `ohf` from
lifting the floor above ``10^{-4}``.

# Arguments

  - $(arg_dict[:model])
  - `obj::MaximumRatio`: The ratio objective, whose `kmin` overrides the size above.
  - `mu`: The resolved aggregate characteristic, or `nothing`.
  - `pr`: Prior result, the fallback when `mu` is `nothing`.
  - `risk_form::Bool`: Whether the caller registered `sr_risk`. `false` is the return form,
    which takes a floor only when the caller named one.

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
  - [`set_max_ratio_return_constraints!`](@ref)
"""
function set_maximum_ratio_scale_floor!(model::JuMP.Model, obj::MaximumRatio,
                                        mu::Option{<:Num_VecNum}, pr::AbstractPriorResult,
                                        risk_form::Bool)
    if !risk_form && isnothing(obj.kmin)
        return nothing
    end
    ohf = shared_get(model, :ohf)
    kmin = if isnothing(obj.kmin)
        mu = isnothing(mu) ? pr.mu : mu
        1e-4 * ohf / max(ohf, maximum(mu) - obj.rf)
    else
        obj.kmin
    end
    JuMP.set_lower_bound(get_k(model), kmin)
    return nothing
end
"""
    add_to_objective_penalty!(model::JuMP.Model, expr)

Accumulate an expression into the objective penalty term `op` in the JuMP model.

Creates the `op` expression if it does not yet exist, then adds `expr` to it.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `expr`: JuMP expression to add to the penalty.

# Returns

  - `nothing`.

# Related

  - [`add_penalty_to_objective!`](@ref)
  - [`set_portfolio_objective_function!`](@ref)
"""
function add_to_objective_penalty!(model::JuMP.Model, expr)
    op = if !shared_has(model, :op) && isa(expr, JuMP.AffExpr)
        JuMP.@expression(model, op, zero(JuMP.AffExpr))
    elseif !shared_has(model, :op) && isa(expr, JuMP.QuadExpr)
        JuMP.@expression(model, op, zero(JuMP.QuadExpr))
    elseif shared_has(model, :op)
        shared_get(model, :op)
    else
        throw(ArgumentError("expr must be a JuMP.AffExpr or JuMP.QuadExpr"))
    end
    if isa(expr, JuMP.QuadExpr) && !isa(op, JuMP.QuadExpr)
        JuMP.unregister(model, :op)
        op = JuMP.@expression(model, op, JuMP.QuadExpr(op))
    end
    JuMP.add_to_expression!(op, expr)
    return nothing
end
"""
    add_penalty_to_objective!(model::JuMP.Model, factor::Integer, expr)

Add the accumulated objective penalty to the main objective expression.

If an `op` penalty term exists in the model, adds `factor * op` to `expr`. Returns `expr` unchanged if no penalty term has been registered.

A quadratic penalty cannot be accumulated into an affine objective in-place, so an affine `expr` is promoted to a `JuMP.QuadExpr` when `op` is quadratic. Promotion allocates a new expression, which is why the caller must use the returned value rather than the one it passed in.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `factor::Integer`: Sign factor (`1` for minimisation, `-1` for maximisation).
  - `expr`: JuMP objective expression.

# Returns

  - `expr`: The objective expression with the penalty added, promoted to a `JuMP.QuadExpr` if that was needed to hold a quadratic penalty.

# Related

  - [`add_to_objective_penalty!`](@ref)
  - [`set_portfolio_objective_function!`](@ref)
"""
function add_penalty_to_objective!(model::JuMP.Model, factor::Integer, expr)
    if !shared_has(model, :op)
        return expr
    end
    op = shared_get(model, :op)
    if !isa(expr, JuMP.QuadExpr) && isa(op, JuMP.QuadExpr)
        JuMP.unregister(model, :obj_expr)
        expr = JuMP.@expression(model, obj_expr, JuMP.QuadExpr(expr))
    end
    JuMP.add_to_expression!(expr, factor, op)
    return expr
end
"""
    set_portfolio_objective_function!(model, obj, optimiser, attrs)

Set the portfolio objective function in the JuMP model.

Dispatches on the objective function type to build the appropriate JuMP objective expression, then folds in the [Objective Penalty](@ref add_to_objective_penalty!) accumulated by the regularisation, soft-constraint, and custom-term builders.

Custom objective terms are applied *before* the penalty is folded in, because they contribute to the same accumulator: `add_penalty_to_objective!` applies the sign factor matching this method's optimisation sense, so a contribution always worsens the objective regardless of which objective is being built.

The return term is **not** a parameter of this function. It used to be passed positionally so that a [`LogarithmicReturn`](@ref) ratio problem could dispatch to its own method, because the logarithmic builder registered `sr_elog_ret_risk` where every other branch registered `sr_risk`. The hoisted ratio constraint registers one name for every term shape, so the two methods collapse into one keyed on the presence of `:sr_risk`, and the positional lost its only dispatch reason.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `obj::ObjectiveFunction`: Portfolio objective (e.g. [`MinimumRisk`](@ref), [`MaximumUtility`](@ref)). This is the objective *being built*, which during a [`Frontier`](@ref) sweep differs from the one the user declared.
  - `optimiser::JuMPOptimisationEstimator`: The outer optimisation estimator (e.g. the [`MeanRisk`](@ref) itself). Supplies the custom objective terms as `optimiser.opt.cobj`.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle.

# Returns

  - `nothing`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`add_penalty_to_objective!`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function set_portfolio_objective_function!(model::JuMP.Model, obj::MinimumRisk,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    risk = get_risk(model)
    JuMP.@expression(model, obj_expr, risk)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumUtility,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = get_ret(model)
    risk = get_risk(model)
    l = obj.l
    JuMP.@expression(model, obj_expr, ret - l * risk)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    if shared_has(model, :sr_risk)
        ret = get_ret(model)
        k = get_k(model)
        rf = obj.rf
        JuMP.@expression(model, obj_expr, ret - rf * k)
        add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
        obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
        JuMP.@objective(model, Max, so * obj_expr)
    else
        risk = get_risk(model)
        JuMP.@expression(model, obj_expr, risk)
        add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
        obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
        JuMP.@objective(model, Min, so * obj_expr)
    end
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumReturn,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = get_ret(model)
    JuMP.@expression(model, obj_expr, ret)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumElementReturn,
                                           optimiser::JuMPOptimisationEstimator, attrs)
    so = get_objective_scale(model)
    ret = state_get(model, Symbol(""), :ret_, obj.i)
    JuMP.@expression(model, obj_expr, ret)
    add_custom_objective_term!(model, obj, optimiser.opt.cobj, optimiser, attrs)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end

export MinimumRisk, MaximumUtility, MaximumRatio, MaximumReturn
