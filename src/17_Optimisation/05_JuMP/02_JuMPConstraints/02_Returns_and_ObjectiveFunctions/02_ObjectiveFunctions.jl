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
`:ret`. It sets no floor on the return, so without a `settings.lb` on a term the solution can
have a negative expected return.

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
\\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret} - l\\, R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:ret_model]) With one [`ArithmeticReturn`](@ref) term it is ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``.
  - $(math_dict[:w_port])
  - $(math_dict[:l_utility])
  - $(math_dict[:R_w])

The coefficient multiplies the whole risk, with no factor of one half. A caller who ports a
``\\tfrac{\\lambda}{2}`` convention halves their own ``\\lambda``.

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

The ratio reads the aggregate return. Its numerator is the model's one return expression,
whatever number of terms built it, so `rf` is one rate on that expression. A term that is not
in return units leaves the numerator through `settings.rte = false`, not through a rate of its
own.

The model solves the ratio in homogenised weights, and [`set_max_ratio_return_constraints!`](@ref)
picks one of two normalisations for it. In the risk form, every row of the model holds at
``\\boldsymbol{y} = \\boldsymbol{0}``, ``k = 0``, so [`set_maximum_ratio_scale_floor!`](@ref)
bounds ``k`` below by ``k_{\\min}``. The floor binds only when no feasible portfolio's return
expression is more than ``r_f``. A mean uncertainty set of a large radius can cause this, and
so can a fee or a market impact cost that is more than every expected return. A term that
deducts a worst-case penalty or a charge forces the risk form, because the return form has no
solution in these cases. Thus a ``k`` equal to its floor shows that the objective never went
above zero. The weights then maximise the return expression at that scale, not the ratio. The
field `kmin` lets a caller read and set the floor, and a `kmin` that the caller names applies
to both forms.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\mathrm{ret} - r_f}{R(\\boldsymbol{w})}\\,.
\\end{align}
```

The quotient is not concave, so the model solves an equivalent programme in
``\\boldsymbol{y} = k \\boldsymbol{w}``, with ``k \\geq 0``, and recovers
``\\boldsymbol{w} = \\boldsymbol{y} / k``. Every constraint of the model is homogeneous in
``(\\boldsymbol{y}, k)``, and one of two normalisations fixes the scale:

```math
\\begin{align}
\\text{return form:} &\\quad \\underset{\\boldsymbol{y},\\, k}{\\min}\\; R(\\boldsymbol{y}) \\quad \\text{s.t.} \\quad \\mathrm{ret}(\\boldsymbol{y}) - r_f k = \\mathrm{ohf}\\,, \\\\
\\text{risk form:} &\\quad \\underset{\\boldsymbol{y},\\, k}{\\max}\\; \\mathrm{ret}(\\boldsymbol{y}) - r_f k \\quad \\text{s.t.} \\quad R(\\boldsymbol{y}) \\leq \\mathrm{ohf}\\,.
\\end{align}
```

The model builds the risk on ``\\boldsymbol{y}``, so a risk expression of degree ``d`` in
``(\\boldsymbol{y}, k)`` equals ``k^{d} R(\\boldsymbol{w})``. The return form then fixes
``k = \\mathrm{ohf} / (\\mathrm{ret} - r_f)`` and minimises
``\\mathrm{ohf}^{d} R(\\boldsymbol{w}) / (\\mathrm{ret} - r_f)^{d}``. The risk form meets its
bound at ``k = (\\mathrm{ohf} / R(\\boldsymbol{w}))^{1/d}`` and maximises
``k (\\mathrm{ret} - r_f)``. Both forms reach the maximiser of

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\mathrm{ret} - r_f}{R(\\boldsymbol{w})^{1/d}}\\,,
\\end{align}
```

and neither maximiser depends on ``\\mathrm{ohf}``.

Where:

  - $(math_dict[:ret_model]) With one [`ArithmeticReturn`](@ref) term it is ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``.
  - $(math_dict[:w_port])
  - $(math_dict[:y_homog])
  - $(math_dict[:k_budget])
  - $(math_dict[:r_f_ratio])
  - $(math_dict[:R_w])
  - $(math_dict[:ohf_ratio])
  - $(math_dict[:d_homog]) Here it is the degree of the expression that the model builds.
  - $(math_dict[:k_min_ratio])
  - ``\\mathbf{\\Sigma}``: The covariance matrix of the variance.
  - ``\\sigma(\\boldsymbol{w})``: The portfolio's standard deviation, ``\\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}``.
  - $(math_dict[:W_lift])

The degree belongs to the formulation, not to the measure, and [`Variance`](@ref) has two
formulations of different degree:

  - A measure of degree one, such as [`StandardDeviation`](@ref) or [`ConditionalValueatRisk`](@ref), gives the first quotient.
  - [`Variance`](@ref) in the [`SquaredSOCRiskExpr`](@ref) or [`QuadRiskExpr`](@ref) formulation builds ``\\boldsymbol{y}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{y} = k^{2} \\sigma^{2}(\\boldsymbol{w})``, of degree two. The model maximises ``(\\mathrm{ret} - r_f) / \\sigma(\\boldsymbol{w})``, the Sharpe ratio.
  - [`Variance`](@ref) in the semidefinite formulation builds ``\\mathrm{tr}(\\mathbf{\\Sigma} \\mathbf{W})`` with ``\\mathbf{W} \\succeq \\boldsymbol{y}\\boldsymbol{y}^\\intercal / k``. The trace is at least ``k \\sigma^{2}(\\boldsymbol{w})``, with equality at ``\\mathbf{W} = \\boldsymbol{y}\\boldsymbol{y}^\\intercal / k``, so it has degree one. The model maximises ``(\\mathrm{ret} - r_f) / \\sigma^{2}(\\boldsymbol{w})``, the excess return per unit of variance. A variance takes this formulation when it holds risk-contribution rows `rc`, when the constraints hold a [`SemiDefinitePhylogeny`](@ref), and always under [`FactorRiskContribution`](@ref).

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
\\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:ret_model])

The objective reads the aggregate expression, not ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``.
A [`LogarithmicReturn`](@ref) term contributes a mean logarithmic return, and several terms
contribute their weighted sum. The objective sets no ceiling on the risk. Without a
`settings.ub` on a risk measure, the solution puts the most weight that the other
constraints let it put on the assets of the highest return.

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

Internal objective that maximises the expression of one return term.

Only the corner solves of the return frontier use it, and it is not part of the public API.
With several terms, the span of term ``i`` comes from a portfolio that maximised term ``i``
alone. The corner that maximises the aggregate return gives a span that changes with the
`scale` of the other terms. That span can have `rt_min > rt_max`, which makes the sweep range
descend.

# Mathematical definition

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret}_i\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:ret_i_term])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumElementReturn(i::Integer) -> MaximumElementReturn

The argument corresponds to the struct's field. The type takes a positional argument alone,
because it has one field and no caller outside the frontier builds it.

## Validation

  - `i > 0`. [`assert_no_return_objective_compatibility`](@ref) checks the other half of the
    domain, `i <= length(ret)`, when the model is built, because it is the first function
    that sees the return terms.

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

The guard tests the state of the expression, not the type of the term. A term is out of
`:ret` when it is a [`NoReturn`](@ref), or when its `settings.rte` is `false`. The predicate
tests the two conditions in one quantifier.

`:ret` is the weighted sum of the terms, so it is zero exactly when every term is out of it.
Each term can be out for either of the two conditions. Two separate tests joined with `||`
miss a vector that mixes the conditions:

```julia
r = [NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(; rte = false))]
```

Every term here is out of `:ret`, yet `all(isa NoReturn) || all(!rte)` returns `false`. Only
`all(isa NoReturn || !rte)` returns `true`. The risk side composes two tests instead, because
its two halves carry different quantifiers, see [`zero_risk_expression_flag`](@ref).

One real term beside a `NoReturn` leaves `:ret` non-zero, so `[ArithmeticReturn(), NoReturn()]`
solves under every objective, and no guard refuses it. [`set_return_constraints!`](@ref)
refuses an empty vector on its own.

# Mathematical definition

```math
\\begin{align}
\\mathrm{flag} &= \\bigwedge_{i=1}^{n} \\left(\\mathrm{none}_i \\lor \\lnot\\, \\mathrm{rte}_i\\right)\\,.
\\end{align}
```

Where:

  - ``n``: Number of return terms, one for a single term. An empty vector gives `false`.
  - ``\\mathrm{none}_i``: Whether term ``i`` is a [`NoReturn`](@ref).
  - ``\\mathrm{rte}_i``: The `settings.rte` flag of term ``i``.

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

The function dispatches on the objective. Three objectives read `:ret`, and the function
refuses them when `:ret` is zero. Every other objective takes the fallback, which does
nothing, because [`MinimumRisk`](@ref) and [`MaximumUtility`](@ref) accept a zero `:ret`.

| Objective                      | Refused when                     |
|:------------------------------ |:-------------------------------- |
| [`MaximumReturn`](@ref)        | every term is out of `:ret`      |
| [`MaximumRatio`](@ref)         | every term is out of `:ret`      |
| [`MaximumElementReturn`](@ref) | term `i` is a [`NoReturn`](@ref) |

The first two read [`zero_return_expression_flag`](@ref), so they see both ways in which a
term leaves the expression. [`MaximumElementReturn`](@ref) reads one index and ignores
`settings.rte`. It maximises `ret_i`, which the builder registers for each value of the flag.
Thus a `false` flag takes the term out of the sum and does not change the objective. Only a
`NoReturn` makes `ret_i` itself zero.

[`set_return_constraints!`](@ref) calls this function when the model is built, not a
constructor. Each optimiser with an objective calls that function. These optimisers are
[`MeanRisk`](@ref), [`FactorRiskContribution`](@ref) and [`NearOptimalCentering`](@ref). The
cost is that a refusal comes when `optimise` runs, not at construction. A
[`TimeDependent`](@ref) schedule is resolved by then, so the function checks each fold on its
own. A callable schedule has no value until its fold exists. A check at construction can thus
reach only the vector schedules.

# Algorithm

 1. Under [`MaximumReturn`](@ref) or [`MaximumRatio`](@ref), refuse when [`zero_return_expression_flag`](@ref) is `true`.
 2. Under [`MaximumElementReturn`](@ref), check that `i` is at most the number of terms, before the next step indexes the terms.
 3. Under [`MaximumElementReturn`](@ref), refuse when term `i` is a [`NoReturn`](@ref).
 4. Under any other objective, do nothing.

# Validation

  - Under [`MaximumReturn`](@ref) or [`MaximumRatio`](@ref): `!zero_return_expression_flag(ret)`. Otherwise the function raises an `ArgumentError`.
  - Under [`MaximumElementReturn`](@ref): `i <= length(ret)`. Otherwise the function raises a `DomainError`, as the constructor does for `i > 0`, because the two checks bound one domain.
  - Under [`MaximumElementReturn`](@ref): `!isa(ret[i], NoReturn)`. Otherwise the function raises an `ArgumentError`.

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
neither. Its logarithmic barrier constrains `exp(log_ret) <= ret - rt`, so with a zero return
expression the model is infeasible rather than degenerate. Without this check, the failure
arrives as a solver `OptimisationFailure` that names no cause.

The test is [`zero_return_expression_flag`](@ref), so it sees both ways in which a term
leaves the expression. This guard runs in the constructor, and
[`assert_no_return_objective_compatibility`](@ref) runs when the model is built. This guard
asks whether the formulation needs a return term, which the estimator alone answers. The other
asks whether the objective needs one, and the objective and the terms first meet when the
model is built.

`T` names the optimiser that calls the function, for the error message. The function skips a
[`TimeDependent`](@ref) schedule, and [`assert_time_dependent_substitution`](@ref) reaches it
instead, because it runs the host's own constructor again on each resolved entry.

# Validation

  - `isa(ret, TimeDependent) || !zero_return_expression_flag(ret)`. Otherwise the function raises an `ArgumentError`.

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

Each optimiser head that shapes `w` from an objective calls this function one time. Thus the
two forms of `k` are here, and not at each head. A head whose formulation is fixed
passes its own objective, [`MinimumRisk`](@ref), and takes the second method.
[`RiskBudgeting`](@ref) is the one head that does not call it, see [`get_k`](@ref).

The function runs before the rest of the model, because [`set_weight_constraints!`](@ref)
reads `k` next. The normalisation factor `ohf` depends on the resolved return characteristic,
which exists only after the return builders run, so [`set_maximum_ratio_normalisation!`](@ref)
registers it later.

The second method takes exactly one objective, not `args...`, so a call with the wrong number
of arguments fails. A variadic method accepts such a call and registers `k = 1` under a
[`MaximumRatio`](@ref) objective, which is wrong.

# JuMP formulation

## Variables

  - `k`: ``k \\geq 0``, created under [`MaximumRatio`](@ref).

## Expressions

  - `k`: the constant ``1``, registered under every other objective.

Where:

  - $(math_dict[:k_budget])

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

The function sizes the factor from the resolved aggregate characteristic when it exists. When
no term carries a per-asset quantity, it uses the prior's own vector. Thus the factor follows
a term's own `mu` and the centre that a set carries. The size is a matter of
numerics alone, because every ``\\mathrm{ohf} > 0`` gives the same weights.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ohf} &= \\min\\left(10^{3},\\, \\max\\left(10^{-3},\\, \\frac{1}{N} \\sum_{j=1}^{N} \\lvert \\bar{\\mu}_j \\rvert\\right)\\right)\\,, \\\\
\\bar{\\boldsymbol{\\mu}} &= \\sum_{i \\,:\\, \\mathrm{rte}_i} s_i\\, \\boldsymbol{\\mu}_i\\,.
\\end{align}
```

Where:

  - $(math_dict[:ohf_ratio]) A stated `obj.ohf` replaces the formula.
  - ``\\bar{\\boldsymbol{\\mu}}``: Aggregate characteristic, the expected returns of the prior when no term in the expression carries one.
  - ``\\boldsymbol{\\mu}_i``: Resolved characteristic of return term ``i``.
  - $(math_dict[:s_i_ret])
  - ``\\mathrm{rte}_i``: The `rte` flag of term ``i``.
  - $(math_dict[:N])

# JuMP formulation

## Expressions

  - `ohf`: ``\\mathrm{ohf}``.

Where:

  - $(math_dict[:ohf_ratio])

# Arguments

  - $(arg_dict[:model])
  - `obj::MaximumRatio`: The ratio objective.
  - `mu`: The resolved aggregate characteristic, or `nothing`.
  - `pr`: Prior result, the fallback when `mu` is `nothing`.

# Validation

  - If `obj.ohf` is provided: `obj.ohf > 0`. Otherwise the function raises a `DomainError`.

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

Raise `k`'s own lower bound to ``k_{\\min}``, which closes the degenerate ray of the ratio problem.

Every row of the homogenised model other than the normalisation is homogeneous in
``(\\boldsymbol{y}, k)``, so it holds at the origin. The risk form's normalisation holds there
too, and [`MaximumRatio`](@ref) states the cost of that. The floor closes the ray to the
origin.

The floor is a variable bound, not a row. [`set_maximum_ratio_factor_variables!`](@ref)
declares `k >= 0`, and this function raises that bound. A row carries a dual, and it moves the
interior point where the solver stops, also where the row cannot bind. Allocators and stacked
optimisers read a `MaximumRatio` answer to a tolerance that is much smaller than that move.

The derived floor applies to the risk form alone, for the same reason. The return form's
normalisation is an equality with a non-zero right-hand side. Under finite weight bounds, it
excludes the ray already. A floor there cannot bind, but it still moves the answer. A `kmin`
that the caller names applies to both forms, because the caller asks for this bound.

``\\mathrm{ohf} / (\\max_i \\bar{\\mu}_i - r_f)`` is the return form's own floor. No long-only
fully invested portfolio earns more than its best asset, so no such model can fix ``k`` below
it. The risk form's scale is ``\\mathrm{ohf}^{1/d} / R(\\boldsymbol{w})`` for a risk measure of
degree ``d``, and that expression does not bound it.

The factor ``10^{-4}`` is a margin with three limits. It must be below the smallest scale that
a well-posed model fixes, which is the scale of `MaximumDrawdown`. It must be above the
feasibility tolerance at which the solver stops on the ray. If not, the recovered weights
``\\boldsymbol{w} = \\boldsymbol{y} / k`` keep a residual of that size. It must be below the
point where a slack bound makes the condition number of the widest model too large. That
model is `ExactOrderedWeightsArray` under a [`LogarithmicReturn`](@ref). With the floor, the
recovered weights meet a bound to about ``10^{-5}``. Without it, the ray can return weights
that break the bound and still report success.

# Mathematical definition

```math
\\begin{align}
k_{\\min} &= \\frac{10^{-4}\\,\\mathrm{ohf}}{\\max\\left(\\mathrm{ohf},\\, \\max_i \\bar{\\mu}_i - r_f\\right)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:k_min_ratio]) A stated `obj.kmin` replaces the formula.
  - $(math_dict[:ohf_ratio])
  - ``\\bar{\\mu}_i``: Entry ``i`` of the aggregate characteristic, the expected returns of the prior when no term in the expression carries one.
  - $(math_dict[:r_f_ratio])
  - $(math_dict[:R_w])
  - $(math_dict[:d_homog])

The outer ``\\max`` in the denominator is at least ``\\mathrm{ohf}``, so
``k_{\\min} \\leq 10^{-4}``. A universe whose characteristic is smaller than ``\\mathrm{ohf}``
thus cannot raise the floor above ``10^{-4}``.

# JuMP formulation

## Variables

  - `k`: read from the model. Its lower bound becomes ``k_{\\min}``.

Where:

  - $(math_dict[:k_budget])
  - $(math_dict[:k_min_ratio])

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

The regularisation, soft-constraint and custom-term builders call it, and
[`add_penalty_to_objective!`](@ref) adds the sum to the objective.

# Algorithm

 1. When the model holds no `op`, register `op` as a zero. Its type is the type of `expr`, a `JuMP.AffExpr` or a `JuMP.QuadExpr`.
 2. When `expr` is quadratic and `op` is affine, register `op` again as a `JuMP.QuadExpr` with the same value.
 3. Add `expr` to `op` in place.

# JuMP formulation

## Expressions

  - `op`: ``\\mathrm{op} + e``.

Where:

  - $(math_dict[:op_penalty])
  - ``e``: The expression `expr`.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `expr`: JuMP expression to add to the penalty.

# Validation

  - When the model holds no `op`: `expr` is a `JuMP.AffExpr` or a `JuMP.QuadExpr`. Otherwise the function raises an `ArgumentError`.

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

An affine expression cannot hold a quadratic penalty in place, so an affine `expr` becomes a `JuMP.QuadExpr` when `op` is quadratic. That makes a new expression, so the caller must use the returned value and not the one it passed in.

# Algorithm

 1. When the model holds no `op`, return `expr` unchanged.
 2. When `op` is quadratic and `expr` is not, register `obj_expr` again as a `JuMP.QuadExpr` that holds `expr`.
 3. Add `factor * op` to `expr` in place, and return it.

# JuMP formulation

## Expressions

  - `obj_expr`: ``e + f\\, \\mathrm{op}``, registered again only in step 2.

Where:

  - ``e``: The objective expression `expr`.
  - ``f``: The sign `factor`.
  - $(math_dict[:op_penalty])

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

The function dispatches on the objective, builds its expression, and adds the [Objective Penalty](@ref add_to_objective_penalty!) that the regularisation, soft-constraint and custom-term builders accumulated.

The custom objective terms go in before the penalty, because they add to the same accumulator. [`add_penalty_to_objective!`](@ref) applies the sign that matches the sense of the objective, so a contribution always worsens the objective, whichever objective the method builds.

The return term is not an argument. Under [`MaximumRatio`](@ref) the method reads the form from the model instead. A registered `:sr_risk` means the risk form, and no `:sr_risk` means the return form.

# Algorithm

 1. Read the objective scale `so`, and the entries that the objective reads.
 2. Register `obj_expr`, the expression of the objective.
 3. Add the custom objective terms of `optimiser.opt.cobj` to the penalty, with [`add_custom_objective_term!`](@ref).
 4. Add the penalty to `obj_expr` with [`add_penalty_to_objective!`](@ref). Use the sign `1` for a minimised objective and `-1` for a maximised one.
 5. Set the objective to `so * obj_expr` in the sense of the objective.

# JuMP formulation

## Expressions

  - `obj_expr`: the expression of the objective, before step 4 adds the penalty to it.

      + ``R(\\boldsymbol{w})`` under [`MinimumRisk`](@ref), and under the return form of [`MaximumRatio`](@ref).
      + ``\\mathrm{ret} - l\\, R(\\boldsymbol{w})`` under [`MaximumUtility`](@ref).
      + ``\\mathrm{ret} - r_f k`` under the risk form of [`MaximumRatio`](@ref).
      + ``\\mathrm{ret}`` under [`MaximumReturn`](@ref).
      + ``\\mathrm{ret}_i`` under [`MaximumElementReturn`](@ref).

## Objective

  - `Min`: ``s_o \\left(R(\\boldsymbol{w}) + \\mathrm{op}\\right)`` under [`MinimumRisk`](@ref) and under the return form of [`MaximumRatio`](@ref).
  - `Max`: ``s_o \\left(\\mathrm{obj\\_expr} - \\mathrm{op}\\right)`` under every other objective.

Where:

  - $(math_dict[:R_w]) It is the model's `:risk`.
  - $(math_dict[:ret_model])
  - $(math_dict[:ret_i_term])
  - $(math_dict[:l_utility])
  - $(math_dict[:r_f_ratio])
  - $(math_dict[:k_budget])
  - $(math_dict[:so_scale])
  - $(math_dict[:op_penalty])
  - ``\\mathrm{obj\\_expr}``: The expression `obj_expr` before step 4.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `obj::ObjectiveFunction`: Portfolio objective, such as [`MinimumRisk`](@ref) or [`MaximumUtility`](@ref). It is the objective that the method builds, which in a [`Frontier`](@ref) sweep is not the one the user declared.
  - `optimiser::JuMPOptimisationEstimator`: The outer optimisation estimator, such as the [`MeanRisk`](@ref) itself. Its `optimiser.opt.cobj` holds the custom objective terms.
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
