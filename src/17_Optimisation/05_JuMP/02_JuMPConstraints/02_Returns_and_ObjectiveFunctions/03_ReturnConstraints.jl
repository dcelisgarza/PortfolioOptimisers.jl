"""
    set_return_bounds!(model, i, ret_expr, lb)

Bound the expression of return term `i` from below.

The `Nothing` method does nothing. A number adds one row. A [`Frontier`](@ref) or a vector
adds the term to the `:ret_frontier` Model State entry for a later sweep, as
[`set_risk_upper_bound!`](@ref) does for a risk measure.

The bound binds on the term's own expression, net of the term's own flagged charges and
before `settings.scale` applies. It binds whether `settings.rte` is `true` or `false`, so a
term can constrain the portfolio and stay out of the objective.

# JuMP formulation

## Variables

  - `k`: read from the model.

## Expressions

  - `ret_frontier`: a vector of pairs `(ret_lb_var_i, ret_lb_i) => (ret_i, lb, i)`, registered by the first term with a [`Frontier`](@ref) or a vector bound. A later term adds its pair to it.

## Constraints

  - `ret_lb_i`: ``s_c \\left(\\mathrm{ret}_i - \\mathrm{lb}_i\\, k\\right) \\geq 0``, for a number bound.

Where:

  - $(math_dict[:k_budget])
  - $(math_dict[:ret_i_term])
  - ``\\mathrm{lb}_i``: The number bound of term ``i``.
  - $(math_dict[:sc_scale])

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term.
  - `ret_expr`: The term's own JuMP return expression.
  - `lb`: Lower bound on the term (scalar, vector, or `Frontier`).

# Returns

  - `nothing`.

# Related

  - [`set_return_constraints!`](@ref)
  - [`JuMPReturnsSettings`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function set_return_bounds!(::JuMP.Model, ::Any, ::Any, ::Nothing)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, i, ret_expr, lb::Number)
    sc = get_constraint_scale(model)
    k = get_k(model)
    state_set!(model, Symbol(""), :ret_lb_, i,
               JuMP.@constraint(model, sc * (ret_expr - lb * k) >= 0))
    return nothing
end
function set_return_bounds!(model::JuMP.Model, i, ret_expr, lb::Front_NumVec)
    bound_key = state_key(Symbol(""), :ret_lb_, i)
    bound_var_key = state_key(Symbol(""), :ret_lb_var_, i)
    if !shared_has(model, :ret_frontier)
        JuMP.@expression(model, ret_frontier,
                         Pair{Tuple{Symbol, Symbol},
                              Tuple{<:JuMP.AbstractJuMPScalar, <:Front_NumVec, <:Integer}}[(bound_var_key, bound_key) => (ret_expr,
                                                                                                                          lb,
                                                                                                                          i)])
    else
        push!(shared_get(model, :ret_frontier),
              (bound_var_key, bound_key) => (ret_expr, lb, i))
    end
    return nothing
end
"""
    set_return_expression!(model, i, ret_expr, scale, rte)

Push the scaled expression of return term `i` onto the `:ret_vec` Model State entry.

If `rte` is `false`, the function does nothing, so the term adds nothing to the model's
return expression, and its own bound still binds. It is the return-side twin of
[`set_risk_expression!`](@ref).

# JuMP formulation

## Expressions

  - `ret_vec`: a vector of expressions, registered empty by the first term that enters it. Each term with `rte = true` adds ``s_i\\, \\mathrm{ret}_i``.

Where:

  - $(math_dict[:s_i_ret])
  - $(math_dict[:ret_i_term])

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term.
  - `ret_expr`: The term's own JuMP return expression.
  - `scale::Number`: The term's weight in the sum.
  - `rte::Bool`: When `false` this method is a no-op.

# Returns

  - `nothing`.

# Related

  - [`scalarise_return_expression!`](@ref)
  - [`set_return_bounds!`](@ref)
"""
function set_return_expression!(model::JuMP.Model, i, ret_expr, scale::Number, rte::Bool)
    if !rte
        return nothing
    end
    if !shared_has(model, :ret_vec)
        JuMP.@expression(model, ret_vec, JuMP.AffExpr[])
    end
    push!(shared_get(model, :ret_vec), scale * ret_expr)
    return nothing
end
"""
    scalarise_return_expression!(model)

Collapse the `:ret_vec` entries into the model's single scalar `:ret` expression.

The return side has no scalariser, only the weighted sum. The library's scalarisers follow the
`scalarize` transforms of cvxpy. Their `max` and `log_sum_exp` ignore the sense of the
objective, so they fail on a concave expression that the objective maximises. cvxpy has no
`min`. A stored `-ret` gives them the right sense. But the objective, the bounds, the ratio and
[`NearOptimalCentering`](@ref) all read the one name `:ret`, and each of them then reads the
wrong sign.

When every term has `settings.rte = false`, `:ret_vec` is absent and `:ret` is zero, with no
error here. The objective makes the refusal. [`MinimumRisk`](@ref) and
[`MaximumUtility`](@ref) accept a zero `:ret`, and
[`assert_no_return_objective_compatibility`](@ref) refuses [`MaximumReturn`](@ref) and
[`MaximumRatio`](@ref) before this function runs.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ret} &= \\sum_{i \\,:\\, \\mathrm{rte}_i} s_i\\, \\mathrm{ret}_i\\,.
\\end{align}
```

Where:

  - $(math_dict[:ret_model])
  - $(math_dict[:s_i_ret])
  - $(math_dict[:ret_i_term])
  - ``\\mathrm{rte}_i``: The `rte` flag of term ``i``.

# JuMP formulation

## Expressions

  - `ret`: the sum of the entries of `ret_vec`, zero when the model holds no `ret_vec`.

# Arguments

  - $(arg_dict[:model])

# Returns

  - `nothing`.

# Related

  - [`set_return_expression!`](@ref)
  - [`set_return_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
"""
function scalarise_return_expression!(model::JuMP.Model)
    JuMP.@expression(model, ret, zero(JuMP.AffExpr))
    if !shared_has(model, :ret_vec)
        return nothing
    end
    for ret_i in shared_get(model, :ret_vec)
        JuMP.add_to_expression!(ret, ret_i)
    end
    return nothing
end
"""
    set_max_ratio_return_constraints!(model, obj, rets, mus, forces_risk, pr)

Add the maximum-ratio homogenisation constraint to the model.

The constraint runs one time for the whole model, not in each term's builder. It reads the
one `:ret` of the model and registers the names `sr_ret` and `sr_risk`. A copy for each term
registers one name several times, and each copy reads the wrong expression.

The test on the aggregate characteristic is the single-term test applied to the sum. A test on
each term sends two terms at ``0.9 r_f`` and `scale = 1` to the risk form, but their sum is
``1.8 r_f``.

The characteristic does not carry a worst-case penalty or a charge. A term that deducts one of
them can leave no feasible portfolio above ``r_f``, although an entry of `mu` is more than
``r_f``. The return form then has no solution, and the solver reports the model infeasible. The
risk form has a solution in each of these cases. Thus a term that deducts a penalty or a charge
forces the risk form.

The test does not see the weight bounds or the linear constraints. A bound that keeps every
feasible portfolio at or below ``r_f`` still takes the return form when an entry of `mu` is
more than ``r_f``.

[`assert_no_return_objective_compatibility`](@ref) refuses an empty numerator before this
function runs, as [`NoRisk`](@ref) is refused under this objective. With every term out of
`:ret`, `k` falls to zero when ``r_f > 0``, and the problem returns an arbitrary feasible point
when ``r_f = 0``.

# Algorithm

 1. Sum the characteristics of the terms in the expression with [`aggregate_return_characteristic`](@ref). The sum is `mu`.

 2. Register `ohf` with [`set_maximum_ratio_normalisation!`](@ref).

 3. Take the risk form if one of these conditions is true, and the return form if not:

      + A term forces it. A term with a mean uncertainty set forces it, and so does a term that deducts a fee or a market impact cost. A norm ball whose map has no column and no charge does not.
      + `mu` is `nothing`, because no term carries a characteristic, as for a [`LogarithmicReturn`](@ref).
      + No entry of `mu` is more than `rf`.

 4. Register `sr_risk` in the risk form, or `sr_ret` in the return form.

 5. Bound `k` below with [`set_maximum_ratio_scale_floor!`](@ref).

# JuMP formulation

## Variables

  - `k`: read from the model. Step 5 can raise its lower bound.

## Constraints

  - `sr_risk`: ``s_c \\left(R(\\boldsymbol{y}) - \\mathrm{ohf}\\right) \\leq 0``, in the risk form.
  - `sr_ret`: ``s_c \\left(\\mathrm{ret} - r_f k - \\mathrm{ohf}\\right) = 0``, in the return form.

Where:

  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])
  - $(math_dict[:R_w]) It is the model's `:risk`, built on the homogenised weights.
  - $(math_dict[:y_homog])
  - $(math_dict[:ohf_ratio])
  - $(math_dict[:ret_model])
  - $(math_dict[:r_f_ratio])

# Arguments

  - $(arg_dict[:model])
  - `obj`: Objective function. The function does nothing unless it is a [`MaximumRatio`](@ref).
  - `rets`: The return terms.
  - `mus`: Each term's resolved characteristic, `nothing` where it has none.
  - `forces_risk`: Whether each term forces the risk form.
  - `pr`: Prior result. Its vector sizes `ohf` when no term carries a characteristic.

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
  - [`set_maximum_ratio_scale_floor!`](@ref)
"""
function set_max_ratio_return_constraints!(::JuMP.Model, ::ObjectiveFunction, args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio, rets,
                                           mus::AbstractVector, forces_risk::AbstractVector,
                                           pr::AbstractPriorResult)
    # The empty-numerator refusal is not here: it is one of the three objective refusals
    # that `assert_no_return_objective_compatibility` makes at the top of
    # `set_return_constraints!`.
    mu = aggregate_return_characteristic(rets, mus)
    set_maximum_ratio_normalisation!(model, obj, mu, pr)
    sc = get_constraint_scale(model)
    k = get_k(model)
    ohf = shared_get(model, :ohf)
    ret = get_ret(model)
    rf = obj.rf
    risk_form = any(forces_risk) || isnothing(mu) || all(x -> x <= rf, mu)
    if risk_form
        risk = get_risk(model)
        JuMP.@constraint(model, sr_risk, sc * (risk - ohf) <= 0)
    else
        JuMP.@constraint(model, sr_ret, sc * (ret - rf * k - ohf) == 0)
    end
    set_maximum_ratio_scale_floor!(model, obj, mu, pr, risk_form)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sum the characteristics of the terms that are in the return expression, each at its own scale.

The function returns `nothing` when no term in the expression carries a per-asset quantity,
as in a problem with [`LogarithmicReturn`](@ref) terms alone. It skips a term whose
`settings.rte` is `false`. That term adds nothing to `:ret`, so it must add nothing to the
aggregate that the tests of the ratio read.

# Mathematical definition

```math
\\begin{align}
\\bar{\\boldsymbol{\\mu}} &= \\sum_{i \\,:\\, \\mathrm{rte}_i,\\, \\boldsymbol{\\mu}_i \\neq \\varnothing} s_i\\, \\boldsymbol{\\mu}_i\\,.
\\end{align}
```

Where:

  - ``\\bar{\\boldsymbol{\\mu}}``: Aggregate characteristic, `nothing` when the sum has no term.
  - $(math_dict[:s_i_ret])
  - ``\\boldsymbol{\\mu}_i``: Resolved characteristic of return term ``i``, a vector or a number, and ``\\varnothing`` when the term carries none.
  - ``\\mathrm{rte}_i``: The `rte` flag of term ``i``.

# Related

  - [`set_max_ratio_return_constraints!`](@ref)
  - [`set_maximum_ratio_normalisation!`](@ref)
"""
function aggregate_return_characteristic(rets, mus::AbstractVector)
    mu = nothing
    for (r, mu_i) in zip(rets, mus)
        if !r.settings.rte || isnothing(mu_i)
            continue
        end
        term = r.settings.scale * mu_i
        mu = isnothing(mu) ? term : mu .+ term
    end
    return mu
end
"""
    add_fees_to_ret!(model, ret, fee::Bool)

Subtract the fees expression from one term's return expression.

The function does nothing when the term's `settings.fee` is `false`, or when the model holds
no fees. It returns whether it deducted a fee, because a charged term forces the risk form of
[`MaximumRatio`](@ref).

The model holds two fee expressions. `:fees` holds the per period terms `l`, `s` and `tn`, and
the return pays it in full. `:one_time_fees` holds the two fixed terms. The period of the
investment pays them one time, so the return pays them divided by `:T`, the observation count
of the fit. An expected return is a per period number. Thus this function always spreads the
one-off cost, for each clock that the fee's `horizon` can name for a realised series.

The builder of each term charges the fees. Thus with several terms, the fees enter the return
expression ``\\sum_{i \\,:\\, \\mathrm{fee}_i} s_i`` times. The library does not constrain that
sum. Two terms at `scale = 0.5` charge the fees one time, and two terms at `scale = 1` charge
them two times.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ret}_i &\\leftarrow \\mathrm{ret}_i - f_r - \\frac{f_o}{T}\\,.
\\end{align}
```

Where:

  - $(math_dict[:ret_i_term])
  - $(math_dict[:f_r_fee]) It is zero when the model holds no `:fees`.
  - $(math_dict[:f_o_fee]) It is zero when the model holds no `:one_time_fees`.
  - $(math_dict[:T])
  - ``\\mathrm{fee}_i``: The `fee` flag of term ``i``.
  - $(math_dict[:s_i_ret])

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `fee::Bool`: The term's `settings.fee`.

# Returns

  - `true` when the function deducted `:fees` or `:one_time_fees`, and `false` otherwise.

# Related

  - [`add_market_impact_cost!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_fees_to_ret!(model::JuMP.Model, ret, fee::Bool)
    if !fee
        return false
    end
    charged = false
    if shared_has(model, :fees)
        JuMP.add_to_expression!(ret, -shared_get(model, :fees))
        charged = true
    end
    # An expected return is a per period number, so a fee charged one time for the whole
    # holding period enters it divided by the observation count of the fit.
    if shared_has(model, :one_time_fees)
        JuMP.add_to_expression!(ret, -shared_get(model, :one_time_fees) / get_T(model))
        charged = true
    end
    return charged
end
"""
    add_market_impact_cost!(model, ret, mic::Bool)

Subtract the market impact cost from one term's return expression.

The function does nothing when the term's `settings.mic` is `false`, or when the model holds
no market impact cost. Only [`BudgetMarketImpact`](@ref) registers one, and the function
detects it by its `:wip` entry. A plain [`BudgetCosts`](@ref) also registers
`cost_bgt_expr`, but its cost only constrains the budget and never reaches the return
expression. The function returns whether it deducted the cost, because a charged term forces
the risk form of [`MaximumRatio`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{ret}_i &\\leftarrow \\mathrm{ret}_i - c(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:ret_i_term])
  - ``c(\\boldsymbol{w})``: Market impact cost, the model's `:cost_bgt_expr`.
  - $(math_dict[:w_port])

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `mic::Bool`: The term's `settings.mic`.

# Returns

  - `true` when the function deducted the market impact cost, and `false` otherwise.

# Related

  - [`add_fees_to_ret!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_market_impact_cost!(model::JuMP.Model, ret, mic::Bool)
    if !mic || !shared_has(model, :wip)
        return false
    end
    JuMP.add_to_expression!(ret, -shared_get(model, :cost_bgt_expr))
    return true
end
"""
    set_return_constraints!(model, pret, obj, pr; kwargs...)
    set_return_constraints!(model, i, pret, pr; kwargs...)

Build the model's return expression and the constraints that go with it.

Every JuMP optimiser calls a four-argument method. It runs the builder of each return term,
sums the results into the one `:ret` expression, and then adds the maximum-ratio constraint.
The five-argument methods are the builders of one term. They dispatch on the type of the term
and on the shape of its uncertainty set. Each builder does these operations:

  - It registers names that end in the index of the term, such as `ret_1` and `t_l1ucs_2`.
  - It charges the flagged costs of the term.
  - It bounds the term.
  - It adds the scaled expression of the term to `:ret_vec`.

# Algorithm

 1. Refuse an objective that reads a zero return expression, with [`assert_no_return_objective_compatibility`](@ref). A vector of terms must not be empty.
 2. For a single term, drop its `scale` with [`unit_scale_returns_estimator`](@ref).
 3. Run the builder of each term `i`. The builder returns the characteristic `mus[i]` and the flag `forces_risk[i]`.
 4. Sum the terms into `ret` with [`scalarise_return_expression!`](@ref).
 5. Add the ratio constraint with [`set_max_ratio_return_constraints!`](@ref), which does nothing unless the objective is a [`MaximumRatio`](@ref).

The builder of an [`ArithmeticReturn`](@ref) with no set registers ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``, with the term's own `mu` or else the prior's. The builder of an [`ArithmeticReturn`](@ref) with a set fits the set with [`mu_ucs`](@ref) and calls [`set_ucs_return_constraints!`](@ref). The builder of a [`LogarithmicReturn`](@ref) raises one exponential cone for each observation. The builder of a [`NoReturn`](@ref) registers a zero and charges nothing. Each other builder charges the flagged costs of the term with [`add_fees_to_ret!`](@ref) and [`add_market_impact_cost!`](@ref). Then each builder calls [`set_return_bounds!`](@ref) and [`set_return_expression!`](@ref).

# JuMP formulation

## Variables

  - `w`, `k`: read from the model.
  - `t_elog_ret_i`: ``q_t`` for ``t = 1, \\dots, T``, created by the [`LogarithmicReturn`](@ref) builder.

## Expressions

  - `ret_i`: ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}`` for an [`ArithmeticReturn`](@ref) with no set, ``\\left(\\sum_{t} w_{t} q_t\\right) / \\sum_{t} w_{t}`` for a [`LogarithmicReturn`](@ref), and ``0`` for a [`NoReturn`](@ref). [`set_ucs_return_constraints!`](@ref) registers it for a term with a set.
  - `kret_i`: ``k + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` for ``t = 1, \\dots, T``, for a [`LogarithmicReturn`](@ref).

## Constraints

  - `elog_ret_ret_i`: ``\\left(s_c q_t,\\; s_c k,\\; s_c (k + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})\\right) \\in \\mathcal{K}_{\\exp}`` for ``t = 1, \\dots, T``, for a [`LogarithmicReturn`](@ref).

Where:

  - $(math_dict[:w_port]) Under [`MaximumRatio`](@ref) it is the model's weight variable, ``k`` times the portfolio weights.
  - $(math_dict[:k_budget])
  - ``q_t``: Epigraph variable of observation ``t``.
  - $(math_dict[:T])
  - $(math_dict[:mu_er])
  - $(math_dict[:w_t_obs]) Every ``w_{t}`` is ``1`` when the term and the prior carry no weights.
  - $(math_dict[:x_t_obs])
  - $(math_dict[:sc_scale])
  - ``\\mathcal{K}_{\\exp} = \\{(x, y, z) : y e^{x / y} \\leq z,\\ y > 0\\}``: Exponential cone, with its closure.

## Relaxation

$(val_dict[:relax])

  - The exponential cone gives ``q_t \\leq k \\ln\\left(1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w} / k\\right)``, so `ret_i` of a [`LogarithmicReturn`](@ref) lies at or below ``k`` times the mean logarithmic return of ``\\boldsymbol{w} / k``. At ``k = 1`` that is the mean logarithmic return of ``\\boldsymbol{w}``.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when its mean logarithmic return meets it.

# Arguments

  - $(arg_dict[:model])
  - `pret`: One return term, or a vector of them.
  - `i`: Index of the return term (per-term builders).
  - `obj::ObjectiveFunction`: Portfolio objective function.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `kwargs...`: Additional keyword arguments (e.g. `rd` for uncertainty sets).

# Returns

  - The four-argument methods return `nothing`. A per-term builder returns
    `(mu, forces_risk)`: the characteristic it resolved (or `nothing`), and whether the term
    forces the risk form of [`MaximumRatio`](@ref). A term with a mean uncertainty set or a
    deducted charge forces it.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`NoReturn`](@ref)
  - [`scalarise_return_expression!`](@ref)
  - [`set_return_bounds!`](@ref)
  - [`add_fees_to_ret!`](@ref)
"""
function set_return_constraints!(model::JuMP.Model, pret::JuMPReturnsEstimator,
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    # `scale` is a combination weight, so it is dropped here: one term is not a combination
    # and the weight has nothing to weigh. The vector method below keeps every element's
    # weight, because there the terms really do combine.
    #
    # `pret` is rebound before *both* uses on purpose. The second use feeds
    # `aggregate_return_characteristic`, which applies `settings.scale` to `mu_i` in its own
    # right. Dropping the weight at the first call alone would leave `MaximumRatio`'s
    # normalisation scaled while `:ret` is not — worse than not dropping it at all.
    assert_no_return_objective_compatibility(pret, obj)
    pret = unit_scale_returns_estimator(pret)
    mu, forces_risk = set_return_constraints!(model, 1, pret, pr; kwargs...)
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, (pret,), [mu], [forces_risk], pr)
    return nothing
end
function set_return_constraints!(model::JuMP.Model, pret::VecJRE, obj::ObjectiveFunction,
                                 pr::AbstractPriorResult; kwargs...)
    @argcheck(!isempty(pret), IsEmptyError("`ret` cannot be an empty vector"))
    assert_no_return_objective_compatibility(pret, obj)
    # A term's resolved characteristic is what its own builder returns, and the three shapes
    # are the whole domain: a per-asset vector, the scalar `dot_scalar` folds against `w`,
    # and `nothing` from a term that holds no characteristic at all — a `LogarithmicReturn`
    # or a `NoReturn`. `aggregate_return_characteristic` reads all three.
    mus = Vector{Option{Num_VecNum}}(undef, length(pret))
    forces_risk = Vector{Bool}(undef, length(pret))
    for (i, pret_i) in enumerate(pret)
        mus[i], forces_risk[i] = set_return_constraints!(model, i, pret_i, pr; kwargs...)
    end
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, pret, mus, forces_risk, pr)
    return nothing
end
function set_return_constraints!(model::JuMP.Model, i,
                                 pret::ArithmeticReturn{<:Any, Nothing, <:Any},
                                 pr::AbstractPriorResult; kwargs...)
    w = get_w(model)
    settings = pret.settings
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w)))
    # A charge is not in the characteristic, so the ratio's test on it cannot see the charge.
    # The charged term takes the risk form, which has a solution for each charge (#1358).
    fee = add_fees_to_ret!(model, ret, settings.fee)
    mic = add_market_impact_cost!(model, ret, settings.mic)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, fee || mic
end
"""
    set_ucs_return_constraints!(model, i, ucs::BoxUncertaintySet, mu, settings)

Build one term's box-robust return expression.

The five methods of this function build the worst case of the five mean uncertainty sets. Each
set forces the risk form of [`MaximumRatio`](@ref), because the characteristic that the ratio
tests does not carry the worst-case penalty. A norm ball whose map has no column has no penalty,
and it forces the risk form only when the term deducts a charge.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\boldsymbol{\\Delta}^\\intercal \\lvert \\boldsymbol{w} \\rvert\\,, \\\\
\\boldsymbol{\\Delta} &= \\frac{\\boldsymbol{u} - \\boldsymbol{\\ell}}{2}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rhat_worst])
  - $(math_dict[:mu_hat_ucs])
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\Delta}``: Half-width of the box uncertainty set. The box is ``\\lvert \\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}} \\rvert \\leq \\boldsymbol{\\Delta}``.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: The `lb` and `ub` fields of the set. The builder reads their half-difference alone.

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `bucs_w_i`: ``\\boldsymbol{b}``, ``N \\times 1``, created.

## Constraints

  - `bucs_ret_i`: ``\\left(s_c b_j,\\; s_c w_j\\right) \\in \\mathcal{K}_{1}`` for ``j = 1, \\dots, N``, so ``b_j \\geq \\lvert w_j \\rvert``.

## Expressions

  - `ret_i`: ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\boldsymbol{\\Delta}^\\intercal \\boldsymbol{b}``, less the term's flagged charges.

Where:

  - ``\\boldsymbol{b}``: Epigraph variable of ``\\lvert \\boldsymbol{w} \\rvert``, with entries ``b_j``.
  - $(math_dict[:w_i_asset])
  - $(math_dict[:sc_scale])
  - ``\\mathcal{K}_{1} = \\{(t, x) : t \\geq \\lvert x \\rvert\\}``: Norm cone of order one in two dimensions.
  - $(math_dict[:N])

## Relaxation

$(val_dict[:relax])

  - The rows give ``\\boldsymbol{b} \\geq \\lvert \\boldsymbol{w} \\rvert``, so `ret_i` lies at or below ``\\hat{r}(\\boldsymbol{w})``, less the charges.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when ``\\hat{r}(\\boldsymbol{w})`` meets it, so the feasible weights are exact. Under [`MinimumRisk`](@ref) with no binding bound, the reported `ret_i` can lie far below the worst case.

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term. Every name that the builder registers ends in it.
  - `ucs`: The uncertainty set.
  - `mu`: Fallback characteristic vector, which the builder uses when the set carries no centre.
  - `settings::JuMPReturnsSettings`: The term's settings. The builder reads `fee` and `mic`.

# Returns

  - `(ret, mu, forces_risk)`: the term's expression, the centre of the set, and whether the term forces the risk form of [`MaximumRatio`](@ref). The centre is the set's own field when it has one, and the fallback otherwise.

# Related

  - [`set_return_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::BoxUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings,
                                     ::Any = nothing)
    sc = get_constraint_scale(model)
    w = get_w(model)
    N = length(w)
    mu = something(ucs.val, mu)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    bucs_w = state_set!(model, Symbol(""), :bucs_w_, i, JuMP.@variable(model, [1:N]))
    state_set!(model, Symbol(""), :bucs_ret_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * bucs_w[j]; sc * w[j]] in JuMP.MOI.NormOneCone(2)))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model,
                                      dot_scalar(mu, w) - LinearAlgebra.dot(d_mu, bucs_w)))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::EllipsoidalUncertaintySet, mu, settings,
                                mtx_sqrt = EigenFallbackSquareRoot())

Build one term's ellipsoid-robust return expression.

The term forces the risk form of [`MaximumRatio`](@ref).

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\kappa \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,, \\\\
\\mathbf{G}^\\intercal \\mathbf{G} &= \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rhat_worst])
  - $(math_dict[:mu_hat_ucs])
  - $(math_dict[:w_port])
  - ``\\kappa``: Radius of the ellipsoid, the `k` field of the set. The set is ``(\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}})^\\intercal \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}^{-1} (\\boldsymbol{\\mu} - \\hat{\\boldsymbol{\\mu}}) \\leq \\kappa^{2}``.
  - ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}``: Shape matrix of the ellipsoid, the `sigma` field of the set.
  - ``\\mathbf{G}``: Transpose of the square root of ``\\mathbf{\\Sigma}_{\\boldsymbol{\\mu}}`` that [`matrix_square_root`](@ref) takes under `mtx_sqrt`, the `mtx_sqrt` of the [`ArithmeticReturn`](@ref). `nothing` takes the plain Cholesky factor, which raises a `LinearAlgebra.PosDefException` on a matrix that is not positive definite. So ``\\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2^{2} = \\boldsymbol{w}^\\intercal \\mathbf{\\Sigma}_{\\boldsymbol{\\mu}} \\boldsymbol{w}``.

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_eucs_gw_i`: ``t``, created.

## Expressions

  - `x_eucs_w_i`: ``\\mathbf{G}\\boldsymbol{w}``.
  - `ret_i`: ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\kappa t``, less the term's flagged charges.

## Constraints

  - `eucs_ret_i`: ``\\left(s_c t,\\; s_c \\mathbf{G}\\boldsymbol{w}\\right) \\in \\mathcal{K}_{2}``, so ``t \\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2``.

Where:

  - ``t``: Epigraph variable of ``\\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2``.
  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])

## Relaxation

$(val_dict[:relax])

  - The row gives ``t \\geq \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2``, so `ret_i` lies at or below ``\\hat{r}(\\boldsymbol{w})``, less the charges.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when ``\\hat{r}(\\boldsymbol{w})`` meets it.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::EllipsoidalUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings,
                                     mtx_sqrt::Option{<:AbstractMatrixSquareRootAlgorithm} = EigenFallbackSquareRoot())
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.val, mu)
    G = transpose(matrix_square_root(mtx_sqrt, ucs.sigma))
    k = ucs.k
    x_eucs_w = state_set!(model, Symbol(""), :x_eucs_w_, i, JuMP.@expression(model, G * w))
    t_eucs_gw = state_set!(model, Symbol(""), :t_eucs_gw_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :eucs_ret_, i,
               JuMP.@constraint(model,
                                [sc * t_eucs_gw; sc * x_eucs_w] in JuMP.SecondOrderCone()))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w) - k * t_eucs_gw))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::L1UncertaintySet, mu, settings)

Build one term's ``\\ell_1``-robust return expression.

The rows are linear, so the model is a linear programme whenever the rest of the problem is,
see [`NoRisk`](@ref). The term forces the risk form of [`MaximumRatio`](@ref). The rows are
linear, but a radius can put every worst-case return at or below ``r_f``, and then the return
form has no solution.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - $(math_dict[:rhat_worst])
  - $(math_dict[:mu_hat_ucs])
  - $(math_dict[:w_port])
  - ``\\epsilon``: Radius of the ``\\ell_1`` uncertainty set, the `eps` field.
  - ``\\boldsymbol{\\sigma}``: Per-asset scale vector, the `sd` field, with entries ``\\sigma_i``. It is ``\\boldsymbol{1}`` when `sd` is `nothing`.

Two ``\\ell_1`` terms whose `sd` are different do not become one term. The sum of their
penalties is one infinity norm only when all the `sd` are equal.

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_l1ucs_i`: ``t``, created.

## Constraints

  - `l1ucs_ret_i`: ``\\left(s_c t,\\; s_c\\, \\boldsymbol{\\sigma} \\odot \\boldsymbol{w}\\right) \\in \\mathcal{K}_{\\infty}``, so ``t \\geq \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty``.

## Expressions

  - `ret_i`: ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\epsilon t``, less the term's flagged charges.

Where:

  - ``t``: Epigraph variable of ``\\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty``.
  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])

## Relaxation

$(val_dict[:relax])

  - The row gives ``t \\geq \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty``, so `ret_i` lies at or below ``\\hat{r}(\\boldsymbol{w})``, less the charges.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when ``\\hat{r}(\\boldsymbol{w})`` meets it.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::L1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings,
                                     ::Any = nothing)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.mu, mu)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    t_l1ucs = state_set!(model, Symbol(""), :t_l1ucs_, i, JuMP.@variable(model))
    state_set!(model, Symbol(""), :l1ucs_ret_, i,
               JuMP.@constraint(model,
                                [sc * t_l1ucs;
                                 sc * sw] in JuMP.MOI.NormInfinityCone(1 + length(w))))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, dot_scalar(mu, w) - ucs.eps * t_l1ucs))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    # The rows are linear, but the characteristic does not carry the penalty (#1358).
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::SignedL1UncertaintySet, mu, settings)

Build one term's signed-``\\ell_1``-robust return expression.

The rows are linear, and the term forces the risk form of [`MaximumRatio`](@ref), as the
``\\ell_1`` set does. The worst case keeps the long-short problem as one problem. Thus it does not need the two
separate problems of equations (27) and (28) of [quintile](@cite). It also does not need the
condition on complementary supports that Remark 12 of that source sets to join the two parts.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
\\end{align}
```

Where:

  - $(math_dict[:rhat_worst])
  - $(math_dict[:mu_hat_ucs])
  - $(math_dict[:w_port])
  - $(math_dict[:w_i_asset])
  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive-error and the negative-error sides, the `ep` and `en` fields.
  - $(math_dict[:sigma_i_ucs])
  - $(math_dict[:pos_part])

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_sl1ucs_p_i`: ``t_{+} \\geq 0``, created.
  - `t_sl1ucs_m_i`: ``t_{-} \\geq 0``, created.

## Constraints

  - `sl1ucs_ret_p_i`: ``s_c \\left(-\\sigma_j w_j - t_{+}\\right) \\leq 0`` for ``j = 1, \\dots, N``.
  - `sl1ucs_ret_m_i`: ``s_c \\left(\\sigma_j w_j - t_{-}\\right) \\leq 0`` for ``j = 1, \\dots, N``.

## Expressions

  - `ret_i`: ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\epsilon_{+} t_{+} - \\epsilon_{-} t_{-}``, less the term's flagged charges.

Where:

  - ``t_{+}``, ``t_{-}``: Epigraph variables of the two positive parts.
  - $(math_dict[:sc_scale])
  - $(math_dict[:N])

## Relaxation

$(val_dict[:relax])

  - The rows give ``t_{+} \\geq \\left[\\max_j (-\\sigma_j w_j)\\right]_{+}`` and ``t_{-} \\geq \\left[\\max_j (\\sigma_j w_j)\\right]_{+}``, so `ret_i` lies at or below ``\\hat{r}(\\boldsymbol{w})``, less the charges.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when ``\\hat{r}(\\boldsymbol{w})`` meets it.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::SignedL1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings,
                                     ::Any = nothing)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.mu, mu)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    t_sl1ucs_p = state_set!(model, Symbol(""), :t_sl1ucs_p_, i,
                            JuMP.@variable(model, lower_bound = 0))
    t_sl1ucs_m = state_set!(model, Symbol(""), :t_sl1ucs_m_, i,
                            JuMP.@variable(model, lower_bound = 0))
    state_set!(model, Symbol(""), :sl1ucs_ret_p_, i,
               JuMP.@constraint(model, sc * (-sw .- t_sl1ucs_p) <= 0))
    state_set!(model, Symbol(""), :sl1ucs_ret_m_, i,
               JuMP.@constraint(model, sc * (sw .- t_sl1ucs_m) <= 0))
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model,
                                      dot_scalar(mu, w) - ucs.ep * t_sl1ucs_p -
                                      ucs.en * t_sl1ucs_m))
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    # The rows are linear, but the characteristic does not carry the penalty (#1358).
    return ret, mu, true
end
"""
    set_ucs_return_constraints!(model, i, ucs::NormBallUncertaintySet, mu, settings)

Build one term's norm-ball-robust return expression.

The builder raises one cone of the dual order on ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}``.
The map of the set takes the place of the Cholesky factor of the ellipsoid, and the builder
factorises nothing. A map with no column raises no cone and leaves the nominal return. That
term forces the risk form of [`MaximumRatio`](@ref) only when it deducts a charge. All other
norm balls force it.
The method takes the mean tag alone, and the [`ArithmeticReturn`](@ref) constructor refuses a
set that carries the covariance tag.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\kappa \\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{q}\\,, \\quad \\frac{1}{p} + \\frac{1}{q} = 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:rhat_worst])
  - $(math_dict[:mu_hat_ucs])
  - $(math_dict[:w_port])
  - ``\\kappa``: Radius of the norm ball, the `kappa` field. The set is ``\\{\\hat{\\boldsymbol{\\mu}} + \\mathbf{L}\\boldsymbol{u} : \\lVert \\boldsymbol{u} \\rVert_{p} \\leq \\kappa\\}``.
  - ``\\mathbf{L}``: Geometry map of the set, ``N \\times r``, the `L` field.
  - ``p``, ``q``: Norm order of the set, the `p` field, and its dual.

# JuMP formulation

## Variables

  - `w`: read from the model.

## Expressions

  - `x_nbucs_w_i`: ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}``, registered only when ``\\mathbf{L}`` has a column.
  - `ret_i`: ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - \\kappa t``, or ``\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w}`` when ``\\mathbf{L}`` has no column, less the term's flagged charges.

Where:

  - ``t``: Epigraph variable of ``\\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{q}``, which [`norm_ball_dual_norm_epigraph!`](@ref) registers with its cone.
  - ``\\kappa``, ``\\mathbf{L}``, ``q``: As above.

## Relaxation

$(val_dict[:relax])

  - The cone gives ``t \\geq \\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{q}``, so `ret_i` lies at or below ``\\hat{r}(\\boldsymbol{w})``, less the charges. A map with no column is exact.
  - The bound is tight when the objective raises `ret_i`, or when the term's lower bound binds. [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) and the risk form of [`MaximumRatio`](@ref) raise `ret_i`. A weight vector meets a lower bound on `ret_i` exactly when ``\\hat{r}(\\boldsymbol{w})`` meets it.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`norm_ball_dual_norm_epigraph!`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i,
                                     ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                                 <:MuUncertaintySetClass},
                                     mu::Num_VecNum, settings::JuMPReturnsSettings,
                                     ::Any = nothing)
    w = get_w(model)
    mu = something(ucs.val, mu)
    L = ucs.L
    # A map with no column spans nothing, so the worst case is the nominal return and no
    # cone is needed.
    robust = size(L, 2) > zero(Int)
    ret = if robust
        x_nbucs_w = state_set!(model, Symbol(""), :x_nbucs_w_, i,
                               JuMP.@expression(model, transpose(L) * w))
        # The covariance builder registers its epigraph under the same prefix and index, so
        # this one tags the index.
        t_nbucs = norm_ball_dual_norm_epigraph!(model, Symbol(""), Symbol(:w_, i),
                                                x_nbucs_w, ucs.p)
        JuMP.@expression(model, dot_scalar(mu, w) - ucs.kappa * t_nbucs)
    else
        JuMP.@expression(model, dot_scalar(mu, w))
    end
    ret = state_set!(model, Symbol(""), :ret_, i, ret)
    fee = add_fees_to_ret!(model, ret, settings.fee)
    mic = add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, robust || fee || mic
end
function set_return_constraints!(model::JuMP.Model, i,
                                 pret::ArithmeticReturn{<:Any, <:UcSE_UcS, <:Any},
                                 pr::AbstractPriorResult; rd::ReturnsResult, kwargs...)
    settings = pret.settings
    # The set is a neighbourhood of the quantity it was calibrated on, so it names the
    # centre. The term's own field and then the prior are the fallbacks (ADR 0050).
    fb = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    # The prior travels beside the returns, because an `AbstractPriorUncertaintySetEstimator`
    # is fitted from the optimisation's own prior result rather than from returns data. An
    # estimator that carries its own `pe` drops it (see [`mu_ucs`](@ref)).
    uc = mu_ucs(pret.ucs, rd, pr; kwargs...)
    ret, mu, forces_risk = set_ucs_return_constraints!(model, i, uc, fb, settings,
                                                       pret.mtx_sqrt)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, forces_risk
end
function set_return_constraints!(model::JuMP.Model, i, pret::LogarithmicReturn,
                                 pr::AbstractPriorResult; kwargs...)
    k = get_k(model)
    sc = get_constraint_scale(model)
    settings = pret.settings
    X = set_portfolio_returns!(model, pr.X)
    T = length(X)
    t_elog_ret = state_set!(model, Symbol(""), :t_elog_ret_, i,
                            JuMP.@variable(model, [1:T]))
    wi = nothing_scalar_array_selector(pret.w, pr.w)
    wi = get_observation_weights(wi, X)
    ret = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(t_elog_ret))
    else
        JuMP.@expression(model, Statistics.mean(t_elog_ret, wi))
    end
    state_set!(model, Symbol(""), :ret_, i, ret)
    fee = add_fees_to_ret!(model, ret, settings.fee)
    mic = add_market_impact_cost!(model, ret, settings.mic)
    kret = state_set!(model, Symbol(""), :kret_, i, JuMP.@expression(model, k .+ X))
    state_set!(model, Symbol(""), :elog_ret_ret_, i,
               JuMP.@constraint(model, [j = 1:T],
                                [sc * t_elog_ret[j], sc * k, sc * kret[j]] in
                                JuMP.MOI.ExponentialCone()))
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    # A logarithmic term holds no per-asset quantity, which forces the ratio's risk form. A
    # charge forces it too, when an arithmetic term supplies the characteristic (#1358).
    return nothing, fee || mic
end
function set_return_constraints!(model::JuMP.Model, i, pret::NoReturn,
                                 ::AbstractPriorResult; kwargs...)
    settings = pret.settings
    ret = state_set!(model, Symbol(""), :ret_, i,
                     JuMP.@expression(model, zero(JuMP.AffExpr)))
    # No charge is applied here, and this is why `settings.fee` and `settings.mic` are inert:
    # the term's expression is identically zero by construction, every guard `NoReturn`
    # carries rests on that, and a fee subtracted here would make it non-zero.
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    # No per-asset quantity, and no robust cone.
    return nothing, false
end
