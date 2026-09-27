"""
    set_return_bounds!(model, i, ret_expr, lb)

Bound the expression of return term `i` from below.

The `Nothing` overload does nothing. With a scalar `lb`, adds `ret_i >= lb * k`. With a
[`Frontier`](@ref) or a vector, pushes the term onto the `:ret_frontier` Model State entry
for a later sweep, exactly as [`set_risk_upper_bound!`](@ref) does on the risk side.

The bound binds on the term's **own** expression, net of that term's own flagged charges and
before `settings.scale` is applied, and it binds whether or not `settings.rte` is `true` — so
a term may constrain the portfolio without entering the objective at all.

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

If `rte` is `false` the function does nothing, so the term contributes nothing to the model's
return expression while its own bound still binds. The twin of
[`set_risk_expression!`](@ref).

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

The collapse is always the **weighted sum** ``\\sum_i s_i\\, \\mathrm{ret}_i``. There is no
scalariser on this side, and there is no configuration in which there is one: the
package's scalarisers follow cvxpy's `scalarize` transforms, whose `max` and `log_sum_exp`
discard the objective's sense and so fail on a maximised concave expression, and cvxpy ships
no `min`. Normalising the sense to rescue them is barred, because `:ret` is a model-global
name that the objective, the bounds, the ratio and [`NearOptimalCentering`](@ref) all read,
and a stored `-ret` leads every one of them astray.

An empty `:ret_vec` — every term opted out through `settings.rte = false` — gives a zero
return expression rather than an error here. The refusal belongs to the objective, not to the
collapse: [`MinimumRisk`](@ref) and [`MaximumUtility`](@ref) read a zero `:ret` legitimately,
while [`MaximumReturn`](@ref) and [`MaximumRatio`](@ref) are refused upstream by
[`assert_no_return_objective_compatibility`](@ref).

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
    set_max_ratio_return_constraints!(model, obj, rets, mus, robust, pr)

Add the maximum-ratio homogenisation constraint to the model.

The constraint is **hoisted** out of the per-term builders and runs exactly once. It reads
the model-global `:ret` and registers the model-global names `sr_ret` and `sr_risk`, so *k*
copies of it would collide and each would read the wrong expression.

Which of the two forms is used is decided by a **structural `any`** and a **numeric
aggregate**:

 1. If any term has no per-asset characteristic (a [`LogarithmicReturn`](@ref)) or builds a
    robust cone (a box or ellipsoidal uncertainty set), the risk form is used.
 2. Otherwise the aggregate ``\\sum_{i:\\,\\mathrm{rte}} s_i \\boldsymbol{\\mu}_i`` decides:
    `all(x -> x <= rf, ·)` selects the risk form.

Step 2 is the exact generalisation of the single-term test. A per-term `any` would send two
terms at `0.9 r_f` and `scale = 1` down the weaker branch, though their sum is `1.8 r_f`.

An **empty numerator** is refused, mirroring [`NoRisk`](@ref) under this objective: with every
term out of `:ret`, `k` collapses to `0` at `rf > 0` and the problem returns an arbitrary
feasible point at `rf = 0`.

# Arguments

  - $(arg_dict[:model])
  - `obj`: Objective function; a no-op unless it is a [`MaximumRatio`](@ref).
  - `rets`: The return terms.
  - `mus`: Each term's resolved characteristic, `nothing` where it has none.
  - `robust`: Whether each term built a robust cone.
  - `pr`: Prior result, the fallback for sizing `ohf`.

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
                                           mus::AbstractVector, robust::AbstractVector,
                                           pr::AbstractPriorResult)
    # The empty-numerator refusal is not here: it is one of the three objective refusals
    # `assert_no_return_objective_compatibility` makes at the top of this seam.
    mu = aggregate_return_characteristic(rets, mus)
    set_maximum_ratio_normalisation!(model, obj, mu, pr)
    sc = get_constraint_scale(model)
    k = get_k(model)
    ohf = shared_get(model, :ohf)
    ret = get_ret(model)
    rf = obj.rf
    risk_form = any(robust) || isnothing(mu) || all(x -> x <= rf, mu)
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

Returns `nothing` when no included term carries a per-asset quantity, which is the state a
pure [`LogarithmicReturn`](@ref) problem is in. A term whose `settings.rte` is `false` is
skipped: it contributes nothing to `:ret`, so it must contribute nothing to the aggregate the
ratio's tests read.

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

Does nothing when the term's `settings.fee` is `false`, or when no fees are registered.

The model carries two fee expressions. `:fees` holds the per period terms `l`, `s` and `tn`, and it
enters the return unchanged. `:one_time_fees` holds the two fixed terms, which are charged one time
for the whole holding period, so it enters divided by `:T`, the observation count of the fit.
An expected return is a per period number, so the one-off cost is always spread here, whatever
clock the fee's `horizon` names for a realised series.

The charge stays **inside** each builder, so with *k* terms the multiplier on the fee is
``\\sum_{i:\\,\\mathrm{fee}} s_i``. That multiplier is deliberately unconstrained: a blend of
two terms at `scale = 0.5` charges the fee once, and two terms at `scale = 1` charge it
twice.

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `fee::Bool`: The term's `settings.fee`.

# Returns

  - `nothing`.

# Related

  - [`add_market_impact_cost!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_fees_to_ret!(model::JuMP.Model, ret, fee::Bool)
    if !fee
        return nothing
    end
    if shared_has(model, :fees)
        JuMP.add_to_expression!(ret, -shared_get(model, :fees))
    end
    # An expected return is a per period number, so a fee charged one time for the whole
    # holding period enters it divided by the observation count of the fit.
    if shared_has(model, :one_time_fees)
        JuMP.add_to_expression!(ret, -shared_get(model, :one_time_fees) / get_T(model))
    end
    return nothing
end
"""
    add_market_impact_cost!(model, ret, mic::Bool)

Subtract the market impact cost from one term's return expression.

Does nothing when the term's `settings.mic` is `false`, or when no market impact cost is
registered. Only [`BudgetMarketImpact`](@ref) registers one; a plain budget cost constrains
the budget and never reaches the return expression, despite sharing the `cost_bgt_expr` name.

# Arguments

  - $(arg_dict[:model])
  - `ret`: JuMP return expression to modify in-place.
  - `mic::Bool`: The term's `settings.mic`.

# Returns

  - `nothing`.

# Related

  - [`add_fees_to_ret!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_market_impact_cost!(model::JuMP.Model, ret, mic::Bool)
    if !mic || !shared_has(model, :wip)
        return nothing
    end
    JuMP.add_to_expression!(ret, -shared_get(model, :cost_bgt_expr))
    return nothing
end
"""
    set_return_constraints!(model, pret, obj, pr; kwargs...)
    set_return_constraints!(model, i, pret, pr; kwargs...)

Build the model's return expression and the constraints that go with it.

The four-argument methods are the seam every JuMP optimiser reaches. They run the per-term
builder once per return term, collapse the results into the single `:ret` expression, and
then add the hoisted maximum-ratio constraint. The five-argument methods are the per-term
builders, which dispatch on the term's type and on the shape of its uncertainty set.

Each per-term builder registers its own index-suffixed names (`ret_1`, `t_l1ucs_2`, …),
applies that term's own flagged charges, bounds that term, and pushes the scaled expression
onto `:ret_vec`.

# Arguments

  - $(arg_dict[:model])
  - `pret`: One return term, or a vector of them.
  - `i`: Index of the return term (per-term builders).
  - `obj::ObjectiveFunction`: Portfolio objective function.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `kwargs...`: Additional keyword arguments (e.g. `rd` for uncertainty sets).

# Returns

  - The four-argument methods return `nothing`. A per-term builder returns
    `(mu, robust)`: the characteristic it resolved (or `nothing`), and whether it built a
    robust cone.

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
    mu, robust = set_return_constraints!(model, 1, pret, pr; kwargs...)
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, (pret,), [mu], [robust], pr)
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
    robust = Vector{Bool}(undef, length(pret))
    for (i, pret_i) in enumerate(pret)
        mus[i], robust[i] = set_return_constraints!(model, i, pret_i, pr; kwargs...)
    end
    scalarise_return_expression!(model)
    set_max_ratio_return_constraints!(model, obj, pret, mus, robust, pr)
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
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, false
end
"""
    set_ucs_return_constraints!(model, i, ucs::BoxUncertaintySet, mu, settings)

Build one term's box-robust return expression.

Introduces a norm-1 cone constraint to model the worst-case characteristic under a box
uncertainty set. The family dispatches on the set type: an [`EllipsoidalUncertaintySet`](@ref)
raises a second-order cone, and the two ``\\ell_1`` sets raise an infinity-norm cone and a pair
of linear epigraphs respectively.

# Mathematical definition

Box uncertainty set (worst-case return):

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\boldsymbol{\\Delta}^\\intercal |\\boldsymbol{w}|\\,, \\\\
\\boldsymbol{\\Delta} &= \\frac{\\boldsymbol{u} - \\boldsymbol{\\ell}}{2}\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\Delta}``: Half-width of the box uncertainty set.
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bounds of the box uncertainty set.

# Arguments

  - $(arg_dict[:model])
  - `i`: Index of the return term, which suffixes every name the builder registers.
  - `ucs`: The uncertainty set.
  - `mu`: Fallback characteristic vector, used when the set carries none of its own.
  - `settings::JuMPReturnsSettings`: The term's settings, read for `fee` and `mic`.

# Returns

  - `(ret, mu, robust)`: the term's expression, the characteristic the set is centred on —
    the set's own field wins over the fallback — and whether the builder raised a
    cone the ratio's `ret == rf k + ohf` normalisation cannot be used with.

# Related

  - [`set_return_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::BoxUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
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
    set_ucs_return_constraints!(model, i, ucs::EllipsoidalUncertaintySet, mu, settings)

Build one term's ellipsoid-robust return expression.

Introduces a second-order cone constraint to model the worst-case characteristic under an
ellipsoidal uncertainty set. The cone is not linear, so the term is reported as `robust`, and
the ratio's `ret == rf k + ohf` normalisation cannot be used with it.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\kappa \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\kappa``: Ellipsoidal uncertainty set radius.
  - ``\\mathbf{G}``: Upper Cholesky factor of the uncertainty set covariance.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::EllipsoidalUncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    sc = get_constraint_scale(model)
    w = get_w(model)
    mu = something(ucs.val, mu)
    G = LinearAlgebra.cholesky(ucs.sigma).U
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

Introduces an infinity-norm cone constraint to model the worst-case characteristic under an
``\\ell_1`` uncertainty set. The constraint is linear, so the resulting model is an LP
whenever the rest of the problem is (see [`NoRisk`](@ref)).

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\epsilon \\lVert \\boldsymbol{\\sigma} \\odot \\boldsymbol{w} \\rVert_\\infty\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\epsilon``: Radius of the ``\\ell_1`` uncertainty set.
  - ``\\boldsymbol{\\sigma}``: Per-asset scaling (`sd`); ``\\boldsymbol{1}`` when `sd` is `nothing`.

Two ``\\ell_1`` terms whose `sd` differ do **not** collapse into one: the sum of their
penalties is not a single infinity norm unless every `sd` matches.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::L1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
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
    return ret, mu, false
end
"""
    set_ucs_return_constraints!(model, i, ucs::SignedL1UncertaintySet, mu, settings)

Build one term's signed-``\\ell_1``-robust return expression.

Introduces one epigraph variable per error sign. Because the objective maximises the return expression, each variable is driven down to its lower bound, so `t_sl1ucs_p` attains ``[\\max_i(-\\sigma_i w_i)]_+`` and `t_sl1ucs_m` attains ``[\\max_i(\\sigma_i w_i)]_+`` at the optimum. The constraints are linear.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\epsilon_{+} \\left[\\underset{i}{\\max}\\, (-\\sigma_i w_i)\\right]_{+} - \\epsilon_{-} \\left[\\underset{i}{\\max}\\, (\\sigma_i w_i)\\right]_{+}\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\epsilon_{+}``, ``\\epsilon_{-}``: Radii of the positive- and negative-error sides.
  - ``\\boldsymbol{\\sigma}``: Per-asset scaling (`sd`); ``\\boldsymbol{1}`` when `sd` is `nothing`.

Modelling this worst case directly keeps the long-short problem *coupled*, so it does not need the decoupling of equations (27) and (28) of [quintile](@cite), nor the complementary-support caveat its Remark 12 attaches to recombining them.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i, ucs::SignedL1UncertaintySet,
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
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
    return ret, mu, false
end
"""
    set_ucs_return_constraints!(model, i, ucs::NormBallUncertaintySet, mu, settings)

Build one term's norm-ball-robust return expression.

Introduces one cone on ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}``, the cone the dual norm
order names, so the ellipsoid's Cholesky factor is replaced by the set's own map and nothing
is factorised. A map with no column raises no cone and leaves the nominal return, and the term
is then not reported as `robust`. The method is defined on the mean tag alone, and the
[`ArithmeticReturn`](@ref) constructor refuses a set that carries the covariance tag.

# Mathematical definition

```math
\\begin{align}
\\hat{r}(\\boldsymbol{w}) &= \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\kappa \\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{q}\\,, \\quad \\frac{1}{p} + \\frac{1}{q} = 1\\,.
\\end{align}
```

Where:

  - ``\\hat{r}(\\boldsymbol{w})``: Worst-case expected return.
  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\kappa``: Norm-ball radius.
  - ``\\mathbf{L}``: Geometry map of the set, ``N \\times r``.
  - ``p``, ``q``: Norm order of the set and its dual.

# JuMP formulation

## Variables

  - `w`: portfolio weights, read from the model.

## Expressions

  - `x_nbucs_w_i`: ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}``, registered only when ``\\mathbf{L}`` has a column.
  - `ret_i`: ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\kappa t``, with ``t`` the epigraph [`norm_ball_dual_norm_epigraph!`](@ref) registers, or ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}`` when ``\\mathbf{L}`` has no column.

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``\\kappa``, ``\\mathbf{L}``: Radius and geometry map of the set.
  - ``t``: Epigraph of ``\\lVert \\mathbf{L}^{\\intercal}\\boldsymbol{w} \\rVert_{q}``.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`norm_ball_dual_norm_epigraph!`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, i,
                                     ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                                 <:MuUncertaintySetClass},
                                     mu::Num_VecNum, settings::JuMPReturnsSettings)
    w = get_w(model)
    mu = something(ucs.val, mu)
    L = ucs.L
    # A map with no column spans nothing, so the worst case is the nominal return and no
    # cone is needed.
    robust = size(L, 2) > zero(Int)
    ret = if robust
        x_nbucs_w = state_set!(model, Symbol(""), :x_nbucs_w_, i,
                               JuMP.@expression(model, transpose(L) * w))
        t_nbucs = norm_ball_dual_norm_epigraph!(model, Symbol(""), i, x_nbucs_w, ucs.p)
        JuMP.@expression(model, dot_scalar(mu, w) - ucs.kappa * t_nbucs)
    else
        JuMP.@expression(model, dot_scalar(mu, w))
    end
    ret = state_set!(model, Symbol(""), :ret_, i, ret)
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    return ret, mu, robust
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
    ret, mu, robust = set_ucs_return_constraints!(model, i, uc, fb, settings)
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    return mu, robust
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
    add_fees_to_ret!(model, ret, settings.fee)
    add_market_impact_cost!(model, ret, settings.mic)
    kret = state_set!(model, Symbol(""), :kret_, i, JuMP.@expression(model, k .+ X))
    state_set!(model, Symbol(""), :elog_ret_ret_, i,
               JuMP.@constraint(model, [j = 1:T],
                                [sc * t_elog_ret[j], sc * k, sc * kret[j]] in
                                JuMP.MOI.ExponentialCone()))
    set_return_bounds!(model, i, ret, settings.lb)
    set_return_expression!(model, i, ret, settings.scale, settings.rte)
    # A logarithmic term holds no per-asset quantity, which forces the ratio's risk form.
    return nothing, false
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
