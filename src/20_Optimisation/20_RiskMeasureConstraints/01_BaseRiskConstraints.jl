"""
    const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}

Alias for JuMP optimisers that do not use factor risk contribution.

Matches [`MeanRisk`](@ref), [`NearOptimalCentering`](@ref), or [`RiskBudgeting`](@ref). Used for dispatch in risk constraint generation functions that apply to these optimiser types but not to factor risk contribution.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`RiskBudgeting`](@ref)
"""
const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collapse the `risk_vec` expression array stored in `model` into a single scalar
`risk` JuMP expression.

The `SumScalariser` method sums all entries into a linear or quadratic expression. The
`LogSumExpScalariser` method introduces auxiliary variables and exponential cone constraints
to encode a log-sum-exp scalarisation. The `MaxScalariser` method introduces a variable and
linear constraints to encode the maximum over all entries.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}_{\\mathrm{sum}} &= \\sum_k \\mathcal{R}_k\\,, \\\\
\\mathcal{R}_{\\mathrm{lse}} &= \\frac{1}{\\gamma}\\ln\\sum_k e^{\\gamma \\mathcal{R}_k}\\,, \\\\
\\mathcal{R}_{\\mathrm{max}} &= \\max_k \\mathcal{R}_k\\,.
\\end{align}
```

Where:

  - ``\\mathcal{R}_{\\mathrm{sum}}``: Sum scalarisation.
  - ``\\mathcal{R}_{\\mathrm{lse}}``: Log-sum-exp scalarisation.
  - ``\\mathcal{R}_{\\mathrm{max}}``: Maximum scalarisation.
  - ``\\mathcal{R}_k``: ``k``-th risk expression.
  - ``\\gamma``: Temperature parameter for log-sum-exp.

# Arguments

  - $(arg_dict[:model])
  - `sca`: Scalariser instance (one of [`SumScalariser`](@ref), [`LogSumExpScalariser`](@ref),
    or [`MaxScalariser`](@ref)).

# Returns

  - `nothing`.

# Related

  - [`SumScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`MaxScalariser`](@ref)
"""
function scalarise_risk_expression!(model::JuMP.Model, ::SumScalariser)
    if !shared_has(model, :risk_vec)
        return nothing
    end
    risk_vec = shared_get(model, :risk_vec)
    if any(x -> isa(x, JuMP.QuadExpr), risk_vec)
        JuMP.@expression(model, risk, zero(JuMP.QuadExpr))
    else
        JuMP.@expression(model, risk, zero(JuMP.AffExpr))
    end
    for risk_i in risk_vec
        JuMP.add_to_expression!(risk, risk_i)
    end
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, sca::LogSumExpScalariser)
    if !shared_has(model, :risk_vec)
        return nothing
    end
    risk_vec = shared_get(model, :risk_vec)
    sc = get_constraint_scale(model)
    N = length(risk_vec)
    gamma = sca.gamma
    JuMP.@variables(model, begin
                        risk
                        u_risk[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          u_risk_lse, sc * (sum(u_risk) - 1) <= 0
                          risk_lse[i = 1:N],
                          [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
                          JuMP.MOI.ExponentialCone()
                      end)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    if !shared_has(model, :risk_vec)
        return nothing
    end
    risk_vec = shared_get(model, :risk_vec)
    JuMP.@variable(model, risk)
    JuMP.@constraint(model, risk_ms, risk .- risk_vec .>= 0)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Number of entries the `:risk_frontier` Model State registry holds, or `0` when it holds none.

Read before a risk measure builds its constraints, so that the entries it adds can be told
from the entries its predecessors added.

# Related

  - [`set_risk_frontier_owner!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function risk_frontier_length(model::JuMP.Model)::Int
    return shared_has(model, :risk_frontier) ? length(shared_get(model, :risk_frontier)) : 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Stamp `i` as the owning risk measure of every `:risk_frontier` entry after position `first`.

[`rebuild_risk_frontier`](@ref) resolves a [`Frontier`](@ref) bound into a span of numbers, and
it needs the risk measure the span belongs to. The registry is **not** parallel to the measure
vector: a measure registers an entry only when its `settings.ub` is a [`Front_NumVec`](@ref),
and a memoised measure such as [`UlcerIndex`](@ref) registers one entry for every copy of
itself in that vector. So the position of an entry does not name its measure, and the owner is
recorded rather than derived. This mirrors the return side, whose `:ret_frontier` entry has
carried its own term index from the start.

# Arguments

  - $(arg_dict[:model])
  - `first::Integer`: The registry length before measure `i` built its constraints, as
    [`risk_frontier_length`](@ref) reads it.
  - `i::Integer`: The position of the measure in the risk measure vector.

# Returns

  - `nothing`.

# Related

  - [`risk_frontier_length`](@ref)
  - [`rebuild_risk_frontier`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function set_risk_frontier_owner!(model::JuMP.Model, first::Integer, i::Integer)
    if !shared_has(model, :risk_frontier)
        return nothing
    end
    risk_frontier = shared_get(model, :risk_frontier)
    for j in (first + 1):length(risk_frontier)
        keys, vals = risk_frontier[j]
        risk_frontier[j] = keys => (vals[1], vals[2], vals[3], Int(i))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Dispatch to index-aware `set_risk_constraints!` for a single risk measure or iterate over a
vector of risk measures.

The single-measure overload calls `set_risk_constraints!(model, 1, r, ...)`. The vector
overload calls `set_risk_constraints!(model, i, rs[i], ...)` for each element.

The single-measure overload also drops the measure's `scale` through
[`unit_scale_risk_measure`](@ref). `scale` weights a measure inside an aggregate built from
several measures, and one measure is not an aggregate, so the weight is inert. The vector
overload keeps every element's `scale`.

Both overloads stamp the owning measure onto the `:risk_frontier` entries that measure
registered ([`set_risk_frontier_owner!`](@ref)). This is the only depth at which the measure
and its entries are both in hand.

Both overloads resolve the measure through [`resolve_deferred_quantities`](@ref), and both
thread the estimator's own solver into that call. This route calls no [`factory`](@ref), so
no selection has run and a measure that states no solver of its own still holds `nothing`. A
**Calibration Rule** that reads the solver would see that `nothing`, while the same rule on
the `factory` route sees the optimiser's. Threading `opt.opt.slv` is what makes the two
routes resolve one measure against one solver.

# Arguments

  - $(arg_dict[:model])
  - `r`: A [`RiskMeasure`](@ref) instance, or `rs` a vector of risk measures.
  - $(arg_dict[:opt_jumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:b1_opt])

# Returns

  - `nothing`.

# Related

  - [`RiskMeasure`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`assert_declared_calibration_resolver`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, r::RiskMeasure,
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                               b1::Option{<:MatNum} = nothing; kwargs...)
    # A `JuMP` model builder reads the measure's slots directly and never calls `factory`,
    # so this is where a Deferred Quantity becomes a value. It resolves the deferred state
    # alone; each builder's own prior fallback is untouched.
    #
    # The estimator's own solver is threaded with the prior. No selection runs on this
    # route, so a measure that states no solver of its own holds `nothing` here, and a
    # Calibration Rule that reads the solver would see that `nothing` rather than the one
    # the optimisation settled on. The `factory` route settles it by selection before it
    # resolves, and this is how the two routes are made to agree (issue #591).
    #
    # `scale` is a combination weight, so it is dropped here: a lone measure is not an
    # aggregate and the weight has nothing to weigh. The vector method below keeps it,
    # because there the measures really do combine.
    #
    # `b1` is typed and named, not absorbed by an `args...` tail. The tail let a caller pass
    # a `Fees` in the slot after `fees` and lose it silently — which is exactly what
    # unconstrained `NearOptimalCentering` did (ADR 0008, amendment 2 §4).
    #
    # The resolution is also where the calibration declaration and its resolver are paired.
    # `expected_risk` refuses a surviving Calibration Rule at the value-level entry point,
    # and a `JuMP` builder reads the slot raw, so this route carried no such refusal. A rule
    # that reaches this line names a type that declared the slot and resolved it nowhere.
    first = risk_frontier_length(model)
    resolved = resolve_deferred_quantities(r, pr, opt.opt.slv)
    assert_declared_calibration_resolver(resolved)
    set_risk_constraints!(model, 1, unit_scale_risk_measure(resolved), opt, pr, pl, fees,
                          b1; kwargs...)
    set_risk_frontier_owner!(model, first, 1)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, rs::VecRM, opt::JuMPOptimisationEstimator,
                               pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                               fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing;
                               kwargs...)
    for (i, r) in enumerate(rs)
        first = risk_frontier_length(model)
        resolved = resolve_deferred_quantities(r, pr, opt.opt.slv)
        assert_declared_calibration_resolver(resolved)
        set_risk_constraints!(model, i, resolved, opt, pr, pl, fees, b1; kwargs...)
        set_risk_frontier_owner!(model, first, i)
    end
    return nothing
end
"""
    prior_high_order_quantity(pr::AbstractPriorResult, key::Symbol)

Read the high-order quantity named by `key` off prior result `pr`, or `nothing` when it carries none.

A [`HighOrderPrior`](@ref) carries `kt`, `sk`, `V` and the three vectorisation matrices. A [`LowOrderPrior`](@ref) carries none of them, and a [`HighOrderPrior`](@ref) fitted with only one of the two tensors leaves the other `nothing`, so the answer is a value or `nothing` rather than an error either way.

# Related

  - [`assert_high_order_quantity`](@ref)
  - [`dup_elim_sum_selector`](@ref)
  - [`HighOrderPrior`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function prior_high_order_quantity(pr::AbstractPriorResult, key::Symbol)
    return hasproperty(pr, key) ? getproperty(pr, key) : nothing
end
"""
    assert_high_order_quantity(q, pr::AbstractPriorResult, rm::Symbol, key::Symbol,
                               est::Symbol)

Refuse a high-order risk measure whose tensor resolves neither on the measure nor on the prior.

`q` is what the measure holds in the slot named `key`, after [`resolve_deferred_quantities`](@ref) has run. The measure is buildable when either side supplies the quantity, so the gate is on the pair rather than on the type of `pr`: a caller who has told the measure how to build its own cokurtosis has already met the requirement, and a [`HighOrderPrior`](@ref) that computed neither tensor does not meet it.

The message names the three ways out: state the quantity, name a `est` or an [`AbstractPriorEstimator`](@ref) in the slot, or give the optimiser a prior estimator that computes one.

# Related

  - [`prior_high_order_quantity`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`Kurtosis`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`VarianceSkewKurtosis`](@ref)
"""
function assert_high_order_quantity(q, pr::AbstractPriorResult, rm::Symbol, key::Symbol,
                                    est::Symbol)
    @argcheck(!isnothing(q) || !isnothing(prior_high_order_quantity(pr, key)),
              ArgumentError("`$rm` needs a `$key`, and neither the measure nor the `$(typeof(pr).name.name)` it was given carries one. State `$key` on the measure, or name a `$est` or an `AbstractPriorEstimator` in it, or give the optimiser a prior estimator that computes a `$key`."))
    return nothing
end
"""
    dup_elim_sum_selector(pr::AbstractPriorResult, N::Integer)

Select the duplication, elimination and summation matrices for `N` assets from prior result `pr`, rebuilding whichever of the three it does not carry.

[`dup_elim_sum_matrices`](@ref) is a pure function of the asset count, with no data in it, so the three matrices a [`HighOrderPrior`](@ref) carries are exactly the ones this rebuilds. That is what lets a measure holding its own tensor be built against a [`LowOrderPrior`](@ref): the vectorisation matrices were the only other thing the kernel took from the prior.

One call builds all three, so the rebuild costs the same whether one of them is missing or all three are.

# Related

  - [`dup_elim_sum_matrices`](@ref)
  - [`prior_high_order_quantity`](@ref)
  - [`assert_high_order_quantity`](@ref)
"""
function dup_elim_sum_selector(pr::AbstractPriorResult, N::Integer)
    D2 = prior_high_order_quantity(pr, :D2)
    L2 = prior_high_order_quantity(pr, :L2)
    S2 = prior_high_order_quantity(pr, :S2)
    if isnothing(D2) || isnothing(L2) || isnothing(S2)
        rD2, rL2, rS2 = dup_elim_sum_matrices(N)
        D2 = nothing_scalar_array_selector(D2, rD2)
        L2 = nothing_scalar_array_selector(L2, rL2)
        S2 = nothing_scalar_array_selector(S2, rS2)
    end
    return D2, L2, S2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add an upper-bound constraint on a risk expression to `model`.

The `Nothing` overload does nothing (no bound was requested). The `Front_NumVec` overload
records the expression and its frontier bound vector in the `:risk_frontier` Model State entry for later
use in Pareto frontier solves. That entry is `(bound_var_key, bound_key) => (r_expr, ub, flag, owner)`,
and `owner` is written here as `0`. The measure that registered the entry is not known at this
depth, so [`set_risk_frontier_owner!`](@ref) stamps it from the loop that enumerates the
measures. The `Number` overload adds the constraint
`sc * (r_expr - ub * k) <= 0` directly to the model. The fall-through method emits a
warning: a non-`nothing` bound with an optimiser outside [`NonFRCJuMPOpt`](@ref) is
ignored, which would otherwise happen silently.

# Arguments

  - $(arg_dict[:model])
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk JuMP expression to bound.
  - `ub`: Upper bound; a scalar number or a frontier specification.
  - `key::Symbol`: Symbol used to name the constraint in the model.
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound (default: `true`).

# Returns

  - `nothing`.

# Related

  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
function set_risk_upper_bound!(::JuMP.Model, ::JuMPOptimisationEstimator, r_expr, ::Nothing,
                               key, flag::Bool = true)
    return nothing
end
function set_risk_upper_bound!(::JuMP.Model, opt::JuMPOptimisationEstimator, r_expr, ub,
                               key, flag::Bool = true)
    return @warn("Risk upper bound `settings.ub = $ub` ($key) is not supported by `$(typeof(opt).name.name)` and would be silently ignored. Remove `ub` from the risk measure settings, or use an optimiser that supports risk upper bounds (`MeanRisk`, `NearOptimalCentering`, `RiskBudgeting`).")
end
#! Using parameters to set the upper bounds would make things more difficult from a user perspective. Keep an eye on this in case things change in the future. We could simplify solve_mean_risk! and solve_noc! for pareto frontiers, we can define ub as a parameter and update it for subsequent solves.
# Solver(; name = :clarabel2,
#        solver = () -> ParametricOptInterface.Optimizer(JuMP.MOI.instantiate(Clarabel.Optimizer;
#                                                                        with_cache_type = Float64)),
#        check_sol = (; allow_local = true, allow_almost = true),
#        settings = Dict("verbose" => false, "max_step_fraction" => 0.75))
# https://discourse.julialang.org/t/solver-attributes-and-set-optimizer-with-parametricoptinterface-jl-and-jump-jl/129935/8?u=dcelisgarza
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Front_NumVec, key,
                               flag::Bool = true)
    bound_key = state_key(key, :_ub)
    bound_var_key = state_key(key, :_ub_var)
    if !shared_has(model, :risk_frontier)
        risk_frontier = JuMP.@expression(model, risk_frontier,
                                         Pair{Tuple{Symbol, Symbol},
                                              Tuple{<:JuMP.AbstractJuMPScalar,
                                                    <:Front_NumVec, Bool, Int}}[(bound_var_key, bound_key) => (r_expr,
                                                                                                               ub,
                                                                                                               flag,
                                                                                                               0)])
    else
        risk_frontier = shared_get(model, :risk_frontier)
        push!(risk_frontier, (bound_var_key, bound_key) => (r_expr, ub, flag, 0))
    end
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Number, key,
                               flag::Bool = true)
    k = get_k(model)
    sc = get_constraint_scale(model)
    d = ifelse(flag, 1, -1)
    state_set!(model, key, :_ub, JuMP.@constraint(model, d * sc * (r_expr - ub * k) <= 0))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Push a scaled risk expression onto the `risk_vec` array in `model`.

If `rke` is `false` the function does nothing. Otherwise it initialises `risk_vec` if needed
and appends `scale * r_expr`.

# Arguments

  - $(arg_dict[:model])
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk JuMP expression to add.
  - `scale::Number`: Scaling factor applied to the expression.
  - `rke::Bool`: When `false` this method is a no-op.

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_expression!(model::JuMP.Model, r_expr::JuMP.AbstractJuMPScalar,
                              scale::Number, rke::Bool)
    if !rke
        return nothing
    end
    if !shared_has(model, :risk_vec)
        JuMP.@expression(model, risk_vec, Union{JuMP.AffExpr, JuMP.QuadExpr}[])
    end
    risk_vec = shared_get(model, :risk_vec)
    push!(risk_vec, scale * r_expr)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply an upper-bound constraint and register the risk expression for the objective.

Calls [`set_risk_upper_bound!`](@ref) with `settings.ub` and [`set_risk_expression!`](@ref)
with `settings.scale` and `settings.rke`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:opt_rjumpe])
  - `r_expr::JuMP.AbstractJuMPScalar`: Risk JuMP expression.
  - `settings::RiskMeasureSettings`: Settings carrying upper bound, scale, and `rke` flag.
  - `name`: Bare Model State entry name seeding the derived bound keys (`<name>_ub`,
    `<name>_ub_var`). The key is resolved here rather than at the call site, so emitters
    never build a key by hand (ADR 0037).
  - `i`: Measure index, for per-measure entry names. The indexed method resolves the same
    key the emitter registered the risk expression under, so the bound keys and the entry
    key cannot drift apart.
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound (default: `true`).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace for `name` (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`state_key`](@ref)
"""
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::RiskJuMPOptimisationEstimator,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, name::Symbol,
                                         flag::Bool = true; prefix::Symbol = Symbol(""))
    key = state_key(prefix, name)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key, flag)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::RiskJuMPOptimisationEstimator,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, name::Symbol, i,
                                         flag::Bool = true; prefix::Symbol = Symbol(""))
    key = state_key(prefix, name, i)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key, flag)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add portfolio drawdown tracking variables and constraints to `model`.

Creates the `dd` variable array (length `T + 1`) together with three constraints:
`cdd_start` (initial drawdown is zero), `cdd_geq_0` (drawdowns are non-negative), and `cdd`
(drawdown recurrence relation). Returns the `dd` array; returns the existing one if already
present in `model`.

# Mathematical definition

Drawdown recurrence:

```math
\\begin{align}
dd_0 &= 0\\,, \\\\
dd_t &\\geq 0\\,, \\\\
dd_t &\\geq dd_{t-1} - \\hat{r}_t
\\quad \\Leftrightarrow \\quad dd_t &= \\max_{s \\leq t} V_s - V_t\\,.
\\end{align}
```

Where:

  - ``dd_t``: Portfolio drawdown at time ``t``.
  - ``\\hat{r}_t``: Portfolio return at time ``t``.
  - ``V_t``: Cumulative portfolio wealth at time ``t``.

where ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` and ``V_t = k + \\sum_{s=1}^t \\hat{r}_s``.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Returns

  - `dd`: JuMP variable array of length `T + 1` tracking portfolio drawdowns.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function set_drawdown_constraints!(model::JuMP.Model, X::MatNum;
                                   prefix::Symbol = Symbol(""))
    return state_build!(model, prefix, :dd) do
        sc = get_constraint_scale(model)
        net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
        T = length(net_X)
        dd = JuMP.@variable(model, [1:(T + 1)])
        state_set!(model, prefix, :cdd_start, JuMP.@constraint(model, sc * dd[1] == 0))
        state_set!(model, prefix, :cdd_geq_0,
                   JuMP.@constraint(model, sc * view(dd, 2:(T + 1)) >= 0))
        state_set!(model, prefix, :cdd,
                   JuMP.@constraint(model,
                                    sc * (net_X + view(dd, 2:(T + 1)) - view(dd, 1:T)) >= 0))
        return dd
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a range risk measure's constraints to `model` by building its two tails and summing them.

A range measure is its base measure applied twice, so this builds it that way. It reads the
two point measures from [`range_tails`](@ref), calls [`set_risk_constraints!`](@ref) on the
loss tail with `loss = true` and on the gain tail with `loss = false`, and registers the sum
under `key`. The gain tail sees the *negated* net portfolio returns, which is the whole of
what "the other tail" means and is what each base builder's `loss` keyword does.

The two tails carry `rke = false` and no upper bound, so only the composite expression
reaches the objective and the bound. Each tail builds under its own measure index, composed
by [`nested_index`](@ref), so a two-tail model names its parts by the side they describe and
a range nested in a range stays collision-free.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The range risk measure.
  - `name::Symbol`: Bare Model State entry name for the composite range expression.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`range_tails`](@ref)
  - [`nested_index`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_range_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                     name::Symbol, opt::RiskJuMPOptimisationEstimator,
                                     pr::AbstractPriorResult, args...;
                                     prefix::Symbol = Symbol(""), kwargs...)
    (; loss, gain) = range_tails(r)
    loss_risk = set_risk_constraints!(model, nested_index(:loss_, i), loss, opt, pr,
                                      args...; loss = true, prefix = prefix, kwargs...)
    gain_risk = set_risk_constraints!(model, nested_index(:gain_, i), gain, opt, pr,
                                      args...; loss = false, prefix = prefix, kwargs...)
    range_risk = state_set!(model, prefix, name, i,
                            JuMP.@expression(model, loss_risk + gain_risk))
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, name, i;
                                    prefix = prefix)
    return range_risk
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the series a conic risk measure reduces.

A conic tail measure is written once against a per-observation series of *returns*. The
returns family reduces the net portfolio returns; the drawdown family reduces the negated
drawdown path, which is the same series with one substitution. [`risk_series`](@ref) is the
one place that substitution is made, so a builder is written once and each twin selects its
series by passing the marker.

# Related

  - [`NetReturnsRiskSeries`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`risk_series`](@ref)
"""
abstract type AbstractRiskSeriesAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Marker selecting the net portfolio returns as the series a risk measure reduces.

This is the series of every returns-tail measure, and the only one that can be
range-composed: the gain tail is the same series negated, which is what `loss = false` means
in [`risk_series`](@ref).

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`risk_series`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
"""
struct NetReturnsRiskSeries <: AbstractRiskSeriesAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Marker selecting the negated drawdown path as the series a risk measure reduces.

`-dd[2:T+1]` is the return-signed drawdown series, so a builder written against net returns
encodes the drawdown twin without a single sign written by hand.

The drawdown series has no gain tail. A run-up is a different recurrence, not the negation
of this one, so [`risk_series`](@ref) takes no `loss` keyword for this marker and no drawdown
measure can be range-composed.

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`NetReturnsRiskSeries`](@ref)
  - [`risk_series`](@ref)
  - [`set_drawdown_constraints!`](@ref)
"""
struct DrawdownRiskSeries <: AbstractRiskSeriesAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the per-observation series a conic risk measure reduces, and its length.

The returns twin and the drawdown twin of a conic tail measure are one programme under the
substitution `net_X -> -dd[2:T+1]`. This function is the one place that substitution is
written, so each builder takes `(series, T)` and encodes both twins.

The series is signed as a *return*: a loss is a negative entry, on both markers. That is why
the drawdown branch negates — `dd` is a non-negative loss path — and it is what lets one
builder body serve both.

# Arguments

  - $(arg_dict[:model])
  - `alg::NetReturnsRiskSeries`: Reduce the net portfolio returns.
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `loss::Bool`: `true` builds the loss tail on the net portfolio returns, `false` the gain
    tail on their negation.
  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `series`: The per-observation return series, length `T`.
  - `T::Int`: The number of observations.

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function risk_series(model::JuMP.Model, ::NetReturnsRiskSeries, pr::AbstractPriorResult;
                     loss::Bool = true, prefix::Symbol = Symbol(""))
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    return net_X, length(net_X)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the negated drawdown path a drawdown risk measure reduces, and its length.

Registers the drawdown variables through [`set_drawdown_constraints!`](@ref), then returns
`-dd[2:T+1]`. The negation is what makes the drawdown path a *return* series, so a builder
written against net portfolio returns encodes the drawdown twin unchanged.

There is no `loss` keyword. A drawdown has no gain tail, so a caller that tries to compose a
range from this series fails at the call site rather than silently building the loss tail
twice.

# Arguments

  - $(arg_dict[:model])
  - `alg::DrawdownRiskSeries`: Reduce the negated drawdown path.
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `series`: The negated drawdown path `-dd[2:T+1]`, length `T`.
  - `T::Int`: The number of observations.

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`NetReturnsRiskSeries`](@ref)
  - [`set_drawdown_constraints!`](@ref)
"""
function risk_series(model::JuMP.Model, ::DrawdownRiskSeries, pr::AbstractPriorResult;
                     prefix::Symbol = Symbol(""))
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    return -view(dd, 2:(T + 1)), T
end
