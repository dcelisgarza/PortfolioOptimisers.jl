"""
    const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}

Groups the JuMP optimisers that do not use factor risk contribution.

[`FactorRiskContribution`](@ref) is the one JuMP optimiser outside the group. Two methods dispatch on this alias. The frontier method of [`set_risk_upper_bound!`](@ref) records a [`Frontier`](@ref) bound, and [`risk_contribution_constraints`](@ref) writes the risk contribution rows of a [`Variance`](@ref). A factor risk contribution optimiser reaches neither method.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RiskBoundOwner`](@ref)
"""
const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}
"""
    const RiskConstraintOwner = Union{<:RiskJuMPOptimisationEstimator, <:AbstractProgrammeAllocationSet}

Groups the owners of the risk constraints that a builder writes. An owner is a JuMP optimiser, whose programme the constraint joins, or a programme Allocation Set, whose projection the constraint joins.

Every `set_risk_constraints!` and `set_risk!` method takes the owner in its `opt` slot. A builder reads the owner through three methods and no others. [`risk_constraint_solver`](@ref) gives the solver that resolves a Deferred Quantity. [`risk_contribution_constraints`](@ref) gives the risk contribution rows of a [`Variance`](@ref). [`set_risk_upper_bound!`](@ref) writes the bound on the risk expression. A [`RiskBoundOwner`](@ref) honours the bound, and a [`FactorRiskContribution`](@ref) ignores it with a warning.

# Related

  - [`RiskBoundOwner`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`AbstractProgrammeAllocationSet`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
const RiskConstraintOwner = Union{<:RiskJuMPOptimisationEstimator,
                                  <:AbstractProgrammeAllocationSet}
"""
    const RiskBoundOwner = Union{<:NonFRCJuMPOpt, <:AbstractProgrammeAllocationSet}

Groups the owners that honour the `settings.ub` of a risk measure.

The group holds every JuMP optimiser except [`FactorRiskContribution`](@ref), and the programme Allocation Set. On a programme Allocation Set, `ub` is the ceiling of its projection.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`NonFRCJuMPOpt`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
const RiskBoundOwner = Union{<:NonFRCJuMPOpt, <:AbstractProgrammeAllocationSet}
"""
    risk_constraint_solver(opt::JuMPOptimisationEstimator)

Returns the solver against which a risk measure builder resolves a Deferred Quantity.

The method for a JuMP optimiser returns the solver of the optimiser, `opt.opt.slv`. A programme Allocation Set has a method of its own, which [`AbstractProgrammeAllocationSet`](@ref) states.

# Arguments

  - `opt`: The JuMP optimiser that owns the constraint.

# Returns

  - `slv`: The solver, or the vector of solvers, of the optimiser.

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function risk_constraint_solver(opt::JuMPOptimisationEstimator)
    return opt.opt.slv
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reduces the entries of `model[:risk_vec]` to the one risk quantity `model[:risk]` that the objective reads.

The scalariser `sca` selects the reduction. Under the sum, `risk` is an expression. Under the log-sum-exp and the maximum, `risk` is a free variable that rows bound from below.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}_{\\mathrm{sum}} &= \\sum_{i=1}^{n} \\mathcal{R}_i\\,, \\\\
\\mathcal{R}_{\\mathrm{lse}} &= \\frac{1}{\\gamma} \\ln \\sum_{i=1}^{n} e^{\\gamma \\mathcal{R}_i}\\,, \\\\
\\mathcal{R}_{\\mathrm{max}} &= \\max_{1 \\leq i \\leq n} \\mathcal{R}_i\\,.
\\end{align}
```

Where:

  - ``\\mathcal{R}_{\\mathrm{sum}}``, ``\\mathcal{R}_{\\mathrm{lse}}``, ``\\mathcal{R}_{\\mathrm{max}}``: The risk under the sum, the log-sum-exp and the maximum scalariser.
  - $(math_dict[:R_i_riskvec])
  - ``n``: Number of entries of `risk_vec`.
  - ``\\gamma > 0``: Smoothing parameter of the log-sum-exp, `sca.gamma`.

# Algorithm

 1. When the model holds no `risk_vec`, return `nothing` and register nothing. This happens when every risk measure sets `rke = false`.
 2. Read `risk_vec`.
 3. Register `risk` through the method that `sca` selects, as the rows below state.

# JuMP formulation

## Variables

  - `risk`: created by the log-sum-exp and the maximum methods, a free scalar.
  - `u_risk`: created by the log-sum-exp method, ``n \\times 1``.

## Expressions

  - `risk`: registered by the sum method, ``\\sum_{i=1}^{n} \\mathcal{R}_i``. It is quadratic when an entry of `risk_vec` is quadratic, and affine otherwise.

## Constraints

  - `u_risk_lse`: ``s_c \\left(\\sum_{i=1}^{n} u_i - 1\\right) \\leq 0``, registered by the log-sum-exp method.
  - `risk_lse`: ``\\left(s_c \\gamma (\\mathcal{R}_i - \\rho),\\, s_c,\\, s_c u_i\\right) \\in \\mathcal{K}_{\\mathrm{exp}}``, ``\\forall\\, i = 1,\\ldots,n``, registered by the log-sum-exp method.
  - `risk_ms`: ``\\rho - \\mathcal{R}_i \\geq 0``, ``\\forall\\, i = 1,\\ldots,n``, registered by the maximum method.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:R_i_riskvec])
  - ``\\rho``: The variable `risk`.
  - ``u_i``: Entry ``i`` of `u_risk`.
  - ``\\mathcal{K}_{\\mathrm{exp}} = \\mathrm{cl}\\,\\{(x, y, z) : y e^{x / y} \\leq z,\\, y > 0\\}``: Exponential cone.

## Relaxation

$(val_dict[:relax])

  - The sum method is exact.
  - The log-sum-exp rows give ``e^{\\gamma (\\mathcal{R}_i - \\rho)} \\leq u_i`` and ``\\sum_{i} u_i \\leq 1``, so ``\\rho \\geq \\mathcal{R}_{\\mathrm{lse}}``. The maximum rows give ``\\rho \\geq \\mathcal{R}_{\\mathrm{max}}``. In both cases `model[:risk]` lies above the scalarised risk.
  - The bound is tight when the objective pulls `risk` down, as a minimum risk objective does.
  - Under the maximum, the objective pulls on the largest entry alone. When a risk measure below the maximum is written through epigraph variables, its entry can lie above the risk that the measure gives. [`expected_risk`](@ref) gives the exact value.

# Arguments

  - $(arg_dict[:model])
  - `sca`: The scalariser, a [`SumScalariser`](@ref), a [`LogSumExpScalariser`](@ref) or a [`MaxScalariser`](@ref).

# Returns

  - `nothing`.

# Related

  - [`SumScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`MaxScalariser`](@ref)
  - [`set_risk_expression!`](@ref)
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

Returns the number of entries in the `:risk_frontier` Model State registry, or `0` when the model holds no registry.

[`set_resolved_risk_constraints!`](@ref) reads the length before a risk measure builds its constraints. The entries after that position are the entries that the measure adds.

# Arguments

  - $(arg_dict[:model])

# Returns

  - `n::Int`: The number of registry entries.

# Related

  - [`set_risk_frontier_owner!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function risk_frontier_length(model::JuMP.Model)::Int
    return shared_has(model, :risk_frontier) ? length(shared_get(model, :risk_frontier)) : 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Records `i` as the owner of every `:risk_frontier` entry after position `first`.

[`rebuild_risk_frontier`](@ref) resolves a [`Frontier`](@ref) bound into a span of numbers, and it needs the risk measure that owns the span. The position of an entry does not give its measure, for two reasons. A measure registers an entry only when its `settings.ub` is a [`Front_NumVec`](@ref). A memoised measure such as [`UlcerIndex`](@ref) registers one entry for each copy of itself in the measure vector. So each entry records its owner. The entries of `:ret_frontier` record their term index in the same way.

# Algorithm

 1. When the model holds no `:risk_frontier`, return `nothing`.
 2. For each entry `j` after position `first`, replace the fourth value of the entry, the owner, with `i`.

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

Builds the constraints of one risk measure, or of each measure in a vector, and gives each measure its index.

This route calls no [`factory`](@ref), so a measure that states no solver of its own still holds `nothing`. Both methods resolve the measure against the solver of the owner, [`risk_constraint_solver`](@ref). A Calibration Rule that reads the solver then sees the same solver here and on the `factory` route.

# Algorithm

 1. The method for one measure drops the `scale` of `r` with [`unit_scale_risk_measure`](@ref), and builds the measure at index `1`. `scale` weighs a measure against the other measures of a vector, so it has no effect on one measure alone.
 2. The method for a vector builds each measure `rs[i]` at index `i`, and keeps its `scale`.
 3. Each build calls [`set_resolved_risk_constraints!`](@ref), which resolves the measure, builds it and records it as the owner of the `:risk_frontier` entries that it adds.

# Arguments

  - $(arg_dict[:model])
  - `r`: A [`RiskMeasure`](@ref), or `rs`, a vector of risk measures.
  - `opt`: The owner of the constraint, a [`JuMPOptimisationEstimator`](@ref) or a programme Allocation Set. [`RiskConstraintOwner`](@ref) states the owners.
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
                               opt::Union{<:JuMPOptimisationEstimator,
                                          <:AbstractProgrammeAllocationSet},
                               pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                               fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing;
                               kwargs...)
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
    set_resolved_risk_constraints!(model, 1, unit_scale_risk_measure(r), opt, pr, pl, fees,
                                   b1; kwargs...)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, rs::VecRM,
                               opt::Union{<:JuMPOptimisationEstimator,
                                          <:AbstractProgrammeAllocationSet},
                               pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                               fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing;
                               kwargs...)
    for (i, r) in enumerate(rs)
        set_resolved_risk_constraints!(model, i, r, opt, pr, pl, fees, b1; kwargs...)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves one risk measure and builds its constraints at index `i`.

The two methods of [`set_risk_constraints!`](@ref) both call this function, so a measure resolves the same way alone and in a vector.

# Algorithm

 1. Read `first`, the length of the `:risk_frontier` registry, with [`risk_frontier_length`](@ref).
 2. Resolve the Deferred Quantities of `r` against the solver of the owner, giving `resolved`. [`resolve_deferred_quantities`](@ref) does the work, and [`risk_constraint_solver`](@ref) gives the solver.
 3. Refuse a Calibration Rule that `resolved` still holds, with [`assert_declared_calibration_resolver`](@ref).
 4. Build the constraints of `resolved` at index `i` with the index method of `set_risk_constraints!`.
 5. Record `i` as the owner of the registry entries after `first`, with [`set_risk_frontier_owner!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: The position of the measure in the risk measure vector.
  - `r::RiskMeasure`: The risk measure.
  - `opt`: The owner of the constraint, a [`RiskConstraintOwner`](@ref).
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])
  - $(arg_dict[:b1_opt])

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
"""
function set_resolved_risk_constraints!(model::JuMP.Model, i::Integer, r::RiskMeasure,
                                        opt::Union{<:JuMPOptimisationEstimator,
                                                   <:AbstractProgrammeAllocationSet},
                                        pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                                        fees::Option{<:Fees}, b1::Option{<:MatNum};
                                        kwargs...)
    first = risk_frontier_length(model)
    resolved = resolve_deferred_quantities(r, pr, risk_constraint_solver(opt))
    assert_declared_calibration_resolver(resolved)
    set_risk_constraints!(model, i, resolved, opt, pr, pl, fees, b1; kwargs...)
    set_risk_frontier_owner!(model, first, i)
    return nothing
end
"""
    prior_high_order_quantity(pr::AbstractPriorResult, key::Symbol)

Returns the high-order quantity that `key` names on the prior result `pr`, or `nothing` when `pr` carries none.

A [`HighOrderPrior`](@ref) has the fields `kt`, `sk`, `V` and the three vectorisation matrices. A [`LowOrderPrior`](@ref) has none of them. A [`HighOrderPrior`](@ref) fitted with one of the two tensors holds `nothing` in the field of the other. The function returns `nothing` in both cases, and never raises.

# Arguments

  - $(arg_dict[:pr])
  - `key::Symbol`: The name of the quantity, for example `:kt`, `:sk`, `:V`, `:D2`, `:L2` or `:S2`.

# Returns

  - `q`: The quantity, or `nothing`.

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

Refuses a high-order risk measure when neither the measure nor the prior carries its tensor.

The builder can build the measure when either side gives the quantity, so the check reads the pair and not the type of `pr`. A measure that states how to build its own cokurtosis passes with a [`LowOrderPrior`](@ref). A [`HighOrderPrior`](@ref) that computed neither tensor fails.

# Arguments

  - `q`: The value in the slot `key` of the measure, after [`resolve_deferred_quantities`](@ref).
  - $(arg_dict[:pr])
  - `rm::Symbol`: The name of the risk measure type, for the message.
  - `key::Symbol`: The name of the quantity, for example `:kt`.
  - `est::Symbol`: The name of the estimator slot of the measure, for the message.

# Validation

  - Raises `ArgumentError` when `q` is `nothing` and [`prior_high_order_quantity`](@ref) finds no `key` on `pr`. The message gives three remedies. State the quantity on the measure, or name an `est` or an [`AbstractPriorEstimator`](@ref) in the measure, or give the optimiser a prior estimator that computes the quantity.

# Returns

  - `nothing`.

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

Returns the duplication, elimination and summation matrices for `N` assets, from the prior result `pr` when it carries them.

[`dup_elim_sum_matrices`](@ref) depends on the asset count alone, so a rebuilt matrix equals the matrix that a [`HighOrderPrior`](@ref) carries. A measure that holds its own tensor can therefore build against a [`LowOrderPrior`](@ref), because the builder reads nothing else from the prior.

# Algorithm

 1. Read `D2`, `L2` and `S2` from `pr` with [`prior_high_order_quantity`](@ref).
 2. When any of the three is `nothing`, build all three with [`dup_elim_sum_matrices`](@ref), and replace each `nothing` with its rebuilt matrix. One call builds all three, so the cost does not depend on how many are missing.

# Arguments

  - $(arg_dict[:pr])
  - `N::Integer`: The number of assets.

# Returns

  - `(D2, L2, S2)`: The duplication, elimination and summation matrices.

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

Bounds the risk expression `r_expr` by `ub`, or records `ub` for a frontier solve.

The type of `ub` and the owner `opt` select the method.

  - A `nothing` bound adds nothing.
  - A number adds one row. The row bounds `r_expr` from above when `flag` is `true`, and from below when it is `false`.
  - A [`Front_NumVec`](@ref) adds an entry to the `:risk_frontier` registry, and the frontier solve reads the entry later. The owner of the entry is written as `0`, because the measure is not known here. [`set_risk_frontier_owner!`](@ref) writes the owner afterwards.

Under a ratio objective the head divides the weights by ``k`` after the solve, so the row bounds the risk of the weights that the head returns. That is exact for a risk expression of degree one in the weights. A builder whose risk expression has a higher degree passes a separate expression of degree one, as [`set_variance_risk_bounds_and_expression!`](@ref) does.

# JuMP formulation

## Expressions

  - `risk_frontier`: registered by the frontier method on its first call. It is a vector of entries `(<key>_ub_var, <key>_ub) => (R, ub, flag, 0)`, and each later call appends one entry.

## Constraints

  - `<key>_ub`: ``s_c \\left(R(\\boldsymbol{w}) - u k\\right) \\leq 0`` when `flag` is `true`, and ``s_c \\left(u k - R(\\boldsymbol{w})\\right) \\leq 0`` when it is `false`, registered by the number method.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - $(math_dict[:R_w])
  - ``u``: The bound, `ub`.

# Arguments

  - $(arg_dict[:model])
  - `opt`: The owner of the constraint, a [`RiskConstraintOwner`](@ref).
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk expression to bound.
  - `ub`: The bound. It is `nothing`, a number, or a [`Front_NumVec`](@ref), which is a vector of bounds or a [`Frontier`](@ref).
  - `key::Symbol`: The Model State key of the risk expression. The row and the frontier entry take their keys from it.
  - `flag::Bool`: `true` for an upper bound and `false` for a lower bound (default: `true`).

# Validation

  - An owner outside [`RiskBoundOwner`](@ref), a [`FactorRiskContribution`](@ref), ignores a bound that is not `nothing`, and the method warns. Without the warning the bound has no effect and nothing says so.
  - A programme Allocation Set raises `ArgumentError` for a [`Front_NumVec`](@ref) bound. Its ceiling is one number. The constructor of the set refuses such a bound first, and this method refuses a set that reaches the builder without that check.

# Returns

  - `nothing`.

# Related

  - [`RiskBoundOwner`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`rebuild_risk_frontier`](@ref)
"""
function set_risk_upper_bound!(::JuMP.Model, ::RiskConstraintOwner, r_expr, ::Nothing, key,
                               flag::Bool = true)
    return nothing
end
function set_risk_upper_bound!(::JuMP.Model, opt::JuMPOptimisationEstimator, r_expr, ub,
                               key, flag::Bool = true)
    return @warn("Risk upper bound `settings.ub = $ub` ($key) is not supported by `$(typeof(opt).name.name)` and would be silently ignored. Remove `ub` from the risk measure settings, or use an optimiser that supports risk upper bounds (`MeanRisk`, `NearOptimalCentering`, `RiskBudgeting`).")
end
function set_risk_upper_bound!(::JuMP.Model, set::AbstractProgrammeAllocationSet, r_expr,
                               ub::Front_NumVec, key, flag::Bool = true)
    return throw(ArgumentError("the ceiling of a risk measure on a `$(nameof(typeof(set)))` is one number, its `settings.ub`; got `$ub` ($key), a frontier or a per-asset vector, which a one-step projection cannot bound."))
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
function set_risk_upper_bound!(model::JuMP.Model, ::RiskBoundOwner,
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

Appends the risk expression `r_expr`, times `scale`, to `model[:risk_vec]`.

[`scalarise_risk_expression!`](@ref) reduces the entries of `risk_vec` to the risk that the objective reads. When `rke` is `false`, the function adds nothing, and the measure reaches the model through its bound alone.

# JuMP formulation

## Expressions

  - `risk_vec`: registered on the first call as an empty vector of affine and quadratic expressions. Each call with `rke = true` appends ``\\mathcal{R}_i = \\omega R(\\boldsymbol{w})``.

Where:

  - $(math_dict[:R_i_riskvec])
  - $(math_dict[:R_w])
  - ``\\omega``: The weight of the measure, `scale`.

# Arguments

  - $(arg_dict[:model])
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk expression to add.
  - `scale::Number`: The weight of the measure.
  - `rke::Bool`: `false` adds nothing.

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

Bounds a risk expression by `settings.ub`, and adds it to the risk that the objective reads.

Every risk measure builder ends with this call, or with a sibling that takes a separate bound expression.

# Algorithm

 1. Compose `key` from `prefix`, `name` and, in the index method, `i`, with [`state_key`](@ref). The builder registered its risk expression under the same key, so the bound keys cannot differ from it.
 2. Bound `r_expr` by `settings.ub` with [`set_risk_upper_bound!`](@ref), in the direction that `flag` gives.
 3. Append `r_expr` times `settings.scale` to `risk_vec` with [`set_risk_expression!`](@ref), when `settings.rke` is `true`.

# Arguments

  - $(arg_dict[:model])
  - `opt`: The owner of the constraint, a [`RiskConstraintOwner`](@ref).
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk expression.
  - `settings::RiskMeasureSettings`: The settings of the measure, which give the bound, the weight and `rke`.
  - `name::Symbol`: The Model State name of the risk expression. The bound keys `<name>_ub` and `<name>_ub_var` derive from it.
  - `i`: The index of the measure, in the index method.
  - `flag::Bool`: `true` for an upper bound and `false` for a lower bound (default: `true`).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace for `name` (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`state_key`](@ref)
"""
function set_risk_bounds_and_expression!(model::JuMP.Model, opt::RiskConstraintOwner,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, name::Symbol,
                                         flag::Bool = true; prefix::Symbol = Symbol(""))
    key = state_key(prefix, name)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key, flag)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_risk_bounds_and_expression!(model::JuMP.Model, opt::RiskConstraintOwner,
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

Adds the drawdown variables of the portfolio to `model`, and returns them.

The function builds the variables once for each `prefix`. A second call returns the variables that the first call registered.

# Mathematical definition

The absolute drawdown of the net portfolio returns is

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} \\hat{r}_s\\,, \\quad c_0 = 0\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,.
\\end{align}
```

It satisfies the recurrence ``-d_0 = 0`` and ``-d_t = \\max(0,\\, -d_{t-1} - \\hat{r}_t)``.

Where:

  - $(math_dict[:rhat_t_net])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

# Algorithm

 1. Read `sc`, the constraint scale.
 2. Read `net_X`, the net portfolio returns, with [`set_net_portfolio_returns!`](@ref), and set `T` to its length.
 3. Create `dd`, ``T + 1`` free variables. Entry `dd[t + 1]` is ``dd_t``.
 4. Register the three rows below.
 5. Register `dd` under the Model State key `<prefix>dd`, and return it.

# JuMP formulation

## Variables

  - `dd`: created, ``(T + 1) \\times 1``, and registered under the Model State key `<prefix>dd`.

## Constraints

  - `cdd_start`: ``s_c\\, dd_0 = 0``.
  - `cdd_geq_0`: ``s_c\\, dd_t \\geq 0``, ``\\forall\\, t = 1,\\ldots,T``.
  - `cdd`: ``s_c \\left(\\hat{r}_t + dd_t - dd_{t-1}\\right) \\geq 0``, ``\\forall\\, t = 1,\\ldots,T``.

Each name is a Model State key, and a caller reads the row back under `prefix` followed by the name.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:dd_t_model])
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:T])

## Relaxation

$(val_dict[:relax])

  - The rows give ``dd_t \\geq \\max(0,\\, dd_{t-1} - \\hat{r}_t)``, so by induction ``dd_t \\geq -d_t`` for every ``t``. Each entry of `dd` lies at or above the drawdown.
  - The bound on an entry is tight when the objective pulls that entry down. A sum over the entries, as in the average drawdown, pulls on every entry. A maximum over the entries, as in the maximum drawdown, pulls on the largest entry alone, and the other entries can lie above their drawdowns.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: The asset returns matrix, ``T \\times N``.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `dd`: The drawdown variables, ``(T + 1) \\times 1``.

# Related

  - [`risk_series`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
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

Adds the constraints of a range risk measure to `model`, as the sum of its loss tail and its gain tail.

The two tails from [`range_tails`](@ref) carry `rke = false` and no bound, so the sum alone reaches the objective and the bound. Each tail builds under its own index, which [`nested_index`](@ref) composes from the side and `i`. The keys of the two tails therefore differ, also when a range holds a range.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}_{\\mathrm{range}}(\\boldsymbol{w}) &= \\mathcal{R}_{\\mathrm{loss}}(\\hat{\\boldsymbol{r}}) + \\mathcal{R}_{\\mathrm{gain}}(-\\hat{\\boldsymbol{r}})\\,.
\\end{align}
```

Where:

  - ``\\mathcal{R}_{\\mathrm{range}}(\\boldsymbol{w})``: The range risk of the portfolio.
  - ``\\mathcal{R}_{\\mathrm{loss}}``, ``\\mathcal{R}_{\\mathrm{gain}}``: The loss tail and the gain tail of the range measure, each a risk measure of a return series.
  - ``\\hat{\\boldsymbol{r}}``: The net portfolio returns, ``T \\times 1``, with entries ``\\hat{r}_t``.
  - $(math_dict[:rhat_t_net])

# Algorithm

 1. Read the tails `loss` and `gain` of `r` with [`range_tails`](@ref).
 2. Build `loss` at index `nested_index(:loss_, i)` with `loss = true`, giving `loss_risk`.
 3. Build `gain` at index `nested_index(:gain_, i)` with `loss = false`, giving `gain_risk`. The builder then reads the negated net portfolio returns.
 4. Register `range_risk = loss_risk + gain_risk` under the Model State key of `name` and `i`.
 5. Bound `range_risk` and add it to `risk_vec` with [`set_risk_bounds_and_expression!`](@ref), from `r.settings`.

# JuMP formulation

## Expressions

  - `<prefix><name><i>`: ``\\mathcal{R}_{\\mathrm{loss}} + \\mathcal{R}_{\\mathrm{gain}}``, registered as a Model State entry.

Where:

  - ``\\mathcal{R}_{\\mathrm{loss}}``, ``\\mathcal{R}_{\\mathrm{gain}}``: The risk expressions that the two tail builds return.

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
                                     name::Symbol, opt::RiskConstraintOwner,
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

Abstract supertype for the series that a conic risk measure reduces.

A builder of a conic tail measure reads a series of returns, one entry for each observation. The returns family reads the net portfolio returns. The drawdown family reads the negated drawdown path, which is a series of returns too. [`risk_series`](@ref) makes the series from the marker, so one builder body serves both families.

# Related

  - [`NetReturnsRiskSeries`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`risk_series`](@ref)
"""
abstract type AbstractRiskSeriesAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the net portfolio returns as the series that a risk measure reduces.

Every tail measure of returns reads this series. A range measure can read it alone, because its gain tail reads the same series negated, which `loss = false` gives in [`risk_series`](@ref).

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`DrawdownRiskSeries`](@ref)
  - [`risk_series`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
"""
struct NetReturnsRiskSeries <: AbstractRiskSeriesAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Selects the negated drawdown path as the series that a risk measure reduces.

`-dd[2:T+1]` carries the sign of a return, so a builder of a measure of returns builds the drawdown measure with no change of sign.

The drawdown series has no gain tail. A run-up follows a different recurrence and is not the negation of the drawdown. So [`risk_series`](@ref) takes no `loss` keyword for this marker, and no range measure reads a drawdown series.

# Related

  - [`AbstractRiskSeriesAlgorithm`](@ref)
  - [`NetReturnsRiskSeries`](@ref)
  - [`risk_series`](@ref)
  - [`set_drawdown_constraints!`](@ref)
"""
struct DrawdownRiskSeries <: AbstractRiskSeriesAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the series of net portfolio returns that a conic risk measure reduces, and its length.

The measure of returns and the measure of drawdowns are one programme under the substitution of `-dd[2:T+1]` for `net_X`. The two methods of this function make that substitution, so each builder takes `(series, T)` and builds both measures. On both markers the series carries the sign of a return, and a loss is a negative entry.

# Mathematical definition

```math
\\begin{align}
s_t &= \\begin{cases} \\hat{r}_t & \\text{for the loss tail}\\,, \\\\ -\\hat{r}_t & \\text{for the gain tail}\\,. \\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:s_t_series])
  - $(math_dict[:rhat_t_net])

# Algorithm

 1. Read `net_X`, the net portfolio returns, with [`set_net_portfolio_returns!`](@ref).
 2. Negate `net_X` when `loss` is `false`.
 3. Return `net_X` and its length.

# Arguments

  - $(arg_dict[:model])
  - `alg::NetReturnsRiskSeries`: The marker that selects the net portfolio returns.
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `loss::Bool`: `true` gives the series of the loss tail, the net portfolio returns. `false` gives the series of the gain tail, their negation (default: `true`).
  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `series`: The series ``s_t``, ``T \\times 1``.
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

Returns the negated drawdown path that a drawdown risk measure reduces, and its length.

The negation gives the drawdown path the sign of a return, so a builder of a measure of returns builds the drawdown measure with no change.

The method takes no `loss` keyword, because a drawdown has no gain tail. A call that asks for the gain tail raises a `MethodError`, and does not build the loss tail a second time.

# Mathematical definition

```math
\\begin{align}
s_t &= -dd_t\\,, \\quad t = 1,\\ldots,T\\,.
\\end{align}
```

Where:

  - $(math_dict[:s_t_series])
  - $(math_dict[:dd_t_model])
  - $(math_dict[:T])

# Algorithm

 1. Build `dd` with [`set_drawdown_constraints!`](@ref), or read it when the model holds it.
 2. Set `T = length(dd) - 1`.
 3. Return `-dd[2:T+1]` and `T`.

# Arguments

  - $(arg_dict[:model])
  - `alg::DrawdownRiskSeries`: The marker that selects the negated drawdown path.
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `series`: The negated drawdown path `-dd[2:T+1]`, ``T \\times 1``.
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
