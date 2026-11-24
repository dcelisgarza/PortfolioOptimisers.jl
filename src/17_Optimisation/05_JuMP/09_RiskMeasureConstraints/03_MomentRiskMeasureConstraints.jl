"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the target return used as the reference level for lower/upper moment risk constraints.

Dispatches on the type of `r.mu`: when `nothing`, uses the prior mean vector `mu`; when a
`VecNum`, uses `r.mu`; when a `VecScalar`, combines the vector and scalar parts with `k`;
when a scalar `Number`, scales it by `k`.

A per-asset target (the first three) is a gross expected return, and the series it centres is
net of the fee. So [`add_fees_to_ret!`](@ref) subtracts the mean fee per period from it, the
same charge the net expected return carries, and the target is the net mean. A per period fee
then cancels from every deviation, as [`moment_target_fees`](@ref) makes it cancel at the value
level. A scalar target is a threshold on the net series, so it takes no fee.

# Arguments

  - $(arg_dict[:model])
  - `r::LoHiOrderMoment`: Risk measure carrying the target specification.
  - `w`: Portfolio weight vector.
  - `mu::VecNum`: Prior mean return vector.
  - `k`: Leverage/scale variable from the model.

# Returns

  - The scalar target return used as the lower moment reference.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`add_fees_to_ret!`](@ref)
  - [`moment_target_fees`](@ref)
"""
function calc_risk_constraint_target(model::JuMP.Model,
                                     ::LoHiOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::VecNum, mu::VecNum, args...)
    tgt = LinearAlgebra.dot(w, mu)
    add_fees_to_ret!(model, tgt, true)
    return tgt
end
function calc_risk_constraint_target(model::JuMP.Model,
                                     r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                                     w::VecNum, args...)
    tgt = LinearAlgebra.dot(w, r.mu)
    add_fees_to_ret!(model, tgt, true)
    return tgt
end
function calc_risk_constraint_target(model::JuMP.Model,
                                     r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     w::VecNum, ::Any, k)
    tgt = LinearAlgebra.dot(w, r.mu.v) + r.mu.s * k
    add_fees_to_ret!(model, tgt, true)
    return tgt
end
function calc_risk_constraint_target(::JuMP.Model,
                                     r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                     ::Any, ::Any, k)
    return r.mu * k
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add first lower moment, mean absolute deviation, or second moment risk constraints to `model`.

Each overload introduces auxiliary non-negative variables (semi-deviations or lower
exceedances) for `T` observations, computes an observation-weighted mean, and adds an
inequality constraint linking the auxiliary variables to the portfolio returns minus the
target. The second-moment overload additionally supports full and lower-half formulations and
multiple variance encodings via [`set_second_moment_risk!`](@ref).

# Mathematical definition

First lower moment / semi-deviation:

```math
\\begin{align}
\\mathrm{FLM}(\\boldsymbol{w}) &= \\frac{1}{T}\\sum_{t=1}^T z_t\\,, \\\\
z_t &\\geq \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\bar{F}(\\boldsymbol{w}) - \\hat{r}_t,\\quad z_t \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{FLM}(\\boldsymbol{w})``: First lower moment.
  - $(math_dict[:T])
  - ``z_t \\geq 0``: Auxiliary variables capturing deviations below the mean.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - ``\\bar{F}(\\boldsymbol{w})``: Mean fee per period, subtracted by [`calc_risk_constraint_target`](@ref) so that the target is the net mean. Zero without a fee.
  - ``\\hat{r}_t``: Net portfolio return at time ``t``, ``\\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` less the fee of the period.

Mean absolute deviation:

```math
\\begin{align}
\\mathrm{MAD}(\\boldsymbol{w}) &= \\frac{2}{T}\\sum_{t=1}^T z_t + \\frac{1}{T}\\sum_{t=1}^T \\hat{r}_t - \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + \\bar{F}(\\boldsymbol{w})\\,, \\\\
z_t &\\geq \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\bar{F}(\\boldsymbol{w}) - \\hat{r}_t,\\quad z_t \\geq 0\\,.
\\end{align}
```

The row is one-sided, and the identity ``\\lvert d \\rvert = 2 \\max(-d, 0) + d`` gives the absolute deviation from it. The mean of the second term is the gap between the mean net return and the target, so the expression is exact for every target, not only for the mean.

Where:

  - ``\\mathrm{MAD}(\\boldsymbol{w})``: Mean absolute deviation.
  - $(math_dict[:T])
  - ``z_t \\geq 0``: Auxiliary variables capturing the deviations below the target.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - ``\\bar{F}(\\boldsymbol{w})``: Mean fee per period, subtracted by [`calc_risk_constraint_target`](@ref) so that the target is the net mean. Zero without a fee.
  - ``\\hat{r}_t``: Net portfolio return at time ``t``, ``\\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` less the fee of the period.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment`: Risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result containing `X` (returns matrix) and `mu`.

# Returns

  - The risk expression: `flm_risk_` under [`FirstLowerMoment`](@ref), `mad_risk_` under [`MeanAbsoluteDeviation`](@ref), and `second_moment_risk_` under [`SecondMoment`](@ref).

# Related

  - [`calc_risk_constraint_target`](@ref)
  - [`set_second_moment_risk!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    k = get_k(model)
    tgt = calc_risk_constraint_target(model, r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    flm = state_set!(model, prefix, :flm_, i, JuMP.@variable(model, [1:T], lower_bound = 0))
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = checked_observation_weights(wi, net_X)
    flm_risk = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(flm))
    else
        JuMP.@expression(model, Statistics.mean(flm, wi))
    end
    state_set!(model, prefix, :flm_risk_, i, flm_risk)
    state_set!(model, prefix, :cflm_mar_, i,
               JuMP.@constraint(model, sc * ((net_X + flm) .- tgt) >= 0))
    set_risk_bounds_and_expression!(model, opt, flm_risk, r.settings, :flm_risk_, i;
                                    prefix = prefix)
    return flm_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `LowOrderMoment` with `MeanAbsoluteDeviation` semi-deviation
to `model`.

Introduces auxiliary `mad` variables and adds constraints encoding the mean absolute
deviation of portfolio returns relative to the target benchmark. Registers the risk
expression and upper-bound constraint.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation}`: The MAD risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `mad_risk`: The risk expression.

# Related

  - [`LowOrderMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:MeanAbsoluteDeviation},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    k = get_k(model)
    tgt = calc_risk_constraint_target(model, r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    mad = state_set!(model, prefix, :mad_, i, JuMP.@variable(model, [1:T], lower_bound = 0))
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = checked_observation_weights(wi, net_X)
    # |x - tgt| = 2 max(tgt - x, 0) + (x - tgt), so the mean of the second term restores the
    # deviation that the one-sided row leaves out when `tgt` is not the mean of `net_X`.
    mad_risk = if isnothing(wi)
        JuMP.@expression(model, 2 * Statistics.mean(mad) + Statistics.mean(net_X) - tgt)
    else
        JuMP.@expression(model,
                         2 * Statistics.mean(mad, wi) + Statistics.mean(net_X, wi) - tgt)
    end
    state_set!(model, prefix, :mad_risk_, i, mad_risk)
    state_set!(model, prefix, :cmar_mad_, i,
               JuMP.@constraint(model, sc * ((net_X + mad) .- tgt) >= 0))
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, :mad_risk_, i;
                                    prefix = prefix)
    return mad_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the second-moment risk JuMP expression in one of four encodings.

The `QuadRiskExpr` overload encodes variance as a quadratic dot product. The `RSOCRiskExpr`
overload uses a rotated second-order cone to encode the squared norm. The `SquaredSOCRiskExpr`
overload squares an existing SOC variable. The `SOCRiskExpr` overload returns the SOC
variable directly (standard deviation form). All methods return the risk expression and a
scaling factor.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `factor::Number`: Variance correction factor (e.g. `1 / (T - 1)`).
  - `second_moment`: Return deviation vector or matrix.
  - `namet`, `namec`: Bare Model State entry names for the auxiliary variable and its
    constraint.
  - `tsecond_moment`: Pre-existing SOC variable (used by SquaredSOC/SOC overloads).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - A 2-tuple `(r_expr, factor)` of the risk JuMP expression and its scaling factor.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`second_moment_bound_val`](@ref)
"""
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, i::Any, factor::Number,
                                 second_moment, args...; prefix::Symbol = Symbol(""))
    return state_set!(model, prefix, :second_moment_risk_, i,
                      JuMP.@expression(model,
                                       factor *
                                       LinearAlgebra.dot(second_moment, second_moment))),
           sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::RSOCRiskExpr, i::Any, factor::Number,
                                 second_moment, namet::Symbol, namec::Symbol, args...;
                                 prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    tsecond_moment = state_set!(model, prefix, namet, i, JuMP.@variable(model))
    state_set!(model, prefix, namec, i,
               JuMP.@constraint(model,
                                [sc * tsecond_moment
                                 sc * 0.5
                                 sc * second_moment] in JuMP.RotatedSecondOrderCone()))
    return state_set!(model, prefix, :second_moment_risk_, i,
                      JuMP.@expression(model, factor * tsecond_moment)), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SquaredSOCRiskExpr, i::Any,
                                 factor::Number, second_moment, namet::Symbol,
                                 namec::Symbol, tsecond_moment::JuMP.AbstractJuMPScalar;
                                 prefix::Symbol = Symbol(""))
    return state_set!(model, prefix, :second_moment_risk_, i,
                      JuMP.@expression(model, factor * tsecond_moment^2)), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SOCRiskExpr, i::Any, factor::Number,
                                 second_moment, namet::Symbol, namec::Symbol,
                                 tsecond_moment::JuMP.AbstractJuMPScalar;
                                 prefix::Symbol = Symbol(""))
    factor = sqrt(factor)
    return state_set!(model, prefix, :second_moment_risk_, i,
                      JuMP.@expression(model, factor * tsecond_moment)), factor
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert an upper-bound value to the appropriate scale for the second-moment bounding variable.

Scales `ub` by `inv(factor)` and, when the formulation is not `SOCRiskExpr`, applies a
square-root transformation to convert from variance to standard-deviation units. Returns
`nothing` when `ub` is `nothing`. A `Frontier` and a [`TrackedRiskBound`](@ref) take the
scale and the transformation as their `factor` and their `bound`.

# Arguments

  - `alg::SecondMomentFormulation`: Second-moment risk formulation (e.g. `SOCRiskExpr`).
  - `ub`: Upper bound value (scalar, vector, `Frontier`, [`TrackedRiskBound`](@ref), or `nothing`).
  - `factor::Number`: Variance correction factor.

# Returns

  - The rescaled upper bound, or `nothing` when `ub` is `nothing`.

# Related

  - [`set_second_moment_risk!`](@ref)
"""
function second_moment_bound_val(::SecondMomentFormulation, ub::Frontier, factor::Number)
    return Frontier(ub.N, inv(factor), SquareRootBound())
end
function second_moment_bound_val(::SOCRiskExpr, ub::Frontier, factor::Number)
    return Frontier(ub.N, inv(factor), LinearBound())
end
function second_moment_bound_val(::SecondMomentFormulation, ub::TrackedRiskBound,
                                 factor::Number)
    return TrackedRiskBound(ub, inv(factor), SquareRootBound())
end
function second_moment_bound_val(::SOCRiskExpr, ub::TrackedRiskBound, factor::Number)
    return TrackedRiskBound(ub, inv(factor), LinearBound())
end
function second_moment_bound_val(::SecondMomentFormulation, ub::VecNum, factor::Number)
    return inv(factor) * sqrt.(ub)
end
function second_moment_bound_val(::SecondMomentFormulation, ub::Number, factor::Number)
    return inv(factor) * sqrt(ub)
end
function second_moment_bound_val(::SOCRiskExpr, ub::VecNum, factor::Number)
    return inv(factor) * ub
end
function second_moment_bound_val(::SOCRiskExpr, ub::Number, factor::Number)
    return inv(factor) * ub
end
function second_moment_bound_val(::Any, ::Nothing, ::Any)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `LowOrderMoment` with `SecondMoment` (semi-variance /
semi-deviation) to `model`.

Introduces a `sqrt_second_moment` variable and adds SOC or quadratic constraints encoding
the second central moment of portfolio returns relative to the target benchmark. Registers
the risk expression and upper-bound constraint.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment}`: The second-moment risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `second_moment_risk`: The risk expression of [`set_second_moment_risk!`](@ref).

# Related

  - [`LowOrderMoment`](@ref)
  - [`SecondMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    w = get_w(model, prefix)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tgt = calc_risk_constraint_target(model, r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    sqrt_second_moment = state_set!(model, prefix, :sqrt_second_moment_, i,
                                    JuMP.@variable(model))
    second_moment = state_set!(model, prefix, :second_moment_, i,
                               JuMP.@expression(model, net_X .- tgt))
    if isa(r.alg.alg1, SemiMoment)
        second_lower_moment = state_set!(model, prefix, :second_lower_moment_, i,
                                         JuMP.@variable(model, [1:T], (lower_bound = 0)))
        state_set!(model, prefix, :csecond_lower_moment_mar_, i,
                   JuMP.@constraint(model, sc * (second_moment + second_lower_moment) >= 0))
        second_moment = second_lower_moment
    end
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = checked_observation_weights(wi, net_X)
    second_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment; prefix = prefix)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        second_moment = state_set!(model, prefix, :scaled_second_moment_, i,
                                   JuMP.@expression(model, wi .* second_moment))
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment; prefix = prefix)
    end
    state_set!(model, prefix, :csqrt_second_moment_soc_, i,
               JuMP.@constraint(model,
                                [sc * sqrt_second_moment
                                 sc * second_moment] in JuMP.SecondOrderCone()))
    ub = second_moment_bound_val(r.alg.alg2, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_moment, ub,
                                             :sqrt_second_moment_, i, second_moment_risk,
                                             r.settings; prefix = prefix)
    return second_moment_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add an even moment to `model`, bound it by `settings.ub`, and add it to the risk that the objective reads.

[`set_even_moment_risk!`](@ref) adds the epigraph of the ``2p``-th central (full) or lower (semi) even moment, which the objective reads.

The target scales with ``k``, because the model subtracts it from the returns of the unscaled weights.

Under a unit budget the moment has degree two in the unscaled weights, and the row of [`set_risk_upper_bound!`](@ref) needs degree one. [`set_even_moment_bound!`](@ref) then adds the model that the bound acts on, with ``k`` in its cones. That model is exact at the level of the bound alone. A [`TrackedRiskBound`](@ref) that has to hold at every level adds a second epigraph, with ``k`` in its cones and in units of the level. A bound whose [`risk_bound_level`](@ref) is zero has no unit, and [`set_risk_epigraphs!`](@ref) then adds a second epigraph in the units of the moment. A measure that no objective reads adds the model of its bound alone. A measure without a bound, and every measure under another head, adds one model.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment}`: The even-moment risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `risk`: The epigraph variable of the even moment, the risk expression. Under a unit budget it is the expression that the bound acts on when the measure is only a bound.

# Related

  - [`LowOrderMoment`](@ref)
  - [`EvenMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_even_moment_risk!`](@ref)
  - [`set_even_moment_bound!`](@ref)
  - [`even_moment_bound_val`](@ref)
  - [`calc_risk_constraint_target`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    w = get_w(model, prefix)
    # The target is subtracted from the returns of the unscaled weights, so it scales with
    # the variable.
    tgt = calc_risk_constraint_target(model, r, w, pr.mu, get_k(model))
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    ow = checked_observation_weights(nothing_scalar_array_selector(r.w, pr.w), net_X)
    # A row of weight zero does not change the moment, but the power cone row of a large loss
    # can stop the solver far from the optimum, as in the EVaR and power-norm programmes.
    rows = positive_weight_rows(ow, T)
    even_moment = state_set!(model, prefix, :even_moment_, i,
                             JuMP.@expression(model, net_X .- tgt))
    # Each name is assigned once, because the closure below captures it.
    Td, wi = if isnothing(ow)
        T - r.alg.ddof, nothing
    else
        (T - r.alg.ddof) / T * sum(ow), view(ow, rows)
    end
    ub = r.settings.ub
    level, exact = risk_bound_level(ub)
    # A level of zero gives the model of the bound no unit.
    if level > zero(level) && is_unit_budget(model)
        bound = if exact
            bp = nested_prefix(prefix, :bound_)
            scaled_moment = state_set!(model, bp, :even_moment_, i,
                                       JuMP.@expression(model, even_moment ./ sqrt(level)))
            set_even_moment_risk!(model, i, r.alg, scaled_moment, rows, wi, Td,
                                  get_k(model); prefix = bp)
        else
            set_even_moment_bound!(model, i, r.alg, even_moment, rows, wi, Td, level;
                                   prefix = prefix)
        end
        ub = even_moment_bound_val(ub)
        # A measure that is only a bound has no term to build.
        risk = if reads_risk_term(r.settings)
            set_even_moment_risk!(model, i, r.alg, even_moment, rows, wi, Td,
                                  effective_k(model); prefix = prefix)
        else
            bound
        end
    else
        risk, bound = set_risk_epigraphs!(model, reads_risk_term(r.settings),
                                          !isnothing(ub); prefix = prefix) do k, p
            return set_even_moment_risk!(model, i, r.alg, even_moment, rows, wi, Td, k;
                                         prefix = p)
        end
    end
    set_variance_risk_bounds_and_expression!(model, opt, bound, ub, :even_moment_risk_, i,
                                             risk, r.settings; prefix = prefix)
    return risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the deviations that an even moment reads, over the observations in `rows`.

  - [`FullMoment`](@ref) reads the deviations from the target.
  - [`SemiMoment`](@ref) reads the lower deviations. The method adds them as variables on the first call for the measure, and a later call returns them.

# JuMP formulation

## Variables

  - `even_lower_moment_`: created by the semi method, ``\\boldsymbol{l} \\geq \\boldsymbol{0}``.

## Constraints

  - `ceven_lower_moment_mar_`: ``s_c \\left(d_t + l_t\\right) \\geq 0``, registered by the semi method.

Where:

  - ``d_t``: Entry ``t`` of the deviations, `even_moment`.
  - ``l_t``: Lower deviation of observation ``t``.
  - $(math_dict[:sc_scale])

Each name carries the index `i` of the measure, and the namespace `prefix`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `alg`: The moment algorithm, [`FullMoment`](@ref) or [`SemiMoment`](@ref).
  - `even_moment::VecJuMPScalar`: The deviations from the target, one for each observation.
  - `rows`: The observations of positive weight.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `dev`: The deviations, a vector of JuMP scalars.

# Related

  - [`EvenMoment`](@ref)
  - [`set_even_moment_risk!`](@ref)
  - [`set_even_moment_bound!`](@ref)
"""
function set_even_moment_deviations!(::JuMP.Model, ::Any, ::FullMoment,
                                     even_moment::VecJuMPScalar, rows;
                                     prefix::Symbol = Symbol(""))
    return view(even_moment, rows)
end
function set_even_moment_deviations!(model::JuMP.Model, i::Any, ::SemiMoment,
                                     even_moment::VecJuMPScalar, rows;
                                     prefix::Symbol = Symbol(""))
    return state_build!(model, prefix, :even_lower_moment_, i) do
        sc = get_constraint_scale(model)
        dev = JuMP.@variable(model, [1:length(rows)], lower_bound = 0)
        state_set!(model, prefix, :ceven_lower_moment_mar_, i,
                   JuMP.@constraint(model, sc * (view(even_moment, rows) + dev) >= 0))
        return dev
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the epigraph of an even moment, the risk that the objective reads, and return its variable.

The method adds the variables `even_moment_u_`, `even_moment_t_` and `even_moment_risk_`, and a chain of power cone rows.

# Mathematical definition

The ``2p``-th even moment is encoded via the following power cone formulation (full variant):

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{u},\\,\\boldsymbol{s},\\,r}{\\mathrm{opt}} \\quad & r \\\\
\\mathrm{s.t.} \\quad & \\sum_{t \\in \\mathcal{I}} u_t \\leq r \\\\
               \\quad & \\left(u_t \\cdot T_d,\\, r,\\, s_t\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{p}\\right),\\quad t \\in \\mathcal{I} \\\\
               \\quad & \\left(s_t,\\, k,\\, \\hat{r}_t - \\mu\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{2}\\right),\\quad t \\in \\mathcal{I}\\,.
\\end{align}
```

For the semi variant, lower-deviation variables ``d_t \\geq 0`` with ``\\hat{r}_t - \\mu + d_t \\geq 0`` replace the centred returns in the innermost power cone. With observation weights ``q_t``, the budget row reads ``\\sum_{t \\in \\mathcal{I}} q_t u_t``.

The variables and the rows cover the observations of positive weight only. An observation of weight zero does not change the moment, but the power cone row of a large loss can stop the solver far from the optimum.

Where:

  - ``r``: Even-moment risk variable.
  - ``\\boldsymbol{u}``: `N × 1` auxiliary variable vector, with one entry for each observation in ``\\mathcal{I}``.
  - ``\\boldsymbol{s}``: `N × 1` auxiliary variable vector, with one entry for each observation in ``\\mathcal{I}``.
  - $(math_dict[:I_pos_obs])
  - ``T_d = T - \\mathrm{ddof}``: Effective sample size.
  - $(math_dict[:T])
  - ``k``: Budget-scaling / homogenisation variable, the `k` argument. The objective reads ``1`` in its place under a unit budget, [`effective_k`](@ref), so the model minimises the moment of the unscaled weights. The target ``\\mu`` always scales with the variable.
  - ``p``: Order parameter; the moment order is ``2p``.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal\\boldsymbol{w}``: Portfolio return at time ``t``.
  - ``\\mu``: Target return.
  - ``\\mathcal{K}_{\\mathrm{pow}}(\\alpha)``: Power cone ``\\{(a, b, c) : a^{\\alpha}\\,b^{1-\\alpha} \\geq |c|,\\; a, b \\geq 0\\}``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `alg::EvenMoment`: The algorithm of the measure, read for its order and its moment.
  - `even_moment::VecJuMPScalar`: The deviations from the target, one for each observation.
  - `rows`: The observations of positive weight.
  - `wi`: The positive observation weights, or `nothing`.
  - `Td::Number`: The effective sample size.
  - `k`: The scale that the cones divide by, [`effective_k`](@ref) or [`get_k`](@ref).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `even_moment_risk`: The epigraph variable of the even moment.

# Related

  - [`EvenMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_even_moment_deviations!`](@ref)
  - [`effective_k`](@ref)
"""
function set_even_moment_risk!(model::JuMP.Model, i::Any, alg::EvenMoment,
                               even_moment::VecJuMPScalar, rows, wi::Option{<:VecNum},
                               Td::Number, k; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    N = length(rows)
    u = state_set!(model, prefix, :even_moment_u_, i, JuMP.@variable(model, [1:N]))
    t = state_set!(model, prefix, :even_moment_t_, i, JuMP.@variable(model, [1:N]))
    even_moment_risk = state_set!(model, prefix, :even_moment_risk_, i,
                                  JuMP.@variable(model, lower_bound = 0))
    state_set!(model, prefix, :ceven_moment_s_, i,
               if isnothing(wi)
                   JuMP.@constraint(model, sc * (sum(u) - even_moment_risk) <= 0)
               else
                   JuMP.@constraint(model,
                                    sc * (LinearAlgebra.dot(u, wi) - even_moment_risk) <= 0)
               end)
    state_set!(model, prefix, :cpoweven_moment_p_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * u[j] * Td, sc * even_moment_risk, sc * t[j]] in
                                JuMP.MOI.PowerCone(inv(alg.p))))
    dev = set_even_moment_deviations!(model, i, alg.alg, even_moment, rows; prefix = prefix)
    state_set!(model, prefix, :cpoweven_moment_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * t[j], sc * k, sc * dev[j]] in JuMP.MOI.PowerCone(0.5)))
    return even_moment_risk
end
"""
    even_moment_bound_val(ub::Number)
    even_moment_bound_val(ub::TrackedRiskBound)

Converts a bound into the units of the expression of [`set_even_moment_bound!`](@ref).

The expression is in units of the bound, so a bound of one number is `1`. A [`TrackedRiskBound`](@ref) takes the inverse of its `level` as its `factor`. Away from the level the expression of [`set_even_moment_bound!`](@ref) is a power of the risk, so the conversion holds at the level alone. The scaled epigraph of a request with `exact = true` is the risk in units of the level, so the conversion holds at every level.

# Arguments

  - `ub`: The bound, whose [`risk_bound_level`](@ref) is positive.

# Returns

  - `ub`: The converted bound.

# Related

  - [`set_even_moment_bound!`](@ref)
  - [`risk_bound_level`](@ref)
"""
function even_moment_bound_val(ub::Number)
    return one(ub)
end
function even_moment_bound_val(ub::TrackedRiskBound)
    return TrackedRiskBound(ub, inv(ub.level), LinearBound())
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the model that the bound `ub` of an even moment acts on under a unit budget, and return its expression.

The expression has degree one in the unscaled weights and ``k``, and it is in units of the bound. So the bound row of [`set_risk_upper_bound!`](@ref) holds it under ``1 \\cdot k``. Each entry of the cones has the size of ``k`` when the bound binds. A copy of the epigraph of the objective with ``k`` in its cones has entries of the size of the moment, about ``10^{-5}`` for daily returns, and an interior point solver can stop on it.

# Mathematical definition

```math
\\begin{align}
b(\\boldsymbol{y}, k) &= \\frac{1}{T_d} \\sum_{t \\in \\mathcal{I}} u_t\\,, \\quad u_t \\geq \\frac{s_t^{p}}{k^{p - 1}}\\,, \\quad s_t \\geq \\frac{q_t^{1/p} d_t^{2}}{u_b k}\\,, \\\\
b(\\boldsymbol{y}, k) \\leq k &\\quad\\Leftrightarrow\\quad \\left(\\frac{1}{T_d} \\sum_{t \\in \\mathcal{I}} q_t \\left\\lvert \\frac{d_t}{k} \\right\\rvert^{2p}\\right)^{1/p} \\leq u_b\\,.
\\end{align}
```

Where:

  - ``b``: The expression.
  - ``d_t``: Deviation ``t`` of the unscaled weights, from [`set_even_moment_deviations!`](@ref).
  - ``u_t``, ``s_t``: Auxiliary variables.
  - ``u_b``: The bound, `ub`.
  - ``q_t``: Weight of observation ``t``, ``1`` without observation weights.
  - ``T_d``: Effective sample size, `Td`.
  - ``p``: Order parameter; the moment order is ``2p``.
  - $(math_dict[:y_homog])
  - $(math_dict[:k_budget])
  - $(math_dict[:I_pos_obs])

# JuMP formulation

## Variables

  - `bound_even_moment_u_`: created, ``\\boldsymbol{u}``.
  - `bound_even_moment_t_`: created, ``\\boldsymbol{s}``.

## Expressions

  - `bound_even_moment_risk_`: ``b``.

## Constraints

  - `bound_cpoweven_moment_p_`: ``\\left(s_c u_t, s_c k, s_c s_t\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1 / p)``.
  - `bound_cpoweven_moment_`: ``\\left(s_c s_t, s_c k, s_c q_t^{1/(2p)} d_t / \\sqrt{u_b}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1 / 2)``.

Where:

  - ``\\mathcal{K}_{\\mathrm{pow}}(\\alpha)``: Power cone ``\\{(a, b, c) : a^{\\alpha}\\,b^{1-\\alpha} \\geq |c|,\\; a, b \\geq 0\\}``.
  - $(math_dict[:sc_scale])

Each name carries the index `i` of the measure, and the namespace `prefix`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `alg::EvenMoment`: The algorithm of the measure, read for its order and its moment.
  - `even_moment::VecJuMPScalar`: The deviations from the target, one for each observation.
  - `rows`: The observations of positive weight.
  - `wi`: The positive observation weights, or `nothing`.
  - `Td::Number`: The effective sample size.
  - `ub::Number`: The bound.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bound`: The expression ``b``.

# Related

  - [`EvenMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_even_moment_deviations!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function set_even_moment_bound!(model::JuMP.Model, i::Any, alg::EvenMoment,
                                even_moment::VecJuMPScalar, rows, wi::Option{<:VecNum},
                                Td::Number, ub::Number; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    k = get_k(model)
    bp = nested_prefix(prefix, :bound_)
    N = length(rows)
    # The weight of an observation enters through its deviation, so every budget coefficient
    # is equal.
    iub = inv(sqrt(ub))
    c = isnothing(wi) ? fill(iub, N) : iub .* wi .^ inv(2 * alg.p)
    dev = set_even_moment_deviations!(model, i, alg.alg, even_moment, rows; prefix = prefix)
    u = state_set!(model, bp, :even_moment_u_, i, JuMP.@variable(model, [1:N]))
    t = state_set!(model, bp, :even_moment_t_, i, JuMP.@variable(model, [1:N]))
    state_set!(model, bp, :cpoweven_moment_p_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * u[j], sc * k, sc * t[j]] in
                                JuMP.MOI.PowerCone(inv(alg.p))))
    state_set!(model, bp, :cpoweven_moment_, i,
               JuMP.@constraint(model, [j = 1:N],
                                [sc * t[j], sc * k, sc * c[j] * dev[j]] in
                                JuMP.MOI.PowerCone(0.5)))
    return state_set!(model, bp, :even_moment_risk_, i,
                      JuMP.@expression(model, sum(u) / Td))
end
