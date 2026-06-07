"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the target return used as the reference level for lower/upper moment risk constraints.

Dispatches on the type of `r.mu`: when `nothing`, uses the prior mean vector `mu`; when a
`VecNum`, uses `r.mu`; when a `VecScalar`, combines the vector and scalar parts with `k`;
when a scalar `Number`, scales it by `k`.

# Arguments

  - `r::LoHiOrderMoment`: Risk measure carrying the target specification.
  - `w`: Portfolio weight vector.
  - `mu::VecNum`: Prior mean return vector.
  - `k`: Leverage/scale variable from the model.

# Returns

  - The scalar target return used as the lower moment reference.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function calc_risk_constraint_target(::LoHiOrderMoment{<:Any, <:Any, Nothing, <:Any},
                                     w::VecNum, mu::VecNum, args...)
    return LinearAlgebra.dot(w, mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                                     w::VecNum, args...)
    return LinearAlgebra.dot(w, r.mu)
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     w::VecNum, ::Any, k)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s * k
end
function calc_risk_constraint_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
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
z_t &\\geq \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\hat{r}_t,\\quad z_t \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{FLM}(\\boldsymbol{w})``: First lower moment.
  - $(math_dict[:T])
  - ``z_t \\geq 0``: Auxiliary variables capturing deviations below the mean.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

Mean absolute deviation:

```math
\\begin{align}
\\mathrm{MAD}(\\boldsymbol{w}) &= \\frac{1}{T}\\sum_{t=1}^T z_t\\,, \\\\
z_t &\\geq |\\hat{r}_t - \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}|,\\quad z_t \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MAD}(\\boldsymbol{w})``: Mean absolute deviation.
  - $(math_dict[:T])
  - ``z_t \\geq 0``: Auxiliary variables capturing absolute deviations.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

where ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` is the net portfolio return at time ``t``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment`: Risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `pr::AbstractPriorResult`: Prior result containing `X` (returns matrix) and `mu`.

# Returns

  - `nothing`.

# Related

  - [`calc_risk_constraint_target`](@ref)
  - [`set_second_moment_risk!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:flm_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    flm = model[Symbol(:flm_, i)] = JuMP.@variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    flm_risk = model[key] = if isnothing(wi)
        JuMP.@expression(model, Statistics.mean(flm))
    else
        JuMP.@expression(model, Statistics.mean(flm, wi))
    end
    model[Symbol(:cflm_mar_, i)] = JuMP.@constraint(model, sc * ((net_X + flm) .- tgt) >= 0)
    set_risk_bounds_and_expression!(model, opt, flm_risk, r.settings, key)
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

  - `nothing`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any,
                                                 <:MeanAbsoluteDeviation},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:mad_risk_, i)
    sc = model[:sc]
    w = model[:w]
    k = model[:k]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    mad = model[Symbol(:mad_, i)] = JuMP.@variable(model, [1:T], lower_bound = 0)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    mad_risk = model[Symbol(:mad_risk_, i)] = if isnothing(wi)
        JuMP.@expression(model, 2 * Statistics.mean(mad))
    else
        JuMP.@expression(model, 2 * Statistics.mean(mad, wi))
    end
    model[Symbol(:cmar_mad_, i)] = JuMP.@constraint(model, sc * ((net_X + mad) .- tgt) >= 0)
    set_risk_bounds_and_expression!(model, opt, mad_risk, r.settings, key)
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
  - $(arg_dict[:key_sym])
  - `keyt`, `keyc`: Symbols for the auxiliary variable and its constraint.
  - `tsecond_moment`: Pre-existing SOC variable (used by SquaredSOC/SOC overloads).

# Returns

  - A 2-tuple `(r_expr, factor)` of the risk JuMP expression and its scaling factor.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`second_moment_bound_val`](@ref)
"""
function set_second_moment_risk!(model::JuMP.Model, ::QuadRiskExpr, ::Any, factor::Number,
                                 second_moment, key::Symbol, args...)
    return model[key] = JuMP.@expression(model,
                                         factor *
                                         LinearAlgebra.dot(second_moment, second_moment)),
                        sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::RSOCRiskExpr, i::Any, factor::Number,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 args...)
    sc = model[:sc]
    tsecond_moment = model[Symbol(keyt, i)] = JuMP.@variable(model)
    model[Symbol(keyc, i)] = JuMP.@constraint(model,
                                              [sc * tsecond_moment;
                                               0.5;
                                               sc * second_moment] in
                                              JuMP.RotatedSecondOrderCone())
    return model[key] = JuMP.@expression(model, factor * tsecond_moment), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SquaredSOCRiskExpr, i::Any,
                                 factor::Number, second_moment, key::Symbol, keyt::Symbol,
                                 keyc::Symbol, tsecond_moment::JuMP.AbstractJuMPScalar)
    return model[key] = JuMP.@expression(model, factor * tsecond_moment^2), sqrt(factor)
end
function set_second_moment_risk!(model::JuMP.Model, ::SOCRiskExpr, i::Any, factor::Number,
                                 second_moment, key::Symbol, keyt::Symbol, keyc::Symbol,
                                 tsecond_moment::JuMP.AbstractJuMPScalar)
    factor = sqrt(factor)
    return model[key] = JuMP.@expression(model, factor * tsecond_moment), factor
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert an upper-bound value to the appropriate scale for the second-moment bounding variable.

Scales `ub` by `inv(factor)` and, when the formulation is not `SOCRiskExpr`, applies a
square-root transformation to convert from variance to standard-deviation units. Returns
`nothing` when `ub` is `nothing`.

# Arguments

  - `alg::SecondMomentFormulation`: Second-moment risk formulation (e.g. `SOCRiskExpr`).
  - `ub`: Upper bound value (scalar, vector, `Frontier`, or `nothing`).
  - `factor::Number`: Variance correction factor.

# Returns

  - The rescaled upper bound, or `nothing` when `ub` is `nothing`.

# Related

  - [`set_second_moment_risk!`](@ref)
"""
function second_moment_bound_val(::SecondMomentFormulation, ub::Frontier, factor::Number)
    return _Frontier(; N = ub.N, factor = inv(factor), bound = SquareRootBound())
end
function second_moment_bound_val(::SOCRiskExpr, ub::Frontier, factor::Number)
    return _Frontier(; N = ub.N, factor = inv(factor), bound = LinearBound())
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

  - `nothing`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`SecondMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:second_moment_risk_, i)
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    bound_key = Symbol(:sqrt_second_moment_, i)
    sqrt_second_moment = model[bound_key] = JuMP.@variable(model)
    second_moment = model[Symbol(:second_moment_, i)] = JuMP.@expression(model,
                                                                         net_X .- tgt)
    if isa(r.alg.alg1, Semi)
        second_lower_moment = model[Symbol(:second_lower_moment_, i)] = JuMP.@variable(model,
                                                                                       [1:T],
                                                                                       (lower_bound = 0))
        model[Symbol(:csecond_lower_moment_mar_, i)] = JuMP.@constraint(model,
                                                                        sc *
                                                                        (second_moment +
                                                                         second_lower_moment) >=
                                                                        0)
        second_moment = second_lower_moment
    end
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    second_moment_risk, factor = if isnothing(wi)
        factor = StatsBase.varcorrection(T, r.alg.ve.corrected)
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment, key,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment)
    else
        factor = StatsBase.varcorrection(wi, r.alg.ve.corrected)
        wi = sqrt.(wi)
        second_moment = model[Symbol(:scaled_second_moment_, i)] = JuMP.@expression(model,
                                                                                    wi .*
                                                                                    second_moment)
        set_second_moment_risk!(model, r.alg.alg2, i, factor, second_moment, key,
                                :tsecond_moment_risk_, :csecond_moment_rsoc_,
                                sqrt_second_moment)
    end
    model[Symbol(:csqrt_second_moment_soc_, i)] = JuMP.@constraint(model,
                                                                   [sc * sqrt_second_moment
                                                                    sc * second_moment] in
                                                                   JuMP.SecondOrderCone())
    ub = second_moment_bound_val(r.alg.alg2, r.settings.ub, factor)
    set_variance_risk_bounds_and_expression!(model, opt, sqrt_second_moment, ub, bound_key,
                                             second_moment_risk, r.settings)
    return second_moment_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add even-moment risk constraints to `model` for [`LowOrderMoment`](@ref) with [`EvenMoment`](@ref).

Introduces auxiliary variables `even_moment_u`, `even_moment_t`, and `even_moment_risk`, and adds a chain of power cone constraints encoding the ``2p``-th central (full) or lower (semi) even moment of portfolio returns relative to the target. Registers the risk expression and upper-bound constraint.

# Mathematical definition

The ``2p``-th even moment is encoded via the following power cone formulation (full variant):

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{u},\\,\\boldsymbol{s},\\,r}{\\mathrm{opt}} \\quad & r \\\\
\\mathrm{s.t.} \\quad & \\sum_{t=1}^{T} u_t \\leq r \\\\
               \\quad & \\left(u_t \\cdot T_d,\\, r,\\, s_t\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{p}\\right),\\quad t = 1,\\ldots,T \\\\
               \\quad & \\left(s_t,\\, k,\\, \\hat{r}_t - \\mu\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{2}\\right),\\quad t = 1,\\ldots,T\\,.
\\end{align}
```

For the semi variant, lower-deviation variables ``d_t \\geq 0`` with ``\\hat{r}_t - \\mu + d_t \\geq 0`` replace the centred returns in the innermost power cone.

Where:

  - ``r``: Even-moment risk variable.
  - ``\\boldsymbol{u}``: `T × 1` auxiliary variable vector.
  - ``\\boldsymbol{s}``: `T × 1` auxiliary variable vector.
  - ``T_d = T - \\mathrm{ddof}``: Effective sample size.
  - $(math_dict[:T])
  - ``k``: Budget-scaling / homogenisation variable.
  - ``p``: Order parameter; the moment order is ``2p``.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal\\boldsymbol{w}``: Portfolio return at time ``t``.
  - ``\\mu``: Target return.
  - ``\\mathcal{K}_{\\mathrm{pow}}(\\alpha)``: Power cone ``\\{(a, b, c) : a^{\\alpha}\\,b^{1-\\alpha} \\geq |c|,\\; a, b \\geq 0\\}``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment}`: The even-moment risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`EvenMoment`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`calc_risk_constraint_target`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    key = Symbol(:even_moment_risk_, i)
    w = model[:w]
    k = ifelse(haskey(model, :crkb), 1, model[:k])
    sc = model[:sc]
    tgt = calc_risk_constraint_target(r, w, pr.mu, k)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    p = r.alg.p
    Td = T - r.alg.ddof
    even_moment_u, even_moment_t, even_moment_risk = model[Symbol(:even_moment_u_, i)], model[Symbol(:even_moment_t_, i)], model[key] = JuMP.@variables(model,
                                                                                                                                                        begin
                                                                                                                                                            [1:T]
                                                                                                                                                            [1:T]
                                                                                                                                                            (),
                                                                                                                                                            (lower_bound = 0)
                                                                                                                                                        end)
    even_moment = model[Symbol(:even_moment_, i)] = JuMP.@expression(model, net_X .- tgt)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    model[Symbol(:ceven_moment_s_, i)] = if isnothing(wi)
        JuMP.@constraint(model, sc * (sum(even_moment_u) - even_moment_risk) <= 0)
    else
        Td = Td / T * sum(wi)
        JuMP.@constraint(model,
                         sc * (LinearAlgebra.dot(even_moment_u, wi) - even_moment_risk) <=
                         0)
    end
    model[Symbol(:cpoweven_moment_p_, i)] = JuMP.@constraint(model, [i = 1:T],
                                                             [sc * even_moment_u[i] * Td,
                                                              sc * even_moment_risk,
                                                              sc * even_moment_t[i]] in
                                                             JuMP.MOI.PowerCone(inv(p)))
    if isa(r.alg.alg, Full)
        model[Symbol(:cpoweven_moment_, i)] = JuMP.@constraint(model, [i = 1:T],
                                                               [sc * even_moment_t[i],
                                                                sc * k,
                                                                sc * even_moment[i]] in
                                                               JuMP.MOI.PowerCone(0.5))
    else
        even_lower_moment = model[Symbol(:even_lower_moment_, i)] = JuMP.@variable(model,
                                                                                   [1:T],
                                                                                   (lower_bound = 0))
        model[Symbol(:ceven_lower_moment_mar_, i)] = JuMP.@constraint(model,
                                                                      sc * (even_moment +
                                                                            even_lower_moment) >=
                                                                      0)
        model[Symbol(:cpoweven_moment_, i)] = JuMP.@constraint(model, [i = 1:T],
                                                               [sc * even_moment_t[i],
                                                                sc * k,
                                                                sc * even_lower_moment[i]] in
                                                               JuMP.MOI.PowerCone(0.5))
    end
    set_risk_bounds_and_expression!(model, opt, even_moment_risk, r.settings, key)
    return even_moment_risk
end
