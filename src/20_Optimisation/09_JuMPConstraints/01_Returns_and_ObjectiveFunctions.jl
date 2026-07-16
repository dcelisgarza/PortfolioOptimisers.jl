"""
$(DocStringExtensions.TYPEDEF)

JuMP returns estimator that computes portfolio returns as the arithmetic (dot-product)
mean return: ``r = \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``.

Optionally supports an uncertainty set on the mean vector (box or ellipsoidal) and a lower
bound on the portfolio return. When `ucs` is set the optimiser maximises the **worst-case**
expected return over the set instead of the point estimate `μ`, giving a robust return.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ArithmeticReturn(;
        ucs::Option{<:UcSE_UcS} = nothing,
        lb::Option{<:RkRtBounds} = nothing,
        mu::Option{<:Num_VecNum} = nothing
    ) -> ArithmeticReturn

Keywords correspond to the struct's fields.

## Details

  - `ucs` accepts either a pre-built mean uncertainty set (the result of [`mu_ucs`](@ref), e.g. a `BoxUncertaintySet` or `EllipsoidalUncertaintySet`) or an uncertainty-set *estimator*. A pre-built set is the simplest path — symmetric with how [`UncertaintySetVariance`](@ref) takes a pre-built [`sigma_ucs`](@ref) result. Passing an estimator defers construction to solve time and requires the returns data (`rd`) to be threaded through the optimiser.

## Validation

  - If `ucs` is an `EllipsoidalUncertaintySet`: must be parameterised by `MuEllipsoidalUncertaintySet`.
  - If `lb` is a number: `isfinite(lb)`.
  - If `lb` is a vector: `!isempty(lb)` and `all(isfinite, lb)`.
  - If `mu` is a number: `isfinite(mu)`.
  - If `mu` is a vector: `!isempty(mu)` and `all(isfinite, mu)`.

# Related

  - [`bounds_returns_estimator`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
"""
@concrete struct ArithmeticReturn <: JuMPReturnsEstimator
    """
    $(field_dict[:ucs])
    """
    ucs
    """
    $(field_dict[:lb])
    """
    lb
    """
    $(field_dict[:mu])
    """
    mu
    function ArithmeticReturn(ucs::Option{<:UcSE_UcS}, lb::Option{<:RkRtBounds},
                              mu::Option{<:Num_VecNum})
        if isa(ucs, EllipsoidalUncertaintySet)
            @argcheck(isa(ucs,
                          EllipsoidalUncertaintySet{<:Any, <:Any,
                                                    <:MuEllipsoidalUncertaintySet}),
                      ArgumentError("ucs must be parameterised by MuEllipsoidalUncertaintySet, got $(typeof(ucs))"))
        end
        if isa(lb, Number)
            @argcheck(isfinite(lb), IsNonFiniteError("lb must be finite, got $lb"))
        elseif isa(lb, VecNum)
            @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
            @argcheck(all(isfinite, lb),
                      IsNonFiniteError("all elements of lb must be finite"))
        end
        if isa(mu, VecNum)
            @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
            @argcheck(all(isfinite, mu),
                      IsNonFiniteError("all elements of mu must be finite"))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
        end
        return new{typeof(ucs), typeof(lb), typeof(mu)}(ucs, lb, mu)
    end
end
function ArithmeticReturn(; ucs::Option{<:UcSE_UcS} = nothing,
                          lb::Option{<:RkRtBounds} = nothing,
                          mu::Option{<:Num_VecNum} = nothing)
    return ArithmeticReturn(ucs, lb, mu)
end
function factory(rt::ArithmeticReturn, pr::AbstractPriorResult, ::Any,
                 ucs::Option{<:UcSE_UcS} = nothing, args...; kwargs...)
    return ArithmeticReturn(; ucs = ucs_selector(rt.ucs, ucs), lb = rt.lb,
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, pr::AbstractPriorResult,
                 ucs::Option{<:UcSE_UcS} = nothing; kwargs...)
    return ArithmeticReturn(; ucs = ucs_selector(rt.ucs, ucs), lb = rt.lb,
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, pr::AbstractPriorResult; kwargs...)
    return ArithmeticReturn(; ucs = ucs_selector(rt.ucs, ucs), lb = rt.lb,
                            mu = nothing_scalar_array_selector(rt.mu, pr.mu))
end
function factory(rt::ArithmeticReturn, ucs::UcSE_UcS, args...; kwargs...)
    return ArithmeticReturn(; ucs = ucs_selector(rt.ucs, ucs), lb = rt.lb, mu = rt.mu)
end
function port_opt_view(r::ArithmeticReturn, i, args...)
    uset = port_opt_view(r.ucs, i)
    mu = nothing_scalar_array_view(r.mu, i)
    return ArithmeticReturn(; ucs = uset, lb = r.lb, mu = mu)
end
"""
    no_bounds_returns_estimator(r, args...)

Create a version of the returns estimator with lower bounds removed.

Used internally in risk frontier sub-problems to remove return lower bounds from the estimator so the sub-problem is unconstrained.

# Arguments

  - `r`: JuMP returns estimator ([`ArithmeticReturn`](@ref) or [`LogarithmicReturn`](@ref)).
  - `args...`: Additional arguments (e.g. `flag::Bool` for uncertainty set handling).

# Returns

  - Returns estimator without bounds.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`no_bounds_optimiser`](@ref)
"""
function no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool = true)
    return flag ? ArithmeticReturn(; ucs = r.ucs, mu = r.mu) : ArithmeticReturn()
end
"""
$(DocStringExtensions.TYPEDEF)

JuMP returns estimator that computes portfolio returns as the logarithmic (geometric)
mean return: ``r = \\prod_{t=1}^T (1 + \\boldsymbol{x}_t^\\intercal \\boldsymbol{w})^{1/T} - 1``.

Optionally supports observation weights and a lower bound on the portfolio return.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogarithmicReturn(;
        w::Option{<:ObsWeights} = nothing,
        lb::Option{<:RkRtBounds} = nothing
    ) -> LogarithmicReturn

Keywords correspond to the struct's fields.

## Validation

  - If `w` is provided: `!isempty(w)`, all elements non-negative and finite.
  - If `lb` is a number: `isfinite(lb)`.
  - If `lb` is a vector: `!isempty(lb)` and `all(isfinite, lb)`.

# Related

  - [`bounds_returns_estimator`](@ref)
  - [`ArithmeticReturn`](@ref)
  - [`JuMPReturnsEstimator`](@ref)
"""
@concrete struct LogarithmicReturn <: JuMPReturnsEstimator
    """
    $(field_dict[:oow])
    """
    w
    """
    $(field_dict[:lb])
    """
    lb
    function LogarithmicReturn(w::Option{<:ObsWeights}, lb::Option{<:RkRtBounds})
        assert_nonempty_nonneg_finite_val(w, :w)
        if isa(lb, Number)
            @argcheck(isfinite(lb), IsNonFiniteError("lb must be finite, got $lb"))
        elseif isa(lb, VecNum)
            @argcheck(!isempty(lb), IsEmptyError("lb cannot be empty"))
            @argcheck(all(isfinite, lb),
                      IsNonFiniteError("all elements of lb must be finite"))
        end
        return new{typeof(w), typeof(lb)}(w, lb)
    end
end
function LogarithmicReturn(; w::Option{<:ObsWeights} = nothing,
                           lb::Option{<:RkRtBounds} = nothing)
    return LogarithmicReturn(w, lb)
end
function factory(rt::LogarithmicReturn, pr::AbstractPriorResult, args...; kwargs...)
    return LogarithmicReturn(; w = nothing_scalar_array_selector(rt.w, pr.w), lb = rt.lb)
end
function no_bounds_returns_estimator(r::LogarithmicReturn, args...)
    return LogarithmicReturn(; w = r.w)
end
#=
mutable struct AKelly <: RetType
    formulation::VarianceFormulation
    a_rc::Union{<:MatNum, Nothing}
    b_rc::Union{<:AbstractVector, Nothing}
end
function AKelly(; formulation::VarianceFormulation = SOC(),
                a_rc::Union{<:MatNum, Nothing} = nothing,
                b_rc::Union{<:AbstractVector, Nothing} = nothing)
    if !isnothing(a_rc) && !isnothing(b_rc) && !isempty(a_rc) && !isempty(b_rc)
        @argcheck(size(a_rc, 1) == length(b_rc),
                  DimensionMismatch("size(a_rc, 1) ($(size(a_rc, 1))) must match length(b_rc) ($(length(b_rc)))"))
    end
    return AKelly(formulation, a_rc, b_rc)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Set properties of [`AKelly`](@ref) with validation. When setting `:a_rc` or `:b_rc`, checks that the new value is consistent with the existing constraint matrix dimensions.
"""
function Base.setproperty!(obj::AKelly, sym::Symbol, val)
    if sym == :a_rc
        if !isnothing(val) && !isnothing(obj.b_rc) && !isempty(val) && !isempty(obj.b_rc)
            @argcheck(size(val, 1) == length(obj.b_rc),
                      DimensionMismatch("size(val, 1) ($(size(val, 1))) must match length(obj.b_rc) ($(length(obj.b_rc)))"))
        end
    elseif sym == :b_rc
        if !isnothing(val) && !isnothing(obj.a_rc) && !isempty(val) && !isempty(obj.a_rc)
            @argcheck(size(obj.a_rc, 1) == length(val),
                      DimensionMismatch("size(obj.a_rc, 1) ($(size(obj.a_rc, 1))) must match length(val) ($(length(val)))"))
        end
    end
    return setfield!(obj, sym, val)
end
function set_objective_function(port, ::Sharpe, ::Union{AKelly, EKelly}, custom_obj)
    model = port.model
    scale_obj = model[:scale_obj]
    ret = get_ret(model)
    JuMP.@expression(model, obj_func, ret)
    add_objective_penalty(model, obj_func, -1)
    custom_objective(port, obj_func, -1, custom_obj)
    JuMP.@objective(model, Max, scale_obj * obj_func)
    return nothing
end
function return_constraints(port, type, ::Any, log_ret::AKelly, mu, sigma, returns,
                            log_ret_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    w = get_w(model)
    fees = model[:fees]
    if isnothing(log_ret_approx_idx) ||
       isempty(log_ret_approx_idx) ||
       iszero(log_ret_approx_idx[1])
        if !haskey(model, :variance_risk)
            a_rc = log_ret.a_rc
            b_rc = log_ret.b_rc
            sdp_rc_variance(model, type, a_rc, b_rc)
            calc_variance_risk(get_ntwk_clust_type(port, a_rc, b_rc), log_ret.formulation,
                               model, mu, sigma, returns)
        end
        variance_risk = model[:variance_risk]
        JuMP.@expression(model, ret, LinearAlgebra.dot(mu, w) - fees - 0.5 * variance_risk)
    else
        variance_risk = model[:variance_risk]
        JuMP.@expression(model, ret,
                    LinearAlgebra.dot(mu, w) - fees - 0.5 * variance_risk[log_ret_approx_idx[1]])
    end

    return_bounds(port)

    return nothing
end
function return_constraints(port, type, obj::Sharpe, log_ret::AKelly, mu, sigma, returns,
                            log_ret_approx_idx)
    a_rc = log_ret.a_rc
    b_rc = log_ret.b_rc
    sdp_rc_variance(port.model, type, a_rc, b_rc)
    return_sharpe_alog_ret_constraints(port, type, obj, log_ret,
                                     get_ntwk_clust_type(port, a_rc, b_rc), mu, sigma,
                                     returns, log_ret_approx_idx)
    return nothing
end
function return_sharpe_alog_ret_constraints(port, type, obj::Sharpe, log_ret::AKelly,
                                          adjacency_constraint::Union{NoAdj, IP}, mu, sigma,
                                          returns, log_ret_approx_idx)
    if isempty(mu)
        return nothing
    end

    model = port.model
    get_fees(model)
    scale_constr = model[:scale_constr]
    w = get_w(model)
    k = get_k(model)
    fees = model[:fees]
    ohf = model[:ohf]
    risk = get_risk(model)
    rf = obj.rf
    JuMP.@variable(model, tapprox_log_ret)
    JuMP.@constraint(model, constr_sr_alog_ret_risk, scale_constr * risk <= scale_constr * ohf)
    JuMP.@expression(model, ret, LinearAlgebra.dot(mu, w) - fees - 0.5 * tapprox_log_ret - k * rf)
    if isnothing(log_ret_approx_idx) ||
       isempty(log_ret_approx_idx) ||
       iszero(log_ret_approx_idx[1])
        if !haskey(model, :variance_risk)
            calc_variance_risk(adjacency_constraint, log_ret.formulation, model, mu, sigma,
                               returns)
        end
        dev = model[:dev]
        JuMP.@constraint(model, constr_sr_alog_ret_ret,
                    [scale_constr * (k + tapprox_log_ret)
                     scale_constr * 2 * dev
                     scale_constr * (k - tapprox_log_ret)] ∈ JuMP.SecondOrderCone())
    else
        dev = model[:dev]
        JuMP.@constraint(model, constr_sr_alog_ret_ret,
                    [scale_constr * (k + tapprox_log_ret)
                     scale_constr * 2 * dev[log_ret_approx_idx[1]]
                     scale_constr * (k - tapprox_log_ret)] ∈ JuMP.SecondOrderCone())
    end
    return_bounds(port)

    return nothing
end
function return_sharpe_alog_ret_constraints(port, type, obj::Sharpe, ::AKelly, ::SDP, ::Any,
                                          ::Any, returns, ::Any)
    return_constraints(port, type, obj, EKelly(), nothing, nothing, returns, nothing)
    return nothing
end
=#
"""
    bounds_returns_estimator(r, lb::Number)

Return a copy of returns estimator `r` with its lower bound set to `lb`.

# Arguments

  - `r`: Returns estimator.
  - `lb::Number`: Lower bound on the portfolio return.

# Returns

  - Returns estimator with updated lower bound.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
"""
function bounds_returns_estimator(r::JuMPReturnsEstimator, lb::Number)
    return Accessors.@set r.lb = lb
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that minimises portfolio risk.

# Related

  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
struct MinimumRisk <: ObjectiveFunction end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises risk-adjusted utility.

# Mathematical definition

```math
\\begin{align}
\\max\\; \\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - \\tfrac{l}{2}\\, R(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``l``: Risk-aversion coefficient.
  - $(math_dict[:R_w])

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
\\max\\; \\frac{\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} - r_f}{R(\\boldsymbol{w})}\\,.
\\end{align}
```

Where:

  - $(math_dict[:mu_er])
  - $(math_dict[:w_port])
  - ``r_f``: Risk-free rate.
  - $(math_dict[:R_w])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing) -> MaximumRatio

Keywords correspond to the struct's fields.

## Validation

  - If `ohf` is provided: `ohf > 0`.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumReturn`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
@concrete struct MaximumRatio <: ObjectiveFunction
    """
    $(field_dict[:rf])
    """
    rf
    """
    Optional objective homogenisation factor for numerical stability of the ratio problem. Defaults to `nothing` (auto-determined).
    """
    ohf
    function MaximumRatio(rf::Number, ohf::Option{<:Number})
        if !isnothing(ohf)
            @argcheck(ohf > zero(ohf), DomainError(ohf, "ohf must be > 0"))
        end
        return new{typeof(rf), typeof(ohf)}(rf, ohf)
    end
end
function MaximumRatio(; rf::Number = 0.0, ohf::Option{<:Number} = nothing)
    return MaximumRatio(rf, ohf)
end
"""
$(DocStringExtensions.TYPEDEF)

Objective function that maximises portfolio return ``\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w}``.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
struct MaximumReturn <: ObjectiveFunction end
"""
    set_maximum_ratio_factor_variables!(model, mu, obj)
    set_maximum_ratio_factor_variables!(model, args...)

Set factor variables for the maximum ratio objective in the JuMP model.

Configures the normalisation factor (`ohf`) and the homogenisation variable (`k`) required for the maximum Sharpe or similar ratio objectives. The no-op overload sets `k` to `1` for non-ratio objectives.

# Arguments

  - `model`: JuMP optimisation model.
  - `mu`: Expected return vector or scalar.
  - `obj`: Objective function (e.g., [`MaximumRatio`](@ref)).
  - `args...`: Arguments (ignored in the fallback overload).

# Returns

  - `nothing`.

# Related

  - [`MaximumRatio`](@ref)
  - [`ObjectiveFunction`](@ref)
"""
function set_maximum_ratio_factor_variables!(model::JuMP.Model, mu::Num_VecNum,
                                             obj::MaximumRatio)
    ohf = if isnothing(obj.ohf)
        min(1e3, max(1e-3, Statistics.mean(abs.(mu))))
    else
        @argcheck(obj.ohf > zero(obj.ohf), DomainError(obj.ohf, "obj.ohf must be > 0"))
        obj.ohf
    end
    JuMP.@expression(model, ohf, ohf)
    JuMP.@variable(model, k >= 0)
    return nothing
end
function set_maximum_ratio_factor_variables!(model::JuMP.Model, args...)
    JuMP.@expression(model, k, 1)
    return nothing
end
"""
    set_return_bounds!(args...)
    set_return_bounds!(model::JuMP.Model, lb::Number)
    set_return_bounds!(model::JuMP.Model, lb::Front_NumVec)

Add a return lower bound constraint to the JuMP model.

The no-op fallback does nothing. With a scalar `lb`, adds a constraint `ret >= lb`. With a `Frontier` or vector `lb`, registers a return frontier expression for efficient frontier sweeps.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `lb`: Lower bound on portfolio return (scalar, vector, or `Frontier`).

# Returns

  - `nothing`.

# Related

  - [`set_return_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
"""
function set_return_bounds!(args...)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Number)
    sc = get_constraint_scale(model)
    k = get_k(model)
    ret = get_ret(model)
    JuMP.@constraint(model, ret_lb, sc * (ret - lb * k) >= 0)
    return nothing
end
function set_return_bounds!(model::JuMP.Model, lb::Front_NumVec)
    JuMP.@expression(model, ret_frontier, lb)
    return nothing
end
"""
    set_max_ratio_return_constraints!(args...)
    set_max_ratio_return_constraints!(model, obj, mu)

Set maximum ratio return constraints in the JuMP model.

Various overloads handle different optimiser types (e.g., [`MaximumRatio`](@ref) objective). No-op fallback when not applicable.

# Arguments

  - `args...`: JuMP model and optimiser parameters (ignored in no-op fallback).
  - `model`: JuMP optimisation model.
  - `obj`: Objective function.
  - `mu`: Expected return vector or scalar.

# Returns

  - `nothing`.

# Related

  - [`set_max_ratio_log_return_constraints!`](@ref)
  - [`MaximumRatio`](@ref)
"""
function set_max_ratio_return_constraints!(args...)
    return nothing
end
function set_max_ratio_return_constraints!(model::JuMP.Model, obj::MaximumRatio,
                                           mu::Num_VecNum)
    sc = get_constraint_scale(model)
    k = get_k(model)
    ohf = model[:ohf]
    ret = get_ret(model)
    rf = obj.rf
    if haskey(model, :bucs_w) || haskey(model, :t_eucs_gw) || all(x -> x <= rf, mu)
        risk = get_risk(model)
        JuMP.@constraint(model, sr_risk, sc * (risk - ohf) <= 0)
    else
        JuMP.@constraint(model, sr_ret, sc * (ret - rf * k - ohf) == 0)
    end
    return nothing
end
"""
    add_fees_to_ret!(model::JuMP.Model, ret)

Subtract the fees expression from the portfolio return expression in the JuMP model.

If no fees are registered in the model, does nothing.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ret`: JuMP return expression to modify in-place.

# Returns

  - `nothing`.

# Related

  - [`add_market_impact_cost!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_fees_to_ret!(model::JuMP.Model, ret)
    if !haskey(model, :fees)
        return nothing
    end
    fees = model[:fees]
    JuMP.add_to_expression!(ret, -fees)
    return nothing
end
"""
    add_market_impact_cost!(model::JuMP.Model, ret)

Subtract market impact costs from the portfolio return expression in the JuMP model.

If no market impact cost expression is registered in the model, does nothing.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ret`: JuMP return expression to modify in-place.

# Returns

  - `nothing`.

# Related

  - [`add_fees_to_ret!`](@ref)
  - [`set_return_constraints!`](@ref)
"""
function add_market_impact_cost!(model::JuMP.Model, ret)
    if !haskey(model, :wip)
        return nothing
    end
    cost_bgt_expr = model[:cost_bgt_expr]
    JuMP.add_to_expression!(ret, -cost_bgt_expr)
    return nothing
end
"""
    set_return_constraints!(model, pret, obj, pr; kwargs...)

Add portfolio return expression and associated constraints to the JuMP model.

Dispatches based on the return estimator type. Registers the `ret` expression, applies fees and market impact costs, configures maximum Sharpe ratio constraints, and adds return lower bounds.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `pret`: JuMP returns estimator ([`ArithmeticReturn`](@ref) or [`LogarithmicReturn`](@ref)).
  - `obj::ObjectiveFunction`: Portfolio objective function.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `kwargs...`: Additional keyword arguments (e.g. `rd` for uncertainty sets).

# Returns

  - `nothing`.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`set_return_bounds!`](@ref)
  - [`add_fees_to_ret!`](@ref)
"""
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{Nothing, <:Any, <:Any},
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    w = get_w(model)
    lb = pret.lb
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    JuMP.@expression(model, ret, dot_scalar(mu, w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
"""
    set_ucs_return_constraints!(model, ucs, mu)

Add uncertainty-set-robust return constraints to the JuMP model.

Dispatches based on the uncertainty set type. For `BoxUncertaintySet`, uses a norm-1 cone constraint. For `EllipsoidalUncertaintySet`, uses a second-order cone constraint.

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

Ellipsoidal uncertainty set (worst-case return):

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

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ucs`: Uncertainty set ([`BoxUncertaintySet`](@ref) or [`EllipsoidalUncertaintySet`](@ref)).
  - `mu`: Expected return vector.

# Returns

  - `nothing`.

# Related

  - [`set_return_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySet,
                                     mu::Num_VecNum)
    sc = get_constraint_scale(model)
    w = get_w(model)
    N = length(w)
    d_mu = (ucs.ub - ucs.lb) * 0.5
    JuMP.@variable(model, bucs_w[1:N])
    JuMP.@constraint(model, bucs_ret[i = 1:N],
                     [sc * bucs_w[i]; sc * w[i]] in JuMP.MOI.NormOneCone(2))
    JuMP.@expression(model, ret, dot_scalar(mu, w) - LinearAlgebra.dot(d_mu, bucs_w))
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
"""
    set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipsoidalUncertaintySet, mu::Num_VecNum)

Add ellipsoidal uncertainty-set-robust return constraints to the JuMP model.

Introduces a second-order cone constraint to model the worst-case expected return under an ellipsoidal uncertainty set for the mean vector.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ucs::EllipsoidalUncertaintySet`: Ellipsoidal uncertainty set with covariance `sigma` and radius `k`.
  - `mu::Num_VecNum`: Expected return vector.

# Returns

  - `nothing`.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`BoxUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipsoidalUncertaintySet,
                                     mu::Num_VecNum)
    sc = get_constraint_scale(model)
    w = get_w(model)
    G = LinearAlgebra.cholesky(ucs.sigma).U
    k = ucs.k
    JuMP.@expression(model, x_eucs_w, G * w)
    JuMP.@variable(model, t_eucs_gw)
    JuMP.@constraint(model, eucs_ret,
                     [sc * t_eucs_gw; sc * x_eucs_w] in JuMP.SecondOrderCone())
    JuMP.@expression(model, ret, dot_scalar(mu, w) - k * t_eucs_gw)
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
"""
    set_ucs_return_constraints!(model::JuMP.Model, ucs::L1UncertaintySet, mu::Num_VecNum)

Add ``\\ell_1``-uncertainty-set-robust return constraints to the JuMP model.

Introduces an infinity-norm cone constraint to model the worst-case characteristic under an ``\\ell_1`` uncertainty set. The constraint is linear, so the resulting model is an LP whenever the rest of the problem is (see [`NoRisk`](@ref)).

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

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ucs::L1UncertaintySet`: ``\\ell_1`` uncertainty set with radius `eps` and scaling `sd`.
  - `mu::Num_VecNum`: Expected return vector.

# Returns

  - `nothing`.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`L1UncertaintySet`](@ref)
  - [`CharacteristicUncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, ucs::L1UncertaintySet,
                                     mu::Num_VecNum)
    sc = get_constraint_scale(model)
    w = get_w(model)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    JuMP.@variable(model, t_l1ucs)
    JuMP.@constraint(model, l1ucs_ret,
                     [sc * t_l1ucs;
                      sc * sw] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@expression(model, ret, dot_scalar(mu, w) - ucs.eps * t_l1ucs)
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
"""
    set_ucs_return_constraints!(model::JuMP.Model, ucs::SignedL1UncertaintySet, mu::Num_VecNum)

Add signed-``\\ell_1``-uncertainty-set-robust return constraints to the JuMP model.

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

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `ucs::SignedL1UncertaintySet`: Signed ``\\ell_1`` uncertainty set with radii `ep`, `em` and scaling `sd`.
  - `mu::Num_VecNum`: Expected return vector.

# Returns

  - `nothing`.

# Related

  - [`set_ucs_return_constraints!`](@ref)
  - [`SignedL1UncertaintySet`](@ref)
  - [`L1UncertaintySet`](@ref)
"""
function set_ucs_return_constraints!(model::JuMP.Model, ucs::SignedL1UncertaintySet,
                                     mu::Num_VecNum)
    sc = get_constraint_scale(model)
    w = get_w(model)
    sd = ucs.sd
    sw = isnothing(sd) ? w : sd .* w
    JuMP.@variables(model, begin
                        t_sl1ucs_p >= 0
                        t_sl1ucs_m >= 0
                    end)
    JuMP.@constraints(model, begin
                          sl1ucs_ret_p, sc * (-sw .- t_sl1ucs_p) <= 0
                          sl1ucs_ret_m, sc * (sw .- t_sl1ucs_m) <= 0
                      end)
    JuMP.@expression(model, ret,
                     dot_scalar(mu, w) - ucs.ep * t_sl1ucs_p - ucs.em * t_sl1ucs_m)
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    return nothing
end
function set_return_constraints!(model::JuMP.Model,
                                 pret::ArithmeticReturn{<:UcSE_UcS, <:Any, <:Any},
                                 obj::ObjectiveFunction, pr::AbstractPriorResult;
                                 rd::ReturnsResult, kwargs...)
    lb = pret.lb
    ucs = pret.ucs
    mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)
    set_ucs_return_constraints!(model, mu_ucs(ucs, rd; kwargs...), mu)
    set_max_ratio_return_constraints!(model, obj, mu)
    set_return_bounds!(model, lb)
    return nothing
end
"""
    set_max_ratio_log_return_constraints!(args...)
    set_max_ratio_log_return_constraints!(model, ::MaximumRatio)

Set maximum ratio log-return constraints in the JuMP model.

Various overloads handle different optimiser types with logarithmic return objectives. No-op fallback when not applicable.

# Arguments

  - `args...`: Arguments (ignored in no-op fallback).
  - `model`: JuMP optimisation model.

# Returns

  - `nothing`.

# Related

  - [`set_max_ratio_return_constraints!`](@ref)
  - [`MaximumRatio`](@ref)
"""
function set_max_ratio_log_return_constraints!(args...)
    return nothing
end
function set_max_ratio_log_return_constraints!(model::JuMP.Model, ::MaximumRatio)
    sc = get_constraint_scale(model)
    ohf = model[:ohf]
    risk = get_risk(model)
    JuMP.@constraint(model, sr_elog_ret_risk, sc * (risk - ohf) <= 0)
end
function set_return_constraints!(model::JuMP.Model, pret::LogarithmicReturn,
                                 obj::ObjectiveFunction, pr::AbstractPriorResult; kwargs...)
    k = get_k(model)
    sc = get_constraint_scale(model)
    lb = pret.lb
    X = set_portfolio_returns!(model, pr.X)
    T = length(X)
    JuMP.@variable(model, t_elog_ret[1:T])
    wi = nothing_scalar_array_selector(pret.w, pr.w)
    wi = get_observation_weights(wi, X)
    if isnothing(wi)
        JuMP.@expression(model, ret, Statistics.mean(t_elog_ret))
    else
        JuMP.@expression(model, ret, Statistics.mean(t_elog_ret, wi))
    end
    add_fees_to_ret!(model, ret)
    add_market_impact_cost!(model, ret)
    set_max_ratio_log_return_constraints!(model, obj)
    JuMP.@expression(model, kret, k .+ X)
    JuMP.@constraint(model, elog_ret_ret[i = 1:T],
                     [sc * t_elog_ret[i], sc * k, sc * kret[i]] in
                     JuMP.MOI.ExponentialCone())
    set_return_bounds!(model, lb)
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
    op = if !haskey(model, :op) && isa(expr, JuMP.AffExpr)
        JuMP.@expression(model, op, zero(JuMP.AffExpr))
    elseif !haskey(model, :op) && isa(expr, JuMP.QuadExpr)
        JuMP.@expression(model, op, zero(JuMP.QuadExpr))
    elseif haskey(model, :op)
        model[:op]
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
    if !haskey(model, :op)
        return expr
    end
    op = model[:op]
    if !isa(expr, JuMP.QuadExpr) && isa(op, JuMP.QuadExpr)
        JuMP.unregister(model, :obj_expr)
        expr = JuMP.@expression(model, obj_expr, JuMP.QuadExpr(expr))
    end
    JuMP.add_to_expression!(expr, factor, op)
    return expr
end
"""
    set_portfolio_objective_function!(model, obj, pret, cobj, opt, pr, args...)

Set the portfolio objective function in the JuMP model.

Dispatches based on the objective function type to configure the appropriate JuMP objective expression. Handles minimum risk, maximum utility, maximum Sharpe ratio, and maximum return objectives, including penalty terms and custom objective contributions.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `obj::ObjectiveFunction`: Portfolio objective (e.g. [`MinimumRisk`](@ref), [`MaximumUtility`](@ref)).
  - `pret::JuMPReturnsEstimator`: Returns estimator.
  - `cobj::Option{<:CustomJuMPObjective}`: Optional custom objective term.
  - `opt::JuMPOptimisationEstimator`: JuMP optimiser configuration.
  - `pr::AbstractPriorResult`: Prior result.

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
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, args...)
    so = get_objective_scale(model)
    risk = get_risk(model)
    JuMP.@expression(model, obj_expr, risk)
    obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumUtility,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, args...)
    so = get_objective_scale(model)
    ret = get_ret(model)
    risk = get_risk(model)
    l = obj.l
    JuMP.@expression(model, obj_expr, ret - l * risk)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::LogarithmicReturn,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, args...)
    so = get_objective_scale(model)
    ret = get_ret(model)
    k = get_k(model)
    rf = obj.rf
    JuMP.@expression(model, obj_expr, ret - rf * k)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumRatio,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, args...)
    so = get_objective_scale(model)
    if haskey(model, :sr_risk)
        ret = get_ret(model)
        k = get_k(model)
        rf = obj.rf
        JuMP.@expression(model, obj_expr, ret - rf * k)
        obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
        add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
        JuMP.@objective(model, Max, so * obj_expr)
    else
        risk = get_risk(model)
        JuMP.@expression(model, obj_expr, risk)
        obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
        add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
        JuMP.@objective(model, Min, so * obj_expr)
    end
    return nothing
end
function set_portfolio_objective_function!(model::JuMP.Model, obj::MaximumReturn,
                                           pret::JuMPReturnsEstimator,
                                           cobj::Option{<:CustomJuMPObjective},
                                           opt::JuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, args...)
    so = get_objective_scale(model)
    ret = get_ret(model)
    JuMP.@expression(model, obj_expr, ret)
    obj_expr = add_penalty_to_objective!(model, -1, obj_expr)
    add_custom_objective_term!(model, obj, pret, cobj, obj_expr, opt, pr, args...)
    JuMP.@objective(model, Max, so * obj_expr)
    return nothing
end

export ArithmeticReturn, LogarithmicReturn, MinimumRisk, MaximumUtility, MaximumRatio,
       MaximumReturn, bounds_returns_estimator
