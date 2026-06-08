"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base JuMP-based portfolio optimisation estimators.

These are configuration-level types (e.g., `JuMPOptimiser`) that define the optimisation problem setup for JuMP-based optimisers.

# Related

  - [`BaseOptimisationEstimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based portfolio optimisation estimators.

JuMP optimisers formulate and solve portfolio optimisation problems using mathematical programming via the JuMP.jl framework.

# Related

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
"""
abstract type JuMPOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk-based JuMP portfolio optimisation estimators.

Subtype `RiskJuMPOptimisationEstimator` to implement optimisers that minimise or constrain risk measures as the primary objective.

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
"""
abstract type RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio objective functions.

Subtype `ObjectiveFunction` to implement portfolio optimisation objectives such as minimum risk, maximum return, or maximum Sharpe ratio.

# Related

  - [`MinimumRisk`](@ref)
  - [`MaximumReturn`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumUtility`](@ref)
"""
abstract type ObjectiveFunction <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based returns estimators used in optimisation models.

`JuMPReturnsEstimator` types define how expected returns are incorporated into JuMP models.

# Related

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
"""
abstract type JuMPReturnsEstimator <: AbstractEstimator end
"""
    jump_returns_view(r, args...; kwargs...)

Get a view or subset of JuMP returns estimator for slicing.

Returns the estimator sliced for a given asset cluster or returns it unchanged.

# Arguments

  - `r`: JuMP returns estimator.
  - `args...`: Additional arguments (index, etc.).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Sliced or unchanged returns estimator.

# Related

  - [`JuMPReturnsEstimator`](@ref)
"""
function jump_returns_view(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP constraint estimators.

Subtype `JuMPConstraintEstimator` to implement custom constraints or objectives for JuMP-based portfolio optimisers.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`CustomJuMPObjective`](@ref)
"""
abstract type JuMPConstraintEstimator <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP constraint implementations.

Implement `add_custom_constraint!` to define custom JuMP model constraints.

# Related

  - [`JuMPConstraintEstimator`](@ref)
  - [`CustomJuMPObjective`](@ref)
"""
abstract type CustomJuMPConstraint <: JuMPConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP objective implementations.

Implement `add_custom_objective_term!` to add custom terms to the JuMP model objective.

# Related

  - [`JuMPConstraintEstimator`](@ref)
  - [`CustomJuMPConstraint`](@ref)
"""
abstract type CustomJuMPObjective <: JuMPConstraintEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: custom JuMP constraints never require previous portfolio weights.
"""
function needs_previous_weights(::CustomJuMPConstraint)
    return false
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: custom JuMP objectives never require previous portfolio weights.
"""
function needs_previous_weights(::CustomJuMPObjective)
    return false
end
"""
    custom_constraint_view(cc, args...; kwargs...)

Get a view or subset of a custom JuMP constraint for slicing.

Returns `nothing` if no custom constraint is provided, or the constraint unchanged otherwise. Used in hierarchical optimisation to propagate custom constraints per cluster.

# Arguments

  - `cc`: Custom JuMP constraint or `nothing`.
  - `args...`: Additional arguments.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Constraint or `nothing`.

# Related

  - [`CustomJuMPConstraint`](@ref)
"""
function custom_constraint_view(::Nothing, args...; kwargs...)
    return nothing
end
function custom_constraint_view(::CustomJuMPConstraint, args...; kwargs...)
    return nothing
end
"""
    custom_objective_view(co, args...; kwargs...)

Get a view or subset of a custom JuMP objective term for slicing.

Returns `nothing` if no custom objective is provided, or the objective unchanged otherwise.

# Arguments

  - `co`: Custom JuMP objective or `nothing`.
  - `args...`: Additional arguments.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - Objective or `nothing`.

# Related

  - [`CustomJuMPObjective`](@ref)
"""
function custom_objective_view(::Nothing, args...; kwargs...)
    return nothing
end
function custom_objective_view(::CustomJuMPObjective, args...; kwargs...)
    return nothing
end
"""
    add_custom_objective_term!(args...; kwargs...)

Add a custom objective term to the JuMP model.

No-op fallback. Override this method for subtypes of [`CustomJuMPObjective`](@ref) to add custom penalty or reward terms to the JuMP model objective.

# Arguments

  - `args...`: JuMP model and custom objective type (ignored in fallback).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPObjective`](@ref)
  - [`add_custom_constraint!`](@ref)
"""
function add_custom_objective_term!(args...; kwargs...)
    return nothing
end
"""
    add_custom_constraint!(args...; kwargs...)

Add a custom constraint to the JuMP model.

No-op fallback. Override this method for subtypes of [`CustomJuMPConstraint`](@ref) to add custom constraints to the JuMP model.

# Arguments

  - `args...`: JuMP model and custom constraint type (ignored in fallback).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `nothing`.

# Related

  - [`CustomJuMPConstraint`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function add_custom_constraint!(args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the solution (portfolio weights) from a JuMP optimisation model.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JuMPOptimisationSolution(; w::ArrNum) -> JuMPOptimisationSolution

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.

# Related

  - [`OptimisationModelResult`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
@concrete struct JuMPOptimisationSolution <: OptimisationModelResult
    """
    $(field_dict[:pw])
    """
    w
    function JuMPOptimisationSolution(w::ArrNum)
        @argcheck(!isempty(w))
        return new{typeof(w)}(w)
    end
end
function JuMPOptimisationSolution(; w::ArrNum)::JuMPOptimisationSolution
    return JuMPOptimisationSolution(w)
end
"""
    set_model_scales!(model::JuMP.Model, so::Number, sc::Number)

Register objective scale `so` and constraint scale `sc` as named expressions in the JuMP model.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `so::Number`: Objective scale factor.
  - `sc::Number`: Constraint scale factor.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
"""
function set_model_scales!(model::JuMP.Model, so::Number, sc::Number)
    JuMP.@expressions(model, begin
                          so, so
                          sc, sc
                      end)
    return nothing
end
# ---------------------------------------------------------------------------
# Model-state read interface
#
# The JuMP model is the shared blackboard between every constraint and risk
# builder. These accessors give its always-present singleton entries a named,
# checked interface: each asserts the entry has been populated and returns it,
# turning an absent-key bug into a clear error at the read site instead of an
# opaque `KeyError` (or, worse, silent misbehaviour) deep in a builder.
#
# Prefer these over raw `model[:sym]` for the entries they cover. See ADR 0004.
# ---------------------------------------------------------------------------
"""
    get_constraint_scale(model::JuMP.Model)

Return the constraint scale expression `model[:sc]`.

Asserts the scale has been registered (via [`set_model_scales!`](@ref)); errors otherwise.

# Related

  - [`set_model_scales!`](@ref)
  - [`get_objective_scale`](@ref)
"""
function get_constraint_scale(model::JuMP.Model)
    @argcheck(haskey(model, :sc))
    return model[:sc]
end
"""
    get_objective_scale(model::JuMP.Model)

Return the objective scale expression `model[:so]`.

Asserts the scale has been registered (via [`set_model_scales!`](@ref)); errors otherwise.

# Related

  - [`set_model_scales!`](@ref)
  - [`get_constraint_scale`](@ref)
"""
function get_objective_scale(model::JuMP.Model)
    @argcheck(haskey(model, :so))
    return model[:so]
end
"""
    get_w(model::JuMP.Model)

Return the portfolio weight variables `model[:w]`.

Asserts the weights have been registered (via [`set_w!`](@ref)); errors otherwise.

# Related

  - [`set_w!`](@ref)
"""
function get_w(model::JuMP.Model)
    @argcheck(haskey(model, :w))
    return model[:w]
end
"""
    get_k(model::JuMP.Model)

Return the homogenisation variable `model[:k]`.

`k >= 0` is the auxiliary scaling variable used to homogenise fractional/ratio
objectives (e.g. maximum ratio); recovered weights are `w / k`. Asserts `:k` has
been registered; errors otherwise.

# Related

  - [`process_model`](@ref)
  - [`get_w`](@ref)
"""
function get_k(model::JuMP.Model)
    @argcheck(haskey(model, :k))
    return model[:k]
end
"""
    get_ret(model::JuMP.Model)

Return the portfolio expected-return expression `model[:ret]`.

Asserts the return expression has been registered; errors otherwise.

# Related

  - [`get_risk`](@ref)
"""
function get_ret(model::JuMP.Model)
    @argcheck(haskey(model, :ret))
    return model[:ret]
end
"""
    get_risk(model::JuMP.Model)

Return the scalarised portfolio risk expression `model[:risk]`.

Asserts the risk expression has been registered (via [`scalarise_risk_expression!`](@ref));
errors otherwise.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`get_ret`](@ref)
"""
function get_risk(model::JuMP.Model)
    @argcheck(haskey(model, :risk))
    return model[:risk]
end
"""
    set_initial_w!(args...)
    set_initial_w!(w::VecNum, wi::VecNum)

Set initial (warm-start) values for portfolio weight variables in the JuMP model.

The no-op fallback does nothing when `wi` is not provided. The two-argument method sets JuMP start values for each weight variable.

# Arguments

  - `w::VecNum`: Vector of JuMP weight variables.
  - `wi::VecNum`: Vector of initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::VecNum, wi::VecNum)
    @argcheck(length(wi) == length(w))
    JuMP.set_start_value.(w, wi)
    return nothing
end
"""
    set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})

Create portfolio weight variables in the JuMP model and optionally set initial values.

Registers a vector of weight variables `w` of length `size(X, 2)` in the model. If `wi` is provided, sets the initial values via [`set_initial_w!`](@ref).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (shape: observations × assets).
  - `wi`: Optional initial weight values.

# Returns

  - `nothing`.

# Related

  - [`set_initial_w!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_w!(model::JuMP.Model, X::MatNum, wi::Option{<:VecNum_VecVecNum})
    JuMP.@variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
"""
    process_model(model, retcode)

Extract the solution from an optimised JuMP model based on the return code.

On success, extracts the optimised weights from the model. On failure, returns an empty solution.

# Arguments

  - `model`: Optimised JuMP model.
  - `retcode`: Optimisation return code ([`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref)).

# Returns

  - Solution object.

# Related

  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function process_model(model::JuMP.Model, ::OptimisationSuccess)
    k = JuMP.value(model[:k])
    ik = !iszero(k) ? inv(k) : 1
    w = JuMP.value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function process_model(model::JuMP.Model, ::OptimisationFailure)
    return JuMPOptimisationSolution(; w = fill(NaN, length(model[:w])))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Attempt to solve the JuMP model using each solver in `opt.opt.slv` in order.

Tries each solver sequentially, checking feasibility and finite non-zero weights. Returns a `(retcode, solution)` tuple where `retcode` is [`OptimisationSuccess`](@ref) or [`OptimisationFailure`](@ref) and `solution` is a [`JuMPOptimisationSolution`](@ref).

# Related

  - [`JuMPOptimisationEstimator`](@ref)
  - [`JuMPOptimisationSolution`](@ref)
  - [`OptimisationSuccess`](@ref)
  - [`OptimisationFailure`](@ref)
"""
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::DataType = Float64)
    trials = Dict()
    success = false
    for solver in opt.opt.slv
        try
            JuMP.set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
        catch err
            trials[solver.name] = Dict(:set_optimizer => err)
            continue
        end
        set_solver_attributes(model, solver.settings)
        try
            JuMP.optimize!(model)
        catch err
            trials[solver.name] = Dict(:optimize! => err)
            continue
        end
        all_finite_weights = all(isfinite, JuMP.value.(model[:w]))
        all_non_zero_weights = !all(x -> isapprox(x, zero(datatype)),
                                    abs.(JuMP.value.(model[:w])))
        try
            JuMP.assert_is_solved_and_feasible(model; solver.check_sol...)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trials[solver.name] = Dict(:assert_is_solved_and_feasible => err,
                                       :settings => solver.settings)
        end
        trials[solver.name] = Dict(:err => JuMP.solution_summary(model),
                                   :settings => solver.settings)
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
        @warn("Failed to solve optimisation problem. Check `retcode.res` for details.")
        OptimisationFailure(; res = trials)
    end
    return retcode, process_model(model, retcode)
end
"""
    set_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register portfolio returns expression `X * w` in the JuMP model.

If the expression already exists in the model, returns it directly (idempotent).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The portfolio returns expression.

# Related

  - [`set_net_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_portfolio_returns!(model::JuMP.Model, X::MatNum)
    if haskey(model, :X)
        return model[:X]
    end
    w = model[:w]
    JuMP.@expression(model, X, X * w)
    return X
end
"""
    set_net_portfolio_returns!(model::JuMP.Model, X::MatNum)

Compute and register net portfolio returns (after fees) in the JuMP model.

Calls [`set_portfolio_returns!`](@ref) and subtracts fees if present.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix.

# Returns

  - The net portfolio returns expression.

# Related

  - [`set_portfolio_returns!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_net_portfolio_returns!(model::JuMP.Model, X::MatNum)
    if haskey(model, :net_X)
        return model[:net_X]
    end
    X = set_portfolio_returns!(model, X)
    if haskey(model, :fees)
        fees = model[:fees]
        JuMP.@expression(model, net_X, X .- fees)
    else
        JuMP.@expression(model, net_X, X)
    end
    return net_X
end
"""
    set_portfolio_returns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register portfolio gross returns `X .+ 1` in the JuMP model.

Used in drawdown and logarithmic return computations.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Portfolio returns expression.

# Returns

  - The gross returns expression `X .+ 1`.

# Related

  - [`set_portfolio_drawdowns_plus_one!`](@ref)
  - [`set_portfolio_returns!`](@ref)
"""
function set_portfolio_returns_plus_one!(model::JuMP.Model, X::MatNum)
    if haskey(model, :Xap1)
        return model[:Xap1]
    end
    JuMP.@expression(model, Xap1, X .+ one(eltype(X)))
    return Xap1
end
"""
    set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum)

Compute and register absolute drawdowns plus one in the JuMP model.

Computes `absolute_drawdown_arr(X) .+ 1` and registers it in the model.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `X::MatNum`: Portfolio returns expression.

# Returns

  - The drawdowns-plus-one expression.

# Related

  - [`set_portfolio_returns_plus_one!`](@ref)
"""
function set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum)
    if haskey(model, :ddap1)
        return model[:ddap1]
    end
    _ddap1 = absolute_drawdown_arr(X) .+ one(eltype(X))
    JuMP.@expression(model, ddap1, _ddap1)
    return ddap1
end
"""
    scalarise_risk_expression!(model, r, X, T, ...) -> nothing

Scalarise a risk expression and add it to the JuMP model objective.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method adds the appropriate risk objective term for a given risk measure type `r`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function scalarise_risk_expression! end
"""
    set_risk_constraints!(model, r, X, T, ...) -> nothing

Set risk constraints in the JuMP model for a given risk measure.

Generic function stub; concrete methods are defined in constraint and risk measure files. Each method configures the appropriate risk constraint expressions for a given risk measure type `r`.

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
function set_risk_constraints! end

export JuMPOptimisationSolution
