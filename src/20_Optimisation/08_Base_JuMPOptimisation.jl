"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for base JuMP-based portfolio optimisation estimators.

These are configuration-level types (e.g., `JuMPOptimiser`) that define the optimisation problem setup for JuMP-based optimisers.

# Related Types

  - [`BaseOptimisationEstimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for JuMP-based portfolio optimisation estimators.

JuMP optimisers formulate and solve portfolio optimisation problems using mathematical programming via the JuMP.jl framework.

# Related Types

  - [`NonFiniteAllocationOptimisationEstimator`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
"""
abstract type JuMPOptimisationEstimator <: NonFiniteAllocationOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk-based JuMP portfolio optimisation estimators.

Subtype `RiskJuMPOptimisationEstimator` to implement optimisers that minimise or constrain risk measures as the primary objective.

# Related Types

  - [`JuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
"""
abstract type RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for portfolio objective functions.

Subtype `ObjectiveFunction` to implement portfolio optimisation objectives such as minimum risk, maximum return, or maximum Sharpe ratio.

# Related Types

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

# Related Types

  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
"""
abstract type JuMPReturnsEstimator <: AbstractEstimator end
function factory(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
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

# Related Types

  - [`CustomJuMPConstraint`](@ref)
  - [`CustomJuMPObjective`](@ref)
"""
abstract type JuMPConstraintEstimator <: AbstractConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP constraint implementations.

Implement `add_custom_constraint!` to define custom JuMP model constraints.

# Related Types

  - [`JuMPConstraintEstimator`](@ref)
  - [`CustomJuMPObjective`](@ref)
"""
abstract type CustomJuMPConstraint <: JuMPConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for custom JuMP objective implementations.

Implement `add_custom_objective_term!` to add custom terms to the JuMP model objective.

# Related Types

  - [`JuMPConstraintEstimator`](@ref)
  - [`CustomJuMPConstraint`](@ref)
"""
abstract type CustomJuMPObjective <: JuMPConstraintEstimator end
function needs_previous_weights(::CustomJuMPConstraint)
    return false
end
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
function add_custom_objective_term!(args...; kwargs...)
    return nothing
end
function add_custom_constraint!(args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Stores the solution (portfolio weights) from a JuMP optimisation model.

# Fields

  - `w`: Optimal portfolio weights (non-empty array of numbers).

# Related

  - [`OptimisationModelResult`](@ref)
  - [`JuMPOptimisationEstimator`](@ref)
"""
@concrete struct JuMPOptimisationSolution <: OptimisationModelResult
    w
    function JuMPOptimisationSolution(w::ArrNum)
        @argcheck(!isempty(w))
        return new{typeof(w)}(w)
    end
end
function JuMPOptimisationSolution(; w::ArrNum)
    return JuMPOptimisationSolution(w)
end
function set_model_scales!(model::JuMP.Model, so::Number, sc::Number)
    JuMP.@expressions(model, begin
                          so, so
                          sc, sc
                      end)
    return nothing
end
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::VecNum, wi::VecNum)
    @argcheck(length(wi) == length(w))
    JuMP.set_start_value.(w, wi)
    return nothing
end
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
function set_portfolio_returns!(model::JuMP.Model, X::MatNum)
    if haskey(model, :X)
        return model[:X]
    end
    w = model[:w]
    JuMP.@expression(model, X, X * w)
    return X
end
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
function set_portfolio_returns_plus_one!(model::JuMP.Model, X::MatNum)
    if haskey(model, :Xap1)
        return model[:Xap1]
    end
    JuMP.@expression(model, Xap1, X .+ one(eltype(X)))
    return Xap1
end
function set_portfolio_drawdowns_plus_one!(model::JuMP.Model, X::MatNum)
    if haskey(model, :ddap1)
        return model[:ddap1]
    end
    _ddap1 = absolute_drawdown_arr(X) .+ one(eltype(X))
    JuMP.@expression(model, ddap1, _ddap1)
    return ddap1
end
function scalarise_risk_expression! end
function set_risk_constraints! end

export JuMPOptimisationSolution
