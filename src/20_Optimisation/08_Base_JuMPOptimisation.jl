abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
abstract type JuMPOptimisationEstimator <: OptimisationEstimator end
abstract type RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator end
abstract type ObjectiveFunction <: AbstractEstimator end
"""
"""
abstract type JuMPReturnsEstimator <: AbstractEstimator end
function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !haskey(model, :G)
        G = isnothing(pr.chol) ? cholesky(pr.sigma).U : pr.chol
        @expression(model, G, G)
    end
    return model[:G]
end
function get_chol_or_V_pm(model::JuMP.Model, pr::HighOrderPrior)
    if !haskey(model, :GV)
        G = cholesky(pr.V).U
        @expression(model, GV, G)
    end
    return model[:GV]
end
function jump_returns_factory(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
function jump_returns_view(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
abstract type JuMPConstraintEstimator <: AbstractConstraintEstimator end
abstract type CustomJuMPConstraint <: JuMPConstraintEstimator end
abstract type CustomJuMPObjective <: JuMPConstraintEstimator end
function custom_constraint_view(::Nothing, args...; kwargs...)
    return nothing
end
function custom_constraint_view(::CustomJuMPConstraint, args...; kwargs...)
    return nothing
end
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
struct JuMPOptimisationSolution{T1} <: OptimisationModelResult
    w::T1
    function JuMPOptimisationSolution(w::AbstractArray)
        @argcheck(!isempty(w))
        return new{typeof(w)}(w)
    end
end
function JuMPOptimisationSolution(; w::AbstractArray)
    return JuMPOptimisationSolution(w)
end
function set_model_scales!(model::JuMP.Model, so::Real, sc::Real)
    @expressions(model, begin
                     so, so
                     sc, sc
                 end)
    return nothing
end
function set_initial_w!(args...)
    return nothing
end
function set_initial_w!(w::AbstractVector, wi::AbstractVector{<:Real})
    @argcheck(length(wi) == length(w))
    set_start_value.(w, wi)
    return nothing
end
function set_w!(model::JuMP.Model, X::AbstractMatrix, wi::Union{Nothing, <:AbstractVector})
    @variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
function process_model(model::JuMP.Model, ::JuMPOptimisationEstimator)
    if termination_status(model) == JuMP.OPTIMIZE_NOT_CALLED
        return JuMPOptimisationSolution(; w = fill(NaN, length(model[:w])))
    end
    k = value(model[:k])
    ik = !iszero(k) ? inv(k) : 1
    w = value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::DataType = Float64)
    trials = Dict()
    success = false
    for solver in opt.opt.slv
        try
            set_optimizer(model, solver.solver; add_bridges = solver.add_bridges)
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
        all_finite_weights = all(isfinite, value.(model[:w]))
        all_non_zero_weights = !all(x -> isapprox(x, zero(datatype)),
                                    abs.(value.(model[:w])))
        try
            assert_is_solved_and_feasible(model; solver.check_sol...)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trials[solver.name] = Dict(:assert_is_solved_and_feasible => err,
                                       :settings => solver.settings)
        end
        trials[solver.name] = Dict(:err => solution_summary(model),
                                   :settings => solver.settings)
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
        @warn("Failed to solve optimisation problem. Check `retcode.res` for details.")
        OptimisationFailure(; res = trials)
    end
    return retcode, process_model(model, opt)
end
function set_portfolio_returns!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :X)
        return model[:X]
    end
    w = model[:w]
    @expression(model, X, X * w)
    return X
end
function set_net_portfolio_returns!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :net_X)
        return model[:net_X]
    end
    X = set_portfolio_returns!(model, X)
    if haskey(model, :fees)
        fees = model[:fees]
        @expression(model, net_X, X .- fees)
    else
        @expression(model, net_X, X)
    end
    return net_X
end
function set_portfolio_returns_plus_one!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :Xap1)
        return model[:Xap1]
    end
    @expression(model, Xap1, one(eltype(X)) .+ X)
    return Xap1
end
function scalarise_risk_expression! end
function set_risk_constraints! end

export JuMPOptimisationSolution
