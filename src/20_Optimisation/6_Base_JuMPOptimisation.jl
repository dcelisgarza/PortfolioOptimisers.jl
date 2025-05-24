abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
abstract type JuMPOptimisationEstimator <: OptimisationEstimator end
abstract type ObjectiveFunction <: AbstractEstimator end
abstract type JuMPReturnsEstimator <: AbstractEstimator end
function jump_returns_factory(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
function jump_returns_view(r::JuMPReturnsEstimator, args...; kwargs...)
    return r
end
abstract type JuMPConstraintEstimator <: AbstractEstimator end
abstract type CustomConstraint <: JuMPConstraintEstimator end
abstract type CustomObjective <: JuMPConstraintEstimator end
function custom_constraint_view(::Nothing, args...; kwargs...)
    return nothing
end
function custom_constraint_view(::CustomConstraint, args...; kwargs...)
    return nothing
end
function custom_objective_view(::Nothing, args...; kwargs...)
    return nothing
end
function custom_objective_view(::CustomObjective, args...; kwargs...)
    return nothing
end
function add_custom_objective_term!(args...; kwargs...)
    return nothing
end
function add_custom_constraint!(args...; kwargs...)
    return nothing
end
struct JuMPOptimisationSolution{T1 <: AbstractVector} <: OptimisationModelResult
    w::T1
end
function JuMPOptimisationSolution(; w::AbstractVector)
    return JuMPOptimisationSolution{typeof(w)}(w)
end
function add_to_objective_penalty!(model::JuMP.Model, expr)
    op = if !haskey(model, :op)
        @expression(model, op, zero(AffExpr))
    else
        model[:op]
    end
    add_to_expression!(op, expr)
    return nothing
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
function set_initial_w!(w::AbstractVector, wi::AbstractVector)
    @smart_assert(length(wi) == length(w))
    set_start_value.(w, wi)
    return nothing
end
function set_w!(model::JuMP.Model, X::AbstractMatrix, wi::Union{Nothing, <:AbstractVector})
    @variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::SumScalariser)
    risk_vec = model[:risk_vec]
    if any(isa.(risk_vec, QuadExpr))
        @expression(model, risk, zero(QuadExpr))
    else
        @expression(model, risk, zero(AffExpr))
    end
    for risk_i ∈ risk_vec
        add_to_expression!(risk, risk_i)
    end
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, sce::LogSumExpScalariser)
    risk_vec = model[:risk_vec]
    sc = model[:sc]
    N = length(risk_vec)
    gamma = sce.gamma
    @variables(model, begin
                   risk
                   u_risk[1:N]
               end)
    @constraints(model,
                 begin
                     u_risk_lse, sc * (sum(u_risk) - 1) <= 0
                     risk_lse[i = 1:N],
                     [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
                     MOI.ExponentialCone()
                 end)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    risk_vec = model[:risk_vec]
    @variable(model, risk)
    @constraint(model, risk_ms, risk_vec .- risk <= 0)
    return nothing
end
function set_risk_constraints!(args...)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, r::RiskMeasure,
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult},
                               nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult}, args...)
    set_risk_constraints!(model, 1, r, opt, pr, cplg, nplg, args...)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, rs::AbstractVector{<:RiskMeasure},
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult},
                               nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult}, args...)
    for (i, r) ∈ enumerate(rs)
        set_risk_constraints!(model, i, r, opt, pr, cplg, nplg, args...)
    end
    return nothing
end
function process_model(model::JuMP.Model, ::JuMPOptimisationEstimator)
    if termination_status(model) == JuMP.OPTIMIZE_NOT_CALLED
        return JuMPOptimisationSolution(; w = fill(NaN, length(model[:w])))
    end
    ik = inv(value(model[:k]))
    w = value.(model[:w]) * ik
    return JuMPOptimisationSolution(; w = w)
end
function optimise_JuMP_model!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                              datatype::Type = Float64)
    trials = Dict()
    success = false
    for solver ∈ opt.opt.slv
        name = solver.name
        solver_i = solver.solver
        settings = solver.settings
        add_bridges = solver.add_bridges
        check_sol = solver.check_sol
        set_optimizer(model, solver_i; add_bridges = add_bridges)
        if !isnothing(settings) && !isempty(settings)
            for (k, v) ∈ settings
                set_attribute(model, k, v)
            end
        end
        try
            JuMP.optimize!(model)
        catch jump_error
            @warn("Failed to solve optimisation problem.")
            println(jump_error)
            push!(trials, name => Dict(:jump_error => jump_error))
            continue
        end
        all_finite_weights = all(isfinite, value.(model[:w]))
        all_non_zero_weights = !all(isapprox.(abs.(value.(model[:w])), zero(datatype)))
        try
            assert_is_solved_and_feasible(model; check_sol...)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            @warn("Failed to solve optimisation problem.")
            println(err.msg)
            push!(trials,
                  name => Dict(:objective_val => objective_value(model), :err => err,
                               :settings => settings))
        end
        push!(trials,
              name => Dict(:objective_val => objective_value(model),
                           :err => solution_summary(model), :settings => settings))
    end
    retcode = if success
        OptimisationSuccess(; res = trials)
    else
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
