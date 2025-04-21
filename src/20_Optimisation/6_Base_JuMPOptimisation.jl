abstract type JuMPOptimisationEstimator <: OptimisationEstimator end
abstract type ObjectiveFunction <: AbstractEstimator end
abstract type ConstraintEstimator <: AbstractEstimator end
abstract type CustomConstraint <: ConstraintEstimator end
abstract type CustomObjective <: ConstraintEstimator end
abstract type JuMPReturnsEstimator <: AbstractEstimator end
function add_custom_objective_term!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                                    pr::AbstractPriorResult, args...)
    return nothing
end
function add_custom_constraint!(model::JuMP.Model, opt::JuMPOptimisationEstimator,
                                pr::AbstractPriorResult, args...)
    return nothing
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
    for r_risk ∈ risk_vec
        add_to_expression!(risk, r_risk)
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
                     u_risk_lse, sc * sum(u_risk) <= sc
                     risk_lse[i = 1:N],
                     [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
                     MOI.ExponentialCone()
                 end)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    risk_vec = model[:risk_vec]
    @variable(model, risk)
    @constraint(model, c_max_risk, risk .>= risk_vec)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, i::Integer, r::RiskMeasure,
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult},
                               nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult})
    return nothing
end
function set_risk_constraints!(model::JuMP.Model,
                               rs::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               cplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult},
                               nplg::Union{Nothing, <:SemiDefinitePhilogenyResult,
                                           <:IntegerPhilogenyResult})
    for (i, r) ∈ enumerate(rs)
        set_risk_constraints!(model, i, r, opt, pr, cplg, nplg)
    end
    return nothing
end
struct JuMPPortfolioResult{T1 <: AbstractVector, T2 <: JuMP.Model, T3 <: JuMPResult} <:
       AbstractResult
    w::T1
    model::T2
    result::T3
end
function process_model(model::JuMP.Model, ::JuMPOptimisationEstimator)
    return value.(model[:w]) / value(model[:k])
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
            push!(trials,
                  name => Dict(:objective_val => objective_value(model), :err => err,
                               :settings => settings))
        end
        push!(trials,
              name => Dict(:objective_val => objective_value(model),
                           :err => solution_summary(model), :settings => settings))
    end
    return JuMPPortfolioResult(process_model(model, opt), model,
                               JuMPResult(trials, success))
end
