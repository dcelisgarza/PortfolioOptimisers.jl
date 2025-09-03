abstract type BaseJuMPOptimisationEstimator <: BaseOptimisationEstimator end
abstract type JuMPOptimisationEstimator <: OptimisationEstimator end
abstract type ObjectiveFunction <: AbstractEstimator end
abstract type JuMPReturnsEstimator <: AbstractEstimator end
Base.length(::JuMPReturnsEstimator) = 1
Base.iterate(::JuMPReturnsEstimator, i = 1) = i <= 1 ? (i, nothing) : nothing
function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !haskey(model, :G)
        G = cholesky(pr.sigma).U
        @expression(model, G, G)
    end
    return model[:G]
end
function get_chol_or_sigma_pm(
    model::JuMP.Model,
    pr::Union{
        <:LowOrderPrior{
            <:Any,
            <:Any,
            <:Any,
            <:AbstractMatrix,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
        },
        <:HighOrderPrior{
            <:LowOrderPrior{
                <:Any,
                <:Any,
                <:Any,
                <:AbstractMatrix,
                <:Any,
                <:Any,
                <:Any,
                <:Any,
                <:Any,
                <:Any,
                <:Any,
                <:Any,
            },
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
            <:Any,
        },
    },
)
    if !haskey(model, :G)
        G = pr.chol
        @expression(model, G, G)
    end
    return model[:G]
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
end
function JuMPOptimisationSolution(; w::AbstractArray)
    return JuMPOptimisationSolution(w)
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
function set_initial_w!(w::AbstractVector, wi::AbstractVector{<:Real})
    @argcheck(length(wi) == length(w))
    set_start_value.(w, wi)
    return nothing
end
function set_w!(model::JuMP.Model, X::AbstractMatrix, wi::Union{Nothing,<:AbstractVector})
    @variable(model, w[1:size(X, 2)])
    set_initial_w!(w, wi)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::SumScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    if any(x -> isa(x, QuadExpr), risk_vec)
        @expression(model, risk, zero(QuadExpr))
    else
        @expression(model, risk, zero(AffExpr))
    end
    for risk_i in risk_vec
        add_to_expression!(risk, risk_i)
    end
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, sce::LogSumExpScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    sc = model[:sc]
    N = length(risk_vec)
    gamma = sce.gamma
    @variables(model, begin
        risk
        u_risk[1:N]
    end)
    @constraints(
        model,
        begin
            u_risk_lse, sc * (sum(u_risk) - 1) <= 0
            risk_lse[i=1:N],
            [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
            MOI.ExponentialCone()
        end
    )
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    @variable(model, risk)
    @constraint(model, risk_ms, risk .- risk_vec .>= 0)
    return nothing
end
function set_risk_constraints!(args...; kwargs...)
    return nothing
end
function set_risk_constraints!(
    model::JuMP.Model,
    r::RiskMeasure,
    opt::JuMPOptimisationEstimator,
    pr::AbstractPriorResult,
    cplg::Union{Nothing,<:SemiDefinitePhylogeny,<:IntegerPhylogeny},
    nplg::Union{Nothing,<:SemiDefinitePhylogeny,<:IntegerPhylogeny},
    args...;
    kwargs...,
)
    set_risk_constraints!(model, 1, r, opt, pr, cplg, nplg, args...; kwargs...)
    return nothing
end
function set_risk_constraints!(
    model::JuMP.Model,
    rs::AbstractVector{<:RiskMeasure},
    opt::JuMPOptimisationEstimator,
    pr::AbstractPriorResult,
    cplg::Union{Nothing,<:SemiDefinitePhylogeny,<:IntegerPhylogeny},
    nplg::Union{Nothing,<:SemiDefinitePhylogeny,<:IntegerPhylogeny},
    args...;
    kwargs...,
)
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, i, r, opt, pr, cplg, nplg, args...; kwargs...)
    end
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
function optimise_JuMP_model!(
    model::JuMP.Model,
    opt::JuMPOptimisationEstimator,
    datatype::DataType = Float64,
)
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
        all_non_zero_weights =
            !all(x -> isapprox(x, zero(datatype)), abs.(value.(model[:w])))
        try
            assert_is_solved_and_feasible(model; solver.check_sol...)
            if all_finite_weights && all_non_zero_weights
                success = true
                break
            end
        catch err
            trials[solver.name] =
                Dict(:assert_is_solved_and_feasible => err, :settings => solver.settings)
        end
        trials[solver.name] =
            Dict(:err => solution_summary(model), :settings => solver.settings)
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
