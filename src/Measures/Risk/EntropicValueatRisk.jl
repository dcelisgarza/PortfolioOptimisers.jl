struct EntropicValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                           T3 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    solvers::T3
end
function EntropicValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             alpha::Real = 0.05,
                             solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EntropicValueatRisk{typeof(settings), typeof(alpha), typeof(solvers)}(settings,
                                                                                 alpha,
                                                                                 solvers)
end
function (r::EntropicValueatRisk)(x::AbstractVector)
    return ERM(x, r.solvers, r.alpha)
end
function risk_measure_factory(r::EntropicValueatRisk;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return EntropicValueatRisk(; settings = r.settings, alpha = r.alpha, solvers = solvers)
end
function cluster_risk_measure_factory(r::EntropicValueatRisk;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export EntropicValueatRisk
