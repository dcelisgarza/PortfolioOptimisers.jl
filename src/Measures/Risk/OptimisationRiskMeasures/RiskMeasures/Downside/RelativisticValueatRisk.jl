struct RelativisticValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                               T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    kappa::T3
    solvers::T4
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 alpha::Real = 0.05, kappa::Real = 0.3,
                                 solvers::Union{Nothing, <:Solver,
                                                <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativisticValueatRisk{typeof(settings), typeof(alpha), typeof(kappa),
                                   typeof(solvers)}(settings, alpha, kappa, solvers)
end
function (r::RelativisticValueatRisk)(x::AbstractVector)
    return RRM(x, r.solvers, r.alpha, r.kappa)
end
function risk_measure_factory(r::RelativisticValueatRisk;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return RelativisticValueatRisk(; settings = r.settings, alpha = r.alpha,
                                   kappa = r.kappa, solvers = solvers)
end
function cluster_risk_measure_factory(r::RelativisticValueatRisk;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export RelativisticValueatRisk
