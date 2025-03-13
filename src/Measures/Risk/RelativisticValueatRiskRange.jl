struct RelativisticValueatRiskRange{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                                    T4 <: Real, T5 <: Real,
                                    T6 <:
                                    Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    kappa_a::T3
    beta::T4
    kappa_b::T5
    solvers::T6
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      alpha::Real = 0.05, kappa_a::Real = 0.3,
                                      beta::Real = 0.05, kappa_b::Real = 0.3,
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa_a) < kappa_a < one(kappa_a))
    @smart_assert(zero(beta) < beta < one(beta))
    @smart_assert(zero(kappa_b) < kappa_b < one(kappa_b))
    return RelativisticValueatRiskRange{typeof(settings), typeof(alpha), typeof(kappa_a),
                                        typeof(beta), typeof(kappa_b), typeof(solvers)}(settings,
                                                                                        alpha,
                                                                                        kappa_a,
                                                                                        beta,
                                                                                        kappa_b,
                                                                                        solvers)
end
function (r::RelativisticValueatRiskRange)(x::AbstractVector)
    return RRM(x, r.solvers, r.alpha, r.kappa_a) + RRM(-x, r.solvers, r.beta, r.kappa_b)
end
function risk_measure_factory(r::RelativisticValueatRiskRange;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return RelativisticValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                        kappa_a = r.kappa_a, beta = r.beta,
                                        kappa_b = r.kappa_b, solvers = solvers)
end
function cluster_risk_measure_factory(r::RelativisticValueatRiskRange;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export RelativisticValueatRiskRange
