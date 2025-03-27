struct EntropicValueatRiskRange{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                                T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    solvers::T4
end
function EntropicValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alpha::Real = 0.05, beta::Real = 0.05,
                                  solvers::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return EntropicValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta),
                                    typeof(solvers)}(settings, alpha, beta, solvers)
end
function (r::EntropicValueatRiskRange)(x::AbstractVector)
    return ERM(x, r.solvers, r.alpha) + ERM(-x, r.solvers, r.beta)
end
function risk_measure_factory(r::EntropicValueatRiskRange, ::Any,
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return EntropicValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta,
                                    solvers = solvers)
end
function cluster_risk_measure_factory(r::EntropicValueatRiskRange, ::Any, ::Any,
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, args...)
    return risk_measure_factory(r, nothing, nothing, solvers, args...)
end

export EntropicValueatRiskRange
