struct EntropicDrawdownatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                              T3 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    solvers::T3
end
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Real = 0.05,
                                solvers::Union{Nothing, <:Solver,
                                               <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EntropicDrawdownatRisk{typeof(settings), typeof(alpha), typeof(solvers)}(settings,
                                                                                    alpha,
                                                                                    solvers)
end
function (r::EntropicDrawdownatRisk)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, r.solvers, r.alpha)
end
function risk_measure_factory(r::EntropicDrawdownatRisk;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return EntropicDrawdownatRisk(; settings = r.settings, alpha = r.alpha,
                                  solvers = solvers)
end
function cluster_risk_measure_factory(r::EntropicDrawdownatRisk;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export EntropicDrawdownatRisk
