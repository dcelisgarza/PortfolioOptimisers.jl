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
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EntropicValueatRisk{typeof(settings), typeof(alpha), typeof(solvers)}(settings,
                                                                                 alpha,
                                                                                 solvers)
end
function (r::EntropicValueatRisk)(x::AbstractVector)
    return ERM(x, r.solvers, r.alpha)
end

export EntropicValueatRisk
