struct EntropicValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                           T3 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    slv::T3
end
function EntropicValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             alpha::Real = 0.05,
                             slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EntropicValueatRisk{typeof(settings), typeof(alpha), typeof(slv)}(settings,
                                                                             alpha, slv)
end
function (r::EntropicValueatRisk)(x::AbstractVector)
    return ERM(x, r.slv, r.alpha)
end

export EntropicValueatRisk
