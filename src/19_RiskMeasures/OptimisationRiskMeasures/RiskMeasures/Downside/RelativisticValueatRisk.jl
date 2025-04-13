struct RelativisticValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                               T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    kappa::T3
    slv::T4
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 alpha::Real = 0.05, kappa::Real = 0.3,
                                 slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativisticValueatRisk{typeof(settings), typeof(alpha), typeof(kappa),
                                   typeof(slv)}(settings, alpha, kappa, slv)
end
function (r::RelativisticValueatRisk)(x::AbstractVector)
    return RRM(x, r.slv, r.alpha, r.kappa)
end

export RelativisticValueatRisk
