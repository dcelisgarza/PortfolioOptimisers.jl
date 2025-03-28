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
    slv::T6
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      alpha::Real = 0.05, kappa_a::Real = 0.3,
                                      beta::Real = 0.05, kappa_b::Real = 0.3,
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa_a) < kappa_a < one(kappa_a))
    @smart_assert(zero(beta) < beta < one(beta))
    @smart_assert(zero(kappa_b) < kappa_b < one(kappa_b))
    return RelativisticValueatRiskRange{typeof(settings), typeof(alpha), typeof(kappa_a),
                                        typeof(beta), typeof(kappa_b), typeof(slv)}(settings,
                                                                                    alpha,
                                                                                    kappa_a,
                                                                                    beta,
                                                                                    kappa_b,
                                                                                    slv)
end
function (r::RelativisticValueatRiskRange)(x::AbstractVector)
    return RRM(x, r.slv, r.alpha, r.kappa_a) + RRM(-x, r.slv, r.beta, r.kappa_b)
end
function risk_measure_factory(r::RelativisticValueatRiskRange, ::Any,
                              slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...; kwargs...)
    slv = risk_measure_solver_factory(r.slv, slv)
    return RelativisticValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                        kappa_a = r.kappa_a, beta = r.beta,
                                        kappa_b = r.kappa_b, slv = slv)
end
function cluster_risk_measure_factory(r::RelativisticValueatRiskRange, ::Any, ::Any,
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}}, args...;
                                      kwargs...)
    return risk_measure_factory(r, nothing, nothing, slv, args...; kwargs...)
end

export RelativisticValueatRiskRange
