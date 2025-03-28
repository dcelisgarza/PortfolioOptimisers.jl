struct EntropicValueatRiskRange{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                                T4 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    slv::T4
end
function EntropicValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  alpha::Real = 0.05, beta::Real = 0.05,
                                  slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return EntropicValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta),
                                    typeof(slv)}(settings, alpha, beta, slv)
end
function (r::EntropicValueatRiskRange)(x::AbstractVector)
    return ERM(x, r.slv, r.alpha) + ERM(-x, r.slv, r.beta)
end
function risk_measure_factory(r::EntropicValueatRiskRange, ::Any,
                              slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...; kwargs...)
    slv = risk_measure_solver_factory(r.slv, slv)
    return EntropicValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta,
                                    slv = slv)
end
function cluster_risk_measure_factory(r::EntropicValueatRiskRange, ::Any, ::Any,
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}}, args...;
                                      kwargs...)
    return risk_measure_factory(r, nothing, nothing, slv, args...; kwargs...)
end

export EntropicValueatRiskRange
