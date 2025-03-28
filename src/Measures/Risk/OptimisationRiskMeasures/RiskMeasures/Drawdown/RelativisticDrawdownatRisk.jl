struct RelativisticDrawdownatRisk{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                                  T4 <:
                                  Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T1
    kappa::T2
    slv::T4
end
function RelativisticDrawdownatRisk(; settings = RiskMeasureSettings(), alpha::Real = 0.05,
                                    kappa = 0.3,
                                    slv::Union{Nothing, <:Solver,
                                               <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativisticDrawdownatRisk{typeof(settings), typeof(alpha), typeof(kappa),
                                      typeof(slv)}(settings, alpha, kappa, slv)
end
function (r::RelativisticDrawdownatRisk)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    return RRM(dd, r.slv, r.alpha, r.kappa)
end

export RelativisticDrawdownatRisk
