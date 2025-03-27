struct RelativisticDrawdownatRisk{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                                  T4 <:
                                  Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T1
    kappa::T2
    solvers::T4
end
function RelativisticDrawdownatRisk(; settings = RiskMeasureSettings(), alpha::Real = 0.05,
                                    kappa = 0.3,
                                    solvers::Union{Nothing, <:Solver,
                                                   <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativisticDrawdownatRisk{typeof(settings), typeof(alpha), typeof(kappa),
                                      typeof(solvers)}(settings, alpha, kappa, solvers)
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
    return RRM(dd, r.solvers, r.alpha, r.kappa)
end

export RelativisticDrawdownatRisk
