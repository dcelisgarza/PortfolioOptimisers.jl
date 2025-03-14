struct RelativeRelativisticDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real,
                                          T3 <: Real,
                                          T4 <: Union{Nothing, <:Solver,
                                                      <:AbstractVector{<:Solver}}} <:
       SolverHierarchicalRiskMeasure
    settings::T1
    alpha::T2
    kappa::T3
    solvers::T4
end
function RelativeRelativisticDrawdownatRisk(;
                                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                            alpha::Real = 0.05, kappa = 0.3,
                                            solvers::Union{Nothing, <:Solver,
                                                           <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativeRelativisticDrawdownatRisk{typeof(settings), typeof(alpha),
                                              typeof(kappa), typeof(solvers)}(settings,
                                                                              alpha, kappa,
                                                                              solvers)
end
function (rldar_r::RelativeRelativisticDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ 1
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return RRM(dd, rldar_r.solvers, rldar_r.alpha, rldar_r.kappa)
end
function risk_measure_factory(r::RelativeRelativisticDrawdownatRisk;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return RelativeRelativisticDrawdownatRisk(; settings = r.settings, alpha = r.alpha,
                                              kappa = r.kappa, solvers = solvers)
end
function cluster_risk_measure_factory(r::RelativeRelativisticDrawdownatRisk;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export RelativeRelativisticDrawdownatRisk
