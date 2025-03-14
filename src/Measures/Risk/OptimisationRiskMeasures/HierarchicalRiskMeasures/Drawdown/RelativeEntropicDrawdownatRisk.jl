struct RelativeEntropicDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real,
                                      T3 <:
                                      Union{Nothing, <:Solver, <:AbstractVector{Solver}}} <:
       SolverHierarchicalRiskMeasure
    settings::T1
    alpha::T2
    solvers::T3
end
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        alpha::Real = 0.05,
                                        solvers::Union{Nothing, <:Solver,
                                                       <:AbstractVector{<:Solver}} = nothing)
    if isa(solvers, AbstractVector)
        @smart_assert(!isempty(solvers))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return RelativeEntropicDrawdownatRisk{typeof(settings), typeof(alpha), typeof(solvers)}(settings,
                                                                                            alpha,
                                                                                            solvers)
end
function (r::RelativeEntropicDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(eltype(dd))
    end
    popfirst!(dd)
    return ERM(dd, r.solvers, r.alpha)
end
function risk_measure_factory(r::RelativeEntropicDrawdownatRisk;
                              solvers::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              kwargs...)
    solvers = risk_measure_solver_factory(r.solvers, solvers)
    return RelativeEntropicDrawdownatRisk(; settings = r.settings, alpha = r.alpha,
                                          solvers = solvers)
end
function cluster_risk_measure_factory(r::RelativeEntropicDrawdownatRisk;
                                      solvers::Union{Nothing, <:Solver,
                                                     <:AbstractVector{<:Solver}}, kwargs...)
    return risk_measure_factory(r; solvers = solvers, kwargs = kwargs)
end

export RelativeEntropicDrawdownatRisk
