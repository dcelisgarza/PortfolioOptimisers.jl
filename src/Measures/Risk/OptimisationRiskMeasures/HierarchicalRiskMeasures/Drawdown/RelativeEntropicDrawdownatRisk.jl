struct RelativeEntropicDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real,
                                      T3 <:
                                      Union{Nothing, <:Solver, <:AbstractVector{Solver}}} <:
       SolverHierarchicalRiskMeasure
    settings::T1
    alpha::T2
    slv::T3
end
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        alpha::Real = 0.05,
                                        slv::Union{Nothing, <:Solver,
                                                   <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return RelativeEntropicDrawdownatRisk{typeof(settings), typeof(alpha), typeof(slv)}(settings,
                                                                                        alpha,
                                                                                        slv)
end
function (r::RelativeEntropicDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(eltype(dd))
    end
    popfirst!(dd)
    return ERM(dd, r.slv, r.alpha)
end

export RelativeEntropicDrawdownatRisk
