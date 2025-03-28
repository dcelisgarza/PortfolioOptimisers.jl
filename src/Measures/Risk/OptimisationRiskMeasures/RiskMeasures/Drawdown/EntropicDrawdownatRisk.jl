struct EntropicDrawdownatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                              T3 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}} <:
       SolverRiskMeasure
    settings::T1
    alpha::T2
    slv::T3
end
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Real = 0.05,
                                slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return EntropicDrawdownatRisk{typeof(settings), typeof(alpha), typeof(slv)}(settings,
                                                                                alpha, slv)
end
function (r::EntropicDrawdownatRisk)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, r.slv, r.alpha)
end

export EntropicDrawdownatRisk
