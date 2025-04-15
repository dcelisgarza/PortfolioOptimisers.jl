function ERM(x::AbstractVector{<:Real}, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
             alpha::Real = 0.05)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    @variables(model, begin
                   t
                   z >= 0
                   u[1:T]
               end)
    @constraints(model, begin
                     sum(u) <= z
                     [i = 1:T], [-x[i] - t, z, u[i]] ∈ MOI.ExponentialCone()
                 end)
    @expression(model, risk, t - z * log(alpha * T))
    @objective(model, Min, risk)
    (; trials, success) = optimise_JuMP_model!(model, slv)
    return if success
        objective_value(model)
    else
        @warn("Model could not be optimised satisfactorily.\nSolvers: $trials.")
        NaN
    end
end
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
function risk_measure_view(r::EntropicValueatRiskRange, ::Any, ::Any,
                           slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                           args...; kwargs...)
    return risk_measure_factory(r, nothing, nothing, slv, args...; kwargs...)
end
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

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
