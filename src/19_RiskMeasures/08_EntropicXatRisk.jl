function ERM(x::AbstractVector{<:Real}, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
             alpha::Real = 0.05, w::WeightsType = nothing)
    if isa(slv, AbstractVector)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    @variables(model, begin
                   t
                   z >= 0
                   u[1:T]
               end)
    aT = if isnothing(w)
        @constraints(model, begin
                         sum(u) - z <= 0
                         [i = 1:T], [-x[i] - t, z, u[i]] in MOI.ExponentialCone()
                     end)
        alpha * T
    else
        @constraints(model, begin
                         dot(w, u) - z <= 0
                         [i = 1:T], [-x[i] - t, z, u[i]] in MOI.ExponentialCone()
                     end)
        alpha * sum(w)
    end
    @expression(model, risk, t - z * log(aT))
    @objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        objective_value(model)
    else
        NaN
    end
end
struct EntropicValueatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    w::T4
    function EntropicValueatRisk(settings::RiskMeasureSettings,
                                 slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                                 alpha::Real, w::WeightsType)
        if isa(slv, AbstractVector)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(w)}(settings, slv,
                                                                            alpha, w)
    end
end
function EntropicValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                             alpha::Real = 0.05, w::WeightsType = nothing)
    return EntropicValueatRisk(settings, slv, alpha, w)
end
function (r::EntropicValueatRisk)(x::AbstractVector)
    return ERM(x, r.slv, r.alpha, r.w)
end
function factory(r::EntropicValueatRisk, prior::AbstractPriorResult,
                 slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}, args...;
                 kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    slv = solver_factory(r.slv, slv)
    return EntropicValueatRisk(; settings = r.settings, slv = slv, alpha = r.alpha, w = w)
end
struct EntropicValueatRiskRange{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    beta::T4
    w::T5
    function EntropicValueatRiskRange(settings::RiskMeasureSettings,
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}}, alpha::Real,
                                      beta::Real, w::WeightsType)
        if isa(slv, AbstractVector)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                                          slv,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w)
    end
end
function EntropicValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                  slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                                  alpha::Real = 0.05, beta::Real = 0.05,
                                  w::WeightsType = nothing)
    return EntropicValueatRiskRange(settings, slv, alpha, beta, w)
end
function (r::EntropicValueatRiskRange)(x::AbstractVector)
    return ERM(x, r.slv, r.alpha, r.w) + ERM(-x, r.slv, r.beta, r.w)
end
function factory(r::EntropicValueatRiskRange, prior::AbstractPriorResult,
                 slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}}, args...;
                 kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    slv = solver_factory(r.slv, slv)
    return EntropicValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                    beta = r.beta, w = w)
end
struct EntropicDrawdownatRisk{T1, T2, T3} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    function EntropicDrawdownatRisk(settings::RiskMeasureSettings,
                                    slv::Union{Nothing, <:Solver,
                                               <:AbstractVector{<:Solver}}, alpha::Real)
        if isa(slv, AbstractVector)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(slv), typeof(alpha)}(settings, slv, alpha)
    end
end
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                                alpha::Real = 0.05)
    return EntropicDrawdownatRisk(settings, slv, alpha)
end
function (r::EntropicDrawdownatRisk)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = -(peak - i)
    end
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, r.slv, r.alpha)
end
struct RelativeEntropicDrawdownatRisk{T1, T2, T3} <: HierarchicalRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    function RelativeEntropicDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                            slv::Union{Nothing, <:Solver,
                                                       <:AbstractVector{<:Solver}},
                                            alpha::Real)
        if isa(slv, AbstractVector)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(slv), typeof(alpha)}(settings, slv, alpha)
    end
end
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        slv::Union{Nothing, <:Solver,
                                                   <:AbstractVector{<:Solver}} = nothing,
                                        alpha::Real = 0.05)
    return RelativeEntropicDrawdownatRisk(settings, slv, alpha)
end
function (r::RelativeEntropicDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(eltype(dd))
    end
    popfirst!(x)
    popfirst!(dd)
    return ERM(dd, r.slv, r.alpha)
end
for r in (EntropicDrawdownatRisk, RelativeEntropicDrawdownatRisk)
    eval(quote
             function factory(r::$(r), ::Any,
                              slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...; kwargs...)
                 slv = solver_factory(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, slv = slv)
             end
         end)
end

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
