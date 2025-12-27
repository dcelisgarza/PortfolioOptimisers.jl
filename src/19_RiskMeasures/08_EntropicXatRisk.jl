function ERM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05,
             w::Option{<:AbstractWeights} = nothing)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    T = length(x)
    JuMP.@variables(model, begin
                        t
                        z >= 0
                        u[1:T]
                    end)
    aT = if isnothing(w)
        JuMP.@constraint(model, sum(u) - z <= 0)
        alpha * T
    else
        JuMP.@constraint(model, LinearAlgebra.dot(w, u) - z <= 0)
        alpha * sum(w)
    end
    JuMP.@constraint(model, [i = 1:T], [-x[i] - t, z, u[i]] in JuMP.MOI.ExponentialCone())
    JuMP.@expression(model, risk, t - z * log(aT))
    JuMP.@objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        JuMP.objective_value(model)
    else
        NaN
    end
end
struct EntropicValueatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    w::T4
    function EntropicValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                 alpha::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
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
                             slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                             w::Option{<:AbstractWeights} = nothing)
    return EntropicValueatRisk(settings, slv, alpha, w)
end
function (r::EntropicValueatRisk)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w)
end
struct EntropicValueatRiskRange{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    beta::T4
    w::T5
    function EntropicValueatRiskRange(settings::RiskMeasureSettings,
                                      slv::Option{<:Slv_VecSlv}, alpha::Number,
                                      beta::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
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
                                  slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                  beta::Number = 0.05,
                                  w::Option{<:AbstractWeights} = nothing)
    return EntropicValueatRiskRange(settings, slv, alpha, beta, w)
end
function (r::EntropicValueatRiskRange)(x::VecNum)
    return ERM(x, r.slv, r.alpha, r.w) + ERM(-x, r.slv, r.beta, r.w)
end
function factory(r::EntropicValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return EntropicValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                    beta = r.beta, w = w)
end
struct EntropicDrawdownatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    w::T4
    function EntropicDrawdownatRisk(settings::RiskMeasureSettings,
                                    slv::Option{<:Slv_VecSlv}, alpha::Number,
                                    w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
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
function EntropicDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                w::Option{<:AbstractWeights} = nothing)
    return EntropicDrawdownatRisk(settings, slv, alpha, w)
end
function (r::EntropicDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end
struct RelativeEntropicDrawdownatRisk{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    w::T4
    function RelativeEntropicDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                            slv::Option{<:Slv_VecSlv}, alpha::Number,
                                            w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
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
function RelativeEntropicDrawdownatRisk(;
                                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                        slv::Option{<:Slv_VecSlv} = nothing,
                                        alpha::Number = 0.05,
                                        w::Option{<:AbstractWeights} = nothing)
    return RelativeEntropicDrawdownatRisk(settings, slv, alpha, w)
end
function (r::RelativeEntropicDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return ERM(dd, r.slv, r.alpha, r.w)
end
for r in (EntropicValueatRisk, EntropicDrawdownatRisk, RelativeEntropicDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv},
                              args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha, w = w)
             end
             function factory(r::$(r), slv::Slv_VecSlv; kwargs...)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha, w = r.w)
             end
         end)
end

export EntropicValueatRisk, EntropicValueatRiskRange, EntropicDrawdownatRisk,
       RelativeEntropicDrawdownatRisk
