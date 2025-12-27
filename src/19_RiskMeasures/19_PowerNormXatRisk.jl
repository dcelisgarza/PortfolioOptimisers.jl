# https://github.com/oxfordcontrol/Clarabel.jl/blob/4915b83e0d900d978681d5e8f3a3a5b8e18086f0/warmstart_test/portfolioOpt/higherorderRiskMeansure.jl#L23
function PRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, p::Number = 2.0,
             w::Option{<:AbstractWeights} = nothing)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, false)
    T = length(x)
    ip = inv(p)
    JuMP.@variables(model, begin
                        pvar_eta
                        pvar_t
                        pvar_w[1:T] >= 0
                        pvar_v[1:T]
                    end)
    iaT = if isnothing(w)
        JuMP.@constraint(model, sum(pvar_v) - pvar_t <= 0)
        inv(alpha * T^ip)
    else
        JuMP.@constraint(model, LinearAlgebra.dot(w, pvar_v) - pvar_t <= 0)
        inv(alpha * sum(w)^ip)
    end
    JuMP.@constraints(model,
                      begin
                          (x + pvar_w) .+ pvar_eta >= 0
                          [i = 1:T],
                          [pvar_v[i], pvar_t, pvar_w[i]] in JuMP.MOI.PowerCone(ip)
                      end)
    JuMP.@objective(model, Min, pvar_eta + iaT * pvar_t)
    return if optimise_JuMP_model!(model, slv).success
        JuMP.objective_value(model)
    else
        NaN
    end
end
struct PowerNormValueatRisk{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    p::T4
    w::T5
    function PowerNormValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                  alpha::Number, p::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                              slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                              p::Number = 2.0, w::Option{<:AbstractWeights} = nothing)
    return PowerNormValueatRisk(settings, slv, alpha, p, w)
end
function (r::PowerNormValueatRisk)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.p, r.w)
end
struct PowerNormValueatRiskRange{T1, T2, T3, T4, T5, T6, T7} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    beta::T4
    pa::T5
    pb::T6
    w::T7
    function PowerNormValueatRiskRange(settings::RiskMeasureSettings,
                                       slv::Option{<:Slv_VecSlv}, alpha::Number,
                                       beta::Number, pa::Number, pb::Number,
                                       w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(pa > one(pa))
        @argcheck(pb > one(pb))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(pa),
                   typeof(pb), typeof(w)}(settings, slv, alpha, beta, pa, pb, w)
    end
end
function PowerNormValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   slv::Option{<:Slv_VecSlv} = nothing,
                                   alpha::Number = 0.05, beta::Number = 0.05,
                                   pa::Number = 2.0, pb::Number = 2.0,
                                   w::Option{<:AbstractWeights} = nothing)
    return PowerNormValueatRiskRange(settings, slv, alpha, beta, pa, pb, w)
end
function (r::PowerNormValueatRiskRange)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.pa, r.w) + PRM(-x, r.slv, r.beta, r.pb, r.w)
end
function factory(r::PowerNormValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerNormValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                     beta = r.beta, pa = r.pa, pb = r.pb, w = w)
end
struct PowerNormDrawdownatRisk{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    p::T4
    w::T5
    function PowerNormDrawdownatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Number, p::Number,
                                     w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerNormDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                 p::Number = 2.0, w::Option{<:AbstractWeights} = nothing)
    return PowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
function (r::PowerNormDrawdownatRisk)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end
struct RelativePowerNormDrawdownatRisk{T1, T2, T3, T4, T5} <: HierarchicalRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    p::T4
    w::T5
    function RelativePowerNormDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                             slv::Option{<:Slv_VecSlv}, alpha::Number,
                                             p::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p >= one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function RelativePowerNormDrawdownatRisk(;
                                         settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                         slv::Option{<:Slv_VecSlv} = nothing,
                                         alpha::Number = 0.05, p::Number = 2.0,
                                         w::Option{<:AbstractWeights} = nothing)
    return RelativePowerNormDrawdownatRisk(settings, slv, alpha, p, w)
end
function (r::RelativePowerNormDrawdownatRisk)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return PRM(dd, r.slv, r.alpha, r.p, r.w)
end
for r in (PowerNormValueatRisk, PowerNormDrawdownatRisk, RelativePowerNormDrawdownatRisk)
    eval(quote
             function factory(r::$(r), pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv},
                              args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, slv = slv, alpha = r.alpha, p = r.p,
                             w = w)
             end
             function factory(r::$(r), slv::Slv_VecSlv; kwargs...)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             p = r.p, slv = slv)
             end
         end)
end

export PowerNormValueatRisk, PowerNormValueatRiskRange, PowerNormDrawdownatRisk,
       RelativePowerNormDrawdownatRisk
