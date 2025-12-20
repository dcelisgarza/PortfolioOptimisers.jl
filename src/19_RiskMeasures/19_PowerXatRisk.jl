# https://github.com/oxfordcontrol/Clarabel.jl/blob/4915b83e0d900d978681d5e8f3a3a5b8e18086f0/warmstart_test/portfolioOpt/higherorderRiskMeansure.jl#L23
function PRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, p::Number = 2.0,
             w::Option{<:AbstractWeights} = nothing)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    T = length(x)
    ip = inv(p)
    iaT = if isnothing(w)
        @constraint(model, sum(a) - b == 0)
        inv(alpha * T^ip)
    else
        @constraint(model, dot(w, a) - b == 0)
        inv(alpha * sum(w)^ip)
    end
    @variables(model, begin
                   eta
                   b
                   c[1:T] >= 0
                   a[1:T]
               end)
    @constraints(model, begin
                     (x + c) .+ eta >= 0
                     [i = 1:T], [a[i], b, c[i]] in MOI.PowerCone(ip)
                 end)
    @objective(model, Min, eta + iaT * b)
    return if optimise_JuMP_model!(model, slv).success
        objective_value(model)
    else
        NaN
    end
end
struct PowerValueatRisk{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    p::T4
    w::T5
    function PowerValueatRisk(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                              alpha::Number, p::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(p > one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(p), typeof(w)}(settings,
                                                                                       slv,
                                                                                       alpha,
                                                                                       p, w)
    end
end
function PowerValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                          p::Number = 2.0, w::Option{<:AbstractWeights} = nothing)
    return PowerValueatRisk(settings, slv, alpha, p, w)
end
function (r::PowerValueatRisk)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.p, r.w)
end
function factory(r::PowerValueatRisk, pr::AbstractPriorResult, slv::Option{<:Slv_VecSlv},
                 args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerValueatRisk(; settings = r.settings, slv = slv, alpha = r.alpha, p = r.p,
                            w = w)
end
struct PowerValueatRiskRange{T1, T2, T3, T4, T5, T6, T7} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    beta::T4
    pa::T5
    pb::T6
    w::T7
    function PowerValueatRiskRange(settings::RiskMeasureSettings, slv::Option{<:Slv_VecSlv},
                                   alpha::Number, beta::Number, pa::Number, pb::Number,
                                   w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(p > one(p))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(beta), typeof(pa),
                   typeof(pb), typeof(w)}(settings, slv, alpha, beta, pa, pb, w)
    end
end
function PowerValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                               slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                               beta::Number = 0.05, pa::Number = 2.0, pb::Number = 2.0,
                               w::Option{<:AbstractWeights} = nothing)
    return PowerValueatRiskRange(settings, slv, alpha, beta, pa, pb, w)
end
function (r::PowerValueatRiskRange)(x::VecNum)
    return PRM(x, r.slv, r.alpha, r.pa, r.w) + PRM(-x, r.slv, r.beta, r.pb, r.w)
end
function factory(r::PowerValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return PowerValueatRiskRange(; settings = r.settings, slv = slv, alpha = r.alpha,
                                 beta = r.beta, pa = r.pa, pb = r.pb, w = w)
end
struct PowerDrawdownatRisk <: RiskMeasure end
struct RelativePowerDrawdownatRisk <: HierarchicalRiskMeasure end

export PowerValueatRisk, PowerValueatRiskRange, PowerDrawdownatRisk,
       RelativePowerDrawdownatRisk