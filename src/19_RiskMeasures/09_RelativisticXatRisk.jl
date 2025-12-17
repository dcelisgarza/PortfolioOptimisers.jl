function RRM(x::VecNum, slv::Slv_VecSlv, alpha::Number = 0.05, kappa::Number = 0.3,
             w::Option{<:AbstractWeights} = nothing)
    if isa(slv, VecSlv)
        @argcheck(!isempty(slv))
    end
    T = length(x)
    model = JuMP.Model()
    set_string_names_on_creation(model, false)
    @variables(model, begin
                   t
                   z >= 0
                   omega[1:T]
                   psi[1:T]
                   theta[1:T]
                   epsilon[1:T]
               end)
    if isnothing(w)
        invat = inv(alpha * T)
        invk2 = inv(2 * kappa)
        ln_k = (invat^kappa - invat^(-kappa)) * invk2
        @expression(model, risk, t + ln_k * z + sum(psi + theta))
    else
        invat = inv(alpha * sum(w))
        invk2 = inv(2 * kappa)
        ln_k = (invat^kappa - invat^(-kappa)) * invk2
        @expression(model, risk, t + ln_k * z + dot(w, psi + theta))
    end
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    invk = inv(kappa)
    invopk = inv(opk)
    invomk = inv(omk)
    @constraints(model,
                 begin
                     [i = 1:T],
                     [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] in
                     MOI.PowerCone(invopk)
                     [i = 1:T],
                     [omega[i] * invomk, theta[i] * invk, -z * invk2] in MOI.PowerCone(omk)
                     (epsilon + omega - x) .- t <= 0
                 end)
    @objective(model, Min, risk)
    return if optimise_JuMP_model!(model, slv).success
        objective_value(model)
    else
        model = JuMP.Model()
        set_string_names_on_creation(model, false)
        @variables(model, begin
                       z[1:T]
                       nu[1:T]
                       tau[1:T]
                   end)
        if isnothing(w)
            @constraints(model, begin
                             sum(z) - 1 == 0
                             sum(nu - tau) * invk2 - ln_k <= 0
                         end)
            @expression(model, risk, -dot(z, x))
        else
            @constraints(model, begin
                             dot(w, z) - 1 == 0
                             dot(w, nu - tau) * invk2 - ln_k <= 0
                         end)
            @expression(model, risk, -dot(z, w .* x))
        end
        @constraints(model, begin
                         [i = 1:T], [nu[i], 1, z[i]] in MOI.PowerCone(invopk)
                         [i = 1:T], [z[i], 1, tau[i]] in MOI.PowerCone(omk)
                     end)
        @objective(model, Max, risk)
        if optimise_JuMP_model!(model, slv).success
            objective_value(model)
        else
            NaN
        end
    end
end
struct RelativisticValueatRisk{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
    w::T5
    function RelativisticValueatRisk(settings::RiskMeasureSettings,
                                     slv::Option{<:Slv_VecSlv}, alpha::Number,
                                     kappa::Number, w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa), typeof(w)}(settings,
                                                                                           slv,
                                                                                           alpha,
                                                                                           kappa,
                                                                                           w)
    end
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Option{<:Slv_VecSlv} = nothing, alpha::Number = 0.05,
                                 kappa::Number = 0.3,
                                 w::Option{<:AbstractWeights} = nothing)
    return RelativisticValueatRisk(settings, slv, alpha, kappa, w)
end
function factory(r::RelativisticValueatRisk, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)
    w = nothing_scalar_array_selector(r.w, pr.w)
    slv = solver_selector(r.slv, slv)
    return RelativisticValueatRisk(; settings = r.settings, alpha = r.alpha,
                                   kappa = r.kappa, slv = slv, w = w)
end
function (r::RelativisticValueatRisk)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa, r.w)
end
struct RelativisticValueatRiskRange{T1, T2, T3, T4, T5, T6, T7} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa_a::T4
    beta::T5
    kappa_b::T6
    w::T7
    function RelativisticValueatRiskRange(settings::RiskMeasureSettings,
                                          slv::Option{<:Slv_VecSlv}, alpha::Number,
                                          kappa_a::Number, beta::Number, kappa_b::Number,
                                          w::Option{<:AbstractWeights})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa_a) < kappa_a < one(kappa_a))
        @argcheck(zero(beta) < beta < one(beta))
        @argcheck(zero(kappa_b) < kappa_b < one(kappa_b))
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa_a),
                   typeof(beta), typeof(kappa_b), typeof(w)}(settings, slv, alpha, kappa_a,
                                                             beta, kappa_b, w)
    end
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      slv::Option{<:Slv_VecSlv} = nothing,
                                      alpha::Number = 0.05, kappa_a::Number = 0.3,
                                      beta::Number = 0.05, kappa_b::Number = 0.3,
                                      w::Option{<:AbstractWeights} = nothing)
    return RelativisticValueatRiskRange(settings, slv, alpha, kappa_a, beta, kappa_b, w)
end
function (r::RelativisticValueatRiskRange)(x::VecNum)
    return RRM(x, r.slv, r.alpha, r.kappa_a, r.w) + RRM(-x, r.slv, r.beta, r.kappa_b, r.w)
end
function factory(r::RelativisticValueatRiskRange, pr::AbstractPriorResult,
                 slv::Option{<:Slv_VecSlv}, args...; kwargs...)
    slv = solver_selector(r.slv, slv)
    return RelativisticValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                        kappa_a = r.kappa_a, beta = r.beta,
                                        kappa_b = r.kappa_b, slv = slv)
end
struct RelativisticDrawdownatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
    function RelativisticDrawdownatRisk(settings, slv::Option{<:Slv_VecSlv}, alpha::Number,
                                        kappa::Number)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa)}(settings,
                                                                                slv, alpha,
                                                                                kappa)
    end
end
function RelativisticDrawdownatRisk(; settings = RiskMeasureSettings(),
                                    slv::Option{<:Slv_VecSlv} = nothing,
                                    alpha::Number = 0.05, kappa::Number = 0.3)
    return RelativisticDrawdownatRisk(settings, slv, alpha, kappa)
end
function (r::RelativisticDrawdownatRisk)(x::VecNum)
    dd = _absolute_drawdown(x)
    return RRM(dd, r.slv, r.alpha, r.kappa)
end
struct RelativeRelativisticDrawdownatRisk{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
    function RelativeRelativisticDrawdownatRisk(settings::HierarchicalRiskMeasureSettings,
                                                slv::Option{<:Slv_VecSlv}, alpha::Number,
                                                kappa::Number)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(kappa) < kappa < one(kappa))
        return new{typeof(settings), typeof(slv), typeof(alpha), typeof(kappa)}(settings,
                                                                                slv, alpha,
                                                                                kappa)
    end
end
function RelativeRelativisticDrawdownatRisk(;
                                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                            slv::Option{<:Slv_VecSlv} = nothing,
                                            alpha::Number = 0.05, kappa::Number = 0.3)
    return RelativeRelativisticDrawdownatRisk(settings, slv, alpha, kappa)
end
function (r::RelativeRelativisticDrawdownatRisk)(x::VecNum)
    dd = _relative_drawdown(x)
    return RRM(dd, r.slv, r.alpha, r.kappa)
end
for r in (RelativisticDrawdownatRisk, RelativeRelativisticDrawdownatRisk)
    eval(quote
             function factory(r::$(r), ::Any, slv::Option{<:Slv_VecSlv}, args...;
                              kwargs...)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             slv = slv)
             end
             function factory(r::$(r), slv::Slv_VecSlv; kwargs...)
                 slv = solver_selector(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             slv = slv)
             end
         end)
end

export RelativisticValueatRisk, RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
       RelativeRelativisticDrawdownatRisk
