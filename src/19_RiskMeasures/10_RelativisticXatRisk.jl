function RRM(x::AbstractVector, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
             alpha::Real = 0.05, kappa::Real = 0.3)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    T = length(x)
    at = alpha * T
    invat = inv(at)
    invk2 = inv(2 * kappa)
    ln_k = (invat^kappa - invat^(-kappa)) * invk2
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    invk = inv(kappa)
    invopk = inv(opk)
    invomk = inv(omk)
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
    @constraints(model,
                 begin
                     [i = 1:T],
                     [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] ∈
                     MOI.PowerCone(invopk)
                     [i = 1:T],
                     [omega[i] * invomk, theta[i] * invk, -z * invk2] ∈ MOI.PowerCone(omk)
                     (epsilon + omega - x) .- t <= 0
                 end)
    @expression(model, risk, t + ln_k * z + sum(psi + theta))
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
        @constraints(model, begin
                         sum(z) - 1 == 0
                         sum(nu - tau) * invk2 - ln_k <= 0
                         [i = 1:T], [nu[i], 1, z[i]] ∈ MOI.PowerCone(invopk)
                         [i = 1:T], [z[i], 1, tau[i]] ∈ MOI.PowerCone(omk)
                     end)
        @expression(model, risk, -dot(z, x))
        @objective(model, Max, risk)
        if optimise_JuMP_model!(model, slv).success
            objective_value(model)
        else
            NaN
        end
    end
end
function RRM(x::AbstractVector, slv::Union{<:Solver, <:AbstractVector{<:Solver}},
             w::AbstractWeights, alpha::Real = 0.05, kappa::Real = 0.3)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    T = sum(w)
    at = alpha * T
    invat = inv(at)
    invk2 = inv(2 * kappa)
    ln_k = (invat^kappa - invat^(-kappa)) * invk2
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    invk = inv(kappa)
    invopk = inv(opk)
    invomk = inv(omk)
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
    @constraints(model,
                 begin
                     [i = 1:T],
                     [z * opk * invk2, psi[i] * opk * invk, epsilon[i]] ∈
                     MOI.PowerCone(invopk)
                     [i = 1:T],
                     [omega[i] * invomk, theta[i] * invk, -z * invk2] ∈ MOI.PowerCone(omk)
                     (epsilon + omega - x) .- t <= 0
                 end)
    @expression(model, risk, t + ln_k * z + dot(w, psi + theta))
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
        @constraints(model, begin
                         dot(w, z) - 1 == 0
                         dot(w, nu - tau) * invk2 - ln_k <= 0
                         [i = 1:T], [nu[i], 1, z[i]] ∈ MOI.PowerCone(invopk)
                         [i = 1:T], [z[i], 1, tau[i]] ∈ MOI.PowerCone(omk)
                     end)
        @expression(model, risk, -dot(z, x))
        @objective(model, Max, risk)
        if optimise_JuMP_model!(model, slv).success
            objective_value(model)
        else
            NaN
        end
    end
end
struct RelativisticValueatRisk{T1 <: RiskMeasureSettings,
                               T2 <: Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                               T3 <: Real, T4 <: Real,
                               T5 <: Union{Nothing, <:AbstractWeights}} <: SolverRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
    w::T5
end
function RelativisticValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}} = nothing,
                                 alpha::Real = 0.05, kappa::Real = 0.3,
                                 w::Union{Nothing, AbstractWeights} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return RelativisticValueatRisk{typeof(settings), typeof(slv), typeof(alpha),
                                   typeof(kappa), typeof(w)}(settings, slv, alpha, kappa, w)
end
function risk_measure_factory(r::RelativisticValueatRisk, prior::AbstractPriorResult,
                              slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...; kwargs...)
    w = risk_measure_nothing_scalar_array_factory(r.w, prior.w)
    slv = risk_measure_solver_factory(r.slv, slv)
    return RelativisticValueatRisk(; settings = r.settings, alpha = r.alpha,
                                   kappa = r.kappa, slv = slv, w = w)
end
function (r::RelativisticValueatRisk{<:Any, <:Any, <:Any, <:Any, Nothing})(x::AbstractVector)
    return RRM(x, r.slv, r.alpha, r.kappa)
end
function (r::RelativisticValueatRisk{<:Any, <:Any, <:Any, <:Any, <:AbstractWeights})(x::AbstractVector)
    return RRM(x, r.slv, r.w, r.alpha, r.kappa)
end
struct RelativisticValueatRiskRange{T1 <: RiskMeasureSettings,
                                    T2 <:
                                    Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                                    T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real,
                                    T7 <: Union{Nothing, <:AbstractWeights}} <:
       SolverRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa_a::T4
    beta::T5
    kappa_b::T6
    w::T7
end
function RelativisticValueatRiskRange(;
                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}} = nothing,
                                      alpha::Real = 0.05, kappa_a::Real = 0.3,
                                      beta::Real = 0.05, kappa_b::Real = 0.3,
                                      w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa_a) < kappa_a < one(kappa_a))
    @smart_assert(zero(beta) < beta < one(beta))
    @smart_assert(zero(kappa_b) < kappa_b < one(kappa_b))
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return RelativisticValueatRiskRange{typeof(settings), typeof(slv), typeof(alpha),
                                        typeof(kappa_a), typeof(beta), typeof(kappa_b),
                                        typeof(w)}(settings, slv, alpha, kappa_a, beta,
                                                   kappa_b, w)
end
function (r::RelativisticValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          Nothing})(x::AbstractVector)
    return RRM(x, r.slv, r.alpha, r.kappa_a) + RRM(-x, r.slv, r.beta, r.kappa_b)
end
function (r::RelativisticValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                          <:AbstractWeights})(x::AbstractVector)
    return RRM(x, r.slv, r.w, r.alpha, r.kappa_a) + RRM(-x, r.slv, r.w, r.beta, r.kappa_b)
end
function risk_measure_factory(r::RelativisticValueatRiskRange, prior::AbstractPriorResult,
                              slv::Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                              args...; kwargs...)
    slv = risk_measure_solver_factory(r.slv, slv)
    return RelativisticValueatRiskRange(; settings = r.settings, alpha = r.alpha,
                                        kappa_a = r.kappa_a, beta = r.beta,
                                        kappa_b = r.kappa_b, slv = slv)
end
struct RelativisticDrawdownatRisk{T1 <: RiskMeasureSettings,
                                  T2 <:
                                  Union{Nothing, <:Solver, <:AbstractVector{<:Solver}},
                                  T3 <: Real, T4 <: Real} <: SolverRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
end
function RelativisticDrawdownatRisk(; settings = RiskMeasureSettings(),
                                    slv::Union{Nothing, <:Solver,
                                               <:AbstractVector{<:Solver}} = nothing,
                                    alpha::Real = 0.05, kappa::Real = 0.3)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativisticDrawdownatRisk{typeof(settings), typeof(slv), typeof(alpha),
                                      typeof(kappa)}(settings, slv, alpha, kappa)
end
function (r::RelativisticDrawdownatRisk)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    return RRM(dd, r.slv, r.alpha, r.kappa)
end
struct RelativeRelativisticDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings,
                                          T2 <: Union{Nothing, <:Solver,
                                                      <:AbstractVector{<:Solver}},
                                          T3 <: Real, T4 <: Real} <:
       SolverHierarchicalRiskMeasure
    settings::T1
    slv::T2
    alpha::T3
    kappa::T4
end
function RelativeRelativisticDrawdownatRisk(;
                                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                            slv::Union{Nothing, <:Solver,
                                                       <:AbstractVector{<:Solver}} = nothing,
                                            alpha::Real = 0.05, kappa = 0.3)
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(kappa) < kappa < one(kappa))
    return RelativeRelativisticDrawdownatRisk{typeof(settings), typeof(slv), typeof(alpha),
                                              typeof(kappa)}(settings, slv, alpha, kappa)
end
function (r::RelativeRelativisticDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    return RRM(dd, r.slv, r.alpha, r.kappa)
end
for r ∈ (RelativisticDrawdownatRisk, RelativeRelativisticDrawdownatRisk)
    eval(quote
             function risk_measure_factory(r::$(r), ::Any,
                                           slv::Union{Nothing, <:Solver,
                                                      <:AbstractVector{<:Solver}}, args...;
                                           kwargs...)
                 slv = risk_measure_solver_factory(r.slv, slv)
                 return $(r)(; settings = r.settings, alpha = r.alpha, kappa = r.kappa,
                             slv = slv)
             end
         end)
end

export RelativisticValueatRisk, RelativisticValueatRiskRange, RelativisticDrawdownatRisk,
       RelativeRelativisticDrawdownatRisk
