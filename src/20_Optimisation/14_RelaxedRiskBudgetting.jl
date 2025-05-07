abstract type RelaxedRiskBudgettingAlgorithm <: OptimisationAlgorithm end
struct BasicRelaxedRiskBudgettingAlgorithm <: RelaxedRiskBudgettingAlgorithm end
struct RegularisationRelaxedRiskBudgettingAlgorithm <: RelaxedRiskBudgettingAlgorithm end
struct RegularisationPenaltyRelaxedRiskBudgettingAlgorithm{T1 <: Real} <:
       RelaxedRiskBudgettingAlgorithm
    p::T1
end
function RegularisationPenaltyRelaxedRiskBudgettingAlgorithm(p::Real = 1.0)
    @smart_assert(isfinite(p))
    return RegularisationPenaltyRelaxedRiskBudgettingAlgorithm{typeof(p)}(p)
end
struct RelaxedRiskBudgettingEstimator{T1 <: RelaxedRiskBudgettingAlgorithm,
                                      T2 <: JuMPOptimiser,
                                      T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                                      T4 <: Union{Nothing, <:AbstractVector{<:Real}},
                                      T5 <: Bool, T6 <: Bool} <: JuMPOptimisationEstimator
    alg::T1
    opt::T2
    rkb::T3
    wi::T4
    str_names::T5
    save::T6
end
function RelaxedRiskBudgettingEstimator(;
                                        alg::RelaxedRiskBudgettingAlgorithm = BasicRelaxedRiskBudgettingAlgorithm(),
                                        opt::JuMPOptimiser = JuMPOptimiser(),
                                        rkb::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                        wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                        str_names::Bool = false, save::Bool = true)
    if isa(rkb, AbstractVector)
        @smart_assert(!isempty(rkb))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return RelaxedRiskBudgettingEstimator{typeof(alg), typeof(opt), typeof(rkb), typeof(wi),
                                          typeof(str_names), typeof(save)}(alg, opt, rkb,
                                                                           wi, str_names,
                                                                           save)
end
function set_relaxed_risk_budgetting_alg_constraints!(::BasicRelaxedRiskBudgettingAlgorithm,
                                                      model::JuMP.Model,
                                                      sigma::AbstractMatrix)
    sc = model[:sc]
    w = model[:w]
    psi = model[:psi]
    G = cholesky(sigma).U
    @constraint(model, cbasic_rrp, [sc * psi; sc * G * w] ∈ SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgetting_alg_constraints!(::RegularisationRelaxedRiskBudgettingAlgorithm,
                                                      model::JuMP.Model,
                                                      sigma::AbstractMatrix)
    sc = model[:sc]
    w = model[:w]
    psi = model[:psi]
    G = cholesky(sigma).U
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     creg_rrp_soc_1,
                     [sc * 2 * psi; sc * 2 * G * w;
                      sc * -2 * rho] ∈ SecondOrderCone()
                     creg_rrp_soc_2, [sc * rho; sc * G * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function set_relaxed_risk_budgetting_alg_constraints!(alg::RegularisationPenaltyRelaxedRiskBudgettingAlgorithm,
                                                      model::JuMP.Model,
                                                      sigma::AbstractMatrix)
    sc = model[:sc]
    w = model[:w]
    psi = model[:psi]
    G = cholesky(sigma).U
    theta = Diagonal(sqrt.(diag(sigma)))
    p = alg.p
    @variable(model, rho >= 0)
    @constraints(model,
                 begin
                     creg_pen_rrp_soc_1,
                     [sc * 2 * psi; sc * 2 * G * w;
                      sc * -2 * rho] ∈ SecondOrderCone()
                     creg_pen_rrp_soc_2,
                     [sc * rho;
                      sc * sqrt(p) * theta * w] ∈ SecondOrderCone()
                 end)
    return nothing
end
function set_relaxed_risk_budgetting_constraints!(model::JuMP.Model,
                                                  rb::RelaxedRiskBudgettingEstimator,
                                                  sigma::AbstractMatrix)
    w = model[:w]
    N = length(w)
    rkb = rb.rkb
    if isnothing(rkb)
        rkb = range(; start = inv(N), stop = inv(N), length = N)
    else
        @smart_assert(length(rkb) == N)
    end
    sc = model[:sc]
    @variables(model, begin
                   psi >= 0
                   gamma >= 0
                   zeta[1:N] >= 0
               end)
    @expression(model, risk, psi - gamma)
    # RRB constraints.
    @constraints(model,
                 begin
                     crrp, sc * (zeta - sigma * w) == 0
                     crrp_soc[i = 1:N],
                     [sc * (w[i] + zeta[i])
                      sc * (2 * gamma * sqrt(rkb[i]))
                      sc * (w[i] - zeta[i])] ∈ SecondOrderCone()
                 end)
    set_relaxed_risk_budgetting_alg_constraints!(rkb.alg, model, sigma)
    return nothing
end
