abstract type RelaxedRiskBudgettingAlgorithm <: OptimisationAlgorithm end
struct BasicRelaxedRiskBudgettingAlgorithm <: RelaxedRiskBudgettingAlgorithm end
struct RegularisationRelaxedRiskBudgettingAlgorithm <: RelaxedRiskBudgettingAlgorithm end
struct RegularisationPenaltyRelaxedRiskBudgettingAlgorithm{T1 <: Real} <:
       RelaxedRiskBudgettingAlgorithm
    p::T1
end
function RegularisationPenaltyRelaxedRiskBudgettingAlgorithm(; p::Real = 1.0)
    @smart_assert(isfinite(p))
    return RegularisationPenaltyRelaxedRiskBudgettingAlgorithm{typeof(p)}(p)
end
struct RelaxedRiskBudgetting{T1 <: JuMPOptimiser,
                             T2 <: Union{Nothing, <:AbstractVector{<:Real}},
                             T3 <: Union{Nothing, <:AbstractVector{<:Real}},
                             T4 <: RelaxedRiskBudgettingAlgorithm} <:
       JuMPOptimisationEstimator
    opt::T1
    rkb::T2
    wi::T3
    alg::T4
end
function RelaxedRiskBudgetting(; opt::JuMPOptimiser = JuMPOptimiser(),
                               rkb::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                               wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                               alg::RelaxedRiskBudgettingAlgorithm = BasicRelaxedRiskBudgettingAlgorithm())
    if isa(rkb, AbstractVector)
        @smart_assert(!isempty(rkb))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return RelaxedRiskBudgetting{typeof(opt), typeof(rkb), typeof(wi), typeof(alg)}(opt,
                                                                                    rkb, wi,
                                                                                    alg)
end
function opt_view(rrb::RelaxedRiskBudgetting, i::AbstractVector, X::AbstractMatrix)
    opt = opt_view(rrb.opt, i, X)
    rkb = nothing_scalar_array_view(rrb.rkb, i)
    wi = nothing_scalar_array_view(rrb.wi, i)
    return RelaxedRiskBudgetting(; opt = opt, rkb = rkb, wi = wi, alg = rrb.alg)
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
                                                  rrb::RelaxedRiskBudgetting,
                                                  sigma::AbstractMatrix)
    w = model[:w]
    N = length(w)
    rkb = rrb.rkb
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
    set_relaxed_risk_budgetting_alg_constraints!(rrb.alg, model, sigma)
    return nothing
end
function optimise!(rrb::RelaxedRiskBudgetting, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    pr, wb, lcs, cent, gcard, sgcard, smtx, nplg, cplg = processed_jump_optimiser_attributes(rrb.opt,
                                                                                             rd;
                                                                                             dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rrb.opt.sc, rrb.opt.so)
    @expression(model, k, 1)
    set_w!(model, pr.X, rrb.wi)
    set_weight_constraints!(model, wb, rrb.opt.bgt, nothing, false)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, rrb.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, rrb.opt.card, gcard, nplg, cplg, rrb.opt.lt, rrb.opt.st,
                         rrb.opt.fees, rrb.opt.ss)
    set_smip_constraints!(model, wb, rrb.opt.scard, sgcard, smtx, rrb.opt.ss)
    set_turnover_constraints!(model, rrb.opt.tn)
    set_tracking_error_constraints!(model, pr, rrb.opt.te)
    set_number_effective_assets!(model, rrb.opt.nea)
    set_l1_regularisation!(model, rrb.opt.l1)
    set_l2_regularisation!(model, rrb.opt.l2)
    set_non_fixed_fees!(model, rrb.opt.fees)
    set_relaxed_risk_budgetting_constraints!(model, rrb, pr.sigma)
    ret = jump_returns_factory(rrb.opt.ret, pr)
    set_return_constraints!(model, ret, MinimumRisk(), pr)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, rrb.opt.ccnt, rrb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rrb.opt.cobj, rrb, pr)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(pr.X))
    return JuMPOptimisationResult(typeof(rrb), pr, wb, lcs, cent, gcard, sgcard, smtx, nplg,
                                  cplg, retcode, sol, ifelse(save, model, nothing))
end

export BasicRelaxedRiskBudgettingAlgorithm, RegularisationRelaxedRiskBudgettingAlgorithm,
       RegularisationPenaltyRelaxedRiskBudgettingAlgorithm, RelaxedRiskBudgetting
