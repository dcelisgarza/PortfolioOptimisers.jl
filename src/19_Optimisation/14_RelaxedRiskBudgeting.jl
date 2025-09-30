abstract type RelaxedRiskBudgetingAlgorithm <: OptimisationAlgorithm end
struct BasicRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
struct RegularisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
struct RegularisedPenalisedRelaxedRiskBudgeting{T1} <: RelaxedRiskBudgetingAlgorithm
    p::T1
end
function RegularisedPenalisedRelaxedRiskBudgeting(; p::Real = 1.0)
    @argcheck(isfinite(p) && p > zero(p))
    return RegularisedPenalisedRelaxedRiskBudgeting(p)
end
struct RelaxedRiskBudgeting{T1, T2, T3, T4, T5} <: JuMPOptimisationEstimator
    opt::T1
    rba::T2
    wi::T3
    alg::T4
    fallback::T5
end
function RelaxedRiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(),
                              rba::Union{Nothing, <:RiskBudgetEstimator,
                                         <:RiskBudgetResult} = nothing,
                              wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
                              fallback::Union{Nothing, <:OptimisationEstimator} = nothing)
    if isa(wi, AbstractVector)
        @argcheck(!isempty(wi))
    end
    if isa(rba, RiskBudgetEstimator)
        @argcheck(!isnothing(opt.sets))
    end
    return RelaxedRiskBudgeting(opt, rba, wi, alg, fallback)
end
function opt_view(rrb::RelaxedRiskBudgeting, i::AbstractVector, X::AbstractMatrix)
    X = isa(rrb.opt.pe, AbstractPriorResult) ? rrb.opt.pe.X : X
    opt = opt_view(rrb.opt, i, X)
    rba = risk_budget_view(rrb.rba, i)
    wi = nothing_scalar_array_view(rrb.wi, i)
    return RelaxedRiskBudgeting(; opt = opt, rba = rba, wi = wi, alg = rrb.alg,
                                fallback = rrb.fallback)
end
function set_relaxed_risk_budgeting_alg_constraints!(::BasicRelaxedRiskBudgeting,
                                                     model::JuMP.Model,
                                                     sigma::AbstractMatrix)
    sc = model[:sc]
    w = model[:w]
    psi = model[:psi]
    G = cholesky(sigma).U
    @constraint(model, cbasic_rrp, [sc * psi; sc * G * w] in SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(::RegularisedRelaxedRiskBudgeting,
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
                     [sc * 2 * psi;
                      sc * 2 * G * w;
                      sc * -2 * rho] in SecondOrderCone()
                     creg_rrp_soc_2, [sc * rho; sc * G * w] in SecondOrderCone()
                 end)
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(alg::RegularisedPenalisedRelaxedRiskBudgeting,
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
                     [sc * 2 * psi;
                      sc * 2 * G * w;
                      sc * -2 * rho] in SecondOrderCone()
                     creg_pen_rrp_soc_2,
                     [sc * rho;
                      sc * sqrt(p) * theta * w] in SecondOrderCone()
                 end)
    return nothing
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting,
                                                 sigma::AbstractMatrix)
    w = model[:w]
    N = length(w)
    rba = risk_budget_constraints(rrb.rba, rrb.opt.sets; N = N, strict = rrb.opt.strict)
    rb = rba.val
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
                      sc * (2 * gamma * sqrt(rb[i]))
                      sc * (w[i] - zeta[i])] in SecondOrderCone()
                 end)
    set_relaxed_risk_budgeting_alg_constraints!(rrb.alg, model, sigma)
    return rba
end
function optimise!(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(rrb.opt,
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
    set_mip_constraints!(model, wb, rrb.opt.card, gcard, nplg, cplg, lt, st, fees,
                         rrb.opt.ss)
    set_smip_constraints!(model, wb, rrb.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, rrb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rrb.opt.te, nothing, nothing, nothing, fees)
    set_number_effective_assets!(model, rrb.opt.nea)
    set_l1_regularisation!(model, rrb.opt.l1)
    set_l2_regularisation!(model, rrb.opt.l2)
    set_non_fixed_fees!(model, fees)
    prb = set_relaxed_risk_budgeting_constraints!(model, rrb, pr.sigma)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_phylogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, rrb.opt.ccnt, rrb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rrb.opt.cobj, rrb, pr)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(pr.X))
    return if isa(retcode, OptimisationSuccess) || isnothing(rrb.fallback)
        JuMPOptimisationRiskBudgeting(typeof(rrb),
                                      ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs,
                                                                       cent, gcard, sgcard,
                                                                       smtx, sgmtx, slt,
                                                                       sst, sglt, sgst,
                                                                       nplg, cplg, tn, fees,
                                                                       ret), prb, retcode,
                                      sol, ifelse(save, model, nothing))
    else
        @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        optimise!(rrb.fallback, rd; dims = dims, str_names = str_names, save = save,
                  kwargs...)
    end
end

export BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting,
       RegularisedPenalisedRelaxedRiskBudgeting, RelaxedRiskBudgeting
