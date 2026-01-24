abstract type RelaxedRiskBudgetingAlgorithm <: OptimisationAlgorithm end
struct BasicRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
struct RegularisedRelaxedRiskBudgeting <: RelaxedRiskBudgetingAlgorithm end
struct RegularisedPenalisedRelaxedRiskBudgeting{T1} <: RelaxedRiskBudgetingAlgorithm
    p::T1
    function RegularisedPenalisedRelaxedRiskBudgeting(p::Number)
        @argcheck(isfinite(p) && p > zero(p))
        return new{typeof(p)}(p)
    end
end
function RegularisedPenalisedRelaxedRiskBudgeting(; p::Number = 1.0)
    return RegularisedPenalisedRelaxedRiskBudgeting(p)
end
struct RelaxedRiskBudgeting{T1, T2, T3, T4, T5} <: JuMPOptimisationEstimator
    opt::T1
    rba::T2
    wi::T3
    alg::T4
    fb::T5
    function RelaxedRiskBudgeting(opt::JuMPOptimiser, rba::RiskBudgetingAlgorithm,
                                  wi::Option{<:VecNum}, alg::RelaxedRiskBudgetingAlgorithm,
                                  fb::Option{<:NonFiniteAllocationOptimisationEstimator})
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        if isa(rba.rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(opt.sets))
        end
        return new{typeof(opt), typeof(rba), typeof(wi), typeof(alg), typeof(fb)}(opt, rba,
                                                                                  wi, alg,
                                                                                  fb)
    end
end
function RelaxedRiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(),
                              rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                              wi::Option{<:VecNum} = nothing,
                              alg::RelaxedRiskBudgetingAlgorithm = BasicRelaxedRiskBudgeting(),
                              fb::Option{<:NonFiniteAllocationOptimisationEstimator} = nothing)
    return RelaxedRiskBudgeting(opt, rba, wi, alg, fb)
end
function opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum)
    X = isa(rrb.opt.pr, AbstractPriorResult) ? rrb.opt.pr.X : X
    opt = opt_view(rrb.opt, i, X)
    rba = risk_budgeting_algorithm_view(rrb.rba, i)
    wi = nothing_scalar_array_view(rrb.wi, i)
    return RelaxedRiskBudgeting(; opt = opt, rba = rba, wi = wi, alg = rrb.alg, fb = rrb.fb)
end
function set_relaxed_risk_budgeting_alg_constraints!(::BasicRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = model[:sc]
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    JuMP.@constraint(model, cbasic_rrp, [sc * psi; sc * G * w] in JuMP.SecondOrderCone())
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(::RegularisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = model[:sc]
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_rrp_soc_1,
                          [sc * 2 * psi;
                           sc * 2 * G * w;
                           sc * -2 * rho] in JuMP.SecondOrderCone()
                          creg_rrp_soc_2, [sc * rho; sc * G * w] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
function set_relaxed_risk_budgeting_alg_constraints!(alg::RegularisedPenalisedRelaxedRiskBudgeting,
                                                     model::JuMP.Model, w::VecJuMPScalar,
                                                     sigma::MatNum,
                                                     chol::Option{<:MatNum} = nothing)
    sc = model[:sc]
    psi = model[:psi]
    G = isnothing(chol) ? LinearAlgebra.cholesky(sigma).U : chol
    theta = LinearAlgebra.Diagonal(sqrt.(LinearAlgebra.diag(sigma)))
    p = alg.p
    JuMP.@variable(model, rho >= 0)
    JuMP.@constraints(model,
                      begin
                          creg_pen_rrp_soc_1,
                          [sc * 2 * psi;
                           sc * 2 * G * w;
                           sc * -2 * rho] in JuMP.SecondOrderCone()
                          creg_pen_rrp_soc_2,
                          [sc * rho;
                           sc * sqrt(p) * theta * w] in JuMP.SecondOrderCone()
                      end)
    return nothing
end
function _set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                  rrb::RelaxedRiskBudgeting,
                                                  w::VecJuMPScalar, sigma::MatNum,
                                                  chol::Option{<:MatNum} = nothing)
    N = length(w)
    rkb = risk_budget_constraints(rrb.rba.rkb, rrb.opt.sets; N = N, strict = rrb.opt.strict)
    rb = rkb.val
    sc = model[:sc]
    JuMP.@variables(model, begin
                        psi >= 0
                        gamma >= 0
                        zeta[1:N] >= 0
                    end)
    JuMP.@expression(model, risk, psi - gamma)
    # RRB constraints.
    JuMP.@constraints(model,
                      begin
                          crrp, sc * (zeta - sigma * w) == 0
                          crrp_soc[i = 1:N],
                          [sc * (w[i] + zeta[i])
                           sc * (2 * gamma * sqrt(rb[i]))
                           sc * (w[i] - zeta[i])] in JuMP.SecondOrderCone()
                      end)
    set_relaxed_risk_budgeting_alg_constraints!(rrb.alg, model, w, sigma, chol)
    return rkb
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:FactorRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rrb.rba.re, rd, rrb.rba.flag,
                                                       rrb.wi)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, model[:w1],
                                                   Matrix(LinearAlgebra.Symmetric(rr.L \
                                                                                  pr.sigma *
                                                                                  b1)))
    set_weight_constraints!(model, wb, rrb.opt.bgt, rrb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
function set_relaxed_risk_budgeting_constraints!(model::JuMP.Model,
                                                 rrb::RelaxedRiskBudgeting{<:Any,
                                                                           <:AssetRiskBudgeting,
                                                                           <:Any, <:Any},
                                                 pr::AbstractPriorResult, wb::WeightBounds,
                                                 args...)
    set_w!(model, pr.X, rrb.wi)
    set_weight_constraints!(model, wb, rrb.opt.bgt, nothing, true)
    rkb = _set_relaxed_risk_budgeting_constraints!(model, rrb, model[:w], pr.sigma, pr.chol)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function _optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, ct, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, pl, tn, fees, ret) = processed_jump_optimiser_attributes(rrb.opt,
                                                                                                                                           rd;
                                                                                                                                           dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rrb.opt.sc, rrb.opt.so)
    JuMP.@expression(model, k, 1)
    prb = set_relaxed_risk_budgeting_constraints!(model, rrb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ct, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, rrb.opt.card, gcard, pl, lt, st, fees, rrb.opt.ss)
    set_smip_constraints!(model, wb, rrb.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, rrb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rrb.opt.tr, nothing, nothing, nothing, fees;
                                    rd = rd)
    set_number_effective_assets!(model, rrb.opt.nea)
    set_l1_regularisation!(model, rrb.opt.l1)
    set_l2_regularisation!(model, rrb.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, pl)
    add_custom_constraint!(model, rrb.opt.ccnt, rrb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rrb.opt.cobj, rrb, pr)
    retcode, sol = optimise_JuMP_model!(model, rrb, eltype(pr.X))
    return RiskBudgetingResult(typeof(rrb),
                               ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, ct,
                                                                gcard, sgcard, smtx, sgmtx,
                                                                slt, sst, sglt, sgst, tn,
                                                                fees, pl, ret), prb,
                               retcode, sol, ifelse(save, model, nothing), nothing)
end
function optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rrb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting,
       RegularisedPenalisedRelaxedRiskBudgeting, RelaxedRiskBudgeting
