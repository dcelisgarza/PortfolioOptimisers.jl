struct RiskBudgetingResult{T1, T2, T3, T4, T5, T6, T7} <: OptimisationResult
    oe::T1
    pa::T2
    prb::T3
    retcode::T4
    sol::T5
    model::T6
    fb::T7
end
function factory(res::RiskBudgetingResult, fb)
    return RiskBudgetingResult(res.oe, res.pa, res.prb, res.retcode, res.sol, res.model, fb)
end
function Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
    return if sym == :w
        r.sol.w
    elseif sym in propertynames(r)
        getfield(r, sym)
    elseif sym in propertynames(r.prb)
        getproperty(r.prb, sym)
    elseif sym in propertynames(r.pa)
        getproperty(r.pa, sym)
    else
        getfield(r, sym)
    end
end
struct ProcessedFactorRiskBudgetingAttributes{T1, T2, T3} <: AbstractResult
    rkb::T1
    b1::T2
    rr::T3
end
struct ProcessedAssetRiskBudgetingAttributes{T1} <: AbstractResult
    rkb::T1
end
abstract type RiskBudgetingAlgorithm <: OptimisationAlgorithm end
struct AssetRiskBudgeting{T1} <: RiskBudgetingAlgorithm
    rkb::T1
    function AssetRiskBudgeting(rkb::Option{<:RkbE_Rkb})
        return new{typeof(rkb)}(rkb)
    end
end
function AssetRiskBudgeting(; rkb::Option{<:RkbE_Rkb} = nothing)
    return AssetRiskBudgeting(rkb)
end
function risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i)
    return AssetRiskBudgeting(; rkb = risk_budget_view(r.rkb, i))
end
struct FactorRiskBudgeting{T1, T2, T3} <: RiskBudgetingAlgorithm
    re::T1
    rkb::T2
    flag::T3
    function FactorRiskBudgeting(re::RegE_Reg, rkb::Option{<:RkbE_Rkb}, flag::Bool)
        return new{typeof(re), typeof(rkb), typeof(flag)}(re, rkb, flag)
    end
end
function FactorRiskBudgeting(; re::RegE_Reg = StepwiseRegression(),
                             rkb::Option{<:RkbE_Rkb} = nothing, flag::Bool = true)
    return FactorRiskBudgeting(re, rkb, flag)
end
function risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)
    re = regression_view(r.re, i)
    return FactorRiskBudgeting(; re = re, rkb = r.rkb, flag = r.flag)
end
struct RiskBudgeting{T1, T2, T3, T4, T5} <: RiskJuMPOptimisationEstimator
    opt::T1
    r::T2
    rba::T3
    wi::T4
    fb::T5
    function RiskBudgeting(opt::JuMPOptimiser, r::RM_VecRM, rba::RiskBudgetingAlgorithm,
                           wi::Option{<:VecNum}, fb::Option{<:OptimisationEstimator})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        if isa(rba.rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(opt.sets))
        end
        return new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fb)}(opt, r, rba,
                                                                                wi, fb)
    end
end
function RiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(), r::RM_VecRM = Variance(),
                       rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                       wi::Option{<:VecNum} = nothing,
                       fb::Option{<:OptimisationEstimator} = nothing)
    return RiskBudgeting(opt, r, rba, wi, fb)
end
function opt_view(rb::RiskBudgeting, i, X::MatNum)
    X = isa(rb.opt.pr, AbstractPriorResult) ? rb.opt.pr.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    rba = risk_budgeting_algorithm_view(rb.rba, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgeting(; opt = opt, r = r, rba = rba, wi = wi, fb = rb.fb)
end
function _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                          w::VecJuMPScalar; strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.opt.sets; N = N, strict = strict)
    rb = rkb.val
    @argcheck(length(rb) == N)
    sc = model[:sc]
    JuMP.@variables(model, begin
                        k
                        log_w[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          clog_w[i = 1:N],
                          [sc * log_w[i], sc, sc * w[i]] in JuMP.MOI.ExponentialCone()
                          crkb, sc * LinearAlgebra.dot(rb, log_w) >= 0
                      end)
    return rkb
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting, <:Any},
                                         pr::AbstractPriorResult, wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:FactorRiskBudgeting, <:Any},
                                         ::Any, wb::WeightBounds, rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rb.rba.re, rd, rb.rba.flag,
                                                       rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w1]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
function _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, ct, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, pl, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                           rd;
                                                                                                                                           dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgeting_constraints!(model, rb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ct, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, rb.opt.card, gcard, pl, lt, st, fees, rb.opt.ss)
    set_smip_constraints!(model, wb, rb.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.tr, rb, pl, fees; rd = rd)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, pl, fees; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sca)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, pl)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return RiskBudgetingResult(typeof(rb),
                               ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, ct,
                                                                gcard, sgcard, smtx, sgmtx,
                                                                slt, sst, sglt, sgst, tn,
                                                                fees, pl, ret), prb,
                               retcode, sol, ifelse(save, model, nothing), nothing)
end
function optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting, RiskBudgetingResult
