@concrete struct RiskBudgetingResult <: NonFiniteAllocationOptimisationResult
    oe
    pa
    prb
    retcode
    sol
    model
    fb
end
function factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt})
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
@concrete struct ProcessedFactorRiskBudgetingAttributes <: AbstractResult
    rkb
    b1
    rr
end
@concrete struct ProcessedAssetRiskBudgetingAttributes <: AbstractResult
    rkb
end
abstract type RiskBudgetingFormulation <: OptimisationAlgorithm end
struct LogRiskBudgeting <: RiskBudgetingFormulation end
struct MixedIntegerRiskBudgeting <: RiskBudgetingFormulation end
abstract type RiskBudgetingAlgorithm <: OptimisationAlgorithm end
@concrete struct AssetRiskBudgeting <: RiskBudgetingAlgorithm
    rkb
    sets
    alg
    function AssetRiskBudgeting(rkb::Option{<:RkbE_Rkb}, sets::Option{<:AssetSets},
                                alg::RiskBudgetingFormulation)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(rkb), typeof(sets), typeof(alg)}(rkb, sets, alg)
    end
end
function AssetRiskBudgeting(; rkb::Option{<:RkbE_Rkb} = nothing,
                            sets::Option{<:AssetSets} = nothing,
                            alg::RiskBudgetingFormulation = LogRiskBudgeting())
    return AssetRiskBudgeting(rkb, sets, alg)
end
function risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i)
    rkb = risk_budget_view(r.rkb, i)
    sets = asset_sets_view(r.sets, i)
    return AssetRiskBudgeting(; rkb = rkb, sets = sets, alg = r.alg)
end
@concrete struct FactorRiskBudgeting <: RiskBudgetingAlgorithm
    re
    rkb
    sets
    flag
    function FactorRiskBudgeting(re::RegE_Reg, rkb::Option{<:RkbE_Rkb},
                                 sets::Option{<:AssetSets}, flag::Bool)
        if isa(rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(sets))
        end
        return new{typeof(re), typeof(rkb), typeof(sets), typeof(flag)}(re, rkb, sets, flag)
    end
end
function FactorRiskBudgeting(; re::RegE_Reg = StepwiseRegression(),
                             rkb::Option{<:RkbE_Rkb} = nothing,
                             sets::Option{<:AssetSets} = nothing, flag::Bool = true)
    return FactorRiskBudgeting(re, rkb, sets, flag)
end
function risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)
    re = regression_view(r.re, i)
    return FactorRiskBudgeting(; re = re, rkb = r.rkb, sets = r.sets, flag = r.flag)
end
@concrete struct RiskBudgeting <: RiskJuMPOptimisationEstimator
    opt
    r
    rba
    wi
    fb
    function RiskBudgeting(opt::JuMPOptimiser, r::RM_VecRM, rba::RiskBudgetingAlgorithm,
                           wi::Option{<:VecNum}, fb::Option{<:OptE_Opt})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        return new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fb)}(opt, r, rba,
                                                                                wi, fb)
    end
end
function RiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(), r::RM_VecRM = Variance(),
                       rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                       wi::Option{<:VecNum} = nothing, fb::Option{<:OptE_Opt} = nothing)
    return RiskBudgeting(opt, r, rba, wi, fb)
end
function needs_previous_weights(opt::RiskBudgeting)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
function factory(rb::RiskBudgeting, w::AbstractVector)
    opt = factory(rb.opt, w)
    r = factory(rb.r, w)
    fb = factory(rb.fb, w)
    return RiskBudgeting(; opt = opt, r = r, rba = rb.rba, wi = rb.wi, fb = fb)
end
function opt_view(rb::RiskBudgeting, i, X::MatNum)
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    rba = risk_budgeting_algorithm_view(rb.rba, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgeting(; opt = opt, r = r, rba = rba, wi = wi, fb = rb.fb)
end
function _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                          w::VecJuMPScalar; strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.rba.sets; N = N, strict = strict)
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
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:LogRiskBudgeting},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
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
###########
function set_rb_mip_w!(model::JuMP.Model, X::MatNum)
    N = size(X, 2)
    JuMP.@variables(model, begin
                        lw[1:N] >= 0
                        sw[1:N] >= 0
                    end)
    JuMP.@expressions(model, begin
                          w, lw - sw
                          w_obj, lw + sw
                      end)
    return nothing
end
function _set_mip_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                              w::VecJuMPScalar, w_obj::VecJuMPScalar;
                                              strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.rba.sets; N = N, strict = strict)
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
                          [sc * log_w[i], sc, sc * w_obj[i]] in JuMP.MOI.ExponentialCone()
                          crkb, sc * LinearAlgebra.dot(rb, log_w) >= 0
                          mipcrkb, sc * (sum(w) - k) >= 0
                      end)
    return rkb
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting{<:Any,
                                                                                <:Any,
                                                                                <:MixedIntegerRiskBudgeting},
                                                           <:Any}, pr::AbstractPriorResult,
                                         wb::WeightBounds, args...)
    set_rb_mip_w!(model, pr.X)
    rkb = _set_mip_risk_budgeting_constraints!(model, rb, model[:w], model[:w_obj];
                                               strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
###########
function _optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, slt, sst, sgmtx, sglt, sgst, plr, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                                rd;
                                                                                                                                                dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgeting_constraints!(model, rb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, rb.opt.card, gcardr, plr, lt, st, fees, rb.opt.ss,
                         isa(rb.rba,
                             AssetRiskBudgeting{<:Any, <:Any, <:MixedIntegerRiskBudgeting}))
    set_smip_constraints!(model, wb, rb.opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt,
                          sgst, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.tr, rb, plr, fees; rd = rd)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_linf_regularisation!(model, rb.opt.linf)
    set_lp_regularisation!(model, rb.opt.lp)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, plr, fees; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sca)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, plr)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return RiskBudgetingResult(typeof(rb),
                               ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr,
                                                                gcardr, sgcardr, smtx,
                                                                sgmtx, slt, sst, sglt, sgst,
                                                                tn, fees, plr, ret), prb,
                               retcode, sol, ifelse(save, model, nothing), nothing)
end
function optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(rb, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting, RiskBudgetingResult,
       LogRiskBudgeting, MixedIntegerRiskBudgeting
