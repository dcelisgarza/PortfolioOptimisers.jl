struct MeanRiskEstimator{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                         T2 <: ObjectiveFunction, T3 <: JuMPOptimiser,
                         T4 <: Union{Nothing, <:AbstractVector{<:Real}}, T5 <: Bool,
                         T6 <: Bool} <: JuMPOptimisationEstimator
    r::T1
    obj::T2
    opt::T3
    wi::T4
    str_names::T5
    save::T6
end
function MeanRiskEstimator(;
                           r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                           obj::ObjectiveFunction = MinimumRisk(),
                           opt::JuMPOptimiser = JuMPOptimiser(),
                           wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                           str_names::Bool = false, save::Bool = true)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return MeanRiskEstimator{typeof(r), typeof(obj), typeof(opt), typeof(wi),
                             typeof(str_names), typeof(save)}(r, obj, opt, wi, str_names,
                                                              save)
end
function optimise!(mr::MeanRiskEstimator, rd::ReturnsResult = ReturnsResult())
    model = JuMP.Model()
    set_string_names_on_creation(model, mr.str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    pr, wb, lcs, cent, gcard, nplg, cplg = processed_jump_optimiser_attributes(mr.opt, rd)
    set_w!(model, pr.X, mr.wi)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, mr.opt.card, gcard, nplg, cplg, mr.opt.lt, mr.opt.st,
                         mr.opt.fees, mr.opt.ss)
    set_turnover_constraints!(model, mr.opt.tn)
    set_tracking_error_constraints!(model, pr.X, mr.opt.te)
    set_number_effective_assets!(model, mr.opt.nea)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, mr.opt.fees)
    set_risk_constraints!(model, mr.r, mr, pr, nplg, cplg)
    scalarise_risk_expression!(model, mr.opt.sce)
    ret = jump_returns_factory(mr.opt.ret, pr)
    set_return_constraints!(model, ret, mr.obj, pr)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, mr.opt.ccnt, mr, pr)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    retcode, sol = optimise_JuMP_model!(model, mr, eltype(pr.X))
    return JuMPOptimisationResult(Type{MeanRiskEstimator}, pr, wb, lcs, cent, gcard, nplg,
                                  cplg, retcode, sol, ifelse(mr.save, model, nothing))
end

export MeanRiskEstimator
