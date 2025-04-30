struct MeanRiskEstimator{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                         T2 <: ObjectiveFunction, T3 <: JuMPOptimiser} <:
       JuMPOptimisationEstimator
    r::T1
    obj::T2
    opt::T3
end
function MeanRiskEstimator(;
                           r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                           obj::ObjectiveFunction = MinimumRisk(),
                           opt::JuMPOptimiser = JuMPOptimiser())
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    return MeanRiskEstimator{typeof(r), typeof(obj), typeof(opt)}(r, obj, opt)
end
function optimise!(mr::MeanRiskEstimator, rd::ReturnsResult = ReturnsResult())
    model = JuMP.Model()
    set_string_names_on_creation(model, mr.opt.str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    pr = prior(mr.opt.pe, rd.X, rd.F)
    datatype = eltype(pr.X)
    set_w!(model, pr.X, mr.opt.wi)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    wb = weight_bounds_constraints(mr.opt.wb, mr.opt.sets; N = size(pr.X, 2),
                                   strict = mr.opt.strict)
    lcs = linear_constraints(mr.opt.lcs, mr.opt.sets; datatype = datatype,
                             strict = mr.opt.strict)
    cent = centrality_constraints(mr.opt.cent, pr.X)
    gcard = cardinality_constraints(mr.opt.gcard, mr.opt.sets; datatype = datatype,
                                    strict = mr.opt.strict)
    nplg = philogeny_constraints(mr.opt.nplg, pr.X)
    cplg = philogeny_constraints(mr.opt.cplg, pr.X)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :clcs_ineq, :clcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :clcm_ineq, :clcm_eq)
    set_mip_constraints!(model, wb, mr.opt.card, gcard, nplg, cplg, mr.opt.bit, mr.opt.fees,
                         mr.opt.ss)
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
    retcode, sol = optimise_JuMP_model!(model, mr, datatype)
    return JuMPOptimisationResult(Type{MeanRiskEstimator}, pr, wb, lcs, cent, gcard, nplg,
                                  cplg, retcode, sol, model)
end

export MeanRiskEstimator
