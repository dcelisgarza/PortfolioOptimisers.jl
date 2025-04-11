struct MeanRiskEstimator{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                         T2 <: ObjectiveFunction, T3 <: JuMPOptimiser} <:
       JuMPOptimisationType
    r::T1
    obj::T2
    opt::T3
end
function MeanRiskEstimator(;
                           r::Union{RiskMeasure, AbstractVector{<:RiskMeasure}} = StandardDeviation(),
                           obj::ObjectiveFunction = MinimumRisk(),
                           opt::JuMPOptimiser = JuMPOptimiser())
    return MeanRiskEstimator{typeof(r), typeof(obj), typeof(opt)}(r, obj, opt)
end
function cleanup_weights(model::JuMP.Model, ::MeanRiskEstimator)
    return value.(model[:w]) / value(model[:k])
end
struct MeanRiskModel{T1 <: AbstractPriorResult, T2 <: Union{Nothing, <:WeightBounds},
                     T3 <: Union{Nothing, <:LinearConstraintResult},
                     T4 <: Union{Nothing, <:LinearConstraintResult},
                     T5 <: Union{Nothing, <:LinearConstraintResult},
                     T6 <: Union{Nothing, <:PhilogenyConstraintResult},
                     T7 <: Union{Nothing, <:PhilogenyConstraintResult},
                     T8 <: JuMPPortfolioSolution} <: PortfolioModel
    pm::T1
    wb::T2
    lcs::T3
    cent::T4
    gcard::T5
    nplg::T6
    cplg::T7
    sol::T8
end
function optimise!(mr::MeanRiskEstimator, rd::ReturnsResult = ReturnsResult())
    model = JuMP.Model()
    set_string_names_on_creation(model, mr.opt.str_names)
    set_objective_penalty!(model)
    set_model_scales!(model, mr.opt.sc, mr.opt.so, mr.opt.ss)
    set_model_fees!(model)
    pm = prior(mr.opt.pe, rd.X, rd.F)
    datatype = eltype(pm.X)
    set_w!(model, pm.X, mr.opt.wi)
    set_maximum_ratio_factor_variables!(model, pm.mu, mr.obj)
    wb = weight_bounds_constraints(mr.opt.wb, mr.opt.sets; scalar = true, N = size(pm.X, 2),
                                   strict = mr.opt.strict)
    lcs = linear_constraints(mr.opt.lcs, mr.opt.sets; datatype = datatype,
                             strict = mr.opt.strict)
    cent = centrality_constraints(mr.opt.cent, pm.X)
    gcard = cardinality_constraints(mr.opt.card, mr.opt.sets; datatype = datatype,
                                    strict = mr.opt.strict)
    nplg = philogeny_constraints(mr.opt.nplg, pm.X)
    cplg = philogeny_constraints(mr.opt.cplg, pm.X)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :clcs_ineq, :clcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :clcm_ineq, :clcm_eq)
    set_mip_constraints!(model, mr.opt.bit, mr.opt.card, gcard, mr.opt.fees, nplg, cplg, wb)
    set_turnover_constraints!(model, mr.opt.tn)
    set_tracking_error_constraints!(model, pm.X, mr.opt.te)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, mr.opt.fees)
    set_risk_constraints!(model, mr.r, mr, pm, nplg, cplg)
    scalarise_risk_expression!(model, mr.opt.sce)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    set_return_constraints!(model, mr.opt.ret, mr.obj, pm)
    set_custom_constraint!(model, mr.opt.ccnt, mr, pm)
    set_portfolio_objective_function!(model, mr.obj, mr.opt.ret, mr.opt.cobj, mr, pm)
    sol = optimise_JuMP_model!(model, mr, datatype)
    return MeanRiskModel(pm, wb, lcs, cent, gcard, nplg, cplg, sol)
end

export MeanRiskEstimator
