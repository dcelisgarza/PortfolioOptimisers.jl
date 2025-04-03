struct MeanRisk{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T2 <: ObjectiveFunction, T3 <: JuMPOptimiser} <: JuMPOptimisationType
    r::T1
    obj::T2
    opt::T3
end
function MeanRisk(;
                  r::Union{RiskMeasure, AbstractVector{<:RiskMeasure}} = StandardDeviation(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  opt::JuMPOptimiser = JuMPOptimiser())
    return MeanRisk{typeof(r), typeof(obj), typeof(opt)}(r, obj, opt)
end
function cleanup_weights(model, ::MeanRisk)
    return value.(model[:w]) / value(model[:k])
end
function optimise!(mr::MeanRisk, rd::ReturnsData = ReturnsData())
    model = JuMP.Model()
    set_string_names_on_creation(model, mr.opt.str_names)
    set_objective_penalty!(model)
    set_model_scales!(model, mr.opt.sc, mr.opt.so, mr.opt.ss)
    set_model_fees!(model)
    pm = prior(mr.opt.pe, rd.X, rd.F)
    datatype = eltype(pm.X)
    set_w!(model, pm.X, mr.opt.wi)
    wb = weight_bounds_constraints(mr.opt.wb, mr.opt.sets; scalar = true, N = size(pm.X, 2),
                                   strict = mr.opt.strict)
    set_maximum_ratio_factor_variables!(model, pm.mu, mr.obj)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model,
                                   linear_constraints(mr.opt.lcs, mr.opt.sets;
                                                      datatype = datatype,
                                                      strict = mr.opt.strict), :clcs_ineq,
                                   :clcs_eq)
    set_linear_weight_constraints!(model, centrality_constraints(mr.opt.cent, pm.X),
                                   :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :clcm_ineq, :clcm_eq)
    cadj = philogeny_constraints(mr.opt.cadj, pm.X)
    nadj = philogeny_constraints(mr.opt.nadj, pm.X)
    set_mip_constraints!(model, mr.opt.bit,
                         cardinality_constraints(mr.opt.card, mr.opt.sets;
                                                 datatype = datatype,
                                                 strict = mr.opt.strict), mr.opt.fees, cadj,
                         nadj, wb)
    set_turnover_constraints!(model, mr.opt.tn)
    set_tracking_error_constraints!(model, pm.X, mr.opt.te)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, mr.opt.fees)
    set_risk_constraints!(model, mr.r, mr, pm, cadj, nadj)
    scalarise_risk_expression!(model, mr.opt.sce)
    set_sdp_philogeny_constraints!(model, cadj, :sdp_cadj)
    set_sdp_philogeny_constraints!(model, nadj, :sdp_nadj)
    set_return_constraints!(model, mr.opt.ret, mr.obj, pm)
    set_custom_constraint!(model, mr.opt.ccnt, mr, pm)
    set_portfolio_objective_function!(model, mr.obj, mr.opt.ret, mr.opt.cobj, mr, pm)
    return optimise_JuMP_model!(model, mr, datatype)
end

export MeanRisk
