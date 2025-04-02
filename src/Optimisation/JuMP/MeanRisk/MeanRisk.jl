struct MeanRisk{T1 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T2 <: ObjectiveFunction, T3 <: JuMPOptimiser} <: JuMPOptimisationType
    risk::T1
    obj::T2
    opt::T3
end
function MeanRisk(;
                  risk::Union{RiskMeasure, AbstractVector{<:RiskMeasure}} = StandardDeviation(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  opt::JuMPOptimiser = JuMPOptimiser())
    return MeanRisk{typeof(risk), typeof(obj), typeof(opt)}(risk, obj, opt)
end
function optimise!(mr::MeanRisk, rd::ReturnsData = ReturnsData())
    model = JuMP.Model()
    set_string_names_on_creation(model, mr.opt.str_names)
    set_objective_penalty!(model)
    set_model_scales!(model, mr.opt.sc, mr.opt.so, mr.opt.ss)
    pm = prior(mr.opt.pe, rd.X, rd.F)
    set_w!(model, pm.X, mr.opt.wi)
    set_maximum_ratio_factor_variables!(model, pm.mu, mr.obj)
    set_weight_constraints!(model, mr.opt.wb)
    set_long_short_bounds_constraints!(model, mr.opt.lss)
    set_linear_weight_constraints!(model,
                                   linear_constraints(mr.opt.lcs, mr.opt.sets;
                                                      datatype = eltype(pm.X),
                                                      strict = mr.opt.strict), :clcs_ineq,
                                   :clcs_eq)
    set_linear_weight_constraints!(model, centrality_constraints(mr.opt.cent, pm.X),
                                   :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :clcm_ineq, :clcm_eq)
    cadj = philogeny_constraints(mr.opt.cadj, pm.X)
    nadj = philogeny_constraints(mr.opt.nadj, pm.X)
    set_mip_constraints!(model, mr.opt.bit,
                         cardinality_constraints(mr.opt.card, mr.opt.sets;
                                                 datatype = eltype(pm.X),
                                                 strict = mr.opt.strict), mr.opt.fees, cadj,
                         nadj, mr.opt.wb)
    set_turnover_constraints!(model, mr.opt.tn)
    set_tracking_error_constraints!(model, pm.X, mr.opt.te)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, mr.opt.wb, mr.opt.fees)
    #!risk constraints
    #!scalarise risk expression
    #!return constraints
    set_sdp_philogeny_constraints!(model, cadj, :sdp_cadj)
    set_sdp_philogeny_constraints!(model, nadj, :sdp_nadj)
    set_custom_constraint!(model, mr.opt.ccnt, pm, mr)
    #!set_portfolio_objective_function!(model, mr.obj, mr.opt.ret, mr.opt.cobj)

    return model
end

export MeanRisk
