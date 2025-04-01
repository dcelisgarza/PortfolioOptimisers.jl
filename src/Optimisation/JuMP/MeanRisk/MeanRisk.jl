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
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    pm = prior(mr.opt.pe, rd.X, rd.F)
    set_w!(model, pm.X, mr.opt.wi)
    set_maximum_ratio_factor_variables!(model, pm.mu, mr.obj)
    set_weight_constraints!(model, mr.opt.wb)
    set_long_short_bounds_constraints!(model, mr.opt.lss)

    return model
end

export MeanRisk
