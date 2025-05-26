struct MeanRisk{T1 <: JuMPOptimiser,
                T2 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T3 <: ObjectiveFunction, T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       JuMPOptimisationEstimator
    opt::T1
    r::T2
    obj::T3
    wi::T4
end
function MeanRisk(; opt::JuMPOptimiser = JuMPOptimiser(),
                  r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return MeanRisk{typeof(opt), typeof(r), typeof(obj), typeof(wi)}(opt, r, obj, wi)
end
function opt_view(mr::MeanRisk, i::AbstractVector, X::AbstractMatrix)
    opt = opt_view(mr.opt, i)
    r = risk_measure_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi)
end
function optimise!(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    pr, wb, lcs, cent, gcard, sgcard, smtx, nplg, cplg = processed_jump_optimiser_attributes(mr.opt,
                                                                                             rd;
                                                                                             dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    set_w!(model, pr.X, mr.wi)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, mr.opt.card, gcard, nplg, cplg, mr.opt.lt, mr.opt.st,
                         mr.opt.fees, mr.opt.ss)
    set_smip_constraints!(model, wb, mr.opt.scard, sgcard, smtx)
    set_turnover_constraints!(model, mr.opt.tn)
    set_noc_tracking_error_constraints!(model, pr, mr.opt.te1)
    set_soc_tracking_error_constraints!(model, pr, mr.opt.te2)
    set_vol_tracking_error_constraints!(model, pr, mr.opt.tev)
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
    return JuMPOptimisationResult(typeof(mr), pr, wb, lcs, cent, gcard, sgcard, smtx, nplg,
                                  cplg, retcode, sol, ifelse(save, model, nothing))
end

export MeanRisk
