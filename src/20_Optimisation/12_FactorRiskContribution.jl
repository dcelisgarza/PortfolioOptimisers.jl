struct FactorRiskContribution{T1, T2, T3, T4, T5, T6, T7, T8, T9} <:
       RiskJuMPOptimisationEstimator
    opt::T1
    re::T2
    r::T3
    obj::T4
    plg::T5
    sets::T6
    wi::T7
    flag::T8
    fb::T9
    function FactorRiskContribution(opt::JuMPOptimiser, re::RegE_Reg, r::RM_VecRM,
                                    obj::ObjectiveFunction,
                                    plg::Option{<:PhCE_PhC_VecPhCE_PhC},
                                    sets::Option{<:AssetSets}, wi::Option{<:VecNum},
                                    flag::Bool, fb::Option{<:OptimisationEstimator})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, VecNum)
            @argcheck(!isempty(wi))
        end
        return new{typeof(opt), typeof(re), typeof(r), typeof(obj), typeof(plg),
                   typeof(sets), typeof(wi), typeof(flag), typeof(fb)}(opt, re, r, obj, plg,
                                                                       sets, wi, flag, fb)
    end
end
function FactorRiskContribution(; opt::JuMPOptimiser = JuMPOptimiser(),
                                re::RegE_Reg = StepwiseRegression(),
                                r::RM_VecRM = Variance(),
                                obj::ObjectiveFunction = MinimumRisk(),
                                plg::Option{<:PhCE_PhC_VecPhCE_PhC} = nothing,
                                sets::Option{<:AssetSets} = nothing,
                                wi::Option{<:VecNum} = nothing, flag::Bool = true,
                                fb::Option{<:OptimisationEstimator} = nothing)
    return FactorRiskContribution(opt, re, r, obj, plg, sets, wi, flag, fb)
end
function opt_view(frc::FactorRiskContribution, i, X::MatNum)
    X = isa(frc.opt.pe, AbstractPriorResult) ? frc.opt.pe.X : X
    opt = opt_view(frc.opt, i, X)
    re = regression_view(frc.re, i)
    r = risk_measure_view(frc.r, i, X)
    return FactorRiskContribution(; opt = opt, re = re, r = r, obj = frc.obj, plg = frc.plg,
                                  sets = frc.sets, wi = frc.wi, flag = frc.flag,
                                  fb = frc.fb)
end
function set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg,
                                                   rd::ReturnsResult, flag::Bool,
                                                   wi::Option{<:VecNum})
    rr = regression(re, rd.X, rd.F)
    Bt = transpose(rr.L)
    b1 = pinv(Bt)
    Nf = size(b1, 2)
    if flag
        b2 = pinv(transpose(nullspace(Bt)))
        N = size(rr.M, 1)
        JuMP.@variables(model, begin
                            w1[1:Nf]
                            w2[1:(N - Nf)]
                        end)
        JuMP.@expression(model, w, b1 * w1 + b2 * w2)
    else
        JuMP.@variable(model, w1[1:Nf])
        JuMP.@expression(model, w, b1 * w1)
    end
    set_initial_w!(w1, wi)
    return b1, rr
end
function _optimise(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, plg, tn, fees, ret) = processed_jump_optimiser_attributes(frc.opt,
                                                                                                                                              rd;
                                                                                                                                              dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, frc.obj)
    b1, rr = set_factor_risk_contribution_constraints!(model, frc.re, rd, frc.flag, frc.wi)
    set_weight_constraints!(model, wb, frc.opt.bgt, frc.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, cent, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, frc.opt.card, gcard, plg, lt, st, fees, frc.opt.ss)
    set_smip_constraints!(model, wb, frc.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, frc.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, frc.opt.te, frc, plg, fees, b1; rd = rd)
    set_number_effective_assets!(model, frc.opt.nea)
    set_l1_regularisation!(model, frc.opt.l1)
    set_l2_regularisation!(model, frc.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, frc.r, frc, pr, plg, fees, b1; rd = rd)
    scalarise_risk_expression!(model, frc.opt.sca)
    set_return_constraints!(model, ret, frc.obj, pr; rd = rd)
    frc_plg = phylogeny_constraints(frc.plg, rd.F)
    set_sdp_frc_phylogeny_constraints!(model, frc_plg)
    add_custom_constraint!(model, frc.opt.ccnt, frc, pr)
    set_portfolio_objective_function!(model, frc.obj, ret, frc.opt.cobj, frc, pr)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(pr.X))
    return JuMPOptimisationFactorRiskContribution(typeof(frc),
                                                  ProcessedJuMPOptimiserAttributes(pr, wb,
                                                                                   lt, st,
                                                                                   lcs,
                                                                                   cent,
                                                                                   gcard,
                                                                                   sgcard,
                                                                                   smtx,
                                                                                   sgmtx,
                                                                                   slt, sst,
                                                                                   sglt,
                                                                                   sgst,
                                                                                   plg, tn,
                                                                                   fees,
                                                                                   ret), rr,
                                                  frc_plg, retcode, sol,
                                                  ifelse(save, model, nothing), nothing)
end
function optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(frc, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export FactorRiskContribution
