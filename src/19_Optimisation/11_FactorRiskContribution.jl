struct FactorRiskContribution{T1 <: JuMPOptimiser,
                              T2 <: Union{<:Regression, <:AbstractRegressionEstimator},
                              T3 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                              T4 <: ObjectiveFunction,
                              T5 <: Union{Nothing, <:SemiDefinitePhilogenyEstimator,
                                          <:SemiDefinitePhilogeny},
                              T6 <: Union{Nothing, <:SemiDefinitePhilogenyEstimator,
                                          <:SemiDefinitePhilogeny},
                              T7 <: Union{Nothing, <:AssetSets,
                                #! Start: to delete
                                          <:DataFrame
                                #! End: to delete
                                }, T8 <: Union{Nothing, <:AbstractVector{<:Real}},
                              T9 <: Bool} <: JuMPOptimisationEstimator
    opt::T1
    re::T2
    r::T3
    obj::T4
    nplg::T5
    cplg::T6
    sets::T7
    wi::T8
    flag::T9
end
function FactorRiskContribution(; opt::JuMPOptimiser = JuMPOptimiser(),
                                re::Union{<:Regression, <:AbstractRegressionEstimator} = StepwiseRegression(),
                                r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                                obj::ObjectiveFunction = MinimumRisk(),
                                nplg::Union{Nothing, <:SemiDefinitePhilogenyEstimator,
                                            <:SemiDefinitePhilogeny} = nothing,
                                cplg::Union{Nothing, <:SemiDefinitePhilogenyEstimator,
                                            <:SemiDefinitePhilogeny} = nothing,
                                sets::Union{Nothing, <:AssetSets,
                                            #! Start: to delete
                                            <:DataFrame
                                            #! Start: to delete
                                            } = nothing,
                                wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                flag::Bool = true)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    @smart_assert(!isa(opt.nplg,
                       Union{<:SemiDefinitePhilogenyEstimator, <:SemiDefinitePhilogeny}))
    @smart_assert(!isa(opt.cplg,
                       Union{<:SemiDefinitePhilogenyEstimator, <:SemiDefinitePhilogeny}))
    return FactorRiskContribution{typeof(opt), typeof(re), typeof(r), typeof(obj),
                                  typeof(nplg), typeof(cplg), typeof(sets), typeof(wi),
                                  typeof(flag)}(opt, re, r, obj, nplg, cplg, sets, wi, flag)
end
function opt_view(frc::FactorRiskContribution, i::AbstractVector, X::AbstractMatrix)
    X = isa(frc.opt.pe, AbstractPriorResult) ? frc.opt.pe.X : X
    opt = opt_view(frc.opt, i, X)
    re = regression_view(frc.re, i)
    r = risk_measure_view(frc.r, i, X)
    sets = nothing_asset_sets_view(frc.sets, i)
    wi = nothing_scalar_array_view(frc.wi, i)
    return FactorRiskContribution(; opt = opt, re = re, r = r, obj = frc.obj,
                                  nplg = frc.nplg, cplg = frc.cplg, sets = sets, wi = wi,
                                  flag = frc.flag)
end
function set_factor_risk_contribution_constraints!(model::JuMP.Model,
                                                   re::Union{<:Regression,
                                                             <:AbstractRegressionEstimator},
                                                   rd::ReturnsResult, flag::Bool,
                                                   wi::Union{Nothing, AbstractVector})
    loadings = regression(re, rd.X, rd.F)
    Bt = transpose(loadings.L)
    b1 = pinv(Bt)
    Nf = size(b1, 2)
    if flag
        b2 = pinv(transpose(nullspace(Bt)))
        N = size(loadings.M, 1)
        @variables(model, begin
                       w1[1:Nf]
                       w2[1:(N - Nf)]
                   end)
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        @variable(model, w1[1:Nf])
        @expression(model, w, b1 * w1)
    end
    set_initial_w!(w1, wi)
    return b1
end
function optimise!(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(frc.opt,
                                                                                                                                         rd;
                                                                                                                                         dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, frc.opt.sc, frc.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, frc.obj)
    b1 = set_risk_budgetting_constraints!(model, frc.re, rd, frc.flag, frc.wi)
    set_weight_constraints!(model, wb, frc.opt.bgt, frc.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, frc.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, frc.opt.card, gcard, nplg, cplg, lt, st, fees,
                         frc.opt.ss)
    set_smip_constraints!(model, wb, frc.opt.scard, sgcard, smtx, sgmtx, slt, sst, nothing,
                          nothing, frc.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, frc.opt.te, frc, nplg, cplg, fees)
    set_number_effective_assets!(model, frc.opt.nea)
    set_l1_regularisation!(model, frc.opt.l1)
    set_l2_regularisation!(model, frc.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, frc.r, frc, pr, nplg, cplg, b1, frc.sets; rd = rd)
    scalarise_risk_expression!(model, frc.opt.sce)
    set_return_constraints!(model, ret, frc.obj, pr; rd = rd)
    frc_nplg = philogeny_constraints(frc.nplg, rd.F)
    frc_cplg = philogeny_constraints(frc.cplg, rd.F)
    set_sdp_frc_philogeny_constraints!(model, frc_nplg, :sdp_frc_nplg)
    set_sdp_frc_philogeny_constraints!(model, frc_cplg, :sdp_frc_cplg)
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
                                                                                   slt, sst,
                                                                                   sgmtx,
                                                                                   #    sglt,
                                                                                   #    sgst,
                                                                                   nplg,
                                                                                   cplg, tn,
                                                                                   fees,
                                                                                   ret),
                                                  frc_nplg, frc_cplg, retcode, sol,
                                                  ifelse(save, model, nothing))
end

export FactorRiskContribution
