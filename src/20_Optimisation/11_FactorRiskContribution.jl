struct FactorRiskContribution{T1 <: JuMPOptimiser,
                              T2 <:
                              Union{<:RegressionResult, <:AbstractRegressionEstimator},
                              T3 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                              T4 <: ObjectiveFunction,
                              T5 <:
                              Union{Nothing, <:SemiDefinitePhilogenyConstraintEstimator,
                                    <:SemiDefinitePhilogenyResult},
                              T6 <:
                              Union{Nothing, <:SemiDefinitePhilogenyConstraintEstimator,
                                    <:SemiDefinitePhilogenyResult},
                              T7 <: Union{Nothing, <:DataFrame},
                              T8 <: Union{Nothing, <:AbstractVector{<:Real}}, T9 <: Bool} <:
       JuMPOptimisationEstimator
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
                                re::Union{<:RegressionResult,
                                          <:AbstractRegressionEstimator} = StepwiseRegression(),
                                r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                                obj::ObjectiveFunction = MinimumRisk(),
                                nplg::Union{Nothing,
                                            <:SemiDefinitePhilogenyConstraintEstimator,
                                            <:SemiDefinitePhilogenyResult} = nothing,
                                cplg::Union{Nothing,
                                            <:SemiDefinitePhilogenyConstraintEstimator,
                                            <:SemiDefinitePhilogenyResult} = nothing,
                                sets::Union{Nothing, <:DataFrame} = nothing,
                                wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                flag::Bool = true)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    @smart_assert(!isa(opt.nplg,
                       Union{<:SemiDefinitePhilogenyConstraintEstimator,
                             <:SemiDefinitePhilogenyResult}))
    @smart_assert(!isa(opt.cplg,
                       Union{<:SemiDefinitePhilogenyConstraintEstimator,
                             <:SemiDefinitePhilogenyResult}))
    return FactorRiskContribution{typeof(opt), typeof(re), typeof(r), typeof(obj),
                                  typeof(nplg), typeof(cplg), typeof(sets), typeof(wi),
                                  typeof(flag)}(opt, re, r, obj, nplg, cplg, sets, wi, flag)
end
function opt_view(frc::FactorRiskContribution, i::AbstractVector, X::AbstractMatrix)
    opt = opt_view(frc.opt, i, X)
    re = regression_view(frc.re, i)
    r = risk_measure_view(frc.r, i, X)
    sets = nothing_dataframe_view(frc.sets, i)
    wi = nothing_scalar_array_view(frc.wi, i)
    return FactorRiskContribution(; opt = opt, re = re, r = r, obj = frc.obj,
                                  nplg = frc.nplg, cplg = frc.cplg, sets = sets, wi = wi,
                                  flag = frc.flag)
end
function set_factor_risk_contribution_constraints!(model::JuMP.Model,
                                                   re::Union{<:RegressionResult,
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
    (; pr, wb, lcs, cent, gcard, sgcard, smtx, nplg, cplg, ret) = processed_jump_optimiser_attributes(frc.opt,
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
    set_mip_constraints!(model, wb, frc.opt.card, gcard, nplg, cplg, frc.opt.lt, frc.opt.st,
                         frc.opt.fees, frc.opt.ss)
    set_smip_constraints!(model, wb, frc.opt.scard, sgcard, smtx, frc.opt.ss)
    set_turnover_constraints!(model, frc.opt.tn)
    set_tracking_error_constraints!(model, pr, frc.opt.te, frc, nplg, cplg)
    set_number_effective_assets!(model, frc.opt.nea)
    set_l1_regularisation!(model, frc.opt.l1)
    set_l2_regularisation!(model, frc.opt.l2)
    set_non_fixed_fees!(model, frc.opt.fees)
    set_risk_constraints!(model, frc.r, frc, pr, nplg, cplg, b1, frc.sets; iv = rd.iv,
                          ivpa = rd.ivpa)
    scalarise_risk_expression!(model, frc.opt.sce)
    set_return_constraints!(model, ret, frc.obj, pr; iv = rd.iv, ivpa = rd.ivpa)
    frc_nplg = philogeny_constraints(frc.nplg, rd.F)
    frc_cplg = philogeny_constraints(frc.cplg, rd.F)
    set_sdp_frc_philogeny_constraints!(model, frc_nplg, :sdp_frc_nplg)
    set_sdp_frc_philogeny_constraints!(model, frc_cplg, :sdp_frc_cplg)
    add_custom_constraint!(model, frc.opt.ccnt, frc, pr)
    set_portfolio_objective_function!(model, frc.obj, ret, frc.opt.cobj, frc, pr)
    retcode, sol = optimise_JuMP_model!(model, frc, eltype(pr.X))
    return JuMPOptimisationFactorRiskContributionResult(typeof(frc), pr, wb, lcs, cent,
                                                        gcard, sgcard, smtx, nplg, cplg,
                                                        frc_nplg, frc_cplg, retcode, sol,
                                                        ifelse(save, model, nothing))
end

export FactorRiskContribution
