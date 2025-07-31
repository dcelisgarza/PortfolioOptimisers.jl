abstract type RiskBudgettingAlgorithm <: OptimisationAlgorithm end
struct AssetRiskBudgettingAlgorithm{T1 <: Union{Nothing, <:AbstractVector}} <:
       RiskBudgettingAlgorithm
    rkb::T1
end
function AssetRiskBudgettingAlgorithm(; rkb::Union{Nothing, <:AbstractVector} = nothing)
    if isa(rkb, AbstractVector)
        @smart_assert(!isempty(rkb))
    end
    return AssetRiskBudgettingAlgorithm{typeof(rkb)}(rkb)
end
function risk_budgetting_algorithm_view(r::AssetRiskBudgettingAlgorithm, i::AbstractVector)
    rkb = nothing_scalar_array_view(r.rkb, i)
    return AssetRiskBudgettingAlgorithm(; rkb = rkb)
end
struct FactorRiskBudgettingAlgorithm{T1 <: Union{<:RegressionResult,
                                                 <:AbstractRegressionEstimator},
                                     T2 <: Union{Nothing, <:AbstractVector}, T3 <: Bool} <:
       RiskBudgettingAlgorithm
    re::T1
    rkb::T2
    flag::T3
end
function FactorRiskBudgettingAlgorithm(;
                                       re::Union{<:RegressionResult,
                                                 <:AbstractRegressionEstimator} = StepwiseRegression(),
                                       rkb::Union{Nothing, <:AbstractVector} = nothing,
                                       flag::Bool = true)
    if isa(rkb, AbstractVector)
        @smart_assert(!isempty(rkb))
    end
    return FactorRiskBudgettingAlgorithm{typeof(re), typeof(rkb), typeof(flag)}(re, rkb,
                                                                                flag)
end
function risk_budgetting_algorithm_view(r::FactorRiskBudgettingAlgorithm, i::AbstractVector)
    re = regression_view(r.re, i)
    return FactorRiskBudgettingAlgorithm(; re = re, rkb = r.rkb, flag = r.flag)
end
struct RiskBudgetting{T1 <: JuMPOptimiser,
                      T2 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                      T3 <: RiskBudgettingAlgorithm,
                      T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       JuMPOptimisationEstimator
    opt::T1
    r::T2
    alg::T3
    wi::T4
end
function RiskBudgetting(; opt::JuMPOptimiser = JuMPOptimiser(),
                        r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                        alg::RiskBudgettingAlgorithm = AssetRiskBudgettingAlgorithm(),
                        wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return RiskBudgetting{typeof(opt), typeof(r), typeof(alg), typeof(wi)}(opt, r, alg, wi)
end
function opt_view(rb::RiskBudgetting, i::AbstractVector, X::AbstractMatrix)
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    alg = risk_budgetting_algorithm_view(rb.alg, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgetting(; opt = opt, r = r, alg = alg, wi = wi)
end
function _set_risk_budgetting_constraints!(model::JuMP.Model, rb::RiskBudgetting, w)
    N = length(w)
    rkb = rb.alg.rkb
    if isnothing(rkb)
        rkb = range(; start = inv(N), stop = inv(N), length = N)
    else
        @smart_assert(length(rkb) == N)
    end
    sc = model[:sc]
    @variables(model, begin
                   k
                   log_w[1:N]
                   c >= 0
               end)
    @constraints(model,
                 begin
                     clog_w[i = 1:N],
                     [sc * log_w[i], sc, sc * w[i]] in MOI.ExponentialCone()
                     crkb, sc * (dot(rkb, log_w) - c) >= 0
                 end)
    return nothing
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgetting{<:Any, <:Any,
                                                             <:AssetRiskBudgettingAlgorithm,
                                                             <:Any},
                                          pr::AbstractPriorResult, wb::WeightBoundsResult,
                                          args...)
    set_w!(model, pr.X, rb.wi)
    _set_risk_budgetting_constraints!(model, rb, model[:w])
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return nothing
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgetting{<:Any, <:Any,
                                                             <:FactorRiskBudgettingAlgorithm,
                                                             <:Any}, ::Any,
                                          wb::WeightBoundsResult, rd::ReturnsResult)
    set_factor_risk_contribution_constraints!(model, rb.alg.re, rd, rb.alg.flag, rb.wi)
    _set_risk_budgetting_constraints!(model, rb, model[:w1])
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return nothing
end
function optimise!(rb::RiskBudgetting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                         rd;
                                                                                                                                         dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    set_risk_budgetting_constraints!(model, rb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, rb.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, rb.opt.card, gcard, nplg, cplg, lt, st, fees, rb.opt.ss)
    set_smip_constraints!(model, wb, rb.opt.scard, sgcard, smtx, sgmtx, nothing, nothing,
                          nothing, nothing, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.te, rb, nplg, cplg, fees)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, nplg, cplg; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sce)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return JuMPOptimisationResult(typeof(rb),
                                  ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs,
                                                                   cent, gcard, sgcard,
                                                                   smtx, slt, sst, sgmtx,
                                                                   nplg, cplg, tn, fees,
                                                                   ret), retcode, sol,
                                  ifelse(save, model, nothing))
end

export AssetRiskBudgettingAlgorithm, FactorRiskBudgettingAlgorithm, RiskBudgetting
