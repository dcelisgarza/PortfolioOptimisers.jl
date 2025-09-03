abstract type RiskBudgettingAlgorithm <: OptimisationAlgorithm end
struct AssetRiskBudgetting{T1} <: RiskBudgettingAlgorithm
    rkb::T1
end
function AssetRiskBudgetting(;
                             rkb::Union{Nothing, <:RiskBudgetEstimator, <:RiskBudgetResult} = nothing)
    return AssetRiskBudgetting(rkb)
end
function risk_budgetting_algorithm_view(r::AssetRiskBudgetting, i::AbstractVector)
    return AssetRiskBudgetting(; rkb = risk_budget_view(r.rkb, i))
end
struct FactorRiskBudgetting{T1, T2, T3} <: RiskBudgettingAlgorithm
    re::T1
    rkb::T2
    flag::T3
end
function FactorRiskBudgetting(;
                              re::Union{<:Regression, <:AbstractRegressionEstimator} = StepwiseRegression(),
                              rkb::Union{Nothing, <:RiskBudgetEstimator,
                                         <:RiskBudgetResult} = nothing, flag::Bool = true)
    return FactorRiskBudgetting(re, rkb, flag)
end
function risk_budgetting_algorithm_view(r::FactorRiskBudgetting, i::AbstractVector)
    re = regression_view(r.re, i)
    return FactorRiskBudgetting(; re = re, rkb = r.rkb, flag = r.flag)
end
struct RiskBudgetting{T1, T2, T3, T4} <: JuMPOptimisationEstimator
    opt::T1
    r::T2
    alg::T3
    wi::T4
end
function RiskBudgetting(; opt::JuMPOptimiser = JuMPOptimiser(),
                        r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                        alg::RiskBudgettingAlgorithm = AssetRiskBudgetting(),
                        wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(r, AbstractVector)
        @argcheck(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @argcheck(!isempty(wi))
    end
    if isa(alg.rkb, RiskBudgetEstimator)
        @argcheck(!isnothing(opt.sets))
    end
    return RiskBudgetting(opt, r, alg, wi)
end
function opt_view(rb::RiskBudgetting, i::AbstractVector, X::AbstractMatrix)
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    alg = risk_budgetting_algorithm_view(rb.alg, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgetting(; opt = opt, r = r, alg = alg, wi = wi)
end
function _set_risk_budgetting_constraints!(model::JuMP.Model, rb::RiskBudgetting,
                                           w::AbstractVector{<:AbstractJuMPScalar},
                                           sets::Union{Nothing, <:AssetSets} = nothing;
                                           strict::Bool = false,
                                           datatype::DataType = Float64)
    N = length(w)
    rkb = risk_budget_constraints(rb.alg.rkb, sets; N = N, strict = strict,
                                  datatype = datatype)
    rb = rkb.val
    @argcheck(length(rb) == N)
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
                     crkb, sc * (dot(rb, log_w) - c) >= 0
                 end)
    return rkb
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgetting{<:Any, <:Any,
                                                             <:AssetRiskBudgetting, <:Any},
                                          sets::Union{Nothing, <:AssetSets},
                                          pr::AbstractPriorResult, wb::WeightBounds,
                                          args...)
    set_w!(model, pr.X, rb.wi)
    rkb = _set_risk_budgetting_constraints!(model, rb, model[:w], sets;
                                            strict = rb.opt.strict, datatype = eltype(pr.X))
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return ProcessedAssetRiskBudgettingAttributes(rkb)
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgetting{<:Any, <:Any,
                                                             <:FactorRiskBudgetting, <:Any},
                                          sets::Union{Nothing, <:AssetSets},
                                          pr::AbstractPriorResult, wb::WeightBounds,
                                          rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rb.alg.re, rd, rb.alg.flag,
                                                       rb.wi)
    rkb = _set_risk_budgetting_constraints!(model, rb, model[:w1], sets;
                                            strict = rb.opt.strict, datatype = eltype(pr.X))
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedFactorRiskBudgettingAttributes(rkb, b1, rr)
end
function optimise!(rb::RiskBudgetting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, nplg, cplg, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                                     rd;
                                                                                                                                                     dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgetting_constraints!(model, rb, rb.opt.sets, pr, wb, rd)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, rb.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, rb.opt.card, gcard, nplg, cplg, lt, st, fees, rb.opt.ss)
    set_smip_constraints!(model, wb, rb.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.te, rb, nplg, cplg, fees)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, nplg, cplg; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sce)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_phylogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return JuMPOptimisationRiskBudgetting(typeof(rb),
                                          ProcessedJuMPOptimiserAttributes(pr, wb, lt, st,
                                                                           lcs, cent, gcard,
                                                                           sgcard, smtx,
                                                                           sgmtx, slt, sst,
                                                                           sglt, sgst, nplg,
                                                                           cplg, tn, fees,
                                                                           ret), prb,
                                          retcode, sol, ifelse(save, model, nothing))
end

export AssetRiskBudgetting, FactorRiskBudgetting, RiskBudgetting
