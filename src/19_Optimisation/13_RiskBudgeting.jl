abstract type RiskBudgetingAlgorithm <: OptimisationAlgorithm end
struct AssetRiskBudgeting{T1} <: RiskBudgetingAlgorithm
    rkb::T1
    function AssetRiskBudgeting(rkb::Union{Nothing, <:RiskBudgetEstimator,
                                           <:RiskBudgetResult})
        return new{typeof(rkb)}(rkb)
    end
end
function AssetRiskBudgeting(;
                            rkb::Union{Nothing, <:RiskBudgetEstimator, <:RiskBudgetResult} = nothing)
    return AssetRiskBudgeting(rkb)
end
function risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i::AbstractVector)
    return AssetRiskBudgeting(; rkb = risk_budget_view(r.rkb, i))
end
struct FactorRiskBudgeting{T1, T2, T3} <: RiskBudgetingAlgorithm
    re::T1
    rkb::T2
    flag::T3
    function FactorRiskBudgeting(re::Union{<:Regression, <:AbstractRegressionEstimator},
                                 rkb::Union{Nothing, <:RiskBudgetEstimator,
                                            <:RiskBudgetResult}, flag::Bool)
        return new{typeof(re), typeof(rkb), typeof(flag)}(re, rkb, flag)
    end
end
function FactorRiskBudgeting(;
                             re::Union{<:Regression, <:AbstractRegressionEstimator} = StepwiseRegression(),
                             rkb::Union{Nothing, <:RiskBudgetEstimator, <:RiskBudgetResult} = nothing,
                             flag::Bool = true)
    return FactorRiskBudgeting(re, rkb, flag)
end
function risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i::AbstractVector)
    re = regression_view(r.re, i)
    return FactorRiskBudgeting(; re = re, rkb = r.rkb, flag = r.flag)
end
struct RiskBudgeting{T1, T2, T3, T4, T5} <: JuMPOptimisationEstimator
    opt::T1
    r::T2
    rba::T3
    wi::T4
    fallback::T5
    function RiskBudgeting(opt::JuMPOptimiser,
                           r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                           rba::RiskBudgetingAlgorithm,
                           wi::Union{Nothing, <:AbstractVector{<:Real}},
                           fallback::Union{Nothing, <:OptimisationEstimator})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if isa(wi, AbstractVector)
            @argcheck(!isempty(wi))
        end
        if isa(rba.rkb, RiskBudgetEstimator)
            @argcheck(!isnothing(opt.sets))
        end
        return new{typeof(opt), typeof(r), typeof(rba), typeof(wi), typeof(fallback)}(opt,
                                                                                      r,
                                                                                      rba,
                                                                                      wi,
                                                                                      fallback)
    end
end
function RiskBudgeting(; opt::JuMPOptimiser = JuMPOptimiser(),
                       r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                       rba::RiskBudgetingAlgorithm = AssetRiskBudgeting(),
                       wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                       fallback::Union{Nothing, <:OptimisationEstimator} = nothing)
    return RiskBudgeting(opt, r, rba, wi, fallback)
end
function opt_view(rb::RiskBudgeting, i::AbstractVector, X::AbstractMatrix)
    X = isa(rb.opt.pe, AbstractPriorResult) ? rb.opt.pe.X : X
    opt = opt_view(rb.opt, i, X)
    r = risk_measure_view(rb.r, i, X)
    rba = risk_budgeting_algorithm_view(rb.rba, i)
    wi = nothing_scalar_array_view(rb.wi, i)
    return RiskBudgeting(; opt = opt, r = r, rba = rba, wi = wi, fallback = rb.fallback)
end
function _set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting,
                                          w::AbstractVector{<:AbstractJuMPScalar};
                                          strict::Bool = false)
    N = length(w)
    rkb = risk_budget_constraints(rb.rba.rkb, rb.opt.sets; N = N, strict = strict)
    rb = rkb.val
    @argcheck(length(rb) == N)
    sc = model[:sc]
    @variables(model, begin
                   k
                   log_w[1:N]
               end)
    @constraints(model,
                 begin
                     clog_w[i = 1:N],
                     [sc * log_w[i], sc, sc * w[i]] in MOI.ExponentialCone()
                     crkb, sc * dot(rb, log_w) >= 0
                 end)
    return rkb
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:AssetRiskBudgeting, <:Any},
                                         pr::AbstractPriorResult, wb::WeightBounds, args...)
    set_w!(model, pr.X, rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return ProcessedAssetRiskBudgetingAttributes(rkb)
end
function set_risk_budgeting_constraints!(model::JuMP.Model,
                                         rb::RiskBudgeting{<:Any, <:Any,
                                                           <:FactorRiskBudgeting, <:Any},
                                         ::Any, wb::WeightBounds, rd::ReturnsResult)
    b1, rr = set_factor_risk_contribution_constraints!(model, rb.rba.re, rd, rb.rba.flag,
                                                       rb.wi)
    rkb = _set_risk_budgeting_constraints!(model, rb, model[:w1]; strict = rb.opt.strict)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return ProcessedFactorRiskBudgetingAttributes(rkb, b1, rr)
end
function optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, plg, tn, fees, ret) = processed_jump_optimiser_attributes(rb.opt,
                                                                                                                                              rd;
                                                                                                                                              dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    prb = set_risk_budgeting_constraints!(model, rb, pr, wb, rd)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, cent, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, rb.opt.card, gcard, plg, lt, st, fees, rb.opt.ss)
    set_smip_constraints!(model, wb, rb.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, rb.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, rb.opt.te, rb, plg, fees; rd = rd)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, rb.r, rb, pr, plg, fees; rd = rd)
    scalarise_risk_expression!(model, rb.opt.sce)
    set_return_constraints!(model, ret, MinimumRisk(), pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, plg)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return if isa(retcode, OptimisationSuccess) || isnothing(rb.fallback)
        JuMPOptimisationRiskBudgeting(typeof(rb),
                                      ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs,
                                                                       cent, gcard, sgcard,
                                                                       smtx, sgmtx, slt,
                                                                       sst, sglt, sgst, plg,
                                                                       tn, fees, ret), prb,
                                      retcode, sol, ifelse(save, model, nothing))
    else
        @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        optimise(rb.fallback, rd; dims = dims, str_names = str_names, save = save,
                 kwargs...)
    end
end

export AssetRiskBudgeting, FactorRiskBudgeting, RiskBudgeting
