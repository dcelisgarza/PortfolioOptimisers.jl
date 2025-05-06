abstract type RiskBudgettingAlgorithm <: OptimisationAlgorithm end
struct AssetRiskBudgettingAlgorithm{T1 <: Union{Nothing, <:AbstractVector}} <:
       RiskBudgettingAlgorithm
    w::T1
end
function AssetRiskBudgettingAlgorithm(; w::Union{Nothing, <:AbstractVector} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return AssetRiskBudgettingAlgorithm{typeof(w)}(w)
end
struct FactorRiskBudgettingAlgorithm{T1 <: Bool, T2 <: Union{Nothing, <:AbstractVector}} <:
       RiskBudgettingAlgorithm
    flag::T1
    w::T2
end
function FactorRiskBudgettingAlgorithm(; flag::Bool = true,
                                       w::Union{Nothing, <:AbstractVector} = nothing)
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return FactorRiskBudgettingAlgorithm{typeof(flag), typeof(w)}(flag, w)
end
struct RiskBudgettingEstimator{T1 <: RiskBudgettingAlgorithm,
                               T2 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                               T3 <: JuMPOptimiser,
                               T4 <: Union{Nothing, <:AbstractVector{<:Real}}, T5 <: Bool,
                               T6 <: Bool} <: JuMPOptimisationEstimator
    alg::T1
    r::T2
    opt::T3
    wi::T4
    str_names::T5
    save::T6
end
function RiskBudgettingEstimator(;
                                 alg::RiskBudgettingAlgorithm = AssetRiskBudgettingAlgorithm(),
                                 r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                                 opt::JuMPOptimiser = JuMPOptimiser(),
                                 wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                                 str_names::Bool = false, save::Bool = true)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return RiskBudgettingEstimator{typeof(alg), typeof(r), typeof(opt), typeof(wi),
                                   typeof(str_names), typeof(save)}(alg, r, opt, wi,
                                                                    str_names, save)
end
function _set_risk_budgetting_constraints!(model::JuMP.Model, rb::RiskBudgettingEstimator,
                                           w)
    N = length(w)
    rkb = rb.alg.w
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
                     [sc * log_w[i], sc, sc * w[i]] ∈ MOI.ExponentialCone()
                     crkb, sc * (dot(rkb, log_w) - c) >= 0
                 end)
    return nothing
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgettingEstimator{<:AssetRiskBudgettingAlgorithm,
                                                                      <:Any, <:Any, <:Any,
                                                                      <:Any, <:Any},
                                          pr::AbstractPriorResult, wb::WeightBoundsResult)
    set_w!(model, pr.X, rb.wi)
    _set_risk_budgetting_constraints!(model, rb, model[:w])
    set_weight_constraints!(model, wb, rb.opt.bgt, nothing, true)
    return nothing
end
function set_risk_budgetting_constraints!(model::JuMP.Model,
                                          rb::RiskBudgettingEstimator{<:FactorRiskBudgettingAlgorithm,
                                                                      <:Any, <:Any, <:Any,
                                                                      <:Any, <:Any},
                                          pr::Union{<:FactorPriorResult,
                                                    <:EmpiricalPartialFactorPriorResult},
                                          wb::WeightBoundsResult)
    B = pr.loadings.frc
    b1 = pinv(transpose(B))
    Nf = size(b1, 2)
    if rb.alg.flag
        b2 = pinv(transpose(nullspace(B)))
        N = size(pr.X, 2)
        @variables(model, begin
                       w1[1:Nf]
                       w2[1:(N - Nf)]
                   end)
        @expression(model, w, b1 * w1 + b2 * w2)
    else
        @variable(model, w1[1:Nf])
        @expression(model, w, b1 * w1)
    end
    set_initial_w!(w1, rb.wi)
    _set_risk_budgetting_constraints!(model, rb, w1)
    set_weight_constraints!(model, wb, rb.opt.bgt, rb.opt.sbgt)
    return nothing
end
function optimise!(rb::RiskBudgettingEstimator, rd::ReturnsResult = ReturnsResult())
    pr, wb, lcs, cent, gcard, nplg, cplg = processed_jump_optimiser_attributes(rb.opt, rd)
    model = JuMP.Model()
    set_string_names_on_creation(model, rb.str_names)
    set_model_scales!(model, rb.opt.sc, rb.opt.so)
    set_risk_budgetting_constraints!(model, rb, pr, wb)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, rb.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, rb.opt.card, gcard, nplg, cplg, rb.opt.lt, rb.opt.st,
                         rb.opt.fees, rb.opt.ss)
    set_turnover_constraints!(model, rb.opt.tn)
    set_tracking_error_constraints!(model, pr.X, rb.opt.te)
    set_number_effective_assets!(model, rb.opt.nea)
    set_l1_regularisation!(model, rb.opt.l1)
    set_l2_regularisation!(model, rb.opt.l2)
    set_non_fixed_fees!(model, rb.opt.fees)
    set_risk_constraints!(model, rb.r, rb, pr, nplg, cplg)
    scalarise_risk_expression!(model, rb.opt.sce)
    ret = jump_returns_factory(rb.opt.ret, pr)
    set_return_constraints!(model, ret, MinimumRisk(), pr)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, rb.opt.ccnt, rb, pr)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, rb.opt.cobj, rb, pr)
    retcode, sol = optimise_JuMP_model!(model, rb, eltype(pr.X))
    return JuMPOptimisationResult(typeof(rb), pr, wb, lcs, cent, gcard, nplg, cplg, retcode,
                                  sol, ifelse(rb.save, model, nothing))
end

export AssetRiskBudgettingAlgorithm, FactorRiskBudgettingAlgorithm, RiskBudgettingEstimator
