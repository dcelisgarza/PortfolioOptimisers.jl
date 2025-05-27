abstract type NearOptimalCenteringAlgorithm <: OptimisationAlgorithm end
struct ConstrainedNearOptimalCenteringAlgorithm <: NearOptimalCenteringAlgorithm end
struct UnconstrainedNearOptimalCenteringAlgorithm <: NearOptimalCenteringAlgorithm end
struct NearOptimalCentering{T1 <: JuMPOptimiser,
                            T2 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                            T3 <: Union{Nothing, <:ObjectiveFunction},
                            T4 <: Union{Nothing, <:Real},
                            T5 <: Union{Nothing, <:AbstractVector},
                            T6 <: Union{Nothing, <:AbstractVector},
                            T7 <: Union{Nothing, <:AbstractVector},
                            T8 <: Union{Nothing, <:AbstractVector},
                            T9 <: Union{Nothing, <:AbstractVector},
                            T10 <: Union{Nothing, <:AbstractVector}, T11 <: Bool,
                            T12 <: NearOptimalCenteringAlgorithm} <:
       JuMPOptimisationEstimator
    opt::T1
    r::T2
    obj::T3
    bins::T4
    w_min::T5
    w_min_ini::T6
    w_opt::T7
    w_opt_ini::T8
    w_max::T9
    w_max_ini::T10
    ucs_flag::T11
    alg::T12
end
function NearOptimalCentering(; opt::JuMPOptimiser = JuMPOptimiser(),
                              r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = StandardDeviation(),
                              obj::Union{Nothing, <:ObjectiveFunction} = MinimumRisk(),
                              bins::Union{Nothing, <:Real} = nothing,
                              w_min::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_min_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_opt::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_opt_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_max::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              w_max_ini::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                              ucs_flag::Bool = true,
                              alg::NearOptimalCenteringAlgorithm = UnconstrainedNearOptimalCenteringAlgorithm())
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
        @smart_assert(!any(isa.(r, Ref(SquaredRiskMeasures))))
    else
        @smart_assert(!isa(r, SquaredRiskMeasures))
    end
    if isa(w_min, AbstractVector)
        @smart_assert(!isempty(w_min))
    end
    if isa(w_min_ini, AbstractVector)
        @smart_assert(!isempty(w_min_ini))
    end
    if isa(w_opt, AbstractVector)
        @smart_assert(!isempty(w_opt))
    end
    if isa(w_opt_ini, AbstractVector)
        @smart_assert(!isempty(w_opt_ini))
    end
    if isa(w_max, AbstractVector)
        @smart_assert(!isempty(w_max))
    end
    if isa(w_max_ini, AbstractVector)
        @smart_assert(!isempty(w_max_ini))
    end
    if isa(bins, Real)
        @smart_assert(isfinite(bins) && bins > 0)
    end
    return NearOptimalCentering{typeof(opt), typeof(r), typeof(obj), typeof(bins),
                                typeof(w_min), typeof(w_min_ini), typeof(w_opt),
                                typeof(w_opt_ini), typeof(w_max), typeof(w_max_ini),
                                typeof(ucs_flag), typeof(alg)}(opt, r, obj, bins, w_min,
                                                               w_min_ini, w_opt, w_opt_ini,
                                                               w_max, w_max_ini, ucs_flag,
                                                               alg)
end
function opt_view(noc::NearOptimalCentering, i::AbstractVector, X::AbstractMatrix)
    r = risk_measure_view(noc.r, i, X)
    opt = opt_view(noc.opt, i, X)
    w_min = nothing_scalar_array_view(noc.w_min, i)
    w_min_ini = nothing_scalar_array_view(noc.w_min_ini, i)
    w_opt = nothing_scalar_array_view(noc.w_opt, i)
    w_opt_ini = nothing_scalar_array_view(noc.w_opt_ini, i)
    w_max = nothing_scalar_array_view(noc.w_max, i)
    w_max_ini = nothing_scalar_array_view(noc.w_max_ini, i)
    return NearOptimalCentering(; alg = noc.alg, ucs_flag = noc.ucs_flag, r = r,
                                obj = noc.obj, opt = opt, bins = noc.bins, w_min = w_min,
                                w_min_ini = w_min_ini, w_opt = w_opt, w_opt_ini = w_opt_ini,
                                w_max = w_max, w_max_ini = w_max_ini)
end
function near_optimal_centering_risks(::Any, r::RiskMeasure, pr::AbstractPriorResult,
                                      fees::Union{Nothing, <:Fees},
                                      slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                      w_min::AbstractVector, w_opt::AbstractVector,
                                      w_max::AbstractVector)
    X = pr.X
    r = factory(r, pr, slv)
    scale = r.settings.scale
    risk_min = expected_risk(r, w_min, X, fees) * scale
    risk_opt = expected_risk(r, w_opt, X, fees) * scale
    risk_max = expected_risk(r, w_max, X, fees) * scale
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(::SumScalariser, rs::AbstractVector{<:RiskMeasure},
                                      pr::AbstractPriorResult, fees::Union{Nothing, <:Fees},
                                      slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                      w_min::AbstractVector, w_opt::AbstractVector,
                                      w_max::AbstractVector)
    X = pr.X
    rs = factory(rs, Ref(pr), Ref(slv))
    datatype = eltype(X)
    risk_min = zero(datatype)
    risk_opt = zero(datatype)
    risk_max = zero(datatype)
    for r ∈ rs
        scale = r.settings.scale
        risk_min += expected_risk(r, w_min, X, fees) * scale
        risk_opt += expected_risk(r, w_opt, X, fees) * scale
        risk_max += expected_risk(r, w_max, X, fees) * scale
    end
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(scalarisation::LogSumExpScalariser,
                                      rs::AbstractVector{<:RiskMeasure},
                                      pr::AbstractPriorResult, fees::Union{Nothing, <:Fees},
                                      slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                      w_min::AbstractVector, w_opt::AbstractVector,
                                      w_max::AbstractVector)
    X = pr.X
    rs = factory(rs, Ref(pr), Ref(slv))
    datatype = eltype(X)
    risk_min = zero(datatype)
    risk_opt = zero(datatype)
    risk_max = zero(datatype)
    gamma = scalarisation.gamma
    for r ∈ rs
        scale = r.settings.scale * gamma
        risk_min += exp(expected_risk(r, w_min, X, fees) * scale)
        risk_opt += exp(expected_risk(r, w_opt, X, fees) * scale)
        risk_max += exp(expected_risk(r, w_max, X, fees) * scale)
    end
    igamma = inv(gamma)
    risk_min = log(risk_min) * igamma
    risk_opt = log(risk_opt) * igamma
    risk_max = log(risk_max) * igamma
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(::MaxScalariser, rs::AbstractVector{<:RiskMeasure},
                                      pr::AbstractPriorResult, fees::Union{Nothing, <:Fees},
                                      slv::Union{Nothing, <:Solver,
                                                 <:AbstractVector{<:Solver}},
                                      w_min::AbstractVector, w_opt::AbstractVector,
                                      w_max::AbstractVector)
    X = pr.X
    rs = factory(rs, Ref(pr), Ref(slv))
    datatype = eltype(X)
    risk_min = typemin(datatype)
    risk_opt = typemin(datatype)
    risk_max = typemin(datatype)
    for r ∈ rs
        scale = r.settings.scale
        risk_min_i = expected_risk(r, w_min, X, fees) * scale
        risk_opt_i = expected_risk(r, w_opt, X, fees) * scale
        risk_max_i = expected_risk(r, w_max, X, fees) * scale
        if risk_min_i >= risk_min
            risk_min = risk_min_i
        end
        if risk_opt_i >= risk_opt
            risk_opt = risk_opt_i
        end
        if risk_max_i >= risk_max
            risk_max = risk_max_i
        end
    end
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult;
                                      dims::Int = 1)
    w_min = noc.w_min
    w_opt = noc.w_opt
    w_max = noc.w_max
    w_min_flag = isnothing(w_min)
    w_opt_flag = isnothing(w_opt)
    w_max_flag = isnothing(w_max)
    unconstrained_flag = isa(noc.alg, UnconstrainedNearOptimalCenteringAlgorithm)
    r = noc.r
    opt = processed_jump_optimiser(noc.opt, rd; dims = dims)
    if w_min_flag || w_max_flag || unconstrained_flag
        nb_r = no_bounds_risk_measure(r, noc.ucs_flag)
        nb_opt = no_bounds_optimiser(opt, noc.ucs_flag)
    end
    if w_min_flag
        res_min = optimise!(MeanRisk(; r = nb_r, obj = MinimumRisk(), opt = nb_opt,
                                     wi = noc.w_min_ini), rd; save = false)
        @smart_assert(isa(res_min.retcode, OptimisationSuccess))
        w_min = res_min.w
    end
    if w_opt_flag
        res_opt = optimise!(MeanRisk(; r = r, obj = noc.obj, opt = opt, wi = noc.w_opt_ini),
                            rd; save = false)
        @smart_assert(isa(res_opt.retcode, OptimisationSuccess))
        w_opt = res_opt.w
    end
    if w_max_flag
        res_max = optimise!(MeanRisk(; r = nb_r, obj = MaximumReturn(), opt = nb_opt,
                                     wi = noc.w_max_ini), rd; save = false)
        @smart_assert(isa(res_max.retcode, OptimisationSuccess))
        w_max = res_max.w
    end
    pr = opt.pe
    fees = opt.fees
    ret = opt.ret
    rk_min, rk_opt, rk_max = near_optimal_centering_risks(opt.sce, r, pr, fees, opt.slv,
                                                          w_min, w_opt, w_max)
    rt_min = expected_returns(ret, w_min, pr, fees)
    rt_opt = expected_returns(ret, w_opt, pr, fees)
    rt_max = expected_returns(ret, w_max, pr, fees)
    ibins = if isnothing(noc.bins)
        T, N = size(pr.X)
        N / T
    else
        inv(noc.bins)
    end
    rk_delta = (rk_max - rk_min) * ibins
    rt_delta = (rt_max - rt_min) * ibins
    rk_opt += rk_delta
    rt_opt -= rt_delta
    if unconstrained_flag
        r, opt = nb_r, nb_opt
    end
    return w_opt, rk_opt, rt_opt, r, opt
end
function set_near_optimal_centering_constraints!(model::JuMP.Model, rk::Real, rt::Real,
                                                 wb::WeightBoundsResult)
    w = model[:w]
    sc = model[:sc]
    w_ub = wb.ub
    risk = model[:risk]
    ret = model[:ret]
    N = length(w)
    @variables(model, begin
                   log_ret
                   log_risk
                   log_w[1:N]
                   log_delta_w[1:N]
               end)
    @constraints(model,
                 begin
                     clog_risk,
                     [sc * log_risk, sc, sc * (rk - risk)] ∈ MOI.ExponentialCone()
                     clog_ret, [sc * log_ret, sc, sc * (ret - rt)] ∈ MOI.ExponentialCone()
                     clog_w[i = 1:N],
                     [sc * log_w[i], sc, sc * w[i]] ∈ MOI.ExponentialCone()
                     clog_delta_w[i = 1:N],
                     [sc * log_delta_w[i], sc, sc * (w_ub[i] - w[i])] ∈
                     MOI.ExponentialCone()
                 end)
    @expression(model, obj_expr, -(log_ret + log_risk + sum(log_w + log_delta_w)))
    return obj_expr
end
function set_unconstrainted_near_optimal_objective_function!(model::JuMP.Model, rk::Real,
                                                             rt::Real,
                                                             wb::WeightBoundsResult)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, wb)
    @objective(model, Min, so * obj_expr)
    return nothing
end
function set_constrained_near_optimal_objective_function!(model::JuMP.Model, rk::Real,
                                                          rt::Real, wb::WeightBoundsResult,
                                                          pret::JuMPReturnsEstimator,
                                                          cobj::Union{Nothing,
                                                                      <:CustomObjective},
                                                          opt::BaseJuMPOptimisationEstimator,
                                                          pr::AbstractPriorResult)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, wb)
    add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, pret, cobj, obj_expr, opt, pr)
    @objective(model, Min, so * obj_expr)
    return nothing
end
function optimise!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:UnconstrainedNearOptimalCenteringAlgorithm},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w_opt, rk_opt, rt_opt, nb_r, nb_opt = near_optimal_centering_setup(noc, rd; dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, nb_opt.sc, nb_opt.so)
    @expression(model, k, 1)
    set_w!(model, nb_opt.pe.X, w_opt)
    set_weight_constraints!(model, nb_opt.wb, nb_opt.bgt, nothing, true)
    set_risk_constraints!(model, nb_r, noc, nb_opt.pe, nothing, nothing)
    scalarise_risk_expression!(model, nb_opt.sce)
    ret = jump_returns_factory(nb_opt.ret, nb_opt.pe)
    set_return_constraints!(model, ret, MinimumRisk(), nb_opt.pe)
    set_unconstrainted_near_optimal_objective_function!(model, rk_opt, rt_opt, nb_opt.wb)
    retcode, sol = optimise_JuMP_model!(model, noc, eltype(nb_opt.pe.X))
    return JuMPOptimisationResult(typeof(noc), nb_opt.pe, nb_opt.wb, nb_opt.lcs,
                                  nb_opt.cent, nb_opt.gcard, nb_opt.sgcard, nb_opt.smtx,
                                  nb_opt.nplg, nb_opt.cplg, retcode, sol,
                                  ifelse(save, model, nothing))
end
function optimise!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:ConstrainedNearOptimalCenteringAlgorithm},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w_opt, rk_opt, rt_opt, r, opt = near_optimal_centering_setup(noc, rd; dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    @expression(model, k, 1)
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, nothing, true)
    set_linear_weight_constraints!(model, opt.lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, opt.cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, opt.wb, opt.card, opt.gcard, opt.nplg, opt.cplg, opt.lt,
                         opt.st, opt.fees, opt.ss)
    set_smip_constraints!(model, opt.wb, opt.scard, opt.sgcard, opt.smtx)
    set_turnover_constraints!(model, opt.tn)
    set_tracking_error_constraints!(model, opt.pe, opt.te)
    set_number_effective_assets!(model, opt.nea)
    set_l1_regularisation!(model, opt.l1)
    set_l2_regularisation!(model, opt.l2)
    set_non_fixed_fees!(model, opt.fees)
    set_risk_constraints!(model, r, noc, opt.pe, opt.nplg, opt.cplg)
    scalarise_risk_expression!(model, opt.sce)
    ret = jump_returns_factory(opt.ret, opt.pe)
    set_return_constraints!(model, ret, MinimumRisk(), opt.pe)
    set_sdp_philogeny_constraints!(model, opt.nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, opt.cplg, :sdp_cplg)
    add_custom_constraint!(model, opt.ccnt, opt, opt.pe)
    set_constrained_near_optimal_objective_function!(model, rk_opt, rt_opt, opt.wb, ret,
                                                     opt.cobj, opt, opt.pe)
    retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
    return JuMPOptimisationResult(typeof(noc), opt.pe, opt.wb, opt.lcs, opt.cent, opt.gcard,
                                  opt.sgcard, opt.smtx, opt.nplg, opt.cplg, retcode, sol,
                                  ifelse(save, model, nothing))
end

export ConstrainedNearOptimalCenteringAlgorithm, UnconstrainedNearOptimalCenteringAlgorithm,
       NearOptimalCentering
