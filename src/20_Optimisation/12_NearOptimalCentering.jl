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
                              w_opt::Union{Nothing, <:AbstractVector} = nothing,
                              w_opt_ini::Union{Nothing, <:AbstractVector} = nothing,
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
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    risk_min = zero(datatype)
    flag = !isa(w_opt, AbstractVector{<:AbstractVector})
    risk_opt = flag ? zero(datatype) : zeros(datatype, length(w_opt))
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
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    risk_min = zero(datatype)
    flag = !isa(w_opt, AbstractVector{<:AbstractVector})
    risk_opt = flag ? zero(datatype) : zeros(datatype, length(w_opt))
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
    risk_opt = (flag ? log(risk_opt) : log.(risk_opt)) * igamma
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
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    risk_min = typemin(datatype)
    flag = !isa(w_opt, AbstractVector{<:AbstractVector})
    risk_opt = flag ? zero(datatype) : zeros(datatype, length(w_opt))
    risk_max = typemin(datatype)
    for r ∈ rs
        scale = r.settings.scale
        risk_min_i = expected_risk(r, w_min, X, fees) * scale
        risk_opt_i = expected_risk(r, w_opt, X, fees) * scale
        risk_max_i = expected_risk(r, w_max, X, fees) * scale
        if risk_min_i >= risk_min
            risk_min = risk_min_i
        end
        if flag
            if risk_opt_i >= risk_opt
                risk_opt = risk_opt_i
            end
        else
            idx = risk_opt_i .>= risk_opt
            risk_opt[idx] = risk_opt_i[idx]
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
    unconstrained = isa(noc.alg, UnconstrainedNearOptimalCenteringAlgorithm)
    r = noc.r
    opt = processed_jump_optimiser(noc.opt, rd; dims = dims)
    if w_min_flag || w_max_flag || unconstrained
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
        if !isa(res_opt.retcode, AbstractVector)
            @smart_assert(isa(res_opt.retcode, OptimisationSuccess))
        else
            @smart_assert(all(isa.(res_opt.retcode, Ref(OptimisationSuccess))))
        end
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
    rk_opt = rk_opt ⊕ rk_delta
    rt_opt = rt_opt ⊖ rt_delta
    if unconstrained
        r, opt = nb_r, nb_opt
    end
    return w_opt, rk_opt, rt_opt, r, opt, rt_min, rt_max, w_min, w_max
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
function set_near_optimal_objective_function!(::UnconstrainedNearOptimalCenteringAlgorithm,
                                              model::JuMP.Model, rk::Real, rt::Real,
                                              opt::BaseJuMPOptimisationEstimator)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, opt.wb)
    @objective(model, Min, so * obj_expr)
    return nothing
end
function set_near_optimal_objective_function!(::ConstrainedNearOptimalCenteringAlgorithm,
                                              model::JuMP.Model, rk::Real, rt::Real,
                                              opt::BaseJuMPOptimisationEstimator)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, opt.wb)
    add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, opt.ret, opt.cobj, obj_expr, opt, opt.pe)
    @objective(model, Min, so * obj_expr)
    return nothing
end
function unregister_noc_variables!(model::JuMP.Model)
    if !haskey(model, :log_ret)
        return nothing
    end
    delete(model, model[:log_ret])
    unregister(model, :log_ret)
    delete(model, model[:log_risk])
    unregister(model, :log_risk)
    delete(model, model[:log_w])
    unregister(model, :log_w)
    delete(model, model[:log_delta_w])
    unregister(model, :log_delta_w)
    delete(model, model[:clog_risk])
    unregister(model, :clog_risk)
    delete(model, model[:clog_ret])
    unregister(model, :clog_ret)
    delete(model, model[:clog_w])
    unregister(model, :clog_w)
    delete(model, model[:clog_delta_w])
    unregister(model, :clog_delta_w)
    unregister(model, :obj_expr)
    return nothing
end
function solve_noc!(noc::NearOptimalCentering, model::JuMP.Model, rk_opt::Real,
                    rt_opt::Real, opt::BaseJuMPOptimisationEstimator, args...)
    set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
    return optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:UnconstrainedNearOptimalCenteringAlgorithm},
                    model::JuMP.Model, rk_opts::AbstractVector{<:Real},
                    rt_opts::AbstractVector{<:Real}, opt::BaseJuMPOptimisationEstimator)
    retcodes = sizehint!(Vector{OptimisationReturnCode}(undef, 0), length(rk_opts))
    sols = sizehint!(Vector{JuMPOptimisationSolution}(undef, 0), length(rk_opts))
    for (rk_opt, rt_opt) ∈ zip(rk_opts, rt_opts)
        unregister_noc_variables!(model)
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function compute_ret_lbs(lbs::Frontier, rt_min::Real, rt_max::Real)
    return range(; start = rt_min, stop = rt_max, length = lbs.N)
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCenteringAlgorithm},
                    model::JuMP.Model, rk_opts::AbstractVector{<:Real},
                    rt_opts::AbstractVector{<:Real}, opt::BaseJuMPOptimisationEstimator,
                    rt_min::Real, rt_max::Real, ::Any, ::Any, ::Val{true}, ::Val{false})
    lbs = compute_ret_lbs(model[:ret_frontier], rt_min, rt_max)
    sc = model[:sc]
    ret_expr = model[:ret]
    retcodes = Vector{OptimisationReturnCode}(undef, length(rk_opts))
    sols = Vector{JuMPOptimisationSolution}(undef, length(rk_opts))
    for (i, (rk_opt, rt_opt, lb)) ∈ enumerate(zip(rk_opts, rt_opts, lbs))
        if i != 1
            delete(model, model[:ret_lb])
            unregister(model, :ret_lb)
            unregister(model, :obj_expr)
        end
        @constraint(model, ret_lb, sc * (ret_expr - lb) >= 0)
        unregister_noc_variables!(model)
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
        retcodes[i] = retcode
        sols[i] = sol
    end
    return retcodes, sols
end
function compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering,
                          pr::AbstractPriorResult, w_min::AbstractVector,
                          w_max::AbstractVector)
    risk_frontier = model[:risk_frontier]
    idx = Vector{Int}(undef, 0)
    for (i, rkf) ∈ enumerate(risk_frontier)
        if !isa(rkf.second[2], AbstractVector)
            push!(idx, i)
        end
    end
    if isempty(idx)
        return risk_frontier
    end
    #! Compute risk upper bounds for the risk frontier using w_min and w_max
    return nothing
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCenteringAlgorithm},
                    model::JuMP.Model, rk_opts::AbstractVector{<:Real},
                    rt_opts::AbstractVector{<:Real}, opt::BaseJuMPOptimisationEstimator,
                    ::Any, ::Any, w_min::AbstractVector, w_max::AbstractVector,
                    ::Val{false}, ::Val{true})
    risk_frontier = compute_risk_ubs(model, noc, opt.pe, w_min, w_max)
    itrs = [(Iterators.repeated(rkf[1], length(rkf[2][2])),
             Iterators.repeated(rkf[2][1], length(rkf[2][2])), rkf[2][2])
            for rkf ∈ risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(Vector{OptimisationReturnCode}(undef, 0), length(rk_opts))
    sols = sizehint!(Vector{JuMPOptimisationSolution}(undef, 0), length(rk_opts))
    sc = model[:sc]
    for (keys, r_exprs, ubs, rk_opt, rt_opt) ∈
        zip(pitrs[1], pitrs[2], pitrs[3], rk_opts, rt_opts)
        unregister_noc_variables!(model)
        for (key, r_expr, ub) ∈ zip(keys, r_exprs, ubs)
            if haskey(model, key)
                delete(model, model[key])
                unregister(model, key)
            end
            model[key] = @constraint(model, sc * (r_expr - ub) <= 0)
        end
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function optimise!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:UnconstrainedNearOptimalCenteringAlgorithm},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w_opt, rk_opt, rt_opt, r, opt = near_optimal_centering_setup(noc, rd; dims = dims)[1:5]
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    @expression(model, k, 1)
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    set_risk_constraints!(model, r, noc, opt.pe, nothing, nothing)
    scalarise_risk_expression!(model, opt.sce)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pe)
    # set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
    # retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
    retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt)
    return JuMPOptimisationResult(typeof(noc), opt.pe, opt.wb, opt.lcs, opt.cent, opt.gcard,
                                  opt.sgcard, opt.smtx, opt.nplg, opt.cplg, opt.ret,
                                  retcode, sol, ifelse(save, model, nothing))
end
function optimise!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:ConstrainedNearOptimalCenteringAlgorithm},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w_opt, rk_opt, rt_opt, r, opt, rt_min, rt_max, w_min, w_max = near_optimal_centering_setup(noc,
                                                                                               rd;
                                                                                               dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    @expression(model, k, 1)
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    set_linear_weight_constraints!(model, opt.lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, opt.cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, opt.wb, opt.card, opt.gcard, opt.nplg, opt.cplg, opt.lt,
                         opt.st, opt.fees, opt.ss)
    set_smip_constraints!(model, opt.wb, opt.scard, opt.sgcard, opt.smtx, opt.ss)
    set_turnover_constraints!(model, opt.tn)
    set_tracking_error_constraints!(model, opt.pe, opt.te, noc, opt.nplg, opt.cplg)
    set_number_effective_assets!(model, opt.nea)
    set_l1_regularisation!(model, opt.l1)
    set_l2_regularisation!(model, opt.l2)
    set_non_fixed_fees!(model, opt.fees)
    set_risk_constraints!(model, r, noc, opt.pe, opt.nplg, opt.cplg)
    scalarise_risk_expression!(model, opt.sce)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pe)
    set_sdp_philogeny_constraints!(model, opt.nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, opt.cplg, :sdp_cplg)
    add_custom_constraint!(model, opt.ccnt, opt, opt.pe)
    # set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
    # retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
    retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt, rt_min, rt_max, w_min, w_max,
                              Val(haskey(model, :ret_frontier)),
                              Val(haskey(model, :risk_frontier)))
    return JuMPOptimisationResult(typeof(noc), opt.pe, opt.wb, opt.lcs, opt.cent, opt.gcard,
                                  opt.sgcard, opt.smtx, opt.nplg, opt.cplg, opt.ret,
                                  retcode, sol, ifelse(save, model, nothing))
end

function near_opt_centering_resolve(model::JuMP.Model, noc::NearOptimalCentering,
                                    w::AbstractVector, rd::ReturnsResult, dims::Int)
    noc.w_opt .= w
    try
        set_start_value.(model[:w], w)
    catch
    end
    #! This can be made more efficient by not recomputing the minimum and maximum risks. But means lifting the computation of rk_opt and rt_opt into a separate function that takes rk_delta and rt_delta. It means creating a new near_optimal_centering_setup function that returns rk_delta, rt_delta and opt which is called before the first invoaction of near_opt_centering_resolve, they would also be outputed by efficient_frontier_bounds.
    rk_opt, rt_opt, opt = near_optimal_centering_setup(noc, rd; dims = dims)[[2, 3, 5]]
    unregister_noc_variables!(model)
    set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
    return optimise_JuMP_model!(model, noc, eltype(w))
end
function efficient_frontier_bounds(noc::NearOptimalCentering, rd::ReturnsResult,
                                   w_max::AbstractVector{<:Real}, dims; kwargs...)
    res = optimise!(noc, rd; dims = dims, kwargs...)
    if isa(res.retcode, OptimisationFailure)
        throw(OptimisationFailure("Failed to compute minimum risk for efficient frontier."))
    end
    model = res.model
    retcode, sol = near_opt_centering_resolve(model, noc, w_max, rd, dims)
    if isa(retcode, OptimisationFailure)
        throw(OptimisationFailure("Failed to compute maximum return for efficient frontier."))
    end
    return res, sol.w
end
function efficient_frontier!(noc::NearOptimalCentering, rd::ReturnsResult = ReturnsResult();
                             N::Integer = 20,
                             wi_min::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                             wi_max::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                             risk_ub::Real = inv(sqrt(eps())),
                             ret_lb::Real = -inv(sqrt(eps())),
                             rf::Union{Nothing, <:Real} = 0.0,
                             ohf::Union{Nothing, <:Real} = nothing, dims::Int = 1,
                             save::Bool = true, kwargs...)
    opt = JuMPOptimiser(; pe = noc.opt.pe, slv = noc.opt.slv, wb = noc.opt.wb,
                        bgt = noc.opt.bgt, sbgt = noc.opt.sbgt, lt = noc.opt.lt,
                        st = noc.opt.st, lcs = noc.opt.lcs, lcm = noc.opt.lcm,
                        cent = noc.opt.cent, gcard = noc.opt.gcard, sgcard = noc.opt.sgcard,
                        smtx = noc.opt.smtx, sets = noc.opt.sets, nplg = noc.opt.nplg,
                        cplg = noc.opt.cplg, tn = noc.opt.tn, te = noc.opt.te,
                        fees = noc.opt.fees,
                        ret = bounds_returns_estimator(noc.opt.ret, ret_lb),
                        sce = noc.opt.sce, ccnt = noc.opt.ccnt, cobj = noc.opt.cobj,
                        sc = noc.opt.sc, so = noc.opt.so, card = noc.opt.card,
                        scard = noc.opt.scard, nea = noc.opt.nea, l1 = noc.opt.l1,
                        l2 = noc.opt.l2, ss = noc.opt.ss, strict = noc.opt.strict)
    mr = MeanRisk(; opt = opt, r = bounds_first_risk_measure(noc.r, risk_ub),
                  obj = MinimumRisk(), wi = wi_min)
    res, w2 = efficient_frontier_bounds(mr, rd, wi_min, wi_max, dims; kwargs...)
    w1 = res.w
    r = factory(mr.r, res.pr, mr.opt.slv)
    ri = isa(r, AbstractVector) ? r[1] : r
    risk1 = expected_risk(ri, w1, res.pr.X, mr.opt.fees; kwargs...)
    risk2 = expected_risk(ri, w2, res.pr.X, mr.opt.fees; kwargs...)
    ret1 = expected_returns(res.ret, w1, res.pr, mr.opt.fees)
    ret2 = expected_returns(res.ret, w2, res.pr, mr.opt.fees)
    mr_model = res.model
    ks = string.(keys(object_dictionary(mr_model)))
    rkey = Symbol(ks[findfirst(x -> contains(x, r".*_risk_1$"), ks)])
    if isa(mr_model[rkey], QuadExpr)
        risk1 = sqrt(risk1)
        risk2 = sqrt(risk2)
    end
    key = Symbol(ks[findfirst(x -> contains(x, r".*_1_ub$"), ks)])
    risks = range(risk1; stop = risk2, length = N)
    rets = range(ret1; stop = ret2, length = N)
    M = length(w1)
    ws = Vector{eltype(w1)}(undef, (N + (!isnothing(rf) ? 1 : 0)) * M)
    opt = JuMPOptimiser(; pe = res.pr, slv = noc.opt.slv, wb = res.wb, bgt = noc.opt.bgt,
                        sbgt = noc.opt.sbgt, lt = noc.opt.lt, st = noc.opt.st,
                        lcs = res.lcs, lcm = noc.opt.lcm, cent = res.cent,
                        gcard = res.gcard, sgcard = res.sgcard, smtx = res.smtx,
                        sets = noc.opt.sets, nplg = res.nplg, cplg = res.cplg,
                        tn = noc.opt.tn, te = noc.opt.te, fees = noc.opt.fees,
                        ret = res.ret, sce = noc.opt.sce, ccnt = noc.opt.ccnt,
                        cobj = noc.opt.cobj, sc = noc.opt.sc, so = noc.opt.so,
                        card = noc.opt.card, scard = noc.opt.scard, nea = noc.opt.nea,
                        l1 = noc.opt.l1, l2 = noc.opt.l2, ss = noc.opt.ss,
                        strict = noc.opt.strict)
    noc = NearOptimalCentering(; opt = opt, r = no_bounds_risk_measure(r, noc.ucs_flag),
                               bins = noc.bins, w_min = copy(w1), w_opt = w1,
                               w_opt_ini = w1, w_max = copy(w2), ucs_flag = noc.ucs_flag,
                               alg = noc.alg)
    res, w2 = efficient_frontier_bounds(noc, rd, w2, dims)
    noc_model = res.model
    ws[1:M] .= res.w
    ws[((N - 1) * M + 1):(N * M)] .= w2
    unregister(mr_model, :obj_expr)
    set_portfolio_objective_function!(mr_model, MaximumReturn(), res.ret, mr.opt.cobj, mr,
                                      res.pr)
    failure = OptimisationFailure[]
    for i ∈ 2:length(risks)
        idx = ((i - 1) * M + 1):(i * M)
        set_normalized_rhs(mr_model[key], risks[i])
        mr_retcode, sol = optimise_JuMP_model!(mr_model, mr, eltype(w1))
        if isa(mr_retcode, OptimisationSuccess)
            noc_retcode, sol = near_opt_centering_resolve(noc_model, noc, sol.w, rd, dims)
            if isa(noc_retcode, OptimisationSuccess)
                ws[idx] .= sol.w
                continue
            end
        end
        set_normalized_rhs(mr_model[key], risk_ub)
        set_normalized_rhs(mr_model[:ret_lb], rets[i])
        unregister(mr_model, :obj_expr)
        set_portfolio_objective_function!(mr_model, MinimumRisk(), res.ret, mr.opt.cobj, mr,
                                          res.pr)
        mr_retcode, sol = optimise_JuMP_model!(mr_model, mr, eltype(res.w))
        set_normalized_rhs(mr_model[:ret_lb], ret_lb)
        if isa(mr_retcode, OptimisationSuccess)
            noc_retcode, sol = near_opt_centering_resolve(noc_model, noc, sol.w, rd, dims)
            if isa(noc_retcode, OptimisationSuccess)
                ws[idx] .= sol.w
            else
                push!(failure, noc_retcode)
                ws[idx] .= NaN
            end
        else
            push!(failure, mr_retcode)
            ws[idx] .= NaN
        end
    end
    if !isnothing(rf)
        delete(mr_model, mr_model[key])
        unregister(mr_model, key)
        delete(mr_model, mr_model[:ret_lb])
        unregister(mr_model, :ret_lb)
        unregister(mr_model, :obj_expr)
        set_portfolio_objective_function!(mr_model, MaximumRatio(; rf = rf, ohf = ohf),
                                          res.ret, mr.opt.cobj, mr, res.pr)
        retcode, sol = optimise_JuMP_model!(mr_model, mr, eltype(res.w))
        if isa(retcode, OptimisationSuccess)
            noc_retcode, sol = near_opt_centering_resolve(noc_model, noc, sol.w, rd, dims)
            if isa(noc_retcode, OptimisationSuccess)
                ws[(N * M + 1):end] .= sol.w
            else
                push!(failure, noc_retcode)
                ws[(N * M + 1):end] .= NaN
            end
        else
            push!(failure, retcode)
            ws[(N * M + 1):end] .= NaN
        end
    end
    return JuMPOptimisationResult(typeof(mr), res.pr, res.wb, res.lcs, res.cent, res.gcard,
                                  res.sgcard, res.smtx, res.nplg, res.cplg, res.ret,
                                  if isempty(failure)
                                      OptimisationSuccess(; res = nothing)
                                  else
                                      failure
                                  end, JuMPOptimisationSolution(; w = reshape(ws, M, :)),
                                  ifelse(save, noc_model, nothing))
end

export ConstrainedNearOptimalCenteringAlgorithm, UnconstrainedNearOptimalCenteringAlgorithm,
       NearOptimalCentering
