abstract type NearOptimalCenteringAlgorithm <: OptimisationAlgorithm end
struct ConstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
struct UnconstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
struct NearOptimalCenteringResult{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10} <:
       OptimisationResult
    oe::T1
    pa::T2
    w_min_retcode::T3
    w_opt_retcode::T4
    w_max_retcode::T5
    noc_retcode::T6
    retcode::T7
    sol::T8
    model::T9
    fb::T10
end
function factory(res::NearOptimalCenteringResult, fb)
    return NearOptimalCenteringResult(res.oe, res.pa, res.w_min_retcode, res.w_opt_retcode,
                                      res.w_max_retcode, res.noc_retcode, res.retcode,
                                      res.sol, res.model, fb)
end
function Base.getproperty(r::NearOptimalCenteringResult, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? r.sol.w : getproperty.(r.sol, :w)
    elseif sym in propertynames(r)
        getfield(r, sym)
    elseif sym in propertynames(r.pa)
        getproperty(r.pa, sym)
    end
end
struct NearOptimalCentering{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13} <:
       RiskJuMPOptimisationEstimator
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
    fb::T13
    function NearOptimalCentering(opt::JuMPOptimiser, r::RM_VecRM,
                                  obj::Option{<:ObjectiveFunction}, bins::Option{<:Number},
                                  w_min::Option{<:VecNum}, w_min_ini::Option{<:VecNum},
                                  w_opt::Option{<:VecNum_VecVecNum},
                                  w_opt_ini::Option{<:VecNum_VecVecNum},
                                  w_max::Option{<:VecNum}, w_max_ini::Option{<:VecNum},
                                  ucs_flag::Bool, alg::NearOptimalCenteringAlgorithm,
                                  fb::Option{<:OptimisationEstimator})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
            if any(x -> isa(x, QuadExpressionRiskMeasures), r)
                @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
            end
        else
            if isa(r, QuadExpressionRiskMeasures)
                @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
            end
        end
        if !isnothing(w_min)
            @argcheck(!isempty(w_min))
        end
        if !isnothing(w_min_ini)
            @argcheck(!isempty(w_min_ini))
        end
        if !isnothing(w_opt)
            @argcheck(!isempty(w_opt))
        end
        if !isnothing(w_opt)
            @argcheck(!isempty(w_opt_ini))
        end
        if !isnothing(w_max)
            @argcheck(!isempty(w_max))
        end
        if !isnothing(w_max_ini)
            @argcheck(!isempty(w_max_ini))
        end
        if isa(bins, Number)
            @argcheck(isfinite(bins) && bins > 0)
        end
        return new{typeof(opt), typeof(r), typeof(obj), typeof(bins), typeof(w_min),
                   typeof(w_min_ini), typeof(w_opt), typeof(w_opt_ini), typeof(w_max),
                   typeof(w_max_ini), typeof(ucs_flag), typeof(alg), typeof(fb)}(opt, r,
                                                                                 obj, bins,
                                                                                 w_min,
                                                                                 w_min_ini,
                                                                                 w_opt,
                                                                                 w_opt_ini,
                                                                                 w_max,
                                                                                 w_max_ini,
                                                                                 ucs_flag,
                                                                                 alg, fb)
    end
end
function NearOptimalCentering(; opt::JuMPOptimiser = JuMPOptimiser(),
                              r::RM_VecRM = StandardDeviation(),
                              obj::Option{<:ObjectiveFunction} = MinimumRisk(),
                              bins::Option{<:Number} = nothing,
                              w_min::Option{<:VecNum} = nothing,
                              w_min_ini::Option{<:VecNum} = nothing,
                              w_opt::Option{<:VecNum_VecVecNum} = nothing,
                              w_opt_ini::Option{<:VecNum_VecVecNum} = nothing,
                              w_max::Option{<:VecNum} = nothing,
                              w_max_ini::Option{<:VecNum} = nothing, ucs_flag::Bool = true,
                              alg::NearOptimalCenteringAlgorithm = UnconstrainedNearOptimalCentering(),
                              fb::Option{<:OptimisationEstimator} = nothing)
    return NearOptimalCentering(opt, r, obj, bins, w_min, w_min_ini, w_opt, w_opt_ini,
                                w_max, w_max_ini, ucs_flag, alg, fb)
end
function opt_view(noc::NearOptimalCentering, i, X::MatNum)
    X = isa(noc.opt.pr, AbstractPriorResult) ? noc.opt.pr.X : X
    opt = opt_view(noc.opt, i, X)
    r = risk_measure_view(noc.r, i, X)
    w_min = nothing_scalar_array_view(noc.w_min, i)
    w_min_ini = nothing_scalar_array_view(noc.w_min_ini, i)
    w_opt = nothing_scalar_array_view(noc.w_opt, i)
    w_opt_ini = nothing_scalar_array_view(noc.w_opt_ini, i)
    w_max = nothing_scalar_array_view(noc.w_max, i)
    w_max_ini = nothing_scalar_array_view(noc.w_max_ini, i)
    return NearOptimalCentering(; alg = noc.alg, ucs_flag = noc.ucs_flag, r = r,
                                obj = noc.obj, opt = opt, bins = noc.bins, w_min = w_min,
                                w_min_ini = w_min_ini, w_opt = w_opt, w_opt_ini = w_opt_ini,
                                w_max = w_max, w_max_ini = w_max_ini, fb = noc.fb)
end
function near_optimal_centering_risks(::Any, r::RiskMeasure, pr::AbstractPriorResult,
                                      fees::Option{<:Fees}, slv::Slv_VecSlv, w_min::VecNum,
                                      w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    r = factory(r, pr, slv)
    scale = r.settings.scale
    risk_min = expected_risk(r, w_min, X, fees) * scale
    risk_opt = expected_risk(r, w_opt, X, fees) * scale
    risk_max = expected_risk(r, w_max, X, fees) * scale
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(::SumScalariser, rs::VecRM, pr::AbstractPriorResult,
                                      fees::Option{<:Fees}, slv::Slv_VecSlv, w_min::VecNum,
                                      w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    risk_min = zero(datatype)
    flag = isa(w_opt, VecNum)
    risk_opt = flag ? zero(datatype) : zeros(datatype, length(w_opt))
    risk_max = zero(datatype)
    for r in rs
        scale = r.settings.scale
        risk_min += expected_risk(r, w_min, X, fees) * scale
        risk_opt += expected_risk(r, w_opt, X, fees) * scale
        risk_max += expected_risk(r, w_max, X, fees) * scale
    end
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(scalarisation::LogSumExpScalariser, rs::VecRM,
                                      pr::AbstractPriorResult, fees::Option{<:Fees},
                                      slv::Slv_VecSlv, w_min::VecNum,
                                      w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    N = length(rs)
    risk_min = zeros(datatype, N)
    flag = isa(w_opt, VecNum)
    risk_opt = if flag
        zeros(datatype, N)
    else
        zeros(datatype, N, length(w_opt))
    end
    risk_max = zeros(datatype, N)
    gamma = scalarisation.gamma
    for (i, r) in enumerate(rs)
        scale = r.settings.scale * gamma
        risk_min[i] = expected_risk(r, w_min, X, fees) * scale
        tmp = expected_risk(r, w_opt, X, fees) * scale
        if flag
            risk_opt[i] = tmp
        else
            risk_opt[i, :] .= tmp
        end
        risk_max[i] = expected_risk(r, w_max, X, fees) * scale
    end
    igamma = inv(gamma)
    risk_min = LogExpFunctions.logsumexp(risk_min) * igamma
    risk_opt = if flag
        LogExpFunctions.logsumexp(risk_opt) * igamma
    else
        vec(LogExpFunctions.logsumexp(risk_opt; dims = 1)) * igamma
    end
    risk_max = LogExpFunctions.logsumexp(risk_max) * igamma
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(::MaxScalariser, rs::VecRM, pr::AbstractPriorResult,
                                      fees::Option{<:Fees}, slv::Option{<:Slv_VecSlv},
                                      w_min::VecNum, w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    rs = factory(rs, pr, slv)
    datatype = eltype(X)
    risk_min = typemin(datatype)
    flag = isa(w_opt, VecNum)
    risk_opt = flag ? zero(datatype) : zeros(datatype, length(w_opt))
    risk_max = typemin(datatype)
    for r in rs
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
            risk_opt[idx] = view(risk_opt_i, idx)
        end
        if risk_max_i >= risk_max
            risk_max = risk_max_i
        end
    end
    return risk_min, risk_opt, risk_max
end
struct NearOptimalSetup{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12}
    w_opt::T1
    rk_opt::T2
    rt_opt::T3
    rt_min::T4
    rt_max::T5
    w_min::T6
    w_max::T7
    r::T8
    opt::T9
    w_min_retcode::T10
    w_opt_retcode::T11
    w_max_retcode::T12
end
function near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult;
                                      dims::Int = 1)
    w_min = noc.w_min
    w_opt = noc.w_opt
    w_max = noc.w_max
    w_min_flag = isnothing(w_min)
    w_opt_flag = isnothing(w_opt)
    w_max_flag = isnothing(w_max)
    w_min_retcode = OptimisationSuccess(nothing)
    w_opt_retcode = OptimisationSuccess(nothing)
    w_max_retcode = OptimisationSuccess(nothing)
    unconstrained = isa(noc.alg, UnconstrainedNearOptimalCentering)
    r = noc.r
    opt = processed_jump_optimiser(noc.opt, rd; dims = dims)
    if w_min_flag || w_max_flag || unconstrained
        nb_r = no_bounds_risk_measure(r, Val(noc.ucs_flag))
        nb_opt = no_bounds_optimiser(opt, noc.ucs_flag)
    end
    if w_min_flag
        res_min = optimise(MeanRisk(; r = nb_r, obj = MinimumRisk(), opt = nb_opt,
                                    wi = noc.w_min_ini), rd; save = false)
        w_min_retcode = res_min.retcode
        w_min = res_min.w
    end
    if w_opt_flag
        res_opt = optimise(MeanRisk(; r = r, obj = noc.obj, opt = opt, wi = noc.w_opt_ini),
                           rd; save = false)
        w_opt_retcode = res_opt.retcode
        w_opt = res_opt.w
    end
    if w_max_flag
        res_max = optimise(MeanRisk(; r = nb_r, obj = MaximumReturn(), opt = nb_opt,
                                    wi = noc.w_max_ini), rd; save = false)
        w_max_retcode = res_max.retcode
        w_max = res_max.w
    end
    pr = opt.pr
    fees = opt.fees
    ret = opt.ret
    rk_min, rk_opt, rk_max = near_optimal_centering_risks(opt.sca, r, pr, fees, opt.slv,
                                                          w_min, w_opt, w_max)
    rt_min = expected_return(ret, w_min, pr, fees)
    rt_opt = expected_return(ret, w_opt, pr, fees)
    rt_max = expected_return(ret, w_max, pr, fees)
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
    return NearOptimalSetup(w_opt, rk_opt, rt_opt, rt_min, rt_max, w_min, w_max, r, opt,
                            w_min_retcode, w_opt_retcode, w_max_retcode)
end
function set_near_optimal_centering_constraints!(model::JuMP.Model, rk::Number, rt::Number,
                                                 wb::WeightBounds)
    w = model[:w]
    sc = model[:sc]
    w_ub = wb.ub
    risk = model[:risk]
    ret = model[:ret]
    N = length(w)
    JuMP.@variables(model, begin
                        log_ret
                        log_risk
                        log_w[1:N]
                        log_delta_w[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          clog_risk,
                          [sc * log_risk, sc, sc * (rk - risk)] in
                          JuMP.MOI.ExponentialCone()
                          clog_ret,
                          [sc * log_ret, sc, sc * (ret - rt)] in JuMP.MOI.ExponentialCone()
                          clog_w[i = 1:N],
                          [sc * log_w[i], sc, sc * w[i]] in JuMP.MOI.ExponentialCone()
                          clog_delta_w[i = 1:N],
                          [sc * log_delta_w[i], sc, sc * (w_ub[i] - w[i])] in
                          JuMP.MOI.ExponentialCone()
                      end)
    JuMP.@expression(model, obj_expr, -(log_ret + log_risk + sum(log_w + log_delta_w)))
    return obj_expr
end
function set_near_optimal_objective_function!(::UnconstrainedNearOptimalCentering,
                                              model::JuMP.Model, rk::Number, rt::Number,
                                              opt::BaseJuMPOptimisationEstimator)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, opt.wb)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_near_optimal_objective_function!(::ConstrainedNearOptimalCentering,
                                              model::JuMP.Model, rk::Number, rt::Number,
                                              opt::BaseJuMPOptimisationEstimator)
    so = model[:so]
    obj_expr = set_near_optimal_centering_constraints!(model, rk, rt, opt.wb)
    add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, opt.ret, opt.cobj, obj_expr, opt, opt.pr)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function unregister_noc_variables!(model::JuMP.Model)
    if !haskey(model, :log_ret)
        return nothing
    end
    JuMP.delete(model, model[:log_ret])
    JuMP.unregister(model, :log_ret)
    JuMP.delete(model, model[:log_risk])
    JuMP.unregister(model, :log_risk)
    JuMP.delete(model, model[:log_w])
    JuMP.unregister(model, :log_w)
    JuMP.delete(model, model[:log_delta_w])
    JuMP.unregister(model, :log_delta_w)
    JuMP.delete(model, model[:clog_risk])
    JuMP.unregister(model, :clog_risk)
    JuMP.delete(model, model[:clog_ret])
    JuMP.unregister(model, :clog_ret)
    JuMP.delete(model, model[:clog_w])
    JuMP.unregister(model, :clog_w)
    JuMP.delete(model, model[:clog_delta_w])
    JuMP.unregister(model, :clog_delta_w)
    JuMP.unregister(model, :obj_expr)
    return nothing
end
function solve_noc!(noc::NearOptimalCentering, model::JuMP.Model, rk_opt::Number,
                    rt_opt::Number, opt::BaseJuMPOptimisationEstimator, args...)
    set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
    return optimise_JuMP_model!(model, noc, eltype(opt.pr.X))
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:UnconstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator)
    retcodes = sizehint!(Vector{OptimisationReturnCode}(undef, 0), length(rk_opts))
    sols = sizehint!(Vector{JuMPOptimisationSolution}(undef, 0), length(rk_opts))
    for (rk_opt, rt_opt) in zip(rk_opts, rt_opts)
        unregister_noc_variables!(model)
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pr.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function compute_ret_lbs(lbs::Frontier, rt_min::Number, rt_max::Number)
    return range(rt_min, rt_max; length = lbs.N)
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator, rt_min::Number, rt_max::Number,
                    ::Any, ::Any, ::Val{true}, ::Val{false})
    lbs = compute_ret_lbs(model[:ret_frontier], rt_min, rt_max)
    sc = model[:sc]
    ret_expr = model[:ret]
    retcodes = Vector{OptimisationReturnCode}(undef, length(rk_opts))
    sols = Vector{JuMPOptimisationSolution}(undef, length(rk_opts))
    for (i, (rk_opt, rt_opt, lb)) in enumerate(zip(rk_opts, rt_opts, lbs))
        if i != 1
            JuMP.delete(model, model[:ret_lb])
            JuMP.unregister(model, :ret_lb)
            JuMP.unregister(model, :obj_expr)
        end
        JuMP.@constraint(model, ret_lb, sc * (ret_expr - lb) >= 0)
        unregister_noc_variables!(model)
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pr.X))
        retcodes[i] = retcode
        sols[i] = sol
    end
    return retcodes, sols
end
function rebuild_risk_frontier(noc::NearOptimalCentering{<:Any, <:AbstractVector, <:Any,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any, <:Any,
                                                         <:ConstrainedNearOptimalCentering},
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, w_min::VecNum, w_max::VecNum,
                               idx::VecInt)
    risk_frontier = copy(risk_frontier)
    r = factory(view(noc.r, idx), pr, noc.opt.slv)
    for (i, ri) in zip(idx, r)
        risk_frontier[i] = _rebuild_risk_frontier(pr, fees, ri, risk_frontier, w_min, w_max,
                                                  i)
    end
    return risk_frontier
end
function rebuild_risk_frontier(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any,
                                                         <:ConstrainedNearOptimalCentering},
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, w_min::VecNum, w_max::VecNum,
                               args...)
    risk_frontier = copy(risk_frontier)
    r = factory(noc.r, pr, noc.opt.slv)
    return [_rebuild_risk_frontier(pr, fees, r, risk_frontier, w_min, w_max)]
end
function compute_risk_ubs(model::JuMP.Model,
                          noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any,
                                                    <:ConstrainedNearOptimalCentering},
                          pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum,
                          w_max::VecNum)
    risk_frontier = model[:risk_frontier]
    idx = Vector{Int}(undef, 0)
    for (i, rkf) in enumerate(risk_frontier)
        if !isa(rkf.second[2], VecNum)
            push!(idx, i)
        end
    end
    if isempty(idx)
        return risk_frontier
    end
    return rebuild_risk_frontier(noc, pr, fees, risk_frontier, w_min, w_max, idx)
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator, ::Any, ::Any, w_min::VecNum,
                    w_max::VecNum, ::Val{false}, ::Val{true})
    risk_frontier = compute_risk_ubs(model, noc, opt.pr, opt.fees, w_min, w_max)
    itrs = [(Iterators.repeated(rkf[1], length(rkf[2][2])),
             Iterators.repeated(rkf[2][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(Vector{OptimisationReturnCode}(undef, 0), length(rk_opts))
    sols = sizehint!(Vector{JuMPOptimisationSolution}(undef, 0), length(rk_opts))
    sc = model[:sc]
    for (keys, r_exprs, ubs, rk_opt, rt_opt) in
        zip(pitrs[1], pitrs[2], pitrs[3], rk_opts, rt_opts)
        unregister_noc_variables!(model)
        for (key, r_expr, ub) in zip(keys, r_exprs, ubs)
            if haskey(model, key)
                JuMP.delete(model, model[key])
                JuMP.unregister(model, key)
            end
            model[key] = JuMP.@constraint(model, sc * (r_expr - ub) <= 0)
        end
        set_near_optimal_objective_function!(noc.alg, model, rk_opt, rt_opt, opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pr.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    msg = ""
    if isa(w_min_retcode, OptimisationFailure)
        msg *= "w_min failed.\n"
    end
    if !isa(w_opt_retcode, AbstractVector) && isa(w_opt_retcode, OptimisationFailure) ||
       isa(w_opt_retcode, AbstractVector) &&
       any(x -> isa(x, OptimisationFailure), w_opt_retcode)
        msg *= "w_opt failed.\n"
    end
    if isa(w_max_retcode, OptimisationFailure)
        msg *= "w_max failed.\n"
    end
    if !isa(noc_retcode, AbstractVector) && isa(noc_retcode, OptimisationFailure) ||
       isa(noc_retcode, AbstractVector) &&
       any(x -> isa(x, OptimisationFailure), noc_retcode)
        msg *= "noc_opt failed."
    end
    return if isempty(msg)
        OptimisationSuccess(nothing)
    else
        @warn("Failed to solve optimisation problem. Check `retcode.res` for details.")
        OptimisationFailure(msg)
    end
end
function _optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:UnconstrainedNearOptimalCentering},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; w_opt, rk_opt, rt_opt, r, opt, w_min_retcode, w_opt_retcode, w_max_retcode) = near_optimal_centering_setup(noc,
                                                                                                                  rd;
                                                                                                                  dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    JuMP.@expression(model, k, 1)
    set_w!(model, opt.pr.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    set_risk_constraints!(model, r, noc, opt.pr, nothing, nothing, opt.fees; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pr; rd = rd)
    noc_retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt)
    retcode = get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    return NearOptimalCenteringResult(typeof(noc),
                                      ProcessedJuMPOptimiserAttributes(opt.pr, opt.wb,
                                                                       opt.lt, opt.st,
                                                                       opt.lcs, opt.ct,
                                                                       opt.gcard,
                                                                       opt.sgcard, opt.smtx,
                                                                       opt.sgmtx, opt.slt,
                                                                       opt.sst, opt.sglt,
                                                                       opt.sgst, opt.pl,
                                                                       opt.tn, opt.fees,
                                                                       opt.ret),
                                      w_min_retcode, w_opt_retcode, w_max_retcode,
                                      noc_retcode, retcode, sol,
                                      ifelse(save, model, nothing), nothing)
end
function _optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:ConstrainedNearOptimalCentering},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; w_opt, rk_opt, rt_opt, r, opt, rt_min, rt_max, w_min, w_max, w_min_retcode, w_opt_retcode, w_max_retcode) = near_optimal_centering_setup(noc,
                                                                                                                                                rd;
                                                                                                                                                dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    JuMP.@expression(model, k, 1)
    set_w!(model, opt.pr.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    set_linear_weight_constraints!(model, opt.lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, opt.ct, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, opt.wb, opt.card, opt.gcard, opt.pl, opt.lt, opt.st,
                         opt.fees, opt.ss)
    set_smip_constraints!(model, opt.wb, opt.scard, opt.sgcard, opt.smtx, opt.sgmtx,
                          opt.slt, opt.sst, opt.sglt, nothing, opt.ss)
    set_turnover_constraints!(model, opt.tn)
    set_tracking_error_constraints!(model, opt.pr, opt.tr, noc, opt.pl, opt.fees; rd = rd)
    set_number_effective_assets!(model, opt.nea)
    set_l1_regularisation!(model, opt.l1)
    set_l2_regularisation!(model, opt.l2)
    set_non_fixed_fees!(model, opt.fees)
    set_risk_constraints!(model, r, noc, opt.pr, opt.pl, opt.fees; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, opt.pl)
    add_custom_constraint!(model, opt.ccnt, opt, opt.pr)
    noc_retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt, rt_min, rt_max, w_min,
                                  w_max, Val(haskey(model, :ret_frontier)),
                                  Val(haskey(model, :risk_frontier)))
    retcode = get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    return NearOptimalCenteringResult(typeof(noc),
                                      ProcessedJuMPOptimiserAttributes(opt.pr, opt.wb,
                                                                       opt.lt, opt.st,
                                                                       opt.lcs, opt.ct,
                                                                       opt.gcard,
                                                                       opt.sgcard, opt.smtx,
                                                                       opt.sgmtx, opt.slt,
                                                                       opt.sst, opt.sglt,
                                                                       opt.sgst, opt.pl,
                                                                       opt.tn, opt.fees,
                                                                       opt.ret),
                                      w_min_retcode, w_opt_retcode, w_max_retcode,
                                      noc_retcode, retcode, sol,
                                      ifelse(save, model, nothing), nothing)
end
function optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                            <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(noc, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export NearOptimalCentering, UnconstrainedNearOptimalCentering,
       ConstrainedNearOptimalCentering, NearOptimalCenteringResult
