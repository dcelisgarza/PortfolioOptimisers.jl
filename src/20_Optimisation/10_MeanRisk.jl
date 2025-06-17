struct MeanRisk{T1 <: JuMPOptimiser,
                T2 <: Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                T3 <: ObjectiveFunction, T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       JuMPOptimisationEstimator
    opt::T1
    r::T2
    obj::T3
    wi::T4
end
function MeanRisk(; opt::JuMPOptimiser = JuMPOptimiser(),
                  r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  wi::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(r, AbstractVector)
        @smart_assert(!isempty(r))
    end
    if isa(wi, AbstractVector)
        @smart_assert(!isempty(wi))
    end
    return MeanRisk{typeof(opt), typeof(r), typeof(obj), typeof(wi)}(opt, r, obj, wi)
end
function opt_view(mr::MeanRisk, i::AbstractVector, X::AbstractMatrix)
    opt = opt_view(mr.opt, i, X)
    r = risk_measure_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, args...)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    return optimise_JuMP_model!(model, mr, eltype(pr.X))
end
function compute_ret_lbs(lbs::AbstractVector, args...)
    return lbs
end
function compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk,
                         ret::JuMPReturnsEstimator, pr::AbstractPriorResult)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    rt_min = expected_returns(ret, sol_min.w, pr, mr.opt.fees)
    rt_max = expected_returns(ret, sol_max.w, pr, mr.opt.fees)
    return range(; start = rt_min, stop = rt_max, length = lbs.N)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{false})
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr)
    sc = model[:sc]
    k = model[:k]
    ret_expr = model[:ret]
    retcodes = Vector{OptimisationReturnCode}(undef, length(lbs))
    sols = Vector{JuMPOptimisationSolution}(undef, length(lbs))
    for (i, lb) ∈ enumerate(lbs)
        if i != 1
            delete(model, model[:ret_lb])
            unregister(model, :ret_lb)
            unregister(model, :obj_expr)
        end
        @constraint(model, ret_lb, sc * (ret_expr - lb * k) >= 0)
        set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(pr.X))
        retcodes[i] = retcode
        sols[i] = sol
    end
    return retcodes, sols
end
function rebuild_risk_frontier(model::JuMP.Model,
                               mr::MeanRisk{<:Any, <:AbstractVector, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               risk_frontier::AbstractVector, idx::AbstractVector)
    risk_frontier = copy(risk_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    r = factory(view(mr.r, idx), pr, mr.opt.slv)
    for (i, ri) ∈ zip(idx, r)
        (; N, factor, flag) = risk_frontier[i].second[2]
        rk_min = expected_risk(ri, sol_min.w, pr.X, mr.opt.fees)
        rk_max = expected_risk(ri, sol_max.w, pr.X, mr.opt.fees)
        rk_min, rk_max = if flag
            factor * sqrt(rk_min), factor * sqrt(rk_max)
        else
            factor * rk_min, factor * rk_max
        end
        ub = range(; start = rk_min, stop = rk_max, length = N)
        risk_frontier[i] = risk_frontier[i].first => (risk_frontier[i].second[1], ub)
    end
    return risk_frontier
end
function rebuild_risk_frontier(model::JuMP.Model, mr::MeanRisk{<:Any, <:Any, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               risk_frontier::AbstractVector, args...)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @smart_assert(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    (; N, factor, flag) = risk_frontier[1].second[2]
    r = factory(mr.r, pr, mr.opt.slv)
    rk_min = expected_risk(r, sol_min.w, pr.X, mr.opt.fees)
    rk_max = expected_risk(r, sol_max.w, pr.X, mr.opt.fees)
    rk_min, rk_max = if flag
        factor * sqrt(rk_min), factor * sqrt(rk_max)
    else
        factor * rk_min, factor * rk_max
    end
    ub = range(; start = rk_min, stop = rk_max, length = N)
    return [risk_frontier[1].first => (risk_frontier[1].second[1], ub)]
end
function compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult)
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
    return rebuild_risk_frontier(model, mr, ret, pr, risk_frontier, idx)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{true})
    risk_frontier = compute_risk_ubs(model, mr, ret, pr)
    itrs = [(Iterators.repeated(rkf[1], length(rkf[2][2])),
             Iterators.repeated(rkf[2][1], length(rkf[2][2])), rkf[2][2])
            for rkf ∈ risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(Vector{OptimisationReturnCode}(undef, 0), length(pitrs))
    sols = sizehint!(Vector{JuMPOptimisationSolution}(undef, 0), length(pitrs))
    k = model[:k]
    sc = model[:sc]
    for (keys, r_exprs, ubs) ∈ zip(pitrs[1], pitrs[2], pitrs[3])
        if haskey(model, :obj_expr)
            unregister(model, :obj_expr)
        end
        for (key, r_expr, ub) ∈ zip(keys, r_exprs, ubs)
            if haskey(model, key)
                delete(model, model[key])
                unregister(model, key)
            end
            model[key] = @constraint(model, sc * (r_expr - ub * k) <= 0)
        end
        set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(pr.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function optimise!(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lcs, cent, gcard, sgcard, smtx, nplg, cplg, ret) = processed_jump_optimiser_attributes(mr.opt,
                                                                                                      rd;
                                                                                                      dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    set_w!(model, pr.X, mr.wi)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq, :lcs_eq)
    set_linear_weight_constraints!(model, cent, :cent_ineq, :cent_eq)
    set_linear_weight_constraints!(model, mr.opt.lcm, :lcm_ineq, :lcm_eq)
    set_mip_constraints!(model, wb, mr.opt.card, gcard, nplg, cplg, mr.opt.lt, mr.opt.st,
                         mr.opt.fees, mr.opt.ss)
    set_smip_constraints!(model, wb, mr.opt.scard, sgcard, smtx, mr.opt.ss)
    set_turnover_constraints!(model, mr.opt.tn)
    set_tracking_error_constraints!(model, pr, mr.opt.te, mr, nplg, cplg)
    set_number_effective_assets!(model, mr.opt.nea)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, mr.opt.fees)
    set_risk_constraints!(model, mr.r, mr, pr, nplg, cplg)
    scalarise_risk_expression!(model, mr.opt.sce)
    set_return_constraints!(model, ret, mr.obj, pr)
    set_sdp_philogeny_constraints!(model, nplg, :sdp_nplg)
    set_sdp_philogeny_constraints!(model, cplg, :sdp_cplg)
    add_custom_constraint!(model, mr.opt.ccnt, mr, pr)
    retcode, sol = solve_mean_risk!(model, mr, ret, pr, Val(haskey(model, :ret_frontier)),
                                    Val(haskey(model, :risk_frontier)))
    return JuMPOptimisationResult(typeof(mr), pr, wb, lcs, cent, gcard, sgcard, smtx, nplg,
                                  cplg, ret, retcode, sol, ifelse(save, model, nothing))
end
function efficient_frontier_bounds(mr::MeanRisk, rd::ReturnsResult,
                                   wi_min::Union{Nothing, <:AbstractVector{<:Real}},
                                   wi_max::Union{Nothing, <:AbstractVector{<:Real}}, dims;
                                   kwargs...)
    res = optimise!(mr, rd; dims = dims, kwargs...)
    if isa(res.retcode, OptimisationFailure)
        throw(OptimisationFailure("Failed to compute minimum risk for efficient frontier."))
    end
    model = res.model
    try
        set_start_value.(model[:w], wi_max)
    catch
    end
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), res.ret, mr.opt.cobj, mr,
                                      res.pr)
    retcode, sol = optimise_JuMP_model!(model, mr, eltype(res.w))
    if isa(retcode, OptimisationFailure)
        throw(OptimisationFailure("Failed to compute maximum return for efficient frontier."))
    end
    return res, sol.w
end
function efficient_frontier!(mr::MeanRisk, rd::ReturnsResult = ReturnsResult();
                             N::Integer = 20,
                             wi_min::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                             wi_max::Union{Nothing, <:AbstractVector{<:Real}} = nothing,
                             risk_ub::Real = inv(sqrt(eps())),
                             ret_lb::Real = -inv(sqrt(eps())),
                             rf::Union{Nothing, <:Real} = 0.0,
                             ohf::Union{Nothing, <:Real} = nothing, dims::Int = 1,
                             save::Bool = true, kwargs...)
    opt = JuMPOptimiser(; pe = mr.opt.pe, slv = mr.opt.slv, wb = mr.opt.wb,
                        bgt = mr.opt.bgt, sbgt = mr.opt.sbgt, lt = mr.opt.lt,
                        st = mr.opt.st, lcs = mr.opt.lcs, lcm = mr.opt.lcm,
                        cent = mr.opt.cent, gcard = mr.opt.gcard, sgcard = mr.opt.sgcard,
                        smtx = mr.opt.smtx, sets = mr.opt.sets, nplg = mr.opt.nplg,
                        cplg = mr.opt.cplg, tn = mr.opt.tn, te = mr.opt.te,
                        fees = mr.opt.fees,
                        ret = bounds_returns_estimator(mr.opt.ret, ret_lb),
                        sce = mr.opt.sce, ccnt = mr.opt.ccnt, cobj = mr.opt.cobj,
                        sc = mr.opt.sc, so = mr.opt.so, card = mr.opt.card,
                        scard = mr.opt.scard, nea = mr.opt.nea, l1 = mr.opt.l1,
                        l2 = mr.opt.l2, ss = mr.opt.ss, strict = mr.opt.strict)
    mr = MeanRisk(; opt = opt, r = bounds_first_risk_measure(mr.r, risk_ub),
                  obj = MinimumRisk(), wi = wi_min)
    res, w2 = efficient_frontier_bounds(mr, rd, wi_min, wi_max, dims; kwargs...)
    w1 = res.w
    r = factory(mr.r, res.pr, mr.opt.slv)
    ri = isa(r, AbstractVector) ? r[1] : r
    risk1 = expected_risk(ri, w1, res.pr.X, mr.opt.fees; kwargs...)
    risk2 = expected_risk(ri, w2, res.pr.X, mr.opt.fees; kwargs...)
    ret1 = expected_returns(res.ret, w1, res.pr, mr.opt.fees)
    ret2 = expected_returns(res.ret, w2, res.pr, mr.opt.fees)
    model = res.model
    ks = string.(keys(object_dictionary(model)))
    rkey = Symbol(ks[findfirst(x -> contains(x, r".*_risk_1$"), ks)])
    if isa(model[rkey], QuadExpr)
        risk1 = sqrt(risk1)
        risk2 = sqrt(risk2)
    end
    key = Symbol(ks[findfirst(x -> contains(x, r".*_1_ub$"), ks)])
    risks = range(risk1; stop = risk2, length = N)
    rets = range(ret1; stop = ret2, length = N)
    M = length(w1)
    ws = Vector{eltype(w1)}(undef, (N + (!isnothing(rf) ? 1 : 0)) * M)
    ws[1:M] .= w1
    ws[((N - 1) * M + 1):(N * M)] .= w2
    try
        set_start_value.(model[:w], w1)
    catch
    end
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), res.ret, mr.opt.cobj, mr,
                                      res.pr)
    failure = OptimisationFailure[]
    for i ∈ 2:length(risks)
        idx = ((i - 1) * M + 1):(i * M)
        set_normalized_rhs(model[key], risks[i])
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(w1))
        if isa(retcode, OptimisationSuccess)
            ws[idx] .= sol.w
        else
            set_normalized_rhs(model[key], risk_ub)
            set_normalized_rhs(model[:ret_lb], rets[i])
            unregister(model, :obj_expr)
            set_portfolio_objective_function!(model, MinimumRisk(), res.ret, mr.opt.cobj,
                                              mr, res.pr)
            retcode, sol = optimise_JuMP_model!(model, mr, eltype(res.w))
            set_normalized_rhs(model[:ret_lb], ret_lb)
            if isa(retcode, OptimisationSuccess)
                ws[idx] .= sol.w
            else
                push!(failure, retcode)
                ws[idx] .= NaN
            end
        end
    end
    if !isnothing(rf)
        delete(model, model[key])
        unregister(model, key)
        delete(model, model[:ret_lb])
        unregister(model, :ret_lb)
        unregister(model, :obj_expr)
        set_portfolio_objective_function!(model, MaximumRatio(; rf = rf, ohf = ohf),
                                          res.ret, mr.opt.cobj, mr, res.pr)
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(res.w))
        if isa(retcode, OptimisationSuccess)
            ws[(N * M + 1):end] .= sol.w
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
                                  ifelse(save, model, nothing))
end

export MeanRisk
