struct MeanRisk{T1, T2, T3, T4, T5} <: RiskJuMPOptimisationEstimator
    opt::T1
    r::T2
    obj::T3
    wi::T4
    fb::T5
    function MeanRisk(opt::JuMPOptimiser,
                      r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}},
                      obj::ObjectiveFunction, wi::Union{Nothing, <:NumVec},
                      fb::Union{Nothing, <:OptimisationEstimator})
        if isa(r, AbstractVector)
            @argcheck(!isempty(r))
        end
        if !isnothing(wi)
            @argcheck(!isempty(wi))
        end
        return new{typeof(opt), typeof(r), typeof(obj), typeof(wi), typeof(fb)}(opt, r, obj,
                                                                                wi, fb)
    end
end
function MeanRisk(; opt::JuMPOptimiser = JuMPOptimiser(),
                  r::Union{<:RiskMeasure, <:AbstractVector{<:RiskMeasure}} = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(),
                  wi::Union{Nothing, <:NumVec} = nothing,
                  fb::Union{Nothing, <:OptimisationEstimator} = nothing)
    return MeanRisk(opt, r, obj, wi, fb)
end
function opt_view(mr::MeanRisk, i, X::NumMat)
    X = isa(mr.opt.pe, AbstractPriorResult) ? mr.opt.pe.X : X
    opt = opt_view(mr.opt, i, X)
    r = risk_measure_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi, fb = mr.fb)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{false}, args...)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    return optimise_JuMP_model!(model, mr, eltype(pr.X))
end
function compute_ret_lbs(lbs::NumVec, args...)
    return lbs
end
function compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk,
                         ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                         fees::Union{Nothing, <:Fees} = nothing)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    rt_min = expected_return(ret, sol_min.w, pr, fees)
    rt_max = expected_return(ret, sol_max.w, pr, fees)
    return range(rt_min, rt_max; length = lbs.N)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{false},
                          fees::Union{Nothing, <:Fees})
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs))
    k = model[:k]
    sc = model[:sc]
    ret_expr = model[:ret]
    for lb in lbs
        if haskey(model, :ret_lb)
            delete(model, model[:ret_lb])
            unregister(model, :ret_lb)
            unregister(model, :obj_expr)
        end
        @constraint(model, ret_lb, sc * (ret_expr - lb * k) >= 0)
        set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(pr.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function _rebuild_risk_frontier(pr::AbstractPriorResult, fees::Union{Nothing, <:Fees},
                                r::RiskMeasure, risk_frontier::PairVec, w_min::NumVec,
                                w_max::NumVec, i::Integer = 1)
    (; N, factor, flag) = risk_frontier[i].second[2]
    rk_min = expected_risk(r, w_min, pr.X, fees)
    rk_max = expected_risk(r, w_max, pr.X, fees)
    rk_min, rk_max = if flag
        factor * rk_min, factor * rk_max
    else
        factor * sqrt(rk_min), factor * sqrt(rk_max)
    end
    ub = range(rk_min, rk_max; length = N)
    return risk_frontier[i].first => (risk_frontier[1].second[1], ub)
end
function rebuild_risk_frontier(model::JuMP.Model,
                               mr::MeanRisk{<:Any, <:AbstractVector, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               fees::Union{Nothing, <:Fees}, risk_frontier::PairVec,
                               idx::IntVec)
    risk_frontier = copy(risk_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    r = factory(view(mr.r, idx), pr, mr.opt.slv)
    for (i, ri) in zip(idx, r)
        risk_frontier[i] = _rebuild_risk_frontier(pr, fees, ri, risk_frontier, sol_min.w,
                                                  sol_max.w, i)
    end
    return risk_frontier
end
function rebuild_risk_frontier(model::JuMP.Model, mr::MeanRisk{<:Any, <:Any, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               fees::Union{Nothing, <:Fees}, risk_frontier::PairVec,
                               args...)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(pr.X))
    @argcheck(isa(retcode, OptimisationSuccess))
    unregister(model, :obj_expr)
    r = factory(mr.r, pr, mr.opt.slv)
    return [_rebuild_risk_frontier(pr, fees, r, risk_frontier, sol_min.w, sol_max.w)]
end
function compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, fees::Union{Nothing, <:Fees})
    risk_frontier = model[:risk_frontier]
    idx = Vector{Int}(undef, 0)
    for (i, rkf) in enumerate(risk_frontier)
        if !isa(rkf.second[2], NumVec)
            push!(idx, i)
        end
    end
    if isempty(idx)
        return risk_frontier
    end
    return rebuild_risk_frontier(model, mr, ret, pr, fees, risk_frontier, idx)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{true},
                          fees::Union{Nothing, <:Fees})
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees)
    itrs = [(Iterators.repeated(rkf[1], length(rkf[2][2])),
             Iterators.repeated(rkf[2][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(pitrs))
    k = model[:k]
    sc = model[:sc]
    for (keys, r_exprs, ubs) in zip(pitrs[1], pitrs[2], pitrs[3])
        if haskey(model, :obj_expr)
            unregister(model, :obj_expr)
        end
        for (key, r_expr, ub) in zip(keys, r_exprs, ubs)
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
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{true},
                          fees::Union{Nothing, <:Fees})
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees)
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees)
    itrs = [(Iterators.repeated(rkf[1], length(rkf[2][2])),
             Iterators.repeated(rkf[2][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs) * length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs) * length(pitrs))
    k = model[:k]
    sc = model[:sc]
    ret_expr = model[:ret]
    for lb in lbs
        if haskey(model, :ret_lb)
            delete(model, model[:ret_lb])
            unregister(model, :ret_lb)
            unregister(model, :obj_expr)
        end
        @constraint(model, ret_lb, sc * (ret_expr - lb * k) >= 0)
        for (keys, r_exprs, ubs) in zip(pitrs[1], pitrs[2], pitrs[3])
            if haskey(model, :obj_expr)
                unregister(model, :obj_expr)
            end
            for (key, r_expr, ub) in zip(keys, r_exprs, ubs)
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
    end
    return retcodes, sols
end
function _optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcs, cent, gcard, sgcard, smtx, slt, sst, sgmtx, sglt, sgst, plg, tn, fees, ret) = processed_jump_optimiser_attributes(mr.opt,
                                                                                                                                              rd;
                                                                                                                                              dims = dims)
    model = JuMP.Model()
    set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    set_w!(model, pr.X, mr.wi)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcs, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, cent, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, mr.opt.card, gcard, plg, lt, st, fees, mr.opt.ss)
    set_smip_constraints!(model, wb, mr.opt.scard, sgcard, smtx, sgmtx, slt, sst, sglt,
                          sgst, mr.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, mr.opt.te, mr, plg, fees; rd = rd)
    set_number_effective_assets!(model, mr.opt.nea)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, mr.r, mr, pr, plg, fees; rd = rd)
    scalarise_risk_expression!(model, mr.opt.sce)
    set_return_constraints!(model, ret, mr.obj, pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, plg)
    add_custom_constraint!(model, mr.opt.ccnt, mr, pr)
    retcode, sol = solve_mean_risk!(model, mr, ret, pr, Val(haskey(model, :ret_frontier)),
                                    Val(haskey(model, :risk_frontier)), fees)
    return JuMPOptimisation(typeof(mr),
                            ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcs, cent,
                                                             gcard, sgcard, smtx, sgmtx,
                                                             slt, sst, sglt, sgst, plg, tn,
                                                             fees, ret), retcode, sol,
                            ifelse(save, model, nothing), nothing)
end
function optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(mr, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export MeanRisk
