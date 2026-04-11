"""
$(DocStringExtensions.TYPEDEF)

Result type for Mean-Risk portfolio optimisation.

# Fields

  - `oe`: Type of the optimisation estimator that produced this result.
  - `pa`: Processed optimisation attributes.
  - `retcode`: Optimisation return code.
  - `sol`: Optimisation solution (or vector of solutions for the efficient frontier).
  - `model`: The JuMP model used for optimisation.
  - `fb`: Fallback result (if a fallback optimiser was used).

The `w` property is forwarded from `sol.w`.

# Related

  - [`NonFiniteAllocationOptimisationResult`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct MeanRiskResult <: NonFiniteAllocationOptimisationResult
    oe
    pa
    retcode
    sol
    model
    fb
end
function factory(res::MeanRiskResult, fb::Option{<:OptE_Opt})
    return MeanRiskResult(res.oe, res.pa, res.retcode, res.sol, res.model, fb)
end
function Base.getproperty(r::MeanRiskResult, sym::Symbol)
    return if sym == :w
        !isa(r.sol, AbstractVector) ? getfield(r.sol, :w) : getfield.(r.sol, :w)
    elseif sym in propertynames(r)
        getfield(r, sym)
    elseif sym in propertynames(r.pa)
        getproperty(r.pa, sym)
    else
        getfield(r, sym)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Mean-Risk portfolio optimiser.

`MeanRisk` formulates and solves a mean-risk portfolio optimisation problem using JuMP. It can optimise a wide variety of objective functions (minimum risk, maximum return, maximum Sharpe ratio, maximum utility) subject to risk, weight, cardinality, and custom constraints.

# Fields

  - `opt`: JuMP optimiser configuration (prior, solver, constraints, bounds, fees, etc.).
  - `r`: Risk measure or vector of risk measures.
  - `obj`: Portfolio objective function.
  - `wi`: Initial portfolio weights for warm-starting the solver (or `nothing`).
  - `fb`: Fallback optimiser.

# Constructors

    MeanRisk(;
        opt::JuMPOptimiser = JuMPOptimiser(),
        r::RM_VecRM = Variance(),
        obj::ObjectiveFunction = MinimumRisk(),
        wi::Option{<:VecNum} = nothing,
        fb::Option{<:OptE_Opt} = nothing
    ) -> MeanRisk

Keywords correspond to the struct's fields.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.

# Examples

```jldoctest
julia> MeanRisk(; opt = JuMPOptimiser(; slv = Clarabel.Optimizer))
MeanRisk
  opt ┼ JuMPOptimiser
  r ┼ Variance
  obj ┼ MinimumRisk
  wi ┼ nothing
  fb ┴ nothing
```

# Related

  - [`scalarise_risk_expression!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`ArithmeticReturn`](@ref)
  - [`LogarithmicReturn`](@ref)
  - [`MinimumRisk`](@ref)
  - [`MaximumUtility`](@ref)
  - [`MaximumRatio`](@ref)
  - [`MaximumReturn`](@ref)
  - [`BudgetRange`](@ref)
  - [`LpRegularisation`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`MeanRiskResult`](@ref)
  - [`ObjectiveFunction`](@ref)
  - [`RiskMeasure`](@ref)
"""
@concrete struct MeanRisk <: RiskJuMPOptimisationEstimator
    opt
    r
    obj
    wi
    fb
    function MeanRisk(opt::JuMPOptimiser, r::RM_VecRM, obj::ObjectiveFunction,
                      wi::Option{<:VecNum}, fb::Option{<:OptE_Opt})
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
function MeanRisk(; opt::JuMPOptimiser = JuMPOptimiser(), r::RM_VecRM = Variance(),
                  obj::ObjectiveFunction = MinimumRisk(), wi::Option{<:VecNum} = nothing,
                  fb::Option{<:OptE_Opt} = nothing)
    return MeanRisk(opt, r, obj, wi, fb)
end
function needs_previous_weights(opt::MeanRisk)
    return (needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
function factory(mr::MeanRisk, w::AbstractVector)
    opt = factory(mr.opt, w)
    r = factory(mr.r, w)
    fb = factory(mr.fb, w)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = mr.wi, fb = fb)
end
function opt_view(mr::MeanRisk, i, X::MatNum)
    X = isa(mr.opt.pe, AbstractPriorResult) ? mr.opt.pe.X : X
    opt = opt_view(mr.opt, i, X)
    r = risk_measure_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi, fb = mr.fb)
end
"""
    solve_mean_risk!(model, mr, ret, pr, ::Val{false}, ::Val{false}, args...)
    solve_mean_risk!(model, mr, ret, pr, ::Val{true}, ::Val{false}, fees)
    solve_mean_risk!(model, mr, ret, pr, ::Val{false}, ::Val{true}, fees)
    solve_mean_risk!(model, mr, ret, pr, ::Val{true}, ::Val{true}, fees)

Solve the Mean-Risk optimisation problem.

Dispatches based on whether a return frontier and/or risk frontier sweep is requested (controlled by `Val` arguments). Single-point, return-frontier, risk-frontier, and combined sweeps are all handled.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `mr::MeanRisk`: MeanRisk estimator configuration.
  - `ret::JuMPReturnsEstimator`: Returns estimator.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `::Val{bool}`: Whether to do a return frontier sweep.
  - `::Val{bool}`: Whether to do a risk frontier sweep.
  - `fees`: Optional fees configuration.

# Returns

  - `(retcode, sol)` or `(retcodes, sols)` depending on the sweep mode.

# Related

  - [`MeanRisk`](@ref)
  - [`compute_ret_lbs`](@ref)
  - [`compute_risk_ubs`](@ref)
"""
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{false}, args...)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    return optimise_JuMP_model!(model, mr, eltype(pr.X))
end
"""
    compute_ret_lbs(lbs, args...)

Compute the return lower bounds for the efficient frontier sweep.

Dispatches based on the type of `lbs`: if a pre-computed vector of lower bounds is provided, returns it directly. If a `Frontier` specification is given, solves the minimum and maximum return sub-problems and constructs a range of bounds.

# Arguments

  - `lbs`: Pre-computed return bounds vector (`VecNum`) or `Frontier` configuration.
  - `args...`: Additional arguments (model, optimiser, prior, etc.) needed when `lbs` is a `Frontier`.

# Returns

  - Vector or range of return lower bounds for frontier sweep.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_mean_risk!`](@ref)
"""
function compute_ret_lbs(lbs::VecNum, args...)
    return lbs
end
"""
    compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator, pr::AbstractPriorResult, fees::Option{<:Fees})

Compute return lower bounds for a `MeanRisk` efficient frontier sweep by solving minimum and maximum return sub-problems.

Solves the minimum-risk and maximum-return portfolios, then constructs a uniformly spaced range of `lbs.N` return targets spanning the two extremes.

# Arguments

  - `lbs::Frontier`: Frontier configuration specifying the number of points.
  - `model::JuMP.Model`: JuMP optimisation model.
  - `mr::MeanRisk`: MeanRisk estimator configuration.
  - `ret::JuMPReturnsEstimator`: Returns estimator.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `fees::Option{<:Fees}`: Optional fees configuration.

# Returns

  - Range of return lower bounds for the frontier sweep.

# Related

  - [`compute_ret_lbs`](@ref)
  - [`MeanRisk`](@ref)
  - [`solve_mean_risk!`](@ref)
"""
function compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk,
                         ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                         fees::Option{<:Fees} = nothing)
    X = pr.X
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    rt_min = expected_return(ret, sol_min.w, pr, fees)
    rt_max = expected_return(ret, sol_max.w, pr, fees)
    return range(rt_min, rt_max; length = lbs.N)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{false},
                          fees::Option{<:Fees})
    X = pr.X
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs))
    k = model[:k]
    sc = model[:sc]
    ret_expr = model[:ret]
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret_expr - ret_lb_var * k) >= 0)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    for lb in lbs
        JuMP.set_parameter_value(ret_lb_var, lb)
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
"""
    _rebuild_risk_frontier(pr, fees, ...)

Internal helper to rebuild the risk frontier from a prior result.

Recomputes the risk range used for the efficient frontier given updated prior information and fee structures.

# Arguments

  - `pr`: Prior result with asset moments.
  - `fees`: Optional fees configuration.
  - Additional parameters.

# Returns

  - Tuple of risk bound values for the frontier.

# Related

  - [`rebuild_risk_frontier`](@ref)
  - [`MeanRisk`](@ref)
"""
function _rebuild_risk_frontier(pr::AbstractPriorResult, fees::Option{<:Fees},
                                r::RiskMeasure, risk_frontier::VecPair, w_min::VecNum,
                                w_max::VecNum, i::Integer = 1)
    (; N, factor, flag) = risk_frontier[i].second[2]
    X = pr.X
    rk_min = expected_risk(r, w_min, X, fees)
    rk_max = expected_risk(r, w_max, X, fees)
    rk_min, rk_max = if flag
        factor * rk_min, factor * rk_max
    else
        factor * sqrt(rk_min), factor * sqrt(rk_max)
    end
    ub = range(rk_min, rk_max; length = N)
    return risk_frontier[i].first => (risk_frontier[1].second[1], ub)
end
"""
    rebuild_risk_frontier(model, mr, ...)

Rebuild the efficient frontier risk bounds from a solved JuMP model.

Extracts and recomputes risk bound values from the optimised model for use in subsequent frontier sweeps.

# Arguments

  - `model`: Solved JuMP model.
  - `mr`: MeanRisk optimiser configuration.
  - Additional parameters.

# Returns

  - Updated risk bounds for the frontier.

# Related

  - [`MeanRisk`](@ref)
  - [`_rebuild_risk_frontier`](@ref)
"""
function rebuild_risk_frontier(model::JuMP.Model,
                               mr::MeanRisk{<:Any, <:AbstractVector, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               fees::Option{<:Fees}, risk_frontier::VecPair, idx::VecInt)
    X = pr.X
    risk_frontier = copy(risk_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    r = factory(view(mr.r, idx), pr, mr.opt.slv)
    for (i, ri) in zip(idx, r)
        risk_frontier[i] = _rebuild_risk_frontier(pr, fees, ri, risk_frontier, sol_min.w,
                                                  sol_max.w, i)
    end
    return risk_frontier
end
function rebuild_risk_frontier(model::JuMP.Model, mr::MeanRisk{<:Any, <:Any, <:Any, <:Any},
                               ret::JuMPReturnsEstimator, pr::AbstractPriorResult,
                               fees::Option{<:Fees}, risk_frontier::VecPair, args...)
    X = pr.X
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess))
    JuMP.unregister(model, :obj_expr)
    r = factory(mr.r, pr, mr.opt.slv)
    return (_rebuild_risk_frontier(pr, fees, r, risk_frontier, sol_min.w, sol_max.w),)
end
"""
    compute_risk_ubs(model, opt, ...)

Compute or rebuild risk upper bounds for the efficient frontier sweep.

Extracts the risk frontier from the model and rebuilds any frontier bounds that have not yet been computed as numeric vectors.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model containing the risk frontier.
  - `opt`: Optimiser configuration.
  - Additional arguments (prior, fees, weights, etc.).

# Returns

  - Updated risk frontier vector of pairs.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_mean_risk!`](@ref)
"""
function compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, fees::Option{<:Fees})
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
    return rebuild_risk_frontier(model, mr, ret, pr, fees, risk_frontier, idx)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{true},
                          fees::Option{<:Fees})
    X = pr.X
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees)
    k = model[:k]
    sc = model[:sc]
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        model[keys[2]] = JuMP.@constraint(model, sc * (vals[1] - ub * k) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(pitrs))
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    for (keys, ubs) in zip(pitrs[1], pitrs[2])
        for (key, ub) in zip(keys, ubs)
            JuMP.set_parameter_value(model[key], ub)
        end
        retcode, sol = optimise_JuMP_model!(model, mr, eltype(X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{true},
                          fees::Option{<:Fees})
    X = pr.X
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees)
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees)
    sc = model[:sc]
    k = model[:k]
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        model[keys[2]] = JuMP.@constraint(model, sc * (vals[1] - ub * k) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs) * length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs) * length(pitrs))
    ret_expr = model[:ret]
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret_expr - ret_lb_var * k) >= 0)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr)
    for lb in lbs
        JuMP.set_parameter_value(ret_lb_var, lb)
        for (keys, ubs) in zip(pitrs[1], pitrs[2])
            for (key, ub) in zip(keys, ubs)
                JuMP.set_parameter_value(model[key], ub)
            end
            retcode, sol = optimise_JuMP_model!(model, mr, eltype(X))
            push!(retcodes, retcode)
            push!(sols, sol)
        end
    end
    return retcodes, sols
end
function _optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, slt, sst, sgmtx, sglt, sgst, plr, tn, fees, ret) = processed_jump_optimiser_attributes(mr.opt,
                                                                                                                                                rd;
                                                                                                                                                dims = dims)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, pr.mu, mr.obj)
    set_w!(model, pr.X, mr.wi)
    set_weight_constraints!(model, wb, mr.opt.bgt, mr.opt.sbgt)
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, mr.opt.card, gcardr, plr, lt, st, fees, mr.opt.ss)
    set_smip_constraints!(model, wb, mr.opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt,
                          sgst, mr.opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, mr.opt.tr, mr, plr, fees; rd = rd)
    set_number_effective_assets!(model, mr.opt.nea)
    set_l1_regularisation!(model, mr.opt.l1)
    set_l2_regularisation!(model, mr.opt.l2)
    set_linf_regularisation!(model, mr.opt.linf)
    set_lp_regularisation!(model, mr.opt.lp)
    set_non_fixed_fees!(model, fees)
    set_risk_constraints!(model, mr.r, mr, pr, plr, fees; rd = rd)
    scalarise_risk_expression!(model, mr.opt.sca)
    set_return_constraints!(model, ret, mr.obj, pr; rd = rd)
    set_sdp_phylogeny_constraints!(model, plr)
    add_custom_constraint!(model, mr.opt.ccnt, mr, pr)
    retcode, sol = solve_mean_risk!(model, mr, ret, pr, Val(haskey(model, :ret_frontier)),
                                    Val(haskey(model, :risk_frontier)), fees)
    return MeanRiskResult(typeof(mr),
                          ProcessedJuMPOptimiserAttributes(pr, wb, lt, st, lcsr, ctr,
                                                           gcardr, sgcardr, smtx, sgmtx,
                                                           slt, sst, sglt, sgst, tn, fees,
                                                           plr, ret), retcode, sol,
                          ifelse(save, model, nothing), nothing)
end
function optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(mr, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export MeanRisk, MeanRiskResult
