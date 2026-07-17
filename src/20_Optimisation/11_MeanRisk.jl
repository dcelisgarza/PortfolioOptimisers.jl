"""
$(DocStringExtensions.TYPEDEF)

Result type for Mean-Risk portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref): the virtual `:w` property and unknown properties resolve through `jr`.

# Constructors

    MeanRiskResult(; jr::JuMPOptimisationResult, fb::Option{<:OptE_Opt}) -> MeanRiskResult

Keywords correspond to the struct's fields.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct MeanRiskResult <: RiskJuMPOptimisationResult
    """
    Shared JuMP result core, see [`JuMPOptimisationResult`](@ref).
    """
    jr
    """
    $(field_dict[:fb])
    """
    fb
    function MeanRiskResult(jr::JuMPOptimisationResult, fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(fb)}(jr, fb)
    end
end
function MeanRiskResult(; jr::JuMPOptimisationResult,
                        fb::Option{<:OptE_Opt})::MeanRiskResult
    return MeanRiskResult(jr, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`MeanRisk`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`MeanRisk`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function mean_risk_td_defaults()::NamedTuple
    return (; r = Variance(), obj = MinimumRisk())
end
"""
$(DocStringExtensions.TYPEDEF)

Mean-Risk portfolio optimiser.

`MeanRisk` formulates and solves a mean-risk portfolio optimisation problem using JuMP. It can optimise a wide variety of objective functions (minimum risk, maximum return, maximum Sharpe ratio, maximum utility) subject to risk, weight, cardinality, and custom constraints.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanRisk(;
        opt::JuMPOptimiser,
        r::TD{<:RM_VecRM} = Variance(),
        obj::TD{<:ObjectiveFunction} = MinimumRisk(),
        wi::TD_Option{<:VecNum} = nothing,
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> MeanRisk

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the risk measure, objective, warm start and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default (`nothing` for `wi` and `fb`, so a scheduled fallback is disabled outside fold loops unless the schedule carries a `default`).

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.
  - `fb` schedules: `bind !== :nearest`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing)))
MeanRisk
  opt ┼ JuMPOptimiser
      │        pe ┼ EmpiricalPrior
      │           │        ce ┼ PortfolioOptimisersCovariance
      │           │           │   ce ┼ Covariance
      │           │           │      │    me ┼ SimpleExpectedReturns
      │           │           │      │       │   w ┴ nothing
      │           │           │      │    ce ┼ GeneralCovariance
      │           │           │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │           │           │      │       │    w ┴ nothing
      │           │           │      │   alg ┴ FullMoment()
      │           │           │   mp ┼ MatrixProcessing
      │           │           │      │     pdm ┼ Posdef
      │           │           │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │           │           │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │           │           │      │      dn ┼ nothing
      │           │           │      │      dt ┼ nothing
      │           │           │      │     alg ┼ nothing
      │           │           │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │           │        me ┼ SimpleExpectedReturns
      │           │           │   w ┴ nothing
      │           │   horizon ┴ nothing
      │       slv ┼ Solver
      │           │          name ┼ String: ""
      │           │        solver ┼ nothing
      │           │      settings ┼ nothing
      │           │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │           │   add_bridges ┴ Bool: true
      │        wb ┼ WeightBounds
      │           │   lb ┼ Float64: 0.0
      │           │   ub ┴ Float64: 1.0
      │       bgt ┼ Float64: 1.0
      │      sbgt ┼ nothing
      │      gbgt ┼ nothing
      │      xbgt ┼ Bool: false
      │        lt ┼ nothing
      │        st ┼ nothing
      │      lcse ┼ nothing
      │       cte ┼ nothing
      │    gcarde ┼ nothing
      │   sgcarde ┼ nothing
      │      smtx ┼ nothing
      │     sgmtx ┼ nothing
      │       slt ┼ nothing
      │       sst ┼ nothing
      │      sglt ┼ nothing
      │      sgst ┼ nothing
      │        tn ┼ nothing
      │      fees ┼ nothing
      │      sets ┼ nothing
      │        tr ┼ nothing
      │       ple ┼ nothing
      │       ret ┼ ArithmeticReturn
      │           │   ucs ┼ nothing
      │           │    lb ┼ nothing
      │           │    mu ┴ nothing
      │       sca ┼ SumScalariser()
      │      ccnt ┼ nothing
      │      cobj ┼ nothing
      │        sc ┼ Int64: 1
      │        so ┼ Int64: 1
      │        ss ┼ nothing
      │      card ┼ nothing
      │     scard ┼ nothing
      │       wn2 ┼ nothing
      │       wnp ┼ nothing
      │     wninf ┼ nothing
      │        l1 ┼ nothing
      │        l2 ┼ nothing
      │      linf ┼ nothing
      │        lp ┼ nothing
      │       brt ┼ Bool: false
      │    cle_pr ┼ Bool: true
      │    strict ┴ Bool: false
    r ┼ Variance
      │   settings ┼ RiskMeasureSettings
      │            │   scale ┼ Float64: 1.0
      │            │      ub ┼ nothing
      │            │     rke ┴ Bool: true
      │      sigma ┼ nothing
      │       chol ┼ nothing
      │         rc ┼ nothing
      │        alg ┴ SquaredSOCRiskExpr()
  obj ┼ MinimumRisk()
   wi ┼ nothing
   fb ┴ nothing
```

# Mathematical definition

The general mean-risk optimisation problem is:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} \\; f(\\boldsymbol{w}) \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Objective ``f`` depends on [`ObjectiveFunction`](@ref):

  - [`MinimumRisk`](@ref): ``f(\\boldsymbol{w}) = \\rho(\\boldsymbol{w})``
  - [`MaximumReturn`](@ref): ``f(\\boldsymbol{w}) = -\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w}``
  - [`MaximumUtility`](@ref): ``f(\\boldsymbol{w}) = -\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} + \\lambda \\rho(\\boldsymbol{w})``
  - [`MaximumRatio`](@ref) (Sharpe): ``f(\\boldsymbol{w}) = -(\\hat{\\boldsymbol{\\mu}}^\\intercal \\boldsymbol{w} - r_f) / \\rho(\\boldsymbol{w})``

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\mathcal{W}``: Feasible weight set defined by portfolio constraints.
  - ``f(\\boldsymbol{w})``: Objective function (depends on [`ObjectiveFunction`](@ref)).
  - ``\\rho(\\boldsymbol{w})``: Portfolio risk measure.
  - ``\\hat{\\boldsymbol{\\mu}}``: Estimated expected return vector.
  - ``\\lambda``: Risk aversion parameter.
  - ``r_f``: Risk-free rate.

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
@propagatable @concrete struct MeanRisk <: RiskJuMPOptimisationEstimator
    """
    $(field_dict[:opt_jmp])
    """
    @fprop opt
    """
    $(field_dict[:r_opt])
    """
    @fprop r
    """
    $(field_dict[:obj])
    """
    obj
    """
    $(field_dict[:wi])
    """
    wi
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function MeanRisk(opt::JuMPOptimiser, r::TD{<:RM_VecRM}, obj::TD{<:ObjectiveFunction},
                      wi::TD_Option{<:VecNum}, fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :MeanRisk)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
        end
        assert_no_risk_objective_compatibility(r, obj)
        if isa(wi, VecNum)
            @argcheck(!isempty(wi), IsEmptyError("wi cannot be empty"))
        end
        assert_time_dependent_substitution(MeanRisk, (; opt, r, obj, wi, fb),
                                           mean_risk_td_defaults())
        return new{typeof(opt), typeof(r), typeof(obj), typeof(wi), typeof(fb)}(opt, r, obj,
                                                                                wi, fb)
    end
end
function MeanRisk(; opt::JuMPOptimiser, r::TD{<:RM_VecRM} = Variance(),
                  obj::TD{<:ObjectiveFunction} = MinimumRisk(),
                  wi::TD_Option{<:VecNum} = nothing,
                  fb::TDO_Option{<:OptE_Opt} = nothing)::MeanRisk
    return MeanRisk(opt, r, obj, wi, fb)
end
function time_dependent_field_defaults(::MeanRisk)::NamedTuple
    return mean_risk_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (JuMP optimiser, risk measure, or fallback).
"""
function needs_previous_weights(opt::MeanRisk)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`MeanRisk`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(mr::MeanRisk, i, X::MatNum, args...)::MeanRisk
    X = isa(mr.opt.pe, AbstractPriorResult) ? mr.opt.pe.X : X
    opt = port_opt_view(mr.opt, i, X)
    r = port_opt_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi, fb = mr.fb)
end
"""
    solve_mean_risk!(model, mr, ret, pr, ::Val{false}, ::Val{false}, args...)
    solve_mean_risk!(model, mr, ret, pr, ::Val{true}, ::Val{false}, fees, args...)
    solve_mean_risk!(model, mr, ret, pr, ::Val{false}, ::Val{true}, fees, args...)
    solve_mean_risk!(model, mr, ret, pr, ::Val{true}, ::Val{true}, fees, args...)

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
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr, args...)
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
    compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator, pr::AbstractPriorResult, fees::Option{<:Fees}, args...)

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
                         fees::Option{<:Fees} = nothing, args...)
    X = pr.X
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("maximum-return solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    rt_min = expected_return(ret, sol_min.w, pr, fees)
    rt_max = expected_return(ret, sol_max.w, pr, fees)
    return range(rt_min, rt_max; length = lbs.N)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{true}, ::Val{false},
                          fees::Option{<:Fees}, args...)
    X = pr.X
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees, args...)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs))
    k = get_k(model)
    sc = get_constraint_scale(model)
    ret_expr = get_ret(model)
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret_expr - ret_lb_var * k) >= 0)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr, args...)
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
    (; N, factor, bound) = risk_frontier[i].second[2]
    X = pr.X
    rk_min = expected_risk(r, w_min, X, fees)
    rk_max = expected_risk(r, w_max, X, fees)
    if bigger_is_better(r)
        rk_min, rk_max = rk_max, rk_min
    end
    rk_min, rk_max = if isa(bound, LinearBound)
        factor * rk_min, factor * rk_max
    elseif isa(bound, SquareRootBound)
        factor * sqrt(rk_min), factor * sqrt(rk_max)
    elseif isa(bound, SquaredBound)
        factor * rk_min^2, factor * rk_max^2
    end
    ub = range(rk_min, rk_max; length = N)
    return risk_frontier[i].first =>
        (risk_frontier[1].second[1], ub, risk_frontier[1].second[3])
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
                               fees::Option{<:Fees}, risk_frontier::VecPair, idx::VecInt,
                               args...)
    X = pr.X
    risk_frontier = copy(risk_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("maximum-return solve failed with retcode $retcode"))
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
    set_portfolio_objective_function!(model, MinimumRisk(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), ret, mr.opt.cobj, mr, pr,
                                      args...)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("maximum-return solve failed with retcode $retcode"))
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
                          pr::AbstractPriorResult, fees::Option{<:Fees}, args...)
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
    return rebuild_risk_frontier(model, mr, ret, pr, fees, risk_frontier, idx, args...)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator,
                          pr::AbstractPriorResult, ::Val{false}, ::Val{true},
                          fees::Option{<:Fees}, args...)
    X = pr.X
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees, args...)
    k = get_k(model)
    sc = get_constraint_scale(model)
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - ub * k) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(pitrs))
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr, args...)
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
                          fees::Option{<:Fees}, args...)
    X = pr.X
    lbs = compute_ret_lbs(model[:ret_frontier], model, mr, ret, pr, fees, args...)
    risk_frontier = compute_risk_ubs(model, mr, ret, pr, fees, args...)
    sc = get_constraint_scale(model)
    k = get_k(model)
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - ub * k) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(lbs) * length(pitrs))
    sols = sizehint!(JuMPOptimisationSolution[], length(lbs) * length(pitrs))
    ret_expr = get_ret(model)
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret_expr - ret_lb_var * k) >= 0)
    set_portfolio_objective_function!(model, mr.obj, ret, mr.opt.cobj, mr, pr, args...)
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
    mr = reset_time_dependent_estimator(mr)
    attrs = processed_jump_optimiser_attributes(mr.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, attrs.pr.mu, mr.obj)
    set_w!(model, attrs.pr.X, mr.wi)
    set_weight_constraints!(model, attrs.wb, mr.opt.bgt, mr.opt.sbgt; gbgt = mr.opt.gbgt)
    assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
    retcode, sol = solve_mean_risk!(model, mr, attrs.ret, attrs.pr,
                                    Val(haskey(model, :ret_frontier)),
                                    Val(haskey(model, :risk_frontier)), attrs.fees, attrs)
    return MeanRiskResult(;
                          jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode,
                                                      sol = sol,
                                                      model = ifelse(save, model, nothing)),
                          fb = nothing)
end
"""
    optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> MeanRiskResult

Run the Mean-Risk portfolio optimisation.

# Arguments

  - `mr`: The mean risk optimiser to use.
  - $(arg_dict[:rd]) If `isa(mr.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`MeanRisk`](@ref)
  - [`MeanRiskResult`](@ref)
"""
function optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(mr, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

export MeanRisk, MeanRiskResult
