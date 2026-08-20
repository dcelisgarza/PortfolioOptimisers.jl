"""
$(DocStringExtensions.TYPEDEF)

Result type for Mean-Risk portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref): the virtual `:w` property and unknown properties resolve through `jr`.

# Constructors

    MeanRiskResult(;
        jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM, fb::Option{<:OptE_Opt}
    ) -> MeanRiskResult

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
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:fb])
    """
    fb
    function MeanRiskResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                            fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(r), typeof(fb)}(jr, r, fb)
    end
end
function MeanRiskResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                        fb::Option{<:OptE_Opt})::MeanRiskResult
    return MeanRiskResult(jr, r, fb)
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
      │           │   settings ┼ JuMPReturnsSettings
      │           │            │   scale ┼ Float64: 1.0
      │           │            │      lb ┼ nothing
      │           │            │     rte ┼ Bool: true
      │           │            │     fee ┼ Bool: true
      │           │            │     mic ┴ Bool: true
      │           │        ucs ┼ nothing
      │           │         mu ┴ nothing
      │       sca ┼ SumScalariser()
      │      ccnt ┼ nothing
      │      cobj ┼ nothing
      │        sc ┼ Int64: 1
      │        so ┼ Int64: 1
      │        ss ┼ nothing
      │      card ┼ nothing
      │     scard ┼ nothing
      │       l2c ┼ nothing
      │       lpc ┼ nothing
      │     linfc ┼ nothing
      │        l1 ┼ nothing
      │        l2 ┼ nothing
      │      linf ┼ nothing
      │        lp ┼ nothing
      │       brt ┼ Bool: false
      │     x_src ┼ Symbol: :prior
      │     z_src ┼ Symbol: :data
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

  - [`optimise`](@ref)
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

# References

  - $(ref_dict[:markowitz1952])
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
    solve_mean_risk!(model, mr, pr, ::Val{false}, ::Val{false}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{true}, ::Val{false}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{false}, ::Val{true}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{true}, ::Val{true}, fees, attrs)

Solve the Mean-Risk optimisation problem.

Dispatches based on whether a return frontier and/or risk frontier sweep is requested (controlled by `Val` arguments). Single-point, return-frontier, risk-frontier, and combined sweeps are all handled.

The return terms are read off `attrs.ret` rather than taken positionally: the positional used to exist so [`set_portfolio_objective_function!`](@ref) could dispatch a logarithmic ratio problem, and it lost that reason when the ratio constraint was hoisted.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `mr::MeanRisk`: MeanRisk estimator configuration.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `::Val{bool}`: Whether to do a return frontier sweep.
  - `::Val{bool}`: Whether to do a risk frontier sweep.
  - `fees`: Optional fees configuration.
  - `attrs`: Pre-computed constraint and prior bundle, which carries the return terms.

# Returns

  - `(retcode, sol)` or `(retcodes, sols)` depending on the sweep mode.

# Related

  - [`MeanRisk`](@ref)
  - [`compute_ret_lbs`](@ref)
  - [`compute_risk_ubs`](@ref)
"""
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          ::Val{false}, ::Val{false}, ::Option{<:Fees},
                          attrs::ProcessedJuMPOptimiserAttributes)
    set_portfolio_objective_function!(model, mr.obj, mr, attrs)
    return optimise_JuMP_model!(model, mr, eltype(pr.X))
end
"""
    return_term(ret, i)

Select return term `i` from the optimiser's `ret` slot.

The slot holds one term or a vector of them, so this is the one place that difference is
resolved for the value-level readers. A single term answers to every index, because a
single-term configuration only ever has index `1`.

# Related

  - [`JRE_VecJRE`](@ref)
  - [`compute_ret_lbs`](@ref)
"""
function return_term(ret::JuMPReturnsEstimator, ::Any)
    return ret
end
function return_term(ret::VecJRE, i)
    return ret[i]
end
"""
    compute_ret_lbs(model, mr, pr, fees, attrs)

Resolve the return frontier's sweep points, one span per swept return term.

Entries whose bound is already a vector of numbers are left alone. Every entry that still
holds a [`Frontier`](@ref) is given a span read off two corner portfolios:

 1. **One** shared minimum-risk solve, which supplies the low end of every span.
 2. **One** [`MaximumElementReturn`](@ref) solve **per swept term**, which supplies that
    term's high end.

So the endpoint cost is `k + 1` solves, not `2`. The per-term maximum is what makes the span
correct: term *i*'s value at the *aggregate* maximum-return corner is an artefact of the
other terms' `scale`, and can fall below its value at the minimum-risk corner, which leaves
`rt_min > rt_max` and makes `range` descend. A portfolio that maximised term *i* alone
maximises over the same feasible set that contains `w_min`, so the span ascends by
construction.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `mr::MeanRisk`: MeanRisk estimator configuration.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `fees::Option{<:Fees}`: Optional fees configuration.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed bundle, which carries the terms.

# Returns

  - The return frontier vector of `(keys, vals)` pairs, with every span resolved.

# Related

  - [`MeanRisk`](@ref)
  - [`set_return_bounds!`](@ref)
  - [`compute_risk_ubs`](@ref)
  - [`solve_mean_risk!`](@ref)
"""
function compute_ret_lbs(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                         fees::Option{<:Fees}, attrs::ProcessedJuMPOptimiserAttributes)
    ret_frontier = shared_get(model, :ret_frontier)
    idx = Vector{Int}(undef, 0)
    for (j, rtf) in enumerate(ret_frontier)
        if !isa(rtf.second[2], VecNum)
            push!(idx, j)
        end
    end
    if isempty(idx)
        return ret_frontier
    end
    X = pr.X
    ret_frontier = copy(ret_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), mr, attrs)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    for j in idx
        expr, front, i = ret_frontier[j].second
        set_portfolio_objective_function!(model, MaximumElementReturn(i), mr, attrs)
        retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
        @argcheck(isa(retcode, OptimisationSuccess),
                  ArgumentError("maximum-return solve for return term $i failed with retcode $retcode"))
        JuMP.unregister(model, :obj_expr)
        rt = return_term(attrs.ret, i)
        rt_min = expected_return(rt, sol_min.w, pr, fees)
        rt_max = expected_return(rt, sol_max.w, pr, fees)
        ret_frontier[j] = ret_frontier[j].first =>
            (expr, range(rt_min, rt_max; length = front.N), i)
    end
    return ret_frontier
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          ::Val{true}, ::Val{false}, fees::Option{<:Fees},
                          attrs::ProcessedJuMPOptimiserAttributes)
    ret_frontier = compute_ret_lbs(model, mr, pr, fees, attrs)
    ret_axis = set_ret_frontier_parameters!(model, ret_frontier)
    set_portfolio_objective_function!(model, mr.obj, mr, attrs)
    return frontier_sweep!(model, mr, eltype(pr.X), frontier_sweep_axes(ret_axis, nothing))
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
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, idx::VecInt,
                               attrs::ProcessedJuMPOptimiserAttributes)
    X = pr.X
    risk_frontier = copy(risk_frontier)
    set_portfolio_objective_function!(model, MinimumRisk(), mr, attrs)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), mr, attrs)
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
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, ::Any,
                               attrs::ProcessedJuMPOptimiserAttributes)
    X = pr.X
    set_portfolio_objective_function!(model, MinimumRisk(), mr, attrs)
    retcode, sol_min = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("minimum-risk solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    set_portfolio_objective_function!(model, MaximumReturn(), mr, attrs)
    retcode, sol_max = optimise_JuMP_model!(model, mr, eltype(X))
    @argcheck(isa(retcode, OptimisationSuccess),
              ArgumentError("maximum-return solve failed with retcode $retcode"))
    JuMP.unregister(model, :obj_expr)
    r = factory(mr.r, pr, mr.opt.slv)
    return [_rebuild_risk_frontier(pr, fees, r, risk_frontier, sol_min.w, sol_max.w)]
end
"""
    unresolved_risk_frontier(model::JuMP.Model)

Read the risk frontier out of `model` and find the entries that still need a rebuild.

An entry is *resolved* when its bound is already a numeric vector; anything else is a frontier
range that [`rebuild_risk_frontier`](@ref) must turn into numbers. This is the shared half of
every [`compute_risk_ubs`](@ref) method — the methods differ only in the `rebuild_risk_frontier`
call they close with.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model containing `risk_frontier`.

# Returns

  - `(risk_frontier, idx)`: The frontier vector of pairs, and the indices of its unresolved
    entries. An empty `idx` means that no rebuild is needed.

# Related

  - [`compute_risk_ubs`](@ref)
  - [`rebuild_risk_frontier`](@ref)
"""
function unresolved_risk_frontier(model::JuMP.Model)
    risk_frontier = shared_get(model, :risk_frontier)
    idx = Vector{Int}(undef, 0)
    for (i, rkf) in enumerate(risk_frontier)
        if !isa(rkf.second[2], VecNum)
            push!(idx, i)
        end
    end
    return risk_frontier, idx
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
  - [`unresolved_risk_frontier`](@ref)
"""
function compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          fees::Option{<:Fees}, attrs::ProcessedJuMPOptimiserAttributes)
    risk_frontier, idx = unresolved_risk_frontier(model)
    if isempty(idx)
        return risk_frontier
    end
    return rebuild_risk_frontier(model, mr, pr, fees, risk_frontier, idx, attrs)
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          ::Val{false}, ::Val{true}, fees::Option{<:Fees},
                          attrs::ProcessedJuMPOptimiserAttributes)
    risk_frontier = compute_risk_ubs(model, mr, pr, fees, attrs)
    risk_axis = set_risk_frontier_parameters!(model, risk_frontier)
    set_portfolio_objective_function!(model, mr.obj, mr, attrs)
    return frontier_sweep!(model, mr, eltype(pr.X), frontier_sweep_axes(nothing, risk_axis))
end
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          ::Val{true}, ::Val{true}, fees::Option{<:Fees},
                          attrs::ProcessedJuMPOptimiserAttributes)
    ret_frontier = compute_ret_lbs(model, mr, pr, fees, attrs)
    risk_frontier = compute_risk_ubs(model, mr, pr, fees, attrs)
    risk_axis = set_risk_frontier_parameters!(model, risk_frontier)
    ret_axis = set_ret_frontier_parameters!(model, ret_frontier)
    set_portfolio_objective_function!(model, mr.obj, mr, attrs)
    return frontier_sweep!(model, mr, eltype(pr.X),
                           frontier_sweep_axes(ret_axis, risk_axis))
end
function _optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    mr = reset_time_dependent_estimator(mr)
    attrs = processed_jump_optimiser_attributes(mr.opt, rd; dims = dims, kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_maximum_ratio_factor_variables!(model, mr.obj)
    set_w!(model, attrs.pr.X, mr.wi)
    set_weight_constraints!(model, attrs.wb, mr.opt)
    assemble_jump_model!(model, mr, mr.opt, attrs, rd, mr.r, mr.obj)
    retcode, sol = solve_mean_risk!(model, mr, attrs.pr,
                                    Val(shared_has(model, :ret_frontier)),
                                    Val(shared_has(model, :risk_frontier)), attrs.fees,
                                    attrs)
    return MeanRiskResult(;
                          jr = JuMPOptimisationResult(; pa = attrs, retcode = retcode,
                                                      sol = sol,
                                                      model = ifelse(save, model, nothing)),
                          r = factory(mr.r, attrs.pr, mr.opt.slv), fb = nothing)
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

@pipe_delegates MeanRisk opt
@pipe_route_sigma_ucs MeanRisk
export MeanRisk, MeanRiskResult
