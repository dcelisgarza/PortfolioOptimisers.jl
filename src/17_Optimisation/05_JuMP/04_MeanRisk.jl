"""
$(DocStringExtensions.TYPEDEF)

Holds the solution, the resolved risk measures and the fallback chain of a Mean-Risk optimisation.

[`optimise`](@ref) returns it for a [`MeanRisk`](@ref). A frontier sweep gives one result that holds one weight vector and one return code for each point of the sweep. The property `w` and every name that is not a field resolve through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MeanRiskResult(;
        jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM, fb::Option{<:OptE_Opt_FbChain}
    ) -> MeanRiskResult

Keywords correspond to the struct's fields.

# Related

  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
  - [`MeanRisk`](@ref)
"""
@concrete struct MeanRiskResult <: RiskJuMPOptimisationResult
    """
    The [`JuMPOptimisationResult`](@ref) that holds the processed attributes, the return code, the solution and the model.
    """
    jr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:fb_res])
    """
    fb
    function MeanRiskResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                            fb::Option{<:OptE_Opt_FbChain})
        return new{typeof(jr), typeof(r), typeof(fb)}(jr, r, fb)
    end
end
function MeanRiskResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                        fb::Option{<:OptE_Opt_FbChain})::MeanRiskResult
    return MeanRiskResult(jr, r, fb)
end
"""
    set_retcode(res::MeanRiskResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`MeanRiskResult`](@ref) with a different return code.

`retcode` is not a field of this result. It resolves through `jr`, the [`JuMPOptimisationResult`](@ref) that the result holds. So the method rebuilds `jr` with the new return code, and keeps `r` and `fb` unchanged.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`MeanRiskResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`MeanRiskResult`](@ref)
"""
function set_retcode(res::MeanRiskResult, retcode::OptRetCode_VecOptRetCode)
    return MeanRiskResult(set_retcode(res.jr, retcode), res.r, res.fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`MeanRisk`](@ref) fields that can hold a [`TimeDependent`](@ref).

The constructor reads these defaults when it checks each schedule with [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) returns them. So the value of each field outside a fold loop is declared once. A field whose static default is `nothing` is not in the result.

# Returns

  - `(; r = Variance(), obj = MinimumRisk())`.

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

Optimises one objective of risk and return over the portfolios that the constraints of a JuMP model admit.

`obj` selects the objective: [`MinimumRisk`](@ref), [`MaximumReturn`](@ref), [`MaximumUtility`](@ref) or [`MaximumRatio`](@ref). `r` gives the risk measures, and `opt` gives the prior, the return terms, the solver and every other constraint. A risk measure whose `settings.ub` is a [`Frontier`](@ref), or a return term whose `settings.lb` is one, makes the optimisation sweep the efficient frontier. The result then holds one portfolio for each point of the sweep.

# Mathematical definition

`obj` selects one of four programmes over the same feasible set:

```math
\\begin{align}
\\text{MinimumRisk:} &\\quad \\underset{\\boldsymbol{w}}{\\min}\\; R(\\boldsymbol{w}) \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,, \\\\
\\text{MaximumReturn:} &\\quad \\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret} \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,, \\\\
\\text{MaximumUtility:} &\\quad \\underset{\\boldsymbol{w}}{\\max}\\; \\mathrm{ret} - l\\, R(\\boldsymbol{w}) \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,, \\\\
\\text{MaximumRatio:} &\\quad \\underset{\\boldsymbol{w}}{\\max}\\; \\frac{\\mathrm{ret} - r_f}{R(\\boldsymbol{w})} \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

``\\mathcal{W}`` holds the ceilings ``R_j(\\boldsymbol{w}) \\leq u_j`` of the risk measures and the floors ``\\mathrm{ret}_i \\geq b_i`` of the return terms. The model solves the ratio in homogenised weights, and [`MaximumRatio`](@ref) states that programme and the degree of the risk that it reads.

A [`Frontier`](@ref) of ``M`` points in the place of a ceiling ``u_j`` or of a floor ``b_i`` gives ``M`` evenly spaced levels between two corner portfolios:

```math
\\begin{align}
u_{j,m} &= c_j \\left( g_j(R_j(\\boldsymbol{w}^{-})) + \\frac{m - 1}{M - 1} \\left( g_j(R_j(\\boldsymbol{w}^{+})) - g_j(R_j(\\boldsymbol{w}^{-})) \\right) \\right)\\,, \\\\
b_{i,m} &= \\mathrm{ret}_i(\\boldsymbol{w}^{-}) + \\frac{m - 1}{M - 1} \\left( \\mathrm{ret}_i(\\boldsymbol{w}^{+}_{i}) - \\mathrm{ret}_i(\\boldsymbol{w}^{-}) \\right)\\,, \\\\
\\boldsymbol{w}^{-} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}_0}{\\arg\\min}\\; R(\\boldsymbol{w})\\,, \\\\
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}_0}{\\arg\\max}\\; \\mathrm{ret}\\,, \\\\
\\boldsymbol{w}^{+}_{i} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}_0}{\\arg\\max}\\; \\mathrm{ret}_i\\,.
\\end{align}
```

The two ends of a ceiling change places when the first end is the larger, so the levels always ascend. The model solves the programme of `obj` once at each combination of levels, so two frontiers of ``M`` points each give ``M^2`` portfolios.

Under [`MinimumRisk`](@ref) with a frontier on a return floor, each point has the minimum risk for its return or more. Under [`MaximumReturn`](@ref) with a frontier on a risk ceiling, each point has the maximum return for its risk or less. With the variance as the risk, these are the two equal definitions of the efficient set on page 82 of [markowitz1952](@cite). Page 87 of the same work states that the efficient set runs from the portfolio of minimum variance to the portfolio of maximum expected return. These two portfolios are the corners ``\\boldsymbol{w}^{-}`` and ``\\boldsymbol{w}^{+}``. Section 8.2 of [cajas2025](@cite) states the four programmes for a general convex risk measure.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:W_feas])
  - $(math_dict[:R_w]) With many measures it is the scalarised risk of the model.
  - $(math_dict[:ret_model])
  - $(math_dict[:ret_i_term])
  - $(math_dict[:l_utility])
  - $(math_dict[:r_f_ratio])
  - ``R_j(\\boldsymbol{w})``, ``u_j``: Risk of measure ``j``, and its ceiling `settings.ub`.
  - ``b_i``: Floor of return term ``i``, its `settings.lb`.
  - ``M``: The `N` field of the [`Frontier`](@ref).
  - ``g_j``, ``c_j``: The transform and the multiplier that the formulation of measure ``j`` writes into the `bound` and the `factor` fields of the [`Frontier`](@ref). The transform is the identity, the square root or the square, see [`FrontierBoundEstimator`](@ref).
  - ``\\mathcal{W}_0``: The feasible set ``\\mathcal{W}`` without the ceilings and the floors that hold a [`Frontier`](@ref).
  - ``\\boldsymbol{w}^{-}``, ``\\boldsymbol{w}^{+}``, ``\\boldsymbol{w}^{+}_{i}``: The corner portfolios of minimum risk, of maximum return, and of maximum return of term ``i``.

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

Keywords correspond to the struct's fields. A field of the type [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. The risk measures, the objective, the initial weights and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. An `optimise` outside a fold loop uses the static default of each field. The default of `wi` and of `fb` is `nothing`. So a scheduled fallback has no effect outside a fold loop, unless the schedule holds a `default`.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `wi` is provided: `!isempty(wi)`.
  - A schedule in `fb`: `bind !== :nearest`.
  - The constructor refuses a risk expression that is identically zero under [`MinimumRisk`](@ref) and under [`MaximumRatio`](@ref). Every feasible portfolio is optimal under the first, and the second is unbounded. A [`NoRisk`](@ref) measure gives such an expression, and so does `settings.rke = false` on every measure.
  - If `r` holds a [`DataBenchmarkTracking`](@ref): `opt.pe` must not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - A part that states no source and reads the raw returns, beside a part that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The constructor then builds the estimator as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `opt`: Recursively updated through [`factory`](@ref).
  - `r`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

`MeanRisk` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method uses `opt.pe.X` in place of `X`. So the views of the children read the observations of the prior, and not the matrix of the caller.
  - `opt` and `r` recurse through [`port_opt_view`](@ref) with that matrix.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - `wi` is sliced to the selected assets.
  - `obj` passes through unchanged.

# Examples

```jldoctest
julia> MeanRisk(; opt = JuMPOptimiser(; slv = Solver(; solver = nothing)))
MeanRisk
  opt ┼ JuMPOptimiser
      │        pe ┼ EmpiricalPrior
      │           │           ce ┼ PortfolioOptimisersCovariance
      │           │              │   ce ┼ Covariance
      │           │              │      │    me ┼ SimpleExpectedReturns
      │           │              │      │       │   w ┴ nothing
      │           │              │      │    ce ┼ GeneralCovariance
      │           │              │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │           │              │      │       │    w ┴ nothing
      │           │              │      │   alg ┼ FullMoment()
      │           │              │      │     w ┴ nothing
      │           │              │   mp ┼ MatrixProcessing
      │           │              │      │     pdm ┼ Posdef
      │           │              │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │           │              │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │           │              │      │      dn ┼ nothing
      │           │              │      │      dt ┼ nothing
      │           │              │      │     alg ┼ nothing
      │           │              │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
      │           │           me ┼ SimpleExpectedReturns
      │           │              │   w ┴ nothing
      │           │      horizon ┼ nothing
      │           │   fill_limit ┼ nothing
      │           │        x_src ┴ nothing
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
      │           │         mu ┼ nothing
      │           │   mtx_sqrt ┴ EigenFallbackSquareRoot()
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
      │        lp ┼ nothing
      │      linf ┼ nothing
      │      xbgt ┼ Bool: false
      │    strict ┴ Bool: false
    r ┼ Variance
      │    settings ┼ RiskMeasureSettings
      │             │   scale ┼ Float64: 1.0
      │             │      ub ┼ nothing
      │             │     rke ┴ Bool: true
      │       sigma ┼ nothing
      │        chol ┼ nothing
      │          rc ┼ nothing
      │         alg ┼ SquaredSOCRiskExpr()
      │    mtx_sqrt ┼ EigenFallbackSquareRoot()
      │   ratio_alg ┼ SquareRootRatio()
      │      corner ┴ nothing
  obj ┼ MinimumRisk()
   wi ┼ nothing
   fb ┴ nothing
```

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
  - [`Frontier`](@ref)
  - [`compute_ret_lbs`](@ref): It resolves a frontier on a return floor.
  - [`rebuild_risk_frontier`](@ref): It resolves a frontier on a risk ceiling.
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:markowitz1952]) Pages 82 and 87.
  - $(ref_dict[:cajas2025]) Section 8.2, Equations 8.7 and 8.10. Equations 8.12 and 8.13 for a general risk measure. $(ref_dict[:no_original_source])
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
        assert_data_benchmark_prior(r, opt.pe, :r, :MeanRisk, "opt.pe")
        est = new{typeof(opt), typeof(r), typeof(obj), typeof(wi), typeof(fb)}(opt, r, obj,
                                                                               wi, fb)
        warn_raw_readers(est)
        return est
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
# `solve_mean_risk!` sweeps every frontier of the model.
function assert_frontier_sweep_owner(::JuMP.Model, ::MeanRisk)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when a part of `opt` needs the previous portfolio weights.

The parts are the schedules of the time-dependent fields, the [`JuMPOptimiser`](@ref) `opt.opt`, the risk measures `opt.r` and the fallback `opt.fb`. A turnover measure or a fee on the turnover is a part that needs them.

# Returns

  - `Bool`: `true` when any part answers `true`.

# Related

  - [`needs_previous_weights`](@ref)
  - [`time_dependent_fields`](@ref)
  - [`MeanRisk`](@ref)
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

Return the view of a [`MeanRisk`](@ref) at the assets `i`.

The `## View parameters` subsection of [`MeanRisk`](@ref) states how the method takes the view of each field.

# Arguments

  - `mr`: The optimiser.
  - `i`: Indices of the assets that the view keeps.
  - `X::MatNum`: Returns matrix of the caller. The method uses the matrix of the prior in its place when `mr.opt.pe` holds a prior result.
  - `args...`: Not read.

# Returns

  - `MeanRisk`: A new optimiser, built through the keyword constructor.

# Related

  - [`port_opt_view`](@ref)
  - [`view_child`](@ref)
  - [`MeanRisk`](@ref)
"""
function port_opt_view(mr::MeanRisk, i, X::MatNum, args...)::MeanRisk
    X = isa(mr.opt.pe, AbstractPriorResult) ? mr.opt.pe.X : X
    opt = port_opt_view(mr.opt, i, X)
    r = port_opt_view(mr.r, i, X)
    wi = nothing_scalar_array_view(mr.wi, i)
    return MeanRisk(; opt = opt, r = r, obj = mr.obj, wi = wi, fb = view_child(mr.fb, i, X))
end
"""
    solve_mean_risk!(model, mr, pr, ::Val{false}, ::Val{false}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{true}, ::Val{false}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{false}, ::Val{true}, fees, attrs)
    solve_mean_risk!(model, mr, pr, ::Val{true}, ::Val{true}, fees, attrs)

Solve the model of a [`MeanRisk`](@ref) once, or once for each point of its frontier sweep.

The first `Val` is `true` when the model holds a `:ret_frontier` registry, and the second is `true` when it holds a `:risk_frontier` registry. Each method reads the return terms from `attrs.ret`.

# Algorithm

 1. With no frontier, set the objective of `mr.obj` with [`set_portfolio_objective_function!`](@ref), and solve once with [`optimise_JuMP_model!`](@ref).
 2. With a return frontier, resolve its spans with [`compute_ret_lbs`](@ref), giving `ret_frontier`.
 3. With a risk frontier, resolve its spans with [`compute_risk_ubs`](@ref), giving `risk_frontier`. Steps 2 and 3 both run before step 4, so every corner solve reads the model without a frontier row.
 4. Register one parameter and one ceiling row for each entry of `risk_frontier` with [`set_risk_frontier_parameters!`](@ref), giving `risk_axis`. Register one parameter and one floor row for each entry of `ret_frontier` with [`set_ret_frontier_parameters!`](@ref), giving `ret_axis`.
 5. Set the objective of `mr.obj`, and solve once at each point of the product of the axes with [`frontier_sweep!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `mr::MeanRisk`: The optimiser.
  - $(arg_dict[:pr])
  - `::Val{Bool}`: `true` for a return frontier.
  - `::Val{Bool}`: `true` for a risk frontier.
  - $(arg_dict[:fees_opt])
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser, which hold the return terms.

# Returns

  - `(retcode, sol)`: One return code and one solution without a frontier, and one of each for each point of the sweep with a frontier.

# Related

  - [`MeanRisk`](@ref)
  - [`compute_ret_lbs`](@ref)
  - [`compute_risk_ubs`](@ref)
  - [`frontier_sweep!`](@ref)
"""
function solve_mean_risk!(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          ::Val{false}, ::Val{false}, ::Option{<:Fees},
                          attrs::ProcessedJuMPOptimiserAttributes)
    set_portfolio_objective_function!(model, mr.obj, mr, attrs)
    return optimise_JuMP_model!(model, mr, eltype(pr.X))
end
"""
    return_term(ret, i)

Return the return term `i` of the `ret` field of an optimiser.

The field holds one term or a vector of terms. A caller that computes the value of a term at a weight vector, such as [`compute_ret_lbs`](@ref), reads the term through this function, so it does not test the form of the field. One term answers every index, because a configuration of one term has the index `1` alone.

# Returns

  - `ret` itself when it is one term, and `ret[i]` when it is a vector.

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

Resolve the sweep points of the return frontier, one span for each swept return term.

An entry whose bound is a vector of numbers keeps it. An entry whose bound is a [`Frontier`](@ref) gets the span of a return floor that the `# Mathematical definition` of [`MeanRisk`](@ref) states. A frontier of `k` swept terms costs `k + 1` corner solves: one minimum-risk solve for the low ends, and one solve for the high end of each term.

The high end of term `i` comes from a portfolio that maximises term `i` alone. The value of term `i` at the portfolio that maximises the sum of the terms depends on the `scale` of the other terms. That value can be less than its value at the minimum-risk portfolio, and then the span descends. The portfolio that maximises term `i` alone is at least as good for term `i` as every other portfolio of the same feasible set, so its span ascends.

# Algorithm

 1. Read the `:ret_frontier` registry from the model, giving `ret_frontier`. Collect the positions of the entries whose bound is not a vector of numbers, giving `idx`.
 2. When `idx` is empty, return `ret_frontier`.
 3. Set the objective [`MinimumRisk`](@ref) and solve, giving `sol_min`. Unregister the objective expression.
 4. For each position `j` of `idx`, read the term index `i` of the entry. Set the objective [`MaximumElementReturn`](@ref)`(i)` and solve, giving `sol_max`. Unregister the objective expression.
 5. Compute the value of term `i` with [`expected_return`](@ref) at `sol_min.w` and at `sol_max.w`, giving `rt_min` and `rt_max`.
 6. Replace the bound of entry `j` with `range(rt_min, rt_max; length = N)`, with the `N` of its `Frontier`.

# Arguments

  - $(arg_dict[:model])
  - `mr::MeanRisk`: The optimiser.
  - $(arg_dict[:pr])
  - $(arg_dict[:fees_opt])
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser, which hold the return terms.

# Validation

  - Each corner solve must succeed. Otherwise an `ArgumentError` names the solve and its return code, because a span from a failed solve is not a span.

# Returns

  - A copy of the registry, with the bound of each entry of `idx` resolved.

# Related

  - [`MeanRisk`](@ref)
  - [`MaximumElementReturn`](@ref)
  - [`return_term`](@ref)
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
    frontier_risk(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum, fees::Option{<:Fees})
    frontier_risk(r::Skewness, w::VecNum, X::MatNum, fees::Option{<:Fees})
    frontier_risk(r::Kurtosis, w::VecNum, X::MatNum, fees::Option{<:Fees})

Value the owner `r` of a `:risk_frontier` entry at the weights `w`, in the units of the expression that the entry bounds.

[`_rebuild_risk_frontier`](@ref) values each corner portfolio with this function. The bounded expression of most measures holds the value of the measure, so the generic method returns [`expected_risk`](@ref). The models of [`Skewness`](@ref) and [`Kurtosis`](@ref) read a comoment matrix, but their functors read the returns, so the two methods for these measures read the matrix that the model reads.

  - `Skewness`: the `lb` bounds the third moment that [`VarianceSkewKurtosis`](@ref) reads off `r.sk`, and the functor gives the standardised skewness. The method returns [`portfolio_third_moment`](@ref).
  - `Kurtosis`: the method returns [`portfolio_fourth_moment`](@ref) of `r.kt` in the units of the functor. These are the square root of the moment under [`SOCRiskExpr`](@ref), and the moment under the other formulations. The functor gives the same value only when `r.kt` is the sample cokurtosis about the Moment Target of the measure. With an integer `r.N`, the model bounds the approximate moment of [`cokurtosis_spectrum`](@ref), which lies at or below the moment of the matrix, so the span is the span of the full moment.

# Arguments

  - `r`: The owner of the entry, already through [`factory`](@ref). The `sk` of a `Skewness` and the `kt` of a `Kurtosis` must be matrices.
  - $(arg_dict[:pw])
  - `X::MatNum`: Returns matrix `observations × assets`.
  - $(arg_dict[:fees_opt])

# Returns

  - `rk::Number`: The value of the bounded expression at `w`.

# Related

  - [`_rebuild_risk_frontier`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`expected_risk`](@ref)
"""
function frontier_risk(r::AbstractBaseRiskMeasure, w::VecNum, X::MatNum,
                       fees::Option{<:Fees})
    return expected_risk(r, w, X, fees)
end
function frontier_risk(r::Skewness, w::VecNum, ::MatNum, ::Option{<:Fees})
    return portfolio_third_moment(r.sk, w)
end
function frontier_risk(r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:SOCRiskExpr},
                       w::VecNum, ::MatNum, ::Option{<:Fees})
    return sqrt(portfolio_fourth_moment(r.kt, w))
end
function frontier_risk(r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                   <:QuadSecondMomentFormulations}, w::VecNum, ::MatNum,
                       ::Option{<:Fees})
    return portfolio_fourth_moment(r.kt, w)
end
"""
    _rebuild_risk_frontier(pr::AbstractPriorResult, fees::Option{<:Fees},
                           r::AbstractBaseRiskMeasure, risk_frontier::VecPair,
                           w_min::VecNum, w_max::VecNum, i::Integer = 1)

Resolve one [`Frontier`](@ref) risk bound into the span of numbers that the sweep writes.

The span is the span of a risk ceiling that the `# Mathematical definition` of [`MeanRisk`](@ref) states, and `r` measures the risk of each corner portfolio. The two ends are sorted, so the span ascends for every measure. The risk of a measure in a vector can be larger at the minimum-risk portfolio than at the maximum-return portfolio, because the minimum-risk solve minimises the scalarised risk of all the measures.

# Algorithm

 1. Read the expression, the `Frontier`, the polarity and the owner of entry `i`.
 2. Compute the risk of `r` at `w_min` and at `w_max` with [`frontier_risk`](@ref), in the units of the bounded expression.
 3. Transform each risk with [`variance_risk_bounds_val`](@ref) and the `bound` of the `Frontier`, and multiply it by `factor`, giving the two ends.
 4. Sort the two ends with `minmax`, and divide the interval into `N` levels with `range`, giving `ub`.

# Arguments

  - $(arg_dict[:pr])
  - $(arg_dict[:fees_opt])
  - `r::AbstractBaseRiskMeasure`: The measure that owns entry `i`, already through [`factory`](@ref).
  - `risk_frontier::VecPair`: The `:risk_frontier` registry.
  - `w_min::VecNum`: Weights of the minimum-risk corner portfolio.
  - `w_max::VecNum`: Weights of the maximum-return corner portfolio.
  - `i::Integer = 1`: Position of the entry in the registry.

# Returns

  - The rebuilt entry. Its keys, its expression, its polarity and its owner are those of entry `i`, and only the bound changes. Each part comes from position `i`. An expression from position `1` bounds the first measure twice, and leaves the measure of entry `i` with no bound.

# Related

  - [`rebuild_risk_frontier`](@ref)
  - [`Frontier`](@ref)
  - [`FrontierBoundEstimator`](@ref)
  - [`MeanRisk`](@ref)
"""
function _rebuild_risk_frontier(pr::AbstractPriorResult, fees::Option{<:Fees},
                                r::AbstractBaseRiskMeasure, risk_frontier::VecPair,
                                w_min::VecNum, w_max::VecNum, i::Integer = 1)
    (expr, front, flag, owner) = risk_frontier[i].second
    (; N, factor, bound) = front
    X = pr.X
    rk_a = factor * variance_risk_bounds_val(bound, frontier_risk(r, w_min, X, fees))
    rk_b = factor * variance_risk_bounds_val(bound, frontier_risk(r, w_max, X, fees))
    ub = range(minmax(rk_a, rk_b)...; length = N)
    return risk_frontier[i].first => (expr, ub, flag, owner)
end
"""
    rebuild_risk_frontier(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                          fees::Option{<:Fees}, risk_frontier::VecPair, idx, attrs)

Resolve every unresolved [`Frontier`](@ref) risk bound into a span of numbers.

Two corner portfolios bound every span, and the method solves each of them once. [`_rebuild_risk_frontier`](@ref) then builds the span of each entry from the measure that owns the entry and from the `Frontier` of the entry.

# Algorithm

 1. Copy the registry, giving `risk_frontier`.
 2. Set the objective [`MinimumRisk`](@ref) and solve, giving `sol_min`. Unregister the objective expression.
 3. Set the objective [`MaximumReturn`](@ref) and solve, giving `sol_max`. Unregister the objective expression.
 4. For each position `i` of `idx`, resolve the owner of entry `i` against the prior and the solver with [`factory`](@ref), giving `r`. [`set_risk_frontier_owner!`](@ref) wrote the owner during Model Assembly.
 5. Replace entry `i` with [`_rebuild_risk_frontier`](@ref) of `r` at `sol_min.w` and `sol_max.w`.

# Arguments

  - $(arg_dict[:model])
  - `mr::MeanRisk`: The optimiser.
  - $(arg_dict[:pr])
  - $(arg_dict[:fees_opt])
  - `risk_frontier::VecPair`: The `:risk_frontier` registry.
  - `idx::VecInt`: Registry positions that need a rebuild, as [`unresolved_risk_frontier`](@ref) returns them.
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser.

# Validation

  - Each corner solve must succeed. Otherwise an `ArgumentError` names the solve and its return code, because a span from a failed solve is not a span.

# Returns

  - A copy of the registry, with every entry of `idx` resolved and every other entry unchanged.

# Related

  - [`MeanRisk`](@ref)
  - [`_rebuild_risk_frontier`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`compute_risk_ubs`](@ref)
"""
function rebuild_risk_frontier(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                               fees::Option{<:Fees}, risk_frontier::VecPair, idx::VecInt,
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
    for i in idx
        r = factory(risk_frontier[i].second[4], pr, mr.opt.slv)
        risk_frontier[i] = _rebuild_risk_frontier(pr, fees, r, risk_frontier, sol_min.w,
                                                  sol_max.w, i)
    end
    return risk_frontier
end
"""
    unresolved_risk_frontier(model::JuMP.Model)

Read the risk frontier out of `model` and find the entries that still need a rebuild.

An entry is resolved when its bound is a vector of numbers. Every other entry holds a [`Frontier`](@ref), which [`rebuild_risk_frontier`](@ref) turns into numbers. Every method of [`compute_risk_ubs`](@ref) calls this function first, and the methods differ only in the call to `rebuild_risk_frontier` that follows.

# Arguments

  - `model::JuMP.Model`: The model, which holds the `:risk_frontier` registry.

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
    compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, pr::AbstractPriorResult,
                     fees::Option{<:Fees}, attrs::ProcessedJuMPOptimiserAttributes)

Resolve the sweep points of the risk frontier of the model.

[`NearOptimalCentering`](@ref) has a method of its own, which takes the two corner portfolios that its setup already solved.

# Algorithm

 1. Read the registry and the positions of its unresolved entries with [`unresolved_risk_frontier`](@ref), giving `risk_frontier` and `idx`.
 2. When `idx` is empty, return `risk_frontier`.
 3. Otherwise resolve the entries of `idx` with [`rebuild_risk_frontier`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `mr::MeanRisk`: The optimiser.
  - $(arg_dict[:pr])
  - $(arg_dict[:fees_opt])
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser.

# Returns

  - The registry, with the bound of every entry resolved into a vector of numbers.

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
"""
    _optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult(); str_names::Bool = false,
              save::Bool = true, kwargs...) -> MeanRiskResult

Build the model of a [`MeanRisk`](@ref), solve it, and return its result.

[`optimise`](@ref) reaches this method through [`shared_series_solve`](@ref), which drops the observations of a Shared Series gap before the solve.

# Algorithm

 1. Replace each schedule of `mr` with its static default through [`reset_time_dependent_estimator`](@ref).
 2. Process the attributes of `mr.opt` against `rd` with [`processed_jump_optimiser_attributes`](@ref), giving `attrs`. It holds the prior, the weight bounds, the fees, the return terms and the Investable Mask.
 3. Take the view of `mr` and of `rd` at the investable assets with [`investable_view`](@ref). Both are unchanged when every asset is investable.
 4. Bind the benchmark of each [`DataBenchmarkTracking`](@ref) measure to `rd` with [`bind_data_benchmark`](@ref).
 5. Build a new model. Register the scales with [`set_model_scales!`](@ref), the number of observations with [`set_model_observations!`](@ref), the homogenisation variables of a ratio with [`set_maximum_ratio_factor_variables!`](@ref), the corner of a lifted variance with [`set_sdp_lift_corner!`](@ref), the weights with [`set_w!`](@ref) and the weight bounds with [`set_weight_constraints!`](@ref).
 6. Add the constraints, the return terms and the risk measures with [`assemble_jump_model!`](@ref).
 7. Solve with [`solve_mean_risk!`](@ref), which sweeps a frontier when the model holds one, giving `retcode` and `sol`.
 8. Build the result. It holds `attrs`, `retcode`, `sol`, the model when `save` is `true`, and the risk measures resolved against the prior.

# Arguments

  - `mr::MeanRisk`: The optimiser.
  - $(arg_dict[:rd])
  - `str_names::Bool`: `true` gives each variable and each row of the model its string name.
  - `save::Bool`: `true` keeps the model in the result.
  - `kwargs...`: Passed on to [`processed_jump_optimiser_attributes`](@ref).

# Returns

  - [`MeanRiskResult`](@ref): The result, with `fb = nothing`. The method of [`optimise`](@ref) for an optimiser with a fallback runs the fallback chain.

# Related

  - [`optimise`](@ref)
  - [`solve_mean_risk!`](@ref)
  - [`MeanRiskResult`](@ref)
"""
function _optimise(mr::MeanRisk, rd::ReturnsResult = ReturnsResult();
                   str_names::Bool = false, save::Bool = true, kwargs...)
    mr = reset_time_dependent_estimator(mr)
    attrs = processed_jump_optimiser_attributes(mr.opt, rd; kwargs...)
    # The bundle reduced what it carries. The head carries the rest — an initial weight
    # vector, a risk measure holding per-asset data, tracking, a custom term — and hands
    # them to `assemble_jump_model!` itself, so it takes the same view of itself and of
    # `rd`. Both are unchanged when every asset is investable.
    mr, rd = investable_view(mr, rd, attrs.pr, attrs.imsk)
    mr = bind_data_benchmark(mr, rd)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, mr.opt.sc, mr.opt.so)
    set_model_observations!(model, size(attrs.pr.X, 1))
    set_maximum_ratio_factor_variables!(model, mr.obj)
    set_sdp_lift_corner!(model, mr.obj, attrs.pr)
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
             rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...) -> MeanRiskResult

Optimise a [`MeanRisk`](@ref) that has no fallback, and return its result.

This method serves an optimiser whose `fb` is `nothing`. It solves through [`shared_series_solve`](@ref), and [`_optimise`](@ref) states the steps of the solve.

# Arguments

  - `mr`: The optimiser.
  - $(arg_dict[:rd]) When `mr.opt.pe` is a prior result, the solve does not read `rd`. A fallback or a clustering step can still read it.
  - `str_names`: `true` gives each variable and each row of the model its string name.
  - `save`: `true` keeps the model in the result.
  - `kwargs`: Passed on to [`_optimise`](@ref).

# Validation

  - No field in the tree of `mr` holds an [`Online`](@ref). Otherwise [`assert_batch_entry`](@ref) throws an `ArgumentError` that names the field. A plain `optimise` is a batch fit, and the fold loop resolves an `Online` only in its online branch.

# Returns

  - [`MeanRiskResult`](@ref): The result of the solve, or of the frontier sweep.

# Related

  - [`MeanRisk`](@ref)
  - [`MeanRiskResult`](@ref)
  - [`_optimise`](@ref)
"""
function optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult;
                  str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(mr, "`optimise`")
    return shared_series_solve(mr, rd; str_names = str_names, save = save, kwargs...)
end

@pipe_delegates MeanRisk opt
@pipe_route_sigma_ucs MeanRisk
export MeanRisk, MeanRiskResult
