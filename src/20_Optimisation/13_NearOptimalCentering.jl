"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for Near Optimal Centering algorithm variants.

# Related Types

  - [`ConstrainedNearOptimalCentering`](@ref)
  - [`UnconstrainedNearOptimalCentering`](@ref)
"""
abstract type NearOptimalCenteringAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Constrained Near Optimal Centering algorithm.

Centres inside the feasible region the portfolio constraints define. Its middle is the
shared [`assemble_jump_model!`](@ref), so every [`JuMPOptimiser`](@ref) setting reaches the
centring model. Use this variant when a setting must bind on the centring solve itself. See
[`UnconstrainedNearOptimalCentering`](@ref) for what the default variant does not read.

# Related Types

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`UnconstrainedNearOptimalCentering`](@ref)

# Related

  - [`assemble_near_optimal_centering_model!`](@ref)
"""
struct ConstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Unconstrained Near Optimal Centering algorithm. This is the default `alg` of
[`NearOptimalCentering`](@ref).

Centres inside the near-optimal region of the *unconstrained* problem. The centring model
carries the weight bounds and the budgets its head applies, the risk expression, the return
expression and the non-fixed fees, and nothing else — the constraint and penalty builders of
the shared middle do not run (ADR 0008, amendment 2). "Unconstrained" names that omission.

The omitted settings are **carried and validated, not rejected**. They are not inert: the
three anchor portfolios are solved as [`MeanRisk`](@ref) sub-problems that do run the whole
middle, so an omitted setting still shapes the anchors and, through them, the centring
target. The one configuration in which such a setting reaches no model at all is `w_min`,
`w_opt` and `w_max` all supplied, because then no sub-problem is solved.

# Settings the centring model reads

`pe`, `slv`, `wb`, `bgt`, `sbgt`, `gbgt`, `sc`, `so`, `sca`, `ret` — the bound-free copy,
see [`no_bounds_optimiser`](@ref) — and the non-fixed part of `fees`.

# Settings the centring model does not read

`lcse`, `cte`, `card`, `gcarde`, `scard`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`,
`sglt`, `sgst`, `lt`, `st`, `xbgt`, `ss`, `tn`, `tr`, `ple`, `l2c`, `lpc`, `linfc`, `l1`,
`l2`, `linf`, `lp`, `ccnt`, `cobj`, and the fixed part of `fees` — a fixed fee is charged
per position held, so it needs the cardinality binaries `set_mip_constraints!` produces.

`cobj` is the one omission a user can observe on the objective rather than on the feasible
set: [`set_near_optimal_objective_function!`](@ref) folds no Objective Penalty into the
barrier, so a Custom Objective Term prices the anchor sub-problems, not the centring solve.

# Related Types

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`ConstrainedNearOptimalCentering`](@ref)

# Related

  - [`assemble_near_optimal_centering_model!`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
struct UnconstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Result type for Near Optimal Centering portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

Property access delegates to the embedded [`JuMPOptimisationResult`](@ref): the unique retcodes resolve directly, the virtual `:w` and unknown properties resolve through `jr`.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`RiskJuMPOptimisationResult`](@ref)
  - [`JuMPOptimisationResult`](@ref)
"""
@concrete struct NearOptimalCenteringResult <: RiskJuMPOptimisationResult
    """
    Shared JuMP result core, see [`JuMPOptimisationResult`](@ref).
    """
    jr
    """
    $(field_dict[:r_res])
    """
    r
    """
    $(field_dict[:w_min_retcode])
    """
    w_min_retcode
    """
    $(field_dict[:w_opt_retcode])
    """
    w_opt_retcode
    """
    $(field_dict[:w_max_retcode])
    """
    w_max_retcode
    """
    $(field_dict[:noc_retcode])
    """
    noc_retcode
    """
    $(field_dict[:fb])
    """
    fb
    function NearOptimalCenteringResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                        w_min_retcode::OptimisationReturnCode,
                                        w_opt_retcode::OptRetCode_VecOptRetCode,
                                        w_max_retcode::OptimisationReturnCode,
                                        noc_retcode::OptRetCode_VecOptRetCode,
                                        fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(r), typeof(w_min_retcode), typeof(w_opt_retcode),
                   typeof(w_max_retcode), typeof(noc_retcode), typeof(fb)}(jr, r,
                                                                           w_min_retcode,
                                                                           w_opt_retcode,
                                                                           w_max_retcode,
                                                                           noc_retcode, fb)
    end
end
function NearOptimalCenteringResult(; jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                    w_min_retcode::OptimisationReturnCode,
                                    w_opt_retcode::OptRetCode_VecOptRetCode,
                                    w_max_retcode::OptimisationReturnCode,
                                    noc_retcode::OptRetCode_VecOptRetCode,
                                    fb::Option{<:OptE_Opt})::NearOptimalCenteringResult
    return NearOptimalCenteringResult(jr, r, w_min_retcode, w_opt_retcode, w_max_retcode,
                                      noc_retcode, fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`NearOptimalCentering`](@ref) fields that may hold a [`TimeDependent`](@ref).

Shared by the constructor's test-substitution pass and [`time_dependent_field_defaults`](@ref), so the fold-less value of a field is declared once. Fields whose static default is `nothing` are omitted.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`time_dependent_field_defaults`](@ref)
  - [`assert_time_dependent_substitution`](@ref)
"""
function near_optimal_centering_td_defaults()::NamedTuple
    return (; r = StandardDeviation(), obj = MinimumRisk())
end
"""
$(DocStringExtensions.TYPEDEF)

Near Optimal Centering (NOC) portfolio optimiser.

`NearOptimalCentering` finds a portfolio that is centrally located within the region of near-optimal solutions. It first solves the minimum-risk, maximum-risk, and user-specified optimal-objective sub-problems, then maximises the minimum distance to the efficient frontier boundaries, yielding a portfolio that is robust to small perturbations in risk-return space.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NearOptimalCentering(;
        opt::JuMPOptimiser,
        r::TD{<:RM_VecRM} = StandardDeviation(),
        obj::TD_Option{<:ObjectiveFunction} = MinimumRisk(),
        bins::Option{<:Number} = nothing,
        w_min::TD_Option{<:VecNum} = nothing,
        w_min_ini::TD_Option{<:VecNum} = nothing,
        w_opt::TD_Option{<:VecNum_VecVecNum} = nothing,
        w_opt_ini::TD_Option{<:VecNum_VecVecNum} = nothing,
        w_max::TD_Option{<:VecNum} = nothing,
        w_max_ini::TD_Option{<:VecNum} = nothing,
        ucs_flag::Bool = true,
        alg::NearOptimalCenteringAlgorithm = UnconstrainedNearOptimalCentering(),
        fb::TDO_Option{<:OptE_Opt} = nothing
    ) -> NearOptimalCentering

Keywords correspond to the struct's fields. Fields typed [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) may hold a [`TimeDependent`](@ref) per-fold schedule instead of a static value: the risk measure, objective, anchor/warm-start weights and fallback are problem definition, so a cross-validation fold loop resolves them per fold, and a fold-less `optimise` runs with each at its static default. `bins`, `ucs_flag` and the `alg` formulation variant are execution control and stay static.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `w_min` is a vector: `!isempty(w_min)`.
  - If `w_min_ini` is a vector: `!isempty(w_min_ini)`.
  - If `w_opt` is a vector: `!isempty(w_opt)`.
  - If `w_opt_ini` is a vector: `!isempty(w_opt_ini)`.
  - If `w_max` is a vector: `!isempty(w_max)`.
  - If `w_max_ini` is a vector: `!isempty(w_max_ini)`.
  - If `bins` is a number: `isfinite(bins) && bins > 0`.
  - `fb` schedules: `bind !== :nearest`.

# Mathematical definition

Let ``\\boldsymbol{w}_{\\min}``, ``\\boldsymbol{w}_{\\mathrm{opt}}``, and ``\\boldsymbol{w}_{\\max}`` be the minimum-risk, user-optimal, and maximum-risk portfolios. The NOC solves:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\max} \\; \\min\\left\\{\\frac{\\boldsymbol{w} - \\boldsymbol{w}_{\\min}}{\\boldsymbol{w}_{\\mathrm{opt}} - \\boldsymbol{w}_{\\min}},\\; \\frac{\\boldsymbol{w}_{\\max} - \\boldsymbol{w}}{\\boldsymbol{w}_{\\max} - \\boldsymbol{w}_{\\mathrm{opt}}}\\right\\} \\quad \\text{s.t.} \\quad \\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: Portfolio weight vector.
  - ``\\mathcal{W}``: Feasible weight set defined by portfolio constraints.
  - ``\\boldsymbol{w}_{\\min}``: Minimum-risk portfolio weights.
  - ``\\boldsymbol{w}_{\\mathrm{opt}}``: User-optimal portfolio weights.
  - ``\\boldsymbol{w}_{\\max}``: Maximum-risk portfolio weights.

The solution yields a portfolio centrally located within the near-optimal region, robust to small perturbations of the objective.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `opt`: Recursively updated via [`factory`](@ref).
  - `r`: Recursively updated via [`factory`](@ref).
  - `fb`: Recursively updated via [`factory`](@ref).

# Related

  - [`optimise`](@ref)
  - [`NearOptimalCenteringResult`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`NearOptimalCenteringAlgorithm`](@ref)
"""
@propagatable @concrete struct NearOptimalCentering <: RiskJuMPOptimisationEstimator
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
    Number of equally-spaced risk bins for the frontier approximation.
    """
    bins
    """
    $(field_dict[:w_min_noc])
    """
    w_min
    """
    $(field_dict[:w_min_ini])
    """
    w_min_ini
    """
    $(field_dict[:w_opt_noc])
    """
    w_opt
    """
    $(field_dict[:w_opt_ini])
    """
    w_opt_ini
    """
    $(field_dict[:w_max_noc])
    """
    w_max
    """
    $(field_dict[:w_max_ini])
    """
    w_max_ini
    """
    $(field_dict[:ucs_flag])
    """
    ucs_flag
    """
    Near Optimal Centering algorithm variant. It selects the middle the centring model runs, so it also selects which `opt` settings that model reads: [`ConstrainedNearOptimalCentering`](@ref) reads all of them, the default [`UnconstrainedNearOptimalCentering`](@ref) reads the subset its docstring lists.
    """
    alg
    """
    $(field_dict[:fb])
    """
    @fprop fb
    function NearOptimalCentering(opt::JuMPOptimiser, r::TD{<:RM_VecRM},
                                  obj::TD_Option{<:ObjectiveFunction},
                                  bins::Option{<:Number}, w_min::TD_Option{<:VecNum},
                                  w_min_ini::TD_Option{<:VecNum},
                                  w_opt::TD_Option{<:VecNum_VecVecNum},
                                  w_opt_ini::TD_Option{<:VecNum_VecVecNum},
                                  w_max::TD_Option{<:VecNum},
                                  w_max_ini::TD_Option{<:VecNum}, ucs_flag::Bool,
                                  alg::NearOptimalCenteringAlgorithm,
                                  fb::TDO_Option{<:OptE_Opt})
        assert_no_nearest_bind_optimiser_schedule(fb, :fb, :NearOptimalCentering)
        assert_risk_measure_required(r, :NearOptimalCentering;
                                     flag = zero_risk_expression_flag)
        assert_return_term_required(opt.ret, :NearOptimalCentering)
        if isa(r, AbstractVector)
            @argcheck(!isempty(r), IsEmptyError("r cannot be empty"))
            if any(x -> isa(x, QuadExpressionRiskMeasures), r)
                @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
            end
        else
            if isa(r, QuadExpressionRiskMeasures)
                @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.")
            end
        end
        if isa(w_min, AbstractVector)
            @argcheck(!isempty(w_min), IsEmptyError("w_min cannot be empty"))
        end
        if isa(w_min_ini, AbstractVector)
            @argcheck(!isempty(w_min_ini), IsEmptyError("w_min_ini cannot be empty"))
        end
        if isa(w_opt, AbstractVector)
            @argcheck(!isempty(w_opt), IsEmptyError("w_opt cannot be empty"))
        end
        if isa(w_opt_ini, AbstractVector)
            @argcheck(!isempty(w_opt_ini), IsEmptyError("w_opt_ini cannot be empty"))
        end
        if isa(w_max, AbstractVector)
            @argcheck(!isempty(w_max), IsEmptyError("w_max cannot be empty"))
        end
        if isa(w_max_ini, AbstractVector)
            @argcheck(!isempty(w_max_ini), IsEmptyError("w_max_ini cannot be empty"))
        end
        if isa(bins, Number)
            @argcheck(isfinite(bins) && bins > 0,
                      DomainError(bins, "bins must be finite and > 0"))
        end
        assert_time_dependent_substitution(NearOptimalCentering,
                                           (; opt, r, obj, bins, w_min, w_min_ini, w_opt,
                                            w_opt_ini, w_max, w_max_ini, ucs_flag, alg, fb),
                                           near_optimal_centering_td_defaults())
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
function NearOptimalCentering(; opt::JuMPOptimiser, r::TD{<:RM_VecRM} = StandardDeviation(),
                              obj::TD_Option{<:ObjectiveFunction} = MinimumRisk(),
                              bins::Option{<:Number} = nothing,
                              w_min::TD_Option{<:VecNum} = nothing,
                              w_min_ini::TD_Option{<:VecNum} = nothing,
                              w_opt::TD_Option{<:VecNum_VecVecNum} = nothing,
                              w_opt_ini::TD_Option{<:VecNum_VecVecNum} = nothing,
                              w_max::TD_Option{<:VecNum} = nothing,
                              w_max_ini::TD_Option{<:VecNum} = nothing,
                              ucs_flag::Bool = true,
                              alg::NearOptimalCenteringAlgorithm = UnconstrainedNearOptimalCentering(),
                              fb::TDO_Option{<:OptE_Opt} = nothing)::NearOptimalCentering
    return NearOptimalCentering(opt, r, obj, bins, w_min, w_min_ini, w_opt, w_opt_ini,
                                w_max, w_max_ini, ucs_flag, alg, fb)
end
function time_dependent_field_defaults(::NearOptimalCentering)::NamedTuple
    return near_optimal_centering_td_defaults()
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if any sub-estimator of `opt` requires previous portfolio weights (JuMP optimiser, risk measure, or fallback).
"""
function needs_previous_weights(opt::NearOptimalCentering)
    return (any(f -> needs_previous_weights(getfield(opt, f)),
                time_dependent_fields(opt)) ||
            needs_previous_weights(opt.opt) ||
            needs_previous_weights(opt.r) ||
            needs_previous_weights(opt.fb))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a cluster-sliced copy of [`NearOptimalCentering`](@ref) for asset index set `i` and returns matrix `X`.
"""
function port_opt_view(noc::NearOptimalCentering, i, X::MatNum,
                       args...)::NearOptimalCentering
    X = isa(noc.opt.pe, AbstractPriorResult) ? noc.opt.pe.X : X
    opt = port_opt_view(noc.opt, i, X)
    r = port_opt_view(noc.r, i, X)
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
"""
    near_optimal_centering_risks(scalariser, r, pr, fees, slv, w_min, w_opt, w_max)

Compute the scaled risk values for the minimum, optimal, and maximum portfolios.

Used internally by Near Optimal Centering to evaluate the risk at the three anchor portfolios (minimum-risk, optimal, maximum-risk) using the given risk measure(s) and scalarisation strategy.

# Arguments

  - `scalariser`: Risk scalarisation strategy (e.g. `SumScalariser`, `LogSumExpScalariser`).
  - `r`: Risk measure or vector of risk measures.
  - `pr`: Prior result containing asset data.
  - `fees`: Optional fees configuration.
  - `slv`: Solver or vector of solvers.
  - $(arg_dict[:w_min])
  - $(arg_dict[:w_opt])
  - $(arg_dict[:w_max])

# Returns

  - `(risk_min, risk_opt, risk_max)`: Tuple of risk values at the three anchor portfolios.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function near_optimal_centering_risks(::Any, r::RiskMeasure, pr::AbstractPriorResult,
                                      fees::Option{<:Fees}, slv::Slv_VecSlv, w_min::VecNum,
                                      w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    r = factory(r, pr, slv)
    # No `scale` here. The barrier compares these targets against the model's `:risk`
    # expression, and the model drops a lone measure's combination weight, so the two sides
    # must agree. The vector method below does apply it, because there the measures combine.
    risk_min = expected_risk(r, w_min, X, fees)
    risk_opt = expected_risk(r, w_opt, X, fees)
    risk_max = expected_risk(r, w_max, X, fees)
    return risk_min, risk_opt, risk_max
end
function near_optimal_centering_risks(sca::Scalariser, rs::VecRM, pr::AbstractPriorResult,
                                      fees::Option{<:Fees}, slv::Option{<:Slv_VecSlv},
                                      w_min::VecNum, w_opt::VecNum_VecVecNum, w_max::VecNum)
    X = pr.X
    rs = factory(rs, pr, slv)
    return scalarise(sca, rs) do r
        scale = r.settings.scale
        return (expected_risk(r, w_min, X, fees) * scale,
                expected_risk(r, w_opt, X, fees) * scale,
                expected_risk(r, w_max, X, fees) * scale)
    end
end
"""
$(DocStringExtensions.TYPEDEF)

Intermediate result type storing the setup data for Near Optimal Centering.

Holds pre-computed portfolio weights, risk and return targets, and sub-problem return codes needed to formulate and solve the NOC optimisation problem.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NearOptimalCentering`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
@concrete struct NearOptimalSetup <: AbstractResult
    """
    $(field_dict[:w_opt_noc])
    """
    w_opt
    """
    $(field_dict[:rk_opt])
    """
    rk_opt
    """
    $(field_dict[:rt_opt])
    """
    rt_opt
    """
    $(field_dict[:rt_min])
    """
    rt_min
    """
    $(field_dict[:rt_max])
    """
    rt_max
    """
    $(field_dict[:rt_ends])
    """
    rt_ends
    """
    $(field_dict[:w_min_noc])
    """
    w_min
    """
    $(field_dict[:w_max_noc])
    """
    w_max
    """
    $(field_dict[:r_opt])
    """
    r
    """
    $(field_dict[:opt_jmp])
    """
    opt
    """
    $(field_dict[:attrs_noc])
    """
    attrs
    """
    $(field_dict[:w_min_retcode])
    """
    w_min_retcode
    """
    $(field_dict[:w_opt_retcode])
    """
    w_opt_retcode
    """
    $(field_dict[:w_max_retcode])
    """
    w_max_retcode
    # Weight and target fields take frontier-safe unions: the sub-problem weights may be a
    # single vector or (in frontier mode) a vector of vectors, and the risk/return targets
    # may likewise be a scalar or a vector.
    function NearOptimalSetup(w_opt::VecNum_VecVecNum, rk_opt::Union{<:Number, <:VecNum},
                              rt_opt::Union{<:Number, <:VecNum},
                              rt_min::Union{<:Number, <:VecNum},
                              rt_max::Union{<:Number, <:VecNum}, rt_ends::Option{<:VecPair},
                              w_min::VecNum_VecVecNum, w_max::VecNum_VecVecNum, r::RM_VecRM,
                              opt::JuMPOptimiser, attrs::ProcessedJuMPOptimiserAttributes,
                              w_min_retcode::OptimisationReturnCode,
                              w_opt_retcode::OptRetCode_VecOptRetCode,
                              w_max_retcode::OptimisationReturnCode)
        return new{typeof(w_opt), typeof(rk_opt), typeof(rt_opt), typeof(rt_min),
                   typeof(rt_max), typeof(rt_ends), typeof(w_min), typeof(w_max), typeof(r),
                   typeof(opt), typeof(attrs), typeof(w_min_retcode), typeof(w_opt_retcode),
                   typeof(w_max_retcode)}(w_opt, rk_opt, rt_opt, rt_min, rt_max, rt_ends,
                                          w_min, w_max, r, opt, attrs, w_min_retcode,
                                          w_opt_retcode, w_max_retcode)
    end
end
function NearOptimalSetup(; w_opt::VecNum_VecVecNum, rk_opt::Union{<:Number, <:VecNum},
                          rt_opt::Union{<:Number, <:VecNum},
                          rt_min::Union{<:Number, <:VecNum},
                          rt_max::Union{<:Number, <:VecNum},
                          rt_ends::Option{<:VecPair} = nothing, w_min::VecNum_VecVecNum,
                          w_max::VecNum_VecVecNum, r::RM_VecRM, opt::JuMPOptimiser,
                          attrs::ProcessedJuMPOptimiserAttributes,
                          w_min_retcode::OptimisationReturnCode,
                          w_opt_retcode::OptRetCode_VecOptRetCode,
                          w_max_retcode::OptimisationReturnCode)::NearOptimalSetup
    return NearOptimalSetup(w_opt, rk_opt, rt_opt, rt_min, rt_max, rt_ends, w_min, w_max, r,
                            opt, attrs, w_min_retcode, w_opt_retcode, w_max_retcode)
end
"""
    near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult; dims::Int = 1)

Compute all prerequisite data for Near Optimal Centering.

Solves the minimum-risk, optimal-objective, and maximum-risk sub-problems (unless pre-computed weights are provided), then computes the risk and return targets for the NOC problem.

# Arguments

  - `noc::NearOptimalCentering`: NOC estimator configuration.
  - `rd::ReturnsResult`: Returns data.
  - `dims::Int`: Observation dimension (default `1`).

# Returns

  - [`NearOptimalSetup`](@ref) containing all setup data needed for the NOC optimisation.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`NearOptimalSetup`](@ref)
  - [`near_optimal_centering_risks`](@ref)
"""
function near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult;
                                      dims::Int = 1, kwargs...)
    w_min = noc.w_min
    w_opt = noc.w_opt
    w_max = noc.w_max
    w_min_flag = isnothing(w_min)
    w_opt_flag = isnothing(w_opt)
    w_max_flag = isnothing(w_max)
    w_min_retcode = OptimisationSuccess()
    w_opt_retcode = OptimisationSuccess()
    w_max_retcode = OptimisationSuccess()
    unconstrained = isa(noc.alg, UnconstrainedNearOptimalCentering)
    r = ucs_risk_measure(noc.r, rd)
    attrs = processed_jump_optimiser_attributes(noc.opt, rd; dims = dims, kwargs...)
    opt = jump_optimiser_from_attributes(noc.opt, attrs)
    # The per-term corner solves need the same unbounded pair the max-return corner uses, so
    # the pair is built whenever a return term declares a frontier bound, even when both
    # anchor portfolios were supplied.
    swept = frontier_return_terms(attrs.ret)
    if w_min_flag || w_max_flag || unconstrained || !isempty(swept)
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
    pr = opt.pe
    fees = opt.fees
    ret = opt.ret
    rk_min, rk_opt, rk_max = near_optimal_centering_risks(opt.sca, r, pr, fees, opt.slv,
                                                          w_min, w_opt, w_max)
    rt_min = expected_return(ret, w_min, pr, fees)
    rt_opt = expected_return(ret, w_opt, pr, fees)
    rt_max = expected_return(ret, w_max, pr, fees)
    rt_ends = if isempty(swept)
        nothing
    else
        return_term_ends(swept, attrs.ret, nb_r, nb_opt, rd, pr, fees, w_min, noc.w_max_ini)
    end
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
    return NearOptimalSetup(; w_opt = w_opt, rk_opt = rk_opt, rt_opt = rt_opt,
                            rt_min = rt_min, rt_max = rt_max, rt_ends = rt_ends,
                            w_min = w_min, w_max = w_max, r = r, opt = opt, attrs = attrs,
                            w_min_retcode = w_min_retcode, w_opt_retcode = w_opt_retcode,
                            w_max_retcode = w_max_retcode)
end
"""
    frontier_return_terms(ret)

Return the indices of the return terms that declare a [`Frontier`](@ref) lower bound.

A term whose `settings.lb` is already a vector of numbers is not swept — its sweep points are
stated — so it does not need a corner solve. An empty answer means the return frontier costs
nothing.

# Related

  - [`return_term_ends`](@ref)
  - [`JuMPReturnsSettings`](@ref)
"""
function frontier_return_terms(ret::JuMPReturnsEstimator)
    return isa(ret.settings.lb, Frontier) ? [1] : Int[]
end
function frontier_return_terms(ret::VecJRE)
    return [i for (i, r) in enumerate(ret) if isa(r.settings.lb, Frontier)]
end
"""
    return_term_ends(swept, ret, nb_r, nb_opt, rd, pr, fees, w_min, w_ini)

Solve one maximum-return corner per swept return term and read off that term's span.

The low end of every span is the shared minimum-risk portfolio `w_min`, which
[`near_optimal_centering_setup`](@ref) already has. The high end of term *i*'s span comes
from a portfolio that maximised term *i* **alone**, which is why this costs `k` extra solves.
Reading it off the aggregate maximum-return corner instead would make the span an artefact of
the other terms' `scale`, and can leave `rt_min > rt_max`.

# Returns

  - A vector of `i => (rt_min_i, rt_max_i)` pairs, one per swept term.

# Related

  - [`frontier_return_terms`](@ref)
  - [`MaximumElementReturn`](@ref)
  - [`compute_ret_lbs`](@ref)
"""
function return_term_ends(swept::VecInt, ret::JRE_VecJRE, nb_r, nb_opt::JuMPOptimiser,
                          rd::ReturnsResult, pr::AbstractPriorResult, fees::Option{<:Fees},
                          w_min::VecNum, w_ini)
    ends = Pair{Int, Tuple{<:Number, <:Number}}[]
    for i in swept
        rt = return_term(ret, i)
        res = optimise(MeanRisk(; r = nb_r, obj = MaximumElementReturn(i), opt = nb_opt,
                                wi = w_ini), rd; save = false)
        @argcheck(isa(res.retcode, OptimisationSuccess),
                  ArgumentError("maximum-return solve for return term $i failed with retcode $(res.retcode)"))
        push!(ends,
              i => (expected_return(rt, w_min, pr, fees),
                    expected_return(rt, res.w, pr, fees)))
    end
    return ends
end
"""
    set_near_optimal_centering_constraints!(model::JuMP.Model, wb::WeightBounds)

Add Near Optimal Centering logarithmic barrier constraints to the JuMP model.

Introduces log variables for portfolio weights, upper bound distances, risk, and return, then adds exponential cone constraints implementing the analytic centre formulation.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `wb::WeightBounds`: Weight bounds configuration.

# Returns

  - Objective expression for the NOC barrier function.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
"""
function set_near_optimal_centering_constraints!(model::JuMP.Model, wb::WeightBounds)
    w = get_w(model)
    sc = get_constraint_scale(model)
    w_ub = wb.ub
    risk = get_risk(model)
    ret = get_ret(model)
    rk = shared_get(model, :noc_rk)
    rt = shared_get(model, :noc_rt)
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
"""
    set_near_optimal_objective_function!(alg, model, noc, attrs)

Set the Near Optimal Centering objective function in the JuMP model.

Formulates the NOC objective based on the algorithm variant. For `UnconstrainedNearOptimalCentering`, uses only the barrier function. For `ConstrainedNearOptimalCentering`, also adds objective penalties and custom objective terms.

The centring problem is a *distance minimisation*, so the objective it reports to a custom term is [`MinimumRisk`](@ref) — the objective actually being built, not `noc.obj`, which describes the reference-point sub-problems (ADR 0036).

The unconstrained variant reaches neither [`add_custom_objective_term!`](@ref) nor [`add_penalty_to_objective!`](@ref), so `opt.cobj` prices the anchor sub-problems and not the centring solve. No builder on that path contributes to the [Objective Penalty](@ref add_to_objective_penalty!) either, so the accumulator is empty and the omitted fold changes no number today. The one case a user can observe is a Custom Objective Term that names itself and supplies no builder: it raises on the anchor solve, and raises nowhere when `w_min`, `w_opt` and `w_max` are all supplied and no anchor solve runs (ADR 0036, amendment).

# Arguments

  - `alg`: NOC algorithm variant ([`UnconstrainedNearOptimalCentering`](@ref) or [`ConstrainedNearOptimalCentering`](@ref)).
  - `model::JuMP.Model`: JuMP optimisation model.
  - `noc::NearOptimalCentering`: The outer optimisation estimator, passed on to custom objective terms.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle.

# Returns

  - `nothing`.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`set_near_optimal_centering_constraints!`](@ref)
  - [`solve_noc!`](@ref)
"""
function set_near_optimal_objective_function!(::UnconstrainedNearOptimalCentering,
                                              model::JuMP.Model, ::NearOptimalCentering,
                                              attrs::ProcessedJuMPOptimiserAttributes)
    so = get_objective_scale(model)
    obj_expr = set_near_optimal_centering_constraints!(model, attrs.wb)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_near_optimal_objective_function!(::ConstrainedNearOptimalCentering,
                                              model::JuMP.Model, noc::NearOptimalCentering,
                                              attrs::ProcessedJuMPOptimiserAttributes)
    so = get_objective_scale(model)
    obj_expr = set_near_optimal_centering_constraints!(model, attrs.wb)
    add_custom_objective_term!(model, MinimumRisk(), noc.opt.cobj, noc, attrs)
    obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
"""
    solve_noc!(noc, model, rk_opt, rt_opt, opt, attrs, args...)

Solve the Near Optimal Centering problem given the model, risk, and return targets.

Sets model parameters for the risk and return targets, configures the NOC objective, and solves the JuMP model. Multiple overloads handle different algorithm variants and frontier sweep modes.

# Arguments

  - `noc::NearOptimalCentering`: NOC estimator configuration.
  - `model::JuMP.Model`: JuMP optimisation model.
  - `rk_opt`: Risk target(s) for the NOC problem.
  - `rt_opt`: Return target(s) for the NOC problem.
  - `opt::BaseJuMPOptimisationEstimator`: JuMP optimiser configuration.
  - `args...`: Additional arguments (frontier bounds, flags, etc.).

# Returns

  - `(retcode, sol)` or `(retcodes, sols)` depending on the overload.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function solve_noc!(noc::NearOptimalCentering, model::JuMP.Model, rk_opt::Number,
                    rt_opt::Number, opt::BaseJuMPOptimisationEstimator,
                    attrs::ProcessedJuMPOptimiserAttributes, args...)
    JuMP.@expression(model, noc_rk, rk_opt)
    JuMP.@expression(model, noc_rt, rt_opt)
    set_near_optimal_objective_function!(noc.alg, model, noc, attrs)
    return optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:UnconstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator,
                    attrs::ProcessedJuMPOptimiserAttributes, args...)
    noc_rk, noc_rt = set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X), length(rk_opts)) do i
        return set_noc_anchor!(noc_rk, noc_rt, rk_opts, rt_opts, i)
    end
end
"""
    set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)

Register the two anchor parameters the centring barrier reads, and set the objective.

`noc_rk` and `noc_rt` are the risk and return of the near-optimal anchor the barrier centres
on. A sweep moves them from point to point, so they are parameters rather than expressions —
the model is assembled once and only their values change between solves.

# Arguments

  - $(arg_dict[:model])
  - `noc::NearOptimalCentering`: Dispatch object for the objective.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed bundle.
  - `rk_opts::VecNum`: The anchor risks, one per sweep point.
  - `rt_opts::VecNum`: The anchor returns, one per sweep point.

# Returns

  - `(noc_rk, noc_rt)`: The two parameter references.

# Related

  - [`set_noc_anchor!`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
  - [`solve_noc!`](@ref)
"""
function set_noc_anchor_parameters!(model::JuMP.Model, noc::NearOptimalCentering,
                                    attrs::ProcessedJuMPOptimiserAttributes,
                                    rk_opts::VecNum, rt_opts::VecNum)
    JuMP.@variable(model, noc_rk in JuMP.Parameter(zero(eltype(rk_opts))))
    JuMP.@variable(model, noc_rt in JuMP.Parameter(zero(eltype(rt_opts))))
    set_near_optimal_objective_function!(noc.alg, model, noc, attrs)
    return noc_rk, noc_rt
end
"""
    set_noc_anchor!(noc_rk, noc_rt, rk_opts, rt_opts, i)

Move the centring anchor onto sweep point `i`.

The [`frontier_sweep!`](@ref) hook of every swept `NearOptimalCentering` solve. The anchors are
the [`MeanRisk`](@ref) solutions of the **same** sweep, computed by
[`near_optimal_centering_setup`](@ref), so anchor `i` belongs to sweep point `i` and the index
is the flat one [`frontier_sweep_axes`](@ref) counts. This is the pairing that makes a swept
centring solve centre on its own point rather than on some other point's optimum.

# Arguments

  - `noc_rk`: The anchor-risk parameter.
  - `noc_rt`: The anchor-return parameter.
  - `rk_opts::VecNum`: The anchor risks, in flat sweep order.
  - `rt_opts::VecNum`: The anchor returns, in flat sweep order.
  - `i::Integer`: The flat index of the sweep point.

# Returns

  - `nothing`.

# Related

  - [`set_noc_anchor_parameters!`](@ref)
  - [`frontier_sweep!`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function set_noc_anchor!(noc_rk, noc_rt, rk_opts::VecNum, rt_opts::VecNum, i::Integer)
    JuMP.set_parameter_value(noc_rk, rk_opts[i])
    JuMP.set_parameter_value(noc_rt, rt_opts[i])
    return nothing
end
"""
    compute_ret_lbs(ret_frontier::VecPair, ::Nothing)

Return a `NearOptimalCentering` return frontier unchanged.

`nothing` means no swept term declared a [`Frontier`](@ref), so every sweep point was stated
outright and there is no span to resolve.

# Arguments

  - `ret_frontier::VecPair`: The `:ret_frontier` Model State entry.
  - `::Nothing`: No per-term spans were computed.

# Returns

  - `ret_frontier`, unchanged.

# Related

  - [`compute_ret_lbs`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_noc!`](@ref)
"""
function compute_ret_lbs(ret_frontier::VecPair, ::Nothing)
    # Every swept term stated its sweep points as a vector, so there is nothing to compute.
    return ret_frontier
end
"""
    compute_ret_lbs(ret_frontier::VecPair, rt_ends::VecPair)

Resolve a `NearOptimalCentering` return frontier from spans computed during setup.

`NearOptimalCentering` does not re-solve for its corners here, unlike [`MeanRisk`](@ref):
[`near_optimal_centering_setup`](@ref) has already paid for one minimum-risk solve and one
maximum-return solve per swept term, so this only turns each span into its `N` sweep points.

# Arguments

  - `ret_frontier::VecPair`: The `:ret_frontier` Model State entry.
  - `rt_ends::VecPair`: The per-term spans, as `i => (rt_min_i, rt_max_i)`.

# Returns

  - The return frontier with every span resolved into a range of sweep points.

# Related

  - [`return_term_ends`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_noc!`](@ref)
"""
function compute_ret_lbs(ret_frontier::VecPair, rt_ends::VecPair)
    ret_frontier = copy(ret_frontier)
    for (j, rtf) in enumerate(ret_frontier)
        expr, front, i = rtf.second
        if isa(front, VecNum)
            continue
        end
        idx = findfirst(x -> x.first == i, rt_ends)
        @argcheck(!isnothing(idx),
                  ArgumentError("no return span was computed for return term $i; `near_optimal_centering_setup` and `set_return_bounds!` disagree about which terms are swept"))
        rt_min, rt_max = rt_ends[idx].second
        ret_frontier[j] = rtf.first => (expr, range(rt_min, rt_max; length = front.N), i)
    end
    return ret_frontier
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator,
                    attrs::ProcessedJuMPOptimiserAttributes, ::Any, ::Any,
                    rt_ends::Option{<:VecPair}, ::Any, ::Any, ::Val{true}, ::Val{false},
                    args...)
    ret_frontier = compute_ret_lbs(shared_get(model, :ret_frontier), rt_ends)
    ret_axis = set_ret_frontier_parameters!(model, ret_frontier)
    noc_rk, noc_rt = set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X),
                           frontier_sweep_axes(ret_axis, nothing)) do i
        return set_noc_anchor!(noc_rk, noc_rt, rk_opts, rt_opts, i)
    end
end
function rebuild_risk_frontier(noc::NearOptimalCentering{<:Any, <:AbstractVector, <:Any,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any, <:Any,
                                                         <:ConstrainedNearOptimalCentering},
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, w_min::VecNum, w_max::VecNum,
                               idx::VecInt, args...)
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
"""
    compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConstrainedNearOptimalCentering}, pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum, w_max::VecNum, args...)

Compute risk upper bounds for a constrained `NearOptimalCentering` frontier sweep.

Identifies risk frontier entries that are not yet resolved (i.e. not concrete weight vectors) and rebuilds them using the minimum and maximum portfolio weights.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model containing `risk_frontier`.
  - `noc::NearOptimalCentering{..., <:ConstrainedNearOptimalCentering}`: Constrained Near Optimal Centering optimiser.
  - `pr::AbstractPriorResult`: Prior result with asset moments.
  - `fees::Option{<:Fees}`: Optional fees configuration.
  - `w_min::VecNum`: Minimum-risk portfolio weights.
  - `w_max::VecNum`: Maximum-risk (maximum-return) portfolio weights.

# Returns

  - Updated risk frontier vector of `(keys, vals)` pairs.

# Related

  - [`compute_risk_ubs`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_noc!`](@ref)
  - [`unresolved_risk_frontier`](@ref)
"""
function compute_risk_ubs(model::JuMP.Model,
                          noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any,
                                                    <:ConstrainedNearOptimalCentering},
                          pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum,
                          w_max::VecNum, args...)
    risk_frontier, idx = unresolved_risk_frontier(model)
    if isempty(idx)
        return risk_frontier
    end
    return rebuild_risk_frontier(noc, pr, fees, risk_frontier, w_min, w_max, idx, args...)
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator,
                    attrs::ProcessedJuMPOptimiserAttributes, ::Any, ::Any, ::Any,
                    w_min::VecNum, w_max::VecNum, ::Val{false}, ::Val{true}, args...)
    risk_frontier = compute_risk_ubs(model, noc, opt.pe, opt.fees, w_min, w_max, args...)
    risk_axis = set_risk_frontier_parameters!(model, risk_frontier)
    noc_rk, noc_rt = set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X),
                           frontier_sweep_axes(nothing, risk_axis)) do i
        return set_noc_anchor!(noc_rk, noc_rt, rk_opts, rt_opts, i)
    end
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator,
                    attrs::ProcessedJuMPOptimiserAttributes, ::Any, ::Any,
                    rt_ends::Option{<:VecPair}, w_min::VecNum, w_max::VecNum, ::Val{true},
                    ::Val{true}, args...)
    ret_frontier = compute_ret_lbs(shared_get(model, :ret_frontier), rt_ends)
    risk_frontier = compute_risk_ubs(model, noc, opt.pe, opt.fees, w_min, w_max, args...)
    risk_axis = set_risk_frontier_parameters!(model, risk_frontier)
    ret_axis = set_ret_frontier_parameters!(model, ret_frontier)
    noc_rk, noc_rt = set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X),
                           frontier_sweep_axes(ret_axis, risk_axis)) do i
        return set_noc_anchor!(noc_rk, noc_rt, rk_opts, rt_opts, i)
    end
end
"""
    get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)

Compute the overall optimisation return code from individual sub-problem return codes.

Combines the return codes from the minimum, optimal, and maximum weight sub-problems with the near-optimal centering return code to determine the overall status.

# Arguments

  - `w_min_retcode`: Return code from the minimum weight sub-problem.
  - `w_opt_retcode`: Return code from the optimal weight sub-problem.
  - `w_max_retcode`: Return code from the maximum weight sub-problem.
  - `noc_retcode`: Return code from the near-optimal centering sub-problem.

# Returns

  - `OptimisationSuccess()` if every sub-problem succeeded; otherwise an
    `OptimisationFailure` whose `res` is a named tuple `(; msg, w_min, w_opt, w_max, noc_opt)` carrying the failure summary and the individual sub-problem return codes
    (including their solver trial diagnostics).

# Related

  - [`NearOptimalCentering`](@ref)
"""
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
        OptimisationSuccess()
    else
        @warn("Failed to solve optimisation problem. Check `retcode.res` for details.")
        OptimisationFailure(;
                            res = (; msg = msg, w_min = w_min_retcode,
                                   w_opt = w_opt_retcode, w_max = w_max_retcode,
                                   noc_opt = noc_retcode))
    end
end
"""
    assemble_near_optimal_centering_model!(alg, model, noc, setup, rd)

Run the model-assembly middle of the Near Optimal Centering variant `alg`.

The two variants share a head and a Result, and differ only in the middle and in the solve.
This is the middle. [`ConstrainedNearOptimalCentering`](@ref) delegates to the shared
[`assemble_jump_model!`](@ref). [`UnconstrainedNearOptimalCentering`](@ref) runs the four
steps of that sequence it needs — non-fixed fees, risk, scalarisation, return — and then the
same [`assert_frontier_sweep_cap`](@ref) tail (ADR 0008, amendments 2 and 3).

[`set_non_fixed_fees!`](@ref) runs before the risk build, as it does in
[`assemble_jump_model!`](@ref). It is what makes the model's return expression net of fees,
and the barrier compares that expression against the `noc_rt` target, which
[`near_optimal_centering_setup`](@ref) computes net of fees as well (ADR 0008, amendment 3).
A fixed fee still does not apply: it needs the cardinality binaries `set_mip_constraints!`
produces, and that builder belongs to the middle this variant does not run.

[`UnconstrainedNearOptimalCentering`](@ref) lists every `opt` setting the resulting model
reads and every setting it does not.

The unconstrained variant reads `setup.opt`, which
[`near_optimal_centering_setup`](@ref) has already replaced with the
[`no_bounds_optimiser`](@ref) copy. So `opt.ret` is the bound-free return estimator, and
`opt.pe`, `opt.fees` and `opt.sca` are the processed values `setup.attrs` carries. The
phylogeny argument is `nothing` because the variant applies no phylogeny constraints.

# Arguments

  - `alg`: NOC algorithm variant ([`UnconstrainedNearOptimalCentering`](@ref) or
    [`ConstrainedNearOptimalCentering`](@ref)).
  - $(arg_dict[:model])
  - `noc::NearOptimalCentering`: Dispatch object for the risk and custom constraint builders.
  - `setup::NearOptimalSetup`: Setup bundle from [`near_optimal_centering_setup`](@ref).
  - $(arg_dict[:rd])

# Returns

  - `nothing`. Mutates `model` in place.

# Related

  - [`solve_near_optimal_centering!`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`set_non_fixed_fees!`](@ref)
  - [`set_risk_and_scalarise!`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assemble_near_optimal_centering_model!(::UnconstrainedNearOptimalCentering,
                                                model::JuMP.Model,
                                                noc::NearOptimalCentering,
                                                setup::NearOptimalSetup, rd::ReturnsResult)
    (; r, opt) = setup
    set_non_fixed_fees!(model, opt.fees)
    set_risk_and_scalarise!(model, r, noc, opt, opt.pe, nothing, opt.fees; rd = rd)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pe; rd = rd)
    assert_frontier_sweep_cap(model)
    return nothing
end
function assemble_near_optimal_centering_model!(::ConstrainedNearOptimalCentering,
                                                model::JuMP.Model,
                                                noc::NearOptimalCentering,
                                                setup::NearOptimalSetup, rd::ReturnsResult)
    (; r, opt, attrs) = setup
    assemble_jump_model!(model, noc, opt, attrs, rd, r)
    return nothing
end
"""
    solve_near_optimal_centering!(alg, model, noc, setup)

Run the solve tail of the Near Optimal Centering variant `alg`.

Reads the arguments each [`solve_noc!`](@ref) overload needs off `setup`, so the two variants
share one `_optimise` head. The constrained variant sweeps both frontier registries, so it
passes the two `Val` flags that select the sweeping overload; the unconstrained variant
registers no frontier bound and passes neither.

# Arguments

  - `alg`: NOC algorithm variant ([`UnconstrainedNearOptimalCentering`](@ref) or
    [`ConstrainedNearOptimalCentering`](@ref)).
  - $(arg_dict[:model])
  - `noc::NearOptimalCentering`: NOC estimator configuration.
  - `setup::NearOptimalSetup`: Setup bundle from [`near_optimal_centering_setup`](@ref).

# Returns

  - `(retcode, sol)` or `(retcodes, sols)`, as [`solve_noc!`](@ref) returns.

# Related

  - [`assemble_near_optimal_centering_model!`](@ref)
  - [`solve_noc!`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function solve_near_optimal_centering!(::UnconstrainedNearOptimalCentering,
                                       model::JuMP.Model, noc::NearOptimalCentering,
                                       setup::NearOptimalSetup)
    (; rk_opt, rt_opt, opt, attrs) = setup
    return solve_noc!(noc, model, rk_opt, rt_opt, opt, attrs)
end
function solve_near_optimal_centering!(::ConstrainedNearOptimalCentering, model::JuMP.Model,
                                       noc::NearOptimalCentering, setup::NearOptimalSetup)
    (; rk_opt, rt_opt, opt, attrs, rt_min, rt_max, rt_ends, w_min, w_max) = setup
    return solve_noc!(noc, model, rk_opt, rt_opt, opt, attrs, rt_min, rt_max, rt_ends,
                      w_min, w_max, Val(shared_has(model, :ret_frontier)),
                      Val(shared_has(model, :risk_frontier)))
end
function _optimise(noc::NearOptimalCentering, rd::ReturnsResult = ReturnsResult();
                   dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
    noc = reset_time_dependent_estimator(noc)
    setup = near_optimal_centering_setup(noc, rd; dims = dims, kwargs...)
    (; w_opt, r, opt, attrs, w_min_retcode, w_opt_retcode, w_max_retcode) = setup
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    set_maximum_ratio_factor_variables!(model, MinimumRisk())
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt)
    assemble_near_optimal_centering_model!(noc.alg, model, noc, setup, rd)
    noc_retcode, sol = solve_near_optimal_centering!(noc.alg, model, noc, setup)
    retcode = get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    return NearOptimalCenteringResult(;
                                      jr = JuMPOptimisationResult(; pa = attrs,
                                                                  retcode = retcode,
                                                                  sol = sol,
                                                                  model = ifelse(save,
                                                                                 model,
                                                                                 nothing)),
                                      r = factory(r, opt.pe, opt.slv),
                                      w_min_retcode = w_min_retcode,
                                      w_opt_retcode = w_opt_retcode,
                                      w_max_retcode = w_max_retcode,
                                      noc_retcode = noc_retcode, fb = nothing)
end
"""
    optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                      <:Any, <:Any, <:Any, <:Any, <:Any, Nothing
                  },
             rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
             str_names::Bool = false, save::Bool = true, kwargs...) -> NearOptimalCenteringResult

Run the Near Optimal Centering portfolio optimisation.

# Arguments

  - `noc`: The near optimal centering optimiser to use.
  - $(arg_dict[:rd]) If `isa(noc.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `dims`: The dimension along which observations advance in time.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`NearOptimalCenteringResult`](@ref)
"""
function optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                            <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
    return _optimise(noc, rd; dims = dims, str_names = str_names, save = save, kwargs...)
end

@pipe_delegates NearOptimalCentering opt
@pipe_route_sigma_ucs NearOptimalCentering
export NearOptimalCentering, UnconstrainedNearOptimalCentering,
       ConstrainedNearOptimalCentering, NearOptimalCenteringResult
