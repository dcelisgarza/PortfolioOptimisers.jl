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

Applies Near Optimal Centering within the feasible region defined by the portfolio constraints.

# Related Types

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`UnconstrainedNearOptimalCentering`](@ref)
"""
struct ConstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Unconstrained Near Optimal Centering algorithm.

Applies Near Optimal Centering ignoring feasibility constraints (weights may temporarily violate bounds).

# Related Types

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`ConstrainedNearOptimalCentering`](@ref)
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
    function NearOptimalCenteringResult(jr::JuMPOptimisationResult,
                                        w_min_retcode::OptimisationReturnCode,
                                        w_opt_retcode::OptRetCode_VecOptRetCode,
                                        w_max_retcode::OptimisationReturnCode,
                                        noc_retcode::OptRetCode_VecOptRetCode,
                                        fb::Option{<:OptE_Opt})
        return new{typeof(jr), typeof(w_min_retcode), typeof(w_opt_retcode),
                   typeof(w_max_retcode), typeof(noc_retcode), typeof(fb)}(jr,
                                                                           w_min_retcode,
                                                                           w_opt_retcode,
                                                                           w_max_retcode,
                                                                           noc_retcode, fb)
    end
end
function NearOptimalCenteringResult(; jr::JuMPOptimisationResult,
                                    w_min_retcode::OptimisationReturnCode,
                                    w_opt_retcode::OptRetCode_VecOptRetCode,
                                    w_max_retcode::OptimisationReturnCode,
                                    noc_retcode::OptRetCode_VecOptRetCode,
                                    fb::Option{<:OptE_Opt})::NearOptimalCenteringResult
    return NearOptimalCenteringResult(jr, w_min_retcode, w_opt_retcode, w_max_retcode,
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
    Near Optimal Centering algorithm variant.
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
  - `w_min`: Minimum-risk portfolio weights.
  - `w_opt`: Optimal portfolio weights (vector or vector of vectors).
  - `w_max`: Maximum-risk portfolio weights.

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
    scale = r.settings.scale
    risk_min = expected_risk(r, w_min, X, fees) * scale
    risk_opt = expected_risk(r, w_opt, X, fees) * scale
    risk_max = expected_risk(r, w_max, X, fees) * scale
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
                              rt_max::Union{<:Number, <:VecNum}, w_min::VecNum_VecVecNum,
                              w_max::VecNum_VecVecNum, r::RM_VecRM, opt::JuMPOptimiser,
                              attrs::ProcessedJuMPOptimiserAttributes,
                              w_min_retcode::OptimisationReturnCode,
                              w_opt_retcode::OptRetCode_VecOptRetCode,
                              w_max_retcode::OptimisationReturnCode)
        return new{typeof(w_opt), typeof(rk_opt), typeof(rt_opt), typeof(rt_min),
                   typeof(rt_max), typeof(w_min), typeof(w_max), typeof(r), typeof(opt),
                   typeof(attrs), typeof(w_min_retcode), typeof(w_opt_retcode),
                   typeof(w_max_retcode)}(w_opt, rk_opt, rt_opt, rt_min, rt_max, w_min,
                                          w_max, r, opt, attrs, w_min_retcode,
                                          w_opt_retcode, w_max_retcode)
    end
end
function NearOptimalSetup(; w_opt::VecNum_VecVecNum, rk_opt::Union{<:Number, <:VecNum},
                          rt_opt::Union{<:Number, <:VecNum},
                          rt_min::Union{<:Number, <:VecNum},
                          rt_max::Union{<:Number, <:VecNum}, w_min::VecNum_VecVecNum,
                          w_max::VecNum_VecVecNum, r::RM_VecRM, opt::JuMPOptimiser,
                          attrs::ProcessedJuMPOptimiserAttributes,
                          w_min_retcode::OptimisationReturnCode,
                          w_opt_retcode::OptRetCode_VecOptRetCode,
                          w_max_retcode::OptimisationReturnCode)::NearOptimalSetup
    return NearOptimalSetup(w_opt, rk_opt, rt_opt, rt_min, rt_max, w_min, w_max, r, opt,
                            attrs, w_min_retcode, w_opt_retcode, w_max_retcode)
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
    pr = opt.pe
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
    return NearOptimalSetup(; w_opt = w_opt, rk_opt = rk_opt, rt_opt = rt_opt,
                            rt_min = rt_min, rt_max = rt_max, w_min = w_min, w_max = w_max,
                            r = r, opt = opt, attrs = attrs, w_min_retcode = w_min_retcode,
                            w_opt_retcode = w_opt_retcode, w_max_retcode = w_max_retcode)
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
    rk = model[:noc_rk]
    rt = model[:noc_rt]
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
    set_near_optimal_objective_function!(alg, model, opt)

Set the Near Optimal Centering objective function in the JuMP model.

Formulates the NOC objective based on the algorithm variant. For `UnconstrainedNearOptimalCentering`, uses only the barrier function. For `ConstrainedNearOptimalCentering`, also adds objective penalties and custom objective terms.

# Arguments

  - `alg`: NOC algorithm variant ([`UnconstrainedNearOptimalCentering`](@ref) or [`ConstrainedNearOptimalCentering`](@ref)).
  - `model::JuMP.Model`: JuMP optimisation model.
  - `opt::BaseJuMPOptimisationEstimator`: JuMP optimiser configuration.

# Returns

  - `nothing`.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`set_near_optimal_centering_constraints!`](@ref)
  - [`solve_noc!`](@ref)
"""
function set_near_optimal_objective_function!(::UnconstrainedNearOptimalCentering,
                                              model::JuMP.Model,
                                              opt::BaseJuMPOptimisationEstimator)
    so = get_objective_scale(model)
    obj_expr = set_near_optimal_centering_constraints!(model, opt.wb)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
function set_near_optimal_objective_function!(::ConstrainedNearOptimalCentering,
                                              model::JuMP.Model,
                                              opt::BaseJuMPOptimisationEstimator)
    so = get_objective_scale(model)
    obj_expr = set_near_optimal_centering_constraints!(model, opt.wb)
    add_penalty_to_objective!(model, 1, obj_expr)
    add_custom_objective_term!(model, opt.ret, opt.cobj, obj_expr, opt, opt.pe)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
"""
    solve_noc!(noc, model, rk_opt, rt_opt, opt, args...)

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
                    rt_opt::Number, opt::BaseJuMPOptimisationEstimator, args...)
    JuMP.@expression(model, noc_rk, rk_opt)
    JuMP.@expression(model, noc_rt, rt_opt)
    set_near_optimal_objective_function!(noc.alg, model, opt)
    return optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:UnconstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator)
    retcodes = sizehint!(OptimisationReturnCode[], length(rk_opts))
    sols = sizehint!(JuMPOptimisationSolution[], length(rk_opts))
    JuMP.@variable(model, noc_rk in JuMP.Parameter(zero(eltype(rk_opts))))
    JuMP.@variable(model, noc_rt in JuMP.Parameter(zero(eltype(rt_opts))))
    set_near_optimal_objective_function!(noc.alg, model, opt)
    for (rk_opt, rt_opt) in zip(rk_opts, rt_opts)
        JuMP.set_parameter_value(noc_rk, rk_opt)
        JuMP.set_parameter_value(noc_rt, rt_opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
"""
    compute_ret_lbs(lbs::Frontier, rt_min::Number, rt_max::Number)

Compute return lower bounds for a `NearOptimalCentering` frontier sweep from pre-computed minimum and maximum return values.

Constructs a uniformly spaced range of `lbs.N` return targets between `rt_min` and `rt_max`.

# Arguments

  - `lbs::Frontier`: Frontier configuration specifying the number of points.
  - `rt_min::Number`: Minimum portfolio return (from the minimum-risk portfolio).
  - `rt_max::Number`: Maximum portfolio return (from the maximum-return portfolio).

# Returns

  - Range of `lbs.N` equally spaced return lower bounds.

# Related

  - [`compute_ret_lbs`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`solve_noc!`](@ref)
"""
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
    sc = get_constraint_scale(model)
    retcodes = Vector{OptimisationReturnCode}(undef, length(rk_opts))
    sols = Vector{JuMPOptimisationSolution}(undef, length(rk_opts))
    ret = get_ret(model)
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret - ret_lb_var) >= 0)
    JuMP.@variable(model, noc_rk in JuMP.Parameter(zero(eltype(rk_opts))))
    JuMP.@variable(model, noc_rt in JuMP.Parameter(zero(eltype(rt_opts))))
    set_near_optimal_objective_function!(noc.alg, model, opt)
    for (i, (rk_opt, rt_opt, lb)) in enumerate(zip(rk_opts, rt_opts, lbs))
        JuMP.set_parameter_value(ret_lb_var, lb)
        JuMP.set_parameter_value(noc_rk, rk_opt)
        JuMP.set_parameter_value(noc_rt, rt_opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
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
"""
    compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConstrainedNearOptimalCentering}, pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum, w_max::VecNum)

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
"""
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
    risk_frontier = compute_risk_ubs(model, noc, opt.pe, opt.fees, w_min, w_max)
    sc = get_constraint_scale(model)
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - ub) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(rk_opts))
    sols = sizehint!(JuMPOptimisationSolution[], length(rk_opts))
    JuMP.@variable(model, noc_rk in JuMP.Parameter(zero(eltype(rk_opts))))
    JuMP.@variable(model, noc_rt in JuMP.Parameter(zero(eltype(rt_opts))))
    set_near_optimal_objective_function!(noc.alg, model, opt)
    for (keys, ubs, rk_opt, rt_opt) in zip(pitrs[1], pitrs[2], rk_opts, rt_opts)
        for (key, ub) in zip(keys, ubs)
            JuMP.set_parameter_value(model[key], ub)
        end
        JuMP.set_parameter_value(noc_rk, rk_opt)
        JuMP.set_parameter_value(noc_rt, rt_opt)
        retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
        push!(retcodes, retcode)
        push!(sols, sol)
    end
    return retcodes, sols
end
function solve_noc!(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:ConstrainedNearOptimalCentering},
                    model::JuMP.Model, rk_opts::VecNum, rt_opts::VecNum,
                    opt::BaseJuMPOptimisationEstimator, rt_min::Number, rt_max::Number,
                    w_min::VecNum, w_max::VecNum, ::Val{true}, ::Val{true})
    lbs = compute_ret_lbs(model[:ret_frontier], rt_min, rt_max)
    risk_frontier = compute_risk_ubs(model, noc, opt.pe, opt.fees, w_min, w_max)
    sc = get_constraint_scale(model)
    for (keys, vals) in risk_frontier
        ub = model[keys[1]] = JuMP.@variable(model,
                                             set = JuMP.Parameter(zero(eltype(vals[2]))))
        d = ifelse(vals[3], 1, -1)
        model[keys[2]] = JuMP.@constraint(model, d * sc * (vals[1] - ub) <= 0)
    end
    itrs = [(Iterators.repeated(rkf[1][1], length(rkf[2][2])), rkf[2][2])
            for rkf in risk_frontier]
    pitrs = Iterators.product.(itrs...)
    retcodes = sizehint!(OptimisationReturnCode[], length(rt_opts) * length(rk_opts))
    sols = sizehint!(JuMPOptimisationSolution[], length(rt_opts) * length(rk_opts))
    ret = get_ret(model)
    JuMP.@variable(model, ret_lb_var in JuMP.Parameter(zero(eltype(lbs))))
    JuMP.@constraint(model, ret_lb, sc * (ret - ret_lb_var) >= 0)
    JuMP.@variable(model, noc_rk in JuMP.Parameter(zero(eltype(rk_opts))))
    JuMP.@variable(model, noc_rt in JuMP.Parameter(zero(eltype(rt_opts))))
    set_near_optimal_objective_function!(noc.alg, model, opt)
    for lb in lbs
        JuMP.set_parameter_value(ret_lb_var, lb)
        for (keys, ubs, rk_opt, rt_opt) in zip(pitrs[1], pitrs[2], rk_opts, rt_opts)
            for (key, ub) in zip(keys, ubs)
                JuMP.set_parameter_value(model[key], ub)
            end
            JuMP.set_parameter_value(noc_rk, rk_opt)
            JuMP.set_parameter_value(noc_rt, rt_opt)
            retcode, sol = optimise_JuMP_model!(model, noc, eltype(opt.pe.X))
            push!(retcodes, retcode)
            push!(sols, sol)
        end
    end
    return retcodes, sols
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
function _optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:UnconstrainedNearOptimalCentering},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    noc = reset_time_dependent_estimator(noc)
    (; w_opt, rk_opt, rt_opt, r, opt, w_min_retcode, w_opt_retcode, w_max_retcode) = near_optimal_centering_setup(noc,
                                                                                                                  rd;
                                                                                                                  dims = dims,
                                                                                                                  kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    JuMP.@expression(model, k, 1)
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    set_risk_constraints!(model, r, noc, opt.pe, nothing, nothing, opt.fees; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    set_return_constraints!(model, opt.ret, MinimumRisk(), opt.pe; rd = rd)
    noc_retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt)
    retcode = get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    return NearOptimalCenteringResult(;
                                      jr = JuMPOptimisationResult(; oe = typeof(noc),
                                                                  pa = ProcessedJuMPOptimiserAttributes(;
                                                                                                        pr = opt.pe,
                                                                                                        wb = opt.wb,
                                                                                                        lt = opt.lt,
                                                                                                        st = opt.st,
                                                                                                        lcsr = opt.lcse,
                                                                                                        ctr = opt.cte,
                                                                                                        gcardr = opt.gcarde,
                                                                                                        sgcardr = opt.sgcarde,
                                                                                                        smtx = opt.smtx,
                                                                                                        sgmtx = opt.sgmtx,
                                                                                                        slt = opt.slt,
                                                                                                        sst = opt.sst,
                                                                                                        sglt = opt.sglt,
                                                                                                        sgst = opt.sgst,
                                                                                                        tn = opt.tn,
                                                                                                        fees = opt.fees,
                                                                                                        plr = opt.ple,
                                                                                                        ret = opt.ret),
                                                                  retcode = retcode,
                                                                  sol = sol,
                                                                  model = ifelse(save,
                                                                                 model,
                                                                                 nothing)),
                                      w_min_retcode = w_min_retcode,
                                      w_opt_retcode = w_opt_retcode,
                                      w_max_retcode = w_max_retcode,
                                      noc_retcode = noc_retcode, fb = nothing)
end
function _optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:ConstrainedNearOptimalCentering},
                   rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                   str_names::Bool = false, save::Bool = true, kwargs...)
    noc = reset_time_dependent_estimator(noc)
    (; w_opt, rk_opt, rt_opt, r, opt, attrs, rt_min, rt_max, w_min, w_max, w_min_retcode, w_opt_retcode, w_max_retcode) = near_optimal_centering_setup(noc,
                                                                                                                                                       rd;
                                                                                                                                                       dims = dims,
                                                                                                                                                       kwargs...)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    JuMP.@expression(model, k, 1)
    set_w!(model, opt.pe.X, w_opt)
    set_weight_constraints!(model, opt.wb, opt.bgt, opt.sbgt)
    assemble_jump_model!(model, noc, opt, attrs, rd, r)
    noc_retcode, sol = solve_noc!(noc, model, rk_opt, rt_opt, opt, rt_min, rt_max, w_min,
                                  w_max, Val(haskey(model, :ret_frontier)),
                                  Val(haskey(model, :risk_frontier)))
    retcode = get_overall_retcode(w_min_retcode, w_opt_retcode, w_max_retcode, noc_retcode)
    return NearOptimalCenteringResult(;
                                      jr = JuMPOptimisationResult(; oe = typeof(noc),
                                                                  pa = attrs,
                                                                  retcode = retcode,
                                                                  sol = sol,
                                                                  model = ifelse(save,
                                                                                 model,
                                                                                 nothing)),
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

export NearOptimalCentering, UnconstrainedNearOptimalCentering,
       ConstrainedNearOptimalCentering, NearOptimalCenteringResult
