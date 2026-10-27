"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for Near Optimal Centering algorithm variants.

# Related

  - [`ConstrainedNearOptimalCentering`](@ref)
  - [`UnconstrainedNearOptimalCentering`](@ref)
"""
abstract type NearOptimalCenteringAlgorithm <: OptimisationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Constrained Near Optimal Centering algorithm.

It selects the branch whose centring model holds every constraint of `opt`. The model is built
by the shared [`assemble_jump_model!`](@ref), so every setting of the [`JuMPOptimiser`](@ref)
reaches it, and ``\\mathcal{W}`` of [`NearOptimalCentering`](@ref) is the whole feasible set.
Use this variant when a setting must bind on the centring solve itself.
[`UnconstrainedNearOptimalCentering`](@ref) lists the settings that the default variant does
not read.

# Related

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`UnconstrainedNearOptimalCentering`](@ref)
  - [`assemble_near_optimal_centering_model!`](@ref)
"""
struct ConstrainedNearOptimalCentering <: NearOptimalCenteringAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Unconstrained Near Optimal Centering algorithm. This is the default `alg` of
[`NearOptimalCentering`](@ref).

It selects the branch whose centring model holds a part of the constraints of `opt`. The model
holds the weight bounds and the budgets, the risk expression, the return expression and the
fees that are not fixed. The other constraint and penalty builders do not run.
"Unconstrained" names that omission.

The constructor reads and checks the omitted settings, and it does not refuse them. They still
have an effect. The anchor of `obj` is a [`MeanRisk`](@ref) problem that builds every
constraint, so an omitted setting changes that anchor, and through it the bounds of the region.
The anchors of minimum risk and of maximum return solve with the same constraints, without
the bounds of the risk measures and of the return terms. An omitted setting reaches no model at
all only when `w_min`, `w_opt` and `w_max` are all given, because then no sub-problem runs.

# Settings the centring model reads

`pe`, `slv`, `wb`, `bgt`, `sbgt`, `gbgt`, `sc`, `so`, `sca`, the part of `fees` that is not
fixed, and `ret`. The model reads the copy of `ret` without bounds, see
[`no_bounds_optimiser`](@ref).

# Settings the centring model does not read

`lcse`, `cte`, `card`, `gcarde`, `scard`, `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`,
`sglt`, `sgst`, `lt`, `st`, `xbgt`, `ss`, `tn`, `tr`, `ple`, `l2c`, `lpc`, `linfc`, `l1`,
`l2`, `linf`, `lp`, `ccnt`, `cobj`, and the fixed part of `fees`. A fixed fee is charged for
each position held, so it needs the binary variables of `set_mip_constraints!`.

`cobj` is the one omission that changes the objective and not the feasible set.
[`set_near_optimal_objective_function!`](@ref) adds no Objective Penalty to the barrier, so a
Custom Objective Term prices the anchor sub-problems and not the centring solve.

# Related

  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`ConstrainedNearOptimalCentering`](@ref)
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

A property that is a field of this type reads the field. Every other property, the weights `w` and the return code `retcode` included, reads the embedded [`JuMPOptimisationResult`](@ref) `jr`.

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
    $(field_dict[:fb_res])
    """
    fb
    function NearOptimalCenteringResult(jr::JuMPOptimisationResult, r::BaseRM_VecBaseRM,
                                        w_min_retcode::OptimisationReturnCode,
                                        w_opt_retcode::OptRetCode_VecOptRetCode,
                                        w_max_retcode::OptimisationReturnCode,
                                        noc_retcode::OptRetCode_VecOptRetCode,
                                        fb::Option{<:OptE_Opt_FbChain})
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
                                    fb::Option{<:OptE_Opt_FbChain})::NearOptimalCenteringResult
    return NearOptimalCenteringResult(jr, r, w_min_retcode, w_opt_retcode, w_max_retcode,
                                      noc_retcode, fb)
end
"""
    set_retcode(res::NearOptimalCenteringResult, retcode::OptRetCode_VecOptRetCode)

Rebuild a [`NearOptimalCenteringResult`](@ref) with a different return code.

`retcode` is not a field of this result. It reads the embedded [`JuMPOptimisationResult`](@ref) `jr`, so the rebuild replaces `jr`. The return codes of the three anchor sub-problems and of the centring solve stay unchanged.

# Arguments

  - `res`: Result to rebuild.
  - `retcode`: Return code, or one per member of the population.

# Returns

  - [`NearOptimalCenteringResult`](@ref): The result, with the new return code.

# Related

  - [`set_retcode`](@ref)
  - [`mark_ruined_members`](@ref)
  - [`NearOptimalCenteringResult`](@ref)
"""
function set_retcode(res::NearOptimalCenteringResult, retcode::OptRetCode_VecOptRetCode)
    return NearOptimalCenteringResult(set_retcode(res.jr, retcode), res.r,
                                      res.w_min_retcode, res.w_opt_retcode,
                                      res.w_max_retcode, res.noc_retcode, res.fb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the static defaults of the [`NearOptimalCentering`](@ref) fields that can hold a [`TimeDependent`](@ref).

The check of the constructor, [`assert_time_dependent_substitution`](@ref), and [`time_dependent_field_defaults`](@ref) both read it, so the value of a field outside a fold loop is stated once. A field whose static default is `nothing` is not in the tuple.

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

Finds the analytic centre of the near optimal region around the portfolio of an objective.

`NearOptimalCentering` solves three anchor portfolios as [`MeanRisk`](@ref) problems: the portfolio of minimum risk, the portfolio of maximum return, and the portfolio of `obj`. The anchors set a floor on the return and a ceiling on the risk. The portfolios that satisfy both bounds are the near optimal region. The result is the analytic centre of that region, the point that maximises the product of the margins of its inequalities.

# Mathematical definition

The near optimal region holds the portfolios whose return is at least ``\\epsilon_{1}`` and whose risk is at most ``\\epsilon_{2}``:

```math
\\begin{align}
\\mathrm{ret} &\\geq \\epsilon_{1}\\,, \\\\
R(\\boldsymbol{w}) &\\leq \\epsilon_{2}\\,, \\\\
\\boldsymbol{w} &\\in \\mathcal{W}\\,.
\\end{align}
```

The two bounds are the return and the risk of the anchor of `obj`, moved by one part in ``m`` of the span of the other two anchors:

```math
\\begin{align}
\\epsilon_{1} &= \\mathrm{ret}(\\boldsymbol{w}_{\\mathrm{opt}}) - \\frac{\\mathrm{ret}(\\boldsymbol{w}_{\\max}) - \\mathrm{ret}(\\boldsymbol{w}_{\\min})}{m}\\,, \\\\
\\epsilon_{2} &= R(\\boldsymbol{w}_{\\mathrm{opt}}) + \\frac{R(\\boldsymbol{w}_{\\max}) - R(\\boldsymbol{w}_{\\min})}{m}\\,.
\\end{align}
```

The result minimises the logarithmic barrier of the region:

```math
\\begin{align}
\\underset{\\boldsymbol{w}}{\\min} \\; &- \\ln\\left(\\mathrm{ret} - \\epsilon_{1}\\right) - \\ln\\left(\\epsilon_{2} - R(\\boldsymbol{w})\\right) - \\sum_{i \\in \\mathcal{L}} \\ln\\left(w_{i} - w_{l,i}\\right) - \\sum_{i \\in \\mathcal{U}} \\ln\\left(w_{u,i} - w_{i}\\right) \\\\
\\text{s.t.} \\quad &\\boldsymbol{w} \\in \\mathcal{W}\\,.
\\end{align}
```

Section 11.4 of [cajas2025](@cite) states the region for a book with ``\\boldsymbol{w} \\geq \\boldsymbol{0}`` and a unit budget, and its barrier holds the term ``-\\ln(w_{i})`` of the lower bound. Its Listing 11.13 also holds the term ``-\\ln(1 - w_{i})`` of the upper bound. The barrier above has one term for each finite weight bound, so it is the barrier of Section 8.5.3 of [boydvandenberghe2004](@cite) for the bounds of `opt.wb`. With the default bounds ``0`` and ``1``, the two barriers are equal. An asset whose two bounds are equal has a fixed weight. That is an equality, so it has no term.

A positive scale of one inequality does not move its analytic centre. So the scale of the constraints, `opt.sc`, does not move the result, and the combination weight of a single risk measure does not move it either.

A [`Frontier`](@ref) in `opt` makes `obj` give one anchor ``\\boldsymbol{w}_{\\mathrm{opt}}`` for each point of the sweep, and the result holds one centre for each point.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:W_feas]) [`UnconstrainedNearOptimalCentering`](@ref) and [`ConstrainedNearOptimalCentering`](@ref) state which constraints of `opt` it holds.
  - $(math_dict[:R_w]) The anchors read it with [`expected_risk`](@ref).
  - $(math_dict[:ret_model]) The anchors read it with [`expected_return`](@ref).
  - ``\\boldsymbol{w}_{\\min}``, ``\\boldsymbol{w}_{\\mathrm{opt}}``, ``\\boldsymbol{w}_{\\max}``: The anchors of minimum risk, of `obj`, and of maximum return, the fields `w_min`, `w_opt` and `w_max` when they are given.
  - ``m``: Number of bins, `bins`. The default is ``T / N``, so a wider universe or a shorter history gives a wider region.
  - ``w_{l,i}``, ``w_{u,i}``: Lower and upper weight bounds of asset ``i``, from `opt.wb`.
  - ``\\mathcal{L}``, ``\\mathcal{U}``: The assets whose lower bound, and whose upper bound, is finite and differs from the other bound, see [`barrier_weight_bounds`](@ref).
  - ``T``, ``N``: Number of observations and number of assets.

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

Keywords correspond to the struct's fields. A field of the type [`TD`](@ref), [`TD_Option`](@ref) or [`TDO_Option`](@ref) can hold a [`TimeDependent`](@ref) schedule in place of a static value. The risk measures, the objective, the anchor weights, the initial weights and the fallback define the problem, so a cross-validation fold loop resolves them for each fold. An `optimise` outside a fold loop uses the static default of each field. `bins`, `ucs_flag` and `alg` control the solve, and they stay static.

## Validation

  - If `r` is a vector: `!isempty(r)`.
  - If `w_min` is a vector: `!isempty(w_min)`.
  - If `w_min_ini` is a vector: `!isempty(w_min_ini)`.
  - If `w_opt` is a vector: `!isempty(w_opt)`.
  - If `w_opt_ini` is a vector: `!isempty(w_opt_ini)`.
  - If `w_max` is a vector: `!isempty(w_max)`.
  - If `w_max_ini` is a vector: `!isempty(w_max_ini)`.
  - If `bins` is a number: `isfinite(bins) && bins > 0`.
  - A schedule in `fb`: `bind !== :nearest`.
  - The constructor refuses a risk expression that is identically zero, and a return expression that is identically zero. The barrier reads both expressions, see [`assert_risk_measure_required`](@ref) and [`assert_return_term_required`](@ref).
  - Each schedule, with the static defaults in the place of the other schedules, must build an estimator, see [`assert_time_dependent_substitution`](@ref).
  - If `r` holds a [`DataBenchmarkTracking`](@ref): `opt.pe` must not state `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).
  - A risk measure whose expression is a `JuMP.QuadExpr` logs a warning, because the barrier needs an affine or a conic risk. The variance with SDP constraints is the trace of a matrix, which is affine, so it works.
  - A part that states no source and reads the raw returns, beside a part that states `x_src = ExcessSource()`, logs one warning that names its path, see [`warn_raw_readers`](@ref). The constructor then builds the estimator as configured.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `opt`: Recursively updated through [`factory`](@ref).
  - `r`: Recursively updated through [`factory`](@ref).
  - `fb`: Recursively updated through [`factory`](@ref).

## View parameters

`NearOptimalCentering` defines its own [`port_opt_view`](@ref) method. It does not derive one from the field tags.

  - The method reads the returns matrix `X` as its third argument. When `opt.pe` holds a prior result, the method uses `opt.pe.X` in place of `X`. So the views of the children read the observations of the prior, and not the matrix of the caller.
  - `opt` and `r` recurse through [`port_opt_view`](@ref) with that matrix.
  - `fb` recurses through [`view_child`](@ref) with that matrix.
  - The six weight vectors `w_min`, `w_min_ini`, `w_opt`, `w_opt_ini`, `w_max` and `w_max_ini` are sliced to the selected assets.
  - `obj`, `bins`, `ucs_flag` and `alg` pass through unchanged.

# Related

  - [`optimise`](@ref)
  - [`NearOptimalCenteringResult`](@ref)
  - [`RiskJuMPOptimisationEstimator`](@ref)
  - [`MeanRisk`](@ref)
  - [`NearOptimalCenteringAlgorithm`](@ref)
  - [`near_optimal_centering_setup`](@ref)
  - [`set_near_optimal_centering_constraints!`](@ref)
  - [`NearOptimalSetup`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2019noc])
  - $(ref_dict[:degraaf2016])
  - $(ref_dict[:cajas2025]) Section 11.4, Equations 11.30 to 11.32. $(ref_dict[:no_original_source])
  - $(ref_dict[:boydvandenberghe2004]) Section 8.5.3, Problem 8.18.
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
    Number of bins ``m``. The near optimal region extends one part in ``m`` of the span of the anchors past the anchor of `obj`. If `nothing`, ``m = T / N``.
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
    Near Optimal Centering algorithm variant. It selects the constraints that the centring model builds, so it selects the settings of `opt` that the model reads. [`ConstrainedNearOptimalCentering`](@ref) reads all of them. The default [`UnconstrainedNearOptimalCentering`](@ref) reads the settings that its docstring lists.
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
                @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.",
                      maxlog = log_limit(:warn), _id = @log_id(typeof(r)))
            end
        elseif isa(r, QuadExpressionRiskMeasures)
            @warn("Risk measures that produce JuMP.QuadExpr risk expressions are not guaranteed to work. The variance with SDP constraints works because the risk measure is the trace of a matrix, an affine expression.",
                  maxlog = log_limit(:warn), _id = @log_id(typeof(r)))
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
        assert_data_benchmark_prior(r, opt.pe, :r, :NearOptimalCentering, "opt.pe")
        est = new{typeof(opt), typeof(r), typeof(obj), typeof(bins), typeof(w_min),
                  typeof(w_min_ini), typeof(w_opt), typeof(w_opt_ini), typeof(w_max),
                  typeof(w_max_ini), typeof(ucs_flag), typeof(alg), typeof(fb)}(opt, r, obj,
                                                                                bins, w_min,
                                                                                w_min_ini,
                                                                                w_opt,
                                                                                w_opt_ini,
                                                                                w_max,
                                                                                w_max_ini,
                                                                                ucs_flag,
                                                                                alg, fb)
        warn_raw_readers(est)
        return est
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
# `solve_noc!` sweeps every frontier of the model.
function assert_frontier_sweep_owner(::JuMP.Model, ::NearOptimalCentering)::Nothing
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` if a part of `opt` needs the previous portfolio weights.

The parts are each field that holds a [`TimeDependent`](@ref) schedule, the JuMP optimiser `opt.opt`, the risk measures `opt.r` and the fallback `opt.fb`.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`PreviousWeightsFunction`](@ref)
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

Return a copy of [`NearOptimalCentering`](@ref) for the assets `i`.

# Algorithm

 1. If `noc.opt.pe` holds a prior result, replace `X` with its matrix `noc.opt.pe.X`.
 2. View `opt` and `r` through [`port_opt_view`](@ref), and `fb` through [`view_child`](@ref), with that matrix.
 3. Slice the six weight vectors `w_min`, `w_min_ini`, `w_opt`, `w_opt_ini`, `w_max` and `w_max_ini` to `i`.
 4. Build the estimator, with `obj`, `bins`, `ucs_flag` and `alg` unchanged.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`port_opt_view`](@ref)
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
                                w_max = w_max, w_max_ini = w_max_ini,
                                fb = view_child(noc.fb, i, X))
end
"""
    near_optimal_centering_risks(scalariser, r, pr, fees, slv, w_min, w_opt, w_max)

Return the risk of the three anchor portfolios of [`NearOptimalCentering`](@ref).

The barrier compares these values with the model's `:risk` expression, so each value has the units of that expression. The model drops the combination weight `scale` of a single measure, so the method for one measure does not apply it. The model combines a vector of measures with their weights, so the method for a vector applies each `scale` and then the scalariser.

# Algorithm

 1. Fit the measures to the prior and the solvers with [`factory`](@ref).
 2. For one measure, return the risk of `w_min`, of `w_opt` and of `w_max` from [`expected_risk`](@ref).
 3. For a vector of measures, compute the three risks of each measure times its `settings.scale`, and combine the measures with `scalariser`.

# Arguments

  - `scalariser`: Scalariser of a vector of measures, such as [`SumScalariser`](@ref). The method for one measure does not read it.
  - `r`: Risk measure or vector of risk measures.
  - `pr`: Prior result.
  - `fees`: Fees, or `nothing`.
  - `slv`: Solver or vector of solvers.
  - $(arg_dict[:w_min])
  - $(arg_dict[:w_opt])
  - $(arg_dict[:w_max])

# Returns

  - `(risk_min, risk_opt, risk_max)`: The risks of the three anchors. `risk_opt` is a vector when `w_opt` holds one anchor for each point of a frontier sweep.

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

The data that [`near_optimal_centering_setup`](@ref) computes before the centring model of [`NearOptimalCentering`](@ref) is built.

It holds the three anchors, the bounds ``\\epsilon_{1}`` and ``\\epsilon_{2}`` of the near optimal region, the returns of the anchors, the risk measures and the optimiser that the centring model reads, and the return codes of the anchor sub-problems.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`NearOptimalCentering`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
@concrete struct NearOptimalSetup <: AbstractResult
    """
    $(field_dict[:w_opt])
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
    $(field_dict[:w_min])
    """
    w_min
    """
    $(field_dict[:w_max])
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
    function NearOptimalSetup(w_opt::VecNum_VecVecNum, rk_opt::Union{Number, VecNum},
                              rt_opt::Union{Number, VecNum}, rt_min::Union{Number, VecNum},
                              rt_max::Union{Number, VecNum}, rt_ends::Option{<:VecPair},
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
function NearOptimalSetup(; w_opt::VecNum_VecVecNum, rk_opt::Union{Number, VecNum},
                          rt_opt::Union{Number, VecNum}, rt_min::Union{Number, VecNum},
                          rt_max::Union{Number, VecNum},
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
    near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult; kwargs...)

Solve the anchors of [`NearOptimalCentering`](@ref), and compute the bounds of its near optimal region.

# Algorithm

 1. Process the optimiser into its attributes with [`processed_jump_optimiser_attributes`](@ref), reduce `noc` and `rd` to the investable universe, and bind a data benchmark to the rows of the view.
 2. If an anchor of minimum risk or of maximum return is missing, if the variant is [`UnconstrainedNearOptimalCentering`](@ref), or if a return term holds a [`Frontier`](@ref), make the copies of the risk measures and of the optimiser without bounds. `ucs_flag` selects whether they keep their uncertainty sets.
 3. Solve each missing anchor as a [`MeanRisk`](@ref) problem. The anchors of minimum risk and of maximum return solve with the copies without bounds, and the anchor of `obj` solves with `r` and `opt` as they are.
 4. Compute the risks of the anchors with [`near_optimal_centering_risks`](@ref), and their returns with [`expected_return`](@ref).
 5. For each return term that holds a [`Frontier`](@ref), compute its span with [`return_term_ends`](@ref).
 6. Compute ``\\epsilon_{2} = R(\\boldsymbol{w}_{\\mathrm{opt}}) + (R(\\boldsymbol{w}_{\\max}) - R(\\boldsymbol{w}_{\\min})) / m`` and ``\\epsilon_{1} = \\mathrm{ret}(\\boldsymbol{w}_{\\mathrm{opt}}) - (\\mathrm{ret}(\\boldsymbol{w}_{\\max}) - \\mathrm{ret}(\\boldsymbol{w}_{\\min})) / m``, with ``m = T / N`` when `bins` is `nothing`.
 7. For the unconstrained variant, give the centring model the copies without bounds.

# Arguments

  - `noc::NearOptimalCentering`: The estimator.
  - $(arg_dict[:rd])
  - `kwargs...`: Keyword arguments passed to [`processed_jump_optimiser_attributes`](@ref).

# Returns

  - `setup::NearOptimalSetup`: The anchors, the bounds of the region and the return codes of the anchor sub-problems.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`NearOptimalSetup`](@ref)
  - [`near_optimal_centering_risks`](@ref)
"""
function near_optimal_centering_setup(noc::NearOptimalCentering, rd::ReturnsResult;
                                      kwargs...)
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
    attrs = processed_jump_optimiser_attributes(noc.opt, rd; kwargs...)
    # The corner solves below run the head's own `opt`, `r` and anchor weights against
    # `rd`, so the head is reduced before any of them, and before the optimiser is
    # repackaged from the bundle. `_optimise` takes the same view of its own locals.
    noc, rd = investable_view(noc, rd, attrs.pr, attrs.imsk)
    # A data benchmark binds to the rows of the view, before the corner solves and the
    # risks of the corners read it.
    noc = bind_data_benchmark(noc, rd)
    r = bind_data_benchmark(r, rd)
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

Return the indices of the return terms whose `settings.lb` is a [`Frontier`](@ref).

A term whose `settings.lb` is a vector of numbers states its levels, so it needs no corner
solve and is not in the answer. An empty answer means that no corner solve runs.

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

Return the span of each return term that holds a [`Frontier`](@ref).

The low end of the span of term ``i`` is its return at the anchor of minimum risk `w_min`.
The high end is its return at the portfolio that maximises term ``i`` alone, so each term
costs one solve. The anchor of maximum return maximises the combined return, so its value
of term ``i`` depends on the `scale` of the other terms, and it can be below the low end.

# Algorithm

 1. For each index ``i`` in `swept`, solve a [`MeanRisk`](@ref) problem with the objective [`MaximumElementReturn`](@ref)`(i)`, the risk measures `nb_r`, the optimiser `nb_opt` and the initial weights `w_ini`.
 2. Make the pair `i => (rt_min_i, rt_max_i)` of the returns of term ``i`` at `w_min` and at that solution.

# Arguments

  - `swept::VecInt`: Indices of the return terms that hold a [`Frontier`](@ref).
  - `ret`: The return terms of the processed optimiser.
  - `nb_r`: The risk measures without bounds.
  - `nb_opt::JuMPOptimiser`: The optimiser without return bounds.
  - $(arg_dict[:rd])
  - `pr::AbstractPriorResult`: Prior result.
  - `fees`: Fees, or `nothing`.
  - `w_min::VecNum`: Anchor of minimum risk.
  - `w_ini`: Initial weights of each solve, or `nothing`.

# Validation

  - Each solve must succeed, `ArgumentError` otherwise.

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
    ends = Pair{Int, Tuple{Number, Number}}[]
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
    barrier_weight_bounds(w_lb::VecNum, w_ub::VecNum)

Return the assets whose lower bound and whose upper bound enter the barrier of
[`NearOptimalCentering`](@ref).

A weight bound is an inequality of the near optimal region, so its margin is a term of the
barrier. An infinite bound states no inequality, and it has no term. An asset with
``w_{l,i} = w_{u,i}`` has a fixed weight. That pair is an equality, and an equality has no
term in the barrier: the two margins are zero at every feasible point, so their logarithms
have no finite value.

# Algorithm

 1. Make `il`, the indices ``i`` with a finite ``w_{l,i}`` and ``w_{l,i} \\neq w_{u,i}``.
 2. Make `iu`, the indices ``i`` with a finite ``w_{u,i}`` and ``w_{l,i} \\neq w_{u,i}``.

# Arguments

  - `w_lb::VecNum`: Lower weight bounds, one per asset.
  - `w_ub::VecNum`: Upper weight bounds, one per asset.

# Returns

  - `(il, iu)::Tuple{Vector{Int}, Vector{Int}}`: The assets of the lower terms and of the
    upper terms.

# Related

  - [`set_near_optimal_centering_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function barrier_weight_bounds(w_lb::VecNum, w_ub::VecNum)
    il = [i for i in eachindex(w_lb, w_ub) if isfinite(w_lb[i]) && w_lb[i] != w_ub[i]]
    iu = [i for i in eachindex(w_lb, w_ub) if isfinite(w_ub[i]) && w_lb[i] != w_ub[i]]
    return il, iu
end
"""
    set_near_optimal_centering_constraints!(model::JuMP.Model, wb::WeightBounds)

Add the logarithmic barrier of [`NearOptimalCentering`](@ref) to the model, and return it.

Each logarithm is a variable bounded above by an exponential cone, so the barrier is a conic objective. The objective maximises each of these variables, so at the optimum each one equals its logarithm. The barrier reads the risk ceiling `noc_rk` and the return floor `noc_rt` that the caller registers, and [`barrier_weight_bounds`](@ref) selects the weight bounds that have a term.

# JuMP formulation

## Variables

  - `log_risk`, `log_ret`: ``t_{R}``, ``t_{\\mathrm{ret}}``.
  - `log_w`: ``\\boldsymbol{t}_{l}``, one entry for each asset in ``\\mathcal{L}``.
  - `log_delta_w`: ``\\boldsymbol{t}_{u}``, one entry for each asset in ``\\mathcal{U}``.

## Constraints

  - `clog_risk`: ``s\\,(t_{R}, 1, \\epsilon_{2} - R(\\boldsymbol{w})) \\in \\mathcal{K}_{\\exp}``, which is ``t_{R} \\leq \\ln(\\epsilon_{2} - R(\\boldsymbol{w}))``.
  - `clog_ret`: ``s\\,(t_{\\mathrm{ret}}, 1, \\mathrm{ret} - \\epsilon_{1}) \\in \\mathcal{K}_{\\exp}``, which is ``t_{\\mathrm{ret}} \\leq \\ln(\\mathrm{ret} - \\epsilon_{1})``.
  - `clog_w[i]`: ``s\\,(t_{l,i}, 1, w_{i} - w_{l,i}) \\in \\mathcal{K}_{\\exp}`` for ``i \\in \\mathcal{L}``.
  - `clog_delta_w[i]`: ``s\\,(t_{u,i}, 1, w_{u,i} - w_{i}) \\in \\mathcal{K}_{\\exp}`` for ``i \\in \\mathcal{U}``.

## Expressions

  - `obj_expr`: ``-\\left(t_{\\mathrm{ret}} + t_{R} + \\sum_{i \\in \\mathcal{L}} t_{l,i} + \\sum_{i \\in \\mathcal{U}} t_{u,i}\\right)``, the barrier.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:R_w])
  - $(math_dict[:ret_model])
  - ``\\epsilon_{1}``, ``\\epsilon_{2}``: The return floor `noc_rt` and the risk ceiling `noc_rk` of the near optimal region.
  - ``w_{l,i}``, ``w_{u,i}``: Lower and upper weight bounds of asset ``i``, from `wb`.
  - ``\\mathcal{L}``, ``\\mathcal{U}``: The assets that [`barrier_weight_bounds`](@ref) returns.
  - ``s``: The constraint scale, `sc`.
  - $(math_dict[:K_exp])

# Arguments

  - $(arg_dict[:model])
  - `wb::WeightBounds`: Weight bounds of the model, one entry for each asset.

# Returns

  - `obj_expr`: The barrier, registered under that name.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
"""
function set_near_optimal_centering_constraints!(model::JuMP.Model, wb::WeightBounds)
    w = get_w(model)
    sc = get_constraint_scale(model)
    w_lb = wb.lb
    w_ub = wb.ub
    risk = get_risk(model)
    ret = get_ret(model)
    rk = shared_get(model, :noc_rk)
    rt = shared_get(model, :noc_rt)
    il, iu = barrier_weight_bounds(w_lb, w_ub)
    JuMP.@variables(model, begin
                        log_ret
                        log_risk
                        log_w[il]
                        log_delta_w[iu]
                    end)
    JuMP.@constraints(model,
                      begin
                          clog_risk,
                          [sc * log_risk, sc, sc * (rk - risk)] in
                          JuMP.MOI.ExponentialCone()
                          clog_ret,
                          [sc * log_ret, sc, sc * (ret - rt)] in JuMP.MOI.ExponentialCone()
                          clog_w[i = il],
                          [sc * log_w[i], sc, sc * (w[i] - w_lb[i])] in
                          JuMP.MOI.ExponentialCone()
                          clog_delta_w[i = iu],
                          [sc * log_delta_w[i], sc, sc * (w_ub[i] - w[i])] in
                          JuMP.MOI.ExponentialCone()
                      end)
    JuMP.@expression(model, obj_expr,
                     -(log_ret +
                       log_risk +
                       sum(log_w[i] for i in il) +
                       sum(log_delta_w[i] for i in iu)))
    return obj_expr
end
"""
    set_near_optimal_objective_function!(alg, model, noc, attrs)

Set the Near Optimal Centering objective function in the JuMP model.

The objective minimises ``s_{o}`` times the barrier of [`set_near_optimal_centering_constraints!`](@ref). For [`ConstrainedNearOptimalCentering`](@ref), it also holds the Custom Objective Terms of `opt.cobj` and the [Objective Penalty](@ref add_to_objective_penalty!).

The centring solve minimises the barrier, so the objective that it gives to a Custom Objective Term is [`MinimumRisk`](@ref), not `noc.obj`. `noc.obj` is the objective of the anchor sub-problem.

[`UnconstrainedNearOptimalCentering`](@ref) calls neither [`add_custom_objective_term!`](@ref) nor [`add_penalty_to_objective!`](@ref), so `opt.cobj` prices the anchor sub-problems and not the centring solve. No builder of that variant adds to the Objective Penalty, so the penalty is empty, and its omission changes no number.

# Algorithm

 1. Add the barrier with [`set_near_optimal_centering_constraints!`](@ref).
 2. For [`ConstrainedNearOptimalCentering`](@ref), add the Custom Objective Terms with [`add_custom_objective_term!`](@ref), and add the Objective Penalty with [`add_penalty_to_objective!`](@ref).
 3. Set the objective to minimise ``s_{o}`` times that expression, where ``s_{o}`` is the objective scale `so`.

# JuMP formulation

## Objective

  - ``\\min\\; s_{o}\\, b``, where ``b`` is `obj_expr` of [`set_near_optimal_centering_constraints!`](@ref). For [`ConstrainedNearOptimalCentering`](@ref), ``b`` also holds the Custom Objective Terms and the Objective Penalty.

# Arguments

  - `alg`: [`UnconstrainedNearOptimalCentering`](@ref) or [`ConstrainedNearOptimalCentering`](@ref).
  - $(arg_dict[:model])
  - `noc::NearOptimalCentering`: The estimator, which the Custom Objective Terms receive.
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser.

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

Register the bounds of the near optimal region, set the barrier objective, and solve the centring model.

# Algorithm

The method dispatches on the type of the bounds, on the variant, and on the frontiers that the model registers.

 1. With one risk ceiling and one return floor, register them as the constant expressions `noc_rk` and `noc_rt`, set the objective with [`set_near_optimal_objective_function!`](@ref), and solve once.
 2. With a vector of bounds, one for each point of a sweep, register `noc_rk` and `noc_rt` as parameters with [`set_noc_anchor_parameters!`](@ref).
 3. For [`UnconstrainedNearOptimalCentering`](@ref), solve once for each point, with the bounds of that point.
 4. For [`ConstrainedNearOptimalCentering`](@ref), resolve the return frontier with [`compute_ret_lbs`](@ref) and the risk frontier with [`compute_risk_ubs`](@ref), make their parameters, and solve once for each point of the product of the two axes, with the bounds of that point.

# JuMP formulation

## Expressions

  - `noc_rk`: ``\\epsilon_{2}``, the risk ceiling, a constant in step 1.
  - `noc_rt`: ``\\epsilon_{1}``, the return floor, a constant in step 1.

[`set_noc_anchor_parameters!`](@ref) registers the two expressions of a sweep.

# Arguments

  - `noc::NearOptimalCentering`: The estimator.
  - $(arg_dict[:model])
  - `rk_opt`: Risk ceiling ``\\epsilon_{2}``, or one for each point of a sweep.
  - `rt_opt`: Return floor ``\\epsilon_{1}``, or one for each point of a sweep.
  - `opt::BaseJuMPOptimisationEstimator`: The optimiser of the centring model.
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser.
  - `args...`: For the constrained variant, the returns of the anchors, the spans of [`return_term_ends`](@ref), the anchors of minimum risk and of maximum return, and one `Val` flag for each frontier registry.

# Returns

  - `(retcode, sol)` for one solve, or `(retcodes, sols)` for a sweep.

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
    p_rk, p_rt, rk_units, rt_units = set_noc_anchor_parameters!(model, noc, attrs, rk_opts,
                                                                rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X), length(rk_opts)) do i
        return set_noc_anchor!(p_rk, p_rt, rk_units, rt_units, i)
    end
end
"""
    set_noc_anchor_parameters!(model, noc, attrs, rk_opts, rt_opts)

Register the risk ceiling and the return floor of a swept centring model as parameters, and set the objective.

`noc_rk` and `noc_rt` are the risk ceiling ``\\epsilon_{2}`` and the return floor ``\\epsilon_{1}`` of the near optimal region. A sweep moves them from point to point. So each one is a parameter times a constant, the model is built once, and only the values of the parameters change between solves.

The constant is the [`parameter_scale`](@ref) of the bounds, so each parameter takes values in ``[-1, 1]``. A parameter with the size of a risk leaves the exponential cones of the barrier badly scaled.

# Algorithm

 1. Compute the constants ``m_{R}`` and ``m_{\\mathrm{ret}}`` with [`parameter_scale`](@ref).
 2. Register the parameters `noc_rk_unit` and `noc_rt_unit`, and the expressions `noc_rk` ``= m_{R}`` `noc_rk_unit` and `noc_rt` ``= m_{\\mathrm{ret}}`` `noc_rt_unit`.
 3. Set the objective with [`set_near_optimal_objective_function!`](@ref).
 4. Divide the bounds by the constants.

# JuMP formulation

## Variables

  - `noc_rk_unit`, `noc_rt_unit`: ``p_{R}``, ``p_{\\mathrm{ret}}``, parameters with the start value zero. [`set_noc_anchor!`](@ref) sets their values.

## Expressions

  - `noc_rk`: ``\\epsilon_{2} = m_{R}\\, p_{R}``.
  - `noc_rt`: ``\\epsilon_{1} = m_{\\mathrm{ret}}\\, p_{\\mathrm{ret}}``.

# Arguments

  - $(arg_dict[:model])
  - `noc::NearOptimalCentering`: The estimator, which selects the objective.
  - `attrs::ProcessedJuMPOptimiserAttributes`: The processed attributes of the optimiser.
  - `rk_opts::VecNum`: The risk ceilings, one for each point of the sweep.
  - `rt_opts::VecNum`: The return floors, one for each point of the sweep.

# Returns

  - `(p_rk, p_rt, rk_units, rt_units)`: The two parameters, and the bounds in the units of the parameters, which [`set_noc_anchor!`](@ref) writes.

# Related

  - [`set_noc_anchor!`](@ref)
  - [`set_near_optimal_objective_function!`](@ref)
  - [`solve_noc!`](@ref)
  - [`parameter_scale`](@ref)
"""
function set_noc_anchor_parameters!(model::JuMP.Model, noc::NearOptimalCentering,
                                    attrs::ProcessedJuMPOptimiserAttributes,
                                    rk_opts::VecNum, rt_opts::VecNum)
    m_rk, m_rt = parameter_scale(rk_opts), parameter_scale(rt_opts)
    p_rk = JuMP.@variable(model, base_name = "noc_rk_unit",
                          set = JuMP.Parameter(zero(eltype(rk_opts))))
    p_rt = JuMP.@variable(model, base_name = "noc_rt_unit",
                          set = JuMP.Parameter(zero(eltype(rt_opts))))
    JuMP.@expression(model, noc_rk, m_rk * p_rk)
    JuMP.@expression(model, noc_rt, m_rt * p_rt)
    set_near_optimal_objective_function!(noc.alg, model, noc, attrs)
    return p_rk, p_rt, rk_opts ./ m_rk, rt_opts ./ m_rt
end
"""
    set_noc_anchor!(p_rk, p_rt, rk_units, rt_units, i)

Set the risk ceiling and the return floor of the centring model to those of sweep point `i`.

It is the hook that [`frontier_sweep!`](@ref) calls before each solve of a swept `NearOptimalCentering`. [`near_optimal_centering_setup`](@ref) computes the bounds of point `i` from the anchor of `obj` at point `i` of the same sweep, and `i` is the flat index that [`frontier_sweep_axes`](@ref) counts. So each point is centred on the near optimal region of its own anchor.

# Arguments

  - `p_rk`: The parameter of the risk ceiling.
  - `p_rt`: The parameter of the return floor.
  - `rk_units::VecNum`: The risk ceilings in the units of `p_rk`, in the flat order of the sweep.
  - `rt_units::VecNum`: The return floors in the units of `p_rt`, in the flat order of the sweep.
  - `i::Integer`: The flat index of the sweep point.

# Returns

  - `nothing`.

# Related

  - [`set_noc_anchor_parameters!`](@ref)
  - [`frontier_sweep!`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function set_noc_anchor!(p_rk, p_rt, rk_units::VecNum, rt_units::VecNum, i::Integer)
    JuMP.set_parameter_value(p_rk, rk_units[i])
    JuMP.set_parameter_value(p_rt, rt_units[i])
    return nothing
end
"""
    compute_ret_lbs(ret_frontier::VecPair, ::Nothing)

Return a `NearOptimalCentering` return frontier unchanged.

`nothing` means that no return term holds a [`Frontier`](@ref). Each term states its levels
as a vector of numbers, so there is no span to resolve.

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

Resolve the return frontier of a `NearOptimalCentering` from the spans of its setup.

Unlike the method of [`MeanRisk`](@ref), this method solves no corner.
[`near_optimal_centering_setup`](@ref) already solved the anchor of minimum risk and one
maximum-return problem for each term that holds a [`Frontier`](@ref), so this method only
turns each span into its levels.

# Algorithm

 1. Keep an entry whose levels are a vector of numbers.
 2. For an entry of term ``i`` that holds a [`Frontier`](@ref) of `N` points, find the span `i => (rt_min_i, rt_max_i)` in `rt_ends`.
 3. Replace the [`Frontier`](@ref) with `range(rt_min_i, rt_max_i; length = N)`.

# Arguments

  - `ret_frontier::VecPair`: The `:ret_frontier` Model State entry.
  - `rt_ends::VecPair`: The spans of the terms, as `i => (rt_min_i, rt_max_i)`.

# Validation

  - Each term that holds a [`Frontier`](@ref) has a span in `rt_ends`, `ArgumentError` otherwise.

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
    p_rk, p_rt, rk_units, rt_units = set_noc_anchor_parameters!(model, noc, attrs, rk_opts,
                                                                rt_opts)
    return frontier_sweep!(model, noc, eltype(opt.pe.X),
                           frontier_sweep_axes(ret_axis, nothing)) do i
        return set_noc_anchor!(p_rk, p_rt, rk_units, rt_units, i)
    end
end
function rebuild_risk_frontier(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any, <:Any, <:Any, <:Any, <:Any,
                                                         <:Any,
                                                         <:ConstrainedNearOptimalCentering},
                               pr::AbstractPriorResult, fees::Option{<:Fees},
                               risk_frontier::VecPair, w_min::VecNum, w_max::VecNum,
                               idx::VecInt, args...)
    risk_frontier = copy(risk_frontier)
    for i in idx
        r = factory(risk_frontier[i].second[4], pr, noc.opt.slv)
        risk_frontier[i] = _rebuild_risk_frontier(pr, fees, r, risk_frontier, w_min, w_max,
                                                  i)
    end
    return risk_frontier
end

export NearOptimalCentering, UnconstrainedNearOptimalCentering,
       ConstrainedNearOptimalCentering, NearOptimalCenteringResult
