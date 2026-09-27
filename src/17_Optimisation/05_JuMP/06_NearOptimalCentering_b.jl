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
same [`assert_frontier_sweep_cap`](@ref) tail.

[`set_non_fixed_fees!`](@ref) runs before the risk build, as it does in
[`assemble_jump_model!`](@ref). It is what makes the model's return expression net of fees,
and the barrier compares that expression against the `noc_rt` target, which
[`near_optimal_centering_setup`](@ref) computes net of fees as well.
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
                   str_names::Bool = false, save::Bool = true, kwargs...)
    noc = reset_time_dependent_estimator(noc)
    setup = near_optimal_centering_setup(noc, rd; kwargs...)
    (; w_opt, r, opt, attrs, w_min_retcode, w_opt_retcode, w_max_retcode) = setup
    # The setup reduced its own locals. These are this method's, and they reach
    # `assemble_near_optimal_centering_model!` directly.
    noc, rd = investable_view(noc, rd, attrs.pr, attrs.imsk)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    set_model_scales!(model, opt.sc, opt.so)
    # Both variants fit on the prior's returns matrix. `rd` defaults to an empty
    # `ReturnsResult` on this path, so the prior is the only matrix that is always there.
    set_model_observations!(model, size(opt.pe.X, 1))
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
             rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...) -> NearOptimalCenteringResult

Run the Near Optimal Centering portfolio optimisation.

# Arguments

  - `noc`: The near optimal centering optimiser to use.
  - $(arg_dict[:rd]) If `isa(noc.opt.pe, AbstractPriorResult)`, `rd` is not necessary if doing a standalone optimisation, but may be required/desired by fallbacks and/or clusterisation.
  - `str_names`: Whether to use string names for the assets in the optimisation.
  - `save`: Whether to save the JuMP model in the optimisation result.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Validation

  - No field in the tree of `noc` holds an [`Online`](@ref). An `ArgumentError` naming the field is thrown otherwise, through [`assert_batch_entry`](@ref): a plain `optimise` is a batch fit, and a wrapper resolves only at the warm-up of the fold loop's online arm.

# Related

  - [`NearOptimalCentering`](@ref)
  - [`NearOptimalCenteringResult`](@ref)
"""
function optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                            <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; str_names::Bool = false, save::Bool = true, kwargs...)
    assert_batch_entry(noc, "`optimise`")
    return _optimise(noc, rd; str_names = str_names, save = save, kwargs...)
end

@pipe_delegates NearOptimalCentering opt
@pipe_route_sigma_ucs NearOptimalCentering
