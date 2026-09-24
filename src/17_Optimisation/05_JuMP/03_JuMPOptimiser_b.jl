"""
    assert_universe_axis_order(sets::Option{<:UniverseSets}, rd::ReturnsResult) -> Nothing

Assert that every axis [`UniverseSets`](@ref) declares agrees, name for name and in order, with the returns data it is about to be used against.

Nothing else in the library ties a declared universe to the data's column order. Names resolve to *positions* in `sets.dict[sets.xkey]`, and those positions index the columns of `pr.X`; if the two disagree the model is still feasible and still solves — it just constrains the wrong assets. Converting the one silent-wrong-answer failure mode into an error is the whole point, which is why this runs even though no supported path is expected to trip it.

Both axes are checked, but only where both sides exist: `rd.nx` and `rd.nf` are optional on [`ReturnsResult`](@ref), and the factor axis is optional on [`UniverseSets`](@ref). The factor check is redundant with the one [`constraint_space_basis`](@ref) makes against the loadings and costs nothing when a caller copies `rd.nf` into the dict; the asset check is new behaviour on a path that has nothing to do with factors, and it may surface pre-existing misconfigurations — that is intended.

`_update_asset_sets` rebuilds `sets.dict[sets.xkey]` from `rdo.nx` before a [`NestedClustered`](@ref) outer solve, and [`port_opt_view`](@ref) slices the asset axis alongside the data, so the synthetic-universe and subset paths satisfy this by construction.

# Related

  - [`UniverseSets`](@ref)
  - [`misaligned_axis_msg`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
"""
function assert_universe_axis_order(sets::Option{<:UniverseSets}, rd::ReturnsResult)
    if isnothing(sets)
        return nothing
    end
    for (axis, key, names, sym) in
        (("asset", sets.xkey, rd.nx, "nx"), ("factor", sets.tfkey, rd.nf, "nf"))
        if isnothing(names) || !haskey(sets.dict, key)
            continue
        end
        declared = sets.dict[key]
        @argcheck(length(declared) == length(names),
                  DimensionMismatch(misaligned_axis_msg(declared, names, axis, key, sym)))
        @argcheck(declared == names,
                  ArgumentError(misaligned_axis_msg(declared, names, axis, key, sym)))
    end
    return nothing
end
"""
    processed_jump_optimiser_attributes(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        dims::Int = 1
    ) -> ProcessedJuMPOptimiserAttributes

Compute all constraint and prior results needed for model assembly.

Resolves every estimator field of `opt` against `rd` — running priors, weight bounds,
thresholds, linear constraints, centrality, cardinality, turnover, fees, and phylogeny
— and returns the fully processed bundle as a
[`ProcessedJuMPOptimiserAttributes`](@ref). The result is consumed directly by
[`assemble_jump_model!`](@ref) and also stored in the per-optimiser Result struct, so
processing happens exactly once per `optimise` call.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - $(arg_dict[:rd])
  - `dims::Int = 1`: Observation dimension passed to the prior estimator.

# Returns

  - [`ProcessedJuMPOptimiserAttributes`](@ref): Fully resolved constraint and prior bundle.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`processed_jump_optimiser`](@ref)
"""
function processed_jump_optimiser_attributes(opt::JuMPOptimiser, rd::ReturnsResult;
                                             dims::Int = 1, kwargs...)
    rd = returns_result_picker(rd, opt.brt)
    assert_universe_axis_order(opt.sets, rd)
    pr = prior(opt.pe, rd; dims = dims)
    # Resolve the fee on the caller's own universe, before the door below narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset, and a carrier keyed by name cannot resolve at all once its `w` sits on the
    # complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    imsk = investable_mask(pr)
    fees = investable_fees_view(fees_constraints(opt.fees, opt.sets;
                                                 datatype = eltype(pr.X),
                                                 strict = opt.strict), imsk, pr.X)
    # The prior fits on the coverage universe and returns a result on the full asset
    # universe, where an asset it could not estimate carries `NaN`. Reduce once, here:
    # every builder below then states its constraint over the investable assets alone, on
    # inputs `port_opt_view` has sliced by the same index. The weights are expanded back
    # in `JuMPOptimisationResult`.
    _, pr, opt, rd = investable_reduction(imsk, pr, opt, rd)
    X = pr.X
    datatype = eltype(X)
    wb = weight_bounds_constraints(opt.wb, opt.sets; N = size(X, 2), strict = opt.strict,
                                   datatype = datatype)
    lt = threshold_constraints(opt.lt, opt.sets; datatype = datatype, strict = opt.strict)
    st = threshold_constraints(opt.st, opt.sets; datatype = datatype, strict = opt.strict)
    lcsr = linear_constraints(opt.lcse, opt.sets; datatype = datatype, strict = opt.strict,
                              rr = pr.rr, rd = rd)
    ctr = centrality_constraints(opt.cte, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                 x_src = opt.x_src, strict = opt.strict, kwargs...)
    gcardr = linear_constraints(opt.gcarde, opt.sets; datatype = Int, strict = opt.strict)
    sgcardr = linear_constraints(opt.sgcarde, opt.sets; datatype = Int, strict = opt.strict)
    # A name-keyed estimator follows the door: a name that left resolves on the
    # Non-Investable Axis. A precomputed constraint cannot, because its `A` is bound to its
    # columns by position. Say so here rather than let the model meet two numbers.
    # `sgcardr` is not checked: its columns are the rows of `sgmtx`, one for each sub-group,
    # and a departed asset removes a column of `sgmtx`, never a sub-group.
    if !isnothing(imsk)
        N = size(X, 2)
        assert_investable_constraint_width(lcsr, N, "lcse")
        assert_investable_constraint_width(gcardr, N, "gcarde")
    end
    if opt.smtx === opt.sgmtx
        smtx = sgmtx = asset_sets_matrix(opt.smtx, opt.sets)
    else
        smtx = asset_sets_matrix(opt.smtx, opt.sets)
        sgmtx = asset_sets_matrix(opt.sgmtx, opt.sets)
    end
    if opt.slt === opt.sglt
        slt = sglt = threshold_constraints(opt.slt, opt.sets; datatype = datatype,
                                           strict = opt.strict)
    else
        slt = threshold_constraints(opt.slt, opt.sets; datatype = datatype,
                                    strict = opt.strict)
        sglt = threshold_constraints(opt.sglt, opt.sets; datatype = datatype,
                                     strict = opt.strict)
    end
    if opt.sst === opt.sgst
        sst = sgst = threshold_constraints(opt.sst, opt.sets; datatype = datatype,
                                           strict = opt.strict)
    else
        sst = threshold_constraints(opt.sst, opt.sets; datatype = datatype,
                                    strict = opt.strict)
        sgst = threshold_constraints(opt.sgst, opt.sets; datatype = datatype,
                                     strict = opt.strict)
    end
    tn = turnover_constraints(opt.tn, opt.sets; datatype = datatype, strict = opt.strict)
    plr = phylogeny_constraints(opt.ple, pr; iv = rd.iv, ivpa = rd.ivpa, rd = rd,
                                x_src = opt.x_src, kwargs...)
    ret = factory(opt.ret, pr)
    return ProcessedJuMPOptimiserAttributes(; pr = pr, wb = wb, lt = lt, st = st,
                                            lcsr = lcsr, ctr = ctr, gcardr = gcardr,
                                            sgcardr = sgcardr, smtx = smtx, sgmtx = sgmtx,
                                            slt = slt, sst = sst, sglt = sglt, sgst = sgst,
                                            tn = tn, fees = fees, plr = plr, ret = ret,
                                            sca = opt.sca, imsk = imsk)
end
"""
    no_bounds_optimiser(opt::JuMPOptimiser, args...) -> JuMPOptimiser

Return a copy of `opt` with every return term replaced by its unbounded variant.

Only the `ret` field changes. [`no_bounds_returns_estimator`](@ref) clears the term's own
return lower bound `settings.lb`, and drops its uncertainty set when the forwarded flag is
`false`; every other field of `opt`, weight bounds and risk bounds included, is carried
through unchanged. A vector of terms is stripped term by term.

[`NearOptimalCentering`](@ref) is the caller: its anchor sub-problems must range over the
whole frontier, so the return bound that shapes the user's own problem is removed from them.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - `args...`: Forwarded to [`no_bounds_returns_estimator`](@ref). The one caller forwards
    `noc.ucs_flag`, which keeps the uncertainty set when `true`.

# Returns

  - `JuMPOptimiser`: Optimiser whose return terms carry no bounds.

# Examples

```jldoctest
julia> opt = JuMPOptimiser(; slv = Solver(; solver = nothing));

julia> PortfolioOptimisers.no_bounds_optimiser(opt) isa JuMPOptimiser
true
```

# Related

  - [`no_bounds_returns_estimator`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function no_bounds_optimiser(opt::JuMPOptimiser, args...)
    pnames = Tuple(setdiff(fieldnames(typeof(opt)), (:ret,)))
    return JuMPOptimiser(; ret = no_bounds_returns_estimator(opt.ret, args...),
                         NamedTuple{pnames}(getfield.(opt, pnames))...)
end
"""
    jump_optimiser_from_attributes(
        opt::JuMPOptimiser,
        attrs::ProcessedJuMPOptimiserAttributes
    ) -> JuMPOptimiser

Repackage a [`ProcessedJuMPOptimiserAttributes`](@ref) into a [`JuMPOptimiser`](@ref).

Maps result-named fields onto the optimiser's estimator-named slots (`lcsr` → `lcse`,
`plr` → `ple`, `pr` → `pe`, …) and carries all remaining settings through from `opt`.
Used where a processed optimiser object is needed for inner sub-problems (e.g.
[`near_optimal_centering_setup`](@ref)) while the same `attrs` is also passed directly to
[`assemble_jump_model!`](@ref) — so processing happens once and is never round-tripped.

# Arguments

  - `opt::JuMPOptimiser`: Source optimiser supplying solver, scalar settings, and any
    fields not present in `attrs`.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle.

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from `attrs`.

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function jump_optimiser_from_attributes(opt::JuMPOptimiser,
                                        attrs::ProcessedJuMPOptimiserAttributes)
    # The remap is derived by reflection so it can never silently drop a field (a hand-written
    # pull-through once omitted `scard`, disabling sub-group cardinality in constrained-NOC;
    # fixed in 9ff28d6). Every `opt` field flows through by name (`base`); every `attrs` result
    # field overrides its optimiser slot. Six result fields carry a different name than their
    # slot — listed in `rename` — the other twelve match by name and map automatically.
    #
    # There is deliberately no inverse. A head that needs the bundle back keeps the `attrs` it
    # already holds (both `NearOptimalCentering` heads do), because a reverse hand list is the
    # same drop the comment above records — only pointing the other way.
    rename = (; pr = :pe, lcsr = :lcse, ctr = :cte, gcardr = :gcarde, sgcardr = :sgcarde,
              plr = :ple)
    of = fieldnames(JuMPOptimiser)
    # `imsk` is the one bundle field that is not a constraint result: it is the Investable
    # Mask the reduction derived, and `JuMPOptimiser` declares no slot for it. A sub-problem
    # built from this optimiser reads an already-reduced prior, so it derives no mask of its
    # own and needs none carried in.
    af = filter(!=(:imsk), fieldnames(ProcessedJuMPOptimiserAttributes))
    base = NamedTuple{of}(getfield.(opt, of))
    overrides = NamedTuple{map(f -> get(rename, f, f), af)}(getfield.(attrs, af))
    return JuMPOptimiser(; merge(base, overrides)...)
end
"""
    processed_jump_optimiser(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        dims::Int = 1
    ) -> JuMPOptimiser

Build a fully-processed [`JuMPOptimiser`](@ref) from raw configuration and returns data.

Calls [`processed_jump_optimiser_attributes`](@ref) then repackages the result via
[`jump_optimiser_from_attributes`](@ref). Used where a processed optimiser is needed for
inner sub-problems (e.g. [`NearOptimalCentering`](@ref)) while the `attrs` bundle is
reused directly by [`assemble_jump_model!`](@ref).

# Arguments

  - `opt::JuMPOptimiser`: Raw optimiser configuration.
  - $(arg_dict[:rd])
  - `dims::Int = 1`: Observation dimension passed to the prior estimator.

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from processed results.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; dims::Int = 1,
                                  kwargs...)
    attrs = processed_jump_optimiser_attributes(opt, rd; dims = dims, kwargs...)
    return jump_optimiser_from_attributes(opt, attrs)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add weight bound and budget constraints to `model` from the optimiser's own budget group.

Every JuMP optimiser head reaches [`set_weight_constraints!`](@ref) through this method, so
the whole budget group (`bgt`, `sbgt`, `gbgt`) travels as one object. A budget field added
to [`JuMPOptimiser`](@ref) is therefore read in one place rather than at every head, which
is how the gross budget once reached [`MeanRisk`](@ref) alone and was silently dropped by
the other four heads.

`long = true` forbids negative weight bounds, so the long/short decomposition is never
built and `sbgt` and `gbgt` cannot apply; they are still forwarded, because the bound check
is what rejects the combination.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `opt::JuMPOptimiser`: Optimiser supplying the budget group.
  - `long::Bool = false`: When `true`, raises an error if any weight bound is negative.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
"""
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds, opt::JuMPOptimiser,
                                 long::Bool = false)
    return set_weight_constraints!(model, wb, opt.bgt, opt.sbgt, long; gbgt = opt.gbgt)
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add risk-measure constraints and scalarise the combined risk expression in `model`.

One step of [`assemble_jump_model!`](@ref), dispatched on `r`: when `r` is `nothing`
(e.g. [`RelaxedRiskBudgeting`](@ref), whose risk lives in its head) this is a no-op.

This is the one place the fee argument is positional. Every head reaches
[`set_risk_constraints!`](@ref) through it, so a head cannot mis-order the list and lose the
fees: unconstrained [`NearOptimalCentering`](@ref) did exactly that while it inlined the step.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model (mutated in place).
  - `r`: Risk measure(s), or `nothing` to skip this step entirely.
  - `optimiser`: Dispatch object for [`set_risk_constraints!`](@ref).
  - `opt::JuMPOptimiser`: Supplies the scalariser `opt.sca`.
  - `pr`: Prior result passed to risk builders.
  - `pl`: Phylogeny result passed to risk builders.
  - `fees`: Fees result passed to risk builders.
  - $(arg_dict[:b1_opt])
  - `rd`: Returns result forwarded as a keyword argument to risk builders.

# Returns

  - `nothing`.

# Related

  - [`assemble_jump_model!`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`scalarise_risk_expression!`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`FactorRiskContribution`](@ref)
"""
function set_risk_and_scalarise!(::JuMP.Model, ::Nothing, optimiser, opt, pr, pl,
                                 fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing;
                                 kwargs...)
    # The no-op carries the same positional list as the builder below, not an `args...` tail:
    # a tail is ambiguous with a typed tail on the sibling method, and it accepts a call the
    # builder would reject.
    return nothing
end
function set_risk_and_scalarise!(model::JuMP.Model, r, optimiser, opt, pr, pl,
                                 fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing; rd)
    set_risk_constraints!(model, r, optimiser, pr, pl, fees, b1; rd = rd)
    scalarise_risk_expression!(model, opt.sca)
    return nothing
end

"""
    assemble_jump_model!(
        model::JuMP.Model,
        optimiser::JuMPOptimisationEstimator,
        opt::JuMPOptimiser,
        attrs::ProcessedJuMPOptimiserAttributes,
        rd::ReturnsResult,
        r::Option{<:RM_VecRM} = nothing,
        obj::ObjectiveFunction = MinimumRisk(),
        b1::Option{<:MatNum} = nothing,
        sdp_asset_phylogeny::Bool = true
    ) -> Nothing

Run the invariant model-assembly sequence shared by all single-JuMP-model optimisers.

Executes the constraint-builder pipeline — from `set_linear_weight_constraints!` through
`add_custom_constraint!` — that sits between the per-optimiser *head* (weight variables)
and *tail* (objective + solve). The head must have populated Model State (`w`/`k` variables)
before calling this function. See `Model Assembly` in `CONTEXT.md` and
`0008-jump-model-assembly.md`.

The tail of the sequence is [`assert_frontier_sweep_cap`](@ref): both frontier registries are
complete here and no sweep solve has started, so this is the one point at which the **total**
sweep — the product across every swept return term and every swept risk measure — is in hand
and can be capped.

# Arguments

  - $(arg_dict[:model])
  - `optimiser::JuMPOptimisationEstimator`: Dispatch object for risk, tracking, and custom
    constraint builders.
  - `opt::JuMPOptimiser`: Supplies scalar settings (`l2c`, `lpc`, `linfc`, `l1`, `l2`,
    `lp`, `linf`, `card`, `scard`, `tr`, `ccnt`, `sca`, `ss`). The four regularisation
    coefficients and the three norm ceilings are the one place a **Calibration Rule**
    reaches the model builders, so they are resolved against `attrs.pr` here rather than
    by [`processed_jump_optimiser_attributes`](@ref): the bundle carries no slot for them
    and this is where both the prior result and the optimiser are in hand. A ceiling is
    read against one norm order, so each of the three sites states its own order in the
    [`CalibrationContext`](@ref) it resolves the slot against.
  - `attrs::ProcessedJuMPOptimiserAttributes`: Pre-computed constraint and prior bundle
    produced by [`processed_jump_optimiser_attributes`](@ref).
  - $(arg_dict[:rd])
  - `r::Option{<:RM_VecRM} = nothing`: Risk measure(s), or `nothing` to skip risk
    constraints and scalarisation (the [`RelaxedRiskBudgeting`](@ref) path).
  - `obj::ObjectiveFunction = MinimumRisk()`: Objective used by the return constraints.
  - $(arg_dict[:b1_opt])
  - `sdp_asset_phylogeny::Bool = true`: Whether to apply the standard asset-space SDP phylogeny
    constraints. [`FactorRiskContribution`](@ref) passes `false` and applies its own
    factor-space variant in its tail instead.

# Returns

  - `nothing`. Mutates `model` in place.

# Related

  - [`ProcessedJuMPOptimiserAttributes`](@ref)
  - [`processed_jump_optimiser_attributes`](@ref)
  - [`set_risk_and_scalarise!`](@ref)
  - [`assert_frontier_sweep_cap`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`FactorRiskContribution`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assemble_jump_model!(model::JuMP.Model, optimiser::JuMPOptimisationEstimator,
                              opt::JuMPOptimiser, attrs::ProcessedJuMPOptimiserAttributes,
                              rd::ReturnsResult, r::Option{<:RM_VecRM} = nothing,
                              obj::ObjectiveFunction = MinimumRisk(),
                              b1::Option{<:MatNum} = nothing,
                              sdp_asset_phylogeny::Bool = true)
    (; pr, wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, tn, fees, plr, ret) = attrs
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, opt.card, gcardr, plr, lt, st, fees, opt.ss, opt.xbgt)
    set_smip_constraints!(model, wb, opt.scard, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                          opt.ss)
    set_turnover_constraints!(model, tn)
    set_tracking_error_constraints!(model, pr, opt.tr, optimiser, plr, fees, b1; rd = rd)
    # The three norm ceilings each take a Calibration Rule, and each resolves here. A
    # ceiling is read against one norm order, which belongs to the constraint and not to
    # the rule, so each site states its own order in the context. `lpc` holds estimators
    # and carries one order per term, so it goes through `norm_ceiling_factory`.
    #
    # These four slots are the ones that reach the model raw. Every other slot is rebuilt
    # by its owner's constructor, which states the range of the number. There is no term to
    # rebuild here, so each of the four builders states the range of the number it is
    # handed, whatever produced it.
    l2c = resolve_calibration_slot(opt.l2c, :l2c, pr, pr.w, opt.slv,
                                   CalibrationContext(; p = 2))
    linfc = resolve_calibration_slot(opt.linfc, :linfc, pr, pr.w, opt.slv,
                                     CalibrationContext(; p = Inf))
    set_weight_norm_2_constraints!(model, l2c)
    set_weight_norm_p_constraints!(model, norm_ceiling_factory(opt.lpc, pr, opt.slv))
    set_weight_norm_inf_constraints!(model, linfc)
    # The four regularisation coefficients are the ambiguity radii of the four ground
    # metrics, so each takes a Calibration Rule and each resolves here, against the
    # optimisation's own prior result and effective solver. `l2` and `lp` hold estimators,
    # so their own `factory` methods resolve them; `l1` and `linf` hold the number itself.
    set_l1_regularisation!(model, resolve_calibration_slot(opt.l1, :l1, pr, pr.w, opt.slv))
    set_l2_regularisation!(model, factory(opt.l2, pr, opt.slv))
    set_lp_regularisation!(model, factory(opt.lp, pr, opt.slv))
    set_linf_regularisation!(model,
                             resolve_calibration_slot(opt.linf, :linf, pr, pr.w, opt.slv))
    set_non_fixed_fees!(model, fees)
    set_risk_and_scalarise!(model, r, optimiser, opt, pr, plr, fees, b1; rd = rd)
    set_return_constraints!(model, ret, obj, pr; rd = rd)
    # After the return constraints, so a `MaximumRatio` has chosen its form, and before
    # every semidefinite phylogeny, including a programme set's in `add_custom_constraint!`.
    mark_risk_minimised!(model, obj)
    set_iplg_constraints!(model, plr)
    if sdp_asset_phylogeny
        set_sdp_phylogeny_constraints!(model, plr)
    end
    add_custom_constraint!(model, opt.ccnt, optimiser, attrs)
    assert_frontier_sweep_cap(model)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Route a mean uncertainty set into a [`JuMPOptimiser`](@ref)'s return estimator.

One of the two [Routing Targets](@ref PIPELINE_ROUTING_TARGETS) that names no plain field: the set lands in `ret.ucs`, and only an [`ArithmeticReturn`](@ref) can bound expected returns, so any other return estimator is an error rather than a silent drop.

A **vector** of return terms is refused for the same reason, from the other side: one set is a neighbourhood of one quantity, so broadcasting it across *k* terms would apply a ball fitted on one fit to every other one. Name the set on the term it belongs to instead.

Internal machinery — not part of the user-facing API.

# Related

  - [`pipe_route`](@ref)
  - [`route_sigma_ucs`](@ref)
"""
function pipe_route(cfg::JuMPOptimiser, ::Val{:mu_ucs}, v)
    @argcheck(isa(cfg.ret, ArithmeticReturn),
              ArgumentError(if isa(cfg.ret, VecJRE)
                                "cannot route a mean uncertainty set into $(length(cfg.ret)) return terms: a set is a neighbourhood of the one quantity it was calibrated on, so it cannot be broadcast across terms. Name the set in the `ucs` field of the term it belongs to."
                            else
                                "cannot route a mean uncertainty set into a $(Base.typename(typeof(cfg.ret)).wrapper); expected returns uncertainty requires an ArithmeticReturn return estimator"
                            end))
    return Accessors.set(cfg, Accessors.PropertyLens{:ret}(),
                         Accessors.set(cfg.ret, Accessors.PropertyLens{:ucs}(), v))
end
pipe_accepts(::JuMPOptimiser, ::Val{:mu_ucs})::Bool = true
