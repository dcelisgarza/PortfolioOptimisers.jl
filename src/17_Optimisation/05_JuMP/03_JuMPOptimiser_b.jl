"""
    resolve_jump_optimiser(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        rmsk::Option{BitVector} = nothing,
        blocks::Tuple = (),
        kwargs...
    ) -> ResolvedJuMPOptimiser

Resolves a [`JuMPOptimiser`](@ref) against the returns data `rd`, to the [`ResolvedJuMPOptimiser`](@ref) that a JuMP model reads.

The function fits the prior, resolves the fees and the return terms, and resolves the [`ConstraintBlock`](@ref) of `opt` in two stages. Stage 1, [`resolve_constraint_block_names`](@ref), resolves the names of the block over the universe of the caller. The reduction to the Investable Mask then views the block, the prior and the rows. Stage 2, [`resolve_constraint_block`](@ref), resolves the block against the reduced prior and rows, to its [`ResolvedConstraintBlock`](@ref). [`assemble_jump_model!`](@ref) reads the bundle, and the Result of the head stores it in `jor`, so the resolution runs once for each `optimise` call.

The entry takes a list of blocks. The block of `opt` is the first, and the keyword `blocks` gives the others. Each entry of `blocks` is the [`ProgrammeAllocationSet`](@ref) of an online head and the rows `X` that the head holds, see [`fit_allocation_set_block`](@ref). Each block resolves against its own prior on its own rows, because a merge of two blocks changes the moments that their readers and their ceilings read. The Investable Mask of the bundle is the intersection of the masks of every block, so an asset that one block cannot read departs from all of them. The bundle then holds the vector of the block Results in `cbr`, the block of `opt` first. The entry takes `blocks` by name, so the keyword does not reach the readers through `kwargs...`.

A distance reader of a block, its `cte` or its `ple`, can refuse an asset that its prior keeps, see [`DistanceReaderRefusalError`](@ref). One refusal branch covers every reader of every block. On a refusal, the function fits again from the same prior results, without that asset, with the mask of [`distance_reader_refusal_mask`](@ref) in `rmsk`. Two readers can refuse different assets, and the fit again meets the refusal of the other one. The readers of `opt` refuse under `opt.strict`, and the readers of a set refuse under the `strict` flag of the online head that holds the set, [`projection_step_strict`](@ref).

**Breaking change.** The verb was `processed_jump_optimiser_attributes`, and it returned `ProcessedJuMPOptimiserAttributes`, which held one field for each resolved constraint. The constraints are now in the block Result in `cbr`. The calibrated slots `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp` and `linf` resolve at stage 2, and not in [`assemble_jump_model!`](@ref). A method of a custom constraint that reads `attrs.cbr` reads a vector when the head has more than one block, and [`owner_block`](@ref) gives the block of `opt`.

# Algorithm

 1. Fit the prior `pr` of `opt.pe` on `rd`. Fit the prior of each entry of `blocks` on its rows with [`fit_allocation_set_block`](@ref).
 2. Compose the Investable Mask `imsk` with [`distance_reader_entry_mask`](@ref). Intersect it with the mask of each entry of `blocks`, [`allocation_set_entry_mask`](@ref). On a fit again, take `rmsk`.
 3. Resolve the fees on the universe of the caller with [`fees_constraints`](@ref), and place them on the axes that the mask leaves with [`investable_fees_view`](@ref).
 4. Resolve stage 1 of the block of `opt` over the universe of the caller, with [`resolve_constraint_block_names`](@ref).
 5. Reduce the prior, the optimiser and `rd` to the mask with [`investable_reduction`](@ref). The reduction views the block of stage 1, and it declares the Non-Investable Axis on its `sets`.
 6. Resolve stage 2 of the reduced block with [`resolve_constraint_block`](@ref), through [`distance_reader_call`](@ref). Resolve each entry of `blocks` at the mask with [`resolve_allocation_set_block`](@ref).
 7. If a block gives a [`DistanceReaderRefusalError`](@ref), fit again from step 2 with the mask of [`distance_reader_refusal_mask`](@ref) and the fitted entries of step 1, and return its value. The first refusal in the order of the blocks wins.
 8. Resolve the return terms against the prior with [`factory`](@ref), and return the bundle. Its `cbr` is the block Result of `opt` when `blocks` is empty, and the vector of every block Result otherwise.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - $(arg_dict[:rd])
  - `rmsk`: The mask of a fit again after a refusal, or `nothing` on the first fit.
  - `blocks`: The blocks after the block of `opt`. Each entry is a `Tuple` `(set, X)` or `(set, X, pr)`, as [`fit_allocation_set_block`](@ref) states. The empty `Tuple` gives one block.
  - `kwargs...`: Keyword arguments passed to the centrality and phylogeny constraint estimators of `opt`.

# Validation

  - The rules of [`resolve_constraint_block`](@ref), such as [`assert_investable_constraint_width`](@ref) for a precomputed constraint.
  - Under `opt.strict`, no distance reader of `opt` refuses an asset. Under [`projection_step_strict`](@ref), no distance reader of a set refuses an asset, and the prior of a set prices every asset that the head keeps. The rules of [`distance_reader_refusal_mask`](@ref) and [`allocation_set_entry_mask`](@ref) raise otherwise.

# Returns

  - `jor::ResolvedJuMPOptimiser`: The resolved bundle.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
  - [`ResolvedConstraintBlock`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`resolve_allocation_set_block`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`processed_jump_optimiser`](@ref)
"""
function resolve_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult;
                                rmsk::Option{BitVector} = nothing, blocks::Tuple = (),
                                kwargs...)
    # `rd` keeps its raw `X` and `B`. Each reader below that reads the returns states its
    # own source, and subtracts the benchmark itself, at the point of use.
    pr = prior(opt.pe, rd)
    # The prior of each set is fitted once. A fit again reads the fitted entries.
    blocks = map(fit_allocation_set_block, blocks)
    # Resolve the fee on the caller's universe before `investable_reduction` narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset. A liquidation charge keyed by name cannot resolve at all once its `w` sits on
    # the complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    # An asset that a `FeatureDistance` of the clustering cannot read in the window of its
    # Asset Panel departs with the non-investable assets (`feature_readable_mask`). So does
    # an asset with a gap in the rows that a plain graph reader reads (`graph_reader_mask`).
    # A fit again after a refusal of a distance reader passes its mask as `rmsk`, which
    # already holds the masks of every block.
    imsk = distance_reader_entry_mask((opt.ple, opt.cte), pr, rd, opt.strict; rmsk = rmsk)
    if isnothing(rmsk)
        imsk = foldl(allocation_set_entry_mask, blocks; init = imsk)
    end
    datatype = eltype(pr.X)
    fees = investable_fees_view(fees_constraints(opt.fees, opt.sets; datatype = datatype,
                                                 strict = opt.strict), imsk, pr.X)
    # Stage 1 resolves the names over the caller's universe, as the fee does, and the
    # reduction views its values. An online head runs the same stage once for each fold,
    # before it knows the mask of a step, so both owners order the names and the reduction
    # by one rule (ADR 0237).
    cbe = resolve_constraint_block_names(opt.cbe, opt.strict, datatype)
    _, prr, optr, rdr = investable_reduction(imsk, pr,
                                             rebuild_estimator(opt, (; cbe = cbe)), rd)
    cbrs = (distance_reader_call(resolve_constraint_block, optr.cbe, prr, rdr;
                                 slv = optr.slv, strict = optr.strict, kwargs...),
            map(b -> resolve_allocation_set_block(b, imsk), blocks)...)
    i = findfirst(x -> x isa DistanceReaderRefusalError, cbrs)
    if !isnothing(i)
        # A distance reader that refuses an asset the prior kept raises before it builds its
        # graph. Fit again from the same prior results, without that asset. The block of
        # `opt` refuses under the flag of the head, and the block of a set under the flag of
        # the online head that holds the set.
        strict = isone(i) ? opt.strict : projection_step_strict()
        return resolve_jump_optimiser(rebuild_estimator(opt, (; pe = pr)), rd;
                                      rmsk = distance_reader_refusal_mask(cbrs[i], imsk, rd,
                                                                          strict),
                                      blocks = blocks, kwargs...)
    end
    cbr = isempty(blocks) ? first(cbrs) : [cbrs...]
    return ResolvedJuMPOptimiser(; pr = prr, fees = fees, ret = factory(optr.ret, prr),
                                 sca = optr.sca, imsk = imsk, cbr = cbr)
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
        attrs::ResolvedJuMPOptimiser
    ) -> JuMPOptimiser

Repackages a [`ResolvedJuMPOptimiser`](@ref) into a [`JuMPOptimiser`](@ref).

The function maps the fields of the bundle onto the fields of the optimiser: `pr` onto `pe`, and `fees`, `ret` and `sca` onto the fields of the same name. It maps the block Result of the owner, [`owner_block`](@ref) of `cbr`, onto the block `cbe` with [`constraint_block_from_resolved`](@ref), which holds the rename table of the block (`lcsr` to `lcse`, `ctr` to `cte`, …). Every other field of `opt` keeps its value. The list of fields comes from the fields of the bundle, so the function drops no field. The Investable Mask `imsk` has no field on the optimiser, and the function leaves it out.

[`near_optimal_centering_setup`](@ref) calls it to build the optimiser of its sub-problems, while it gives the same `attrs` to [`assemble_jump_model!`](@ref). So the resolution runs once, and no sub-problem resolves a slot again. An optimiser holds one block, so the blocks after the first reach a sub-problem as the keyword `blocks` of the entry, from [`constraint_block_steps`](@ref).

# Arguments

  - `opt::JuMPOptimiser`: Source optimiser supplying solver, scalar settings, and any
    fields not present in `attrs`.
  - `attrs::ResolvedJuMPOptimiser`: Pre-computed constraint and prior bundle.

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from `attrs`.

# Related

  - [`ResolvedJuMPOptimiser`](@ref)
  - [`processed_jump_optimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`assemble_jump_model!`](@ref)
"""
function jump_optimiser_from_attributes(opt::JuMPOptimiser, attrs::ResolvedJuMPOptimiser)
    # The remap is derived by reflection so it can never silently drop a field (a hand-written
    # pull-through once omitted `scard`, disabling sub-group cardinality in constrained-NOC;
    # fixed in 9ff28d6). Every `attrs` field overrides its optimiser field, and the block
    # Result maps onto the block through the rename table of the block.
    #
    # There is deliberately no inverse. A head that needs the bundle back keeps the `attrs` it
    # already holds (both `NearOptimalCentering` heads do), because a reverse hand list is the
    # same drop the comment above records — only pointing the other way.
    rename = (; pr = :pe, cbr = :cbe)
    # `imsk` is the one bundle field with no field on the optimiser: it is the Investable
    # Mask the reduction derived. A sub-problem built from this optimiser reads an
    # already-reduced prior, so it derives no mask of its own and needs none carried in.
    af = filter(!=(:imsk), fieldnames(ResolvedJuMPOptimiser))
    vals = map(af) do f
        return if f === :cbr
            constraint_block_from_resolved(opt.cbe, owner_block(attrs.cbr))
        else
            getfield(attrs, f)
        end
    end
    return rebuild_estimator(opt, NamedTuple{map(f -> get(rename, f, f), af)}(vals))
end
"""
    constraint_block_steps(blocks::Tuple{}, attrs::ResolvedJuMPOptimiser) -> Tuple{}
    constraint_block_steps(blocks::Tuple, attrs::ResolvedJuMPOptimiser) -> Tuple

Gives the blocks after the first of a resolved bundle, as the keyword `blocks` of a sub-problem on the reduced universe.

A sub-problem of [`NearOptimalCentering`](@ref) solves on the prior and the rows that the bundle reduced, and its optimiser comes from [`jump_optimiser_from_attributes`](@ref). An optimiser holds one block, so this function gives the other blocks to the sub-problem through the keyword of the entry. Each entry holds the set viewed at the Investable Mask, its rows on the mask, and the prior Result of its block, as [`allocation_set_block_step`](@ref) gives them. So a sub-problem reads the reduced blocks, and it fits no prior of a set again.

# Arguments

  - `blocks`: The entries that the bundle resolved, in the keyword `blocks` of [`resolve_jump_optimiser`](@ref).
  - `attrs`: The bundle. Its `cbr` holds the block Result of each entry after the block of the owner.

# Returns

  - `steps::Tuple`: One entry `(set, X, pr)` for each entry of `blocks`, or the empty `Tuple`.

# Related

  - [`jump_optimiser_from_attributes`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`near_optimal_centering_setup`](@ref)
"""
function constraint_block_steps(::Tuple{}, ::ResolvedJuMPOptimiser)
    return ()
end
function constraint_block_steps(blocks::Tuple, attrs::ResolvedJuMPOptimiser)
    cbr = attrs.cbr
    return ntuple(i -> allocation_set_block_step(blocks[i], cbr[i + 1], attrs.imsk),
                  length(blocks))
end
"""
    processed_jump_optimiser(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        kwargs...
    ) -> JuMPOptimiser

Build a fully-processed [`JuMPOptimiser`](@ref) from raw configuration and returns data.

Calls [`resolve_jump_optimiser`](@ref) then repackages the result via
[`jump_optimiser_from_attributes`](@ref). Used where a processed optimiser is needed for
inner sub-problems (e.g. [`NearOptimalCentering`](@ref)) while the `attrs` bundle is
reused directly by [`assemble_jump_model!`](@ref).

# Arguments

  - `opt::JuMPOptimiser`: Raw optimiser configuration.
  - $(arg_dict[:rd])
  - `kwargs...`: Keyword arguments passed to [`resolve_jump_optimiser`](@ref).

# Returns

  - `JuMPOptimiser`: Optimiser with all estimator slots populated from processed results.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`ResolvedJuMPOptimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`jump_optimiser_from_attributes`](@ref)
"""
function processed_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; kwargs...)
    attrs = resolve_jump_optimiser(opt, rd; kwargs...)
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
        attrs::ResolvedJuMPOptimiser,
        rd::ReturnsResult,
        r::Option{<:RM_VecRM} = nothing,
        obj::ObjectiveFunction = MinimumRisk(),
        b1::Option{<:MatNum} = nothing
    ) -> Nothing

Run the invariant model-assembly sequence shared by all single-JuMP-model optimisers.

Executes the constraint-builder pipeline — from `set_linear_weight_constraints!` through
`add_custom_constraint!` — that sits between the per-optimiser *head* (weight variables)
and *tail* (objective + solve). The head must have populated Model State (`w`/`k` variables)
before calling this function. See `Model Assembly` in `GLOSSARY.md` and
`0008-jump-model-assembly.md`.

The tail of the sequence is [`assert_frontier_sweep_cap`](@ref): both frontier registries are
complete here and no sweep solve has started, so this is the one point at which the **total**
sweep — the product across every swept return term and every swept risk measure — is in hand
and can be capped. [`assert_frontier_sweep_owner`](@ref) follows it, and refuses a frontier
when the optimiser solves its model once.

# Arguments

  - $(arg_dict[:model])
  - `optimiser::JuMPOptimisationEstimator`: Dispatch object for risk, tracking, and custom
    constraint builders.
  - `opt::JuMPOptimiser`: Supplies the tracking errors `tr`, which the head binds to its
    rows, and the scalariser `sca` of the risk step.
  - `attrs::ResolvedJuMPOptimiser`: The resolved bundle of [`resolve_jump_optimiser`](@ref).
    The builders read the prior, the fees and the return terms of the bundle, and every
    constraint, setting and calibrated slot of the [`ResolvedConstraintBlock`](@ref) in
    `attrs.cbr`. A Calibration Rule in a norm ceiling or a penalty resolved at stage 2 of
    the block, [`resolve_constraint_block`](@ref), so no builder resolves it here.
  - $(arg_dict[:rd])
  - `r::Option{<:RM_VecRM} = nothing`: Risk measure(s), or `nothing` to skip risk
    constraints and scalarisation (the [`RelaxedRiskBudgeting`](@ref) path).
  - `obj::ObjectiveFunction = MinimumRisk()`: Objective used by the return constraints.
  - $(arg_dict[:b1_opt])

# Returns

  - `nothing`. Mutates `model` in place.

# Related

  - [`ResolvedJuMPOptimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`set_risk_and_scalarise!`](@ref)
  - [`assert_frontier_sweep_cap`](@ref)
  - [`assert_frontier_sweep_owner`](@ref)
  - [`JuMPOptimiser`](@ref)
  - [`MeanRisk`](@ref)
  - [`RiskBudgeting`](@ref)
  - [`RelaxedRiskBudgeting`](@ref)
  - [`FactorRiskContribution`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function assemble_jump_model!(model::JuMP.Model, optimiser::JuMPOptimisationEstimator,
                              opt::JuMPOptimiser, attrs::ResolvedJuMPOptimiser,
                              rd::ReturnsResult, r::Option{<:RM_VecRM} = nothing,
                              obj::ObjectiveFunction = MinimumRisk(),
                              b1::Option{<:MatNum} = nothing)
    (; pr, fees, ret) = attrs
    # The builder of a block after the first adds its rows (ADR 0237). Until it lands, the
    # model holds the rows of the block of the owner, and the other blocks narrow the mask.
    cbr = owner_block(attrs.cbr)
    (; wb, lt, st, lcsr, ctr, gcardr, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst, tn, plr) = cbr
    set_linear_weight_constraints!(model, lcsr, :lcs_ineq_, :lcs_eq_)
    set_linear_weight_constraints!(model, ctr, :cent_ineq_, :cent_eq_)
    set_mip_constraints!(model, wb, cbr.card, gcardr, plr, lt, st, fees, cbr.ss, cbr.xbgt)
    set_smip_constraints!(model, wb, cbr.scard, sgcardr, smtx, sgmtx, slt, sst, sglt, sgst,
                          cbr.ss)
    set_turnover_constraints!(model, tn)
    # The head binds a data benchmark of its tracking errors to its rows, so `tr` is read
    # from the head and not from the block Result.
    set_tracking_error_constraints!(model, pr, opt.tr, optimiser, plr, fees, b1; rd = rd)
    # The norm ceilings and the penalties resolved their Calibration Rules at stage 2 of the
    # block, `resolve_constraint_block`, which also checked the range of each number.
    set_weight_norm_2_constraints!(model, cbr.l2c)
    set_weight_norm_p_constraints!(model, cbr.lpc)
    set_weight_norm_inf_constraints!(model, cbr.linfc)
    set_l1_regularisation!(model, cbr.l1)
    set_l2_regularisation!(model, cbr.l2)
    set_lp_regularisation!(model, cbr.lp)
    set_linf_regularisation!(model, cbr.linf)
    set_non_fixed_fees!(model, fees)
    set_risk_and_scalarise!(model, r, optimiser, opt, pr, plr, fees, b1; rd = rd)
    set_return_constraints!(model, ret, obj, pr; rd = rd)
    # After the return constraints, so a `MaximumRatio` has chosen its form, and before
    # every semidefinite phylogeny, including a programme set's in `add_custom_constraint!`.
    mark_risk_minimised!(model, obj)
    set_iplg_constraints!(model, plr)
    set_sdp_phylogeny_constraints!(model, plr)
    add_custom_constraint!(model, cbr.ccnt, optimiser, attrs)
    assert_frontier_sweep_cap(model)
    assert_frontier_sweep_owner(model, optimiser)
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
    return rebuild_estimator(cfg, (; ret = rebuild_estimator(cfg.ret, (; ucs = v))))
end
pipe_accepts(::JuMPOptimiser, ::Val{:mu_ucs})::Bool = true
# A routing target that is a slot of the Constraint Block, such as `:wb` or `:ple`, goes into
# the block, and `:pe` stays on the optimiser (ADR 0237).
pipe_block_field(::JuMPOptimiser)::Symbol = :cbe
"""
    reader_frames(opt::JuMPOptimiser, path::AbstractString, refit::Bool) -> ReturnsFrames
    group_frames(opt::JuMPOptimiser) -> ReturnsFrames
    group_frames(est::JuMPOptimisationEstimator) -> ReturnsFrames

Walk the readers of a JuMP head, for the warning of a forgotten raw reader and the gap rule of the head.

The group of a JuMP head is the head, the configuration that it holds in `opt`, and every reader in their slots: the prior `opt.pe`, the graph readers `opt.cte` and `opt.ple`, the regressions of `opt.lcse`, and the readers inside the risk measures, the tracking errors and the return terms. [`field_frames`](@ref) walks every field. The walk stops at a fallback, because it is a head with its own group. A configuration that takes an online step with no head answers for its own slots. A regression reader at `x_src = nothing` counts as raw only when the prior `opt.pe` gives no loadings, see [`refits_regression`](@ref).

# Arguments

  - `opt`: The configuration of a JuMP head.
  - $(arg_dict[:frame_path])
  - $(arg_dict[:frame_refit])
  - `est`: The JuMP head.

# Returns

  - `fr::ReturnsFrames`: The frames of the slots of `opt`, at `path` or from the configuration itself, or of the group of `est`.

# Related

  - [`field_frames`](@ref)
  - [`warn_raw_readers`](@ref)
  - [`group_source`](@ref)
"""
function reader_frames(opt::JuMPOptimiser, path::AbstractString, refit::Bool)
    return field_frames(opt, path, refit)
end
function group_frames(opt::JuMPOptimiser)
    return field_frames(opt, "", refits_regression(opt.pe))
end
function group_frames(est::JuMPOptimisationEstimator)
    return field_frames(est, "", refits_regression(est.opt.pe))
end
"""
    bind_data_benchmark(opt::JuMPOptimiser, rd::ReturnsResult)
    bind_data_benchmark(est::JuMPOptimisationEstimator, rd::ReturnsResult)
    bind_data_benchmark(est::RiskJuMPOptimisationEstimator, rd::ReturnsResult)

Bind every [`DataBenchmarkTracking`](@ref) of a JuMP optimiser to the benchmark of the rows it fits on.

Each head calls this function after [`investable_view`](@ref), so the benchmark has the rows and the investable columns of the model. Every builder, every corner solve and every risk of a frontier then reads the bound benchmark. The constructor of each holder already refused a data benchmark beside a prior that states `x_src = ExcessSource()`, see [`assert_data_benchmark_prior`](@ref).

  - A [`JuMPOptimiser`](@ref) binds its tracking errors `tr`.
  - A head binds its `opt`. A head with risk measures, a [`RiskJuMPOptimisationEstimator`](@ref), also binds its `r`.

Each rebuilds through its keyword constructor only when a field changes, so a head with no data benchmark comes back as itself.

# Algorithm

 1. A [`JuMPOptimiser`](@ref) binds `opt.tr` to `rd` with [`bind_data_benchmark`](@ref), giving `tr`.
 2. A head binds its `est.opt` with step 1, giving `opt`. A [`RiskJuMPOptimisationEstimator`](@ref) also binds `est.r`, giving `r`.
 3. Return the object itself when every bound field is the field it replaces, by `===`.
 4. Otherwise rebuild the object with the bound fields through its keyword constructor, with [`rebuild_estimator`](@ref).

# Arguments

  - `opt`, `est`: The optimiser or the head.
  - $(arg_dict[:rd])

# Returns

  - The optimiser or the head with its data benchmarks bound, or itself when it holds none.

# Related

  - [`bind_data_benchmark`](@ref)
  - [`DataBenchmarkTracking`](@ref)
  - [`investable_view`](@ref)
"""
function bind_data_benchmark(opt::JuMPOptimiser, rd::ReturnsResult)
    tr = bind_data_benchmark(opt.tr, rd)
    return tr === opt.tr ? opt : rebuild_estimator(opt, (; tr = tr))
end
function bind_data_benchmark(est::JuMPOptimisationEstimator, rd::ReturnsResult)
    opt = bind_data_benchmark(est.opt, rd)
    return opt === est.opt ? est : rebuild_estimator(est, (; opt = opt))
end
function bind_data_benchmark(est::RiskJuMPOptimisationEstimator, rd::ReturnsResult)
    opt = bind_data_benchmark(est.opt, rd)
    r = bind_data_benchmark(est.r, rd)
    return if opt === est.opt && r === est.r
        est
    else
        rebuild_estimator(est, (; opt = opt, r = r))
    end
end
