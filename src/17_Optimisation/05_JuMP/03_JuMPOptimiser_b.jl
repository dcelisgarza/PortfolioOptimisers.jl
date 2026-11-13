"""
    resolve_jump_optimiser(
        opt::JuMPOptimiser,
        rd::ReturnsResult;
        blocks::Tuple = (),
        kwargs...
    ) -> ResolvedJuMPOptimiser

Resolves a [`JuMPOptimiser`](@ref) against the returns data `rd`, to the [`ResolvedJuMPOptimiser`](@ref) that a JuMP model reads.

The function fits the prior, resolves the fees and the return terms, and resolves the [`ConstraintBlock`](@ref) of `opt` in two stages. Stage 1, [`resolve_constraint_block_names`](@ref), resolves the names of the block over the universe of the caller. The reduction to the Investable Mask then views the block, the prior and the rows. Stage 2, [`resolve_constraint_block`](@ref), resolves the block against the reduced prior and rows, to its [`ResolvedConstraintBlock`](@ref). [`assemble_jump_model!`](@ref) reads the bundle, and the Result of the head stores it in `jor`, so the resolution runs once for each `optimise` call.

The entry takes a list of blocks. The block of `opt` is the first, and the keyword `blocks` gives the others. Each entry of `blocks` is the [`ProgrammeAllocationSet`](@ref) of an online head and the rows `X` that the head holds, see [`fit_allocation_set_block`](@ref). Each block resolves against its own prior on its own rows, because a merge of two blocks changes the moments that their readers and their ceilings read. The Investable Mask of the bundle is the intersection of the masks of every block, so an asset that one block cannot read departs from all of them. The bundle then holds the vector of the block Results in `cbr`, the block of `opt` first. The entry takes `blocks` by name, so the keyword does not reach the readers through `kwargs...`.

A distance reader of a block, its `cte` or its `ple`, can refuse an asset that its prior keeps, see [`DistanceReaderRefusalError`](@ref). The shared entry [`resolve_block_entry`](@ref) holds one refusal branch for every reader of every block. The projection programme of a [`ProgrammeAllocationSet`](@ref) uses the same entry as a bare head. On a refusal, the entry fits again from the same prior results, without that asset. The readers of `opt` refuse under `opt.strict`, and the readers of a set refuse under the `strict` flag of the online head that holds the set, [`projection_step_strict`](@ref).

# Algorithm

 1. Fit the prior `pr` of `opt.pe` on `rd`. Fit the prior of each entry of `blocks` on its rows with [`fit_allocation_set_block`](@ref).
 2. Compose the Investable Mask `imsk` of the head with [`distance_reader_entry_mask`](@ref).
 3. Resolve the fees on the universe of the caller with [`fees_constraints`](@ref).
 4. Resolve stage 1 of the block of `opt` over the universe of the caller, with [`resolve_constraint_block_names`](@ref).
 5. Resolve every block through the shared entry [`resolve_block_entry`](@ref). The entry intersects the mask with the mask of each entry of `blocks`. The resolver of the head, [`resolve_jump_head_block`](@ref), places the fees on the axes that the mask leaves, reduces the prior, the optimiser and `rd` to the mask, and resolves stage 2 of the block of `opt`. The entry resolves each entry of `blocks` at the mask, and it fits again on a refusal.
 6. Resolve the return terms against the prior with [`factory`](@ref), and return the bundle. Its `cbr` is the block Result of `opt` when `blocks` is empty, and the vector of every block Result otherwise.

# Arguments

  - `opt::JuMPOptimiser`: JuMP optimiser configuration.
  - $(arg_dict[:rd])
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
  - [`resolve_block_entry`](@ref)
  - [`resolve_allocation_set_block`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`processed_jump_optimiser`](@ref)
"""
function resolve_jump_optimiser(opt::JuMPOptimiser, rd::ReturnsResult; blocks::Tuple = (),
                                kwargs...)
    # `rd` keeps its raw `X` and `B`. Each reader below that reads the returns states its
    # own source, and subtracts the benchmark itself, at the point of use.
    pr = prior(opt.pe, rd)
    # The prior of each set is fitted once. A fit again reads the fitted entries.
    blocks = map(fit_allocation_set_block, blocks)
    # An asset that a `FeatureDistance` of the clustering cannot read in the window of its
    # Asset Panel departs with the non-investable assets (`feature_readable_mask`). So does
    # an asset with a gap in the rows that a plain graph reader reads (`graph_reader_mask`).
    imsk = distance_reader_entry_mask((opt.ple, opt.cte), pr, rd, opt.strict)
    datatype = eltype(pr.X)
    # Resolve the fee on the caller's universe before `investable_reduction` narrows `sets`.
    # A name stated over that universe must not be refused because the data delisted the
    # asset. A liquidation charge keyed by name cannot resolve at all once its `w` sits on
    # the complement while `sets` sits on the mask. `investable_fees_view` then places the
    # resolved fee on the axes the mask leaves.
    feesu = fees_constraints(opt.fees, opt.sets; datatype = datatype, strict = opt.strict)
    # Stage 1 resolves the names over the caller's universe, as the fee does, and the
    # reduction views its values. An online head runs the same stage once for each fold,
    # before it knows the mask of a step, so both owners order the names and the reduction
    # by one rule (ADR 0237).
    opt1 = rebuild_struct(opt,
                          (;
                           cbe = resolve_constraint_block_names(opt.cbe, opt.strict,
                                                                datatype)))
    # The shared entry holds the one refusal branch for the readers of every block.
    imsk, (fees, prr, optr), cbrs = resolve_block_entry(resolve_jump_head_block,
                                                        (opt1, pr, rd, feesu, kwargs), imsk,
                                                        blocks, rd, opt.strict)
    cbr = isempty(blocks) ? first(cbrs) : [cbrs...]
    return ResolvedJuMPOptimiser(; pr = prr, fees = fees, ret = factory(optr.ret, prr),
                                 sca = optr.sca, imsk = imsk, cbr = cbr)
end
"""
    resolve_jump_head_block(imsk::Option{BitVector}, opt::JuMPOptimiser,
                            pr::AbstractPriorResult, rd::ReturnsResult, fees, kwargs) -> Tuple

Resolves the block of a JuMP head at the Investable Mask `imsk`. This is the resolver of the block of the head that [`resolve_jump_optimiser`](@ref) gives to the shared entry [`resolve_block_entry`](@ref).

The entry calls it once on the first fit, and once more on each fit again after a refusal, with the mask of that fit. The block of `opt` already holds the values of stage 1.

# Algorithm

 1. Place the fees on the axes that the mask leaves with [`investable_fees_view`](@ref).
 2. Reduce the prior, the optimiser and `rd` to the mask with [`investable_reduction`](@ref). The reduction views the block of stage 1, and it declares the Non-Investable Axis on its `sets`.
 3. Resolve stage 2 of the reduced block with [`resolve_constraint_block`](@ref), through [`distance_reader_call`](@ref), under the solver and the `strict` flag of the head.

# Arguments

  - `imsk`: The Investable Mask of the fit, or `nothing` when every asset is investable.
  - `opt`: The JuMP optimiser, with its block after stage 1.
  - $(arg_dict[:pr])
  - $(arg_dict[:rd])
  - `fees`: The fees of `opt`, resolved on the universe of the caller, or `nothing`.
  - `kwargs`: The keyword arguments of the centrality and phylogeny constraint estimators of `opt`.

# Returns

  - `((fees, pr, opt), (cbr,))::Tuple`: The fees, the prior and the optimiser at the mask, and the block Result of `opt`, or the [`DistanceReaderRefusalError`](@ref) of one of its readers.

# Related

  - [`resolve_jump_optimiser`](@ref)
  - [`resolve_block_entry`](@ref)
  - [`resolve_constraint_block`](@ref)
"""
function resolve_jump_head_block(imsk::Option{BitVector}, opt::JuMPOptimiser,
                                 pr::AbstractPriorResult, rd::ReturnsResult, fees, kwargs)
    _, prr, optr, rdr = investable_reduction(imsk, pr, opt, rd)
    cbr = distance_reader_call(resolve_constraint_block, optr.cbe, prr, rdr; slv = optr.slv,
                               strict = optr.strict, kwargs...)
    return (investable_fees_view(fees, imsk, pr.X), prr, optr), (cbr,)
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
    return rebuild_struct(opt, NamedTuple{map(f -> get(rename, f, f), af)}(vals))
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
    later_constraint_blocks(blocks::Tuple{}, attrs::ResolvedJuMPOptimiser) -> Tuple{}
    later_constraint_blocks(blocks::Tuple, attrs::ResolvedJuMPOptimiser) -> Tuple

Gives each block after the first of a resolved bundle, with the set that owns it and the namespace of the block in the model of the head.

A JuMP head adds every block of its bundle to its model, in [`assemble_jump_model!`](@ref). The block of the head is the first, and it takes the empty prefix. Block `i` after it takes the Model State namespace `blk_i_`, from [`nested_prefix`](@ref), so the second block registers its rows under `blk_2_`. The owner of a later block is its [`ProgrammeAllocationSet`](@ref), viewed at the Investable Mask of the bundle with [`constraint_block_steps`](@ref). The set holds the risk ceilings `r` and the return floor `ret` that the bounds builder adds, the tracking errors of the block, and the owner that its custom constraints and its custom objective terms get.

# Arguments

  - `blocks`: The entries that the bundle resolved, in the keyword `blocks` of [`resolve_jump_optimiser`](@ref).
  - `attrs`: The bundle. Its `cbr` holds the block Result of each entry after the block of the owner.

# Returns

  - `later::Tuple`: One entry `(set, cbr, prefix)` for each entry of `blocks`, or the empty `Tuple`.

# Related

  - [`assemble_jump_model!`](@ref)
  - [`constraint_block_steps`](@ref)
  - [`set_block_constraints!`](@ref)
"""
function later_constraint_blocks(::Tuple{}, ::ResolvedJuMPOptimiser)
    return ()
end
function later_constraint_blocks(blocks::Tuple, attrs::ResolvedJuMPOptimiser)
    steps = constraint_block_steps(blocks, attrs)
    return ntuple(i -> (first(steps[i]), attrs.cbr[i + 1],
                        nested_prefix(Symbol(""), :blk_, i + 1)), length(blocks))
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

[`UnconstrainedNearOptimalCentering`](@ref) reaches [`set_weight_constraints!`](@ref) through this method, because its centring solve adds the bounds of the block and no other row of it. Every other JuMP head adds its bounds and its budgets in phase 1 of its block, [`set_block_constraints!`](@ref), with the net budget of the head. Both routes pass the whole budget group (`bgt`, `sbgt`, `gbgt`) as one object, so a budget field cannot reach one head and miss the others.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `opt::JuMPOptimiser`: Optimiser supplying the budget group.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
"""
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds, opt::JuMPOptimiser)
    return set_weight_constraints!(model, wb, opt.bgt, opt.sbgt; gbgt = opt.gbgt)
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
        b1::Option{<:MatNum} = nothing;
        blocks::Tuple = ()
    ) -> Nothing

Run the invariant model-assembly sequence shared by all single-JuMP-model optimisers.

Executes the constraint-builder pipeline that sits between the per-optimiser *head* (weight
variables) and *tail* (objective + solve). The head must have populated Model State (`w`/`k`
variables) before calling this function. See `Model Assembly` in `GLOSSARY.md` and
`0008-jump-model-assembly.md`.

The middle runs once for each block, through the block builder. The block of the head is the
first, and the keyword `blocks` gives the others, as the head gave them to
[`resolve_jump_optimiser`](@ref). Each block after the first takes its own namespace, and its
owner is its [`ProgrammeAllocationSet`](@ref), see [`later_constraint_blocks`](@ref).

 1. Phase 1 of each block, [`set_block_constraints!`](@ref), adds the weight bounds, the
    budgets and every row before the risk step. The block of the head takes the net budget
    `opt.bgt`, the fees and the tracking errors of the head. A block of a set takes no net
    budget and no fees, and its own tracking errors.
 2. The head adds its fees, its risk measures and its return terms, and marks the objective
    with [`mark_risk_minimised!`](@ref). The risk measures read the phylogeny `plr` of the
    block of the head.
 3. The bounds builder [`set_allocation_risk_return_bounds!`](@ref) adds the risk ceilings
    `r` and the return floor `ret` of each set.
 4. Phase 2 of each block, [`set_block_post_risk_constraints!`](@ref), adds the phylogeny
    rows and the custom constraints. Each block then adds its custom objective terms with
    [`add_custom_objective_term!`](@ref), with `obj`. The block of the head gives the head
    and `attrs`, and a block of a set gives the set and the Result of its block.

A frontier corner solves this model again with another objective. Every custom objective term
keeps the `obj` of the assembly, as the penalties do, and counts once in each solve.

The tail of the sequence is [`assert_frontier_sweep_cap`](@ref): both frontier registries are
complete here and no sweep solve has started, so this is the one point at which the **total**
sweep — the product across every swept return term and every swept risk measure — is in hand
and can be capped. [`assert_frontier_sweep_owner`](@ref) follows it, and refuses a frontier
when the optimiser solves its model once.

# Arguments

  - $(arg_dict[:model])
  - `optimiser::JuMPOptimisationEstimator`: Dispatch object for risk, tracking, and custom
    constraint builders, and the owner that the custom objective terms of the head get.
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
  - `obj::ObjectiveFunction = MinimumRisk()`: Objective used by the return constraints, and
    by the custom objective terms of every block.
  - $(arg_dict[:b1_opt])
  - `blocks`: The blocks after the block of the head, the keyword `blocks` of
    [`resolve_jump_optimiser`](@ref) that gave `attrs`. The empty `Tuple` gives one block.

# Returns

  - `nothing`. Mutates `model` in place.

# Related

  - [`ResolvedJuMPOptimiser`](@ref)
  - [`resolve_jump_optimiser`](@ref)
  - [`later_constraint_blocks`](@ref)
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
                              b1::Option{<:MatNum} = nothing; blocks::Tuple = ())
    (; pr, fees, ret) = attrs
    cbr = owner_block(attrs.cbr)
    # Each block after the first comes with the set that owns it and its own namespace
    # (ADR 0237).
    later = later_constraint_blocks(blocks, attrs)
    # The head binds a data benchmark of its tracking errors to its rows, so `tr` is read
    # from the head and not from the block Result.
    set_block_constraints!(model, cbr, optimiser, opt.bgt, fees; tr = opt.tr, b1 = b1,
                           rd = rd)
    # The head owns the net budget and the fees, so a block of a set takes neither.
    for (set, cbri, prefix) in later
        set_block_constraints!(model, cbri, set, nothing, nothing;
                               tr = allocation_set_tracking(set.tr, cbri), prefix = prefix)
    end
    set_non_fixed_fees!(model, fees)
    set_risk_and_scalarise!(model, r, optimiser, opt, pr, cbr.plr, fees, b1; rd = rd)
    set_return_constraints!(model, ret, obj, pr; rd = rd)
    # After the return constraints, so a `MaximumRatio` has chosen its form, and before
    # every semidefinite phylogeny, including a programme set's in `add_custom_constraint!`.
    mark_risk_minimised!(model, obj)
    for (set, cbri, prefix) in later
        set_allocation_risk_return_bounds!(model, set, cbri; prefix = prefix)
    end
    set_block_post_risk_constraints!(model, cbr, optimiser, attrs)
    # Every block adds its custom objective terms here, once, with the objective of the
    # assembly, as its penalties go into the Objective Penalty in phase 1. The objective
    # builder runs once for each corner of a frontier, and the Objective Penalty keeps every
    # term it gets, so a term added there counts once for each corner (#2065).
    add_custom_objective_term!(model, obj, cbr.cobj, optimiser, attrs)
    for (set, cbri, prefix) in later
        set_block_post_risk_constraints!(model, cbri, set, nothing; prefix = prefix)
        add_custom_objective_term!(model, obj, cbri.cobj, set, cbri)
    end
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
    return rebuild_struct(cfg, (; ret = rebuild_struct(cfg.ret, (; ucs = v))))
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
 4. Otherwise rebuild the object with the bound fields through its keyword constructor, with [`rebuild_struct`](@ref).

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
    return tr === opt.tr ? opt : rebuild_struct(opt, (; tr = tr))
end
function bind_data_benchmark(est::JuMPOptimisationEstimator, rd::ReturnsResult)
    opt = bind_data_benchmark(est.opt, rd)
    return opt === est.opt ? est : rebuild_struct(est, (; opt = opt))
end
function bind_data_benchmark(est::RiskJuMPOptimisationEstimator, rd::ReturnsResult)
    opt = bind_data_benchmark(est.opt, rd)
    r = bind_data_benchmark(est.r, rd)
    return if opt === est.opt && r === est.r
        est
    else
        rebuild_struct(est, (; opt = opt, r = r))
    end
end
