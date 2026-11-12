---
status: accepted
---

# A Constraint Block holds the feasible set of a programme, and the entry of a head takes a list of blocks

## Context

`JuMPOptimiser` and `ProgrammeAllocationSet` spell one concept twice: the feasible set of a
programme over `w`, and the terms that the programme adds to the Objective Penalty.
[ADR 0159](0159-a-constrained-online-update-projects-onto-an-allocation-set-in-the-rules-own-geometry-and-the-default-set-needs-no-solver.md)
widened the set in place under the names and bounds of the optimiser, and it sent the shared
struct to map #1213. The two types now hold the same fields, but they keep two copies of one
pipeline. `resolve_allocation_set_rows` resolves the set as `processed_jump_optimiser_attributes`
resolves a JuMP head, and `assemble_allocation_set!` adds the kinds in the order of
`assemble_jump_model!` ([ADR 0008](0008-jump-model-assembly.md)).

The copies drifted, and the drift gave defects:

- #2030: a reader of the set, `ple` or `cte`, refuses an asset that the prior keeps. The refusal
  branch of #2015 and #2028 exists only in the entry of the JuMP head, so the
  `DistanceReaderRefusalError` escapes the leader re-solve of `FollowTheLeader`.
- #2032: the `cobj` of the set has no effect in the leader re-solve, against
  [ADR 0164](0164-a-follow-the-leader-rule-solves-its-programme-on-the-allocation-set-and-projects-only-its-damped-mix.md).
- #2035: a fallback leader of `FollowTheLeader` solves without the Allocation Set of the head.

The leader re-solve puts the set on the `ccnt` of the held optimiser, through
`AllocationSetConstraint` (ADR 0164). This route has three defects of design. `ccnt` is the
extension point for a constraint that the library does not name, and the library uses it for a
whole programme. `AllocationSetConstraint.X` is a `ReturnsResult`, so an estimator holds a Result,
against `src/CLAUDE.md`. And the `:aset_` prefix and the rule "state a kind on the set or on the
held optimiser, not on both" exist only because the set enters the model as one opaque part.

The leader re-solve also shows a requirement. **One programme can hold more than one block, and
each block resolves against its own prior on its own rows.** The block of the held optimiser reads
the prior of the head on the sample. The block of the set reads the prior of the set on all the
rows that the head holds. A merge of the two blocks into one changes the moments of the readers,
and `ple`, `card` and the risk measures have no rule of intersection.

The maintainer ruled the four decisions of the map on 2026-11-11, and the names on 2026-11-12
(#2036). This ADR records them.

## Decision

### 1. The block

The **Constraint Block**, `ConstraintBlock`, holds the feasible set of a programme over `w` and
the terms that the programme adds to the Objective Penalty. `JuMPOptimiser` holds it in the field
`cbe`, beside its prior, solver, budget, fees, return term, scalariser and execution settings.
`ProgrammeAllocationSet` holds it in the field `cbe`, beside its prior, solver, risk ceilings `r`
and return floor `ret`. A constraint kind is added to the block, and it reaches both owners.

### 2. The partition rule

**A field is a block field when it narrows the feasible set of `w` or adds a term to the
Objective Penalty, and both owners hold it.** A block field never writes the objective. A field
that the objective reads stays on the owner, and so does the net budget. A bound on the risk or on
the return of `w` that stands alone, and that no objective reads, stays a field of the set.

| Field | Place | Reason |
| --- | --- | --- |
| `sbgt`, `gbgt` | Block. `bgt` stays on the owner. | They cap the leverage of the feasible set. `set_long_short_budget_constraints!(model, bgt, sbgt)` also reads the net budget, so the builder takes the net budget of the owner of the block. The set's block passes `nothing`. `assert_gross_budget_admissible` reads both sides, so it runs in the constructor of the owner. |
| `xbgt` | Block. | It pins `lw` and `sw`, which belong to the weights, so the builder writes the pin once, when any block asks for it. |
| `lcse`, with its exposure rows | Block, with no special case. | Both owners make one call, `linear_constraints(lcse, sets; rr = pr.rr, rd = rd)`. `assert_investable_constraint_width` applies to every block. |
| `ple`, with the semidefinite kind | Block. | It reads the lifted `W` and the marks of the owner's objective. So the builder has two phases: the rows before the owner's risk step, and the floor, the phylogeny rows and `ccnt` after `mark_risk_minimised!`. The risk measures of the owner read the `plr` of its first block. |
| `l1`, `l2`, `lp`, `linf`, `cobj` | Block, as terms of the Objective Penalty. | ADR 0164 sends the penalties and the `cobj` of the set into the leader's objective, and only the block reaches the leader. The builders add to `op`. Two blocks give the sum of their terms. The owner decides whether its objective reads `op`. |
| `ret` of `JuMPOptimiser` | Owner. | It is the objective term, the normalisation of `MaximumRatio`, a frontier, the target of `no_bounds_optimiser` and of `pipe_route(:mu_ucs)`. |
| `r` and `ret` of `ProgrammeAllocationSet` | Flat fields of the set. | A JuMP head holds no bound that stands alone, so a ceiling slot and a floor slot in its block are always empty. The bounds of the set stay one family: `assert_risk_ceiling`, `assert_return_floor`, `risk_reads_rows` and `set_allocation_risk_ceiling!`. |
| Every other shared slot (`wb`, `sets`, `cte`, `gcarde`, `sgcarde`, `smtx`, `sgmtx`, the thresholds, `tn`, `tr`, `ccnt`, `card`, `scard`, `l2c`, `lpc`, `linfc`, `ss`) | Block. | Both owners hold it, and it narrows the feasible set of `w` or names the assets of a slot that does. |

A block field can read these data:

1. The caller's object and `sets`.
2. The prior result and the rows of its own block.
3. The solver of its owner. A Calibration Rule in `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp` or
   `linf` resolves against `slv`.
4. The Price-Adjusted Allocation, through `factory`.
5. Four values of its owner, which the block builder takes as arguments: the net budget, for
   `sbgt` and `gbgt`; `fees`, for the MIP kinds; the marks `risk_minimised` and `variance_flag`,
   for the semidefinite `ple`; and the objective with its owner, for `cobj`.

A block field writes to three places: its own namespace, the state of the shared weights once
(`lw` and `sw`, `W`, the MIP bits), and the Objective Penalty.

**The budget of one is enforced on the owner.** The online head owns the budget of one
(ADR 0159). The bare projection writes `Σw = k`. The block of the set in a leader model carries no
budget row, so the leader keeps `Σw = k` in its model. `FollowTheLeader` refuses, at construction,
a held JuMP optimiser whose `bgt` excludes one, on each entry of a `TimeDependent` schedule
(#2034).

### 3. The resolver, the builder and the entry for many blocks

**Both halves split.** One resolver resolves one block against its own prior and rows. One builder
adds one resolved block to a model. The entry of a head takes a list of blocks, intersects their
Investable Masks, and holds one refusal branch for the readers of every block. The batch head
passes one block. The leader re-solve passes two. The projection programme passes the block of the
set to a bare head.

**The block Result.** A resolved block is the **Resolved Constraint Block**,
`ResolvedConstraintBlock <: AbstractResolved`. It holds:

1. the resolved slots;
2. the settings that need no resolution: `card`, `scard`, `ss`, `xbgt`, `sbgt`, `gbgt`, `ccnt`,
   `cobj`;
3. the prior result and the rows of its own block.

The bundle of a JuMP head, `ResolvedJuMPOptimiser`, holds `pr`, `fees`, `ret`, `sca`, `imsk` and
the resolved blocks in `cbr`. `cbr` holds one block Result or a vector of them, as `plr` holds one
phylogeny Result or a vector. The `cobj` of a block gets the model's `obj`, the owner of its block,
and the Result of its block. `jump_optimiser_from_attributes` keeps its rename table, and the table
moves to the block.

**A second block reaches the entry as an internal keyword of `_optimise`.** The precedent is
`rmsk`, which reaches the resolver of a JuMP head as a keyword. The keyword carries the set after
`factory(set, w)`, which is configuration, and the rows `X`, which are an argument. So no
estimator holds a Result, and `AllocationSetConstraint` goes. The keyword also carries the `r` and
the `ret` of the set, and a bounds builder adds them after `mark_risk_minimised!`. The entry takes
the keyword by name, so the keyword does not reach `centrality_constraints` and
`phylogeny_constraints` through `kwargs...`. The fallback chain passes `kwargs...` to each
`_optimise`, so the block reaches each member of the chain (#2035).

**Each block after the first takes a Model State namespace.** It registers the weights under its
prefix with `w_owner`, as `weights_prefix` states for the risk ceiling of the set and for
`DependentVariableTracking`. Every builder with a fixed name takes a `prefix` (#2041). The prefix
replaces the `:aset_` prefix, and the rule "not on both" goes. A kind on both the set and the held
optimiser gives two constraints.

**The MIP bits belong to the weights.** There is one bit for each asset, written once, as `lw`,
`sw` and the pin of `xbgt` are. Each block adds its own rows `w_i ≤ ub_i^b z_i`,
`w_i ≥ lt_i^b z_i` and `Σ z ≤ card_b` on the shared `z`. "Asset `i` is held" is a fact of `w`, so
the rows of two blocks intersect exactly, with half the binaries. The fixed fee of the owner reads
the same bits once.

**A block resolves in two stages.**

1. Stage 1 resolves the names against `sets`. This is configuration. It gives the block with
   values in its slots, which the `XE_X` unions take, so it holds no Result. An online arm runs it
   once for each fold, where the swap of #1243 runs.
2. Stage 2 resolves against the prior and the rows. It gives the Resolved Constraint Block. An
   online arm runs it at each step.

The batch entry runs the two stages one after the other. A slot whose resolver reads `rd`, such as
`ScoreWeightBounds` in `wb`, is a stage 2 slot.

### 4. The keyword surface

**The released keywords stay, and `opt.wb` stays readable.** The 235 reads under `src/` and the
934 `JuMPOptimiser(` call sites keep their form. `ProgrammeAllocationSet` takes the same rule.

1. **The keyword constructor forwards the flat keywords into the block**, as
   `KernelTrendPatternTracking` builds its nested parts from flat keywords
   ([ADR 0223](0223-the-elastic-net-is-a-penalised-regression-target.md), item 7).
2. **A named `forward` rule keeps each slot readable on the owner**, as `HighOrderPriorEstimator`
   hoists `me` and `ce` ([ADR 0013](0013-property-forwarding-macro.md), Shape A). The rule lists
   the slots, `forward(cbe, wb, sets, …)`, and not `forward(cbe)`. A slot that the block gets later
   becomes a property of the owner only through an edit of the list. The `r` and the `ret` of the
   set are fields, so they need no forward rule.
3. **The keyword constructor also takes the block by its field name**, because
   `forward_setproperties` and `rebuild_estimator` call the constructor with every field by name.
4. **A flat keyword replaces that slot of a given block.** `rebuild_estimator` passes on each name
   of `repl` that is not a field. So a rebuild that names `wb` calls the constructor with the old
   block and the new `wb`. The flat default of a keyword must not overwrite a slot of a given
   block. The schedules of #1243 rebuild through this path at each fold.
5. **A search key that names a slot keeps working.** A search sets each key with
   `Accessors.set(opt, lens, val)`, and the keys `"opt.wb"`, `"opt.l1"` and `"opt.l2"` are in use.
   The `settable` option of the `forward` rule lets a patch of `Accessors.set` pass its names to
   the keyword constructor (#2037, ADR 0013).
6. **Pipe routing reaches a slot of the block.** `pipe_route` and `pipe_accepts` test `hasfield`,
   and a forwarded slot is not a field. So the owner routes a slot target of
   `PIPELINE_ROUTING_TARGETS` into the block, as `@pipe_delegates` routes every target into `opt`
   ([ADR 0038](0038-optimisers-own-pipeline-routing.md)), and it keeps `pe` on itself.
7. **The reads of the stored bundle stay**, through a named `forward` rule on the bundle. The
   reads `pa.wb`, `pa.tn` and `pa.lcsr` become `jor.wb`, `jor.tn` and `jor.lcsr`.

### 5. The names

| Item | Name before | Name |
| --- | --- | --- |
| The block | (none) | `ConstraintBlock` |
| The block Result | (none) | `ResolvedConstraintBlock <: AbstractResolved` |
| The bundle of a JuMP head | `ProcessedJuMPOptimiserAttributes` | `ResolvedJuMPOptimiser` |
| The bundles of risk budgeting | `ProcessedAssetRiskBudgetingAttributes`, `ProcessedFactorRiskBudgetingAttributes` | `ResolvedAssetRiskBudgeting`, `ResolvedFactorRiskBudgeting` |
| The abstract types | `ProcessedAttributes`, `ProcessedRiskBudgetingAttributes` | `AbstractResolved`, `AbstractResolvedRiskBudgeting` |
| The resolver verb of a JuMP head | `processed_jump_optimiser_attributes` | `resolve_jump_optimiser` |
| The resolver verb of a block | (none) | `resolve_constraint_block` |
| The field of the block on both owners | (none) | `cbe` |
| The field of the resolved blocks on `ResolvedJuMPOptimiser` | (none) | `cbr` |
| The Result fields that hold the bundles | `pa`, `prb` | `jor`, `rbr` |

In JuMP an attribute is a setting of the solver, so `ProcessedJuMPOptimiserAttributes` read as the
solver settings. "Resolved" is the word of the rulings and of `resolve_allocation_set`, and every
member of the family takes it. `cbe` and `cbr` follow the `e`/`r` pair of `lcse`/`lcsr`,
`cte`/`ctr` and `ple`/`plr`. `jor` and `rbr` put the thing first and then `r`. `rrb` is taken,
because RRB is `RelaxedRiskBudgeting`.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| #2030 alone | A resolve hook on `CustomJuMPConstraint` that the entry calls. | The three defects of the `ccnt` route stay, and so do the two copies of the pipeline. |
| #2030 alone | A named slot for the set on `JuMPOptimiser`. | The set still goes through `JuMPOptimiser`, and the two copies stay. |
| #2030 alone | Merge the set into the slots of the held optimiser. | It changes the moments that the set reads. |
| #2030 alone | A leader verb for the log-optimal leader only, and a refusal of every other held estimator under a programme set. | It removes the mean-risk, fee-aware, risk-budgeting and minimum-variance leaders under a programme set. |
| The partition | `ret` splits: the term on the owner, a floor slot in the block. | The floor slot is empty in every `JuMPOptimiser` block. It needed a floor keyword on `JuMPOptimiser`, and the block builder had to clear `rte` and `rke`. |
| The partition | A ceiling slot and a floor slot in the block. | Empty in every `JuMPOptimiser` block, and the cause of the problems of the row above. |
| The partition | A separate struct for the bounds. | A type with no second owner. The flat fields hold the same data. |
| The partition | The bounds on `FollowTheLeader` alone. | The projection programme of every online rule reads the ceilings and the floor of the set. On `FollowTheLeader` alone they leave `ExponentiatedGradient` and the other projection rules. |
| The split | Keep the two copies. | #2030, #2032 and #2035 stay. Each new kind lands twice. |
| The split | Split the resolver only. | It fixes the refusal of #2030. The order of the builder stays in two places, the collisions of the fixed names stay, and #2032 stays. It is a stage, not an end. |
| The split | Make the set a JuMP head, with no split. | It fixes the bare projection only. The leader re-solve still needs two blocks with two priors in one model. |
| The block Result | A `NamedTuple`, as `resolve_allocation_set_rows` returns. | No type, no docstring, no dispatch. |
| The block Result | The block estimator with resolved values in its slots. | A builder cannot tell a resolved block from a raw one by type, and the prior and the rows of the block have no slot. Stage 1 gives this form, and it is not the Result. |
| The block Result | The bundle of a JuMP head once for each block. | It carries `ret`, `sca` and `fees` once for each block, and the partition keeps them on the owner. |
| The second block | A field on the owner, as `ccnt` carries the set now, or a vector of blocks. | It puts the data of one step, `w` and `X`, in an estimator, which then holds a Result. |
| Two blocks in one model | Anonymous rows, as `add_allocation_set_bounds!` writes. | A builder whose entry another builder reads, such as the MIP bits or a turnover index, needs a name. |
| Two blocks in one model | Free indices alone, as `free_state_index` gives. | They serve the indexed builders. The builders with fixed names still collide. |
| The stages | One stage for each call. | The online arms then resolve the names at every step, where the set resolves them once for each fold. |
| The keywords | A read through the block, `opt.cbe.wb`, at every site. | It changes 235 reads under `src/` and breaks the search keys such as `"opt.wb"`. The maintainer kept the released surface. |
| The forward rule | `forward(cbe)`, which forwards every name of the block. | ADR 0013 states that an estimator hoists named accessors and does not forward every name. |
| The name of the block | `ProgrammeBlock`. | "Programme" in `ProgrammeAllocationSet` means a set that needs a solver. |
| The name of the block | `FeasibleSet`. | It omits the penalties, and it sits beside Allocation Set and Universe Sets. |
| The prefix of the family | `Processed…Data`, or `Resolved…Result`. | The first keeps the vague verb. The second sits next to `JuMPOptimisationResult`. |

## Consequences

- **Breaks.** The positional constructors of `JuMPOptimiser` and `ProgrammeAllocationSet` change,
  because the block becomes a field. `AllocationSetConstraint` is `public`, and it leaves the API.
  The exported `ProcessedJuMPOptimiserAttributes` and the Result fields `pa` and `prb` take their
  new names. A prior of the set that prices fewer assets than the leader drops them with the
  warning of `distance_reader_refusal_mask`, and `strict` refuses them. Before, it raised an
  `ArgumentError`. Each break takes a row in `docs/src/migration.md`.
- **`resolve_allocation_set_rows` and `assemble_allocation_set!` go.** The projection programme
  uses the shared entry, and the refusal rebuild of #2028 in its method of seven arguments goes.
- **`port_opt_view`, `factory`, `rebuild_estimator`, `needs_previous_weights` and the
  schedules of issue #1243 move to the block.** A field of the set keeps the schedule that #1243 gives it.
- **The bounds builder of the set clears `rte` for a floor, as a ceiling clears `rke`.** Before,
  the floor of the set kept `rte`. The set stays the Risk Constraint Owner of its ceilings.
- **`NearOptimalCentering` keeps the slots of a block, and the `r` and `ret` of the set, in its
  anchor sub-problems**, as it keeps `wb`.
- **One rule orders the names and the investable reduction (#2042).** Stage 1,
  `resolve_constraint_block_names`, holds only the slots whose value follows a view at the
  Investable Mask: the thresholds, `sgcarde`, the sub-group matrices and thresholds, and `tn`. It
  resolves them over the universe of the caller, before the reduction, in both owners, and the
  reduction views the values. Stage 2, `resolve_constraint_block`, resolves after the reduction,
  over the `sets` that carry the Non-Investable Axis: `wb` and `lcse`, whose resolvers read `rd`;
  `gcarde`, because a linear constraint over the assets does not follow a view; and the slots that
  read the prior. So a row that names a departed asset drops in both owners, and
  `assert_investable_constraint_width` refuses a precomputed `lcse` or `gcarde` of the wrong width
  in every block. Before #2042 the set resolved `lcse` and `gcarde` once for each fold, and it did
  not cut them: the reduction kept their width, and the model raised a bare `DimensionMismatch`.
  The set now declares the Non-Investable Axis on its `sets` in `programme_investable_reduction`,
  as `investable_reduction` does for a head. A test covers both arms.
- **`ScoreWeightBounds` in a `ProgrammeAllocationSet`.** The set refuses it now. A stage 2 slot
  reads `rd` at each step, so the set can take it after the resolver lands. Map #1213 records it
  under *Not yet specified*.
- **Other decision records.** The builder ticket amends ADR 0008: the middle of the assembly runs
  once for each block. The ticket of the leader re-solve amends or rewrites ADR 0164: the set no
  longer goes on `ccnt`. The closing ticket of map #1213 amends ADR 0159.
