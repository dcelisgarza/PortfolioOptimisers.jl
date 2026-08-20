---
status: accepted
---

# Extract the shared JuMP model-assembly pipeline into one deep module

## Context

Every single-JuMP-model Optimisation Estimator builds its `JuMP.Model` by running the
same long sequence of constraint and risk builders inline in its own `_optimise` method.
The sequence — `set_linear_weight_constraints!` (×2) → `set_mip_constraints!` →
`set_smip_constraints!` → `set_turnover_constraints!` → `set_tracking_error_constraints!`
→ `set_weight_norm_2_constraints!` → `set_l1/l2/linf/lp_regularisation!` →
`set_non_fixed_fees!` → `set_risk_constraints!` → `scalarise_risk_expression!` →
`set_return_constraints!` → `set_sdp_phylogeny_constraints!` → `add_custom_constraint!` —
is repeated, in the same order, across five files:

- [`MeanRisk`](../../src/20_Optimisation/11_MeanRisk.jl) (`_optimise` ~631–656)
- [`RiskBudgeting`](../../src/20_Optimisation/14_RiskBudgeting.jl) (`_optimise` ~688–713)
- [`RelaxedRiskBudgeting`](../../src/20_Optimisation/15_RelaxedRiskBudgeting.jl) (`_optimise` ~382)
- [`FactorRiskContribution`](../../src/20_Optimisation/12_FactorRiskContribution.jl) (`_optimise` ~299–341)
- [`NearOptimalCentering`](../../src/20_Optimisation/13_NearOptimalCentering.jl) (constrained `_optimise` ~1053–1105)

The middle is the seam between builders, but it is not a module: it has no interface, so a
change to constraint ordering or a new constraint type must be applied in five places, and
there is no way to test "are the right constraints added in the right order" without a full
prior + solver + CSV-baseline `optimise()` run.

The duplication is not perfectly uniform. The optimisers differ at three kinds of point:

1. **Head** — how the weight variables `w` and scalar `k` are shaped: `set_w!` (MeanRisk,
   NOC), `set_risk_budgeting_constraints!` (RB), `set_relaxed_risk_budgeting_constraints!`
   (RRB), `set_factor_risk_contribution_constraints!` (FRC, which also produces `b1`).
2. **Per-optimiser context inside the middle** — FRC threads `b1` into tracking and risk;
   the return-constraint objective is `mr.obj`/`frc.obj` for objective-bearing optimisers
   versus `MinimumRisk()` for RB/NOC/RRB; RRB carries **no** risk measure at all (it has no
   `.r` field — its risk lives in the `rba` head) and so omits `set_risk_constraints!` and
   `scalarise_risk_expression!`.
3. **Tail** — the objective function and the solve diverge fundamentally: `solve_mean_risk!`
   (frontier iteration), `optimise_JuMP_model!` (plain), `solve_noc!` (inner/outer).

There are also two parallel representations of the processed inputs: MeanRisk/RB/FRC/RRB use
`processed_jump_optimiser_attributes` → a flat `ProcessedJuMPOptimiserAttributes` with
result-suffixed names (`lcsr`, `ctr`, `gcardr`, `plr`); constrained-NOC uses
`processed_jump_optimiser` → a processed `JuMPOptimiser` in place, read with
estimator-suffixed names (`lcse`, `cte`, `gcarde`, `ple`) and carrying the scalar settings on
the same object, wrapped in a [`NearOptimalSetup`](../../src/20_Optimisation/13_NearOptimalCentering.jl).

This builds on the Model State interface of [ADR 0004](0004-typed-jump-model-state.md): that
ADR gave the data the builders share a named interface; this ADR gives the *ordering* of the
builders a named interface. The two are complementary — `Model State` is the data, `Model
Assembly` is the sequence (see `CONTEXT.md`).

## Decision

Extract the invariant middle into one deep module and leave the per-optimiser head and tail
in place.

1. **Inner core.** A mutating builder

   ```julia
   assemble_jump_model!(model::JuMP.Model, optimiser::JuMPOptimisationEstimator,
                        opt::JuMPOptimiser, attrs::ProcessedJuMPOptimiserAttributes,
                        rd::ReturnsResult, r::Option{<:RM_VecRM} = nothing,
                        obj::ObjectiveFunction = MinimumRisk(),
                        miprb_flag::Bool = false, b1::Option{<:MatNum} = nothing,
                        sdp_asset_phylogeny::Bool = true)
   ```

   runs the middle from `set_linear_weight_constraints!` through `add_custom_constraint!`
   and returns `nothing`. It reads constraint **results** from `attrs`
   (`ProcessedJuMPOptimiserAttributes`) and scalar **settings** from `opt` (the
   `JuMPOptimiser`); `optimiser` is the dispatch object for `set_risk_constraints!`,
   tracking, and custom constraints. It reads `w`/`k` through the ADR 0004 Model State
   accessors — the contract is that the head has populated Model State before the call.

2. **Per-optimiser context as optional kwargs, not interface width.** The things that vary
   inside the middle ride in as kwargs defaulting to the common case: `r` (risk measures, or
   `nothing`), `b1` (factor loadings, `nothing`), `obj` (`MinimumRisk()`), `miprb_flag`
   (`false`), and `sdp_asset_phylogeny` (`true`). The `b1` value is splatted in via an optional
   trailing-argument tuple — `extra = isnothing(b1) ? () : (b1,)` — so the non-factor
   tracking/risk calls are reproduced argument-for-argument, and FRC's calls (which dispatch
   on `opt::FactorRiskContribution` with a trailing `b1::MatNum`) get exactly `(b1,)`. The
   risk + scalarise step is a `::Nothing`-dispatched pair so RRB's absent risk measure is a
   no-op; the compiler constant-propagates the defaults and specialises away the branches —
   no runtime cost.

   > **Implementation note.** The ADR originally anticipated only `r`/`b1`/`obj`. Building it
   > surfaced two more divergences in the middle that the static read missed: RB alone passes
   > a mixed-integer flag to `set_mip_constraints!` (→ `miprb_flag`), and FRC applies a
   > factor-space SDP phylogeny in its tail *instead of* the standard asset-space one
   > (→ `sdp_asset_phylogeny = false`, with FRC's `set_sdp_frc_phylogeny_constraints!` left
   > caller-side).
   >
   > **Superseded (ADR 0033).** `miprb_flag` has since been removed from the signature: RB and
   > the user-facing `xbgt` were emitting the same exact-decomposition constraints under two
   > names, so both now route through a Model State head contract (`WeightsFromParts` vs
   > `PartsBoundWeights`) read by `set_exact_budget_constraints!`, not a flag on the middle. The
   > `sdp_asset_phylogeny` divergence is unaffected.

3. **Head and tail stay per-optimiser.** Each `_optimise` keeps its own head (including
   `set_weight_constraints!`, which is intrinsic to how each head shapes `w`) and its own
   tail (objective function + `solve_*!` + Result construction). The middle never touches
   weight/budget bounds or the objective.

4. **NOC integrates through a bespoke outer method, not by bending the core.** A thin
   `assemble_jump_model!(noc, model, setup::NearOptimalSetup)` unwraps the
   `NearOptimalSetup` (extracting `setup.opt` for settings and building the
   `ProcessedJuMPOptimiserAttributes` NOC already constructs for its Result), then delegates
   to the shared `assemble_jump_model!`. The `lcse`↔`lcsr` field-name divergence is confined
   to this one wrapper. The inner core sees a single concrete shape.

5. **FRC factor phylogeny stays caller-side.** FRC's `set_sdp_frc_phylogeny_constraints!`
   (factor-space, computed from `frc.frc_ple`/`rd.F`) remains in FRC's tail. The middle keeps
   only the standard `set_sdp_phylogeny_constraints!`. One caller does not justify a
   phylogeny-applier seam in the shared interface.

6. **Scope.** The five single-JuMP-model optimisers above. **Unconstrained-NOC is excluded**
   (it has no middle — it skips straight from head to solve) and keeps its short bespoke body.
   Meta-optimisers (NCO/Stacking/SubsetResampling) and clustering optimisers (HRP/HERC) are
   out of scope — they are not single-model builders.

## Considered options

- **Dispatch the bundle type inside the core** (accept `ProcessedJuMPOptimiserAttributes` or
  `NearOptimalSetup`, unwrap via dispatched accessors in the body). Rejected: it pushes NOC's
  field-name quirk into the shared core. The bespoke outer wrapper localises it instead.
- **Widen the bundle so a single `(model, optimiser, setup)` 3-arg call works** (give
  `ProcessedJuMPOptimiserAttributes` the scalar settings). Rejected: that struct is a stored
  public field of every Result, so widening it ripples into results and the factory/propagation
  machinery for a cosmetic argument saving.
- **Lift risk+scalarise out of the middle for everyone** to accommodate RRB. Rejected: it
  re-duplicates risk+scalarise across the four optimisers that do share it. The `r`-kwarg +
  `::Nothing` guard keeps the middle shared and RRB a single omission.
- **Drop RRB and/or constrained-NOC from scope.** Rejected: both share most of the middle;
  the `r`-kwarg and the bespoke NOC wrapper bring them in without bending the core.

## Consequences

- A solver-free test surface appears: `assemble_jump_model!` can be tested by building a
  model, running a minimal head, calling it with a hand-built `attrs`, and asserting on the
  constructed model through the Model State accessors and JuMP's constraint listings — no
  solve required. Existing per-optimiser CSV + solver tests are retained as numeric
  integration coverage; they stop being the only way to exercise assembly.
- Constraint ordering and new constraint types are edited in one place instead of five.
- The two processed-input representations still coexist, but the round-trip between them is
  gone (this was the ADR's deferred follow-up, now done): `processed_jump_optimiser` is a thin
  wrapper over `processed_jump_optimiser_attributes` + a new `jump_optimiser_from_attributes`
  repackaging helper, and constrained-NOC carries the `ProcessedJuMPOptimiserAttributes`
  through a new `NearOptimalSetup.attrs` field straight to the assembler — no `lcse`→`lcsr`
  remap, the processing happens once. Fully eliminating `processed_jump_optimiser` would
  require NOC's inner sub-problem solves to stop needing a `JuMPOptimiser`; left as-is.
- Rather than a separate public `assemble_jump_model!` wrapper, each `_optimise` builds its
  `attrs` once (reused for both assembly and the Result) and calls `assemble_jump_model!`
  directly. Constrained-NOC builds `attrs` from its processed in-place `opt` (the `NearOptimalSetup`
  fields); unconstrained-NOC has no middle and is untouched.

### Two latent inconsistencies in constrained-NOC, normalised on the way in

Routing constrained-NOC through the shared middle quietly normalised two oddities the
original code carried. Both are behaviour-neutral for the current test suite (the relevant
inputs are never configured for NOC), but they are real changes if those inputs are ever set:

1. The original passed `nothing` for the `set_smip_constraints!` sub-group-short-threshold
   slot while *storing* `opt.sgst` in the Result. The shared path uses `attrs.sgst = opt.sgst`
   for both, making them consistent.
2. The original passed `opt` (the `JuMPOptimiser`) as the `add_custom_constraint!` dispatch
   object, inconsistent with its own risk/tracking calls which pass `noc`. The shared path
   passes `noc` (the optimiser) uniformly.

The NOC tests configure neither sub-group short thresholds nor custom constraints, so both
are no-ops there; flagged here so a future reader knows they were deliberate, not accidental.

## Verification

Behaviour-preservation was checked against the existing CSV-baseline regression suites in a
live Julia session (edit → Revise → re-run), with a final cold reload to rule out load-order
or world-age artifacts. All green: MeanRisk (88/88 slice), Asset Risk Budgeting (235/235),
MIP Risk Budgeting (26/26 — exercises `miprb_flag = true`), Relaxed Risk Budgeting (37/37 —
exercises the `r = nothing` no-op), Factor Risk Contribution (4/4 — exercises the `b1` splat
and `sdp_asset_phylogeny = false`), and Near Optimal Centering (24/24, cold load).

The promised solver-free surface is realised in `test/test_03b_jump_model_assembly.jl`: it
runs the head + `assemble_jump_model!` and asserts on the registered Model-State keys
without solving — `r` routing (one indexed risk key per measure), the `r = nothing` no-op
(risk keys absent, head and return constraints still present), and `wn2`/`l1` toggling their
keys. 15 assertions, ~1 s — versus the multi-minute solver suites.

## Amendment (2026-08-17)

The architecture review of 2026-08-16 (candidate A) found that §3 conflated two jobs. How a
head shapes `w` does vary per optimiser, and §3 is right about that. Which `JuMPOptimiser`
settings reach the model does not vary, and it had no interface — so it was six hand-written
argument lists, and they had drifted. This amendment records the seam that closes the gap.
§3 is narrowed, not overturned: the head still owns `set_weight_constraints!`.

1. **The budget group travels as one object.** `set_weight_constraints!(model, wb, opt, long)`
   takes the `JuMPOptimiser` and forwards `bgt`, `sbgt` and `gbgt` together. Every head reaches
   the builder through it, so a budget field added to `JuMPOptimiser` is read in one place.
   Before this, `gbgt` was named at one head of five, so `JuMPOptimiser(; gbgt = …)`
   constructed, validated and solved under the other four with the leverage cap dropped.

   `gbgt` binds only where the long/short decomposition exists, which a negative weight bound
   forces. The two long-only head shapes — default `RiskBudgeting` and default
   `RelaxedRiskBudgeting` — therefore reject the combination through the `long = true` bound
   check rather than dropping it.

2. **`k` has one head-level producer.** `set_maximum_ratio_factor_variables!` registers both
   spellings: `k >= 0` under `MaximumRatio`, the literal `1` under every other objective. The
   heads whose formulation is fixed pass `MinimumRisk()` and take the second branch instead of
   hand-writing `@expression(model, k, 1)`. Its second method now takes exactly one objective;
   the former `args...` fallback absorbed a wrong-arity call and registered the constant under
   a ratio objective, which is how `test_03b_jump_model_assembly.jl` called it — so the ratio
   branch went untested while the call still looked correct.

   `RiskBudgeting` is the one head outside this producer. Its log barrier pins the scale, so
   `_set_risk_budgeting_constraints!` declares a *free* `k` and then `set_unit_budget!`.
   `get_k`'s error message names both routes.

3. **Unconstrained NOC reports the bundle it already holds.** §4 and the Consequences section
   describe unconstrained NOC as building its own `ProcessedJuMPOptimiserAttributes`. It was
   in fact hand-listing all 19 fields off a processed `JuMPOptimiser` that
   `jump_optimiser_from_attributes` had itself built from the `attrs` the setup already
   returns — a round trip whose hand list is the same silent drop that function's own comment
   records, pointing the other way. It now uses `attrs` directly, as the constrained head does.
   There is deliberately no inverse remap helper: the bundle is never reconstructed, it is kept.

§6 still holds — unconstrained NOC has no middle, and this amendment does not give it one.

## Amendment 2 (2026-08-17)

The maintainability review of 2026-08-17 (finding 5) found that §6 is wrong on its own terms,
and that the exclusion had cost a defect. §6 is withdrawn. The amendment above ends with the
sentence "§6 still holds"; that sentence is superseded here, and the rest of that amendment is
untouched.

### The finding

Unconstrained NOC does not "skip straight from head to solve". It ran three steps of the
shared middle inline — `set_risk_constraints!` → `scalarise_risk_expression!` →
`set_return_constraints!` — and its argument list had drifted from the sequence it copied:

```julia
set_risk_constraints!(model, r, noc, opt.pe, nothing, nothing, opt.fees; rd = rd)
```

The dispatched signature is `(model, r, opt, pr, pl, fees, args...)`. So `nothing` bound
`fees`, and `opt.fees` fell into `args...`. Unconstrained NOC built every risk constraint
fee-free from `4cd418494` — the commit that appended `opt.fees` to a call that already carried
two `nothing`s — until this amendment. The intent never took effect, and nothing raised,
because the callee's `args...` tail absorbed the overshoot. This is the second instance of the
failure §2 of the first amendment records, and the second one found in this file.

### Decision

1. **Unconstrained NOC has a middle.** It is not `assemble_jump_model!`'s middle: the variant
   applies no linear, cardinality, turnover, tracking, norm, regularisation, phylogeny or
   custom constraint, and that omission is what "unconstrained" names. So the middle is named
   and dispatched instead of excluded. `assemble_near_optimal_centering_model!(alg, model,
   noc, setup, rd)` has one method per variant: the constrained one delegates to
   `assemble_jump_model!`, the unconstrained one runs risk + scalarise + return and the same
   `assert_frontier_sweep_cap` tail.

2. **One head, one Result.** The two `_optimise` methods differed in 6 of 34 lines. They are
   now one method that dispatches the middle through
   `assemble_near_optimal_centering_model!` and the solve through
   `solve_near_optimal_centering!(alg, model, noc, setup)`, which reads each `solve_noc!`
   overload's arguments off the `NearOptimalSetup`. The 12-line Result block is written once.

3. **The fee argument is positional in one place.** The unconstrained middle reaches
   `set_risk_constraints!` through `set_risk_and_scalarise!`, as every other head does. A head
   can no longer order that list for itself.

4. **The absorber is removed.** Both generic `set_risk_constraints!` methods and both
   `set_risk_and_scalarise!` methods take a named, typed `b1::Option{<:MatNum} = nothing` in
   place of the `args...` tail. `b1` is the only trailing value the middle has ever passed, and
   every caller already passed exactly one, so no call site changes arity. A `Fees` in that
   slot is now a `MethodError` instead of a silent loss. The `::Nothing` no-op method carries
   the same positional list rather than an `args...` tail — a tail is ambiguous against a typed
   one, and it accepts calls its sibling would reject.

5. **The unconstrained middle ends at `assert_frontier_sweep_cap` too.** The call is a no-op
   there today: `near_optimal_centering_setup` hands the variant the `no_bounds_risk_measure`
   and `no_bounds_optimiser` copies, so neither frontier registry can be populated. The sweep
   the variant does run is parameterised over the `rk_opt`/`rt_opt` vectors its sub-problem
   solves produce, and the cap applies inside those sub-problems. Making the call keeps the
   invariant structural instead of resting on an argument about which bounds survive.

§3 of the first amendment still holds: the head keeps `set_weight_constraints!`, and the two
variants share it.

### Verification

`test_20_near_optimal_centering_optimisation.jl`, `test_03b_jump_model_assembly.jl`,
`test_19_factor_risk_contribution.jl`, `test_18n_frontier_sweep_cap.jl`,
`test_18m_return_multiplicity.jl`, `test_18o_no_return.jl`,
`test_15_relaxed_risk_budgetting_optimisation.jl` (the `r = nothing` no-op),
`test_16a_asset_risk_budgeting.jl`, `test_18a_mean_risk_1.jl` and `test_28_seam_lock.jl` all
pass. No CSV baseline moved: no test configures fees on a `NearOptimalCentering`, which is why
the dropped fees went unnoticed.

## Amendment 3 (2026-08-18)

The architecture review of 2026-08-17 (finding 1) revisits the same seam from the caller's
side. Amendment 2 gave unconstrained NOC a named middle; this amendment says what that middle
reads, fixes the one setting it read inconsistently, and records why the rest is documented
rather than refused.

### The census

Every `JuMPOptimiser` setting was set one at a time on a default
[`NearOptimalCentering`](../../src/20_Optimisation/13_NearOptimalCentering.jl) with the three
anchor portfolios supplied, and the assembled centring model was compared byte for byte
against the same model without the setting. `lcse`, `card`, `l1`, `linf`, `lp`, `l2c`,
`linfc`, `tn`, `tr` and `ss` all leave the model identical. They are carried and validated,
and the builder each one drives belongs to the middle this variant does not run.

They are **not** inert, which is why "carried and dropped" overstates the case. The three
anchor portfolios are `MeanRisk` sub-problems that run the whole `assemble_jump_model!`, so an
omitted setting shapes the anchors and, through them, the centring target. It reaches no model
at all only when `w_min`, `w_opt` and `w_max` are all supplied and no sub-problem is solved.

### The defect: the return expression was gross, the target net

`near_optimal_centering_setup` computes `rt_min`, `rt_opt` and `rt_max` with
`expected_return(ret, w, pr, fees)` — net of fees. The barrier constrains `ret - rt`, where
`ret` is the model's return expression. `add_fees_to_ret!` subtracts the model's `:fees`
expression only when one is registered, and `set_non_fixed_fees!` is what registers it. The
unconstrained middle did not run it, so the two halves of one comparison used different units
whenever a fee was set: with `Fees(; l = 0.05)` on a five-asset problem the model's return
expression was unchanged from the fee-free one, while every coefficient should have moved by
`0.05` — about fifty times the gross expected return itself.

`set_non_fixed_fees!(model, opt.fees)` now runs first in the unconstrained middle, in the same
position it holds in `assemble_jump_model!`. A **fixed** fee still does not apply on this path
and cannot: it is charged per position held, so it needs the cardinality binaries
`set_mip_constraints!` produces, and that builder is part of the omitted middle. The fee shapes
`l`, `s`, `tn`, `fl` were each checked to build without error on the unconstrained path, under
long-only and long-short bounds.

This is the third fee defect on this one path, after amendment 2's argument-position drift. All
three had the same shape: a value that the setup computes one way and the model consumes
another, with nothing comparing the two.

### What is documented rather than refused

The review asks for a membership declaration that the head checks, so that a setting the
formulation cannot honour raises instead of being ignored. That half is **not** done, and the
exclusion is stated in the docstrings instead —
[`UnconstrainedNearOptimalCentering`](../../src/20_Optimisation/13_NearOptimalCentering.jl)
now lists every setting the centring model reads and every setting it does not, and the `alg`
field text names the choice as the thing that selects between them.

Three reasons:

1. **The settings are meaningful on this variant.** They bind on the anchors. A head that
   raised on them would refuse configurations that produce a correct answer today, and would
   force a user who wants a constrained anchor and an unconstrained centring to give up one of
   the two.
2. **It is breaking on the default path.** `UnconstrainedNearOptimalCentering` is the default
   `alg`, so every `NearOptimalCentering` that sets one of these fields and solves today would
   start throwing, with no migration but a change of `alg` that changes the answer.
3. **The gap the review names is a documentation gap first.** "No configuration error can be
   raised" was true, but the reader had no way to learn the exclusion either. One of those two
   is cheap to close and reversible.

The declaration remains a reasonable future step, and this amendment does not argue against
it. It belongs with a decision about what the raise should be — an error, a warning, or a
`strict`-gated pair — and that is a wider change than one path.

### Verification

`test_20_near_optimal_centering_optimisation.jl`, "Unconstrained NOC return expression is net
of fees": builds the head and the middle with the three anchors supplied, so no solve runs, and
asserts that the model's return expression and the `rt_opt` target both move with the fees, and
that a fixed fee moves neither. Proved to discriminate — before the fix the two return
expressions are byte-identical.
