---
status: accepted
---

# Extract the shared JuMP model-assembly pipeline into one deep module

## Context

Every single-JuMP-model Optimisation Estimator builds its `JuMP.Model` by running the
same long sequence of constraint and risk builders inline in its own `_optimise` method.
The sequence — `set_linear_weight_constraints!` (×2) → `set_mip_constraints!` →
`set_smip_constraints!` → `set_turnover_constraints!` → `set_tracking_error_constraints!`
→ `set_number_effective_assets!` → `set_l1/l2/linf/lp_regularisation!` →
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
   _assemble_jump_model!(model, optimiser, opt, attrs, rd;
                         r = nothing, b1 = nothing, obj = MinimumRisk(),
                         miprb_flag = false, sdp_phylogeny = true)
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
   (`false`), and `sdp_phylogeny` (`true`). The `b1` value is splatted in via an optional
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
   > (→ `sdp_phylogeny = false`, with FRC's `set_sdp_frc_phylogeny_constraints!` left
   > caller-side).

3. **Head and tail stay per-optimiser.** Each `_optimise` keeps its own head (including
   `set_weight_constraints!`, which is intrinsic to how each head shapes `w`) and its own
   tail (objective function + `solve_*!` + Result construction). The middle never touches
   weight/budget bounds or the objective.

4. **NOC integrates through a bespoke outer method, not by bending the core.** A thin
   `assemble_jump_model!(noc, model, setup::NearOptimalSetup)` unwraps the
   `NearOptimalSetup` (extracting `setup.opt` for settings and building the
   `ProcessedJuMPOptimiserAttributes` NOC already constructs for its Result), then delegates
   to the shared `_assemble_jump_model!`. The `lcse`↔`lcsr` field-name divergence is confined
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

- A solver-free test surface appears: `_assemble_jump_model!` can be tested by building a
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
  `attrs` once (reused for both assembly and the Result) and calls `_assemble_jump_model!`
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
and `sdp_phylogeny = false`), and Near Optimal Centering (24/24, cold load).

The promised solver-free surface is realised in `test/test_03b_jump_model_assembly.jl`: it
runs the head + `_assemble_jump_model!` and asserts on the registered Model-State keys
without solving — `r` routing (one indexed risk key per measure), the `r = nothing` no-op
(risk keys absent, head and return constraints still present), and `nea`/`l1` toggling their
keys. 15 assertions, ~1 s — versus the multi-minute solver suites.
