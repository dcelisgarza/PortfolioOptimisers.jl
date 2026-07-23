---
status: accepted
---

# Split the MIP constraint file into an indicator layer and topical emitters

## Context

[ADR 0021](0021-unify-mip-constraints-over-mip-space.md) unified the asset-space and sub-group
MIP builders over an `AbstractMIPSpace`, but the file it left behind — `05_MIPConstraints.jl`,
~986 lines — still conflated two unrelated jobs:

- **Declaring the binary indicators** and the constraints that give them meaning (the
  weight-bound gating that makes a bit *mean* held or short).
- **Emitting every feature that happens to need an indicator** — minimum-holding thresholds,
  fixed fees, cardinality, group cardinality, integer phylogeny, and the exact-budget (`xbgt`)
  decomposition — each parked in the same file only because it consumes a bit.

Because everything lived together, six unrelated features shared one file, and the coupling hid
two bugs:

1. `set_iplg_constraints!` read the held indicator back out of the model as `model[:ib]`, a key
   only the long-only builder registers; a short + (threshold | fixed fee | `xbgt`) + integer
   phylogeny model registers `ilb`/`isb` instead and threw `KeyError(:ib)`. It went untested
   because every `IntegerPhylogeny` test is long-only.
2. The RB mixed-integer flag (`miprb_flag`, ADR 0008) and the user-facing `xbgt` were emitting
   the *same* constraints under two names, kept in sync by hand.

Two further mechanisms had drifted from what they meant:

- `assemble_jump_model!` carried `miprb_flag::Bool` (ADR 0008's implementation note) purely so
  risk budgeting could ask for the decomposition to be pinned — a second name for `xbgt`.
- A `:crkb` model-key sniff at four sites asked "may I gate with `1`?" under a docstring that
  called it a *fixed budget*, when RB's `k` is a free variable whose *scale* the log-barrier
  normalisation pins.

## Decision

**Keep the MIP file as an indicator *layer* and move every indicator *consumer* to a file named
for the feature.**

1. **Indicator layer (`01_MIPIndicators.jl`).** Holds only: the `AbstractMIPSpace` types and
   interface (ADR 0021), the typed indicator bundles, the three declare-and-delegate builders
   (`mip_constraints`, `short_mip_threshold_constraints`, `sign_mip_constraints`), the shared
   builder selector `run_mip_builder!`, and the `set_mip_constraints!` orchestrator. Each builder
   makes the gated weight expression, declares its binaries, applies the weight-bound gating,
   delegates feature emission, and returns the *held* indicator.

2. **Typed indicator bundles, read by accessor, never by key.** `HeldIndicators` (`ib`),
   `LongShortIndicators` (`ilb`/`isb`), `SignIndicators` (`xb`) — a *held* bit says an asset is
   in the portfolio (what cardinality counts), a *sign* bit says which side. Emitters take the
   bundle and read it through `held` / `lb_gate` / `long_gate` / …, not `model[...]`. `held`
   deliberately has no `SignIndicators` method: a caller that needs a held bit where none exists
   fails at the seam with a `MethodError` instead of reading a sign bit and silently miscounting.
   This is the durable fix for bug 1 — and the held indicator is now *returned* from
   `set_mip_constraints!` and threaded to the late `set_iplg_constraints!` call, never re-read by
   key.

3. **Topical emitters.** Fixed fees → `10_FeesConstraints.jl` (beside the non-fixed fees),
   integer phylogeny → `06_IntegerPhylogenyConstraints.jl` (applied *late* in
   `assemble_jump_model!` beside the SDP phylogeny, since FRC skips SDP but still needs it),
   cardinality + group cardinality → `07_CardinalityConstraints.jl`, minimum-holding thresholds
   → `08_ThresholdConstraints.jl`, the exact-budget decomposition → `03_BudgetConstraints.jl`.
   Each file is named for what it *is*, not for the fact that it needs a bit.

4. **`xbgt` and `miprb` are one concept, dispatched on a head contract.** They pin the same
   long/short decomposition; they differed only because RB's head makes `w = lw − sw` an
   *identity* (a pin-pair suffices) while a standard head makes `lw`/`sw` *bounds* (two more
   constraints close the slack). Both now route to a single `set_exact_budget_constraints!` that
   reads a **decomposition contract** off Model State — `WeightsFromParts` (RB, asks by
   construction, no flag) or `PartsBoundWeights` (explicit `xbgt`) — and emits what the contract
   needs. `miprb_flag` is deleted from `assemble_jump_model!` (superseding ADR 0008's
   implementation note).

5. **`is_unit_budget` / `effective_k` replace the `:crkb` sniff.** One named predicate at the
   four sites, honest that `k` is free and only its scale is pinned.

6. **Fold the sub-group path onto the seam (completing ADR 0021).** ADR 0021 unified the
   *builders*; the cardinality/group-cardinality *emission* and the *builder selection* were
   still triplicated across the sub-group functions. `set_card_constraints!` /
   `set_gcard_constraints!` now take an `AbstractMIPSpace` and key via `mip_key`, and
   `run_mip_builder!` is the single builder selector. The asset orchestrator and all three
   sub-group emitters route through them; a sub-group differs from the asset space only through
   its `sp` (card 3→1, gcard 3→1, builder selection 4→1).

## Considered options

- **Leave the file as one unit.** Rejected: the conflation is what hid both bugs and forced the
  double-emit of `xbgt`/`miprb`. Naming files for features, not for the mechanism they borrow,
  is what lets a reader find "where are fixed fees" without knowing they need a binary.
- **Keep `miprb_flag` as a distinct flag.** Rejected: it is `xbgt` under another name. A head
  contract states the *reason* the pin is wanted (identity vs bound) and picks the closing
  constraints from that, instead of a boolean that means "RB is calling."
- **Read indicators back from the model by key in each emitter.** Rejected: that is exactly bug
  1. A typed bundle with no held accessor for the sign-only case turns a silent miscount into a
  compile-time seam error.

## Consequences

- The MIP file drops from ~986 lines to the indicator layer alone; five features live in files
  named for themselves.
- **Behavioural neutrality where claimed.** The split, the bundle, the head-contract merge, and
  the sub-group fold are all structural — emitted constraints are unchanged (verified against the
  MIP-constraint, risk-budgeting, and model-assembly test sets). The one deliberate behaviour fix
  is bug 1 (the `KeyError` path).
- Asset-space model keys are unchanged; sub-group keys remain `mip_key`-namespaced. The indicator
  keys are now locked against bare `model[:key]` reads by the seam-lock test
  (`test_28_seam_lock.jl`), so bug 1's regression class cannot return.
- A new indicator-consuming feature is now a new topical file that takes the bundle; a new *kind*
  of bit is a new bundle plus its accessors; a new weight space is still just the three
  `AbstractMIPSpace` methods of ADR 0021.

## Amends

- **[ADR 0008](0008-jump-model-assembly.md)** — the `miprb_flag` in the `assemble_jump_model!`
  signature (item 2's implementation note) is removed; risk budgeting asks for the exact
  decomposition through the Model State head contract read by `set_exact_budget_constraints!`, not
  a middle-of-assembly flag. The middle still never touches weight/budget bounds (ADR 0008 §3
  holds).
- **[ADR 0021](0021-unify-mip-constraints-over-mip-space.md)** — extends its "write the builders
  once against `AbstractMIPSpace`" decision to the cardinality/group-cardinality emitters and the
  builder selector, which ADR 0021 left space-specific.
