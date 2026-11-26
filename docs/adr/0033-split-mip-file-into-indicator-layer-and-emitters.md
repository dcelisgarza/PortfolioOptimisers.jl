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

## Amendment (2026-10-27)

Item 5 calls `k` free. Since #1844 and #1845 the `k` of `RiskBudgeting` is a variable bounded by
`k >= kmin >= 0` (ADR 0126), so "free" means only that `k` is not the number `1`. The log barrier still fixes the scale of the
unscaled weights, and `effective_k` still answers `1` under a unit budget.

## Amendment (2026-11-23)

Item 5 gave one predicate to four sites. One of them, the choice of the gates of the asset space, read
it wrongly (#2275). `effective_k` is `1` for the builder of a risk or of a penalty, because the
objective of `RiskBudgeting` measures the scaled weights alone. A bound, a threshold, a target or
the gate of a binary reads `get_k` under every head, because the head divides the weights by `k`
after the solve. `is_unit_budget` stays the one predicate of the record. `effective_k` reads it, and
so does a builder that gives one risk to the objective and to a bound (#2279).

A risk expression that reads `effective_k` has a degree above one in the scaled weights, so it cannot
also be the expression of the bound row `risk <= ub * k` (#2279). `set_risk_epigraphs!` builds such a
risk once with each scale when a measure has a bound under a unit budget, and once in every other
case. The bound of a semidefinite variance takes the second-order cone of the deviation, which is
exact on the weights and needs no second lift. The bound of an even moment takes the model
of `set_even_moment_bound!`, in units of the bound. On 96 small models Clarabel solved all of them in
that form, and 32 with the norm of order `2p` or with a copy of the epigraph with `k` in its
cones.

A tracking builder has the same need for a tracked measure of any degree (#2283), and it cannot call
`set_risk_epigraphs!` for it. It puts a `TrackedRiskBound` into the bound slot of the tracked measure.
The builder of the measure converts this request as it converts a bound, and `set_risk_upper_bound!`
registers the expression of degree one in place of a row. The tracking builder then writes its own row,
so the keys of the rows do not move, and every head honours the bound. The rejected form gave the
tracked measure the bound as its own `settings.ub`. A `FactorRiskContribution` ignores that bound, and the
dependent mode needs an expression and not a row.

## Amendment (2026-11-26)

The bounds of an asset give the side of its held bit (#2372). Item 2 said that a held bit has no
sign. That is true only for an asset whose bounds admit both signs. An asset with `lb >= 0` is
long when it is held, and an asset with `ub <= 0` and `lb < 0` is short when it is held.

The selector `run_mip_builder!` read the short part `sw` of the model to choose a builder, and a
long threshold alone took the held builder. That builder wrote `w >= lt * ib` on every asset, so
a long-short book could hold no short, and a short-only book was infeasible. A short threshold on
a short-only book took the long-short builder and its `2N` binaries.

Now `mip_sides` reads the bounds of each entry of the space, and the rule is:

1. A pin with no consumer of a held bit takes the sign builder, for an explicit `xbgt` and for a
   `WeightsFromParts` head alike. The held builder had served that head, and its bounds forbade
   every short when they were finite.
2. An explicit `xbgt` with a consumer of a held bit takes the long-short builder, as ADR 0032
   states.
3. A threshold, a fixed fee or a pin takes the long-short builder when an entry admits both
   signs.
4. Every other call takes the held builder. `HeldIndicators` carries the field `sht`, the entries
   whose bounds admit a short weight and no long one. The thresholds, the fixed fees and the gates
   of the pin read the side of each entry from it. `held_sides` sets it for each call, because two
   calls with two sets of bounds can read one bundle.

So a short-only book is the mirror of a long-only book: `N` binaries, and rows with no big-M
slack. A book whose assets each admit one sign takes `N` binaries too. The rejected form added
`lt_flag` to the old condition of the selector. It corrects the long-short case, but a short-only
book keeps `2N` binaries, the selector still reads the model and not the bounds, and the
`WeightsFromParts` head keeps its defect.
