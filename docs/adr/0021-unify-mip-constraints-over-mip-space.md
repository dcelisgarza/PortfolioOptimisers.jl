---
status: accepted
---

# One MIP constraint family dispatched over an `AbstractMIPSpace`

## Context

[05_MIPConstraints.jl](../../src/20_Optimisation/09_JuMPConstraints/05_MIPConstraints.jl) carried
two nearly-parallel families of binary-indicator constraint builders:

- **Asset-space MIP** (`mip_wb`, `short_mip_threshold_constraints`, `mip_constraints`): binary
  indicators gate the portfolio weights `w` directly; model keys are the bare names (`:ib`,
  `:i_mip`, `:w_mip_lt`, …).
- **Sub-group SMIP** (`smip_wb`, `short_smip_threshold_constraints`, `smip_constraints`): the same
  cardinality / threshold / weight-bound logic, but indicators gate the *sub-group* weights
  `smtx * w` (through a selection matrix), weight bounds map through that matrix, and keys are
  namespaced per group so multiple sub-groups do not collide.

The two families differed only in three things — **which weight expression the indicators gate**,
**how weight bounds map into that space**, and **how model keys are named** — yet each duplicated
the full cardinality / group-cardinality / long-short threshold / weight-bound machinery. Every fix
or new constraint had to be written twice and kept in sync; the file was ~640 lines of largely
mirrored code.

## Decision

**Extract the three points of difference into a small `AbstractMIPSpace` interface and write the
constraint builders once against it.**

- `abstract type AbstractMIPSpace end` with two members:
  - `AssetMIPSpace` — indicators gate `w`; bare key names.
  - `SubsetMIPSpace{T1,T2}(smtx, pfx, i)` — indicators gate `smtx * w`; bounds map through `smtx`;
    keys namespaced as `Symbol(pfx, name, :_, i)` so cardinality, group-cardinality and per-group
    variants never collide.
- The interface is three methods: `mip_wx!(model, sp)` (the gated weight expression),
  `mip_bounds(sp, b)` (bound mapping) and `mip_key(sp, name)` (key naming), plus
  `use_direct_mip_indicators(model, sp, k)` for the asset-space shortcut.
- The former `smip_*` functions are deleted. `mip_constraints` / `mip_wb` /
  `short_mip_threshold_constraints` now take an `sp::AbstractMIPSpace`. The asset entry point
  constructs `AssetMIPSpace()`; the sub-group entry point constructs a `SubsetMIPSpace(smtx, …)`
  per selection matrix (`:s` for cardinality, `:sg` for group cardinality) and loops.
- Asset and sub-group builders share the single `:ss` short-selling expression via
  `set_mip_ss_expr!` / `get_mip_ss`: the first caller registers it, later callers reuse it.

## Considered options

- **Keep two families.** Rejected: the duplication is the whole problem — every threshold/cardinality
  change is a double edit with a silent-drift failure mode.
- **Parameterise with flags/closures instead of a type.** Rejected: the differences are a small
  fixed set of behaviours (gate expression, bound map, key naming) with exactly two implementations
  — dispatch over a two-case abstract type states that cleanly and lets a future third space (should
  one arise) slot in by implementing three methods, whereas threading closures obscures the seam.

## Consequences

- Net ~90 fewer lines and, more importantly, one place to change MIP constraint logic. Asset and
  sub-group cardinality/threshold constraints are provably the same code on different spaces.
- **Structural, not behavioural**: the emitted constraints and model keys are unchanged, so this is
  transparent to callers and to solved outputs.
- Adding a new "MIP space" (a different weight expression the indicators should gate) is now a
  `struct <: AbstractMIPSpace` plus `mip_wx!` / `mip_bounds` / `mip_key` methods — the builders come
  for free.

## Extended by

- **[ADR 0033](0033-split-mip-file-into-indicator-layer-and-emitters.md)** carries the same
  "write it once against `AbstractMIPSpace`" decision into the cardinality/group-cardinality
  emitters (`set_card_constraints!` / `set_gcard_constraints!`) and the builder selector
  (`run_mip_builder!`), which this ADR left space-specific — folding the sub-group cardinality
  path onto the seam. It also splits the unified file into an indicator layer plus topical
  emitter files (`05_MIPConstraints.jl` → `01_MIPIndicators.jl` + siblings).
