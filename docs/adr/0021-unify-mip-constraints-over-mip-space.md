---
status: accepted
---

# One MIP constraint family dispatched over an `AbstractMIPSpace`

## Context

`05_MIPConstraints.jl` (superceded by `01_MIPIndicators.jl`) carried
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

## Amendment (2026-11-23)

The concentration limit (#2018, ADR 0245) gates a binary of its own, and it would have been the
fifth copy of the rows that link a binary to a free budget `k`. The held bit, the long bit, the
short bit and the exact cost budget held the other four.

- **`declare_mip_gate!(model, sp, bin, gate, N, wb, ss)` declares the binaries and their gates.**
  Under a numeric `k` the gate is the binary. Under a variable `k` the gate is a continuous
  variable with its three big-M rows. The five sites call it, and it registers each key through
  `state_set!`, so a second declaration under one key raises.
- **`use_direct_mip_indicators` is gone.** The function tests `isa(k, Number)` in every space. The
  shortcut of the asset space bound the unscaled weights of a `RiskBudgeting` head (#2275).
- **`ConcentrationMIPSpace` is a member of the family.** It implements `mip_wx!` and `mip_key`. A
  space that gates no weight bound has no `mip_bounds`.
- The rows of the exact cost budget take the keys `cost_bgt_gd_ub` and `cost_bgt_gd_lb`, after
  the form of the other gates, where they had `cost_bgt_gz_ub` and `cost_bgt_gz_lb`. Under a
  numeric `k` the model holds no `cost_bgt_g`.
