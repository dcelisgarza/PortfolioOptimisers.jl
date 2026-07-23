---
status: accepted
---

# The MIP indicator bundle lives in Model State, not in a return value

## Context

[ADR 0033](0033-split-mip-file-into-indicator-layer-and-emitters.md) gave the MIP builders a
typed indicator bundle (`HeldIndicators`, `LongShortIndicators`, `SignIndicators`) so that every
feature reads a bit through an accessor (`held` / `long_gate` / …) rather than out of the model
by key. That closed the *emitter* seam: a consumer running inside the builder cannot fabricate
the wrong key.

One consumer runs *outside* the builder, though. Integer phylogeny is applied late in
`assemble_jump_model!`, beside its semidefinite sibling, long after `set_mip_constraints!` has
returned — because a factor-risk-contribution model skips the SDP block but still needs the
integer one. ADR 0033 fed it the held indicator by **returning it from `set_mip_constraints!`
and hand-threading it through `assemble_jump_model!`** to the late `set_iplg_constraints!` call.

That left the abstraction only *externally* policed:

- The held indicator's identity still crossed the seam as a bare value; the "reach an indicator
  only through the bundle" rule was enforced by a source-regex seam-lock test
  (`test_28_seam_lock.jl`), not by the interface itself.
- `assemble_jump_model!` carried a `mip_held` local for the sole purpose of relaying one value
  between two of its own calls — a thread whose only reason to exist was that the bundle had
  nowhere to live between them.
- Every alternative builder had to agree on *which* extracted value to return
  (`ib` vs `i_mip`), re-deriving the "held binary" concept at the return statement instead of
  behind one accessor.

## Decision

**Register the asset-space indicator bundle as a first-class Model State entry, and let the late
emitter read it back through the typed accessors — the same named-interface pattern
[ADR 0004](0004-typed-jump-model-state.md) gave the rest of Model State, and that
`decomposition_contract` already uses.**

1. **`set_mip_indicators!(model, ind)` / `mip_indicators(model)`.** The builder that runs in
   `set_mip_constraints!` registers its bundle under one canonical entry (`:mip_indicators`);
   `mip_indicators` returns it, or `nothing` when no MIP builder ran. Only the asset space
   registers a bundle — sub-group builders consume theirs in the same call and never cross the
   late seam.

2. **The builders return the bundle, not an extracted value.** `mip_constraints`,
   `short_mip_threshold_constraints`, and `sign_mip_constraints` all return their
   `AbstractMIPIndicators`; `run_mip_builder!` passes it through. `set_mip_constraints!` returns
   `nothing`.

3. **`held_bin` names the held *binary*.** The value the old return threaded — `ib` for the
   long-only bundle, `ilb + isb` for the long-short one — is the *held binary* that cardinality
   sums and integer phylogeny multiplies. It is distinct from `held` (the held *gate*, which is
   the continuous relaxation `ibf` under a free budget) and now has its own accessor, mirroring
   how `long_bin` sits beside `long_gate`. Cardinality, group cardinality, and integer phylogeny
   key on `held_bin`; the minimum-holding threshold keeps keying on `held`. Like `held`, it has
   no `SignIndicators` method, so a caller that needs a held count where none exists fails at the
   seam with a `MethodError` — reached only inside the `IntegerPhylogeny` branch, which
   `set_mip_constraints!`'s own guards prove a sign-only bundle rules out.

4. **`set_iplg_constraints!` reads the bundle back.** It takes only `(model, plgs)`, calls
   `mip_indicators(model)`, and gates on `held_bin`. `assemble_jump_model!` drops the `mip_held`
   local.

## Considered options

- **Keep threading the return value (ADR 0033 status quo).** Rejected: it is the one place the
  bundle abstraction is enforced by a regex over source rather than by the interface, and it
  makes `assemble_jump_model!` carry a value between two calls purely because the bundle had no
  home.
- **Reuse `held` for cardinality/phylogeny instead of adding `held_bin`.** Rejected: `held`
  returns the *gate*, which under a free budget is the continuous relaxation `ibf`; counting must
  key on a bit that is exactly 0 or 1. Reusing it would silently change behaviour on the
  free-budget path. `held_bin` reproduces the old threaded value exactly.
- **Put the accessors in the Model State module (`08_Base_JuMPOptimisation.jl`), beside
  `decomposition_contract`.** Rejected: the bundle *types* are defined in the indicator layer,
  included after that module, so a typed setter there cannot name them. The indicator layer is
  also where "indicator identity is decided", so the accessors belong with the bundle.

## Consequences

- **Behaviourally neutral.** `held_bin` returns exactly what each builder used to return, so the
  emitted cardinality, group-cardinality, and integer-phylogeny constraints are unchanged
  (verified against the model-assembly and MIP/phylogeny test sets). The change is purely where
  the bundle lives between declaration and its late consumer.
- The seam is now held by the interface: a late emitter *cannot* get a raw indicator except
  through `mip_indicators` + an accessor. The `test_28_seam_lock.jl` regex stays as a backstop
  against a fresh bare `model[:ib]` read, but it is no longer the only thing holding the seam.
- A future late emitter that needs the held indicator has a place to read it, with no new
  argument to thread through `assemble_jump_model!`.

## Extends

- **[ADR 0033](0033-split-mip-file-into-indicator-layer-and-emitters.md)** — completes the bundle
  seam it introduced: the bundle is now Model State, so even the late integer-phylogeny consumer
  reaches it through the typed interface rather than a hand-threaded return value.
- **[ADR 0004](0004-typed-jump-model-state.md)** — one more named Model State entry with a setter and a
  reader, in the same spirit as `decomposition_contract`.
