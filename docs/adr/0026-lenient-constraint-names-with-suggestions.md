---
status: accepted
---

# Keep the lenient `strict = false` constraint default; improve the diagnostic instead of failing closed

## Context

Finding 1 (High) of the 2026-07-06 security review recommended flipping `get_linear_constraints`
(and the Black-Litterman view generator) from `strict = false` to `strict = true` by default, so a
misspelled asset name in an untrusted constraint/view string fails closed with a typed error rather
than dropping a term/row behind a `@warn` — the one silent-wrong-answer path reachable from the
primary API with default settings.

The recommendation is sound *for a caller who names a fixed, fully-known universe*. But it collides
with a first-class use of the library: **meta-optimisers** (`NestedClustered`, `SubsetResampling`,
Cross-Validation) run the same user constraints against many *subsets* of the universe. A constraint
like `AAPL >= 0.05` is meaningful even when `AAPL` is absent from a particular cluster or resampled
subset — the caller writes the constraint once, over the full universe, without knowing which assets
each inner problem will contain. Dropping the unmatched term/row is the *intended* behaviour there,
not a mistake; `strict = true` as a default would make the meta-optimisers throw on every subset that
happens to exclude a named asset.

So the fail-open default is load-bearing. The real defect the review identified is narrower: when a
dropped name *is* a typo, the only trace is a warning that (a) said nothing about which valid name was
meant and (b) dumped the whole universe/struct into the log (finding 4).

## Decision

**Keep `strict = false` as the default and keep `strict = true` as the explicit opt-in for callers
with a fixed universe. Instead of flipping the default, make the diagnostic good enough that a real
typo is obvious while a legitimately-absent asset stays quiet.**

- The "variable not in asset universe" message (both the `@warn` and the `strict` `ArgumentError`)
  gains a fuzzy `" (did you mean \`AAPL\`?)"` suggestion, produced by `did_you_mean`using
  [`StringDistances.jl`](<https://github.com/matthieugomez/StringDistances.jl>).
- The suggestion is **threshold-gated**: it only appears when the nearest candidate clears a
  normalised-similarity cutoff (`findnearest(...; min_score)`). A typo (`APL` vs `AAPL`, score 0.75)
  fires; an asset that is simply absent from this subset has no close neighbour and draws *no*
  suggestion — so the meta-optimiser loops are not flooded with nonsense guesses.
- The distance and threshold live in a global, mutable config, `STRING_DISTANCE`, set via
  `set_string_distance!(; dist, min_score)` — mirroring the existing `COMPACT_SHOW` /
  `set_compact_show!` pretty-printing config. Setting `min_score > 1` disables suggestions entirely.
- The message builders (`unknown_variable_msg`, `empty_row_msg`) and the suggestion helper live once
  in `02_Tools.jl` and are called by both `get_linear_constraints`
  ([02_LinearConstraintGeneration.jl](../../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl))
  and the Black-Litterman view generator
  ([05_BlackLittermanViewsGeneration.jl](../../src/13_Prior/05_BlackLittermanViewsGeneration.jl)),
  so the two previously-drifted copies of this message can never diverge again.

## Considered options

- **Flip the default to `strict = true` (the review's recommendation).** Rejected: it breaks the
  meta-optimiser / cross-validation use of constraints over asset subsets, which depends on unmatched
  terms being dropped. A breaking default change that also removes a needed behaviour.
- **Self-contained edit-distance instead of a dependency.** Considered (it would avoid adding a
  dependency to the same codebase we are hardening). Rejected in favour of the battle-tested
  `StringDistances.jl`, which supplies `findnearest` with `min_score` gating and multiple metrics
  directly.
- **Hardcode the similarity threshold.** Rejected: a global config consistent with the existing
  pretty-print config lets callers tune or disable suggestions (e.g. silence them in tight
  meta-optimiser loops, or widen them for short tickers) without a code change.
- **Add the suggestion to the all-zero-row message too.** Rejected: that message only fires after the
  per-variable messages have already emitted their suggestions, or when valid coefficients cancelled
  (no typo to suggest) — so a suggestion there would be redundant or meaningless.

## Consequences

- The silent-wrong-answer path the review flagged is mitigated without a breaking default change: a
  real typo now names the likely intended asset in the log (and in the `strict` error), while the
  meta-optimiser use of absent-by-design names is preserved and un-noised.
- New dependency: `StringDistances.jl` (`[compat] = "1"`). Weighed against the security posture, but
  small and pure-Julia.
- New public-ish surface: `set_string_distance!` / `STRING_DISTANCE`, callable qualified like
  `set_compact_show!` (neither is exported).
- Finding 1 remains *open* in the sense that the default is still fail-open by design; this ADR
  records that this is intentional. Callers who want fail-closed opt in with `strict = true`.
- Regression coverage in
  [test_02_equation_parsing.jl](../../test/test_02_equation_parsing.jl) ("Asset name suggestions").
  Related hardening of the same boundary is in
  [ADR 0025](0025-enumerated-parser-allowlist.md).
