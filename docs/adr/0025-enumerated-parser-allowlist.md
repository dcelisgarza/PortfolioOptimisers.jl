---
status: accepted
---

# Enumerate the equation parser's callable functions instead of resolving names against `Base`

## Context

Constraint equations, Black-Litterman view strings, entropy-pooling view strings and asset-set
names are untrusted input: a config file, spreadsheet, or UI feeds them into a caller, and they all
funnel through the exported [`parse_equation`](../../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl).
That parser calls `Meta.parse` on the caller string and then evaluates the numeric parts of the
resulting expression in `eval_numeric_functions` — so `eval_numeric_functions` *is* the library's
trust boundary for string input.

Its original design gated call heads against a `Set` of allowed symbols and then dispatched with
`Base.invokelatest(getfield(Base, fname), args...)`. Two problems followed from that shape:

1. **The capability was the whole `Base` namespace.** The `Set` gated the *name*, but the sink still
   resolved that name against all of `Base`. Correctness of the boundary rested entirely on the
   allowlist staying perfectly in sync with the sink — two lists that had to agree.
2. **The two lists had already drifted.** `:prior` was a member of the `Set` (it is a structural
   marker naming assets/groups, expanded later by `replace_group_by_assets`) but has no `Base`
   binding. A string like `prior(2)` — all-numeric args — reached `getfield(Base, :prior)` and threw
   a raw `UndefVarError` from deep inside `Base`, rather than a typed parse error at the boundary.
   The allowlist's own docstring also omitted `:prior`, so the documented set, the enforced set, and
   the callable set were three different things.

A related defect sat at the same sink (finding 3 of the 2026-07-06 security review): numeric
literals were combined in machine `Int64` *before* the `datatype` (float) conversion downstream, so
`2^64` silently wrapped to `0`, `10^19` produced a negative bound, and `2^-1` raised a raw
`DomainError` instead of a parse error.

## Decision

**Replace the name-lookup-in-`Base` sink with an explicit `Symbol => Function` table, make that table
the single source of truth for what may be called, and handle `:prior` structurally.**

- `allowed_functions` becomes a `const Dict{Symbol, Function}` mapping each of the 16 permitted math
  functions (`+ - * / ^ sqrt cbrt exp exp2 exp10 log log2 log10 abs min max`) directly to its
  function object. Membership and dispatch are now the same list — the set of callable functions
  cannot drift from the set of allowed names, because they are literally the keys and values of one
  `Dict`.
- `eval_numeric_functions` looks the head up with `get(allowed_functions, fname, nothing)` and throws
  a typed `Meta.ParseError` when it is absent. It can only ever call those 16 functions; `getfield`
  and `invokelatest` are gone (the functions are statically known Base functions, so no world-age
  dance is needed).
- `:prior` is deliberately **absent** from the table and branched first: it stays symbolic for later
  structural expansion, and all-numeric arguments (`prior(2)`) throw a typed `Meta.ParseError`
  ("`prior(...)` takes asset/group names, not numbers") instead of the old `UndefVarError`.
- Numeric arguments are coerced to `datatype` at the point an allowlisted function is *evaluated*
  (all args numeric), so arithmetic happens in the optimiser's float domain: `2^64` →
  `1.8446744073709552e19`, `2^-1` → `0.5`. Literals that survive inside an unevaluated (nonlinear)
  subexpression are left untouched, so `2^z` still renders as `2 ^ z`.

## Considered options

- **Keep `getfield(Base, fname)`, add a guard rejecting names with no binding.** Rejected: it patches
  the `:prior` symptom but leaves the capability equal to all of `Base` and keeps two lists to sync.
- **Keep the `Set` as the gate and add a parallel dispatch `Dict`.** Rejected: it removes the
  `getfield` sink but reintroduces exactly the two-lists-must-agree drift that produced the `:prior`
  bug; one `Dict` whose keys *are* the allowlist makes drift impossible by construction.
- **Coerce every numeric leaf to `datatype` at the recursion base case** (rather than at the call
  boundary). Rejected: it re-renders integer literals inside symbolic subexpressions (`2^z` →
  `2.0 ^ z`), changing user-visible constraint strings and breaking an existing round-trip test, for
  no correctness gain — overflow only ever occurs where a function is actually evaluated, which is
  exactly where call-boundary coercion applies.

## Consequences

- Attack surface at the parser trust boundary shrinks from the whole `Base` namespace to 16
  enumerated functions; anything else fails closed with a typed `Meta.ParseError`.
- The doc/const/callable drift that produced the `prior(2)` `UndefVarError` is eliminated — there is
  one list.
- Overflowing and negative-exponent literals become correct floats instead of silently-wrong bounds
  or raw `DomainError`s. Domain crashes for genuinely undefined math (e.g. `sqrt(-1)`) are left as
  raw errors — out of scope for this change.
- The same log-hygiene fix (finding 4) reworded the "variable not found" / "row dropped" warnings in
  both `get_linear_constraints` and the drifted copy in
  [Black-Litterman views generation](../../src/13_Prior/05_BlackLittermanViewsGeneration.jl) to name a
  variable/equation and an asset count rather than dumping the whole universe or `ParsingResult`
  struct into logs.
- **Internal only** — no public signature changes; `parse_equation`'s contract is unchanged except
  that previously-raw crashes are now typed `Meta.ParseError`s. Carried over from item S1 of the
  2026-07-02 combined review. Regression coverage in
  [test_02_equation_parsing.jl](../../test/test_02_equation_parsing.jl).
