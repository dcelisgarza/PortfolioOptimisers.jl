---
status: accepted
---

# Cap the equation parser's string length and AST depth to close a stack-exhaustion DoS

## Context

Constraint equations, Black-Litterman view strings and entropy-pooling view strings are untrusted
input: a config file, spreadsheet, or UI feeds them into a caller, and they all funnel through the
exported [`parse_equation`](../../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl). That
function calls `Meta.parse` on the caller string and then walks the resulting expression tree with
several recursive functions (`eval_numeric_functions`, `collect_terms!`, `has_invalid_plus`).

Finding 2 (Medium) of the 2026-07-06 security review (`docs/reports/security-review-20260706-143415.html`)
identified that nothing bounded that recursion. A deeply nested string — e.g. tens of thousands of
parentheses around a single variable — produces an AST deep enough that the recursive walk (or
`Meta.parse` itself) exhausts the stack and takes down the host process. The path is reachable on the
default constraint API with default settings; the blast radius is a **process kill** (denial of
service), the worst reach category, though it is not injection or a silent wrong result — the earlier
allowlist hardening ([ADR 0025](0025-enumerated-parser-allowlist.md)) already bounds *what* can be
called to 16 enumerated functions, so this is a resource bound, not a capability bound.

The `Expr` form of `parse_equation` (a caller passing a pre-built AST rather than a string) is a lower
trust boundary — the caller is program code, not config/UI — but it takes an unbounded tree that no
string-length cap covers, and the same recursive walks run on it.

## Decision

**Add two conservative, static, runtime-overridable resource caps at the parser trust boundary, and
fail closed with a typed `Meta.ParseError` when either is exceeded.**

- A global config `EQUATION_LIMITS` (a mutable struct holding `max_length = 4096` and
  `max_depth = 256`) with a `set_equation_limits!(; max_length, max_depth)` setter, living in
  [01_Base.jl](../../src/01_Base.jl) alongside — and mirroring — the existing `STRING_DISTANCE` /
  `set_string_distance!` and `COMPACT_SHOW` / `set_compact_show!` configuration idiom. Neither the
  const nor the setter is exported; both are used qualified, like `set_string_distance!`.
- **String form:** `parse_equation(eqn::AbstractString)` rejects any string longer than
  `max_length` characters *before* `Meta.parse` runs. Because achieving AST nesting depth `d` from a
  string requires at least `d` characters, the length cap also bounds the depth of the string form —
  one cheap check at the true trust boundary closes the reachable path.
- **`Expr` form:** `parse_equation(expr::Expr)` rejects any tree deeper than `max_depth` via a
  bounded helper `_expr_depth_exceeds`, which recurses at most `max_depth + 1` frames and
  short-circuits the instant the limit is breached — so the checker cannot exhaust the stack it
  protects.
- The defaults are conservative: a legitimate linear constraint is short and shallow, so `4096` and
  `256` sit far above any real constraint and far below the depth that threatens the stack. Callers
  generating genuinely large machine-authored constraint sets raise them with `set_equation_limits!`;
  callers wanting a tighter boundary lower them. Both must be positive.

## Considered options

- **Auto-detect the limit during precompilation from system characteristics** (the original
  proposal). Rejected. Julia precompilation runs on the machine that *builds* the pkgimage — a CI
  runner, a Docker build stage, a developer laptop — and bakes any computed constant into the image,
  which is then loaded on a *different* deployment machine; the cap would capture build-time state,
  not runtime state. Even on one machine there is no single "system stack" to read (worker-task
  stacks differ from the main thread's and are set by `JULIA_THREAD_STACK_SIZE`), and per-frame stack
  cost varies by Julia version and optimisation level, so "system characteristics" do not map to a
  safe recursion depth. The value we want is a cap *far below any legitimate constraint*, not the
  machine's survivable maximum — a static default expresses that directly and portably.
- **Auto-detect at `__init__` (runtime) instead of precompile.** Rejected. It fixes the build-vs-run
  machine mismatch but keeps the other two problems (no single stack size; version-dependent frame
  cost) and adds load-time complexity for no gain over a conservative static default that callers can
  override.
- **A single length cap, no depth cap.** Rejected. It closes the string trust boundary completely,
  but leaves the `Expr` form's recursive walks unbounded; the depth guard is cheap defence-in-depth
  for that path.

## Findings 1 and 3 will not be pursued

The same security review raised two sibling findings at adjacent boundaries. Both are deliberately
**left open** and should not be re-flagged by a future review:

- **Finding 1 — arity/type checking at the allowlisted call boundary** (`sqrt(1, 2)`, `min()` raise a
  raw `MethodError` rather than a typed `Meta.ParseError`). Not pursued: enforcing correct call shapes
  for the 16 allowlisted functions means encoding and maintaining a per-function arity/type contract
  that must track Julia's own method tables — a **large ongoing maintenance burden** for a boundary
  whose failure mode is already a fail-closed crash, not injection or a silent wrong result. The name
  allowlist ([ADR 0025](0025-enumerated-parser-allowlist.md)) already bounds *which* functions are
  reachable; a raw `MethodError` from one of those 16 is an acceptable, contained failure.
- **Finding 3 — index-literal type allowlist in `parse_lens`** (`a[1.5]` raises a raw `MethodError`
  rather than a typed parse error). Not pursued: `_eval_index` accepting only `Integer`/`Symbol`/
  `:vect` and letting other literal types fall through keeps the lens grammar open to whatever index
  types `Accessors.jl` may support, and these keys are developer-authored code, not config/UI input
  (a latent, not reachable, path — recorded in the [ADR 0025 amendment](0025-enumerated-parser-allowlist.md)).
  Enumerating the permitted literal types would **make the feature less flexible** for no reachable
  security gain.

## Consequences

- The one reachable process-kill (DoS) path the review flagged is closed: an over-long or over-deep
  untrusted string now fails closed with a typed `Meta.ParseError` at the boundary instead of
  exhausting the stack.
- New public-ish surface: `EQUATION_LIMITS` / `set_equation_limits!`, callable qualified like
  `set_string_distance!` (neither exported). No public signature of `parse_equation` changes;
  previously-crashing pathological input is now a typed parse error.
- No new dependency; the caps are pure static config.
- Findings 1 and 3 remain open by design (see above); this ADR records the rationale so future
  security reviews do not re-litigate them.
- Regression coverage in the "Equation parser recursion caps" testset of
  [test_02_equation_parsing.jl](../../test/test_02_equation_parsing.jl). Related hardening of the same
  boundary is in [ADR 0025](0025-enumerated-parser-allowlist.md) and
  [ADR 0026](0026-lenient-constraint-names-with-suggestions.md).
