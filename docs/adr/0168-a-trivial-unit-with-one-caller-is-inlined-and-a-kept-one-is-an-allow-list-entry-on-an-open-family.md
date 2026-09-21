---
status: proposed
---

# A trivial unit with one caller is inlined, and a kept one is an allow-list entry on an Open Family

## Context

[Issue #1208](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1208) names three
units as needless indirection: `diagonal_geometry`, `DiagonalProjection` and `resolve_turnover`.
It asks for a rule that separates a wrapper that earns its place, an interface a later capability
extends through, from one that only adds a name between a caller and the expression it calls.
And it asks for a census of the rest of the library under that rule.

Six facts measured on `dev` at `d96073d819` shaped the decision.

1. **`diagonal_geometry` is one method with one caller and a one-expression body.** The caller
   is `online_update!` of `AdaptiveSubgradient`
   (`10_OnlinePortfolioSelection/09_AdaptiveOptimistic.jl:292`), and the body is
   `DiagonalProjection(; h = h)`. Its sibling `gram_geometry` is not the same shape: it has a
   pass-through method on `AbstractProjectionGeometry` and a bind method on `GramProjection`, and
   its caller's slot, `NewtonStep.proj`, admits `Union{<:EuclideanProjection, <:GramProjection}`.
   The dispatch does work at that call site today.
2. **`AdaptiveSubgradient.proj` is dead configuration.** Its bound is `T3 <: DiagonalProjection`,
   its default is `DiagonalProjection()`, and the rule overwrites the geometry's `h` at every
   step. A caller's `h` never reaches a projection. The `h = nothing` state of
   `DiagonalProjection` exists to fill that slot before the first step, and
   `assert_diagonal_bound` exists at three sites to refuse that state. The type is exported, on
   the public API page and in the capability catalogue, and none of it is on `origin/main`
   (`63286db307`), so no release carries it.
3. **`DiagonalProjection` itself is live dispatch.** `project`, `set_projection_objective!`,
   `projection_solver` and `assert_geometry_admits_set` select their arms on the geometry, next
   to `EuclideanProjection`, `EntropicProjection`, `TsallisProjection`, `LogBarrierProjection`
   and `GramProjection`. On a `BoundedAllocationSet` the diagonal projection is the closed form
   `bounded_quadratic_projection(q, set.wb, h)`; on a `ProgrammeAllocationSet` it is a
   second-order cone objective that `projection_programme` writes through
   `set_projection_objective!`. Passing `h` alone through that machinery means a bare vector in
   the geometry slot, or a branch on the set type inside the rule.
4. **`resolve_turnover` is a `Nothing`/`Number` pair with one caller.** The caller is
   `summarise_returns` (`18_ExpectedReturns.jl:1593`), and `isnothing(turnover) ?
   oftype(ann_ret, NaN) : turnover` is the same expression. The one property the pair can hold
   over the ternary is concreteness of the caller's inferred return type.
5. **The library defines 2004 function names in `src/` and `ext/`.** A grep heuristic finds 580
   with one reference and 226 with none, outside their own definitions and docstring signature
   lines. The heuristic undercounts a bare indented call and a reference in value position, so it
   is a candidate list and not a census; the census is the build's, with a parser.
6. **An `# Interfaces` section is already mandatory, so it discriminates nothing on its own.**
   `test/test_61_interfaces_section_census.jl` requires the section on every abstract type that
   is the direct supertype of a concrete type, or a debt-list entry, and
   [ADR 0154](0154-a-private-abstract-type-earns-a-public-declaration-when-its-interfaces-section-names-the-seam.md)
   makes the section the only route to a `public` declaration. A rule that read the section as
   an automatic exemption would exempt nearly every family.

## Decision

### The rule

A unit is a dead-end, and is removed, when all three hold.

1. It has **one method**. Two methods are dispatch: a `Nothing`/value pair and a two-family pair
   save their caller an `isa` branch and keep the caller's inferred type concrete, so they are
   never candidates, whatever their bodies hold. `gram_geometry` on `NewtonStep.proj` is the
   shape, and so is `carrier_asset_names` over a `Nothing` and a returns carrier.
2. Its body is **one depth-one forward**: a bare argument, a literal, a field of an argument, or
   one call (or constructor) whose callee is a name and whose arguments are each of those, a
   global name, a splat of one or a keyword forward. `diagonal_geometry`, whose body is
   `DiagonalProjection(; h = h)`, is the shape. A decision, an operator, an index, a nested
   call, a comprehension or a string template is logic the unit owns: a ternary that picks a
   formulation, a message template, an accessor that slices and a helper that names one rule
   all reduce their caller's complexity, and every one of them stays.
3. It has **at most one reference** in `src/` and `ext/` outside its own definition. A
   reference in value position, such as `map(f, xs)`, counts as a call does. A call from `test/`,
   `docs/`, `examples/` or `user_guide/` is not a use: the direct tests of an inlined unit move
   to its caller, and a unit with no reference in `src/` or `ext/` at all is dead.

A unit that offers nothing but indirection is the whole of the target. Anything that reduces
complexity, keeps a caller's inferred type concrete, or is simple and convenient earns its keep,
and the complexity gate forces most named steps into existence in the first place.

The type-level case is an abstract type with one concrete subtype that **nothing codes against**:
no method dispatches on it, no field is bound to it, and no union alias names it, so its only
reference is the subtype's own declaration. Such a type stays when it has a realistic future
application, a second member the family is shaped for, and the allow-list is where that
application is named. A type any code reads through is an abstraction the code is written
against, and is never a candidate.

### The brake

The census flags every candidate, and a name is kept only by a hand-written entry in the gate's
allow-list, each with a one-line reason: for a function, the reason it is simple and convenient;
for an abstract type, the future application or the `# Interfaces` section that already states
its contract. The entry is the whole of the brake, and the reviewer of that edit is the
maintainer. An entry whose name is no longer flagged is stale and fails the gate, so the list can
only shrink on its own.

A public forward is public API, and a user's call is a use the census cannot see. It is flagged
all the same and kept by an allow-list entry, so the list shows every public forward the library
carries: the descriptor factories, the risk-measure aliases and the out-of-place arm of
`matrix_processing_algorithm` are the shapes.

### The gate

`test/test_70_trivial_unit_census.jl` parses `src/` and `ext/` with the parser
`code_health/CodeHealth.jl` holds, which is JuliaSyntax, and resolves every name against the
loaded module. It lists every one-method function whose body is a depth-one forward and whose
name has at most one reference, and every abstract type with one concrete subtype and no
reference outside the declarations, and fails on any name outside its allow-list and on any
allow-list entry that is no longer flagged. Two shapes are not candidates: a constructor method
named after its type, the library's keyword-constructor idiom, and a method that extends another
module's function, such as `Base.show`, which is an interface by definition.

### The three named units

- `diagonal_geometry` is deleted. The call site constructs `DiagonalProjection(h)`.
- `DiagonalProjection` stays as the geometry token of fact 3 and becomes private. `h` is a
  required positive vector, so the `h = nothing` state and `assert_diagonal_bound` go, and
  `AdaptiveSubgradient` loses its `proj` field. The export, the public API entry and the
  catalogue entry go with it. `GramProjection` keeps its `A = nothing` state, because
  `NewtonStep.proj` admits two geometries and the user configures the Gram solver on it.
- `resolve_turnover` is replaced by the ternary at its one call site, if `Base.return_types`
  shows `summarise_returns` concretely typed for both a `Nothing` and a `Number` turnover. If the
  ternary loses concreteness, the pair stays and the reason is recorded on the build ticket.

### Delivery

One build ticket, one branch, commits per directory, so no parallel session contends on the
sweep manifest or the API pages. The rule is written in
`.github/instructions/julia-source-code.instructions.md` and routed in `STANDARDS.md` to the
gate.

## Considered options

1. **Every single-caller unit is a candidate, whatever its body** — refused. It makes hundreds
   of named steps defend themselves, and the complexity gate put most of them there.
2. **No exception at all, `gram_geometry` included** — refused. Inlining live dispatch as a
   branch at the call site is the shape the library dispatches to avoid.
3. **The `# Interfaces` section as an automatic exemption** — refused by fact 6.
4. **A named future subtype in an open ticket as the mark of an open family** — refused. A
   ticket is not a property of the code, and the gate cannot read it.
5. **A bare `AbstractVector` in the geometry slot**, so `h` is passed on with no struct —
   refused. It loosens `projection_programme` and `set_projection_objective!` to a type that is
   not a geometry.
6. **A branch on the set type inside `online_update!`** — refused. It moves the set-branch out of
   `project` for one rule.
7. **Exported forwarders exempt from the gate** — refused. The allow-list should show every
   public forwarder, so a later API decision reads them off one place.
8. **A count ratchet instead of an allow-list** — refused. A number names nothing, and a kept
   unit needs a name so its justification can be checked.
9. **A map with a sub-issue per directory, or folding the census into the sweep** — refused for
   this pass. The removals share three files, and one branch pays no merge contention.
10. **Every one-expression unit as a candidate, a ternary and a `Nothing`/value pair included** —
    the rule as first drafted, refused during the build. The wide reading flagged the
    formulation pickers of the entropy-pooling tail views, the per-term projection of a
    re-based constraint row, the group view of a tensor Panel Field and every `Nothing`/value
    accessor pair, each of which reduces its caller's complexity or keeps its inferred type
    concrete. Only a unit that offers nothing but indirection is a dead-end, and the rule above
    says so.
11. **Every single-subtype abstract type removed, its subtype taking the supertype** — refused
    during the build for the same reason. A type nothing codes against and with no realistic
    second member is the only dead-end; a family shaped for a second member stays, and names it
    on the allow-list.

## Consequences

- `CONTEXT.md`: **Open Family** is written. The *Projection Geometry* entry no longer says every
  rule holds its geometry on a `proj` slot, because `AdaptiveSubgradient` builds its own at the
  step.
- The build owes: the census test and its allow-list; the deletion of `diagonal_geometry` and
  `assert_diagonal_bound`; the private `DiagonalProjection(h)` with `h` required, its export,
  public API entry and catalogue entry removed and a private API entry added; the `proj` field of
  `AdaptiveSubgradient` removed from the struct, the constructor, the docstring and the test that
  reads it; the `return_types` measurement of `summarise_returns` and, on a pass, the ternary;
  the census over the whole library and the removal of every dead-end it flags, with each
  touched file's sweep manifest row and private API page corrected; the rule in the
  instructions file and its `STANDARDS.md` row. Under the rule above the census found one
  dead-end beyond the three named units, the no-op `set_budget_costs!`, and no abstract type.
- Built by [#1212](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1212).
- No released number moves. `AdaptiveSubgradient` and `DiagonalProjection` are on `dev` alone,
  and the removal of the `proj` keyword is a clean break with no deprecation, as ADR 0167 made
  for `ff`.
