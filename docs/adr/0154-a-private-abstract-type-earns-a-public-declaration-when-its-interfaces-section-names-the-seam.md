---
status: accepted
---

# A private abstract type earns a `public` declaration when its `# Interfaces` section names the seam

## Context

[Issue #1120](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1120), on
[the delineation map](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/553), asked
which of the names now documented under `docs/src/private_api/` earn a `public` declaration, and
what rule decides it. The migration that map ran (#1102–#1118) classifies a name by
`Base.isexported`/`Base.ispublic` alone — [ADR 0128](0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md)
fixed that classifier, not what the source declares. Nothing before this decision says when a name
that is neither exported nor `public`-declared should become one.

Four candidate rules were on the table: promoting the abstract types a user is meant to subtype
(and the verb they implement for it); promoting a name a public docstring links with `@ref`;
promoting any name a user guide or example calls qualified (`PortfolioOptimisers.foo`); or
promoting nothing until the maintainer asks for it by name, the precedent set when
[#1110](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1110) and
[#1113](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1113) declined to widen the
foreign-owned class for `Base.copy` and `Base.vcat` on the same ground. A search of
`docs/src/public_api/` found no page linking `@ref` into `docs/src/private_api/`, so the `@ref`
route promotes nothing today.

**The seam rule does not need a new marker — the library already carries one.** A `# Interfaces`
docstring section, shipped 2026-09-15, already names the method(s) a new subtype of an abstract
type must implement, and `test/test_61_interfaces_section_census.jl` already requires the section
on every abstract type that is the direct supertype of a concrete type the package defines, or
keeps the type on a shrinking debt list in that file. `AbstractSelectionRule`'s docstring, for
example, already reads:

```julia
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that turn per-asset scores into a keep-mask.
...
# Interfaces

In order to implement a new selection rule that works seamlessly with the library, subtype
`AbstractSelectionRule` with all necessary parameters as part of the struct, and implement the
following method:

  - `rule_keep(rule::AbstractSelectionRule, scores::VecNum, bib::Bool) -> BitVector`: Turn the
    per-asset scores into a keep-mask.
...
"""
abstract type AbstractSelectionRule <: AbstractAlgorithm end
```

That is already the extension seam this decision needs a signal for — a first pass that measured
only 6 hand-picked candidates against an invented `# Extend` marker missed it, because it read only
the two lines immediately before each `abstract type` statement rather than the docstring's own
`# Interfaces` heading further up. Measured properly: **95 abstract types across `src/` and `ext/`
already carry a `# Interfaces` section, and 94 of them are still neither exported nor
`public`-declared.** They span every top-level directory that has an abstract-type family:
`01_Base/`, `03_InputData/`, `04_MatrixProcessing/`, `05_Moments/` (28 types, the largest group),
`08_Phylogeny/`, `09_ConstraintGeneration/`, `10_Prior/`, `11_UncertaintySets/`,
`16_RiskMeasures/`, `17_Optimisation/` (17 types), `20_AssetSelection.jl`, plus the standalone
files `02_Tools.jl`, `14_NetReturnsDrawdowns.jl` and `15_Tracking.jl`.

## Decision

### A `# Interfaces` section is the seam signal; a name a guide calls qualified is not

An abstract type is a genuine extension point — one a caller is meant to subtype — exactly when
its docstring carries a `# Interfaces` section. **Documenting the contract already is the public
promise.** A type whose docstring commits to "implement this method to extend the library" and is
then left undiscoverable in `public_api/` is the inconsistency this decision closes, not a reason
to hold back. The type, and every verb its `# Interfaces` section names under a `` ## `verb` ``
heading or a call-shaped bullet, carry a `public` declaration in source.

This is the same doctrine [ADR 0128](0128-the-docs-split-into-mirrored-public-and-private-api-trees-and-a-census-gates-placement.md)
already applies to the docs split — **the source declares, nothing is inferred** — extended one
step earlier, to what makes a name public in the first place, not only to how the docs mirror that
fact. Reusing the existing `# Interfaces` section rather than inventing a parallel one means the
signal is already written, already reviewed at the point the interface was designed, and already
gated by `test_61` — nothing new to keep in sync.

A guide or example calling a private name qualified (`PortfolioOptimisers.foo`) does not, by
itself, promote it — that is a symptom of a name being a seam, not the cause, and every qualified
seam reference checked while grilling this ticket already resolves to a type or verb the
`# Interfaces` rule promotes on its own. `@ref`-linking a private name from a public docstring does
not promote it either: a public page may still point a reader at private machinery worth naming
without committing to its stability.

### The rule does not, by itself, promote anything today

This decision fixes the rule; it does not run it. None of the 94 still-private `# Interfaces`-
marked abstract types, nor their named verbs, gain a `public` declaration in this ticket. Applying
the rule is a shared infrastructure ticket — the census below, plus the STANDARDS.md row — followed
by one promotion ticket per top-level `src/`/`ext/` directory, mirroring the shape the mirror-tree
migration itself used ([#1101](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1101)
→ #1102–#1118): each directory ticket promotes every `# Interfaces`-marked type it holds and moves
the entries the promotion affects between the mirror-tree pages, confirming with the maintainer any
type that turns out, on inspection, not to be a genuine outside seam.

### A census gates the two declarations against each other

An abstract type carrying a `# Interfaces` section without a matching `public`/`export`
declaration — on the type or on a verb the section names — is a defect from the day this decision
lands; so is a `public` abstract type whose docstring documents no `# Interfaces` contract. A new
census file, parallel to `test/test_43_exported_abstract_type_census.jl`, checks both directions
for every abstract type in `src/` and `ext/`, reusing `test_61`'s existing section parser rather
than writing a second one. Its exact shape is the infrastructure ticket's job to design; this
decision only fixes that the gate must exist.

## Consequences

- `STANDARDS.md` gains a row: subject = a private name's promotion to `public`, authority = this
  ADR, gate = the new census file, not yet created.
- The infrastructure ticket owes the census and the STANDARDS.md row it gates. Each directory
  ticket it unblocks owes, per abstract type it promotes: the `public` declaration on the type and
  every verb its `# Interfaces` section names, and the mirror-tree page updates the promotion moves
  entries on.
- `@ref`-linking a private name from a public docstring, and calling a private name qualified from
  a guide or example, both remain unrestricted and carry no promotion obligation on their own — the
  `# Interfaces` section is the only signal that does.
- This decision leaves ADR 0128's foreign-owned class untouched: `Base.copy`, `Base.vcat`, and
  every other foreign-owned method that ADR already classifies stay governed by it. Widening that
  class is still a separate, one-at-a-time maintainer call.
