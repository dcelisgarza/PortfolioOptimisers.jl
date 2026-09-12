---
status: accepted
---

# The docs split into mirrored public and private API trees, and a census gates placement

## Context

[Issue #553](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/553) measured
`docs/src/api/`: 194 pages mirroring `src/`, 2824 `@docs` entries, and 62 % of those entries
neither `export`ed nor `public`-declared. 143 pages mix an exported or `public` name with an
internal one on the same page, and a reader cannot tell the 35 internal-only pages from the 14
public-only ones without reading every entry. `docs/make.jl` already computes the three sets —
`exported_symbols`, `public_symbols`, `private_symbols` — from `Base.isexported` and
`Base.ispublic`, and nothing downstream reads the distinction.

**The source declares, and the docs derive.** `export` and `public` in `src/` and `ext/` are the
single source of truth for what is public API; the docs must never state a boundary the source does
not declare. Before this decision, nothing enforced that: a page could put a name on the wrong
side and no check would fail.

[Issue #557](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/557) prototyped four
same-page candidates against the charting-time constraint that the docs keep mirroring `src` and
that the delineation not become a split into two doc sets: two headed blocks on one page, a
per-entry admonition, a collapsed disclosure, and a sibling page. It recommended one page per
source file with headed sections, plus a badge stamped into the built HTML from
`Base.isexported`/`Base.ispublic` — the only candidate that cannot go stale, because it reads the
declaration at build time rather than being hand-split.

[Issue #558](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/558) reversed both the
recommendation and the constraint that shaped it: several API pages already trip the size warning,
and the maintainer judged a hard directory boundary easier to read and use than a same-page marker.

[Issue #559](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/559) then had to decide
what a directory-boundary design still leaves open: how a name that extends a `Base` or `StatsAPI`
generic — one the package does not own — classifies, and what proves a name landed on the correct
side once the split exists.

## Decision

### Two mirrored top-level trees replace `docs/src/api/` outright

`docs/src/public_api/` and `docs/src/private_api/` reproduce the `docs/src/api/` directory
structure file-for-file, and together they replace it — one boundary to maintain, not three. Each
pair of mirror pages is generated programmatically from one classification pass over the same
source file's entries: a name lands on exactly one of the two pages, never both.

Within a page, entries are ordered **abstract types/consts → concrete types → functions**,
replacing the source order the old `docs/src/api/` pages used. A source file with nothing on one
side still gets that side's page, holding a one-line note that the file has no public (or no
internal) names — there is no more "stays in place" case, because every file maps to both trees.
Page-level prose between `@docs` blocks (headers, explanations) carries no blanket placement rule;
whoever migrates a given page decides, per page, which mirror it belongs on.

The per-file migration itself is not part of this decision. It is chartable — two mirror pages, the
ordering rule, `make.jl` walking two roots — only once the classifier and the gate below exist, and
it is tracked as fog on [the map](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/553)
rather than as a ticket here.

### The classifier: the source declares; a foreign-owned binding is always public

For a name `PortfolioOptimisers` itself owns, the classifier is `Base.isexported` /
`Base.ispublic`, unchanged: exported → public, `public`-declared → public, neither → internal.

A name naming a method the package adds to a function it does not own — extending a `Base` or
`StatsAPI` generic on a library type, for example `Base.iterate`, `Base.getproperty`,
`Base.getindex`, `Base.propertynames`, `Base.showerror`, `Base.split`, `StatsAPI.fit`,
`StatsAPI.predict` — classifies **always public**, unconditionally. Extending a foreign generic on
a library type is, by construction, part of that type's public interface: a caller is meant to
`split(x)`, iterate over `x`, do `x.field`, `fit(x)`. This is a fourth case layered on the existing
two, not a gap in them, and it needs no allow-list: the criterion is that the function is
foreign-owned, not that the name appears on a list. A page may spell the name qualified
(`StatsAPI.fit`) or bare (`fit`); both classify identically, so the spelling used on a page carries
no separate rule.

### The gate: placement correctness only, absolute, over migrated pages

A new census file, parallel to `test/test_43_exported_abstract_type_census.jl`, parses the
`` ```@docs ``` `` fenced blocks of `docs/src/public_api/**` and `docs/src/private_api/**` directly
from their `.md` source — no docs build. For every entry it finds, the entry's classification
(by the rule above) must match the tree it was parsed from. A bare name and its qualified form
resolve to the same binding before classification, per the classifier rule.

That is the entire assertion. The gate does **not** check completeness — that every exported and
`public` name is documented somewhere — because
[issue #554](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/554) already proved it
holds by direct measurement, and completeness is not a per-page migration risk the way placement
is. It does not check for an empty section either; the one-line note on a page with nothing on one
side is a page-authoring convention, not something a census enforces.

**The gate is absolute, not a ratchet, and it binds only over pages already migrated.** A page
enters `public_api/` or `private_api/` only when someone deliberately migrates it into the mirror
tree, so there is no legacy debt to grandfather the way `test/test_26_docs.jl`'s docstring ratchets
do. Any `@docs` entry inside either tree must be on the correct side from the day the page is
migrated. How many source files still sit under a pre-migration path is the migration ticket's
concern, not this gate's.

## Consequences

- `docs/src/api/` is retired once migration completes; `docs/src/public_api/` and
  `docs/src/private_api/` are the only API tree from then on. Until migration finishes, only the
  pages already moved are gated — the rest carry no placement guarantee yet.
- `STANDARDS.md` gains a row: subject = an `@docs` entry's mirror-tree placement, authority = this
  ADR, gate = the new census file. `test/test_46_standards_citation_census.jl` requires that row to
  name a file that resolves.
- `make.jl`'s API-page discovery must walk two roots and build two top-level navigation groups,
  `Public API` and `Private API`, instead of the single `walkdir(docs/src/api)` it uses today. That
  edit belongs to the migration work this ADR unblocks, not to this decision.
- The four same-page candidates [issue #557](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/557)
  built — two headings, a per-entry marker, a collapsed disclosure, a sibling page — and its
  recommended badge-plus-headings combination are superseded. None of that prototype work is lost:
  the badge mechanism (reading a name's classification from `Base.isexported`/`Base.ispublic` at
  build time) is exactly the classification pass the mirror trees are generated from.
- The 21-entry foreign-owned class this ADR classifies always-public was untested by #557's two
  prototype pages, because neither carried one; the rule here is the first place it is decided.
