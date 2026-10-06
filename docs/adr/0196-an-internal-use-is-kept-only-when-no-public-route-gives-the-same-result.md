---
status: accepted
---

# An internal use is kept only when no public route gives the same result

## Context

The library reaches into the internals of Julia and of its dependencies at about 200 sites in
`src/` and `ext/`: a qualified access to a name that its owner does not make public, a read of a
field of a Julia object such as `typeof(x).name.wrapper`, and a read of a field of a dependency
type. Such a use breaks without notice when the owner changes the internal, and no gate found one.
The maintainer asked to rely on internals only where it is necessary.

[#1499](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1499) made the inventory on
Julia 1.13.1. It found that `Base.ispublic` alone is a wrong measure: MOI, Optim, Roots and
StatsBase document names in their manuals that they never declare `public`, and 88 of the 136
qualified accesses were such names.
[#1500](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1500) decided the rule below
and gave each use a verdict, after a check of the effect of each change on performance,
architecture, maintainability and ergonomics.

## Decision

**A name or a field is public when its owner says so.**

1. Julia says so with `export` or `public`, so a name of `Base` or `Core` is public when
   `Base.ispublic` is `true` for it. A docstring alone does not make it public. A field of a Julia
   object, such as `.name`, `.wrapper`, `.parameters` or `.a`, is never public.
2. A dependency says so with `export`, with `public`, or with its manual. A name that the manual
   documents is public, and `Base.ispublic` cannot see this, so the census holds each such name
   with a pointer to the manual page.
3. A field of a dependency type is public only when the manual documents the field and the owner
   gives no accessor for it. When an accessor exists, the code calls the accessor.

**An internal use** is a call of, a method added to, or a read of a name or a field that is not
public. A read by `ext/` of a name of PortfolioOptimisers is not an internal use, because the
package owns both sides. `test/`, `docs/` and `code_health/` are out of scope: they inspect the
library.

**Each internal use takes one of two verdicts, Keep or Replace.**

A **Keep** needs one of five reasons:

| Reason | What the census entry records |
| --- | --- |
| No public route gives the same result. | The Upstream mark and the draft request. |
| The use extends a function of the owner through an internal hook. | The Upstream mark and the draft request. |
| The public route needs a new direct dependency. | The name of the dependency. |
| The public route has a cost in a hot path. | A benchmark in `code_health/` that measures both routes. |
| A supported Julia version lacks the public route. | The first version that has it. |

More own code is never a reason for a Keep. A short formula, a loop, a dispatch, or one method for
each type is a public route. But **a port of an algorithm of the owner is not a public route**: the
package would own a copy of the owner's defects, and the owner's fixes would not reach it. A use
whose only route is a port is a Keep with the reason "no public route".

The **Upstream mark** is not a third verdict. It marks a Keep with the reason "no public route" or
"extension", for any owner, Julia included. A request for a public route is drafted on a ticket in
the maintainer's voice, the maintainer posts it, and the census entry records the upstream issue.
The entry keeps the mark until the owner answers. A refusal makes it a plain Keep.

A **Replace** keeps the qualification of a name. A qualified name such as `DataFrames.nrow`
disambiguates the name, so a Replace changes the route, not the spelling of the module.

**A generic rebuild over a family of types calls `Accessors.setproperties`.** Accessors exports it,
Accessors is a direct dependency, and it is the documented extension point of ConstructionBase. It
gave the same result as `typeof(x).name.wrapper` on 161 estimators and 54 risk measures, at the
same cost, and a type that a user writes works with no extra code. A type with forwarded properties
gets its `setproperties` method from `@forward_properties`, which calls the keyword constructor by
name. The default method of ConstructionBase reads `.name.wrapper` itself, but that read is an
internal use of the owner, not of this package.

A Replace goes to the sweep sub-issue of #404 that owns the file, reopened and scoped to this rule.
A census in `code_health/` holds each Keep with its reason, so a new internal use fails the build.

## Considered options

- **`Base.ispublic` alone.** Rejected: it flags the documented API of MOI, Optim, Roots and
  StatsBase, and forces a Keep or an Upstream request where the owner already gives the route.
- **The documented manual for Julia too.** Rejected: Julia defines public itself, since 1.11, so
  its own definition applies. Only `Base.@__doc__` differs, and it is a Keep.
- **One method for each type, emitted by the declaring macro, for a rebuild.** Rejected after the
  impact check: `@propagatable` declares 95 of the 256 concrete estimators and no optimisation
  result, and `rebuild_estimator` runs on every `AbstractEstimator`, which users subtype through 51
  `# Interfaces` sections. A user type would get a `MethodError` from a function that is not
  public.
- **A port where no public route exists.** Rejected, for the reason above.

## Consequences

- The per-use verdicts are on #1500. At the time of the decision, the Keeps are `Base.@__doc__`,
  `Base.uniontypes`, the hint path of `Base.kwarg_decl` (Julia); `StatsBase.varcorrection`;
  `Distances._pairwise!` (an extension); the branch ordering of Clustering and `HclustMerges`; and
  `FLoops.Transducers.Executor` (a new dependency). The other uses are Replace, or they are not
  internal.
- What is public moves with the Julia version and with each release of a dependency, so the census
  states the version it measures, and an entry that becomes public is dropped.
