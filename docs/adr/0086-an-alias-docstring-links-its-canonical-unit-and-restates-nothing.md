---
status: accepted
---

# An alias docstring links its canonical unit and restates nothing

## Context

[#436](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/436), a child of
[#478](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/478), opened on a gap. The
docstring Authority, `.github/instructions/julia-docstrings.instructions.md`, gives a section
structure for a type and a section structure for a function. It gives none for an alias. Read
literally, a factory alias such as `MAD` needs `# Arguments`, `# Returns` and `# Related`, and an
acronym alias such as `HRP` needs `# Related`. Nothing in the tree does either.

`.github/instructions/julia-source-code.instructions.md` § *Union Type Aliases and Dispatch Groups*
carried the only written statement: a union alias is documented "explaining which types it groups
and why". That is a rule about content, and it says nothing about sections. It also sat in a file
that is not the Authority for docstrings, which is the drift `STANDARDS.md` § *Changing a standard*
forbids.

### The measurement

The ticket counted three kinds of alias by a grep for `Union{` at column zero. A parse of every
file under `src/` finds a fourth shape and a wider third kind.

| Kind | Declaration | Units | Where |
| --- | --- | ---: | --- |
| Acronym alias | `const HRP = HierarchicalRiskParity` | 111 | `src/25_Aliases.jl` |
| Factory alias | `MAD(; kwargs...)::LowOrderMoment` | 21 | `src/25_Aliases.jl` |
| Dispatch alias | a `const` bound to a type expression | 249 | 57 files under `src/` |

The dispatch count is wider than the ticket's 183 because a grep for `Union{` cannot see two
shapes that a caller meets identically. `const VecLc = AbstractVector{<:LinearConstraint}` is a
container, not a union. `const RMCVaR{T} = Union{...}` is parametrised, so its declaration does not
start `const RMCVaR = Union{`. Both are the type a method signature dispatches on, which is the
only thing that distinguishes the kind, so both are the same kind.

The convention the tree already follows is near-uniform, and it contradicts the literal reading of
the Authority.

- 111 of 111 acronym aliases carry a signature line and one sentence. **None carries a section.**
- 21 of 21 factory aliases carry a signature line and a sentence naming the types they compose.
  One carried `# Related`, and none carried `# Arguments`, `# Returns` or `# Examples`.
- 249 of 249 dispatch aliases are documented. 226 carry `# Related`, three carry `# References`,
  and **none carries any other section**.

Three defects were found while measuring, and all three are fixed in the commit that records this
decision. `GAO` indented its one sentence into its signature block, so Documenter rendered the
sentence as code. `ZeroVarianceFilter` carried a `# Related` block whose three bullets its own
first sentence already links, and it raises a `DomainError` with no `# Validation` section.
`Prices_RR` carried neither a `const` header line nor `# Related`, in a file marked `swept = true`.

### The tension

An alias points at a canonical unit whose docstring already carries every section. Repeating those
sections on the alias makes a second copy that drifts, which is the argument that put the field
descriptions into `field_dict`. Against that, a reader who reaches `MAD` through autocomplete sees
only the alias, and a docstring that states nothing sends that reader away.

## Decision

**An alias docstring links its canonical unit and restates nothing.** The reader gets one click,
not a copy. The Authority is `.github/instructions/julia-docstrings.instructions.md` §
*Section Structure for Aliases*, and `STANDARDS.md` routes an alias docstring to it.

**1. Three kinds, and the sections each carries.**

| Kind | Lives in | Header line | Sections |
| --- | --- | --- | --- |
| Acronym alias | `src/25_Aliases.jl` only | the alias name alone | none |
| Factory alias | `src/25_Aliases.jl` only | the signature, ending `-> T` | `# Validation`, and only when its own body raises |
| Dispatch alias | any file under `src/` | the declaration | `# Related`, and `# References` when the grouping itself is published |

A dispatch alias is a `const` bound to a type expression rather than to a bare name. A union, a
container and a parametrised form are one kind.

**2. No alias carries `# Fields`, `# Constructors`, `# Arguments`, `# Returns` or `# Examples`.**
The canonical unit owns each of them, and the summary sentence puts it one click away. A copy on
the alias is a second text to maintain, and it is the one a reader reaches first when it goes
stale.

**3. An acronym alias and a factory alias carry no `# Related`.** Their summary sentence already
`@ref`s every canonical name, and the Authority's own `## What each section holds` states that
`# Related` does not hold a copy of the related unit's own text. One link, once. A dispatch alias
does carry it, because a group of members is a list and a summary sentence must not be one.

**4. A factory alias may carry a second summary sentence**, and only for a choice the composition
fixes that a reader would otherwise get wrong. That sentence is about the unit as a whole, so the
summary paragraph owns it under `## What each section holds`, exactly as
[ADR 0085](0085-the-docstring-standard-is-rules-and-pointers.md) settled for a fact that
`# Details` used to hold. `ZeroVarianceFilter` is the worked case: it states why it scores with
`SCM()` and not with `Variance`.

**5. `# Algorithm`, `# Mathematical definition` and `# JuMP formulation` reach no alias.** An alias
runs no steps and registers no row.

**6. The Gate is three checks in `test/test_26_docs.jl`.**

 1. **Library-wide, absolute.** No alias carries a section outside its kind's row. Every kind sits
    at its allowance after the three fixes above, so the check carries no allow-list and a new
    breach is the only way it can red.
 2. **A swept file, presence.** A dispatch alias in a file marked `swept = true` in
    `sweep/manifest.toml` carries `# Related`.
 3. **Library-wide, a ratchet.** The count of dispatch aliases carrying no `# Related` may not
    rise. It stands at 22 and the check retires at zero.

Check 2 reads the `swept` flag and check 1 does not, which is the split
[ADR 0081](0081-the-docstring-standard-states-the-model-it-builds.md) drew and
[ADR 0085](0085-the-docstring-standard-is-rules-and-pointers.md) reused. A check that DEMANDS a
section may not red a file that no child map of #404 has swept. A check that FORBIDS one may,
because a file passes it by changing nothing.

**7. The `## Reference docstrings` table gains a row.** `RhoDistanceAlgorithm` in
`src/09_Distance/02_Distance.jl` is the dispatch-alias reference, and its file is swept, so a Gate
holds the pointer.

## Consequences

The 22 dispatch aliases that carry no `# Related` are a debt the ratchet bounds, and the prose
ticket of each file under #404 pays its own share. 19 of the 22 sit in four files under
`src/20_Optimisation/02_CrossValidation/` and `src/19_RiskMeasures/`, so the debt is concentrated
rather than spread.

Nine dispatch aliases carry no header line at all, and 33 write a header in one of four shapes.
This decision states the shape and does not gate it, because the units are unswept and a presence
demand on an unswept file is what the split in check 6 refuses. The prose sweep of each file
corrects its own.

Two tickets of child map 1 waited on this decision: the documentation of `src/01_Base.jl`, which
holds 26 dispatch aliases, and the sweep of `src/25_Aliases.jl`, which is 132 alias units. Both are
unblocked.
