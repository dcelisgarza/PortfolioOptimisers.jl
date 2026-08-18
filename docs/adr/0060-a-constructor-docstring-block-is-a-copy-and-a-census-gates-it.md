---
status: accepted
---

# A constructor docstring block is a copy, and a census gates it

## Context

`field_dict` and `arg_dict` in `src/01_Base.jl` centralise the *prose* of a field. The keyword
*signature* of a constructor is not centralised. Each type restates it by hand inside its own
`# Constructors` docstring block:

```julia
# Constructors

    SubsetResampling(;
        pe::TD{<:PrE_Pr} = EmpiricalPrior(),
        subset_size::TD{<:SubsetSizeE} = 0.8,
        n_subsets::TD{<:NumberSubsetsE} = 2,
        max_comb::Integer = 1_000_000_000,
        …
    ) -> SubsetResampling
```

The block is a copy. Nothing tied the copy to the definition, so the two drifted, and the
drift was invisible to a reader of either side alone. A census of `src/` on 2026-08-17 found
321 documented keyword signatures over 197 files, and **20 of them disagreed with the code**:

- Seven carried a stale default. `SubsetResampling` advertised `subset_size = 0.5`,
  `n_subsets = 100` and `max_comb = 1000` against a definition that has held `0.8`, `2` and
  `1_000_000_000` since `67c236315`. `HierarchicalRiskMeasureSettings` documented `scale` as
  mandatory while its own jldoctest called `HierarchicalRiskMeasureSettings()` and printed
  `scale ┴ Float64: 1.0`.
- Thirteen more carried a stale type annotation, a stale keyword order, or had lost a keyword
  entirely. `MeanRiskResult` documented two of its three keywords. `JuMPOptimiser` documented
  neither `sca` nor the vector half of `plr`.

One block did not even parse: `OptimEntropyPooling`'s last keyword had lost its trailing comma.

## Decision

**The block stays hand-written, and a test makes the drift loud.**

`test/test_41_constructor_docstring_drift.jl` parses every `# Constructors` block and every
keyword-taking definition out of the *source text* of `src/**/*.jl`, normalises both, and
requires each documented signature to equal one real definition of that name.

Generation was the stronger candidate and is blocked.
`DocStringExtensions.TYPEDSIGNATURES`, used 578 times in `src/`, renders a signature **without
its default values** — and the defaults are the half that drifts. A generator that drops them
replaces a stale block with a less useful one.

The census reads text and reflects on nothing, so it loads no package and costs about four
seconds. It names no type, so a type added tomorrow is covered the day its docstring is written.

### What it normalises

A census that fails on noise is a census that gets deleted. Four differences are cosmetic and
are absorbed:

| Difference             | Documented          | Code           | Absorbed by                    |
| ---------------------- | ------------------- | -------------- | ------------------------------ |
| module qualification   | `ThreadedEx()`      | `FLoops.ThreadedEx()` | `normex` drops the `A.b` prefix |
| the `pi` spelling      | `pi`                | `π`            | `normex` rewrites `:pi` to `:π` |
| numeric literal width  | `1`                 | `1.0`          | every literal goes through `Float64` |
| operator spacing       | `2/3`               | `2 / 3`        | `string` re-prints canonically |

Anything else is drift.

### What it does not cover

- A block that declares only *positional* constructors takes no keywords and is out of scope.
  Forty blocks are in that class today; the census records them and asserts nothing about them.
- The census matches a documented signature against *any* definition of that name, because a
  name can carry both an outer constructor and a `@concrete` inner one. It therefore proves
  that the block describes a real signature, not that it describes the one a user reaches.

### Guards on the census itself

An extractor that quietly stops finding anything would pass every assertion. Three floors stop
that: at least 190 files, at least 1000 keyword-taking definitions, at least 300 documented
keyword signatures. The block must also parse — the `OptimEntropyPooling` comma is the case
that motivates it.

## Consequences

- Adding a keyword to a documented type now fails the census until the block is updated. That
  is the point.
- Changing a default now fails it too, which is the case that produced all seven stale defaults.
- The `# Constructors` block keeps its defaults, which `TYPEDSIGNATURES` cannot show.
- If `DocStringExtensions` ever renders default values, generation becomes possible again and
  this ADR should be amended, not deleted.
