---
status: accepted
---

# A matrix source is named, not flagged: `cle_pr::Bool` becomes `x_src::Symbol`

## Context

Three optimisers — [`JuMPOptimiser`](../../src/20_Optimisation/10_JuMPOptimiser.jl),
[`HierarchicalOptimiser`](../../src/20_Optimisation/04_Base_ClusteringOptimisation.jl) and
[`NestedClustered`](../../src/20_Optimisation/17_NestedClustered.jl) — carried a field
`cle_pr::Bool = true`, forwarded as a keyword argument into the prior-layer bridge in
[`01_Base_Prior.jl`](../../src/13_Prior/01_Base_Prior.jl), where eight sites spelled the same line:

```julia
X = isnothing(rd) || cle_pr ? pr.X : rd.X
```

Two asset-returns matrices are reachable at that point — the prior result's `X` and the raw
`ReturnsResult`'s `X` — and the flag picked between them. Both the name and the documented meaning
were wrong about that:

- **The documented meaning was wrong.** `field_dict[:cle_pr]` read *"whether to pass the prior
  result to the clustering estimator"*. It does no such thing; the clustering estimator only ever
  receives a raw matrix. It selects which matrix.
- **The `cle_` prefix was wrong.** The keyword reaches `clusterise`, but also `phylogeny_matrix`,
  `phylogeny_constraints`, `centrality_vector` and `centrality_constraints`. It is not a clustering
  knob.

Issue [#160](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/160)'s `FeatureDistance`
work introduces a second selector of exactly the same kind. Issue
[#167](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/167) settled that the feature
matrix `Z` also has two carriers — a user-supplied one on `ReturnsResult` and a derived one on the
prior result — chosen by `z_src::Symbol = :data`, and that `z_src` must be an optimiser field for
the same reason `cle_pr` is one: the prior layer loads after the distance layer and cannot resolve
it. A `Bool` next to a `Symbol` would have made two instances of one idea read as two ideas.

## Decision

**Rename `cle_pr::Bool = true` to `x_src::Symbol = :prior`, as a clean break with no deprecation
path.** `:prior` reads `pr.X`, `:data` reads `rd.X`; behaviour at the default is unchanged. The two
selectors then read as one idea:

```julia
x_src::Symbol = :prior    # which RETURNS matrix   (pr.X vs rd.X)
z_src::Symbol = :data     # which FEATURE matrix   (pr.Z vs rd.Z)
```

Their defaults differ deliberately and that difference is the point of naming them: the returns
matrix defaults to the derived carrier because the prior is what the optimisation is defined on,
while the feature matrix defaults to the user's own data because explicit data outranks derived
(#161). A shared `Bool` spelling could not have expressed opposite defaults without one of them
reading as "false means yes".

Three consequences follow.

**A clean break, not a deprecation.** `cle_pr` was never right — accepting it for a release would
keep a name that misdescribes what it does. The library has taken this route before; ADR
[0041](0041-one-resource-cap-per-sink.md) renamed `ResourceLimits`' fields outright. A `Bool` passed
to `x_src` is a `MethodError` at the inner constructor, so no silent misbehaviour survives the
rename.

**`x_src` stays static.** It is ADR [0030](0030-time-dependent-constraints.md) execution control —
*how* the problem is solved, not *what* the problem is — so it is not a `TD_Option` and takes no
`TimeDependent` schedule. Provenance is not a per-fold knob: a schedule that read the prior on some
folds and the raw data on others would produce a weight path no single problem definition explains.

**The Symbol is validated.** A `Bool` could not be wrong; a `Symbol` can. `assert_source_selector`
in [`01_Base/10_Assertions.jl`](../../src/01_Base/10_Assertions.jl)
enforces `src in (:prior, :data)` and is called from all
three inner constructors, so a typo throws where it was written rather than silently selecting the
other carrier. The bridge is validated too, via a single `returns_matrix_picker(pr, rd, x_src)` that
replaces the eight copies of the ternary — the direct `clusterise(cle, pr; x_src = …)` entry points
are public and do not pass through an optimiser constructor.

## Consequences

- **Breaking.** Every caller spelling `cle_pr` must be rewritten; there is no shim.
- `field_dict[:cle_pr]` becomes `field_dict[:x_src]` and its text now describes selection rather
  than the fictional "pass the prior result" behaviour.
- The five doctest blocks that print an optimiser gain a `x_src ┼ Symbol: :prior` row in place of
  `cle_pr ┼ Bool: true`; the label column is anchored by `strict`, so alignment is unchanged for
  every other row.
- `returns_matrix_picker` gives issue [#168](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/168)
  a template for the `z_src` half: one picker per matrix, validated once, called from every bridge
  site.
