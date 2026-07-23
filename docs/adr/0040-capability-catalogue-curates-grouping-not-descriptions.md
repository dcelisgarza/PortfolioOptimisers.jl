---
status: accepted
---

# The capability catalogue curates grouping, not descriptions

## Context

`docs/src/api/00_API.md` carried a hand-written **Features** section: 748 of the
page's 796 lines, 401 distinct `@ref` targets, 113 nested `::: details`
collapsibles. It was the only place in the docs that answered "what can this
library do?" organised by *capability* rather than by source file, and it had
drifted badly.

Measured against the live package at the time of writing:

- **96 concrete leaf Estimators and Algorithms had no entry at all** — including
  whole families added by later ADRs: `Pipeline`/`PipelineStep` (0028),
  `TimeDependent` (0030), `TrainTestSplit` (0031), `L1UncertaintySet` (0032),
  the `Windowed*` estimators (0039), plus `MissingDataFilter`, `Imputer`,
  `PricesToReturns`, `RedundancySelector`, `Solver`, `L2Regularisation`,
  `CustomJuMPConstraint`, all four entropy-pooling optimisers, both
  opinion-pooling algorithms and both Schur-complement variants.
- **The Plotting section listed 12 of 36 plotting functions.**
- **The "Traditional optimisation features" subsection is a restatement of
  `fieldnames(JuMPOptimiser)`** (43 fields) and was missing nine of them.
- **Four `@ref`s were malformed** — `` [`Variance`] ``,
  `` [`ThirdCentralMoment`]-@(ref) ``, `` [`GreedyAllocation`] `` and a
  placeholder — rendering as dead literal text. Documenter cannot warn about
  these: a link with no target is not a broken link, it is not a link.
- **Eight `@id readme-*` anchors had zero inbound references** repo-wide,
  vestigial from when the content lived in the README.
- The opening sentence claimed entries "lack docstrings", which
  `test/test_26_docs.jl` had already made false.

Nothing checked any of it, so every drift was silent.

Three facts shaped the fix.

**The descriptions were already written, somewhere else.** Every type docstring
in the package opens with `$(DocStringExtensions.TYPEDEF)`, a blank line, then a
summary paragraph — 664 of 664 types, median 80 characters. Those summaries say
what the Features entries said. The section was therefore maintaining a *second,
unchecked description of every type*: it called `GeneralCovariance` "optionally
weighted covariance with custom covariance estimator" while the docstring called
it "a simple wrapper around a `StatsBase.CovarianceEstimator`…". Two
descriptions of one thing, neither validated against the other.

**The grouping was not.** The section's nesting is not the type hierarchy — it
groups by the job a thing does, cutting across `src/` files and across the
subtype tree, and its ordering deliberately follows the `CONTEXT.md` pipeline
spine. `FullMoment` and `SemiMoment` each appear in about eight different
groups. None of that is derivable.

**Functions have no equivalent convention.** A function has many methods and
many docstrings with no single summary; extraction fails for `prior`,
`optimise`, `posdef`, `owa_gmd` and `plot_efficient_frontier` alike. And
"exported" is not a usable boundary for them either: `concrete_typed_array`,
`is_leaf`, `pre_order` and `to_tree` are all genuinely `export`ed.

## Decision

Replace the hand-written section with a **generated page**, `Capability
Catalogue`, built from a declaration that curates **only the grouping**.

1. **`docs/capability_catalogue.jl`** holds the curated tree as plain data —
   `Section`, `Group`, `Cap`, `Note`, `Prose` — with no dependencies, so it can
   be `include`d from both `docs/` and `test/`.

2. **Descriptions come from docstrings.** A `Cap` names its targets and nothing
   else; `docs/generate_capability_catalogue.jl` resolves each one's description
   from the first sentence of its docstring at build time. There is now exactly
   one description of every type in the repo.

3. **`label` is an override, not the norm.** 347 of 418 entries carry no label.
   The 71 that do fall into two kinds: 26 whose sentence reads *through* its own
   links ("Mean-Risk [`MeanRisk`](@ref) returns a [`MeanRiskResult`](@ref)"),
   which are rendered verbatim; and 45 where a group's children would otherwise
   all repeat the same prefix — eight centrality algorithms each opening
   "Centrality algorithm type for …" is noise, and `Betweenness` is better.

4. **Coverage is checked in `test/test_26_docs.jl`, not only at docs-build
   time.** Every concrete leaf subtype of `AbstractEstimator` or
   `AbstractAlgorithm` must appear; every exported function must either appear
   or be listed in `NOT_A_FEATURE` with a reason (`:alias`, `:base_overload`,
   `:trait`, `:internal`). `NOT_A_FEATURE` is checked in **both** directions, so
   an exemption for something no longer exported fails too.

5. **The generator also refuses** to render an incomplete page. The test is
   authoritative and fires far sooner, but the generator's failure mode is worse
   than a red test: a page that quietly omits a capability *looks* complete,
   which is the defect being eliminated.

6. **The page moves out of the API section**, to a top-level nav slot between
   Home and User Guide. `00_API.md` shrinks to the design-philosophy essay its
   title promises. The API section is organised 1:1 with `src/`; a
   capability-oriented view was a foreign body in it, and a reader deciding
   whether the library suits them never gets that far.

The catalogue is the **selective** view — Estimators and Algorithms, the things
a user chooses. `26_TypeHierarchy.md` remains the **exhaustive** view, including
every abstract type. They answer different questions and are kept apart.

## Consequences

- Adding an estimator or algorithm now forces a placement decision, on the PR
  that adds it. This is enforced by CI; `TestOnPRs.yml` gained
  `docs/capability_catalogue.jl` as a trigger path so that editing the catalogue
  alone still runs the check that guards it.
- **The docstring summary convention became load-bearing** and is now written
  down in `.github/instructions/julia-docstrings.instructions.md`. It was
  previously true in all 664 cases but recorded nowhere, so it could have lapsed
  silently. Adopting it also required fixing the docstrings it exposed: 150
  redundant "in `PortfolioOptimisers.jl`" qualifiers removed across 45 files,
  eight run-on summaries split into a crisp first sentence plus a detail
  sentence (losing nothing from the API page), nine filler openers rewritten,
  and `Detone`'s summary reworded because it linked its own verbs and so
  rendered them twice.
- Typos become impossible rather than merely checked: a `Cap` names a `Symbol`
  resolved against the module, so a misspelling fails at load. The four
  malformed links that nothing caught cannot recur in this form.
- Roughly 200 curated strings — group headings, orderings, the 71 labels, the
  prose — remain hand-maintained. That is accepted: they are structural and
  near-static, whereas the 343 generated descriptions are the part that grows
  with the codebase, and growth is where drift came from.
- Coverage is "at least once", not a partition, because `FullMoment` and
  `SemiMoment` legitimately recur across groups. An accidental duplicate entry
  is therefore not caught.

## Alternatives considered

**Keep the markdown and add a coverage checker.** Cheapest, and it would have
caught the 96 missing types. Rejected because it leaves 664 duplicated
descriptions in place — the same defect one level down.

**Generate everything, including the grouping, from the type tree.** Rejected:
the grouping is real information that the subtype tree does not contain. The
result would be a second type-hierarchy page, which already exists.

**Tag each type in `src/` with `@feature "group/subgroup"`.** Attractive because
the grouping would live next to the type it describes. Rejected on three counts
checked against the actual content: a tag cannot express *order* (neither of
groups nor of entries within one, and `Gerber0`/`Gerber1`/`Gerber2` is not
alphabetical), cannot place one type in several groups (`FullMoment` needs
about eight), and cannot state that the top-level order follows the `CONTEXT.md`
pipeline spine. It also couples the package to a docs concern.

**Range the check over exported symbols.** Rejected: 133 of 704 exports are
alias constructors, and the rest include error types and `Base` overloads, so
the check would need a ~200-name denylist — the drifting list again, relocated.
Ranging over concrete leaf Estimators and Algorithms instead gives a boundary
derived from the domain model in `CONTEXT.md`, which a contributor can apply
without consulting a list.
