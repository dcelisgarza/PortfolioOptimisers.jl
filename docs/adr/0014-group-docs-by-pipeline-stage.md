---
status: accepted
---

# Group the user guide and examples by pipeline stage

## Context

The docs have two narrative halves besides the API reference: a **User Guide**
(`user_guide/`, one monolithic `01_Basic_Optimisation.jl`, ~45 KB) positioned as a fast,
skimmable tour, and **Examples** (`examples/`, 11 files `01`–`11`) positioned as in-depth
single-feature walkthroughs. `CONTEXT.md` frames the package as a pipeline
`data → moments → prior → optimisation → post-processing` plus cross-cutting abstractions.

Both halves are assembled by `docs/make.jl`'s `generate_files`, which **globs a flat
directory and sorts by numeric filename prefix**; the resulting `pages` list is dropped into
Documenter as a flat `"User Guide" => […]` / `"Examples" => […]` with no sub-sections. (By
contrast the **API** section in the same `make.jl` *is* hand-grouped into nested subsections —
"Moments", "Distance", "Phylogeny", "Prior", …)

A knowledge-graph pass over the guide and example sources (graphify) made the coverage gap
concrete: the 11 examples cover roughly the MeanRisk path plus moment/factor estimation,
validation, and tuning, while whole glossary families have **no worked example** —
Black-Litterman / Entropy Pooling / Opinion Pooling, Uncertainty Sets, the constraint-
generation layer (linear/group/threshold/phylogeny/centrality), Turnover, Tracking, Fees /
Net Returns, Finite Allocation, and the alternative optimiser families (Risk Budgeting,
clustering optimisers, meta-optimisers, Near Optimal Centering) as deep dives in their own
right. Two "gaps" turned out to already exist under different names: example 07 *Risk Factor
Optimisation* is really **Factor Priors**, and example 08 *Improving Moment Estimation*
already covers low-order **and** high-order moment estimation.

Three forces had to be reconciled:

- **The guide-vs-examples boundary was fuzzy.** "Quick" vs "in-depth" does not by itself say
  where a new topic goes. Resolved as a *correlated* split rather than three independent
  axes: the **guide** is breadth-first + common-path + task-recipe (a shallow tour, one
  minimal blessed call per task), and the **examples** are depth-first + long-tail + concept
  (one deep dive per cross-cutting feature or optimiser family). These co-occur, so a topic's
  placement is unambiguous.
- **Scale.** "Tour every optimiser in the guide **and** give each its own deep example", plus
  filling the family gaps, roughly doubles the example count to ~30. A flat scroll of 30
  numbered files is not navigable.
- **The spine already exists.** `CONTEXT.md`'s pipeline order is the natural grouping for both
  halves, and the API nav already demonstrates nested grouping in this `make.jl`.

## Decision

Restructure both narrative halves around the `CONTEXT.md` pipeline spine, realised as
**subdirectories per group**, and rewrite `generate_files` to recurse and emit nested
Documenter pages (the same nested shape the API section already uses).

**User Guide** — one shallow page per pipeline group (flat files; it is small):

```text
user_guide/
  00_Introduction.jl
  01_Data_and_Priors.jl
  02_Optimisers.jl              # the full breadth tour: naive, JuMP (MeanRisk, RB, RRB,
                                #   FRC, NOC), clustering (HRP/HERC/SCHRP), meta (NCO/
                                #   Stacking/SubsetResampling) — minimal call each
  03_Constraints_and_Costs.jl
  04_Validation_and_Tuning.jl
  05_Post_Processing.jl
  06_Choosing_a_Strategy.jl     # the decision framework (compute × frequency × risk × budget
                                #   → which tools); the worked profiles live in examples
```

**Examples** — subdirectories per pipeline group, deep single-topic pages:

```text
examples/
  1_foundations/
    01_Getting_Started.jl              [keep ex01]
  2_moments_priors/
    01_Expected_Returns_Estimation.jl  ⬩ (from ex08, split)
    02_Covariance_Estimation.jl        ⬩ (from ex08, split: Gerber/denoise/sparsify/…)
    03_Higher_Moment_Estimation.jl     ⬩ (from ex08, split: coskew/cokurt)
    04_Factor_Priors.jl                [rename ex07]
    05_Black_Litterman.jl              ⬩  ─┐ sequenced arc:
    06_Entropy_Pooling.jl              ⬩   │ each builds on the
    07_Opinion_Pooling.jl              ⬩  ─┘ previous
    08_Uncertainty_Sets.jl             ⬩ (robust optimisation)
  3_optimisers/
    01_MeanRisk_Objectives.jl          [keep ex02]
    02_Efficient_Frontier.jl           [keep ex03]
    03_Pareto_Surface.jl               [keep ex04]
    04_Multiple_Risk_Measures.jl       [keep ex06]
    05_Risk_Budgeting.jl               ⬩ (asset/factor/relaxed + FRC)
    06_Clustering_Optimisers.jl        ⬩ (HRP/HERC/SCHRP)
    07_Meta_Optimisers.jl              ⬩ (NCO/Stacking/SubsetResampling)
    08_Near_Optimal_Centering.jl       ⬩
  4_constraints_costs/
    01_Budget_Constraints.jl           [keep ex05]
    02_Linear_Group_Constraints.jl     ⬩ (asset sets, weight bounds, threshold)
    03_Phylogeny_Centrality.jl         ⬩
    04_Turnover_and_Tracking.jl        ⬩
    05_Fees_and_Net_Returns.jl         ⬩
    06_Regularisation.jl               [keep ex09]
  5_validation_tuning/
    01_Cross_Validation.jl             [keep ex10]
    02_Hyperparameter_Tuning.jl        [keep ex11]
  6_post_processing/
    01_Finite_Allocation.jl            ⬩ (Discrete/Greedy)
    02_Plotting_and_Reporting.jl       ⬩
  7_putting_it_together/
    01_Profile_Retail_Daily.jl         ⬩ end-to-end (low compute/budget, high frequency)
    02_Profile_Desk_Monthly.jl         ⬩ end-to-end (high compute, monthly)
    03_Profile_Institutional.jl        ⬩ end-to-end (constrained, large budget)
```

~30 examples (11 kept, every one re-homed into a group; ~19 new ⬩). The guide page for each
group cross-links to the matching examples group, and vice versa.

**Granularity is hybrid** — one example per family by default, split into its own page only
where a mechanism is genuinely deep. Applications recorded above: the views family is three
sequenced pages (BL → EP → OP, because each builds on the last); moment estimation is split
three ways along the §3.1/§3.2/§3.3 glossary boundary; meta-optimisers stay one page;
constraints split into "linear/group" vs "phylogeny/centrality" vs cost-bearing
(turnover/tracking, fees).

**The capstone is distributed**, not one page: the *decision framework* lives shallow in the
guide (`06_Choosing_a_Strategy`); the *worked end-to-end profiles* live deep as a short series
in `7_putting_it_together/`; and every deep example carries a short **"When to reach for
this"** admonition so the strategy guidance is also reachable in situ.

**Two cross-cutting authoring requirements** apply to every page:

- **A closing plot.** Each example/guide page must end with a digestible visualisation
  (`plot_prior`, `plot_measures`, weight/contribution/frontier plots, …). Plotting is a
  per-page requirement, not only its own example.
- **A `## Findings` block in Literate `#src` lines.** Authoring a page is also a dogfooding
  sweep for documentation gaps, ergonomics friction, plotting gaps, latent bugs, and
  unexpected interactions. Findings are recorded in `#src` notes next to the provoking code
  (visible in source, stripped from rendered docs) and rolled up into **one tracking issue
  per pipeline group**.

**Authoring template** (two shapes, observed from the existing examples):

- *Prior/constraint/moment* topics: `1. ReturnsResult data → 2. Prior/feature → 3. Compare
  via a stock MeanRisk efficient frontier`.
- *Optimiser* topics invert it: `fixed stock prior → the optimiser's variants → compare`.

Each shape closes with the mandatory plot, the "When to reach for this" callout, and the
`#src` findings block.

## Considered options

- **Flat, append-only** (keep `01`–`11`, add `12`–`30` in any order) — rejected: zero
  renumbering and no `make.jl` change, but TOC order matches no learning path and related
  topics scatter; the weakest pedagogy at exactly the scale where navigability matters most.
- **Flat, renumbered into numeric theme-bands** — rejected: still a flat list of 30,
  encodes grouping fragilely in number ranges, and breaks existing URLs anyway without
  buying real sections.
- **Flat files + hand-maintained nested page-map in `make.jl`** (the API nav's approach) —
  viable and preserves existing example URLs, but the file layout stays flat and
  undiscoverable, the map must be hand-edited on every add, and we are already accepting URL
  changes by regrouping. Subdirectories make the grouping self-documenting on disk and let
  `generate_files` derive the nesting.
- **Coarse granularity** (one dense page per family: "Views-based Priors", "Meta-optimisers",
  "Constraints") — rejected: fewer pages but each mixes mechanisms and grows long; loses the
  per-tool navigability that motivated the restructure.
- **Fine granularity** (every named mechanism its own page) — rejected as the *default* but
  applied selectively via the hybrid rule where depth earns it.
- **Single monolithic guide page** — rejected: fastest to Ctrl-F but does not mirror the
  examples' spine, and grows unbounded as families are added.
- **One-example capstone** / **framework-only-with-callouts** — both folded into the chosen
  distributed capstone rather than picked exclusively.

## Consequences

- **`generate_files` must be rewritten to recurse subdirectories** and emit nested
  `"Group" => [pages…]` structures, preserving the existing per-file Literate `markdown` +
  `notebook` generation and the `.csv`/`.csv.gz` data-file copy/`git diff` handling. The
  shared data files (`SP500.csv.gz`, `Factors.csv.gz`, `SP500_idx.csv.gz`) are referenced by
  relative path from the example sources; moving examples into subdirectories changes their
  depth, so either the data files are centralised and the `joinpath(@__DIR__, "..")` paths
  updated, or copied per group.
- **Existing example URLs change** (`examples/03_Efficient_Frontier` →
  `examples/3_optimisers/02_Efficient_Frontier`). Inbound links (README, blog posts, the
  cross-links between guide and examples) must be updated; consider Documenter redirect stubs
  for the old paths.
- **Notebooks regenerate** at the new paths; the `.ipynb` outputs committed alongside sources
  move with them.
- **More pages to write and maintain** (~19 new examples + 5 new guide pages). The per-group
  findings issues double as the work tracker for this expansion.
- **`CONTEXT.md` is unaffected** — this is a docs-structure decision, not a glossary change;
  no new domain terms were introduced (the plan uses existing glossary vocabulary
  throughout).
- **The graphify graph already indexes the guide/example concept layer** (added in the same
  session), so coverage can be re-checked against the graph as pages are written.
