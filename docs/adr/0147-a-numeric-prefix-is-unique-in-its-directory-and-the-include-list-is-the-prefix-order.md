---
status: accepted
---

# A numeric prefix is unique in its directory, and the include list is the prefix order

## Context

`.github/instructions/julia-source-code.instructions.md` § *Code Organization* states that a
source file's numeric prefix is its load order. Three directories under `src/` broke the rule,
and nothing gated it:

| Directory | Entries under one prefix | Since |
| --------- | ------------------------ | ----- |
| `src/` | `21_ExpectedReturns.jl`, `21_FactorAttribution.jl`, `21_FactorAttributionRealised.jl` | `9de0fc4386` (#782) |
| `src/19_RiskMeasures/` | `16_NoRisk.jl`, `16_Range.jl` | `4709360dcc` (#154) |
| `src/20_Optimisation/20_RiskMeasureConstraints/` | `16_NoRiskConstraints.jl`, `16_RangeConstraints.jl` | `4709360dcc` (#154) |

`docs/src/api/` mirrors `src/` and is numbered the same way, and it carried a fourth:
`docs/src/api/19_RiskMeasures/26_AdjustRiskContributions.md` beside `26_NoRisk.md`.

The load works in every case, because `src/PortfolioOptimisers.jl` lists every file by hand and
that list is the order. Under a shared prefix the number is a label there, not the order.
Issue #1060 found the first row while reviewing PR 625 and stated that every other directory
kept its numbers unique, which the census below showed to be false: the two `16_` pairs had
stood for far longer than the `21_` triple.

ADR 0104 met the same defect at the top level of `src/` once before, for four files under `03_`,
and settled it by giving them a directory. It did not add a gate, and the defect returned.

## Decision

**Every entry of a directory under `src/`, `ext/` and `docs/src/api/` carries a distinct numeric
prefix, and the `include` list of `src/PortfolioOptimisers.jl` is the listing of `src/` sorted by
that prefix.** The number on a file is the order the module loads it in, and two files cannot
state one order. `test/test_47_alias_and_module_census.jl` gates both claims beside the module
census that already reads the include list.

**The three directories are renumbered so the file that loads first keeps its number and every
later file takes the next.** No line of any renumbered file changes except a comment or a
docstring that names a sibling by path.

| Was | Is |
| --- | -- |
| `src/21_FactorAttribution.jl` | `src/22_FactorAttribution.jl` |
| `src/21_FactorAttributionRealised.jl` | `src/23_FactorAttributionRealised.jl` |
| `src/22_Preselection.jl` | `src/24_Preselection.jl` |
| `src/23_Pipeline/` | `src/25_Pipeline/` |
| `src/24_Plotting.jl` | `src/26_Plotting.jl` |
| `src/25_Aliases.jl` | `src/27_Aliases.jl` |
| `src/19_RiskMeasures/16_Range.jl` … `28_RiskMeasureTools.jl` | `17_Range.jl` … `29_RiskMeasureTools.jl`, each one higher |
| `src/20_Optimisation/20_RiskMeasureConstraints/16_RangeConstraints.jl` … `21_GenericValueatRiskRangeConstraints.jl` | `17_RangeConstraints.jl` … `22_GenericValueatRiskRangeConstraints.jl`, each one higher |

`16_NoRisk.jl` and `16_NoRiskConstraints.jl` keep their numbers: each loads before the file it
shared the number with.

**The API pages move with the files.** ADR 0104 left `docs/src/api/03_Preprocessing.md` in place
because a page is checked by the units it renders and never by the file they are declared in.
That rule says what a page must contain; it says nothing about what a page is called, and the
API tree is numbered so that it reads beside `src/`. So `docs/src/api/` takes the same
renumbering: `22_FactorAttribution.md`, `24_AssetSelection.md`, `25_Pipeline/`, `26_Plotting.md`,
`27_Aliases.md`, and the two risk-measure directories one higher from `17_` up. `26_NoRisk.md`
becomes `16_NoRisk.md`, where its source file sits. `docs/generate_type_hierarchy.jl` numbers
the type-hierarchy page one past the largest prefix in the tree, so that page becomes
`28_TypeHierarchy.md` on the next docs build, and the tracked copy is moved with it.

### What follows the renumbering

- `src/PortfolioOptimisers.jl` includes every file at its new path, in the same order.
- The rows of `code_health/sweep_manifest.toml`, `complexity_baseline.toml`,
  `coverage_baseline.toml`, `size_baseline.toml` and `jet_baseline.toml` keep every number under
  the new path. A refresh would have written the same rows: all four gates call
  `CodeHealth.pair_renames`, which pairs a dead row with a new row that measures the same.
- `docs/make.jl` does not change. It indexes the API directories by position in a `walkdir`,
  and no directory is added or removed.
- The relative links of ADRs 0004, 0005, 0047 and 0051, of
  `.github/instructions/julia-docstrings.instructions.md` and of `docs/src/api/15_Turnover.md`
  point at the new paths, because a dead relative link reds the link checker.
- The two links into the deployed site, `stable/api/26_Plotting` and `stable/api/27_Aliases`,
  answer 404 until the next release and are excluded in `.lychee.toml` under the same reason as
  `tree/main/src/01_Base`.
- The `research/` notes name the new paths, because two of them link to `blob/dev/`.

**An ADR that names an old path in prose keeps it**, as ADR 0104 settled. ADRs 0038, 0039, 0040,
0067, 0081, 0086, 0092, 0101, 0103, 0111, 0132 and 0142 name a path that this ADR moved, and every
one was true when written.

## Consequences

A reader of any directory under `src/` reads the load order off the names, and a reader of
`src/PortfolioOptimisers.jl` reads a list that a sort would reproduce. A file added under a
number a sibling holds, or included out of its numbered place, reds
`test/test_47_alias_and_module_census.jl` on the commit that adds it, which is what would have
caught #782 and #154.

A reader who knows an old path finds nothing at it. Sixty-one paths moved and every reference
under `src/`, `test/`, `code_health/`, `docs/src/api/`, `user_guide/` and `.github/` moved with
them.

## Alternatives considered

- **Accept the shared number and amend the naming rule** to say the include list is the order
  and the prefix is the stage. The cheaper change, and it makes the rule describe the defect.
  A reader of `src/` would still be unable to tell which of two files loads first.
- **Give the three attribution files a directory**, as ADR 0104 did for the input data. The
  expected-returns file is not attribution, so the directory would hold two files of one
  subject and one of another, or the two attribution files alone under a prefix that still
  needs a shift of everything after it.
- **Renumber the top level only, and allow-list the two `16_` pairs in the census.** An
  allow-list of collisions is a ratchet on a count that should be zero, and each pair is six
  renames of the same shape as the rest.
