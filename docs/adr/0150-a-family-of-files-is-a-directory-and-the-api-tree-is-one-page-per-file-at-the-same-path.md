---
status: accepted
---

# A family of files is a directory, and the API tree is one page per file at the same path

## Context

`src/` held 301 files in fourteen directories, and the directories did not follow the families.
`src/08_Moments/` was fifty-one files and two subdirectories under one prefix run, where the three
Gerber-family covariances stood at `05_`, `06_` and `35_`, the five `Windowed*` files at
`28_`–`32_`, the three `ExpWeighted*` files at `51_`–`53_`, and the cross-sectional factor model
— its regression, its weights, its exposures, its family basis, its return forecasts and its
diagnostics — at `38_`–`49_` with `40_HigherMomentPartialFit.jl` in the middle. `src/11_Phylogeny/`
held the DBHT machinery at `04_` and `07_`–`12_`, the centrality files at `14_`, `15_`, `18_` and
`23_`, and the network files at `16_`, `17_`, `19_` and `20_`. `src/13_Prior/` held five
Black-Litterman files and three entropy-pooling files in one run; `src/19_RiskMeasures/` held the
X-at-Risk family at `06_`–`09_` and `20_`; `src/20_Optimisation/` held the hierarchical, JuMP, meta
and finite-allocation optimisers in one run of twenty-five entries. The top level held the four
matrix-processing files and the two factor-attribution files as loose siblings of the directories.

`docs/src/api/` is numbered to read beside `src/` (ADR 0147), and `docs/src/api/00_API.md`
states a one-to-one correspondence between pages and source files. It did not hold: `01_Base.md`
was one page for the sixteen files of `src/01_Base/`, `03_Preprocessing.md` one page for the
eighteen of `src/03_InputData/`, `22_FactorAttribution.md` one page for two files,
`24_AssetSelection.md` the page of `src/24_Preselection.jl`, `02_DeltaUncertaintySet.md` and
`10_OWARiskMeasure.md` were off their files by a letter, and the JuMP method of `NoRisk` was on no
page at all. `docs/make.jl` indexed the API directories by position in a `walkdir`, so every added
directory shifted every later index by hand.

Issue #1058 asked for the reorganisation and carried execution.

## Decision

**A family of two or more files under one subject is a directory, numbered as one entry of its
parent, and its files are numbered inside it.** The rule is written into
`.github/instructions/julia-source-code.instructions.md` § *Code Organization*. The families are
the ones the issue named, and the placements the issue left open are settled here:

| Directory | Holds |
| --------- | ----- |
| `src/04_MatrixProcessing/` | posdef, denoise, detone, matrix processing |
| `src/05_Moments/05_Gerber/` | Gerber, Smyth-Broby, Gerber IQ |
| `src/05_Moments/21_TimeSeriesRegression/` | stepwise and dimension-reduction regression; `20_Base_Regression.jl` stays above because it roots the cross-sectional family too |
| `src/05_Moments/26_Windowed/` | the five windowed estimators |
| `src/05_Moments/29_ExpWeighted/` | the three exponentially weighted estimators, placed before the regime-adjusted ones that build on the same weighting |
| `src/05_Moments/30_RegimeAdjusted/` | the two regime-adjusted estimators |
| `src/05_Moments/32_CrossSectionalFactorModel/` | the regression, the weights, the factor model, `04_FactorExposures/`, the family basis and its transforms, `07_ReturnForecasts/`, and the three diagnostics and the summary — one subject, so one directory, rather than a "cross-sectional regression" directory that leaves its exposures and forecasts outside |
| `src/08_Phylogeny/06_DBHT/` | the DBHT estimator, PMFG, graph traversal, clique hierarchy, bubble tree, dendrogram and clustering; `07_LoGo.jl` stays outside, because it is a matrix-processing algorithm that reads the PMFG rather than DBHT functionality |
| `src/08_Phylogeny/08_Network/` | minimum spanning tree, network estimator, network graph, separation |
| `src/08_Phylogeny/09_Centrality/` | centrality, polarity, estimator, queries; after the network, because `CentralityEstimator` binds `AbstractNetworkEstimator` |
| `src/10_Prior/05_BlackLitterman/` | the views and the four Black-Litterman priors |
| `src/10_Prior/06_EntropyPooling/` | the base, Meucci and general entropy-pooling priors; opinion pooling stays outside, because it pools priors rather than views |
| `src/16_RiskMeasures/06_XatRisk/` | VaR, CVaR, EVaR, RLVaR and their power-norm form; `18_GenericValueatRiskRange.jl` stays outside, because its `ValueatRiskRMs` union names `WorstRealisation`, which loads after the family |
| `src/17_Optimisation/04_Hierarchical/` | the clustering base, HRP, Schur HRP, HERC |
| `src/17_Optimisation/05_JuMP/` | the JuMP base, `02_JuMPConstraints/`, the optimiser, mean-risk, factor risk contribution, near-optimal centering, the two risk budgetings, and `09_RiskMeasureConstraints/`, which mirrors `16_RiskMeasures/` entry for entry including its own `06_XatRisk/` |
| `src/17_Optimisation/06_Meta/` | the meta base, nested clustered, stacking, subset resampling |
| `src/17_Optimisation/07_FiniteAllocation/` | the finite base, discrete and greedy |
| `src/19_FactorAttribution/` | the attribution and its realised form |

The top level closes its gaps: `01_Base/`, `02_Tools.jl`, `03_InputData/`, `04_MatrixProcessing/`,
`05_Moments/`, `06_Distance/`, `07_JuMPModelOptimisation.jl`, `08_Phylogeny/`,
`09_ConstraintGeneration/`, `10_Prior/`, `11_UncertaintySets/`, `12_Turnover.jl`, `13_Fees.jl`,
`14_NetReturnsDrawdowns.jl`, `15_Tracking.jl`, `16_RiskMeasures/`, `17_Optimisation/`,
`18_ExpectedReturns.jl`, `19_FactorAttribution/`, `20_AssetSelection.jl`, `21_Pipeline/`,
`22_Plotting.jl`, `23_Aliases.jl`. Every file keeps its relative order except where a family
gathers, and no line of any moved file changes except a comment or a docstring that names a
sibling by path.

**Two files are renamed for the glossary.** `CONTEXT.md` names the concept *Asset Selector*, so
`src/24_Preselection.jl` becomes `src/20_AssetSelection.jl`, the name its page already carried, and
`src/03_InputData/13_AssetSelection.jl`, which holds the root `AbstractAssetSelector` and the
fit/apply seam, becomes `13_Base_AssetSelection.jl` under the `Base_` convention of every other
family root.

**`docs/src/api/` is one page per source file at the same path, and a directory of files is a
directory of pages.** `01_Base.md` becomes the sixteen pages of `01_Base/`, `03_Preprocessing.md`
the eighteen of `03_InputData/`, and `22_FactorAttribution.md` the two of `19_FactorAttribution/`.
Each `@docs` entry goes to the page of the file that declares the docstring it renders, resolved
from the docstring's own path metadata and, for a signature-qualified entry, from the method whose
signature it names; a section heading follows its entries, its prose follows the first page that
takes an entry from it, and the directory intro opens the group's first page.
`02_DeltaUncertaintySets.md` and `07_OWARiskMeasures.md` take their files' names.
`16_NoRiskConstraints.md` is added for the one JuMP method no page rendered. The type-hierarchy
page, numbered one past the largest prefix, becomes `24_TypeHierarchy.md`. `docs/make.jl` builds
the API navigation by walking the tree: a file is a page, a directory is a group labelled by its
name, and nothing is positional.

### What follows the move

- `src/PortfolioOptimisers.jl` includes every file at its new path, in the listing order, which
  `test_47_alias_and_module_census.jl` gates. Load order was checked statically before the move
  — every pair of files the gathering reorders, against every top-level name the later one
  defines — and one real dependency surfaced: `GenericValueatRiskRange.jl` names
  `WorstRealisation` in a `const` union, which is why it stays outside `06_XatRisk/`. The package
  loads, and JET reports the same rows.
- The rows of `code_health/sweep_manifest.toml`, `complexity_baseline.toml`,
  `coverage_baseline.toml`, `size_baseline.toml`, `jet_baseline.toml` and the exemptions of
  `rulings.toml` keep every number under the new path, re-sorted as a refresh writes them.
- Every reference under `src/`, `test/`, `code_health/`, `docs/src/api/`, `docs/src/contribute/`,
  `user_guide/`, `examples/`, `research/`, `.github/`, `STANDARDS.md` and `.lychee.toml` names the
  new path. In `docs/adr/` only a link target moves; **an ADR that names an old path in prose keeps
  it**, as ADRs 0104 and 0147 settled, and the two dead links ADR 0042's amendment retains stay as
  written.
- The open issues whose title or body names a moved path are retitled and rewritten to the new
  one, because `code_health/sweep_check.jl` matches a sweep sub-issue to its file by title.
- The links into the deployed site answer 404 until the next release, as ADR 0147 recorded for
  the previous renumbering, and are excluded in `.lychee.toml` under the same reason.

## Consequences

A reader finds a family by its directory and reads the load order off the names in it; a family
that grows adds a file to its directory rather than a number to a run of fifty. A page is found at
the path of its file, and a page that documents two files is a defect the API intro already
names. Adding a directory under `docs/src/api/` no longer edits `docs/make.jl`.

Every path under `src/` except `01_Base/`, `02_Tools.jl` and seventeen of `03_InputData/` changed, so every
reader who knows an old path finds nothing at it, and every sweep sub-issue, map and memory that
named one had to be rewritten in the same change.

## Alternatives considered

- **Group without reordering**: include a directory's files where they stood, so `05_Gerber/`
  loads at three separate points. That keeps the load order untouched and breaks ADR 0147, whose
  rule is that the listing sorted by prefix is the include list.
- **A "cross-sectional regression" directory of three files**, as the issue's list read, with
  the exposures, the family basis, the return forecasts and the diagnostics left in the parent.
  The exposures and forecasts exist only for the cross-sectional model, so the parent would still
  interleave one subject across nine entries.
- **Leave `01_Base.md` and `03_Preprocessing.md` as they were**, by ADR 0104's rule that a page is
  checked by the units it renders. The rule says what a page must contain, not how many files it
  may cover, and the API intro promises one page per file; ADR 0147 already moved the pages with
  the files for the same reason.
- **Keep the positional `walkdir` indexing in `docs/make.jl`** and add the eighteen new indices
  by hand. The list would be rewritten by the next directory anyone adds.
