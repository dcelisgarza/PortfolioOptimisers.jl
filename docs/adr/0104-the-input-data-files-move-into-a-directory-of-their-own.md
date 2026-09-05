---
status: accepted
---

# The input-data files move into a directory of their own

## Context

The top level of `src/` held four files under the prefix `03_`:

| File | Code lines | What it holds |
| ---- | ---------- | ------------- |
| `03_AssetPanel.jl` | 175 | `AssetPanel`, the field index and the two universe masks |
| `03_AssetPanelBuilder.jl` | 330 | `asset_panel` and the fill policies that resolve a blank |
| `03_Preprocessing.jl` | 726 | `ReturnsResult`, `PricesResult`, `prices_to_returns`, the preprocessing estimators and `train_test_split` |
| `03_SyntheticAssetPanel.jl` | 310 | `synthetic_asset_panel` and the constants it draws from |

One prefix over four files is the shape every other multi-file subject in `src/` already gave a
directory. `src/01_Base/`, `src/08_Moments/`, `src/13_Prior/` and seven more each hold a numbered
sequence under one name. The four files above are one subject by the same test: they are the
input data a portfolio optimisation reads, before any moment is estimated. Nothing else at the
top level of `src/` shares a prefix with a sibling.

The prefix also carried no order. A reader of `src/` could not tell from the names that
`03_AssetPanel.jl` loads before `03_Preprocessing.jl`, and the four names sort alphabetically
rather than in load order.

## Decision

**The four files move into `src/03_InputData/`, numbered in load order.**

| Was | Is |
| --- | -- |
| `src/03_AssetPanel.jl` | `src/03_InputData/01_AssetPanel.jl` |
| `src/03_AssetPanelBuilder.jl` | `src/03_InputData/02_AssetPanelBuilder.jl` |
| `src/03_Preprocessing.jl` | `src/03_InputData/03_Preprocessing.jl` |
| `src/03_SyntheticAssetPanel.jl` | `src/03_InputData/04_SyntheticAssetPanel.jl` |

**The move is structural.** No line of the four files changes. No type, no bound, no verb, no
docstring and no `export` moves relative to another. Every gate number the four rows carried is
carried by the same row under the new path: the complexity maxima and sums, the coverage lines
and misses, the three JET runs, the size counts, the sweep units and the one Exemption of
`code_health/rulings.toml`.

**A new input-data file joins the directory.** A file that reads or builds the data an
optimisation starts from belongs here, and takes the next number.

### What follows the move

- `src/PortfolioOptimisers.jl` includes the four files at their new paths, in the same order.
- The four rows of `sweep/manifest.toml` keep `map = 2`, their units and their `swept` state.
  The directory uses exactly one map, which is the state `CodeHealth.candidate_maps` reads.
- The four rows of `complexity_baseline.toml`, `coverage_baseline.toml`, `size_baseline.toml`
  and the twelve rows of `jet_baseline.toml` keep every number. A refresh would have written
  the same rows: all four gates call `CodeHealth.pair_renames`, which pairs a dead row with a
  new row that measures the same.
- The `ReturnsResult` argument-count Exemption of `code_health/rulings.toml` names the new path.
- The API pages do not move. `docs/src/api/03_Preprocessing.md` stays one page and keeps its
  name, because a page is checked by the units it renders and never by the file they are
  declared in. ADR 0100 settled this for `docs/src/api/01_Base.md`.
- `ext/PortfolioOptimisersImputeExt.jl` and `test/test_26_docs.jl` name the new path in prose.
- `CodeHealth.candidate_maps` no longer states a count of subdirectories or of top-level files.
  Both numbers had drifted before this move, and the sentence they stood in does not need them.

**An ADR that names an old path in prose keeps it.** ADR 0028, ADR 0029 and ADR 0042 name
`src/03_Preprocessing.jl`, and all three reached `main`. Each statement was true when it was
written, and ADR 0100 settled this rule for the same kind of move.

## Consequences

No numeric prefix appears twice at the top level of `src/` any more. A reader who opens
`src/03_InputData/` reads the load order off the names.

A reader who knows an old path finds nothing at it. Every reference under `src/`, `ext/`,
`test/`, `code_health/` and `sweep/` moves with the files. The historical measurement dumps under
`docs/reports/` keep the old paths: they record what was measured on the day they were written.

`synthetic_asset_panel` stays in `src/`, where issue #656 put it. It is exported public API, it
has an API page entry and a Capability Catalogue entry, and an example does `using
PortfolioOptimisers` and nothing else, so a generator under `test/` cannot reach one. This move
changes neither that decision nor the file's content.

## Alternatives considered

- **Renumber the four files at the top level.** `03_`, `04_`, `05_`, `06_` and a shift of every
  later prefix. It fixes the collision and not the shape: four files of one subject stay four
  top-level entries, and the shift rewrites twenty file names to move four.
- **Move only the panel files, and leave `03_Preprocessing.jl` where it is.** The prefix
  collision would remain, and the preprocessing file is the one the other three are input data
  beside.
- **Move `04_SyntheticAssetPanel.jl` under `test/`.** Refused. It is exported public API and the
  deep-dive example it was built for cannot reach a fixture under `test/`. Issue #656 measured
  this.
