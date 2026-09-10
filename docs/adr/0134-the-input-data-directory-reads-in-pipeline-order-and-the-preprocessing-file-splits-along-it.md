---
status: accepted
---

# The input-data directory reads in pipeline order, and the preprocessing file splits along it

## Context

ADR 0104 gave the input data a directory of its own and numbered its files in load order. Load order
is not the same thing as reading order, and four commits of map #955 have pulled the two apart. The
ingestion layer arrived after the feature-side files and took the next free numbers, so the price
path ran 03, 07, 08, 09 with three files about features and panels sitting between its halves:

| File | What it holds |
| ---- | ------------- |
| `03_Preprocessing.jl` | `PricesResult`, `ReturnsResult`, `prices_to_returns`, the gap-return family, the preprocessing hierarchy, `TrainTestSplit`, the asset-selector seam, `MissingDataFilter`, `Imputer` |
| `04_CrossSectionalTransforms.jl` | the cross-sectional transforms |
| `05_FeatureSelector.jl` | `feature_matrix` and the Feature Selector |
| `06_PanelDataFrame.jl` | `panel_dataframe` |
| `07_ListingSpan.jl` | `ListingSpan`, `listing_span`, `universe_masks` |
| `08_PriceGapFill.jl` | `PriceGapFill`, the span-bounded fill |
| `09_PriceIngestion.jl` | `PriceIngestion`, `price_ingestion`, the door of the layer |

Two things are wrong with that, and they are the same thing twice.

**The order does not read as the pipeline.** A caller's data enters at `price_ingestion`, becomes a
`PricesResult`, is converted by `prices_to_returns`, and becomes a `ReturnsResult`. The directory
says that sequence is 09, 03, 03, 03, with the price-level steps at 08 before the door at 09 that
produces what they consume, and with the listing span they all read at 07, after the file that reads
it. Nothing in the numbering is wrong as *load* order — a verb resolves at call time, so the layer's
three files may load last — but a reader looking for the path the data takes cannot find it in the
names.

**One file holds both data levels and every step that touches either.** At 2747 lines
`03_Preprocessing.jl` was the second-largest file under `src/`, and its subjects are not one subject:
the price carrier and the returns carrier, the arithmetic that converts one into the other, the
fit/apply seam that every preprocessing estimator implements, two price-level filters, a split, and
the asset-selector root. A reader who wants `PricesResult` reads past `ReturnsResult`; a reader who
wants either reads past the conversion; and three sessions working on the price path, the returns
path and the split all queue on one file.

## Decision

**The directory is renumbered so that its names read in pipeline order, and
`03_Preprocessing.jl` splits along the same line.**

| Was | Is |
| --- | -- |
| `01_AssetPanel.jl` | `01_AssetPanel.jl` |
| `02_AssetPanelBuilder.jl` | `02_AssetPanelBuilder.jl` |
| `07_ListingSpan.jl` | `03_ListingSpan.jl` |
| `03_Preprocessing.jl` (the fit/apply seam) | `04_Preprocessing.jl` |
| `03_Preprocessing.jl` (the shared carrier views) | `05_CarrierViews.jl` |
| `03_Preprocessing.jl` (`PricesResult`) | `06_PricesResult.jl` |
| `09_PriceIngestion.jl` | `07_PriceIngestion.jl` |
| `08_PriceGapFill.jl` | `08_PriceGapFill.jl` |
| `03_Preprocessing.jl` (`MissingDataFilter`, `Imputer`) | `09_PriceFilters.jl` |
| `03_Preprocessing.jl` (`ReturnsResult`) | `10_ReturnsResult.jl` |
| `03_Preprocessing.jl` (the conversion) | `11_PricesToReturns.jl` |
| `03_Preprocessing.jl` (`TrainTestSplit`) | `12_TrainTestSplit.jl` |
| `03_Preprocessing.jl` (the asset-selector seam) | `13_AssetSelection.jl` |
| `04_CrossSectionalTransforms.jl` | `14_CrossSectionalTransforms.jl` |
| `05_FeatureSelector.jl` | `15_FeatureSelector.jl` |
| `06_PanelDataFrame.jl` | `16_PanelDataFrame.jl` |

The order states the path: the panel and its span, the seam every step implements, the price carrier,
the door, the two price-level steps, the returns carrier, the conversion, and then the steps and
views that read a returns carrier.

**The split is verbatim.** Every one of the 2743 non-`export` lines of `03_Preprocessing.jl` appears
once in exactly one new file, in its original order, and no line changes. The old file's single
`export` list becomes one `export` line per new file, holding the same sixteen names. That is the
whole content change, and it is why the gate numbers below partition rather than move.

**The fit/apply seam loads before every estimator, which the pipeline order alone would not give.**
A reader's order would put the door first, at the head of the price run. A struct's supertype is
resolved when the struct is defined, so `PriceGapFill`, `MissingDataFilter`, `Imputer`,
`PricesToReturns`, `TrainTestSplit` and `AbstractAssetSelector` each need
`AbstractPreprocessingEstimator` to exist already. `04_Preprocessing.jl` therefore holds the six
abstract types and the two generic verbs, and nothing else; it is the declaration the rest of the
directory implements. For the same reason the door sits at 07 rather than 01: `price_ingestion`
returns a `PricesResult`, so the carrier is defined first.

**Each carrier file declares its own result root.** `AbstractPricesResult` moves to
`06_PricesResult.jl` and `AbstractReturnsResult` to `10_ReturnsResult.jl`, beside the one concrete
type each of them has. `Prices_RR`, the alias over both, is declared in `10_ReturnsResult.jl`, the
later of the two.

### What follows the split

- `src/PortfolioOptimisers.jl` includes sixteen files where it included nine, in the order above.
- **`sweep/manifest.toml` partitions exactly.** The eight new rows carry 9, 4, 3, 4, 8, 8, 7 and 6
  units, which sum to the dead row's 49. They carry `map = 2` and `swept = false`, which is the state
  the dead row was in. Each one also carries the `# Algorithm` count of its own file, so the floor
  the dead row held at 22 is carried rather than dropped; the eight sum to 28.
- **`size_baseline.toml` partitions exactly** on every column but one: doc 1414, comment 30 and
  blank 596 sum to the dead row's, and code sums to 710 against 707 because seven `export` statements
  stand where one did.
- **`complexity_baseline.toml` records each new file's own measurement.** No definition changed, so
  every maximum is a number the dead row already carried, now driving a smaller file.
- **ADR 0074's entry test needs five Exemptions, and they are bookkeeping.** A definition that stood
  grandfathered under the old file's row now drives its own file's maximum: `ReturnsResult` at
  cyc 18 and cog 25, `port_opt_view` at cog 16, and `prices_to_returns` at cyc 11 and arg 13. All
  five cite `a-definition-a-file-split-moved-unchanged`, the Rationale the maintainer approved for
  exactly this case in the split of `src/01_Base.jl`. The `ReturnsResult` argument-count Exemption
  follows the definition to its new path.
- **`jet_baseline.toml` partitions exactly.** The dead row's fifteen reports of the `main` run land
  as 1 on `05_CarrierViews.jl`, 1 on `09_PriceFilters.jl`, 12 on `11_PricesToReturns.jl` and 1 on
  `12_TrainTestSplit.jl`. Every new row of the `plots_ext` run is zero, as the dead row was.
- **`coverage_baseline.toml` carries `misses = 0` on every new row, which is proved rather than
  measured.** Each new file's lines are a subset of the dead file's, and that file stood terminal at
  zero misses. `lines` is context, and the new rows carry a local measurement that undercounts a
  method the ten input-data test files never compile; CI's next refresh corrects it.
- **The API pages do not move.** `docs/src/api/03_Preprocessing.md` keeps its name and renders the
  same units, by ADR 0100's rule that a page is checked by the units it renders rather than by the
  file they are declared in. ADR 0104 settled the same point for the same directory.

**An ADR that names an old path in prose keeps it.** ADR 0028, ADR 0029, ADR 0042, ADR 0102 and
ADR 0104 name files this ADR renumbers, and all five reached `main`. Each statement was true when it
was written. A relative *link* is different: ADR 0114 linked `03_Preprocessing.jl` for
`ReturnsResult` and now links `10_ReturnsResult.jl`, because a dead link is a broken record rather
than history.

## Consequences

The names of `src/03_InputData/` are now a description of the path a caller's data takes, and the
largest file in the directory is 695 lines rather than 2747. A session working on the conversion and
a session working on the split no longer queue on one file.

A reader who knows an old path finds nothing at it. The sixteen files are one subject and one map, so
no sweep ticket changes hands.

Reading order and load order agree wherever a type bound does not forbid it, and the two places they
disagree are stated above: the seam loads before the estimators that subtype it, and the carrier
before the door that builds it. Both are consequences of when Julia resolves a supertype and a return
type, not of the subject.

## Alternatives considered

- **Renumber only, and leave `03_Preprocessing.jl` whole.** It buys the reading order and leaves the
  2747-line file holding both data levels, which is half of what makes the directory hard to read.
- **Split only, and leave the numbering.** The split would land eight new files in the middle of the
  old numbering, so the price run would still be interleaved with the feature files and the new names
  would have to be chosen to fit gaps rather than to read in order.
- **Put the door at `01_`, where a reader starts.** Refused by the type system rather than by taste:
  `price_ingestion` is annotated to return `PricesResult`, so that type is defined first.
- **Give the price path and the returns path a subdirectory each.** The two share the carrier views,
  the fit/apply seam, the `Prices_RR` alias and the conversion that spans them, so the boundary would
  cut through four shared units. One directory of sixteen ordered files states the same thing without
  asking where a shared unit lives.
