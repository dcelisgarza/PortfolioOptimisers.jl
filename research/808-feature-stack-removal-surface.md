# 808 — The removal surface of the feature-distance stack

Research ticket #808 of wayfinder map #802. Written 2026-09-05.

Measured at `7e3d080c61`, the tip of `dev` on 2026-09-05 and the commit the map's ground truth
names. The sources are the files under `src/`, `ext/`, `test/`, `examples/`, `user_guide/` and
`docs/src/api/`, plus `sweep/manifest.toml` and the five baselines under `code_health/`. No Julia
was run. Every count is a regular-expression pass over every line of those files, and every site
carries its file and line. The rules are in section 1, and the scripts are described in the
appendix. A build ticket should take the tables as its checklist and re-grep only what it moves.

---

## Summary

- **The stack is 8 mechanisms and about 3,300 sites.** 32 source files name a removed symbol
  (732 lines) and 23 more read a carrier field; 17 test files name a symbol (413 lines) and 26
  read a carrier field; 2 examples (73 lines), 1 user guide page (6) and 9 hand-written API
  pages (43) name a symbol. Section 2 has the table per mechanism. The home files are `src/13_Prior/15_FeaturePrior.jl` (854 lines, 166 lines name a removed symbol),
  `src/09_Distance/05_FeatureDistance.jl` (1,462 lines, 83), `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl`
  (1,439 lines, 119), `src/13_Prior/01_Base_Prior.jl` (1,971 lines, 104) and
  `src/03_InputData/03_Preprocessing.jl` (2,604 lines, 87).
- **One source file goes whole, and every other one is rewritten in place.**
  `15_FeaturePrior.jl` defines nothing the destination keeps. It leaves seven baseline rows in
  five files (section 5), one `include` line at `src/PortfolioOptimisers.jl:193`, one API page
  (`docs/src/api/13_Prior/15_FeaturePrior.md`, 14 `@docs` entries), one catalogue group
  (`docs/capability_catalogue.jl:600`–`645`) and one export line (`:853`, seven names).
- **The eight forwarders are one shape.** Each is four lines at
  `src/13_Prior/01_Base_Prior.jl:858`, `:902`, `:943`, `:985`, `:1030`, `:1119`, `:1166` and
  `:1207`: a head that takes `z_src::Symbol = :data`, a call to `feature_matrix_picker`, and a
  forward of `Z`, `nz` and `z_diag`. Their docstrings repeat the same three-step algorithm and
  the same two `# Related` entries, so a build ticket edits 8 × 4 code lines and 8 × 7 docstring
  lines with one recipe. `average_centrality` at `:1074` forwards to the `centrality_vector`
  forwarder rather than calling the picker, and is the ninth head that takes `z_src`.
- **`z_src` has 250 sites, 159 of them in `src/`.** 74 sit in `01_Base_Prior.jl`, 25 in
  `05_FeatureDistance.jl`, 42 in the three files that declare the field
  (`04_Base_ClusteringOptimisation.jl` 13, `10_JuMPOptimiser.jl` 16, `17_NestedClustered.jl` 13),
  and 9 in the four optimisers that forward it. The three struct fields are at `04_Base_ClusteringOptimisation.jl:470`,
  `10_JuMPOptimiser.jl:720` and `17_NestedClustered.jl:517`. The example
  `16_Feature_Distance_Clustering.jl` passes it 18 times.
- **The carrier fields are the widest net.** `ReturnsResult(; nz = …, Z = …)` is built at 53
  sites (4 in `src/`, 43 in `test/`, 6 in the examples and the user guide) and `PricesResult(; nz
  = …, Z = …)` at 13. `pnl` is passed at 8 sites, all of them `ReturnsResult`. Seven
  `port_opt_view` methods slice `Z`, `nz` or `pnl`. The dotted reads are `rd.Z` 51, `rd.nz` 40,
  `rd.pnl` 77, and 219 `.Z` reads over every receiver name.
- **The tests.** The thirteen ground-truth-11 files hold 2,753 `@test` lines, not the 3,500 the
  map estimated. 187 of them name a removed name: 100 name a symbol, and 87 read or pass a
  carrier field (`.Z`, `.nz`, `Z =`, `nz =`) and nothing else. Four files carry 130 of the 187:
  `test_12e_asset_sets_features.jl` 44, `test_12c_feature_prior.jl` 31,
  `test_12d_phylogeny_features.jl` 28 and `test_13b_feature_routing.jl` 27. Three of the
  thirteen are almost untouched: `test_12b_prior_core.jl` (603 `@test`, 1),
  `test_13_phylogeny.jl` (830, 1) and `test_37_time_dependent_constraints.jl` (411, 0; its two
  lines that name `z_src`, `:771` and `:1689`, are comments). Eleven
  files outside the ground truth hold 66 more, 56 of them carrier-field reads, and
  `test_06_preprocessing.jl` alone holds 39. Section 3 has the table.
- **Doctests.** Nine `ReturnsResult` prints in `03_Preprocessing.jl` each show an `nz`, a `Z` and
  a `pnl` row (27 rows). Three `LowOrderPrior` prints in `01_Base_Prior.jl` each show a `Z` row,
  one of them nested inside the `HighOrderPrior` print at `:1662`. No doctest prints a
  `PricesResult`. Eight `UniverseSets` prints show a `zkey` row, in seven files. Section 6 lists
  every line.
- **Ground truth.** Every one of the map's 34 file:line anchors reads at `7e3d080c61` as the map
  says. One attribution is corrected in section 1.3: `01_Base_Prior.jl:1487` opens
  `LowOrderPrior`'s own `@forward_properties` block, and `HighOrderPrior` forwards `Z` through
  `forward(pr)` at `:1890`, inside the block that opens at `:1882`. Four counts are re-measured
  there.
- **Outside the census scope**, and owed anyway: `CONTEXT.md` (§2 entries at lines 100–115,
  and lines 209, 294, 314, 456), twelve ADRs that name the stack (amended, never rewritten, except
  0102 which is a draft), the catalogue, and `test/test_43_exported_abstract_type_census.jl:81`–`82`,
  which lists `AbstractFeatureMatrixEstimator`, `AbstractPhylogenyFeatureAlgorithm` and
  `AbstractFeatureValue` in the census of abstract types that are *not* exported. Section 7.

---

## 1. How the census was taken

### 1.1 Scope and kinds

Every `.jl` and `.md` file under `src/`, `ext/`, `test/`, `examples/`, `user_guide/` and
`docs/src/api/` is read line by line. A site is one line that matches one name pattern. A line
that names two symbols is two sites, one per name, so the per-mechanism totals overlap where
two names share a line. Each site is classified by where it sits:

| Kind | Rule |
| --- | --- |
| definition | A `struct`, `abstract type`, `function`, `const` or short-form head whose name is the symbol, in `src/` or `ext/`. |
| field declaration | A bare field line inside a struct, or a keyword in a constructor signature, whose name is the symbol. |
| export | An `export` line. |
| docstring | Any line inside a triple-quoted docstring block, including a jldoctest and its printed output. |
| comment | A `#` line outside a docstring. |
| code | Every other `src/` or `ext/` line. |
| dict key | A line of `src/01_Base/01_DocstringDictionaries.jl` that defines a key named for the symbol. |
| dict text | A line of that file whose description text names the symbol. |
| test assertion | A `test/` line that carries `@test`, `@test_throws`, `@test_logs` or another `@test_*` macro, but not `@testset`. |
| test code, test comment | Every other `test/` line. |
| example code, example prose | A line of `examples/` or `user_guide/`, split on whether it opens with `#`. |
| doc page | A line of `docs/src/api/`. `26_TypeHierarchy.md` is generated, so its rows are context and not work. |

A name matches on word boundaries, so `feature_matrix` does not match `feature_matrix_picker`.
Two names needed a narrower rule. `Scale` matches only when it is not followed by a lower-case
word, because the English verb opens twenty algorithm steps across the library; the type is
always backticked, called, or declared. `sets`, `sel` and `strict` are counted inside
`05_FeatureDistance.jl` only, because every optimiser and every Black-Litterman estimator has a
`sets` and a `strict` of its own; outside the home file they are counted only on a line that
also builds a `FeatureDistance`.

### 1.2 Construction sites, wide nets and limits

A construction site is found by balancing parentheses from `LowOrderPrior(`,
`ReturnsResult(`, `PricesResult(`, `FeatureDistance(` or `UniverseSets(` across up to 60
lines, and reading the keywords inside. A keyword pass on a line other than the opening one is
therefore found.

Three patterns are wide nets and are labelled as such in the tables: `any .Z read`, `any .nz
read` and `Z keyword / field`. The last one matches `Z =` and `Z::` on any line, and so also
catches the positional `Z::ArrNum` argument of every kernel in `05_FeatureDistance.jl` and the
`Z::Arr3Num` exposure tensor of `08_Moments/38_CrossSectionalRegression.jl`. Both are sites the
carrier rewrite touches, so they stay. Five files whose `Z` is a linkage matrix or a local are
excluded from that net: `11_Phylogeny/11_DBHTDendrogram.jl`, `11_Phylogeny/12_DBHTClustering.jl`,
`08_Moments/21_Base_Regression.jl`, `14_UncertaintySets/06_CalibrationRules.jl` and
`08_Moments/10_Histogram.jl`.

Three limits. A `@test` that spans several lines is counted once, on the line that carries the
macro, and only when a name sits on that line. A `[`Proximity`](@ref)` in a docstring of
`11_Phylogeny/` is a site, because the reference dangles when the type goes. A prose mention on
an example line is a site for the same reason.

### 1.3 The ground-truth anchors, re-read at `7e3d080c61`

Every anchor the map gives reads as stated. The list below is the line at each anchor.

- `src/09_Distance/05_FeatureDistance.jl:531`: `@propagatable @concrete struct FeatureDistance <: AbstractDistanceEstimator`
- `src/09_Distance/05_FeatureDistance.jl:830`: `function select_features(::FeatureDistance{<:Any, <:Any, <:Any, Nothing}, Z::ArrNum,`
- `src/09_Distance/05_FeatureDistance.jl:1410`: `z_src::Symbol = :none, kwargs...)`
- `src/09_Distance/05_FeatureDistance.jl:1456`: `nz::Option{<:VecStr} = nothing, z_src::Symbol = :none, kwargs...)`
- `src/03_InputData/03_Preprocessing.jl:768`: `pnl`
- `src/03_InputData/03_Preprocessing.jl:1053`: `pnl = port_opt_view(rd.pnl, i)`
- `src/03_InputData/03_Preprocessing.jl:1076`: `pnl = port_opt_view(rd.pnl, i, j, k)`
- `src/03_InputData/03_Preprocessing.jl:1662`: `pnl = isnothing(pnl) ? nothing : port_opt_view(pnl, rows, acols)`
- `src/03_InputData/01_AssetPanel.jl:518`: `@concrete struct AssetPanel <: AbstractResult`
- `src/03_InputData/01_AssetPanel.jl:425`: `@concrete struct PanelField <: AbstractResult`
- `src/03_InputData/01_AssetPanel.jl:73`: `struct NumericPanelField <: AbstractPanelFieldKind end`
- `src/03_InputData/01_AssetPanel.jl:161`: `@concrete struct CategoricalPanelField <: AbstractPanelFieldKind`
- `src/03_InputData/01_AssetPanel.jl:221`: `@concrete struct TensorPanelField <: AbstractPanelFieldKind`
- `src/03_InputData/01_AssetPanel.jl:636`: `function panel_field(pnl::AssetPanel, name::AbstractString)::PanelField`
- `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl:131`: `function panel_field_values(rd::ReturnsResult, name::AbstractString)::Matrix{<:Real}`
- `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl:138`: `Z = rd.Z`
- `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl:202`: `return cross_sectional_groups(rd.pnl, rd.Z, group)`
- `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl:278`: `Z = rd.Z`
- `src/13_Prior/01_Base_Prior.jl:1398`: `Z`
- `src/13_Prior/01_Base_Prior.jl:1487`: `@forward_properties LowOrderPrior begin`
- `src/13_Prior/01_Base_Prior.jl:816`: `function feature_matrix_picker(pr::Pr_RR, rd::Option{<:ReturnsResult}, z_src::Symbol)`
- `src/13_Prior/01_Base_Prior.jl:862`: `Z, nz, z_diag = feature_matrix_picker(pr, rd, z_src)`
- `src/13_Prior/01_Base_Prior.jl:1211`: `Z, nz, z_diag = feature_matrix_picker(pr, rd, z_src)`
- `src/20_Optimisation/04_Base_ClusteringOptimisation.jl:470`: `z_src`
- `src/20_Optimisation/10_JuMPOptimiser.jl:720`: `z_src`
- `src/20_Optimisation/17_NestedClustered.jl:517`: `z_src`
- `src/13_Prior/15_FeaturePrior.jl:75`: `function feature_matrix(ze::MatNum_Arr3Num, ::AbstractPriorResult, args...; kwargs...)`
- `src/13_Prior/15_FeaturePrior.jl:603`: `function feature_matrix(ze::Arr3Num, ::AbstractPriorResult, X::MatNum, args...; kwargs...)`
- `src/13_Prior/15_FeaturePrior.jl:111`: `struct RegressionFeatures <: AbstractFeatureMatrixEstimator end`
- `src/13_Prior/15_FeaturePrior.jl:467`: `@propagatable @concrete struct PhylogenyFeatures <: AbstractFeatureMatrixEstimator`
- `src/13_Prior/15_FeaturePrior.jl:200`: `@concrete struct Proximity <: AbstractPhylogenyFeatureAlgorithm`
- `src/13_Prior/15_FeaturePrior.jl:564`: `@concrete struct AssetSetsFeatures <: AbstractFeatureMatrixEstimator`
- `src/13_Prior/15_FeaturePrior.jl:637`: `function feature_estimator_view(ze::AbstractFeatureMatrixEstimator, i, args...)`
- `src/22_Preselection.jl:1298`: `clr = clusterise(alg.cle, rd.X; Z = rd.Z, nz = rd.nz, z_src = :data_only)`

Four things to correct or refine in the map's ground truth:

- **Ground truth 5.** `:1487` opens `@forward_properties LowOrderPrior begin`, the block that
  derives `f_mu`, `f_sigma` and `original_X`. It does not forward `Z`. `HighOrderPrior`'s block
  opens at `:1882` and forwards every `LowOrderPrior` field, `Z` included, through `forward(pr)`
  at `:1890`. So `HighOrderPrior` gains and loses `Z` with `LowOrderPrior` and needs no edit of
  its own, but its doctest at `:1662` prints the nested carrier and loses the `Z` row at `:1675`.
- **Ground truth 6.** `z_src` has 159 lines in 14 files of `src/`, counting the dictionary key
  at `01_DocstringDictionaries.jl:572` and the one prose line in `03_Preprocessing.jl:2107`. The
  map's 158 in 13 files is the same set without the dictionary.
- **Ground truth 8 and 9.** `zkey` and `feature_universe` have 87 + 19 = 106 sites, not 70;
  `Scale` and `AbstractFeatureValue` have 38 + 22 = 60, not 31; `collapse_feature_matrix`,
  `features_are_assets`, `feature_matrix_view` and `feature_row_indices` have 25 + 41 + 34 + 27 =
  127, not 97. The difference is docstring cross-references and test lines, which the map's
  count left out and this census keeps.
- **Ground truth 11.** The thirteen test files hold 2,753 `@test` lines, not about 3,500.

## 2. The tables, one mechanism at a time

Each mechanism has a per-name summary and a site table. A row of the site table is one name,
one file and one kind, with every line listed.

### Mechanism 1: `FeaturePrior`, the producers, and the taxonomy route

821 sites. The producers live in `src/13_Prior/15_FeaturePrior.jl`: `AbstractFeatureMatrixEstimator`
(`:17`), the literal-matrix producer (`feature_matrix` on a bare matrix at `:75` and `:603`, and
its `feature_estimator_view` at `:640`), `RegressionFeatures` (`:111`), `AbstractPhylogenyFeatureAlgorithm`
(`:137`), `Proximity` (`:200`), `phylogeny_features` (`:274`, `:277`, `:345`), `_proximity_features`
(`:324`), `PhylogenyFeatures` (`:467`), `AssetSetsFeatures` (`:564`), `feature_estimator_view`
(`:637`), `FeaturePrior` (`:737`), and the export at `:853`. The taxonomy route lives in
`src/12_ConstraintGeneration/06_AssetSetsMatrix.jl`: `AbstractFeatureValue` (`:331`), `Scale` (`:386`),
`asset_sets_features` (`:561` for the key-list contract, `:1226` for the graded program),
`feature_program_candidates` (`:595`), `feature_unknown_name_msg` (`:778`), `asset_sets_feature_names`
(`:1289`, `:1383`), `taxonomy_feature_names` (`:1332`) and the export at `:1438`.
`carrier_feature_names` is at `src/13_Prior/01_Base_Prior.jl:769` and `:772`.

Every `Proximity` site under `src/11_Phylogeny/` is a `[`Proximity`](@ref)` in a `# Related`
list or a sentence of the separation family's docstrings: seventeen in `01_Base_Phylogeny.jl`,
two each in `17_NetworkEstimator.jl` and `20_Separation.jl`, one in `22_PhylogenyMatrix.jl`.
The separation and decay families stay; only the references go.

### Mechanism 1: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `asset_sets_features` | 2 | 1 | 51 | 3 | 0 | 1 | 0 | 18 | 48 | 6 | 2 | 132 |
| `asset_sets_feature_names` | 2 | 0 | 15 | 4 | 0 | 0 | 0 | 5 | 7 | 3 | 1 | 37 |
| `taxonomy_feature_names` | 1 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 1 | 12 |
| `FeaturePrior` | 3 | 1 | 31 | 13 | 0 | 0 | 0 | 10 | 59 | 12 | 4 | 133 |
| `carrier_feature_names` | 2 | 0 | 4 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 10 |
| `AbstractFeatureMatrixEstimator` | 1 | 0 | 8 | 7 | 0 | 0 | 0 | 2 | 3 | 2 | 2 | 25 |
| `feature_matrix` | 6 | 1 | 12 | 2 | 0 | 0 | 0 | 2 | 4 | 2 | 1 | 30 |
| `RegressionFeatures` | 1 | 1 | 16 | 2 | 0 | 0 | 0 | 7 | 19 | 4 | 2 | 52 |
| `PhylogenyFeatures` | 3 | 1 | 15 | 3 | 0 | 0 | 0 | 1 | 23 | 10 | 2 | 58 |
| `phylogeny_features` | 3 | 0 | 14 | 2 | 0 | 0 | 0 | 10 | 22 | 5 | 1 | 57 |
| `AssetSetsFeatures` | 4 | 0 | 11 | 7 | 0 | 0 | 0 | 3 | 19 | 5 | 2 | 51 |
| `Proximity` | 3 | 1 | 35 | 4 | 0 | 0 | 0 | 15 | 37 | 10 | 2 | 107 |
| `AbstractPhylogenyFeatureAlgorithm` | 1 | 0 | 5 | 4 | 0 | 0 | 0 | 1 | 3 | 0 | 2 | 16 |
| `_proximity_features` | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 1 | 7 |
| `Scale` | 3 | 0 | 17 | 1 | 0 | 0 | 0 | 7 | 8 | 0 | 2 | 38 |
| `feature_estimator_view` | 2 | 0 | 6 | 1 | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 13 |
| `AbstractFeatureValue` | 1 | 0 | 13 | 3 | 0 | 0 | 0 | 1 | 2 | 0 | 2 | 22 |
| `feature_program_candidates` | 1 | 0 | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 |
| `feature_unknown_name_msg` | 1 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 8 |

Total: 821.

### Mechanism 1: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `asset_sets_features` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 16 |
| `asset_sets_features` | `docs/src/api/22_AssetSelection.md` | doc page | 1 | 41 |
| `asset_sets_features` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 4 | 93, 100, 256, 860 |
| `asset_sets_features` | `src/01_Base/01_DocstringDictionaries.jl` | dict text | 1 | 751 |
| `asset_sets_features` | `src/03_InputData/01_AssetPanel.jl` | docstring | 2 | 125, 158 |
| `asset_sets_features` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 661 |
| `asset_sets_features` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 2 | 438, 526 |
| `asset_sets_features` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | docstring | 4 | 743, 766, 877, 896 |
| `asset_sets_features` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 1231 |
| `asset_sets_features` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 2 | 561, 1226 |
| `asset_sets_features` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 36 | 167, 289, 329, 383, 420, 439, 447, 464, 479, 537, 592, 627, 654, 683, 715, 742, 773, 799, 839, 891, 955, 1016, 1110, 1148, 1208, 1243, 1255, 1265, 1285, 1300, 1327, 1339, 1359, 1378, 1394, 1415 |
| `asset_sets_features` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | export | 1 | 1438 |
| `asset_sets_features` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 598 |
| `asset_sets_features` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 6 | 52, 496, 522, 526, 536, 556 |
| `asset_sets_features` | `src/22_Preselection.jl` | docstring | 1 | 1225 |
| `asset_sets_features` | `test/test_12e_asset_sets_features.jl` | test assertion | 18 | 91, 94, 122, 182, 396, 403, 406, 408, 416, 419, 439, 627, 683, 692, 724, 759, 763, 772 |
| `asset_sets_features` | `test/test_12e_asset_sets_features.jl` | test code | 44 | 46, 47, 98, 130, 131, 145, 150, 155, 170, 248, 340, 390, 395, 401, 414, 448, 465, 466, 474, 475, 479, 486, 497, 501, 506, 507, 532, 533, 557, 558, 583, 626, 649, 677, 691, 693, 710, 714, 723, 725, 739, 754, 771, 773 |
| `asset_sets_features` | `test/test_12e_asset_sets_features.jl` | test comment | 1 | 253 |
| `asset_sets_features` | `test/test_13b_feature_routing.jl` | test code | 1 | 366 |
| `asset_sets_features` | `test/test_13b_feature_routing.jl` | test comment | 1 | 368 |
| `asset_sets_features` | `test/test_13e_feature_preselection.jl` | test code | 1 | 39 |
| `asset_sets_features` | `user_guide/02_Optimisers.jl` | example code | 2 | 149, 172 |
| `asset_sets_feature_names` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 17 |
| `asset_sets_feature_names` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 2 | 94, 101 |
| `asset_sets_feature_names` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 661 |
| `asset_sets_feature_names` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 2 | 438, 654 |
| `asset_sets_feature_names` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | docstring | 1 | 897 |
| `asset_sets_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 3 | 1293, 1385, 1439 |
| `asset_sets_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 2 | 1289, 1383 |
| `asset_sets_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 12 | 130, 167, 527, 543, 555, 559, 1221, 1275, 1300, 1326, 1337, 1369 |
| `asset_sets_feature_names` | `test/test_12e_asset_sets_features.jl` | test assertion | 5 | 444, 445, 446, 447, 455 |
| `asset_sets_feature_names` | `test/test_12e_asset_sets_features.jl` | test code | 4 | 79, 156, 256, 632 |
| `asset_sets_feature_names` | `test/test_13b_feature_routing.jl` | test code | 1 | 367 |
| `asset_sets_feature_names` | `test/test_13b_feature_routing.jl` | test comment | 1 | 368 |
| `asset_sets_feature_names` | `test/test_13e_feature_preselection.jl` | test code | 1 | 38 |
| `asset_sets_feature_names` | `user_guide/02_Optimisers.jl` | example code | 1 | 171 |
| `taxonomy_feature_names` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 18 |
| `taxonomy_feature_names` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 702 |
| `taxonomy_feature_names` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 4 | 438, 669, 674, 694 |
| `taxonomy_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 1293 |
| `taxonomy_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 1 | 1332 |
| `taxonomy_feature_names` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 1 | 1296 |
| `taxonomy_feature_names` | `test/test_13b_feature_routing.jl` | test code | 2 | 371, 372 |
| `taxonomy_feature_names` | `test/test_13b_feature_routing.jl` | test comment | 1 | 369 |
| `FeaturePrior` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 3 | 15, 17, 18 |
| `FeaturePrior` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 356 |
| `FeaturePrior` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 11 | 211, 212, 217, 222, 228, 232, 281, 302, 360, 718, 739 |
| `FeaturePrior` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 1403 |
| `FeaturePrior` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 1 | 1362 |
| `FeaturePrior` | `src/13_Prior/01_Base_Prior.jl` | docstring | 3 | 1274, 1278, 1346 |
| `FeaturePrior` | `src/13_Prior/03_FactorPrior.jl` | comment | 1 | 423 |
| `FeaturePrior` | `src/13_Prior/03_FactorPrior.jl` | docstring | 1 | 47 |
| `FeaturePrior` | `src/13_Prior/06_BlackLittermanPrior.jl` | comment | 1 | 726 |
| `FeaturePrior` | `src/13_Prior/08_FactorBlackLittermanPrior.jl` | docstring | 1 | 39 |
| `FeaturePrior` | `src/13_Prior/15_FeaturePrior.jl` | code | 9 | 597, 605, 761, 762, 766, 789, 790, 835, 849 |
| `FeaturePrior` | `src/13_Prior/15_FeaturePrior.jl` | comment | 1 | 486 |
| `FeaturePrior` | `src/13_Prior/15_FeaturePrior.jl` | definition | 3 | 737, 750, 759 |
| `FeaturePrior` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 25 | 6, 14, 24, 70, 93, 106, 461, 496, 526, 542, 561, 612, 632, 648, 656, 660, 673, 680, 682, 693, 772, 781, 785, 795, 830 |
| `FeaturePrior` | `src/13_Prior/15_FeaturePrior.jl` | export | 1 | 853 |
| `FeaturePrior` | `test/test_12b_prior_core.jl` | test code | 1 | 1391 |
| `FeaturePrior` | `test/test_12c_feature_prior.jl` | test assertion | 4 | 193, 306, 327, 329 |
| `FeaturePrior` | `test/test_12c_feature_prior.jl` | test code | 21 | 40, 48, 58, 67, 73, 81, 86, 99, 102, 112, 121, 128, 150, 189, 198, 205, 247, 251, 257, 290, 313 |
| `FeaturePrior` | `test/test_12c_feature_prior.jl` | test comment | 2 | 4, 325 |
| `FeaturePrior` | `test/test_12d_phylogeny_features.jl` | test assertion | 3 | 125, 127, 173 |
| `FeaturePrior` | `test/test_12d_phylogeny_features.jl` | test code | 11 | 118, 138, 149, 158, 170, 179, 188, 219, 229, 239, 732 |
| `FeaturePrior` | `test/test_12e_asset_sets_features.jl` | test assertion | 1 | 206 |
| `FeaturePrior` | `test/test_12e_asset_sets_features.jl` | test code | 10 | 180, 196, 205, 211, 242, 258, 599, 604, 608, 635 |
| `FeaturePrior` | `test/test_12e_asset_sets_features.jl` | test comment | 2 | 178, 254 |
| `FeaturePrior` | `test/test_12g_forwarding_rule.jl` | test assertion | 1 | 400 |
| `FeaturePrior` | `test/test_12g_forwarding_rule.jl` | test code | 3 | 165, 214, 415 |
| `FeaturePrior` | `test/test_12g_forwarding_rule.jl` | test comment | 1 | 211 |
| `FeaturePrior` | `test/test_13b_feature_routing.jl` | test code | 1 | 35 |
| `FeaturePrior` | `test/test_13c_feature_views.jl` | test code | 6 | 337, 394, 401, 402, 446, 469 |
| `FeaturePrior` | `test/test_13e_feature_preselection.jl` | test assertion | 1 | 140 |
| `FeaturePrior` | `test/test_13e_feature_preselection.jl` | test comment | 1 | 128 |
| `FeaturePrior` | `user_guide/02_Optimisers.jl` | example code | 1 | 196 |
| `carrier_feature_names` | `docs/src/api/13_Prior/01_Base_Prior.md` | doc page | 1 | 34 |
| `carrier_feature_names` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 820 |
| `carrier_feature_names` | `src/13_Prior/01_Base_Prior.jl` | definition | 2 | 769, 772 |
| `carrier_feature_names` | `src/13_Prior/01_Base_Prior.jl` | docstring | 4 | 738, 739, 789, 806 |
| `carrier_feature_names` | `test/test_53_panel_carrier_census.jl` | test assertion | 1 | 84 |
| `carrier_feature_names` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 17 |
| `AbstractFeatureMatrixEstimator` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 6 |
| `AbstractFeatureMatrixEstimator` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 276 |
| `AbstractFeatureMatrixEstimator` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 2 | 210, 547 |
| `AbstractFeatureMatrixEstimator` | `src/13_Prior/15_FeaturePrior.jl` | code | 7 | 111, 467, 564, 605, 637, 751, 760 |
| `AbstractFeatureMatrixEstimator` | `src/13_Prior/15_FeaturePrior.jl` | definition | 1 | 17 |
| `AbstractFeatureMatrixEstimator` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 8 | 6, 69, 104, 457, 555, 609, 658, 728 |
| `AbstractFeatureMatrixEstimator` | `test/test_12c_feature_prior.jl` | test code | 1 | 9 |
| `AbstractFeatureMatrixEstimator` | `test/test_12d_phylogeny_features.jl` | test assertion | 1 | 108 |
| `AbstractFeatureMatrixEstimator` | `test/test_12e_asset_sets_features.jl` | test assertion | 1 | 176 |
| `AbstractFeatureMatrixEstimator` | `test/test_43_exported_abstract_type_census.jl` | test code | 2 | 8, 81 |
| `feature_matrix` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 7 |
| `feature_matrix` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 1 | 550 |
| `feature_matrix` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 966 |
| `feature_matrix` | `src/13_Prior/15_FeaturePrior.jl` | code | 2 | 605, 843 |
| `feature_matrix` | `src/13_Prior/15_FeaturePrior.jl` | definition | 6 | 74, 75, 112, 484, 594, 603 |
| `feature_matrix` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 12 | 8, 12, 19, 20, 105, 460, 536, 560, 688, 730, 807, 831 |
| `feature_matrix` | `src/13_Prior/15_FeaturePrior.jl` | export | 1 | 853 |
| `feature_matrix` | `test/test_12c_feature_prior.jl` | test code | 1 | 12 |
| `feature_matrix` | `test/test_12e_asset_sets_features.jl` | test comment | 1 | 597 |
| `feature_matrix` | `test/test_12i_cross_sectional_factor_carrier.jl` | test assertion | 2 | 173, 176 |
| `feature_matrix` | `test/test_12i_cross_sectional_factor_carrier.jl` | test code | 2 | 174, 177 |
| `RegressionFeatures` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 8 |
| `RegressionFeatures` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 279 |
| `RegressionFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 3 | 271, 281, 324 |
| `RegressionFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 269 |
| `RegressionFeatures` | `src/13_Prior/03_FactorPrior.jl` | comment | 1 | 423 |
| `RegressionFeatures` | `src/13_Prior/03_FactorPrior.jl` | docstring | 1 | 47 |
| `RegressionFeatures` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 112 |
| `RegressionFeatures` | `src/13_Prior/15_FeaturePrior.jl` | definition | 1 | 111 |
| `RegressionFeatures` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 15 | 13, 20, 28, 40, 71, 83, 93, 98, 99, 522, 648, 693, 696, 729, 800 |
| `RegressionFeatures` | `src/13_Prior/15_FeaturePrior.jl` | export | 1 | 853 |
| `RegressionFeatures` | `test/test_12b_prior_core.jl` | test code | 1 | 1391 |
| `RegressionFeatures` | `test/test_12c_feature_prior.jl` | test assertion | 2 | 327, 329 |
| `RegressionFeatures` | `test/test_12c_feature_prior.jl` | test code | 9 | 56, 58, 68, 75, 84, 87, 194, 195, 257 |
| `RegressionFeatures` | `test/test_12d_phylogeny_features.jl` | test assertion | 2 | 125, 173 |
| `RegressionFeatures` | `test/test_12d_phylogeny_features.jl` | test code | 2 | 158, 174 |
| `RegressionFeatures` | `test/test_12e_asset_sets_features.jl` | test comment | 1 | 215 |
| `RegressionFeatures` | `test/test_12g_forwarding_rule.jl` | test assertion | 1 | 400 |
| `RegressionFeatures` | `test/test_12g_forwarding_rule.jl` | test code | 1 | 415 |
| `RegressionFeatures` | `test/test_12i_cross_sectional_factor_carrier.jl` | test assertion | 2 | 173, 176 |
| `RegressionFeatures` | `test/test_12i_cross_sectional_factor_carrier.jl` | test code | 2 | 174, 177 |
| `RegressionFeatures` | `test/test_12i_cross_sectional_factor_carrier.jl` | test comment | 1 | 172 |
| `RegressionFeatures` | `test/test_13b_feature_routing.jl` | test code | 1 | 35 |
| `RegressionFeatures` | `test/test_13c_feature_views.jl` | test code | 1 | 337 |
| `PhylogenyFeatures` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 13 |
| `PhylogenyFeatures` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 278 |
| `PhylogenyFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 7 | 291, 303, 325, 337, 358, 622, 926 |
| `PhylogenyFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 289 |
| `PhylogenyFeatures` | `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` | example code | 1 | 163 |
| `PhylogenyFeatures` | `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` | example prose | 1 | 558 |
| `PhylogenyFeatures` | `src/11_Phylogeny/01_Base_Phylogeny.jl` | docstring | 2 | 564, 591 |
| `PhylogenyFeatures` | `src/13_Prior/15_FeaturePrior.jl` | code | 3 | 481, 482, 484 |
| `PhylogenyFeatures` | `src/13_Prior/15_FeaturePrior.jl` | definition | 3 | 467, 476, 480 |
| `PhylogenyFeatures` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 13 | 45, 121, 127, 132, 197, 220, 265, 368, 371, 423, 424, 522, 633 |
| `PhylogenyFeatures` | `src/13_Prior/15_FeaturePrior.jl` | export | 1 | 853 |
| `PhylogenyFeatures` | `test/test_12d_phylogeny_features.jl` | test assertion | 1 | 89 |
| `PhylogenyFeatures` | `test/test_12d_phylogeny_features.jl` | test code | 16 | 90, 91, 106, 107, 118, 138, 149, 170, 179, 193, 195, 196, 219, 229, 238, 732 |
| `PhylogenyFeatures` | `test/test_12d_phylogeny_features.jl` | test comment | 2 | 4, 164 |
| `PhylogenyFeatures` | `test/test_13_phylogeny.jl` | test code | 1 | 1772 |
| `PhylogenyFeatures` | `test/test_13c_feature_views.jl` | test code | 3 | 392, 401, 403 |
| `PhylogenyFeatures` | `test/test_13e_feature_preselection.jl` | test comment | 1 | 82 |
| `phylogeny_features` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 12 |
| `phylogeny_features` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 5 | 634, 787, 789, 814, 816 |
| `phylogeny_features` | `src/11_Phylogeny/01_Base_Phylogeny.jl` | docstring | 3 | 514, 547, 857 |
| `phylogeny_features` | `src/11_Phylogeny/20_Separation.jl` | docstring | 1 | 443 |
| `phylogeny_features` | `src/13_Prior/15_FeaturePrior.jl` | code | 2 | 489, 854 |
| `phylogeny_features` | `src/13_Prior/15_FeaturePrior.jl` | definition | 3 | 274, 277, 345 |
| `phylogeny_features` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 10 | 47, 121, 133, 213, 215, 287, 309, 318, 418, 459 |
| `phylogeny_features` | `test/test_12d_phylogeny_features.jl` | test assertion | 9 | 50, 120, 182, 386, 488, 489, 740, 748, 759 |
| `phylogeny_features` | `test/test_12d_phylogeny_features.jl` | test code | 15 | 12, 13, 14, 15, 240, 380, 458, 461, 494, 498, 508, 509, 517, 725, 742 |
| `phylogeny_features` | `test/test_12d_phylogeny_features.jl` | test comment | 1 | 265 |
| `phylogeny_features` | `test/test_13_phylogeny.jl` | test assertion | 1 | 2765 |
| `phylogeny_features` | `test/test_13_phylogeny.jl` | test code | 4 | 2140, 2144, 2160, 2767 |
| `phylogeny_features` | `test/test_13e_feature_preselection.jl` | test code | 2 | 83, 116 |
| `AssetSetsFeatures` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 14 |
| `AssetSetsFeatures` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 277 |
| `AssetSetsFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 4 | 227, 233, 324, 924 |
| `AssetSetsFeatures` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 225 |
| `AssetSetsFeatures` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 4 | 447, 465, 557, 1223 |
| `AssetSetsFeatures` | `src/13_Prior/15_FeaturePrior.jl` | code | 7 | 574, 579, 588, 589, 594, 597, 854 |
| `AssetSetsFeatures` | `src/13_Prior/15_FeaturePrior.jl` | definition | 4 | 564, 573, 578, 585 |
| `AssetSetsFeatures` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 7 | 49, 408, 496, 504, 507, 547, 548 |
| `AssetSetsFeatures` | `test/test_12e_asset_sets_features.jl` | test assertion | 3 | 423, 592, 595 |
| `AssetSetsFeatures` | `test/test_12e_asset_sets_features.jl` | test code | 19 | 5, 132, 133, 146, 173, 174, 196, 210, 242, 258, 422, 424, 502, 588, 589, 594, 605, 608, 635 |
| `Proximity` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 10 |
| `Proximity` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 609 |
| `Proximity` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 8 | 307, 343, 359, 634, 787, 789, 814, 816 |
| `Proximity` | `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` | example code | 1 | 165 |
| `Proximity` | `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` | example prose | 1 | 557 |
| `Proximity` | `src/11_Phylogeny/01_Base_Phylogeny.jl` | docstring | 12 | 138, 158, 343, 385, 450, 590, 633, 806, 826, 891, 915, 953 |
| `Proximity` | `src/11_Phylogeny/17_NetworkEstimator.jl` | docstring | 2 | 95, 101 |
| `Proximity` | `src/11_Phylogeny/20_Separation.jl` | docstring | 3 | 124, 191, 443 |
| `Proximity` | `src/11_Phylogeny/22_PhylogenyMatrix.jl` | docstring | 1 | 104 |
| `Proximity` | `src/13_Prior/15_FeaturePrior.jl` | code | 4 | 210, 324, 345, 481 |
| `Proximity` | `src/13_Prior/15_FeaturePrior.jl` | definition | 3 | 200, 205, 209 |
| `Proximity` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 17 | 123, 131, 151, 173, 175, 182, 183, 213, 236, 240, 267, 287, 305, 319, 370, 423, 451 |
| `Proximity` | `src/13_Prior/15_FeaturePrior.jl` | export | 1 | 853 |
| `Proximity` | `test/test_12d_phylogeny_features.jl` | test assertion | 14 | 50, 110, 111, 112, 182, 386, 467, 468, 488, 489, 716, 740, 748, 759 |
| `Proximity` | `test/test_12d_phylogeny_features.jl` | test code | 29 | 13, 14, 15, 117, 193, 195, 240, 380, 458, 461, 466, 471, 494, 498, 508, 509, 517, 549, 559, 575, 652, 653, 672, 680, 706, 720, 721, 722, 742 |
| `Proximity` | `test/test_12d_phylogeny_features.jl` | test comment | 3 | 284, 410, 473 |
| `Proximity` | `test/test_13_phylogeny.jl` | test assertion | 1 | 2765 |
| `Proximity` | `test/test_13_phylogeny.jl` | test code | 3 | 2145, 2160, 2767 |
| `Proximity` | `test/test_13e_feature_preselection.jl` | test code | 2 | 83, 116 |
| `AbstractPhylogenyFeatureAlgorithm` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 9 |
| `AbstractPhylogenyFeatureAlgorithm` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 608 |
| `AbstractPhylogenyFeatureAlgorithm` | `src/13_Prior/15_FeaturePrior.jl` | code | 4 | 200, 277, 476, 481 |
| `AbstractPhylogenyFeatureAlgorithm` | `src/13_Prior/15_FeaturePrior.jl` | definition | 1 | 137 |
| `AbstractPhylogenyFeatureAlgorithm` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 5 | 190, 215, 266, 370, 458 |
| `AbstractPhylogenyFeatureAlgorithm` | `test/test_12d_phylogeny_features.jl` | test assertion | 1 | 111 |
| `AbstractPhylogenyFeatureAlgorithm` | `test/test_12d_phylogeny_features.jl` | test code | 1 | 113 |
| `AbstractPhylogenyFeatureAlgorithm` | `test/test_43_exported_abstract_type_census.jl` | test code | 2 | 8, 81 |
| `_proximity_features` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 11 |
| `_proximity_features` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 353 |
| `_proximity_features` | `src/13_Prior/15_FeaturePrior.jl` | definition | 1 | 324 |
| `_proximity_features` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 245 |
| `_proximity_features` | `test/test_12d_phylogeny_features.jl` | test code | 3 | 549, 559, 578 |
| `Scale` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 13 |
| `Scale` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 555 |
| `Scale` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 397 |
| `Scale` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 3 | 386, 391, 396 |
| `Scale` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 16 | 289, 327, 337, 353, 361, 372, 384, 405, 414, 419, 438, 800, 1018, 1176, 1208, 1219 |
| `Scale` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 558 |
| `Scale` | `test/test_12e_asset_sets_features.jl` | test assertion | 7 | 379, 380, 381, 382, 406, 408, 683 |
| `Scale` | `test/test_12e_asset_sets_features.jl` | test code | 7 | 324, 383, 384, 390, 395, 401, 556 |
| `Scale` | `test/test_12e_asset_sets_features.jl` | test comment | 1 | 363 |
| `feature_estimator_view` | `docs/src/api/13_Prior/15_FeaturePrior.md` | doc page | 1 | 16 |
| `feature_estimator_view` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 791 |
| `feature_estimator_view` | `src/13_Prior/15_FeaturePrior.jl` | definition | 2 | 637, 640 |
| `feature_estimator_view` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 6 | 609, 610, 676, 774, 779, 786 |
| `feature_estimator_view` | `test/test_12d_phylogeny_features.jl` | test comment | 1 | 166 |
| `feature_estimator_view` | `test/test_12e_asset_sets_features.jl` | test assertion | 1 | 217 |
| `feature_estimator_view` | `test/test_12e_asset_sets_features.jl` | test comment | 1 | 215 |
| `AbstractFeatureValue` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 12 |
| `AbstractFeatureValue` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 554 |
| `AbstractFeatureValue` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 3 | 386, 441, 694 |
| `AbstractFeatureValue` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 1 | 331 |
| `AbstractFeatureValue` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 13 | 291, 295, 299, 313, 381, 405, 409, 418, 429, 437, 826, 877, 1176 |
| `AbstractFeatureValue` | `test/test_12e_asset_sets_features.jl` | test assertion | 1 | 382 |
| `AbstractFeatureValue` | `test/test_43_exported_abstract_type_census.jl` | test code | 2 | 9, 82 |
| `feature_program_candidates` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 19 |
| `feature_program_candidates` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 1234 |
| `feature_program_candidates` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 1 | 595 |
| `feature_program_candidates` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 10 | 567, 757, 764, 774, 830, 882, 946, 1003, 1101, 1163 |
| `feature_unknown_name_msg` | `docs/src/api/12_ConstraintGeneration/06_AssetSetsMatrix.md` | doc page | 1 | 25 |
| `feature_unknown_name_msg` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 3 | 847, 968, 1060 |
| `feature_unknown_name_msg` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | definition | 1 | 778 |
| `feature_unknown_name_msg` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 3 | 753, 817, 938 |

Sites in the table: 821.

### Mechanism 1: the literal-matrix producer

- `src/13_Prior/15_FeaturePrior.jl:75`: `function feature_matrix(ze::MatNum_Arr3Num, ::AbstractPriorResult, args...; kwargs...)`
- `src/13_Prior/15_FeaturePrior.jl:603`: `function feature_matrix(ze::Arr3Num, ::AbstractPriorResult, X::MatNum, args...; kwargs...)`
- `src/13_Prior/15_FeaturePrior.jl:610`: `feature_estimator_view(ze::MatNum_Arr3Num, i, args...)`
- `src/13_Prior/15_FeaturePrior.jl:640`: `function feature_estimator_view(ze::MatNum_Arr3Num, i, args...)`

### Mechanism 2: `LowOrderPrior.Z` and its forwarding

109 sites, plus 12 construction sites that pass `Z`. The field is declared at
`src/13_Prior/01_Base_Prior.jl:1398`, its `field_dict` key `:Z_prior` at
`01_DocstringDictionaries.jl:234` has that one user and is deleted with it, the inner constructor
takes it at `:1404` and checks it with `check_feature_matrix` at `:1459`, the keyword
constructor takes it at `:1471`, and `port_opt_view` cuts it at `:1535`. The prose that
explains the field is `:1239`–`:1304`. `HighOrderPrior` forwards it through `forward(pr)` at
`:1890`.

Seven `src/` construction sites pass `Z`: the view at `:1532`, and one per wrapping prior that
rebuilds the carrier by hand rather than through `forward_prior` —
`09_AugmentedBlackLittermanPrior.jl:553`, `11_MeucciEntropyPoolingPrior.jl:840` and `:980`,
`12_EntropyPoolingPrior.jl:3649` and `:3739`, `13_OpinionPoolingPrior.jl:533`. The producer itself
writes the field through `forward_prior(pr; Z = Z)` at `15_FeaturePrior.jl:846`, the one
`forward_prior` call in `src/` that passes `Z`. The other 66 `LowOrderPrior(` sites pass no `Z` and
change only if the keyword is removed from the signature they call.

### Mechanism 2: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `Z (in 01_Base_Prior.jl)` | `src/13_Prior/01_Base_Prior.jl` | code | 25 | 818, 819, 820, 821, 862, 863, 905, 906, 947, 948, 989, 990, 1034, 1035, 1123, 1124, 1170, 1171, 1211, 1212, 1404, 1459, 1462, 1472, 1535 |
| `Z (in 01_Base_Prior.jl)` | `src/13_Prior/01_Base_Prior.jl` | docstring | 40 | 59, 340, 342, 521, 743, 745, 782, 788, 789, 791, 805, 806, 833, 834, 876, 877, 918, 919, 960, 961, 1003, 1004, 1091, 1092, 1137, 1138, 1183, 1184, 1239, 1246, 1260, 1274, 1278, 1280, 1304, 1313, 1334, 1501, 1509, 1675 |
| `Z (in 01_Base_Prior.jl)` | `src/13_Prior/01_Base_Prior.jl` | field declaration | 2 | 1398, 1471 |
| `Z_prior dict key` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 234 |
| `pr.Z / fpr.Z / prior .Z` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 2 | 1234, 1239 |
| `pr.Z / fpr.Z / prior .Z` | `src/03_InputData/03_Preprocessing.jl` | code | 7 | 608, 609, 630, 632, 2316, 2466, 2598 |
| `pr.Z / fpr.Z / prior .Z` | `src/13_Prior/03_FactorPrior.jl` | docstring | 1 | 47 |
| `pr.Z / fpr.Z / prior .Z` | `src/13_Prior/08_FactorBlackLittermanPrior.jl` | docstring | 1 | 39 |
| `pr.Z / fpr.Z / prior .Z` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 414 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_06b_asset_panel.jl` | test assertion | 7 | 200, 201, 202, 204, 205, 206, 207 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12b_prior_core.jl` | test assertion | 1 | 2782 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12c_feature_prior.jl` | test assertion | 3 | 61, 62, 201 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12c_feature_prior.jl` | test code | 2 | 260, 261 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12d_phylogeny_features.jl` | test assertion | 4 | 119, 120, 141, 733 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12d_phylogeny_features.jl` | test code | 1 | 220 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12d_phylogeny_features.jl` | test comment | 1 | 237 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12e_asset_sets_features.jl` | test assertion | 5 | 182, 260, 264, 601, 637 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12g_forwarding_rule.jl` | test assertion | 2 | 168, 196 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12g_forwarding_rule.jl` | test code | 2 | 58, 82 |
| `pr.Z / fpr.Z / prior .Z` | `test/test_12g_forwarding_rule.jl` | test comment | 1 | 163 |

Sites in the table: 109.

### Mechanism 2: `LowOrderPrior(` construction sites that pass `Z`

| File | Line | Ends | Kind |
| --- | ---: | ---: | --- |
| `src/13_Prior/01_Base_Prior.jl` | 1532 | 1535 | code |
| `src/13_Prior/12_EntropyPoolingPrior.jl` | 3649 | 3650 | code |
| `src/13_Prior/12_EntropyPoolingPrior.jl` | 3739 | 3740 | code |
| `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | 553 | 556 | code |
| `src/13_Prior/13_OpinionPoolingPrior.jl` | 533 | 534 | code |
| `src/13_Prior/11_MeucciEntropyPoolingPrior.jl` | 840 | 841 | code |
| `src/13_Prior/11_MeucciEntropyPoolingPrior.jl` | 980 | 981 | code |
| `test/test_13c_feature_views.jl` | 386 | 386 | test code |
| `test/test_12g_forwarding_rule.jl` | 58 | 58 | test code |
| `test/test_12g_forwarding_rule.jl` | 82 | 82 | test code |
| `test/test_12c_feature_prior.jl` | 175 | 176 | test code |
| `test/test_53_panel_carrier_census.jl` | 83 | 83 | test code |

Sites: 12.

### Mechanism 2: every `LowOrderPrior(` construction site, with or without `Z`

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `src/13_Prior/01_Base_Prior.jl` | code | 5 | 1399, 1465, 1472, 1532, 1928 |
| `src/13_Prior/01_Base_Prior.jl` | docstring | 5 | 43, 556, 1227, 1320, 1656 |
| `src/13_Prior/02_EmpiricalPrior.jl` | code | 2 | 167, 251 |
| `src/13_Prior/03_FactorPrior.jl` | code | 1 | 435 |
| `src/13_Prior/08_FactorBlackLittermanPrior.jl` | code | 1 | 417 |
| `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | code | 1 | 553 |
| `src/13_Prior/11_MeucciEntropyPoolingPrior.jl` | code | 2 | 840, 980 |
| `src/13_Prior/12_EntropyPoolingPrior.jl` | code | 2 | 3649, 3739 |
| `src/13_Prior/13_OpinionPoolingPrior.jl` | code | 1 | 533 |
| `test/test_05_tools.jl` | test code | 1 | 706 |
| `test/test_09e_deferred_quantity.jl` | test code | 3 | 122, 183, 611 |
| `test/test_09g_calibration_rules.jl` | test code | 14 | 425, 664, 674, 684, 686, 687, 698, 699, 783, 785, 787, 1140, 1169, 1194 |
| `test/test_09h_ambiguity_calibration.jl` | test code | 1 | 837 |
| `test/test_10_uncertainty_set.jl` | test code | 1 | 974 |
| `test/test_12b_prior_core.jl` | test assertion | 2 | 1933, 1945 |
| `test/test_12b_prior_core.jl` | test code | 9 | 1360, 1362, 1367, 1476, 1923, 1935, 1942, 2132, 2216 |
| `test/test_12c_feature_prior.jl` | test code | 2 | 175, 217 |
| `test/test_12f_forward_prior.jl` | test code | 7 | 18, 22, 27, 152, 172, 223, 331 |
| `test/test_12g_forwarding_rule.jl` | test code | 4 | 58, 82, 109, 134 |
| `test/test_12i_cross_sectional_factor_carrier.jl` | test assertion | 1 | 106 |
| `test/test_12i_cross_sectional_factor_carrier.jl` | test code | 5 | 76, 80, 115, 120, 125 |
| `test/test_13c_feature_views.jl` | test code | 1 | 386 |
| `test/test_50_investable_reduction.jl` | test assertion | 3 | 60, 65, 71 |
| `test/test_50_investable_reduction.jl` | test code | 3 | 45, 48, 76 |
| `test/test_53_panel_carrier_census.jl` | test code | 1 | 83 |

Sites: 78.

### Mechanism 3: `z_src`, the picker, the forwarders, the bridge and the diagnostics

555 sites. `feature_matrix_picker` is defined at `src/13_Prior/01_Base_Prior.jl:816` and called
at the eight forwarders listed below; its docstring ends at `:815` and names
`carrier_feature_names` (`:769`, `:772`) and `assert_source_selector` (`01_Base/10_Assertions.jl:475`,
shared with `x_src` and kept). `assert_feature_matrix_supplied` is at
`09_Distance/05_FeatureDistance.jl:1392`, called from the two keyword bridge methods at `:1450`
and `:1455`; its five-branch message is the home of `:none`, `:data`, `:prior`, `:data_only`
and the `else` arm that `:neither` falls into. The `arg_dict` key `:z_src` at
`01_DocstringDictionaries.jl:572` has 13 users, all of them in this mechanism.

`Pr_RR` is defined at `:351` and has 93 sites. Twenty-nine are in `01_Base_Prior.jl` (the
bridge prose, the picker, `returns_matrix_picker` and the forwarders). The other 64 are the plotting
verbs (`24_Plotting.jl` 12, `ext/PortfolioOptimisersPlotsExt.jl` 15), `expected_risk`
(`19_RiskMeasures/27_ExpectedRisk.jl` 8), `calc_net_returns` and `extract_pr`
(`20_Optimisation/01_Base_Optimisation.jl` 9), `NaiveOptimisationResult` (3),
`get_subset_size` and `get_window_size` (`02_CrossValidation/05_MultipleRandomised.jl` 7), and
`test_53_panel_carrier_census.jl` (7). Those read `X` off the bridge and keep it; only the
`Z` half of the bridge's contract goes.

`z_src` reaches the optimisers through three struct fields and their constructors, and through
the four clustering optimisers that pass `z_src = opt.z_src` into `clusterise`: the table lists
`04_Base_ClusteringOptimisation.jl` (field `:470`, keyword `:504`), `10_JuMPOptimiser.jl` (field
`:720`, and the phylogeny-constraint call at `:1362`), `17_NestedClustered.jl` (field `:517`),
`05_HierarchicalRiskParity.jl:332` and `:445`, `06_SchurComplementHierarchicalRiskParity.jl:1027`
and `:1060`, `07_HierarchicalEqualRiskContribution.jl:748`, and `11_MeanRisk.jl:222`.

### Mechanism 3: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Pr_RR` | 1 | 0 | 41 | 27 | 0 | 0 | 15 | 1 | 6 | 0 | 2 | 93 |
| `z_src` | 4 | 0 | 79 | 75 | 1 | 0 | 0 | 19 | 50 | 21 | 1 | 250 |
| `assert_feature_matrix_supplied` | 1 | 0 | 8 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 13 |
| `:data_only` | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 2 | 3 | 0 | 0 | 10 |
| `z_src = :data / :prior / :none` | 0 | 0 | 9 | 10 | 0 | 0 | 0 | 9 | 33 | 15 | 0 | 76 |
| `assert_source_selector` | 1 | 0 | 4 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 14 |
| `feature_matrix_picker` | 1 | 0 | 25 | 8 | 0 | 0 | 0 | 3 | 8 | 0 | 1 | 46 |
| `:neither` | 0 | 0 | 6 | 1 | 0 | 0 | 0 | 3 | 5 | 0 | 0 | 15 |
| `returns_matrix_picker` | 1 | 0 | 20 | 8 | 0 | 0 | 0 | 6 | 2 | 0 | 1 | 38 |

Total: 555.

### Mechanism 3: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `Pr_RR` | `docs/src/api/13_Prior/01_Base_Prior.md` | doc page | 2 | 16, 27 |
| `Pr_RR` | `ext/PortfolioOptimisersPlotsExt.jl` | code | 15 | 7, 47, 61, 134, 152, 409, 418, 470, 554, 594, 625, 1802, 1866, 1914, 1956 |
| `Pr_RR` | `src/03_InputData/03_Preprocessing.jl` | docstring | 1 | 12 |
| `Pr_RR` | `src/13_Prior/01_Base_Prior.jl` | code | 11 | 733, 816, 858, 902, 943, 985, 1030, 1075, 1119, 1166, 1207 |
| `Pr_RR` | `src/13_Prior/01_Base_Prior.jl` | definition | 1 | 351 |
| `Pr_RR` | `src/13_Prior/01_Base_Prior.jl` | docstring | 17 | 338, 342, 743, 745, 756, 765, 789, 853, 897, 938, 980, 1025, 1048, 1070, 1113, 1161, 1202 |
| `Pr_RR` | `src/19_RiskMeasures/27_ExpectedRisk.jl` | code | 3 | 321, 329, 336 |
| `Pr_RR` | `src/19_RiskMeasures/27_ExpectedRisk.jl` | docstring | 5 | 32, 33, 35, 50, 51 |
| `Pr_RR` | `src/20_Optimisation/01_Base_Optimisation.jl` | code | 3 | 2857, 2993, 3029 |
| `Pr_RR` | `src/20_Optimisation/01_Base_Optimisation.jl` | comment | 1 | 3032 |
| `Pr_RR` | `src/20_Optimisation/01_Base_Optimisation.jl` | docstring | 5 | 2833, 2839, 2847, 2985, 3010 |
| `Pr_RR` | `src/20_Optimisation/02_CrossValidation/05_MultipleRandomised.jl` | code | 7 | 405, 412, 417, 449, 456, 463, 500 |
| `Pr_RR` | `src/20_Optimisation/03_NaiveOptimisation.jl` | code | 2 | 191, 199 |
| `Pr_RR` | `src/20_Optimisation/03_NaiveOptimisation.jl` | docstring | 1 | 142 |
| `Pr_RR` | `src/24_Plotting.jl` | docstring | 12 | 15, 23, 80, 306, 342, 425, 529, 1099, 1130, 1194, 1206, 1207 |
| `Pr_RR` | `test/test_53_panel_carrier_census.jl` | test assertion | 1 | 137 |
| `Pr_RR` | `test/test_53_panel_carrier_census.jl` | test code | 5 | 9, 105, 108, 123, 138 |
| `Pr_RR` | `test/test_53_panel_carrier_census.jl` | test comment | 1 | 140 |
| `z_src` | `docs/src/api/22_AssetSelection.md` | doc page | 1 | 41 |
| `z_src` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 18 | 146, 155, 180, 195, 368, 474, 530, 577, 580, 583, 586, 598, 604, 677, 687, 757, 882, 929 |
| `z_src` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 964 |
| `z_src` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 572 |
| `z_src` | `src/03_InputData/03_Preprocessing.jl` | docstring | 1 | 2107 |
| `z_src` | `src/09_Distance/05_FeatureDistance.jl` | code | 12 | 768, 1392, 1394, 1396, 1397, 1398, 1399, 1400, 1451, 1452, 1456, 1457 |
| `z_src` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 13 | 437, 723, 1353, 1357, 1361, 1369, 1375, 1379, 1410, 1412, 1422, 1430, 1435 |
| `z_src` | `src/13_Prior/01_Base_Prior.jl` | code | 30 | 816, 817, 820, 821, 860, 862, 863, 903, 905, 906, 945, 947, 948, 987, 989, 990, 1032, 1034, 1035, 1076, 1078, 1121, 1123, 1124, 1168, 1170, 1171, 1209, 1211, 1212 |
| `z_src` | `src/13_Prior/01_Base_Prior.jl` | docstring | 44 | 780, 786, 788, 790, 796, 797, 801, 807, 833, 834, 840, 842, 876, 877, 883, 885, 918, 919, 925, 927, 960, 961, 967, 969, 1003, 1004, 1011, 1013, 1048, 1057, 1059, 1091, 1092, 1101, 1103, 1137, 1138, 1145, 1147, 1183, 1184, 1190, 1192, 1274 |
| `z_src` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 605 |
| `z_src` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 2 | 526, 536 |
| `z_src` | `src/20_Optimisation/04_Base_ClusteringOptimisation.jl` | code | 6 | 480, 482, 488, 491, 492, 506 |
| `z_src` | `src/20_Optimisation/04_Base_ClusteringOptimisation.jl` | docstring | 5 | 321, 325, 330, 417, 468 |
| `z_src` | `src/20_Optimisation/04_Base_ClusteringOptimisation.jl` | field declaration | 2 | 470, 504 |
| `z_src` | `src/20_Optimisation/05_HierarchicalRiskParity.jl` | code | 2 | 332, 445 |
| `z_src` | `src/20_Optimisation/05_HierarchicalRiskParity.jl` | docstring | 1 | 154 |
| `z_src` | `src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl` | code | 2 | 1027, 1060 |
| `z_src` | `src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl` | docstring | 1 | 508 |
| `z_src` | `src/20_Optimisation/07_HierarchicalEqualRiskContribution.jl` | code | 1 | 748 |
| `z_src` | `src/20_Optimisation/07_HierarchicalEqualRiskContribution.jl` | docstring | 1 | 174 |
| `z_src` | `src/20_Optimisation/10_JuMPOptimiser.jl` | code | 11 | 752, 754, 957, 967, 1009, 1049, 1054, 1142, 1222, 1362, 1393 |
| `z_src` | `src/20_Optimisation/10_JuMPOptimiser.jl` | docstring | 4 | 513, 517, 522, 718 |
| `z_src` | `src/20_Optimisation/10_JuMPOptimiser.jl` | field declaration | 1 | 720 |
| `z_src` | `src/20_Optimisation/11_MeanRisk.jl` | docstring | 1 | 222 |
| `z_src` | `src/20_Optimisation/17_NestedClustered.jl` | code | 9 | 528, 530, 551, 555, 568, 581, 584, 708, 764 |
| `z_src` | `src/20_Optimisation/17_NestedClustered.jl` | docstring | 3 | 397, 412, 515 |
| `z_src` | `src/20_Optimisation/17_NestedClustered.jl` | field declaration | 1 | 517 |
| `z_src` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `z_src` | `src/22_Preselection.jl` | docstring | 3 | 1225, 1272, 1277 |
| `z_src` | `test/test_12c_feature_prior.jl` | test assertion | 1 | 308 |
| `z_src` | `test/test_12d_phylogeny_features.jl` | test code | 1 | 192 |
| `z_src` | `test/test_12e_asset_sets_features.jl` | test code | 4 | 269, 282, 641, 657 |
| `z_src` | `test/test_12e_asset_sets_features.jl` | test comment | 2 | 252, 630 |
| `z_src` | `test/test_13b_feature_routing.jl` | test assertion | 13 | 65, 66, 154, 155, 163, 164, 190, 191, 215, 219, 227, 228, 229 |
| `z_src` | `test/test_13b_feature_routing.jl` | test code | 13 | 60, 69, 70, 149, 158, 185, 188, 192, 216, 425, 427, 429, 436 |
| `z_src` | `test/test_13b_feature_routing.jl` | test comment | 6 | 49, 58, 64, 213, 224, 335 |
| `z_src` | `test/test_13c_feature_views.jl` | test code | 9 | 345, 349, 356, 409, 416, 450, 457, 476, 488 |
| `z_src` | `test/test_13e_feature_preselection.jl` | test assertion | 4 | 56, 141, 157, 158 |
| `z_src` | `test/test_13e_feature_preselection.jl` | test code | 4 | 12, 51, 58, 202 |
| `z_src` | `test/test_13e_feature_preselection.jl` | test comment | 3 | 172, 196, 268 |
| `z_src` | `test/test_37_time_dependent_constraints.jl` | test comment | 2 | 771, 1689 |
| `z_src` | `test/test_53_panel_carrier_census.jl` | test assertion | 1 | 59 |
| `z_src` | `test/test_53_panel_carrier_census.jl` | test code | 6 | 14, 55, 56, 64, 65, 192 |
| `z_src` | `user_guide/02_Optimisers.jl` | example code | 2 | 178, 182 |
| `assert_feature_matrix_supplied` | `docs/src/api/09_Distance/05_FeatureDistance.md` | doc page | 1 | 27 |
| `assert_feature_matrix_supplied` | `src/09_Distance/05_FeatureDistance.jl` | code | 2 | 1452, 1457 |
| `assert_feature_matrix_supplied` | `src/09_Distance/05_FeatureDistance.jl` | definition | 1 | 1392 |
| `assert_feature_matrix_supplied` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 4 | 1353, 1422, 1435, 1446 |
| `assert_feature_matrix_supplied` | `src/13_Prior/01_Base_Prior.jl` | docstring | 2 | 782, 813 |
| `assert_feature_matrix_supplied` | `src/22_Preselection.jl` | docstring | 2 | 1225, 1250 |
| `assert_feature_matrix_supplied` | `test/test_13e_feature_preselection.jl` | test code | 1 | 145 |
| `:data_only` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 1400 |
| `:data_only` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 1 | 1362 |
| `:data_only` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `:data_only` | `src/22_Preselection.jl` | docstring | 2 | 1272, 1277 |
| `:data_only` | `test/test_13e_feature_preselection.jl` | test assertion | 2 | 147, 159 |
| `:data_only` | `test/test_13e_feature_preselection.jl` | test code | 2 | 127, 154 |
| `:data_only` | `test/test_13e_feature_preselection.jl` | test comment | 1 | 162 |
| `z_src = :data / :prior / :none` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 13 | 155, 180, 195, 368, 474, 580, 583, 598, 604, 677, 687, 757, 882 |
| `z_src = :data / :prior / :none` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 964 |
| `z_src = :data / :prior / :none` | `src/09_Distance/05_FeatureDistance.jl` | code | 7 | 768, 1394, 1396, 1397, 1398, 1399, 1400 |
| `z_src = :data / :prior / :none` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 2 | 437, 723 |
| `z_src = :data / :prior / :none` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 820 |
| `z_src = :data / :prior / :none` | `src/13_Prior/01_Base_Prior.jl` | docstring | 2 | 780, 801 |
| `z_src = :data / :prior / :none` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 605 |
| `z_src = :data / :prior / :none` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 536 |
| `z_src = :data / :prior / :none` | `src/20_Optimisation/04_Base_ClusteringOptimisation.jl` | docstring | 1 | 330 |
| `z_src = :data / :prior / :none` | `src/20_Optimisation/10_JuMPOptimiser.jl` | docstring | 1 | 522 |
| `z_src = :data / :prior / :none` | `src/20_Optimisation/17_NestedClustered.jl` | docstring | 1 | 412 |
| `z_src = :data / :prior / :none` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `z_src = :data / :prior / :none` | `src/22_Preselection.jl` | docstring | 1 | 1272 |
| `z_src = :data / :prior / :none` | `test/test_12c_feature_prior.jl` | test assertion | 1 | 308 |
| `z_src = :data / :prior / :none` | `test/test_12d_phylogeny_features.jl` | test code | 1 | 192 |
| `z_src = :data / :prior / :none` | `test/test_12e_asset_sets_features.jl` | test code | 4 | 269, 282, 641, 657 |
| `z_src = :data / :prior / :none` | `test/test_12e_asset_sets_features.jl` | test comment | 2 | 252, 630 |
| `z_src = :data / :prior / :none` | `test/test_13b_feature_routing.jl` | test assertion | 6 | 65, 66, 154, 155, 163, 164 |
| `z_src = :data / :prior / :none` | `test/test_13b_feature_routing.jl` | test code | 12 | 60, 69, 70, 149, 158, 188, 192, 216, 425, 427, 429, 436 |
| `z_src = :data / :prior / :none` | `test/test_13b_feature_routing.jl` | test comment | 3 | 49, 58, 335 |
| `z_src = :data / :prior / :none` | `test/test_13c_feature_views.jl` | test code | 6 | 409, 416, 450, 457, 476, 488 |
| `z_src = :data / :prior / :none` | `test/test_13e_feature_preselection.jl` | test assertion | 2 | 157, 158 |
| `z_src = :data / :prior / :none` | `test/test_13e_feature_preselection.jl` | test code | 1 | 202 |
| `z_src = :data / :prior / :none` | `test/test_53_panel_carrier_census.jl` | test code | 4 | 14, 55, 64, 192 |
| `z_src = :data / :prior / :none` | `user_guide/02_Optimisers.jl` | example code | 1 | 178 |
| `assert_source_selector` | `docs/src/api/01_Base.md` | doc page | 1 | 148 |
| `assert_source_selector` | `src/01_Base/10_Assertions.jl` | definition | 1 | 475 |
| `assert_source_selector` | `src/13_Prior/01_Base_Prior.jl` | code | 2 | 734, 817 |
| `assert_source_selector` | `src/13_Prior/01_Base_Prior.jl` | docstring | 4 | 709, 728, 786, 811 |
| `assert_source_selector` | `src/20_Optimisation/04_Base_ClusteringOptimisation.jl` | code | 2 | 481, 482 |
| `assert_source_selector` | `src/20_Optimisation/10_JuMPOptimiser.jl` | code | 2 | 753, 754 |
| `assert_source_selector` | `src/20_Optimisation/17_NestedClustered.jl` | code | 2 | 529, 530 |
| `feature_matrix_picker` | `docs/src/api/13_Prior/01_Base_Prior.md` | doc page | 1 | 33 |
| `feature_matrix_picker` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 5 | 1357, 1388, 1416, 1429, 1445 |
| `feature_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | code | 8 | 862, 905, 947, 989, 1034, 1123, 1170, 1211 |
| `feature_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | definition | 1 | 816 |
| `feature_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | docstring | 20 | 342, 349, 745, 764, 833, 855, 876, 899, 918, 940, 960, 982, 1003, 1027, 1091, 1115, 1137, 1163, 1183, 1204 |
| `feature_matrix_picker` | `test/test_13b_feature_routing.jl` | test code | 2 | 411, 416 |
| `feature_matrix_picker` | `test/test_13e_feature_preselection.jl` | test code | 1 | 237 |
| `feature_matrix_picker` | `test/test_53_panel_carrier_census.jl` | test assertion | 3 | 92, 93, 97 |
| `feature_matrix_picker` | `test/test_53_panel_carrier_census.jl` | test code | 5 | 12, 56, 65, 85, 109 |
| `:neither` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 3 | 1360, 1362, 1364 |
| `:neither` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 821 |
| `:neither` | `src/13_Prior/01_Base_Prior.jl` | docstring | 3 | 782, 790, 807 |
| `:neither` | `test/test_13e_feature_preselection.jl` | test assertion | 2 | 148, 160 |
| `:neither` | `test/test_13e_feature_preselection.jl` | test code | 2 | 154, 168 |
| `:neither` | `test/test_13e_feature_preselection.jl` | test comment | 2 | 128, 163 |
| `:neither` | `test/test_53_panel_carrier_census.jl` | test assertion | 1 | 97 |
| `:neither` | `test/test_53_panel_carrier_census.jl` | test comment | 1 | 95 |
| `returns_matrix_picker` | `docs/src/api/13_Prior/01_Base_Prior.md` | doc page | 1 | 15 |
| `returns_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | code | 8 | 861, 904, 946, 988, 1033, 1122, 1169, 1210 |
| `returns_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | definition | 1 | 733 |
| `returns_matrix_picker` | `src/13_Prior/01_Base_Prior.jl` | docstring | 20 | 342, 348, 780, 812, 832, 854, 875, 898, 917, 939, 959, 981, 1002, 1026, 1090, 1114, 1136, 1162, 1182, 1203 |
| `returns_matrix_picker` | `test/test_12b_prior_core.jl` | test assertion | 4 | 1424, 1425, 1426, 1428 |
| `returns_matrix_picker` | `test/test_12b_prior_core.jl` | test code | 1 | 1432 |
| `returns_matrix_picker` | `test/test_53_panel_carrier_census.jl` | test assertion | 2 | 72, 73 |
| `returns_matrix_picker` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 110 |

Sites in the table: 555.

### Mechanism 3: the eight forwarders in `src/13_Prior/01_Base_Prior.jl`

- `src/13_Prior/01_Base_Prior.jl:858` head `function clusterise(cle::AbstractClustersEstimator, pr::Pr_RR;`, picker call at `:862`, forward at `:863`
- `src/13_Prior/01_Base_Prior.jl:902` head `function phylogeny_matrix(pl::NwE_ClE_Cl, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing,`, picker call at `:905`, forward at `:906`
- `src/13_Prior/01_Base_Prior.jl:943` head `function phylogeny_constraints(plc::AbstractPhylogenyConstraintEstimator, pr::Pr_RR;`, picker call at `:947`, forward at `:948`
- `src/13_Prior/01_Base_Prior.jl:985` head `function centrality_vector(cte::CentralityEstimator, pr::Pr_RR;`, picker call at `:989`, forward at `:990`
- `src/13_Prior/01_Base_Prior.jl:1030` head `function centrality_vector(pl::NwE_ClE_Cl, ct::AbstractCentralityAlgorithm, pr::Pr_RR;`, picker call at `:1034`, forward at `:1035`
- `src/13_Prior/01_Base_Prior.jl:1119` head `function average_centrality(cte::CentralityEstimator, w::VecNum, pr::Pr_RR;`, picker call at `:1123`, forward at `:1124`
- `src/13_Prior/01_Base_Prior.jl:1166` head `function asset_phylogeny(pl::NwE_ClE_Cl, w::VecNum, pr::Pr_RR;`, picker call at `:1170`, forward at `:1171`
- `src/13_Prior/01_Base_Prior.jl:1207` head `function centrality_constraints(ccs::CC_VecCC, pr::Pr_RR;`, picker call at `:1211`, forward at `:1212`

`average_centrality(pl, ct, w, pr::Pr_RR; …)` at `:1074` is the ninth head that takes `z_src`.
It forwards `rd`, `x_src` and `z_src` to the `centrality_vector` forwarder at `:1078` and never
calls the picker.

### Mechanism 3: `z_src` per file

- `docs/src/api/22_AssetSelection.md`: 1
- `examples/3_optimisers/16_Feature_Distance_Clustering.jl`: 19
- `src/01_Base/01_DocstringDictionaries.jl`: 1
- `src/03_InputData/03_Preprocessing.jl`: 1
- `src/09_Distance/05_FeatureDistance.jl`: 25
- `src/13_Prior/01_Base_Prior.jl`: 74
- `src/13_Prior/15_FeaturePrior.jl`: 3
- `src/20_Optimisation/04_Base_ClusteringOptimisation.jl`: 13
- `src/20_Optimisation/05_HierarchicalRiskParity.jl`: 3
- `src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl`: 3
- `src/20_Optimisation/07_HierarchicalEqualRiskContribution.jl`: 2
- `src/20_Optimisation/10_JuMPOptimiser.jl`: 16
- `src/20_Optimisation/11_MeanRisk.jl`: 1
- `src/20_Optimisation/17_NestedClustered.jl`: 13
- `src/22_Preselection.jl`: 4
- `test/test_12c_feature_prior.jl`: 1
- `test/test_12d_phylogeny_features.jl`: 1
- `test/test_12e_asset_sets_features.jl`: 6
- `test/test_13b_feature_routing.jl`: 32
- `test/test_13c_feature_views.jl`: 9
- `test/test_13e_feature_preselection.jl`: 11
- `test/test_37_time_dependent_constraints.jl`: 2
- `test/test_53_panel_carrier_census.jl`: 7
- `user_guide/02_Optimisers.jl`: 2

### Mechanism 4: the selector, `sets`, `strict` and the resolution helpers

155 sites in `src/09_Distance/05_FeatureDistance.jl`, plus the construction sites below and 12
`select_features` sites. The fields are declared at `:547` (`sel`), `:551` (`sets`) and `:555`
(`strict`); the inner constructor calls `assert_feature_selector` (`:610`) at `:561`, and the
keyword constructor at `:566` takes all three. The helpers are `feature_selector_msg` (`:656`, `:660`),
`feature_selection_push!` (`:698`), `feature_selection_indices` (`:763`, `:766`) and
`select_features` (`:830`, `:834`), called from the two `distance` methods at `:1280` and
`:1287`. The `field_dict` keys are `:fdsel` (`01_DocstringDictionaries.jl:151`, 3 users),
`:fdsets` (`:152`, 4 users) and `:fdstrict` (`:153`, 3 users); every user is in this file.

No file outside `05_FeatureDistance.jl` reads `.sel`, `.sets` or `.strict` off a distance. The
`sets`, `strict` and `sel` reads everywhere else belong to optimisers, the Black-Litterman
family and the risk-measure constraints, and stay.

Of the 163 `FeatureDistance(` construction sites in the tree, 32 pass `sel`, 7 pass `sets` and
7 pass `strict`, all of them in `test_13b_feature_routing.jl`, `test_13e_feature_preselection.jl`
and `test_53_panel_carrier_census.jl`, except the constructor itself and its docstring
example. The 18 sites in `16_Feature_Distance_Clustering.jl` and the one in
`user_guide/02_Optimisers.jl` pass none of the three.

### Mechanism 4: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `select_features` | 2 | 0 | 6 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 12 |
| `assert_feature_selector` | 1 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 |
| `feature_selection_indices` | 2 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 11 |
| `feature_selector_msg` | 2 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 9 |
| `sets (in 05_FeatureDistance.jl)` | 1 | 0 | 24 | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 42 |
| `sel (in 05_FeatureDistance.jl)` | 3 | 0 | 27 | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51 |
| `strict (in 05_FeatureDistance.jl)` | 1 | 0 | 11 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |

Total: 155.

### Mechanism 4: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `select_features` | `docs/src/api/09_Distance/05_FeatureDistance.md` | doc page | 1 | 18 |
| `select_features` | `src/09_Distance/05_FeatureDistance.jl` | code | 2 | 1280, 1287 |
| `select_features` | `src/09_Distance/05_FeatureDistance.jl` | definition | 2 | 830, 834 |
| `select_features` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 6 | 441, 524, 607, 759, 777, 779 |
| `select_features` | `test/test_13b_feature_routing.jl` | test assertion | 1 | 309 |
| `assert_feature_selector` | `docs/src/api/09_Distance/05_FeatureDistance.md` | doc page | 1 | 14 |
| `assert_feature_selector` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 561 |
| `assert_feature_selector` | `src/09_Distance/05_FeatureDistance.jl` | definition | 1 | 610 |
| `assert_feature_selector` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 3 | 482, 523, 574 |
| `feature_selection_indices` | `docs/src/api/09_Distance/05_FeatureDistance.md` | doc page | 1 | 17 |
| `feature_selection_indices` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 836 |
| `feature_selection_indices` | `src/09_Distance/05_FeatureDistance.jl` | definition | 2 | 763, 766 |
| `feature_selection_indices` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 7 | 525, 651, 693, 718, 719, 801, 825 |
| `feature_selector_msg` | `docs/src/api/09_Distance/05_FeatureDistance.md` | doc page | 1 | 15 |
| `feature_selector_msg` | `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 709 |
| `feature_selector_msg` | `src/09_Distance/05_FeatureDistance.jl` | definition | 2 | 656, 660 |
| `feature_selector_msg` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 5 | 623, 624, 675, 683, 695 |
| `sets (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | code | 17 | 560, 561, 562, 563, 570, 571, 610, 611, 612, 657, 698, 700, 702, 766, 769, 772, 836 |
| `sets (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 24 | 131, 171, 438, 473, 482, 499, 508, 523, 527, 574, 576, 578, 584, 596, 634, 641, 664, 669, 673, 718, 719, 723, 736, 808 |
| `sets (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | field declaration | 1 | 551 |
| `sel (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | code | 21 | 561, 562, 563, 571, 610, 612, 613, 616, 617, 618, 619, 657, 661, 763, 764, 766, 768, 771, 836, 840, 843 |
| `sel (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 27 | 433, 435, 443, 472, 482, 498, 507, 523, 526, 574, 576, 580, 585, 586, 587, 597, 598, 680, 718, 719, 731, 737, 739, 754, 791, 808, 815 |
| `sel (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | field declaration | 3 | 547, 559, 569 |
| `strict (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | code | 12 | 560, 563, 570, 571, 657, 661, 699, 710, 766, 772, 836, 840 |
| `strict (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 11 | 441, 474, 500, 509, 665, 675, 718, 719, 725, 791, 808 |
| `strict (in 05_FeatureDistance.jl)` | `src/09_Distance/05_FeatureDistance.jl` | field declaration | 1 | 555 |

Sites in the table: 155.

### Mechanism 4: `FeatureDistance(` construction sites that pass `sel`, `sets` or `strict`

#### `sel =`: 32 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `test/test_13b_feature_routing.jl` | test assertion | 11 | 306, 313, 315, 316, 327, 329, 336, 374, 377, 392, 395 |
| `test/test_13b_feature_routing.jl` | test code | 15 | 296, 297, 298, 299, 319, 338, 345, 347, 352, 356, 358, 385, 405, 422, 440 |
| `test/test_13e_feature_preselection.jl` | test code | 3 | 250, 257, 273 |
| `test/test_53_panel_carrier_census.jl` | test code | 3 | 158, 184, 204 |

#### `sets =`: 7 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `src/09_Distance/05_FeatureDistance.jl` | code | 1 | 566 |
| `src/09_Distance/05_FeatureDistance.jl` | docstring | 1 | 468 |
| `test/test_13b_feature_routing.jl` | test assertion | 2 | 374, 377 |
| `test/test_13b_feature_routing.jl` | test code | 2 | 302, 385 |
| `test/test_13e_feature_preselection.jl` | test code | 1 | 250 |

#### `strict =`: 7 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `test/test_13b_feature_routing.jl` | test code | 3 | 347, 385, 440 |
| `test/test_13e_feature_preselection.jl` | test code | 1 | 273 |
| `test/test_53_panel_carrier_census.jl` | test code | 3 | 158, 184, 204 |

Every `FeatureDistance(` construction site: 163.
- `examples/3_optimisers/16_Feature_Distance_Clustering.jl`: 18
- `src/09_Distance/05_FeatureDistance.jl`: 8
- `test/test_12c_feature_prior.jl`: 1
- `test/test_12d_phylogeny_features.jl`: 8
- `test/test_12e_asset_sets_features.jl`: 9
- `test/test_13_phylogeny.jl`: 31
- `test/test_13b_feature_routing.jl`: 34
- `test/test_13c_feature_views.jl`: 30
- `test/test_13d_feature_collapse.jl`: 13
- `test/test_13e_feature_preselection.jl`: 6
- `test/test_53_panel_carrier_census.jl`: 4
- `user_guide/02_Optimisers.jl`: 1

### Mechanism 5: the carrier fields, the checks, the views and the collapse

1,354 sites. The definitions are in `src/03_InputData/03_Preprocessing.jl`:
`check_feature_names` (`:111`), `check_feature_matrix` (`:167`, `:171`, `:181`),
`check_names_and_feature_matrix` (`:244`, `:251`, `:260`), `features_are_assets` (`:297`),
`feature_matrix_view` (`:339`–`:351`, five methods) and `feature_row_indices` (`:398`–`:407`,
four). `PricesResult`'s inner constructor takes `nz` and `Z` at `:507` and its keyword
constructor at `:536`–`:537`; `ReturnsResult` declares `pnl` as its last field at `:768`, its
inner constructor takes `nz` and `Z` at `:773` and `pnl` at `:774` and checks the panel with
`check_asset_panel` at `:831`, and its keyword constructor takes all three at `:841`–`:843`. `check_asset_panel` is at
`01_AssetPanel.jl:755` and `:758`. `prices_to_returns` takes them at `:1573`–`:1575` and slices
them at `:1658`–`:1662`. The collapse is `collapse_feature_matrix`
(`20_Optimisation/01_Base_Optimisation.jl:2938`–`:2963`, six methods), called from
`prepare_outer_rd` (`16_Base_MetaOptimisation.jl:326`, the call at `:350`) and, under
cross-validation, from `fold_feature_matrix` (`:500`–`:511`), `fold_feature_anchors` (`:535`)
and `rebuild_feature_matrix` (`:564`), which `rebuild_returns_result` (`:623`) calls.

Seven `port_opt_view` methods slice one of the three fields, listed after the tables. Two are
the `ReturnsResult` arities the map names (`:1043` and `:1057`), two are the `PricesResult`
arities (`:602`, `:612`), one is `LowOrderPrior`'s (`01_Base_Prior.jl:1527`), and two are
`AssetPanel`'s own (`01_AssetPanel.jl:708`, `:712`), which the carrier views call.

`ReturnsResult(; nz = …, Z = …)` is built at 53 sites and `PricesResult(; nz = …, Z = …)` at
13; the `nz` and `Z` lists coincide except for one line, `test_06_preprocessing.jl:118` versus
`:120`. `pnl` is passed at 8 sites, all `ReturnsResult`: three in `03_Preprocessing.jl` (the two
views and `prices_to_returns`), four in `test_06b_asset_panel.jl` and one in
`test_08t_exposures.jl`. There are 280 `ReturnsResult(` and 76 `PricesResult(` constructions
in all, so 227 and 63 of them pass no feature field and change only with the signature.

The two `field_dict` entries that describe the fields on the carriers are `:nz_feat`
(`01_DocstringDictionaries.jl:621`, 2 users) and `:Z` (`:233`, 12 users, most of them the
distance kernels). Neither key is named for a carrier, so the census lists them under `dict
keys naming nz / Z / pnl` and leaves the deletion decision to the carrier ticket.

### Mechanism 5: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `features_are_assets` | 1 | 0 | 19 | 8 | 0 | 0 | 0 | 7 | 3 | 2 | 1 | 41 |
| `prepare_outer_rd` | 1 | 0 | 7 | 2 | 0 | 0 | 0 | 0 | 8 | 0 | 1 | 19 |
| `collapse_feature_matrix` | 6 | 0 | 13 | 3 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 25 |
| `rebuild_returns_result` | 1 | 0 | 14 | 2 | 0 | 0 | 0 | 1 | 6 | 0 | 1 | 25 |
| `feature_row_indices` | 4 | 0 | 11 | 5 | 0 | 0 | 0 | 3 | 2 | 1 | 1 | 27 |
| `rebuild_feature_matrix` | 1 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 8 |
| `fold_feature_matrix` | 3 | 0 | 6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 11 |
| `fold_feature_anchors` | 1 | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 6 |
| `check_feature_matrix` | 3 | 0 | 15 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 24 |
| `feature_matrix_view` | 5 | 0 | 20 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 34 |
| `check_names_and_feature_matrix` | 3 | 0 | 14 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 21 |
| `check_feature_names` | 1 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 9 |
| `check_asset_panel` | 2 | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 8 |
| `rd.Z` | 0 | 0 | 14 | 13 | 1 | 0 | 0 | 8 | 14 | 1 | 0 | 51 |
| `any .Z read` | 0 | 0 | 23 | 24 | 1 | 0 | 0 | 120 | 39 | 12 | 0 | 219 |
| `rd.nz` | 0 | 0 | 5 | 12 | 0 | 0 | 0 | 8 | 15 | 0 | 0 | 40 |
| `any .nz read` | 0 | 0 | 8 | 19 | 0 | 0 | 0 | 22 | 18 | 0 | 0 | 67 |
| `nz keyword / field` | 2 | 0 | 11 | 51 | 0 | 0 | 0 | 7 | 68 | 7 | 0 | 146 |
| `rd.pnl` | 0 | 0 | 26 | 22 | 0 | 0 | 0 | 12 | 17 | 0 | 0 | 77 |
| `any .pnl read` | 0 | 0 | 26 | 22 | 0 | 0 | 0 | 23 | 25 | 0 | 0 | 96 |
| `pr.nz` | 0 | 0 | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `pr.Z on a PricesResult / prior` | 0 | 0 | 1 | 9 | 0 | 0 | 0 | 13 | 6 | 0 | 0 | 29 |
| `Z keyword / field` | 6 | 0 | 74 | 107 | 0 | 0 | 0 | 3 | 108 | 10 | 1 | 309 |
| `pnl keyword / field` | 1 | 0 | 14 | 29 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 52 |
| `dict keys naming nz / Z / pnl` | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 2 |

Total: 1354.

### Mechanism 5: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `features_are_assets` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 14 |
| `features_are_assets` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 1 | 628 |
| `features_are_assets` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 947 |
| `features_are_assets` | `src/03_InputData/03_Preprocessing.jl` | code | 5 | 629, 1050, 1073, 1658, 2464 |
| `features_are_assets` | `src/03_InputData/03_Preprocessing.jl` | definition | 1 | 297 |
| `features_are_assets` | `src/03_InputData/03_Preprocessing.jl` | docstring | 9 | 270, 311, 336, 561, 863, 944, 1014, 1479, 2347 |
| `features_are_assets` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 2 | 789, 826 |
| `features_are_assets` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 2 | 26, 72 |
| `features_are_assets` | `src/20_Optimisation/01_Base_Optimisation.jl` | docstring | 3 | 2910, 2920, 2933 |
| `features_are_assets` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 2 | 349, 573 |
| `features_are_assets` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | comment | 1 | 348 |
| `features_are_assets` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 3 | 324, 484, 599 |
| `features_are_assets` | `test/test_06_preprocessing.jl` | test assertion | 2 | 227, 237 |
| `features_are_assets` | `test/test_13b_feature_routing.jl` | test assertion | 1 | 404 |
| `features_are_assets` | `test/test_13d_feature_collapse.jl` | test assertion | 4 | 107, 108, 211, 427 |
| `features_are_assets` | `test/test_13d_feature_collapse.jl` | test code | 1 | 47 |
| `features_are_assets` | `test/test_13d_feature_collapse.jl` | test comment | 2 | 208, 344 |
| `prepare_outer_rd` | `docs/src/api/20_Optimisation/16_Base_MetaOptimisation.md` | doc page | 1 | 13 |
| `prepare_outer_rd` | `src/20_Optimisation/01_Base_Optimisation.jl` | docstring | 2 | 2881, 2934 |
| `prepare_outer_rd` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 1 | 763 |
| `prepare_outer_rd` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | comment | 1 | 571 |
| `prepare_outer_rd` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | definition | 1 | 326 |
| `prepare_outer_rd` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 5 | 543, 560, 599, 621, 753 |
| `prepare_outer_rd` | `test/test_13d_feature_collapse.jl` | test code | 7 | 200, 202, 209, 214, 224, 396, 430 |
| `prepare_outer_rd` | `test/test_13d_feature_collapse.jl` | test comment | 1 | 232 |
| `collapse_feature_matrix` | `docs/src/api/20_Optimisation/01_Base_Optimisation.md` | doc page | 1 | 115 |
| `collapse_feature_matrix` | `src/20_Optimisation/01_Base_Optimisation.jl` | definition | 6 | 2938, 2941, 2946, 2957, 2960, 2963 |
| `collapse_feature_matrix` | `src/20_Optimisation/01_Base_Optimisation.jl` | docstring | 6 | 2895, 2896, 2897, 2898, 2899, 2900 |
| `collapse_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 3 | 350, 504, 512 |
| `collapse_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 7 | 315, 323, 449, 476, 495, 543, 599 |
| `collapse_feature_matrix` | `test/test_13d_feature_collapse.jl` | test code | 2 | 45, 475 |
| `rebuild_returns_result` | `docs/src/api/20_Optimisation/16_Base_MetaOptimisation.md` | doc page | 1 | 20 |
| `rebuild_returns_result` | `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` | docstring | 7 | 305, 320, 576, 605, 766, 1097, 1113 |
| `rebuild_returns_result` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 2 | 776, 784 |
| `rebuild_returns_result` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | definition | 1 | 623 |
| `rebuild_returns_result` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 7 | 360, 374, 448, 559, 587, 726, 754 |
| `rebuild_returns_result` | `test/test_13d_feature_collapse.jl` | test assertion | 1 | 508 |
| `rebuild_returns_result` | `test/test_13d_feature_collapse.jl` | test code | 6 | 409, 424, 468, 495, 527, 533 |
| `feature_row_indices` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 16 |
| `feature_row_indices` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 968 |
| `feature_row_indices` | `src/03_InputData/03_Preprocessing.jl` | code | 3 | 608, 630, 1659 |
| `feature_row_indices` | `src/03_InputData/03_Preprocessing.jl` | comment | 1 | 801 |
| `feature_row_indices` | `src/03_InputData/03_Preprocessing.jl` | definition | 4 | 398, 401, 404, 407 |
| `feature_row_indices` | `src/03_InputData/03_Preprocessing.jl` | docstring | 8 | 355, 356, 357, 358, 553, 562, 681, 1479 |
| `feature_row_indices` | `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` | docstring | 1 | 575 |
| `feature_row_indices` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 1 | 424 |
| `feature_row_indices` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 2 | 402, 417 |
| `feature_row_indices` | `test/test_06_preprocessing.jl` | test assertion | 3 | 283, 285, 287 |
| `feature_row_indices` | `test/test_06_preprocessing.jl` | test code | 2 | 279, 430 |
| `rebuild_feature_matrix` | `docs/src/api/20_Optimisation/16_Base_MetaOptimisation.md` | doc page | 1 | 19 |
| `rebuild_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 1 | 661 |
| `rebuild_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | definition | 1 | 564 |
| `rebuild_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 5 | 418, 498, 539, 599, 619 |
| `fold_feature_matrix` | `docs/src/api/20_Optimisation/16_Base_MetaOptimisation.md` | doc page | 1 | 17 |
| `fold_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 1 | 574 |
| `fold_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | definition | 3 | 500, 503, 511 |
| `fold_feature_matrix` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 6 | 470, 471, 472, 517, 532, 561 |
| `fold_feature_anchors` | `docs/src/api/20_Optimisation/16_Base_MetaOptimisation.md` | doc page | 1 | 18 |
| `fold_feature_anchors` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 1 | 575 |
| `fold_feature_anchors` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | definition | 1 | 535 |
| `fold_feature_anchors` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 2 | 515, 562 |
| `fold_feature_anchors` | `test/test_13d_feature_collapse.jl` | test assertion | 1 | 567 |
| `check_feature_matrix` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 12 |
| `check_feature_matrix` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example prose | 1 | 960 |
| `check_feature_matrix` | `src/03_InputData/02_AssetPanelBuilder.jl` | code | 1 | 859 |
| `check_feature_matrix` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 4 | 8, 85, 830, 1190 |
| `check_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | code | 2 | 255, 264 |
| `check_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | definition | 3 | 167, 171, 181 |
| `check_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | docstring | 8 | 88, 108, 120, 121, 122, 204, 213, 238 |
| `check_feature_matrix` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 1459 |
| `check_feature_matrix` | `src/13_Prior/01_Base_Prior.jl` | docstring | 2 | 1304, 1348 |
| `check_feature_matrix` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 418 |
| `feature_matrix_view` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 15 |
| `feature_matrix_view` | `src/03_InputData/01_AssetPanel.jl` | docstring | 1 | 496 |
| `feature_matrix_view` | `src/03_InputData/03_Preprocessing.jl` | code | 6 | 609, 632, 1052, 1075, 1660, 2466 |
| `feature_matrix_view` | `src/03_InputData/03_Preprocessing.jl` | definition | 5 | 339, 342, 345, 348, 351 |
| `feature_matrix_view` | `src/03_InputData/03_Preprocessing.jl` | docstring | 13 | 301, 302, 303, 304, 305, 393, 553, 562, 864, 945, 1014, 1479, 2348 |
| `feature_matrix_view` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 1535 |
| `feature_matrix_view` | `src/13_Prior/01_Base_Prior.jl` | docstring | 4 | 1313, 1345, 1509, 1525 |
| `feature_matrix_view` | `src/13_Prior/15_FeaturePrior.jl` | code | 1 | 641 |
| `feature_matrix_view` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 2 | 628, 634 |
| `check_names_and_feature_matrix` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 13 |
| `check_names_and_feature_matrix` | `src/03_InputData/01_AssetPanel.jl` | docstring | 2 | 722, 752 |
| `check_names_and_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | code | 3 | 526, 829, 1586 |
| `check_names_and_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | definition | 3 | 244, 251, 260 |
| `check_names_and_feature_matrix` | `src/03_InputData/03_Preprocessing.jl` | docstring | 12 | 88, 107, 128, 140, 161, 196, 197, 198, 450, 473, 685, 1465 |
| `check_feature_names` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 11 |
| `check_feature_names` | `src/03_InputData/03_Preprocessing.jl` | code | 2 | 254, 263 |
| `check_feature_names` | `src/03_InputData/03_Preprocessing.jl` | definition | 1 | 111 |
| `check_feature_names` | `src/03_InputData/03_Preprocessing.jl` | docstring | 5 | 162, 204, 213, 226, 237 |
| `check_asset_panel` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 79 |
| `check_asset_panel` | `src/03_InputData/01_AssetPanel.jl` | definition | 2 | 755, 758 |
| `check_asset_panel` | `src/03_InputData/01_AssetPanel.jl` | docstring | 2 | 717, 718 |
| `check_asset_panel` | `src/03_InputData/03_Preprocessing.jl` | code | 1 | 831 |
| `check_asset_panel` | `src/03_InputData/03_Preprocessing.jl` | docstring | 2 | 686, 713 |
| `rd.Z` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 1 | 586 |
| `rd.Z` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 234 |
| `rd.Z` | `src/03_InputData/03_Preprocessing.jl` | code | 3 | 1052, 1075, 1367 |
| `rd.Z` | `src/03_InputData/03_Preprocessing.jl` | docstring | 1 | 2107 |
| `rd.Z` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 1 | 138 |
| `rd.Z` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 2 | 51, 82 |
| `rd.Z` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 1 | 202 |
| `rd.Z` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 1 | 278 |
| `rd.Z` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 1 | 230 |
| `rd.Z` | `src/13_Prior/01_Base_Prior.jl` | code | 1 | 819 |
| `rd.Z` | `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` | docstring | 3 | 305, 766, 1097 |
| `rd.Z` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 5 | 350, 424, 536, 566, 574 |
| `rd.Z` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 4 | 297, 408, 543, 599 |
| `rd.Z` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `rd.Z` | `src/22_Preselection.jl` | docstring | 3 | 1225, 1229, 1272 |
| `rd.Z` | `test/test06c_setup.jl` | test code | 1 | 379 |
| `rd.Z` | `test/test_06b_asset_panel.jl` | test assertion | 2 | 252, 294 |
| `rd.Z` | `test/test_06b_asset_panel.jl` | test code | 1 | 279 |
| `rd.Z` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 3 | 31, 37, 132 |
| `rd.Z` | `test/test_06c_synthetic_asset_panel.jl` | test code | 4 | 70, 150, 166, 175 |
| `rd.Z` | `test/test_08s_descriptors.jl` | test code | 2 | 23, 42 |
| `rd.Z` | `test/test_08t_exposures.jl` | test code | 1 | 370 |
| `rd.Z` | `test/test_08u_ew_descriptors.jl` | test code | 2 | 97, 403 |
| `rd.Z` | `test/test_13b_feature_routing.jl` | test assertion | 1 | 412 |
| `rd.Z` | `test/test_13e_feature_preselection.jl` | test code | 1 | 235 |
| `rd.Z` | `test/test_53_panel_carrier_census.jl` | test assertion | 2 | 57, 66 |
| `rd.Z` | `test/test_53_panel_carrier_census.jl` | test code | 2 | 4, 190 |
| `any .Z read` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 12 | 236, 284, 311, 328, 329, 360, 451, 586, 645, 649, 652, 729 |
| `any .Z read` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 234 |
| `any .Z read` | `src/03_InputData/01_AssetPanel.jl` | docstring | 1 | 471 |
| `any .Z read` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 2 | 1234, 1239 |
| `any .Z read` | `src/03_InputData/03_Preprocessing.jl` | code | 10 | 608, 609, 630, 632, 1052, 1075, 1367, 2316, 2466, 2598 |
| `any .Z read` | `src/03_InputData/03_Preprocessing.jl` | docstring | 1 | 2107 |
| `any .Z read` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 1 | 138 |
| `any .Z read` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 2 | 51, 82 |
| `any .Z read` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 1 | 202 |
| `any .Z read` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 1 | 278 |
| `any .Z read` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 1 | 230 |
| `any .Z read` | `src/13_Prior/01_Base_Prior.jl` | code | 3 | 818, 819, 1535 |
| `any .Z read` | `src/13_Prior/01_Base_Prior.jl` | docstring | 3 | 1246, 1260, 1274 |
| `any .Z read` | `src/13_Prior/03_FactorPrior.jl` | docstring | 1 | 47 |
| `any .Z read` | `src/13_Prior/08_FactorBlackLittermanPrior.jl` | docstring | 1 | 39 |
| `any .Z read` | `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | code | 1 | 556 |
| `any .Z read` | `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | comment | 1 | 540 |
| `any .Z read` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 414 |
| `any .Z read` | `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` | docstring | 3 | 305, 766, 1097 |
| `any .Z read` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 5 | 350, 424, 536, 566, 574 |
| `any .Z read` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 4 | 297, 408, 543, 599 |
| `any .Z read` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `any .Z read` | `src/22_Preselection.jl` | docstring | 3 | 1225, 1229, 1272 |
| `any .Z read` | `test/test06c_setup.jl` | test code | 1 | 379 |
| `any .Z read` | `test/test_06_preprocessing.jl` | test assertion | 27 | 191, 194, 195, 200, 202, 203, 207, 210, 223, 228, 233, 250, 256, 257, 258, 264, 296, 297, 298, 300, 307, 310, 316, 317, 318, 326, 335 |
| `any .Z read` | `test/test_06_preprocessing.jl` | test code | 2 | 211, 239 |
| `any .Z read` | `test/test_06b_asset_panel.jl` | test assertion | 10 | 200, 201, 202, 204, 205, 206, 207, 252, 264, 294 |
| `any .Z read` | `test/test_06b_asset_panel.jl` | test code | 1 | 279 |
| `any .Z read` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 3 | 31, 37, 132 |
| `any .Z read` | `test/test_06c_synthetic_asset_panel.jl` | test code | 4 | 70, 150, 166, 175 |
| `any .Z read` | `test/test_08s_descriptors.jl` | test code | 2 | 23, 42 |
| `any .Z read` | `test/test_08t_exposures.jl` | test code | 3 | 245, 302, 370 |
| `any .Z read` | `test/test_08u_ew_descriptors.jl` | test code | 2 | 97, 403 |
| `any .Z read` | `test/test_12b_prior_core.jl` | test assertion | 1 | 2782 |
| `any .Z read` | `test/test_12c_feature_prior.jl` | test assertion | 26 | 50, 52, 61, 62, 69, 70, 89, 90, 107, 114, 126, 151, 157, 178, 180, 183, 201, 209, 224, 225, 244, 279, 291, 315, 316, 321 |
| `any .Z read` | `test/test_12c_feature_prior.jl` | test code | 8 | 130, 132, 136, 140, 145, 153, 260, 261 |
| `any .Z read` | `test/test_12d_phylogeny_features.jl` | test assertion | 11 | 119, 120, 125, 127, 140, 141, 154, 159, 231, 232, 733 |
| `any .Z read` | `test/test_12d_phylogeny_features.jl` | test code | 5 | 152, 160, 180, 220, 230 |
| `any .Z read` | `test/test_12d_phylogeny_features.jl` | test comment | 1 | 237 |
| `any .Z read` | `test/test_12e_asset_sets_features.jl` | test assertion | 7 | 182, 198, 200, 260, 264, 601, 637 |
| `any .Z read` | `test/test_12e_asset_sets_features.jl` | test code | 1 | 230 |
| `any .Z read` | `test/test_12f_forward_prior.jl` | test assertion | 1 | 240 |
| `any .Z read` | `test/test_12g_forwarding_rule.jl` | test assertion | 5 | 167, 168, 195, 196, 215 |
| `any .Z read` | `test/test_12g_forwarding_rule.jl` | test code | 2 | 58, 82 |
| `any .Z read` | `test/test_12g_forwarding_rule.jl` | test comment | 1 | 163 |
| `any .Z read` | `test/test_13b_feature_routing.jl` | test assertion | 4 | 61, 204, 412, 417 |
| `any .Z read` | `test/test_13c_feature_views.jl` | test assertion | 13 | 243, 244, 245, 304, 342, 343, 366, 367, 372, 378, 385, 387, 435 |
| `any .Z read` | `test/test_13c_feature_views.jl` | test code | 2 | 439, 494 |
| `any .Z read` | `test/test_13d_feature_collapse.jl` | test assertion | 5 | 425, 426, 471, 474, 508 |
| `any .Z read` | `test/test_13d_feature_collapse.jl` | test code | 1 | 434 |
| `any .Z read` | `test/test_13e_feature_preselection.jl` | test assertion | 5 | 105, 109, 122, 124, 193 |
| `any .Z read` | `test/test_13e_feature_preselection.jl` | test code | 1 | 235 |
| `any .Z read` | `test/test_53_panel_carrier_census.jl` | test assertion | 2 | 57, 66 |
| `any .Z read` | `test/test_53_panel_carrier_census.jl` | test code | 2 | 4, 190 |
| `rd.nz` | `src/03_InputData/03_Preprocessing.jl` | code | 5 | 1050, 1051, 1073, 1074, 1367 |
| `rd.nz` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 1 | 51 |
| `rd.nz` | `src/13_Prior/01_Base_Prior.jl` | code | 2 | 773, 820 |
| `rd.nz` | `src/13_Prior/01_Base_Prior.jl` | docstring | 1 | 789 |
| `rd.nz` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 4 | 349, 351, 573, 584 |
| `rd.nz` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 1 | 554 |
| `rd.nz` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `rd.nz` | `src/22_Preselection.jl` | docstring | 2 | 1229, 1272 |
| `rd.nz` | `test/test06c_setup.jl` | test code | 1 | 379 |
| `rd.nz` | `test/test_06b_asset_panel.jl` | test assertion | 2 | 257, 296 |
| `rd.nz` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 5 | 32, 33, 34, 35, 133 |
| `rd.nz` | `test/test_06c_synthetic_asset_panel.jl` | test code | 4 | 70, 150, 166, 175 |
| `rd.nz` | `test/test_08s_descriptors.jl` | test code | 3 | 23, 43, 44 |
| `rd.nz` | `test/test_08u_ew_descriptors.jl` | test code | 3 | 98, 99, 403 |
| `rd.nz` | `test/test_13b_feature_routing.jl` | test assertion | 1 | 413 |
| `rd.nz` | `test/test_13b_feature_routing.jl` | test code | 1 | 285 |
| `rd.nz` | `test/test_13e_feature_preselection.jl` | test code | 2 | 235, 239 |
| `rd.nz` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 4 |
| `any .nz read` | `src/03_InputData/01_AssetPanel.jl` | docstring | 1 | 471 |
| `any .nz read` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 1 | 1229 |
| `any .nz read` | `src/03_InputData/03_Preprocessing.jl` | code | 12 | 610, 629, 631, 1050, 1051, 1073, 1074, 1367, 2316, 2464, 2465, 2597 |
| `any .nz read` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 1 | 51 |
| `any .nz read` | `src/13_Prior/01_Base_Prior.jl` | code | 2 | 773, 820 |
| `any .nz read` | `src/13_Prior/01_Base_Prior.jl` | docstring | 2 | 752, 789 |
| `any .nz read` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 4 | 349, 351, 573, 584 |
| `any .nz read` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 1 | 554 |
| `any .nz read` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `any .nz read` | `src/22_Preselection.jl` | docstring | 2 | 1229, 1272 |
| `any .nz read` | `test/test06c_setup.jl` | test code | 1 | 379 |
| `any .nz read` | `test/test_06_preprocessing.jl` | test assertion | 7 | 192, 224, 227, 234, 251, 311, 327 |
| `any .nz read` | `test/test_06b_asset_panel.jl` | test assertion | 3 | 199, 257, 296 |
| `any .nz read` | `test/test_06b_asset_panel.jl` | test code | 2 | 272, 279 |
| `any .nz read` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 5 | 32, 33, 34, 35, 133 |
| `any .nz read` | `test/test_06c_synthetic_asset_panel.jl` | test code | 4 | 70, 150, 166, 175 |
| `any .nz read` | `test/test_08s_descriptors.jl` | test code | 3 | 23, 43, 44 |
| `any .nz read` | `test/test_08t_exposures.jl` | test code | 1 | 304 |
| `any .nz read` | `test/test_08u_ew_descriptors.jl` | test code | 3 | 98, 99, 403 |
| `any .nz read` | `test/test_12e_asset_sets_features.jl` | test assertion | 1 | 633 |
| `any .nz read` | `test/test_13b_feature_routing.jl` | test assertion | 1 | 413 |
| `any .nz read` | `test/test_13b_feature_routing.jl` | test code | 1 | 285 |
| `any .nz read` | `test/test_13d_feature_collapse.jl` | test assertion | 3 | 203, 427, 443 |
| `any .nz read` | `test/test_13e_feature_preselection.jl` | test assertion | 2 | 106, 123 |
| `any .nz read` | `test/test_13e_feature_preselection.jl` | test code | 2 | 235, 239 |
| `any .nz read` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 4 |
| `nz keyword / field` | `examples/2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.jl` | example code | 1 | 230 |
| `nz keyword / field` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 5 | 101, 184, 636, 722, 867 |
| `nz keyword / field` | `src/03_InputData/01_AssetPanel.jl` | code | 2 | 758, 761 |
| `nz keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | code | 4 | 1041, 1091, 1264, 1299 |
| `nz keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 1 | 1269 |
| `nz keyword / field` | `src/03_InputData/03_Preprocessing.jl` | code | 22 | 111, 244, 251, 260, 297, 507, 536, 610, 631, 633, 841, 1051, 1055, 1074, 1078, 1367, 1661, 1683, 2316, 2465, 2467, 2597 |
| `nz keyword / field` | `src/03_InputData/03_Preprocessing.jl` | docstring | 7 | 196, 197, 198, 270, 437, 667, 1433 |
| `nz keyword / field` | `src/03_InputData/03_Preprocessing.jl` | field declaration | 2 | 773, 1573 |
| `nz keyword / field` | `src/08_Moments/10_Histogram.jl` | code | 1 | 668 |
| `nz keyword / field` | `src/09_Distance/05_FeatureDistance.jl` | code | 8 | 766, 834, 1278, 1285, 1451, 1453, 1456, 1458 |
| `nz keyword / field` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 3 | 719, 778, 779 |
| `nz keyword / field` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 1231 |
| `nz keyword / field` | `src/13_Prior/01_Base_Prior.jl` | code | 9 | 820, 863, 906, 948, 990, 1035, 1124, 1171, 1212 |
| `nz keyword / field` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 3 | 351, 668, 769 |
| `nz keyword / field` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `nz keyword / field` | `test/test_06_preprocessing.jl` | test assertion | 1 | 238 |
| `nz keyword / field` | `test/test_06_preprocessing.jl` | test code | 24 | 106, 113, 115, 119, 123, 128, 133, 139, 142, 145, 149, 151, 183, 184, 206, 221, 231, 243, 244, 274, 292, 293, 324, 331 |
| `nz keyword / field` | `test/test_06b_asset_panel.jl` | test code | 3 | 268, 272, 279 |
| `nz keyword / field` | `test/test_08t_exposures.jl` | test code | 1 | 304 |
| `nz keyword / field` | `test/test_12c_feature_prior.jl` | test code | 1 | 319 |
| `nz keyword / field` | `test/test_12e_asset_sets_features.jl` | test code | 6 | 79, 256, 350, 386, 632, 648 |
| `nz keyword / field` | `test/test_13b_feature_routing.jl` | test assertion | 6 | 306, 313, 316, 329, 336, 346 |
| `nz keyword / field` | `test/test_13b_feature_routing.jl` | test code | 12 | 33, 178, 320, 341, 349, 355, 375, 378, 388, 396, 405, 423 |
| `nz keyword / field` | `test/test_13c_feature_views.jl` | test code | 6 | 85, 90, 91, 279, 296, 318 |
| `nz keyword / field` | `test/test_13d_feature_collapse.jl` | test code | 7 | 196, 197, 198, 376, 491, 505, 555 |
| `nz keyword / field` | `test/test_13e_feature_preselection.jl` | test code | 5 | 38, 41, 86, 117, 242 |
| `nz keyword / field` | `test/test_53_panel_carrier_census.jl` | test code | 3 | 42, 44, 191 |
| `nz keyword / field` | `user_guide/02_Optimisers.jl` | example code | 1 | 171 |
| `rd.pnl` | `src/03_InputData/03_Preprocessing.jl` | code | 2 | 1053, 1076 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 4 | 132, 134, 189, 191 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 5 | 51, 56, 88, 94, 175 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | code | 4 | 371, 495, 504, 509 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | docstring | 1 | 445 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | code | 3 | 345, 357, 370 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | docstring | 1 | 285 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | docstring | 3 | 240, 583, 587 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/05_EWVolatilityDescriptors.jl` | docstring | 2 | 176, 180 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | code | 2 | 65, 69 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | docstring | 3 | 41, 46, 294 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 2 | 132, 202 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | docstring | 3 | 51, 112, 180 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/09_CompositeExposure.jl` | docstring | 1 | 340 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/10_DerivedExposure.jl` | docstring | 1 | 123 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 3 | 89, 91, 283 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 4 | 69, 74, 107, 238 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | code | 2 | 106, 108 |
| `rd.pnl` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | docstring | 2 | 75, 79 |
| `rd.pnl` | `test/test06c_setup.jl` | test code | 1 | 380 |
| `rd.pnl` | `test/test_06b_asset_panel.jl` | test assertion | 4 | 251, 258, 293, 295 |
| `rd.pnl` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 7 | 38, 39, 40, 84, 135, 136, 167 |
| `rd.pnl` | `test/test_06c_synthetic_asset_panel.jl` | test code | 5 | 58, 91, 101, 148, 174 |
| `rd.pnl` | `test/test_08s_descriptors.jl` | test assertion | 1 | 197 |
| `rd.pnl` | `test/test_08s_descriptors.jl` | test code | 2 | 201, 420 |
| `rd.pnl` | `test/test_08t_exposures.jl` | test code | 6 | 160, 163, 167, 170, 370, 371 |
| `rd.pnl` | `test/test_08t_rolling_descriptors.jl` | test code | 1 | 272 |
| `rd.pnl` | `test/test_08u_ew_descriptors.jl` | test code | 1 | 387 |
| `rd.pnl` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 4 |
| `any .pnl read` | `src/03_InputData/03_Preprocessing.jl` | code | 2 | 1053, 1076 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 4 | 132, 134, 189, 191 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 5 | 51, 56, 88, 94, 175 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | code | 4 | 371, 495, 504, 509 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | docstring | 1 | 445 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | code | 3 | 345, 357, 370 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | docstring | 1 | 285 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | docstring | 3 | 240, 583, 587 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/05_EWVolatilityDescriptors.jl` | docstring | 2 | 176, 180 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | code | 2 | 65, 69 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | docstring | 3 | 41, 46, 294 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 2 | 132, 202 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | docstring | 3 | 51, 112, 180 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/09_CompositeExposure.jl` | docstring | 1 | 340 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/10_DerivedExposure.jl` | docstring | 1 | 123 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 3 | 89, 91, 283 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 4 | 69, 74, 107, 238 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | code | 2 | 106, 108 |
| `any .pnl read` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | docstring | 2 | 75, 79 |
| `any .pnl read` | `test/test06c_setup.jl` | test code | 1 | 380 |
| `any .pnl read` | `test/test_06b_asset_panel.jl` | test assertion | 15 | 209, 210, 211, 212, 217, 238, 239, 251, 256, 258, 262, 263, 282, 293, 295 |
| `any .pnl read` | `test/test_06b_asset_panel.jl` | test code | 6 | 4, 267, 269, 273, 277, 278 |
| `any .pnl read` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 7 | 38, 39, 40, 84, 135, 136, 167 |
| `any .pnl read` | `test/test_06c_synthetic_asset_panel.jl` | test code | 5 | 58, 91, 101, 148, 174 |
| `any .pnl read` | `test/test_08s_descriptors.jl` | test assertion | 1 | 197 |
| `any .pnl read` | `test/test_08s_descriptors.jl` | test code | 2 | 201, 420 |
| `any .pnl read` | `test/test_08t_exposures.jl` | test code | 8 | 160, 163, 167, 170, 245, 304, 370, 371 |
| `any .pnl read` | `test/test_08t_rolling_descriptors.jl` | test code | 1 | 272 |
| `any .pnl read` | `test/test_08u_ew_descriptors.jl` | test code | 1 | 387 |
| `any .pnl read` | `test/test_53_panel_carrier_census.jl` | test code | 1 | 4 |
| `pr.nz` | `src/03_InputData/03_Preprocessing.jl` | code | 7 | 610, 629, 631, 2316, 2464, 2465, 2597 |
| `pr.nz` | `src/13_Prior/01_Base_Prior.jl` | docstring | 1 | 752 |
| `pr.Z on a PricesResult / prior` | `src/03_InputData/03_Preprocessing.jl` | code | 7 | 608, 609, 630, 632, 2316, 2466, 2598 |
| `pr.Z on a PricesResult / prior` | `src/13_Prior/01_Base_Prior.jl` | code | 2 | 818, 1535 |
| `pr.Z on a PricesResult / prior` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 414 |
| `pr.Z on a PricesResult / prior` | `test/test_12b_prior_core.jl` | test assertion | 1 | 2782 |
| `pr.Z on a PricesResult / prior` | `test/test_12c_feature_prior.jl` | test assertion | 3 | 61, 62, 201 |
| `pr.Z on a PricesResult / prior` | `test/test_12c_feature_prior.jl` | test code | 2 | 260, 261 |
| `pr.Z on a PricesResult / prior` | `test/test_12d_phylogeny_features.jl` | test assertion | 4 | 119, 120, 141, 733 |
| `pr.Z on a PricesResult / prior` | `test/test_12d_phylogeny_features.jl` | test code | 1 | 220 |
| `pr.Z on a PricesResult / prior` | `test/test_12d_phylogeny_features.jl` | test comment | 1 | 237 |
| `pr.Z on a PricesResult / prior` | `test/test_12e_asset_sets_features.jl` | test assertion | 5 | 182, 260, 264, 601, 637 |
| `pr.Z on a PricesResult / prior` | `test/test_12g_forwarding_rule.jl` | test code | 2 | 58, 82 |
| `Z keyword / field` | `docs/src/api/08_Moments/38_CrossSectionalRegression.md` | doc page | 1 | 25 |
| `Z keyword / field` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 9 | 100, 155, 184, 368, 474, 554, 636, 723, 867 |
| `Z keyword / field` | `src/03_InputData/01_AssetPanel.jl` | code | 1 | 761 |
| `Z keyword / field` | `src/03_InputData/01_AssetPanel.jl` | field declaration | 1 | 759 |
| `Z keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | code | 8 | 940, 944, 955, 1131, 1169, 1174, 1263, 1264 |
| `Z keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 7 | 454, 905, 906, 907, 1139, 1140, 1186 |
| `Z keyword / field` | `src/03_InputData/03_Preprocessing.jl` | code | 25 | 171, 181, 251, 260, 342, 345, 348, 351, 507, 609, 610, 632, 633, 773, 1052, 1055, 1075, 1078, 1367, 1660, 1683, 2316, 2466, 2467, 2598 |
| `Z keyword / field` | `src/03_InputData/03_Preprocessing.jl` | docstring | 18 | 120, 121, 122, 196, 197, 198, 301, 302, 303, 304, 305, 355, 356, 357, 358, 438, 668, 1434 |
| `Z keyword / field` | `src/03_InputData/03_Preprocessing.jl` | field declaration | 3 | 537, 842, 1574 |
| `Z keyword / field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | code | 1 | 1496 |
| `Z keyword / field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | docstring | 1 | 1431 |
| `Z keyword / field` | `src/08_Moments/38_CrossSectionalRegression.jl` | code | 5 | 513, 713, 766, 835, 902 |
| `Z keyword / field` | `src/08_Moments/38_CrossSectionalRegression.jl` | docstring | 11 | 574, 613, 674, 710, 723, 749, 770, 797, 819, 858, 888 |
| `Z keyword / field` | `src/08_Moments/38_CrossSectionalRegression.jl` | field declaration | 1 | 634 |
| `Z keyword / field` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 1 | 138 |
| `Z keyword / field` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 2 | 210, 278 |
| `Z keyword / field` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 2 | 184, 185 |
| `Z keyword / field` | `src/09_Distance/05_FeatureDistance.jl` | code | 26 | 830, 834, 882, 926, 957, 995, 1038, 1078, 1081, 1084, 1087, 1114, 1153, 1195, 1199, 1203, 1208, 1277, 1280, 1284, 1287, 1344, 1348, 1392, 1450, 1455 |
| `Z keyword / field` | `src/09_Distance/05_FeatureDistance.jl` | docstring | 20 | 777, 779, 847, 888, 934, 965, 1013, 1043, 1091, 1119, 1163, 1228, 1229, 1262, 1292, 1293, 1326, 1353, 1409, 1411 |
| `Z keyword / field` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 7 | 843, 895, 1020, 1025, 1068, 1115, 1232 |
| `Z keyword / field` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 5 | 537, 806, 856, 977, 1076 |
| `Z keyword / field` | `src/13_Prior/01_Base_Prior.jl` | code | 10 | 863, 906, 948, 990, 1035, 1124, 1171, 1212, 1404, 1535 |
| `Z keyword / field` | `src/13_Prior/01_Base_Prior.jl` | docstring | 1 | 1239 |
| `Z keyword / field` | `src/13_Prior/01_Base_Prior.jl` | field declaration | 1 | 1471 |
| `Z keyword / field` | `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | code | 1 | 556 |
| `Z keyword / field` | `src/13_Prior/11_MeucciEntropyPoolingPrior.jl` | code | 2 | 841, 981 |
| `Z keyword / field` | `src/13_Prior/12_EntropyPoolingPrior.jl` | code | 2 | 3650, 3740 |
| `Z keyword / field` | `src/13_Prior/13_OpinionPoolingPrior.jl` | code | 1 | 534 |
| `Z keyword / field` | `src/13_Prior/15_FeaturePrior.jl` | code | 3 | 333, 843, 846 |
| `Z keyword / field` | `src/20_Optimisation/01_Base_Optimisation.jl` | code | 4 | 2941, 2946, 2960, 2963 |
| `Z keyword / field` | `src/20_Optimisation/01_Base_Optimisation.jl` | docstring | 6 | 2895, 2896, 2897, 2898, 2899, 2900 |
| `Z keyword / field` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 7 | 350, 503, 511, 576, 661, 668, 769 |
| `Z keyword / field` | `src/20_Optimisation/16_Base_MetaOptimisation.jl` | docstring | 3 | 470, 471, 472 |
| `Z keyword / field` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `Z keyword / field` | `test/test_06_preprocessing.jl` | test code | 24 | 114, 116, 121, 124, 129, 134, 140, 143, 146, 150, 152, 183, 185, 206, 221, 231, 239, 243, 244, 274, 292, 293, 324, 332 |
| `Z keyword / field` | `test/test_06b_asset_panel.jl` | test code | 4 | 180, 269, 272, 279 |
| `Z keyword / field` | `test/test_06d_cross_sectional_transforms.jl` | test code | 1 | 571 |
| `Z keyword / field` | `test/test_07_linear_algebra.jl` | test code | 1 | 266 |
| `Z keyword / field` | `test/test_08k_regression_algorithms.jl` | test code | 1 | 331 |
| `Z keyword / field` | `test/test_08n_cross_sectional_regression.jl` | test code | 1 | 70 |
| `Z keyword / field` | `test/test_08q_cross_sectional_factor_model.jl` | test code | 1 | 209 |
| `Z keyword / field` | `test/test_08s_descriptors.jl` | test code | 1 | 42 |
| `Z keyword / field` | `test/test_08t_exposures.jl` | test code | 2 | 302, 304 |
| `Z keyword / field` | `test/test_08u_ew_descriptors.jl` | test code | 1 | 97 |
| `Z keyword / field` | `test/test_10_uncertainty_set.jl` | test code | 1 | 908 |
| `Z keyword / field` | `test/test_12c_feature_prior.jl` | test assertion | 2 | 224, 225 |
| `Z keyword / field` | `test/test_12c_feature_prior.jl` | test code | 11 | 17, 176, 222, 228, 229, 230, 231, 237, 240, 241, 319 |
| `Z keyword / field` | `test/test_12d_phylogeny_features.jl` | test code | 4 | 380, 498, 706, 725 |
| `Z keyword / field` | `test/test_12e_asset_sets_features.jl` | test code | 11 | 47, 98, 248, 256, 340, 414, 626, 632, 649, 677, 710 |
| `Z keyword / field` | `test/test_12f_forward_prior.jl` | test assertion | 1 | 240 |
| `Z keyword / field` | `test/test_12f_forward_prior.jl` | test code | 1 | 239 |
| `Z keyword / field` | `test/test_12g_forwarding_rule.jl` | test code | 2 | 58, 82 |
| `Z keyword / field` | `test/test_13_phylogeny.jl` | test code | 3 | 403, 426, 555 |
| `Z keyword / field` | `test/test_13b_feature_routing.jl` | test code | 3 | 33, 178, 423 |
| `Z keyword / field` | `test/test_13c_feature_views.jl` | test code | 12 | 30, 32, 34, 36, 72, 85, 90, 91, 279, 296, 318, 386 |
| `Z keyword / field` | `test/test_13d_feature_collapse.jl` | test code | 12 | 31, 33, 35, 37, 69, 196, 197, 198, 377, 491, 505, 555 |
| `Z keyword / field` | `test/test_13e_feature_preselection.jl` | test code | 5 | 39, 41, 86, 117, 242 |
| `Z keyword / field` | `test/test_36_redundancy_selection.jl` | test code | 2 | 207, 208 |
| `Z keyword / field` | `test/test_53_panel_carrier_census.jl` | test code | 4 | 41, 44, 83, 190 |
| `Z keyword / field` | `user_guide/02_Optimisers.jl` | example code | 1 | 172 |
| `pnl keyword / field` | `src/03_InputData/01_AssetPanel.jl` | code | 4 | 636, 708, 712, 758 |
| `pnl keyword / field` | `src/03_InputData/01_AssetPanel.jl` | docstring | 5 | 594, 621, 674, 717, 718 |
| `pnl keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | code | 1 | 1265 |
| `pnl keyword / field` | `src/03_InputData/02_AssetPanelBuilder.jl` | docstring | 1 | 1186 |
| `pnl keyword / field` | `src/03_InputData/03_Preprocessing.jl` | code | 8 | 774, 1053, 1055, 1076, 1078, 1575, 1662, 1683 |
| `pnl keyword / field` | `src/03_InputData/03_Preprocessing.jl` | docstring | 2 | 669, 1435 |
| `pnl keyword / field` | `src/03_InputData/03_Preprocessing.jl` | field declaration | 1 | 843 |
| `pnl keyword / field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | code | 1 | 1496 |
| `pnl keyword / field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | docstring | 1 | 1431 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 3 | 132, 189, 267 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 2 | 227, 249 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | code | 3 | 631, 646, 654 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/05_EWVolatilityDescriptors.jl` | code | 1 | 211 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | code | 3 | 65, 346, 372 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 3 | 158, 269, 272 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | docstring | 3 | 136, 242, 243 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 1 | 89 |
| `pnl keyword / field` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | code | 1 | 106 |
| `pnl keyword / field` | `test/test06c_setup.jl` | test code | 1 | 422 |
| `pnl keyword / field` | `test/test_06b_asset_panel.jl` | test code | 5 | 65, 267, 269, 273, 280 |
| `pnl keyword / field` | `test/test_06d_cross_sectional_transforms.jl` | test code | 1 | 570 |
| `pnl keyword / field` | `test/test_08t_exposures.jl` | test code | 1 | 304 |
| `dict keys naming nz / Z / pnl` | `src/01_Base/01_DocstringDictionaries.jl` | dict text | 2 | 233, 234 |

Sites in the table: 1354.

### Mechanism 5: `ReturnsResult(` and `PricesResult(` construction sites that pass `nz`, `Z` or `pnl`

#### `ReturnsResult(; nz = …)`: 53 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 4 | 184, 636, 720, 867 |
| `src/03_InputData/03_Preprocessing.jl` | code | 4 | 1054, 1077, 1366, 1682 |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 2 | 667, 768 |
| `test/test_06_preprocessing.jl` | test assertion | 1 | 238 |
| `test/test_06_preprocessing.jl` | test code | 16 | 113, 115, 118, 123, 127, 132, 139, 141, 145, 183, 184, 205, 221, 231, 323, 331 |
| `test/test_06b_asset_panel.jl` | test code | 3 | 268, 271, 279 |
| `test/test_08t_exposures.jl` | test code | 1 | 304 |
| `test/test_12c_feature_prior.jl` | test code | 1 | 319 |
| `test/test_12e_asset_sets_features.jl` | test code | 3 | 255, 631, 648 |
| `test/test_13b_feature_routing.jl` | test code | 3 | 32, 177, 423 |
| `test/test_13c_feature_views.jl` | test code | 3 | 85, 89, 91 |
| `test/test_13d_feature_collapse.jl` | test code | 6 | 195, 197, 198, 376, 491, 504 |
| `test/test_13e_feature_preselection.jl` | test code | 4 | 41, 86, 117, 242 |
| `test/test_53_panel_carrier_census.jl` | test code | 1 | 44 |
| `user_guide/02_Optimisers.jl` | example code | 1 | 170 |

#### `ReturnsResult(; Z = …)`: 53 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 4 | 184, 636, 720, 867 |
| `src/03_InputData/03_Preprocessing.jl` | code | 4 | 1054, 1077, 1366, 1682 |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | code | 2 | 667, 768 |
| `test/test_06_preprocessing.jl` | test assertion | 1 | 238 |
| `test/test_06_preprocessing.jl` | test code | 16 | 113, 115, 120, 123, 127, 132, 139, 141, 145, 183, 184, 205, 221, 231, 323, 331 |
| `test/test_06b_asset_panel.jl` | test code | 3 | 268, 271, 279 |
| `test/test_08t_exposures.jl` | test code | 1 | 304 |
| `test/test_12c_feature_prior.jl` | test code | 1 | 319 |
| `test/test_12e_asset_sets_features.jl` | test code | 3 | 255, 631, 648 |
| `test/test_13b_feature_routing.jl` | test code | 3 | 32, 177, 423 |
| `test/test_13c_feature_views.jl` | test code | 3 | 85, 89, 91 |
| `test/test_13d_feature_collapse.jl` | test code | 6 | 195, 197, 198, 376, 491, 504 |
| `test/test_13e_feature_preselection.jl` | test code | 4 | 41, 86, 117, 242 |
| `test/test_53_panel_carrier_census.jl` | test code | 1 | 44 |
| `user_guide/02_Optimisers.jl` | example code | 1 | 170 |

#### `ReturnsResult(; pnl = …)`: 8 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `src/03_InputData/03_Preprocessing.jl` | code | 3 | 1054, 1077, 1682 |
| `test/test_06b_asset_panel.jl` | test code | 4 | 267, 268, 271, 279 |
| `test/test_08t_exposures.jl` | test code | 1 | 304 |

Every `ReturnsResult(` construction site: 280.

#### `PricesResult(; nz = …)`: 13 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `src/03_InputData/03_Preprocessing.jl` | code | 4 | 610, 633, 2467, 2597 |
| `test/test_06_preprocessing.jl` | test code | 6 | 149, 151, 243, 244, 292, 293 |
| `test/test_13c_feature_views.jl` | test code | 3 | 279, 296, 318 |

#### `PricesResult(; Z = …)`: 13 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `src/03_InputData/03_Preprocessing.jl` | code | 4 | 610, 633, 2467, 2597 |
| `test/test_06_preprocessing.jl` | test code | 6 | 149, 151, 243, 244, 292, 293 |
| `test/test_13c_feature_views.jl` | test code | 3 | 279, 296, 318 |

#### `PricesResult(; pnl = …)`: 0 sites

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |

Every `PricesResult(` construction site: 76.

### Mechanism 5: `port_opt_view` methods that slice `Z`, `nz` or `pnl`

- `src/13_Prior/01_Base_Prior.jl:1527` `function port_opt_view(pr::LowOrderPrior, i, args...)::LowOrderPrior` touches `Z`, `feature_matrix_view`
- `src/03_InputData/03_Preprocessing.jl:602` `function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},` touches `Z`, `nz`, `feature_matrix_view`, `feature_row_indices`
- `src/03_InputData/03_Preprocessing.jl:612` `function port_opt_view(pr::PricesResult, i::AbstractVector{<:Dates.AbstractTime},` touches `Z`, `nz`, `feature_matrix_view`, `feature_row_indices`, `features_are_assets`
- `src/03_InputData/03_Preprocessing.jl:1043` `function port_opt_view(rd::ReturnsResult, i)` touches `Z`, `nz`, `pnl`, `feature_matrix_view`, `features_are_assets`
- `src/03_InputData/03_Preprocessing.jl:1057` `function port_opt_view(rd::ReturnsResult, i, j, k = :)` touches `Z`, `nz`, `pnl`, `feature_matrix_view`, `features_are_assets`
- `src/03_InputData/01_AssetPanel.jl:708` `function port_opt_view(pnl::AssetPanel, i)` touches `pnl`
- `src/03_InputData/01_AssetPanel.jl:712` `function port_opt_view(pnl::AssetPanel, i, j, ::Any = :)` touches `pnl`

### Mechanism 6: `UniverseSets.zkey` and `feature_universe`

108 sites. The field is declared at `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl:678`,
taken by the inner constructor at `:686` and the keyword constructor at `:732` with the default
`"nz"`, checked for prefix collisions in the `knames` tuple at `:691`–`:692` and for a duplicate
node at `:698`–`:700`, carried through `port_opt_view` at `:783`, and read by
`feature_universe` at `:899`. The `field_dict` key `:us_zkey` at `01_DocstringDictionaries.jl:738`
has that one user. `feature_universe` is called twice, both in `06_AssetSetsMatrix.jl`
(`:1231` and `:1385`), and its API entry is `02_LinearConstraintGeneration.md:48`.

Thirteen `UniverseSets(` constructions pass `zkey`: the view at `:783`, two docstring examples
in `06_AssetSetsMatrix.jl` (`:1202`, `:1364`), two in `test_01_structs.jl`, seven in
`test_12e_asset_sets_features.jl`, and one in the example at `16_Feature_Distance_Clustering.jl:249`.
Eight printed `UniverseSets` show a `zkey ┼ String: "nz"` row; the list is after the table.

### Mechanism 6: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zkey` | 3 | 0 | 37 | 16 | 1 | 1 | 0 | 8 | 19 | 2 | 0 | 87 |
| `us_zkey dict key` | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `feature_universe` | 1 | 0 | 12 | 2 | 0 | 0 | 0 | 1 | 2 | 0 | 1 | 19 |

Total: 108.

### Mechanism 6: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `zkey` | `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 2 | 244, 249 |
| `zkey` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 738 |
| `zkey` | `src/01_Base/01_DocstringDictionaries.jl` | dict text | 1 | 751 |
| `zkey` | `src/01_Base/08_TypeAliases.jl` | docstring | 1 | 336 |
| `zkey` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | code | 15 | 690, 691, 692, 698, 699, 700, 702, 723, 726, 734, 783, 900, 901, 902, 903 |
| `zkey` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | docstring | 11 | 464, 568, 578, 595, 605, 606, 620, 634, 750, 875, 886 |
| `zkey` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | field declaration | 3 | 678, 686, 732 |
| `zkey` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 1 | 847 |
| `zkey` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 18 | 584, 757, 763, 828, 880, 1001, 1099, 1151, 1181, 1188, 1195, 1197, 1202, 1339, 1351, 1355, 1359, 1364 |
| `zkey` | `src/13_Prior/06_BlackLittermanPrior.jl` | docstring | 1 | 127 |
| `zkey` | `src/13_Prior/07_BayesianBlackLittermanPrior.jl` | docstring | 1 | 175 |
| `zkey` | `src/13_Prior/08_FactorBlackLittermanPrior.jl` | docstring | 1 | 143 |
| `zkey` | `src/13_Prior/09_AugmentedBlackLittermanPrior.jl` | docstring | 1 | 251 |
| `zkey` | `src/13_Prior/11_MeucciEntropyPoolingPrior.jl` | docstring | 1 | 136 |
| `zkey` | `src/13_Prior/12_EntropyPoolingPrior.jl` | docstring | 1 | 3294 |
| `zkey` | `src/13_Prior/15_FeaturePrior.jl` | docstring | 1 | 516 |
| `zkey` | `test/test_01_structs.jl` | test assertion | 4 | 2236, 2243, 2358, 2388 |
| `zkey` | `test/test_01_structs.jl` | test code | 4 | 2318, 2334, 2360, 2371 |
| `zkey` | `test/test_01_structs.jl` | test comment | 4 | 2253, 2262, 2316, 2317 |
| `zkey` | `test/test_12b_prior_core.jl` | test code | 1 | 1828 |
| `zkey` | `test/test_12e_asset_sets_features.jl` | test assertion | 4 | 549, 569, 570, 585 |
| `zkey` | `test/test_12e_asset_sets_features.jl` | test code | 8 | 311, 431, 515, 542, 571, 572, 618, 703 |
| `zkey` | `test/test_12e_asset_sets_features.jl` | test comment | 2 | 543, 567 |
| `us_zkey dict key` | `src/01_Base/01_DocstringDictionaries.jl` | dict key | 1 | 738 |
| `us_zkey dict key` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | docstring | 1 | 676 |
| `feature_universe` | `docs/src/api/12_ConstraintGeneration/02_LinearConstraintGeneration.md` | doc page | 1 | 48 |
| `feature_universe` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | definition | 1 | 899 |
| `feature_universe` | `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | docstring | 3 | 645, 820, 873 |
| `feature_universe` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | code | 2 | 1231, 1385 |
| `feature_universe` | `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 9 | 109, 121, 132, 1161, 1188, 1222, 1345, 1355, 1380 |
| `feature_universe` | `test/test_18k_constraints.jl` | test assertion | 1 | 862 |
| `feature_universe` | `test/test_18k_constraints.jl` | test code | 1 | 860 |
| `feature_universe` | `test/test_18k_constraints.jl` | test comment | 1 | 854 |

Sites in the table: 108.

### Mechanism 6: the `zkey` row of every printed `UniverseSets`

- `src/01_Base/08_TypeAliases.jl:336`
- `src/13_Prior/06_BlackLittermanPrior.jl:127`
- `src/13_Prior/12_EntropyPoolingPrior.jl:3294`
- `src/13_Prior/07_BayesianBlackLittermanPrior.jl:175`
- `src/13_Prior/08_FactorBlackLittermanPrior.jl:143`
- `src/13_Prior/09_AugmentedBlackLittermanPrior.jl:251`
- `src/13_Prior/11_MeucciEntropyPoolingPrior.jl:136`
- `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl:634`

Printed `zkey` rows: 8.

### Mechanism 6: `UniverseSets(` construction sites that pass `zkey`

| File | Kind | Sites | Lines |
| --- | --- | ---: | --- |
| `examples/3_optimisers/16_Feature_Distance_Clustering.jl` | example code | 1 | 249 |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | code | 1 | 781 |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | docstring | 2 | 1202, 1364 |
| `test/test_01_structs.jl` | test code | 2 | 2331, 2369 |
| `test/test_12e_asset_sets_features.jl` | test code | 7 | 311, 431, 515, 571, 572, 618, 703 |

Sites: 13.

### Mechanism 7: `ClusterGroups`' read of `rd.Z`

One code line, `src/22_Preselection.jl:1298`, and four docstring lines that explain it (`:1225`,
`:1229`, `:1272`, `:1277`). The struct carries no `z_src` field, as ADR 0045's third amendment
says, so nothing else in the file changes. The tests are `test_13e_feature_preselection.jl` (14
sites) and `test_36_redundancy_selection.jl` (9, none of which name a feature). The example
`examples/1_foundations/03_Asset_Preselection.jl` uses `ClusterGroups` with a returns distance
at `:286`, `:291` and `:296` and needs no change.

### Mechanism 7: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `rd.Z / rd.nz / z_src in 22_Preselection.jl` | `src/22_Preselection.jl` | code | 1 | 1298 |
| `rd.Z / rd.nz / z_src in 22_Preselection.jl` | `src/22_Preselection.jl` | docstring | 4 | 1225, 1229, 1272, 1277 |
| `ClusterGroups` | `docs/src/api/22_AssetSelection.md` | doc page | 4 | 37, 39, 41, 48 |
| `ClusterGroups` | `docs/src/api/26_TypeHierarchy.md` | doc page | 1 | 613 |
| `ClusterGroups` | `examples/1_foundations/03_Asset_Preselection.jl` | example code | 3 | 286, 291, 296 |
| `ClusterGroups` | `test/test_13e_feature_preselection.jl` | test assertion | 3 | 55, 56, 57 |
| `ClusterGroups` | `test/test_13e_feature_preselection.jl` | test code | 10 | 6, 12, 45, 49, 51, 176, 223, 248, 255, 271 |
| `ClusterGroups` | `test/test_13e_feature_preselection.jl` | test comment | 1 | 268 |
| `ClusterGroups` | `test/test_36_redundancy_selection.jl` | test assertion | 3 | 23, 146, 158 |
| `ClusterGroups` | `test/test_36_redundancy_selection.jl` | test code | 4 | 22, 127, 135, 152 |
| `ClusterGroups` | `test/test_36_redundancy_selection.jl` | test comment | 2 | 21, 150 |

Sites in the table: 36.

### Mechanism 8: map 643's readers

131 sites. `panel_field_values` is defined at
`src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl:131` and `:152`, and is the seam every
descriptor reads through: 19 code sites in five files of `42_FactorExposures/`. `panel_field` is
defined at `03_InputData/01_AssetPanel.jl:636` and read at `04_CrossSectionalTransforms.jl:1498`,
`01_Base_Descriptor.jl:135` and `11_OneHotExposure.jl:92`. `rd.pnl` is read at 20 code lines in
seven files of `42_FactorExposures/`, and at 26 docstring lines in eleven; the only reads outside that
directory are the two views at `03_Preprocessing.jl:1053` and `:1076`. `rd.Z` is read directly
at three code lines, `01_Base_Descriptor.jl:138`, `08_Base_Exposure.jl:202` and
`11_OneHotExposure.jl:278`, as the map says. `04_CrossSectionalTransforms.jl` reads no `rd.pnl`
and no `rd.Z`; it reads the panel through `panel_field` at `:1498`, inside
`cross_sectional_groups(pnl::AssetPanel, Z::Arr3Num, name)` at `:1496`, which
`08_Base_Exposure.jl:202` calls with `rd.pnl` and `rd.Z`.

### Mechanism 8: per name

| Name | src def | src export | src docstring | src code | dict key | dict text | ext | test assertion | test other | example | doc page | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `panel_field_values` | 2 | 0 | 29 | 19 | 0 | 0 | 0 | 1 | 7 | 0 | 1 | 59 |
| `panel_field` | 2 | 0 | 7 | 3 | 0 | 0 | 0 | 5 | 1 | 0 | 1 | 19 |
| `rd.pnl (map 643 readers)` | 0 | 0 | 26 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 46 |
| `rd.Z (map 643 readers)` | 0 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| `rd.nz (map 643 readers)` | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

Total: 131.

### Mechanism 8: every site

| Name | File | Kind | Sites | Lines |
| --- | --- | --- | ---: | --- |
| `panel_field_values` | `docs/src/api/08_Moments/42_FactorExposures/01_Base_Descriptor.md` | doc page | 1 | 13 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 2 | 156, 158 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | definition | 2 | 131, 152 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 7 | 65, 72, 73, 112, 117, 167, 185 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | code | 5 | 370, 414, 493, 499, 508 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | docstring | 10 | 7, 27, 319, 344, 363, 387, 404, 430, 449, 487 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | code | 5 | 338, 349, 350, 361, 362 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | docstring | 3 | 270, 289, 332 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | code | 6 | 261, 265, 269, 271, 655, 656 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | docstring | 5 | 234, 235, 245, 255, 588 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 1 | 131 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | docstring | 3 | 103, 117, 127 |
| `panel_field_values` | `src/08_Moments/42_FactorExposures/09_CompositeExposure.jl` | docstring | 1 | 344 |
| `panel_field_values` | `test/test_08s_descriptors.jl` | test assertion | 1 | 194 |
| `panel_field_values` | `test/test_08s_descriptors.jl` | test code | 6 | 169, 175, 179, 181, 184, 191 |
| `panel_field_values` | `test/test_08t_exposures.jl` | test code | 1 | 247 |
| `panel_field` | `docs/src/api/03_Preprocessing.md` | doc page | 1 | 50 |
| `panel_field` | `src/03_InputData/01_AssetPanel.jl` | definition | 1 | 636 |
| `panel_field` | `src/03_InputData/01_AssetPanel.jl` | docstring | 3 | 515, 594, 625 |
| `panel_field` | `src/03_InputData/01_AssetPanel.jl` | field declaration | 1 | 776 |
| `panel_field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | code | 1 | 1498 |
| `panel_field` | `src/03_InputData/04_CrossSectionalTransforms.jl` | docstring | 1 | 1482 |
| `panel_field` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 1 | 135 |
| `panel_field` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 2 | 82, 126 |
| `panel_field` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 1 | 92 |
| `panel_field` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 1 | 85 |
| `panel_field` | `test/test_06b_asset_panel.jl` | test assertion | 3 | 67, 209, 210 |
| `panel_field` | `test/test_06b_asset_panel.jl` | test code | 1 | 68 |
| `panel_field` | `test/test_06c_synthetic_asset_panel.jl` | test assertion | 2 | 39, 40 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 4 | 132, 134, 189, 191 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 5 | 51, 56, 88, 94, 175 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | code | 4 | 371, 495, 504, 509 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/02_PanelFieldDescriptors.jl` | docstring | 1 | 445 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | code | 3 | 345, 357, 370 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/03_LagDescriptors.jl` | docstring | 1 | 285 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/04_EWMeanDescriptors.jl` | docstring | 3 | 240, 583, 587 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/05_EWVolatilityDescriptors.jl` | docstring | 2 | 176, 180 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | code | 2 | 65, 69 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/07_RollingDescriptors.jl` | docstring | 3 | 41, 46, 294 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 2 | 132, 202 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | docstring | 3 | 51, 112, 180 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/09_CompositeExposure.jl` | docstring | 1 | 340 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/10_DerivedExposure.jl` | docstring | 1 | 123 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 3 | 89, 91, 283 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 4 | 69, 74, 107, 238 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | code | 2 | 106, 108 |
| `rd.pnl (map 643 readers)` | `src/08_Moments/42_FactorExposures/12_ConstantExposure.jl` | docstring | 2 | 75, 79 |
| `rd.Z (map 643 readers)` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | code | 1 | 138 |
| `rd.Z (map 643 readers)` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 2 | 51, 82 |
| `rd.Z (map 643 readers)` | `src/08_Moments/42_FactorExposures/08_Base_Exposure.jl` | code | 1 | 202 |
| `rd.Z (map 643 readers)` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | code | 1 | 278 |
| `rd.Z (map 643 readers)` | `src/08_Moments/42_FactorExposures/11_OneHotExposure.jl` | docstring | 1 | 230 |
| `rd.nz (map 643 readers)` | `src/08_Moments/42_FactorExposures/01_Base_Descriptor.jl` | docstring | 1 | 51 |

Sites in the table: 131.

## 3. The tests

A removed name is any symbol of mechanisms 1 to 7 (the list is in the appendix), a
`FeatureDistance(` keyword `sets =`, or a carrier feature field: `.Z`, `.nz`, `Z =`, `nz =`. The
symbol and carrier-field halves are split out because the carrier ticket rewrites the second
half and the producer and selector tickets rewrite the first. A reshaped name is one the
destination keeps under a changed contract: `.pnl`, `pnl =`, `panel_field_values`,
`panel_field`, `check_asset_panel`, `prepare_outer_rd`, `rebuild_returns_result`, `sel`,
`strict`, `FeatureDistance` and `Pr_RR`. A file appears when it holds at least one of either, or
is one of the thirteen files of ground truth 11.

| Test file | `@test` lines | naming a removed name | of which a symbol | of which a carrier field only | naming a reshaped name only | `@testset` blocks | file lines |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test/test_01_structs.jl` | 221 | 4 | 4 | 0 | 0 | 26 | 2625 |
| `test/test_05_tools.jl` | 290 | 0 | 0 | 0 | 3 | 7 | 901 |
| `test/test_06_preprocessing.jl` | 109 | 39 | 5 | 34 | 0 | 17 | 576 |
| `test/test_06b_asset_panel.jl` | 75 | 13 | 0 | 13 | 16 | 8 | 298 |
| `test/test_06c_synthetic_asset_panel.jl` | 84 | 8 | 0 | 8 | 7 | 9 | 253 |
| `test/test_08s_descriptors.jl` | 152 | 0 | 0 | 0 | 2 | 27 | 527 |
| `test/test_09c_risk_input_kind.jl` | 15 | 0 | 0 | 0 | 1 | 6 | 221 |
| `test/test_09f_calibration_slot.jl` | 161 | 0 | 0 | 0 | 5 | 18 | 980 |
| `test/test_09h_ambiguity_calibration.jl` | 334 | 0 | 0 | 0 | 2 | 25 | 1554 |
| `test/test_12b_prior_core.jl` (ground truth 11) | 603 | 1 | 0 | 1 | 0 | 42 | 2922 |
| `test/test_12c_feature_prior.jl` (ground truth 11) | 56 | 31 | 5 | 26 | 0 | 9 | 337 |
| `test/test_12d_phylogeny_features.jl` (ground truth 11) | 198 | 28 | 20 | 8 | 1 | 19 | 763 |
| `test/test_12e_asset_sets_features.jl` (ground truth 11) | 157 | 44 | 37 | 7 | 1 | 20 | 803 |
| `test/test_12f_forward_prior.jl` | 70 | 1 | 0 | 1 | 0 | 9 | 354 |
| `test/test_12g_forwarding_rule.jl` (ground truth 11) | 102 | 6 | 1 | 5 | 0 | 8 | 421 |
| `test/test_12i_cross_sectional_factor_carrier.jl` (ground truth 11) | 58 | 2 | 2 | 0 | 0 | 9 | 331 |
| `test/test_13_phylogeny.jl` (ground truth 11) | 830 | 1 | 1 | 0 | 12 | 131 | 4481 |
| `test/test_13b_feature_routing.jl` (ground truth 11) | 86 | 27 | 16 | 11 | 11 | 21 | 448 |
| `test/test_13c_feature_views.jl` (ground truth 11) | 82 | 13 | 0 | 13 | 2 | 10 | 531 |
| `test/test_13d_feature_collapse.jl` (ground truth 11) | 90 | 11 | 4 | 7 | 0 | 13 | 571 |
| `test/test_13e_feature_preselection.jl` (ground truth 11) | 48 | 16 | 9 | 7 | 1 | 11 | 280 |
| `test/test_18k_constraints.jl` | 189 | 1 | 1 | 0 | 0 | 8 | 935 |
| `test/test_37_time_dependent_constraints.jl` (ground truth 11) | 411 | 0 | 0 | 0 | 0 | 70 | 1962 |
| `test/test_53_panel_carrier_census.jl` (ground truth 11) | 32 | 7 | 5 | 2 | 1 | 6 | 218 |
| Total over the 24 files | 4453 | 253 | 110 | 143 | 65 | | |

Ground truth 11: 13 files, 2753 `@test` lines, 187 naming a removed name (100 a symbol, 87 a carrier field only), 29 naming a reshaped name only.
Outside ground truth 11: 11 files, 1700 `@test` lines, 66 naming a removed name (10 a symbol, 56 a carrier field only), 36 naming a reshaped name only.

Three observations for the build tickets:

- **Eight files are wholly about the stack** and are rewritten or replaced rather than edited:
  `test_12c_feature_prior.jl` (337 lines, 9 top-level testsets), `test_12d_phylogeny_features.jl`
  (763, 19), `test_12e_asset_sets_features.jl` (803, 20), `test_13b_feature_routing.jl` (448, 1
  outer testset with 9 nested), `test_13c_feature_views.jl` (531, 9 nested),
  `test_13d_feature_collapse.jl` (571, 12 nested), `test_13e_feature_preselection.jl` (280, 10
  nested) and `test_53_panel_carrier_census.jl` (218, 5 nested). Their testset titles are a
  ready-made list of the behaviours the destination must show again.
- **The separation and decay families are tested inside `test_12d_phylogeny_features.jl`**:
  `HopCount`, `PathLength`, the quantile budgets and the four decays, from `:251` to `:718`.
  The roughly 170 `@test` lines that name no removed symbol keep their assertions and need a new home
  or a new fixture, because the file's fixture is a `PhylogenyFeatures` producer.
- **`test_06_preprocessing.jl` is the largest file outside the ground truth**, with 39 carrier
  assertions on `rd.Z`, `rd.nz`, `pr.Z` and `pr.nz` across the `ReturnsResult`, `PricesResult`,
  `prices_to_returns`, `MissingDataFilter` and `Imputer` paths. The carrier ticket owns it. The
  map's ground truth 11 does not name it.

### Lines naming a removed name, any kind of line, per ground-truth-11 file

| Test file | lines naming a removed symbol | lines naming a carrier field only | lines naming a reshaped name |
| --- | ---: | ---: | ---: |
| `test/test_12b_prior_core.jl` | 2 | 1 | 0 |
| `test/test_12c_feature_prior.jl` | 37 | 45 | 1 |
| `test/test_12d_phylogeny_features.jl` | 78 | 14 | 11 |
| `test/test_12e_asset_sets_features.jl` | 135 | 11 | 13 |
| `test/test_12g_forwarding_rule.jl` | 5 | 8 | 0 |
| `test/test_12i_cross_sectional_factor_carrier.jl` | 5 | 0 | 0 |
| `test/test_13_phylogeny.jl` | 7 | 3 | 37 |
| `test/test_13b_feature_routing.jl` | 45 | 24 | 40 |
| `test/test_13c_feature_views.jl` | 17 | 27 | 31 |
| `test/test_13d_feature_collapse.jl` | 9 | 21 | 29 |
| `test/test_13e_feature_preselection.jl` | 29 | 13 | 11 |
| `test/test_37_time_dependent_constraints.jl` | 2 | 0 | 0 |
| `test/test_53_panel_carrier_census.jl` | 16 | 9 | 14 |

## 4. The examples, the user guide and the API pages

**Examples.** `examples/3_optimisers/16_Feature_Distance_Clustering.jl` (968 lines) names a removed
symbol on 69 lines and a carrier field on 16 more. It builds a `ReturnsResult` with `nz` and `Z`
at `:184`, `:636`, `:720` and `:867`, constructs a `FeatureDistance` 18 times, passes `z_src`
18 times, and uses every producer: `FeaturePrior` at 11 lines, `AssetSetsFeatures` at 4,
`RegressionFeatures` at 3, `PhylogenyFeatures` at 7 and `Proximity` at 8, `phylogeny_features`
at 5, `asset_sets_features` at 4, `zkey` at `:244` and `:249`, and `features_are_assets` at
`:628`. Its closing prose at `:947`–`:968` names `features_are_assets`, `check_feature_matrix`,
`z_src`, `feature_matrix` and `feature_row_indices`, and `:966` names `15_FeaturePrior.jl` by
path. `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` (567 lines) names
`PhylogenyFeatures` and `Proximity` on four lines: the code at `:163` and `:165`, and the prose
at `:557`–`:558` that contrasts the constraint's `sep` with the producer's `decay`.
`examples/2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.jl:230` passes an `nz =` and
is the only other example line the carrier fields reach.

**User guide.** `user_guide/02_Optimisers.jl` (248 lines) builds the feature carrier at
`:170`–`:172` with `asset_sets_feature_names` and `asset_sets_features`, passes `z_src = :data`
at `:178`, and explains `z_src` and `FeaturePrior` at `:182`–`:196`. `user_guide/04_Constraints_and_Costs.jl`
names none of the removed symbols.

**API pages.** Nine hand-written pages under `docs/src/api/` name a removed symbol on 43 lines,
and the generated `26_TypeHierarchy.md` on nine more:

| Page | Entries that go | Entries that change |
| --- | --- | --- |
| `13_Prior/15_FeaturePrior.md` | The whole page: 14 `@docs` entries (`:6`–`:18`) and the intro at `:3`. | — |
| `09_Distance/05_FeatureDistance.md` | `assert_feature_selector` `:14`, `feature_selector_msg` `:15`, `feature_selection_push!` `:16`, `feature_selection_indices` `:17`, `select_features` `:18`, `assert_feature_matrix_supplied` `:27`. | `FeatureDistance` `:13`. |
| `13_Prior/01_Base_Prior.md` | `feature_matrix_picker` `:33`, `carrier_feature_names` `:34`. | `clusterise(cle, pr::Pr_RR; …)` `:16`, `Pr_RR` `:27`, `LowOrderPrior` `:4`. |
| `12_ConstraintGeneration/06_AssetSetsMatrix.md` | `AbstractFeatureValue` `:12`, `Scale` `:13`, `resolve_feature_value` `:14`, `Num_AFeatVal` `:15`, `asset_sets_features` `:16`, `asset_sets_feature_names` `:17`, `taxonomy_feature_names` `:18`, `feature_program_candidates` `:19`, `feature_unknown_name_msg` `:25`, and the grammar and writer entries at `:20`–`:24` and `:26`–`:31` if the graded program goes with them. | `UniverseSets` `:4`. |
| `12_ConstraintGeneration/02_LinearConstraintGeneration.md` | `feature_universe` `:48`. | — |
| `03_Preprocessing.md` | `check_feature_names` `:11`, `check_feature_matrix` `:12`, `check_names_and_feature_matrix` `:13`, `features_are_assets` `:14`, `feature_matrix_view` `:15`, `feature_row_indices` `:16`. | `ReturnsResult` `:9`, `PricesResult` `:143`, `check_asset_panel` `:79`, and the panel prose at `:35`–`:42` that says `nz` and `Z` hold the panel's numbers. |
| `20_Optimisation/01_Base_Optimisation.md` | `collapse_feature_matrix` `:115`. | — |
| `20_Optimisation/16_Base_MetaOptimisation.md` | `fold_feature_matrix` `:17`, `fold_feature_anchors` `:18`, `rebuild_feature_matrix` `:19`. | — |
| `22_AssetSelection.md` | — | The `ClusterGroups` prose at `:41`, which names `z_src` and `asset_sets_features`. |
| `08_Moments/42_FactorExposures/01_Base_Descriptor.md` | — | `panel_field_values` `:13` keeps its entry and changes its contract. |

`docs/make.jl` builds the page list with `walkdir` and indexes the top-level API folders
positionally (`:184`, `:232`–`:250`), so deleting `15_FeaturePrior.md` inside `13_Prior/` shifts
nothing.

## 5. The baseline rows a deleted source file leaves

A file under `src/` has seven rows in five files: one in `sweep/manifest.toml`, one in
`code_health/coverage_baseline.toml`, three in `code_health/jet_baseline.toml` (one per run:
`[run.main]` at `:13`, `[run.plots_ext]` at `:266`, `[run.impute_ext]` at `:519`), one in
`code_health/complexity_baseline.toml` and one in `code_health/size_baseline.toml`. Every one
of them is generated and says "do not edit by hand", and every one of them goes stale when the
file goes, because the generators add rows and never remove one. Each stale row is deleted by
hand in the commit that deletes the file. The table lists the rows of the ten files the mechanisms live in. Only
`15_FeaturePrior.jl` is a deletion candidate; the other nine are rewritten in place and their
rows are re-measured, not deleted.

| File | Baseline | Line | Row |
| --- | --- | ---: | --- |
| `src/13_Prior/15_FeaturePrior.jl` | `sweep/manifest.toml` | 174 | `"src/13_Prior/15_FeaturePrior.jl" = { map = 6, units = 13, algorithm = 6, swept = true }` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/coverage_baseline.toml` | 139 | `"src/13_Prior/15_FeaturePrior.jl" = { lines = 62, misses = 0 }` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/jet_baseline.toml` | 146 | `"src/13_Prior/15_FeaturePrior.jl" = { raw = 0, reviewed = 0 }` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/jet_baseline.toml` | 399 | `"src/13_Prior/15_FeaturePrior.jl" = { raw = 0, reviewed = 0 }` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/jet_baseline.toml` | 652 | `"src/13_Prior/15_FeaturePrior.jl" = { raw = 0, reviewed = 0 }` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/complexity_baseline.toml` | 140 | `"src/13_Prior/15_FeaturePrior.jl" = { cyc = 3, cog = 4, arg = 6, cyc_sum = 28, cog_sum = 13, arg_sum = 42, macros = ["@f` |
| `src/13_Prior/15_FeaturePrior.jl` | `code_health/size_baseline.toml` | 141 | `"src/13_Prior/15_FeaturePrior.jl" = { code = 140, doc = 476, comment = 27, blank = 211, total = 854 }` |
| `src/09_Distance/05_FeatureDistance.jl` | `sweep/manifest.toml` | 126 | `"src/09_Distance/05_FeatureDistance.jl" = { map = 2, units = 28, algorithm = 20, swept = true }` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/coverage_baseline.toml` | 91 | `"src/09_Distance/05_FeatureDistance.jl" = { lines = 138, misses = 0 }` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/jet_baseline.toml` | 98 | `"src/09_Distance/05_FeatureDistance.jl" = { raw = 2, reviewed = 2 }` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/jet_baseline.toml` | 351 | `"src/09_Distance/05_FeatureDistance.jl" = { raw = 0, reviewed = 0 }` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/jet_baseline.toml` | 604 | `"src/09_Distance/05_FeatureDistance.jl" = { raw = 0, reviewed = 0 }` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/complexity_baseline.toml` | 92 | `"src/09_Distance/05_FeatureDistance.jl" = { cyc = 9, cog = 11, arg = 7, cyc_sum = 80, cog_sum = 67, arg_sum = 100, macro` |
| `src/09_Distance/05_FeatureDistance.jl` | `code_health/size_baseline.toml` | 93 | `"src/09_Distance/05_FeatureDistance.jl" = { code = 309, doc = 782, comment = 2, blank = 369, total = 1462 }` |
| `src/03_InputData/03_Preprocessing.jl` | `sweep/manifest.toml` | 64 | `"src/03_InputData/03_Preprocessing.jl" = { map = 2, units = 44, algorithm = 22, swept = true }` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/coverage_baseline.toml` | 29 | `"src/03_InputData/03_Preprocessing.jl" = { lines = 403, misses = 0 }` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/jet_baseline.toml` | 36 | `"src/03_InputData/03_Preprocessing.jl" = { raw = 19, reviewed = 19 }` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/jet_baseline.toml` | 289 | `"src/03_InputData/03_Preprocessing.jl" = { raw = 0, reviewed = 0 }` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/jet_baseline.toml` | 542 | `"src/03_InputData/03_Preprocessing.jl" = { raw = 0, reviewed = 0 }` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/complexity_baseline.toml` | 30 | `"src/03_InputData/03_Preprocessing.jl" = { cyc = 20, cog = 31, arg = 16, cyc_sum = 161, cog_sum = 164, arg_sum = 139, ma` |
| `src/03_InputData/03_Preprocessing.jl` | `code_health/size_baseline.toml` | 31 | `"src/03_InputData/03_Preprocessing.jl" = { code = 726, doc = 1331, comment = 11, blank = 536, total = 2604 }` |
| `src/03_InputData/01_AssetPanel.jl` | `sweep/manifest.toml` | 62 | `"src/03_InputData/01_AssetPanel.jl" = { map = 2, units = 14, swept = false }` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/coverage_baseline.toml` | 27 | `"src/03_InputData/01_AssetPanel.jl" = { lines = 93, misses = 0 }` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/jet_baseline.toml` | 34 | `"src/03_InputData/01_AssetPanel.jl" = { raw = 0, reviewed = 0 }` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/jet_baseline.toml` | 287 | `"src/03_InputData/01_AssetPanel.jl" = { raw = 0, reviewed = 0 }` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/jet_baseline.toml` | 540 | `"src/03_InputData/01_AssetPanel.jl" = { raw = 0, reviewed = 0 }` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/complexity_baseline.toml` | 28 | `"src/03_InputData/01_AssetPanel.jl" = { cyc = 7, cog = 12, arg = 4, cyc_sum = 35, cog_sum = 28, arg_sum = 17, macros = [` |
| `src/03_InputData/01_AssetPanel.jl` | `code_health/size_baseline.toml` | 29 | `"src/03_InputData/01_AssetPanel.jl" = { code = 175, doc = 411, comment = 0, blank = 190, total = 776 }` |
| `src/13_Prior/01_Base_Prior.jl` | `sweep/manifest.toml` | 160 | `"src/13_Prior/01_Base_Prior.jl" = { map = 6, units = 41, algorithm = 19, swept = true }` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/coverage_baseline.toml` | 125 | `"src/13_Prior/01_Base_Prior.jl" = { lines = 180, misses = 0 }` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/jet_baseline.toml` | 132 | `"src/13_Prior/01_Base_Prior.jl" = { raw = 1, reviewed = 1 }` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/jet_baseline.toml` | 385 | `"src/13_Prior/01_Base_Prior.jl" = { raw = 0, reviewed = 0 }` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/jet_baseline.toml` | 638 | `"src/13_Prior/01_Base_Prior.jl" = { raw = 0, reviewed = 0 }` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/complexity_baseline.toml` | 126 | `"src/13_Prior/01_Base_Prior.jl" = { cyc = 10, cog = 8, arg = 12, cyc_sum = 96, cog_sum = 85, arg_sum = 113, macros = ["@` |
| `src/13_Prior/01_Base_Prior.jl` | `code_health/size_baseline.toml` | 127 | `"src/13_Prior/01_Base_Prior.jl" = { code = 377, doc = 1101, comment = 48, blank = 445, total = 1971 }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `sweep/manifest.toml` | 157 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { map = 5, units = 33, algorithm = 12, swept = true }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/coverage_baseline.toml` | 122 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { lines = 174, misses = 1 }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/jet_baseline.toml` | 129 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { raw = 0, reviewed = 0 }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/jet_baseline.toml` | 382 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { raw = 0, reviewed = 0 }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/jet_baseline.toml` | 635 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { raw = 0, reviewed = 0 }` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/complexity_baseline.toml` | 123 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { cyc = 9, cog = 16, arg = 5, cyc_sum = 75, cog_sum = 61, arg_sum ` |
| `src/12_ConstraintGeneration/06_AssetSetsMatrix.jl` | `code_health/size_baseline.toml` | 124 | `"src/12_ConstraintGeneration/06_AssetSetsMatrix.jl" = { code = 297, doc = 767, comment = 10, blank = 365, total = 1439 }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `sweep/manifest.toml` | 153 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { map = 5, units = 51, algorithm = 29, swept = true }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/coverage_baseline.toml` | 118 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { lines = 404, misses = 1 }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/jet_baseline.toml` | 125 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { raw = 1, reviewed = 1 }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/jet_baseline.toml` | 378 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { raw = 0, reviewed = 0 }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/jet_baseline.toml` | 631 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { raw = 0, reviewed = 0 }` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/complexity_baseline.toml` | 119 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { cyc = 20, cog = 47, arg = 7, cyc_sum = 149, cog_sum =` |
| `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl` | `code_health/size_baseline.toml` | 120 | `"src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl" = { code = 697, doc = 1291, comment = 47, blank = 565, to` |
| `src/22_Preselection.jl` | `sweep/manifest.toml` | 284 | `"src/22_Preselection.jl" = { map = 8, units = 30, algorithm = 11, swept = true }` |
| `src/22_Preselection.jl` | `code_health/coverage_baseline.toml` | 249 | `"src/22_Preselection.jl" = { lines = 168, misses = 0 }` |
| `src/22_Preselection.jl` | `code_health/jet_baseline.toml` | 256 | `"src/22_Preselection.jl" = { raw = 1, reviewed = 1 }` |
| `src/22_Preselection.jl` | `code_health/jet_baseline.toml` | 509 | `"src/22_Preselection.jl" = { raw = 0, reviewed = 0 }` |
| `src/22_Preselection.jl` | `code_health/jet_baseline.toml` | 762 | `"src/22_Preselection.jl" = { raw = 0, reviewed = 0 }` |
| `src/22_Preselection.jl` | `code_health/complexity_baseline.toml` | 250 | `"src/22_Preselection.jl" = { cyc = 7, cog = 12, arg = 4, cyc_sum = 78, cog_sum = 77, arg_sum = 31, macros = [] }` |
| `src/22_Preselection.jl` | `code_health/size_baseline.toml` | 251 | `"src/22_Preselection.jl" = { code = 313, doc = 762, comment = 0, blank = 365, total = 1440 }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `sweep/manifest.toml` | 216 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { map = 10, units = 113, swept = false }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/coverage_baseline.toml` | 181 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { lines = 475, misses = 24 }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/jet_baseline.toml` | 188 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { raw = 1, reviewed = 1 }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/jet_baseline.toml` | 441 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { raw = 0, reviewed = 0 }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/jet_baseline.toml` | 694 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { raw = 0, reviewed = 0 }` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/complexity_baseline.toml` | 182 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { cyc = 9, cog = 10, arg = 7, cyc_sum = 198, cog_sum = 144, arg_sum = 17` |
| `src/20_Optimisation/01_Base_Optimisation.jl` | `code_health/size_baseline.toml` | 183 | `"src/20_Optimisation/01_Base_Optimisation.jl" = { code = 812, doc = 1501, comment = 13, blank = 733, total = 3059 }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `sweep/manifest.toml` | 253 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { map = 10, units = 19, swept = false }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/coverage_baseline.toml` | 218 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { lines = 167, misses = 17 }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/jet_baseline.toml` | 225 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { raw = 7, reviewed = 7 }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/jet_baseline.toml` | 478 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { raw = 0, reviewed = 0 }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/jet_baseline.toml` | 731 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { raw = 0, reviewed = 0 }` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/complexity_baseline.toml` | 219 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { cyc = 10, cog = 16, arg = 8, cyc_sum = 64, cog_sum = 61, arg_sum =` |
| `src/20_Optimisation/16_Base_MetaOptimisation.jl` | `code_health/size_baseline.toml` | 220 | `"src/20_Optimisation/16_Base_MetaOptimisation.jl" = { code = 252, doc = 349, comment = 17, blank = 167, total = 785 }` |

Beyond the five baselines, a deleted or rewritten file is named at:

- `src/PortfolioOptimisers.jl:193` includes `15_FeaturePrior.jl`; `:145` includes
  `05_FeatureDistance.jl`, `:176` `06_AssetSetsMatrix.jl`, `:172` `02_LinearConstraintGeneration.jl`,
  `:179` `01_Base_Prior.jl`, `:83` `03_Preprocessing.jl`, `:303` `22_Preselection.jl`.
- `code_health/rulings.toml` carries an `arg` exemption for `LowOrderPrior` in `01_Base_Prior.jl`
  (`:219`, rationale `keyword-constructor-arity`), a coverage exemption for `feature_target!` in
  `06_AssetSetsMatrix.jl` (`:385`) and one for `LinearConstraint` in `02_LinearConstraintGeneration.jl`
  (`:375`). The `LowOrderPrior` exemption is keyed on a constructor whose arity falls by one
  when `Z` goes, so it is re-measured. No ruling names `15_FeaturePrior.jl` or
  `05_FeatureDistance.jl`.
- `sweep/manifest.toml` rows carry an `algorithm` floor for the swept files:
  `15_FeaturePrior.jl` 6, `05_FeatureDistance.jl` 20, `01_Base_Prior.jl` 19,
  `03_Preprocessing.jl` 22, `06_AssetSetsMatrix.jl` 12, `02_LinearConstraintGeneration.jl` 29,
  `22_Preselection.jl` 11. A deleted docstring with a `# Algorithm` section lowers the file's
  count below its floor, so the row is edited in the same commit. `test_45_sweep_census.jl` also
  reds when a file's `units` count leaves its row.
- `test/test_26_docs.jl` names `03_Preprocessing.jl` at `:488`, `:739` and `:930`, in the
  explanation of how `port_opt_view`'s multi-method docstring is counted; the ratchets there are
  `DETAILS_TOTAL = 57` (`:941`), `NO_RELATED_TOTAL = 20` (`:1029`) and `MATH_COPY_TOTAL = 7`
  (`:1227`), and each falls when a docstring that contributes to it goes.
- `src/09_Distance/04_Similarity.jl:538` and `test/test_13_phylogeny.jl:56`, `:115` say that
  `default_similarity(::AngularDist)` is declared in `05_FeatureDistance.jl`;
  `test/test_12i_cross_sectional_factor_carrier.jl:172` and `examples/3_optimisers/16_Feature_Distance_Clustering.jl:966`
  name `15_FeaturePrior.jl` by path; `test/test_49_coverage_attribution_census.jl:32` names
  `06_AssetSetsMatrix.jl`.
- `test/test_43_exported_abstract_type_census.jl:81`–`82` asserts that
  `AbstractFeatureMatrixEstimator`, `AbstractPhylogenyFeatureAlgorithm` and `AbstractFeatureValue`
  exist, are abstract and are not exported. Deleting any of them reds that loop unless its name
  leaves the tuple.

## 6. The doctests that print a carrier

Each printed `ReturnsResult` shows an `nz`, a `Z` and a `pnl` row; each printed `LowOrderPrior`
shows a `Z` row; the `HighOrderPrior` print at `01_Base_Prior.jl:1662` nests a `LowOrderPrior`
and shows its `Z` row at `:1675`. No doctest prints a `PricesResult`; the three `docs/src/api`
rows are `@docs` entries, not output. Every one of these blocks is regenerated when a field
goes or comes, under the `run-doctests` skill's fresh-process rule.

| File | Line | Printed head | `nz`/`Z`/`pnl` rows at |
| --- | ---: | --- | --- |
| `docs/src/api/03_Preprocessing.md` | 9 | `ReturnsResult` | (none printed) |
| `docs/src/api/03_Preprocessing.md` | 143 | `PricesResult` | (none printed) |
| `docs/src/api/13_Prior/01_Base_Prior.md` | 4 | `LowOrderPrior` | (none printed) |
| `src/03_InputData/03_Preprocessing.jl` | 692 | `ReturnsResult` | 702:nz, 703:Z, 704:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 883 | `ReturnsResult` | 893:nz, 894:Z, 895:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 898 | `ReturnsResult` | 908:nz, 909:Z, 910:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 976 | `ReturnsResult` | 986:nz, 987:Z, 988:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 991 | `ReturnsResult` | 1001:nz, 1002:Z, 1003:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 1300 | `ReturnsResult` | 1310:nz, 1311:Z, 1312:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 1315 | `ReturnsResult` | 1325:nz, 1326:Z, 1327:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 1333 | `ReturnsResult` | 1343:nz, 1344:Z, 1345:pnl |
| `src/03_InputData/03_Preprocessing.jl` | 1534 | `ReturnsResult` | 1544:nz, 1545:Z, 1546:pnl |
| `src/13_Prior/01_Base_Prior.jl` | 47 | `LowOrderPrior` | 59:Z |
| `src/13_Prior/01_Base_Prior.jl` | 1322 | `LowOrderPrior` | 1334:Z |
| `src/13_Prior/01_Base_Prior.jl` | 1663 | `pr ┼ LowOrderPrior` | 1675:Z |

Printed carriers: 15.

### Every printed `nz`, `Z` or `pnl` row (any carrier)

| File | Row | Count | Lines |
| --- | --- | ---: | --- |
| `src/03_InputData/03_Preprocessing.jl` | `Z` | 9 | 703, 894, 909, 987, 1002, 1311, 1326, 1344, 1545 |
| `src/03_InputData/03_Preprocessing.jl` | `nz` | 9 | 702, 893, 908, 986, 1001, 1310, 1325, 1343, 1544 |
| `src/03_InputData/03_Preprocessing.jl` | `pnl` | 9 | 704, 895, 910, 988, 1003, 1312, 1327, 1345, 1546 |
| `src/13_Prior/01_Base_Prior.jl` | `Z` | 3 | 59, 1334, 1675 |

## 7. Outside the census scope

These files are not in the ticket's list and are recorded here so the build tickets do not find
them by surprise.

- **`CONTEXT.md`** carries *Feature Matrix* (`:100`), *Feature Matrix Estimator* (`:104`–`:105`,
  naming `RegressionFeatures`, `AssetSetsFeatures` and `PhylogenyFeatures`), *Feature Program*
  (`:107`–`:108`), *Asset Panel* (`:111`–`:112`, which says `rd.nz` and `rd.Z` hold the values)
  and *Panel Field* (`:115`); *Feature Distance* at `:294`; *Phylogeny Features* at `:314`; the
  Universe Sets key list with `zkey` at `:456`; and a cross-reference at `:209`. The map's first
  decision ticket re-cuts §2.
- **ADRs.** Twelve files under `docs/adr/` name the stack: `0030`, `0044`, `0045` (the owner, with
  six amendments), `0046`, `0047`, `0048`, `0049`, `0064`, `0066`, `0068`, `0102` and
  `examples-coverage.md`. `FeaturePrior` is on 12 lines in 4 of them, `z_src` on 22 lines in 4,
  `Proximity` on 12 in 3, `rd.Z` on 9 in 3. ADR 0102 is a draft on `dev` and is rewritten; the
  rest are released and are amended.
- **`docs/capability_catalogue.jl`** carries the `FeaturePrior` group at `:600`–`:645`:
  `Cap(:FeaturePrior)`, `Cap(:feature_matrix, :AbstractFeatureMatrixEstimator)`,
  `Cap(:RegressionFeatures)`, `Group(Cap(:PhylogenyFeatures, :phylogeny_features))`,
  `Group(Cap(:Proximity))` — which is the *parent* of the separation and decay groups
  (`separation_matrix`, `HopCount`, `PathLength`, the quantile budgets, `separation_decay` and
  the four decays), so those caps need a new parent when `Proximity` goes —
  `Group(Cap(:AssetSetsFeatures, :asset_sets_features, :asset_sets_feature_names))`,
  `Group(Cap(:resolve_feature_value, :AbstractFeatureValue))` and `Cap(:Scale)`. The
  `FeatureDistance` group at `:427`–`:436` stays. `test_26_docs.jl`'s catalogue census requires
  every exported function to have an entry, so the removed exports leave the catalogue in the
  same commit.
- **Exports.** `15_FeaturePrior.jl:853` exports `RegressionFeatures`, `FeaturePrior`,
  `feature_matrix`, `Proximity`, `PhylogenyFeatures`, `phylogeny_features` and
  `AssetSetsFeatures`. `06_AssetSetsMatrix.jl:1438` exports `asset_sets_features`,
  `asset_sets_feature_names`, `Scale` and `resolve_feature_value`. `README.md` names none of
  them.
- **Dictionary keys that lose their last user** when their mechanism goes: `:Z_prior` (`:234`,
  1 user), `:us_zkey` (`:738`, 1), `:ze` (`:235`, 1), `:plfalg` (`:237`, 1), `:asets_vals`
  (`:751`, 1), `:asets_strict` (`:752`, 1), `:fdsets` (`:152`, 4, all in `05_FeatureDistance.jl`),
  `:z_src` (`:572`, 13, all forwarders and optimisers). `:fdsel` (3) and `:fdstrict` (3) keep
  users if the selector keeps its names. `:pr_rr` (`:158`, 11 users) describes the bridge and
  says it carries `Z`.

---

## Appendix: the patterns and the scripts

The census is two Python scripts that hold the file loader, the docstring tracker, the kind
classifier, the parenthesis balancer and the pattern lists. They are not committed. What
matters for reproduction is the pattern set:

- **Mechanism 1 names**: `FeaturePrior`, `feature_matrix`, `feature_estimator_view`,
  `AbstractFeatureMatrixEstimator`, `RegressionFeatures`, `PhylogenyFeatures`, `Proximity`,
  `AbstractPhylogenyFeatureAlgorithm`, `_proximity_features`, `phylogeny_features`,
  `AssetSetsFeatures`, `asset_sets_features`, `asset_sets_feature_names`, `Scale` (as
  `` `Scale` `` or `Scale` not followed by a lower-case word), `AbstractFeatureValue`,
  `taxonomy_feature_names`, `carrier_feature_names`, `feature_program_candidates`,
  `feature_unknown_name_msg`.
- **Mechanism 2**: `\bZ\b` inside `01_Base_Prior.jl`; `(pr|fpr|…)\.Z\b` elsewhere; the key
  `Z_prior`; `LowOrderPrior(` balanced and searched for `Z =`.
- **Mechanism 3**: `z_src`, `feature_matrix_picker`, `Pr_RR`, `assert_feature_matrix_supplied`,
  `assert_source_selector`, `returns_matrix_picker`, `:neither`, `:data_only`, and
  `z_src\s*(=|==|in|∈)`.
- **Mechanism 4**: the five helpers by name; `sel`, `sets`, `strict` inside
  `05_FeatureDistance.jl`; `FeatureDistance(` balanced and searched for each keyword;
  `(de|ce|d|fd|dist)\.(sel|sets|strict)` elsewhere.
- **Mechanism 5**: the thirteen verbs by name; `rd\.Z`, `rd\.nz`, `rd\.pnl`, `pr\.Z`,
  `pr\.nz`; `\.Z\b`, `\.nz\b`, `\.pnl\b`; `(nz|Z|pnl)\s*(=[^=]|::)` at a token boundary,
  with the five excluded files; `ReturnsResult(` and `PricesResult(` balanced; every
  `function port_opt_view(` body scanned for `Z`, `nz`, `pnl` and the three view verbs.
- **Mechanism 6**: `zkey`, `feature_universe`, `us_zkey`; `zkey [┼┴]` for the printed row;
  `UniverseSets(` balanced.
- **Mechanism 7**: `rd\.Z|rd\.nz|z_src|:data_only` inside `22_Preselection.jl`; `ClusterGroups`
  in the tests, examples and pages.
- **Mechanism 8**: `panel_field_values`, `panel_field`; `rd\.pnl`, `rd\.Z`, `rd\.nz` inside
  `42_FactorExposures/` and `04_CrossSectionalTransforms.jl`.
- **Removed set for the `@test` count**: every mechanism 1, 3, 4, 5, 6 symbol above except
  `Pr_RR`, `returns_matrix_picker`, `assert_source_selector`, `check_asset_panel`,
  `prepare_outer_rd`, `rebuild_returns_result`, `panel_field_values` and `panel_field`; plus
  `:neither`, `:data_only`, a `sets =` on a `FeatureDistance(` line, and the four carrier-field
  forms.
- **Reshaped set**: `.pnl`, `pnl =`, `panel_field_values`, `panel_field`, `check_asset_panel`,
  `prepare_outer_rd`, `rebuild_returns_result`, `sel =`, `.sel`, `.strict`, `FeatureDistance`,
  `Pr_RR`.
