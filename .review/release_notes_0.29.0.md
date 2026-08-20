# Release notes — v0.29.0

**Where these go.** The repository publishes release notes as the body of the GitHub release, and
has never carried a `CHANGELOG.md`. This file is the draft for the **v0.29.0 release body**. It
lives under `.review/`, which is deleted before the pull request closes, so the hand-off step is:
paste this document into the v0.29.0 release body, then delete `.review/`. Nothing else in the
repository carries these notes.

## The base of this diff is `v0.28.0`, not the branch point

The API ledger in `.review/api_ledger.txt` diffs the branch point `c975293745`. That is the right
base for a **review** of the pull request, and the wrong base for **release notes**: `c975293745`
is an ancestor of the `v0.28.0` tag, so part of that diff is already released and is not a 0.29.0
break. `x_src` is the clear case — `v0.28.0` already ships it and already has no `cle_pr`, and its
own release notes announce the change.

Everything below is measured `v0.28.0..HEAD`.

| Quantity | Against the branch point | **Against `v0.28.0`** |
| --- | --- | --- |
| Exports removed | 8 | **8** |
| Exports added | 63 | **62** |
| Exported types gone | 2 | **2** |
| Exported types whose fields change | 44 | **40** |
| …of those, types that **lose** a field | 12 | **9** |

The last two rows also correct a parser fault in the committed ledger. Its parser reads a field
declaration as a bare indented name and misses one carrying a propagation macro — `@vprop sets`,
`@fprop mu`, `@pprop`, `@wprop`, `@cprop` — so it reported four removals that never happened
(`FactorBlackLittermanPrior` `sets` and `w`, `FactorRiskBudgeting` `sets`,
`BayesianBlackLittermanPrior` `sets`) and missed that `AugmentedBlackLittermanPrior` collapses two
fields into one. Those four types are listed correctly below.

---

## Breaking changes

### 1. Six exported names are renamed. There is no shim

Consistent with the library's practice: `src/` has never carried an `@deprecate`.

| Old name | New name | Why |
| --- | --- | --- |
| `AssetSets` | `UniverseSets` | The type declares the axes a name can land on — assets, factors, features — so it is not about assets alone. **Its fields are renamed too**; see below. |
| `CVaREntropyPooling` | `ConditionalValueatRiskEntropyPooling` | Spelled out, matching every other measure name in the library. |
| `MaxSR` | `MaxRa` | The alias targets `MaximumRatio`, which is any ratio, not the Sharpe ratio alone. |
| `RVaR` | `RLVaR` | Disambiguates **rel**ativistic from **rel**ative. `RVaR` read as either. |
| `RVaR_RG` | `RLVaR_RG` | Same reason. |
| `R_RDaR` | `R_RLDaR` | Same reason. `R_` is the relative prefix and `RL` is relativistic, so the old spelling collided with itself. |

All six old names have zero occurrences in the tracked tree. For five of them, replacing the name is
the whole migration. `UniverseSets` is the exception.

#### `AssetSets` → `UniverseSets` renames its fields as well

| `AssetSets` (v0.28.0) | `UniverseSets` (0.29.0) |
| --- | --- |
| `key` | `xkey` — the asset axis |
| `ukey` | `uxkey` |
| — | `fkey`, `ufkey` — the factor axis |
| — | `zkey` — the feature axis |
| `dict` | `dict` |

```julia
# Before
AssetSets(; key = "nx", dict = Dict("nx" => tickers))

# After
UniverseSets(; xkey = "nx", dict = Dict("nx" => tickers))
```

The `x` prefix is what makes the axis explicit: a bare `key` could not say which of three axes it
named, which is why a Black-Litterman caller now states its **axis** rather than its key (ADR 0068).
The two factor keys and `zkey` are new; a caller that only ever used the asset axis renames one
field and is done.

### 2. Nine exported types lose a field

Reading a removed name is only broken where the table says so — several survive as virtual reads.
Constructing with the removed keyword breaks in every case.

| Type | Removed | Replacement |
| --- | --- | --- |
| `ArithmeticReturn` | `lb` | `settings = JuMPReturnsSettings(; lb = …)` |
| `LogarithmicReturn` | `lb` | `settings = JuMPReturnsSettings(; lb = …)` |
| `AugmentedBlackLittermanPrior` | `a_sets`, `f_sets` | one `sets`, a `UniverseSets` carrying both axes |
| `EntropyPoolingPrior` | `var_alpha`, `cvar_alpha`, `ds_opt`, `dm_opt` | the level and the formulation move onto the view |
| `HierarchicalResult` | `fb` | `fb` moves to the leaf result type |
| `HighOrderPrior` | `f_kt`, `f_sk`, `f_V` | one nested `fpr` |
| `LowOrderPrior` | `f_mu`, `f_sigma`, `f_w` | one nested `fpr` |
| `NetworkEstimator` | `n` | `sep`, and the count moves onto `HopCount` |
| `SubsetResampling` | `scale` | removed with no replacement |

Each is spelled out below.

#### `lb` moves into a settings bundle, and `settings` takes the first slot

Every return term now carries a `JuMPReturnsSettings` bundle — `scale`, `lb`, `rte`, `fee`, `mic` —
in a field named `settings`, placed **first**. That matches `Variance(settings, sigma, chol, rc,
alg)` on the risk side, where `settings` has always come first.

```julia
# Before
ArithmeticReturn(; lb = 0.02)

# After
ArithmeticReturn(; settings = JuMPReturnsSettings(; lb = 0.02))
```

Two things break and no more. `lb` moved out of the estimator into the bundle, and `settings` took
the first positional slot — which `@concrete`'s positional constructor makes a real break rather
than a cosmetic one. A keyword caller who never set `lb` needs no edit. See ADR 0052.

#### An augmented Black-Litterman prior carries one `sets`, not two

`a_sets` and `f_sets` collapse into a single `sets`. A `UniverseSets` declares both axes — `xkey`
for assets and `fkey` for factors — so two fields were two halves of one object. The caller states
which axis a view lands on rather than which key it reads. See ADR 0068.

```julia
# Before
AugmentedBlackLittermanPrior(; a_sets = asset_sets, f_sets = factor_sets, …)

# After
AugmentedBlackLittermanPrior(; sets = UniverseSets(; xkey = "assets", fkey = "factors",
                                                   dict = …), …)
```

`BayesianBlackLittermanPrior`, `FactorBlackLittermanPrior` and `FactorRiskBudgeting` **keep** their
`sets` field. The committed ledger says otherwise; it is wrong.

#### A tail view carries its own level and formulation

`EntropyPoolingPrior` loses the estimator-level `var_alpha`, `cvar_alpha`, `ds_opt` and `dm_opt`,
and gains `evar_views`. A tail view is now stated as a group: the view equations, the level they
are read under, and the formulation that writes them.

```julia
# Before
EntropyPoolingPrior(; cvar_views = views, cvar_alpha = 0.05, ds_opt = …, dm_opt = …)

# After
EntropyPoolingPrior(; cvar_views = ConditionalValueatRiskView(; views = views, alpha = 0.05))
```

`alg` defaults to `nothing` and resolves per view to the cheapest formulation that is exact. A
`prior(...)` reference inside a group resolves at that group's own level, which is what makes a
view stated against the prior move with the level it is read under. `ValueatRiskView` carries
`views` and `alpha` and no `alg`, because a VaR view is linear in the posterior probabilities.
See ADR 0069.

#### `HierarchicalResult` becomes a shared field core

`HierarchicalResult` is now the common core embedded as the first field, `hr`, of two new exported
leaf types:

- `HierarchicalRiskParityResult` — `hr`, `r`, `sca`, `fb`
- `HierarchicalEqualRiskContributionResult` — `hr`, `ri`, `ro`, `scai`, `scao`, `fb`

ADR 0011 fixes `fb` as the **last** field of each concrete result and keeps it out of the core, so
the one generic `factory(res, fb)` rebuilds both leaves by the same convention. A caller who reads
`res.fb` off an HRP or HERC result is unaffected — `res` is the leaf, and the leaf has `fb`. What
breaks is constructing a `HierarchicalResult` with `fb`, or dispatching on `HierarchicalResult`
where the concrete result is now a leaf.

#### The factor block of a prior carrier is nested

`LowOrderPrior` and `HighOrderPrior` no longer carry the factor moments as flat fields. One field,
`fpr`, carries a whole nested prior for the factor block, or `nothing` when there is none.

| Carrier | Fields removed | Fields added |
| --- | --- | --- |
| `LowOrderPrior` | `f_mu`, `f_sigma`, `f_w` | `o_X`, `fpr`, `Z` |
| `HighOrderPrior` | `f_kt`, `f_sk`, `f_V` | `fpr` |

**The break bites a caller who constructs a carrier with a flat factor keyword. It does not bite a
caller who reads one.** The removed names survive as *virtual reads* through `@forward_properties`
(`src/13_Prior/01_Base_Prior.jl:1089` and `:1379`), so `pr.f_mu` still answers, and answers
`nothing` where there is no factor block — exactly what the flat field did.

```julia
# Still works. No change needed.
pr.f_mu, pr.f_sigma, pr.f_w      # LowOrderPrior
hop.f_kt, hop.f_sk, hop.f_V      # HighOrderPrior

# Breaks: MethodError — got unsupported keyword argument "f_mu".
LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr, f_mu = f_mu, f_sigma = f_sigma)

# The replacement: build the factor block as its own prior and pass it as `fpr`.
LowOrderPrior(; X = X, mu = mu, sigma = sigma, rr = rr,
              fpr = LowOrderPrior(; X = F, mu = f_mu, sigma = f_sigma))
```

The togetherness invariant is unchanged, only respelled. The old constructor needed `rr`, `f_mu`
and `f_sigma` together. The new one needs `rr` and `fpr` together, and says so:
`ArgumentError: rr and fpr are the factor block and must be provided together or not at all`.

Anything that destructures or splats the field set is also affected, because the field set itself
changed: `fieldnames`, `getfield`, a positional constructor call, and `prior_field_values`.

**Reads that come free with the nesting.** Seven, none of which exists at `v0.28.0`. They need no
migration, and they answer on any carrier that has a factor block: `f_ens`, `f_kld`, `f_ow` on
`LowOrderPrior`, and `f_D2`, `f_L2`, `f_S2`, `f_skmp` on `HighOrderPrior`.

**Which read is idiomatic.** `pr.fpr.mu` is the public read. ADR 0046's 2026-08-03 amendment
freezes the flat `f_`-prefixed set at thirteen names: it stays readable and stays supported, and no
name is added to it. The two reads differ only where the block is absent — `pr.f_mu` answers
`nothing`, and `pr.fpr.mu` raises. Guard on `fpr`, then read through it.

Read ADR 0046 for the reason behind every dropped field.

#### A network relates by its separation

`NetworkEstimator.n` is removed in favour of `sep`. The count moved onto `HopCount`, because `sep`
answers which pairs are related and the count is a parameter of one way of answering it.

```julia
# Before
NetworkEstimator(; n = 2)

# After
NetworkEstimator(; sep = HopCount(; n = 2))
```

`sep` decides which pairs are related, which every consumer of the estimator needs; leaving it on
the producer would hide it from the constraint path. See ADR 0048.

#### `SubsetResampling` loses `scale`, with no replacement

A Combination Weight is the weight an element carries inside a combination, and it belongs where an
**outer optimiser** chooses the blend. `Stacking` owns one and keeps its `scale`. `SubsetResampling`
averages randomly drawn asset subsets and has no outer optimiser, so the field had no meaning
there. See ADR 0053, and `combination_weights` for the shape that survives.

### 3. A PMFG weight must be strictly positive

`PMFG_T2s` carries the graph's structure and its weights in one matrix, so an exactly zero weight is
an **absent edge** rather than a weak one. `assert_pmfg_weights` now counts the stored edges against
the `3N - 6` a maximal planar graph has, and raises a `DomainError` naming the shortfall. It runs at
`DBHTs`, `calc_weighted_adjacency_graph` and `calc_distance_weighted_graph`. `logo!` reads only the
cliques and is deliberately unguarded.

**What breaks.** A similarity that returns an exact zero on the PMFG path. Two shipped routes reach
one:

- `ExponentialSimilarity` or `GeneralExponentialSimilarity` under `LogDistance`, which maps an
  exactly zero correlation to `Inf`, and `exp(-Inf)` is `0` exactly.
- `ComplementSimilarity` or `MaximumDistanceSimilarity` at `D = 1`, which is a correlation of `-1`.

An exactly zero sample correlation is not an everyday event, but `Denoise()` makes one. The measured
case is `NetworkEstimator(; ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn =
Denoise())), de = Distance(; alg = LogDistance()), alg = ExponentialSimilarity())` over noise, which
denoises to the identity correlation. It used to return a network of `0` of its `54` edges. It now
raises.

```text
DomainError with count(!iszero, A) / 2 == 3 * size(A, 1) - 6 must hold. Got
edges => 0
3 * N - 6 => 54
```

The replacement is a similarity that stays strictly positive on the data at hand. Nothing that
worked and returned a full graph changes: the refusal is exactly the set of inputs whose graph was
already missing edges. Before this change the same inputs died later, in `turn_into_Hclust_merges`,
with a `BoundsError` about a matrix index. See ADR 0049, *Non-negative reaches the check, positive
reaches the graph*.

### 4. Two exported names lose their export. The names stay

`TimeDependentCallable` and `TimeDependentOptimiserCallable` are exported at `v0.28.0` and are not
exported at 0.29.0. This is a **withdrawal**, not a rename: both types still exist, keep their
names, keep their docstrings and keep their `docs/src/api/` and catalogue entries. Only the export
goes, so a caller that writes the bare name after `using PortfolioOptimisers` now gets an
`UndefVarError`.

| Old spelling | New spelling |
| --- | --- |
| `TimeDependentCallable` | `PortfolioOptimisers.TimeDependentCallable` |
| `TimeDependentOptimiserCallable` | `PortfolioOptimisers.TimeDependentOptimiserCallable` |

```julia
# Before
struct MySchedule <: TimeDependentCallable end

# After
struct MySchedule <: PortfolioOptimisers.TimeDependentOptimiserCallable end
```

**Why.** ADR 0030's amendment of 2026-08-19 reclassifies the family. The root moves from
`AbstractAlgorithm` to `AbstractEstimator`, and a third name, `TimeDependentConstraintCallable`,
splits the constraint case out of the root so that a bare root no longer means two things at once.
All three are abstract types, and `CLAUDE.md` does not export an abstract type unasked, so all
three are unexported and pinned by name in `test/test_43_exported_abstract_type_census.jl`. No
dispatch changes: every generic method that reached the root already named both supertypes in one
`Union`.

The module prefix is what the examples and `test/test_37_time_dependent_constraints.jl` already
used, so the migration is a prefix and nothing else.

### 5. Four exported types keep their arity and change their field order

A **keyword** call is unaffected by everything in this section. This is about a **positional**
call, which `@concrete` gives every type in the library.

`.review/api_ledger.txt` now reports the positional shape of every exported type present at both
refs, because a set-membership reading cannot see a reorder — a field inserted anywhere but the end
reads as a plain addition while every argument from that slot onwards rebinds. Of the 428 exported
types present at both `v0.28.0` and 0.29.0:

| Shape | Count | What a stale positional call does |
| --- | --- | --- |
| Identical | 388 | binds as before |
| Pure append | 15 | binds as before |
| Arity change | 20 | **`MethodError`** — the argument count no longer matches |
| Truncation | 1 | **`MethodError`** — same reason |
| **Same arity, reordered** | **4** | **builds a wrong object, silently** |

The last row is the only one that needs care, and it needs it badly.

| Type | Fields | First divergence | What a `v0.28.0` positional call produces |
| --- | --- | --- | --- |
| `ArithmeticReturn` | 3 | slot 1, `ucs` → `settings` | `settings` holds the old `ucs`, `ucs` holds the old `lb` |
| `LogarithmicReturn` | 2 | slot 1, `w` → `settings` | `settings` holds the old `w` |
| `LowOrderPrior` | 12 | slot 2, `mu` → `o_X` | `o_X` holds `mu`, `mu` holds `sigma`, `sigma` holds `nothing` |
| `NetworkEstimator` | 4 | slot 4, `n` → `sep` | `sep` holds an integer |

**It does not raise.** Each of these types declares a typed inner constructor, and each is
`@concrete`, and `ConcreteStructs` emits an *unconstrained* generic constructor beside it:

```julia
LowOrderPrior(X::__T_X, o_X::__T_o_X, mu::__T_mu, …) where {__T_X, __T_o_X, __T_mu, …}
```

Where the argument types do not match the typed constructor, that one does, and it validates
nothing. Verified by running each of the four, not by reading the signatures:

```julia
julia> v = ArithmeticReturn(BoxUncertaintySet(; lb = [0.1], ub = [0.2]), 0.05, nothing);

julia> typeof(v.settings), typeof(v.ucs)
(BoxUncertaintySet{…}, Float64)
```

**Migration.** Call these four by keyword. A keyword call names its slots, so it either binds
correctly or raises. This is the general advice for the library — the positional constructor is an
artefact of `@concrete`, not a supported interface — and here it is the difference between a
`MethodError` and a wrong number.

### Not a migration item

`LowOrderPrior.z_sq` is neither a field nor a forwarded property at `HEAD`. It was added and removed
inside this development cycle, so no release ever carried it and it needs no note.

---

## New in 0.29.0

62 new exported names.

### The feature matrix, the distance layer and phylogeny

A **Feature Matrix** is data, not estimator configuration: it is carried on the Result and routed to
the consumer, selected by `z_src`. ADRs 0044, 0045, 0048.

- Feature producers — `PhylogenyFeatures`, `AssetSetsFeatures`, `RegressionFeatures`,
  `AggregateFeatures`, `FeaturePrior`, `feature_matrix`, `phylogeny_features`,
  `asset_sets_features`, `asset_sets_feature_names`.
- Feature values and collapses — `Scale`, `Proximity`, `resolve_feature_value`, `MeanCollapse`,
  `MedianCollapse`, `LastObservation`, `StackObservations`.
- Distance and similarity — `FeatureDistance`, `AggregateDistances`, `AngularDist`,
  `AngularSimilarity`, `ComplementSimilarity`, `DistancePolarity`, `SimilarityPolarity`.
- Separation and decay — `HopCount`, `HopCountQuantile`, `PathLength`, `PathLengthQuantile`,
  `resolve_separation`, `separation_matrix`, `separation_budget`, `separation_decay`, `NoDecay`,
  `LinearDecay`, `ExponentialDecay`, `ReciprocalDecay`.
- Centrality — `centrality_polarity`, `TopologyOnly`. Five centrality types gain
  `ov::Option{TopologyOnly}`, which withdraws the type's own polarity declaration and reads the
  topology alone.

### Universe sets and constraint generation

`UniverseSets` replaces `AssetSets` and declares the axis a name lands on. `ExposureConstraintEstimator`
(alias `ECE`) and `FactorSpace` generate factor-exposure constraints; a constraint is re-based only
where the type admits it, which is a bound rather than a runtime check. ADR 0047.

### Entropy pooling views

A tail view is a group carrying its own level and formulation. ADRs 0064, 0069.

- Views — `ValueatRiskView`, `ConditionalValueatRiskView`, `EntropicValueatRiskView`.
- Formulations — `LinearConditionalValueatRiskView`, `IntegerConditionalValueatRiskView`,
  `ConicEntropicValueatRiskView`, `GridEntropicValueatRiskView`.
- Priors — `ConditionalValueatRiskEntropyPooling`, `MeucciEntropyPoolingPrior`.

### The return expression

A return expression is a weighted sum of terms, and `scale` is the combination weight. ADRs 0052,
0053, 0054.

- `JuMPReturnsSettings` — the per-term bundle (`scale`, `lb`, `rte`, `fee`, `mic`).
- `NoReturn` — a problem with no return term.
- `unit_scale_risk_measure` — drops the weight on the singular route, where one element is not a
  combination.

### Results and reporting

`HierarchicalRiskParityResult`, `HierarchicalEqualRiskContributionResult`,
`RelaxedRiskBudgetingResult`, `PerformanceSummaryResult`, `performance_summary`.

### Renamed aliases

`MaxRa`, `RLVaR`, `RLVaR_RG`, `R_RLDaR` — see the rename table above.

---

## Non-breaking field additions

31 exported types gain a field and lose none. No caller needs an edit.

| Field | Types | What it is |
| --- | --- | --- |
| `settings` | `ExpectedReturn`, `MeanReturn`, `ThirdCentralMoment`, `ExpectedReturnRiskRatio`, `MeanReturnRiskRatio`, `NonOptimisationRiskRatio` | The return-term bundle. The ratio types also gain `sca` (`sca1`/`sca2` on `NonOptimisationRiskRatio`). |
| `r` | `MeanRiskResult`, `NearOptimalCenteringResult`, `FactorRiskContributionResult`, `RiskBudgetingResult`, `SchurComplementHierarchicalRiskParityResult` | The risk measures the result was produced under. |
| `ov` | `BetweennessCentrality`, `ClosenessCentrality`, `EigenvectorCentrality`, `RadialityCentrality`, `StressCentrality` | `Option{TopologyOnly}`, defaulting to `nothing`. Withdraws the polarity declaration. |
| `val` | `BoxUncertaintySet`, `EllipsoidalUncertaintySet` | The quantity the set bounds. ADR 0050. |
| `mu` | `L1UncertaintySet`, `SignedL1UncertaintySet` | The quantity the set bounds. ADR 0050. |
| `pe` | `Kurtosis`, `Skewness`, `VarianceSkewKurtosis`, `DistributionValueatRisk` | A prior estimator; a slot may hold the method instead of the value. ADR 0051. |
| `z_src` | `JuMPOptimiser`, `HierarchicalOptimiser`, `NestedClustered` | Selects the feature-matrix source, `:data` or `:prior`. |
| `nz`, `Z` | `PricesResult`, `ReturnsResult` | The feature axis names and the feature matrix. |
| `f_mp` | `BayesianBlackLittermanPrior` | Factor-side matrix post-processing. |
| `sca` | `ProcessedJuMPOptimiserAttributes` | The scalariser the assembly resolved. |
