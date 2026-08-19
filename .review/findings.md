# PR #261 review — findings

## Block 1 — feature matrix and the distance layer  (`c975293745..b7944d6fe5`)

### Boxes — Block 1

| Box | Verdict |
| --- | --- |
| `33de810633` formatter pass changes no behaviour | **FAILS — see B1-1** |
| Similarity family moved to the distance layer | PASS |
| ADR 0044 flag-to-name conversion is total | PASS |
| ADR 0043 `nothing` means unweighted | PASS |
| ADR 0042 Impute is a weak dependency | PASS |
| 14 new exports in the capability catalogue | PASS, but the gate has a hole — see B1-2 |
| `test_08c` / `test_12c` test the decision | PASS |

Evidence for the passes:

- `cle_pr` survives nowhere in `src/`. `assert_source_selector` is called from all three inner
  constructors (`04_Base_ClusteringOptimisation.jl:420`, `10_JuMPOptimiser.jl:633`,
  `17_NestedClustered.jl:457`) and from both prior-layer pickers (`13_Prior/01_Base_Prior.jl:497,532`).
- ADR 0043's resolver split is present as three methods at `src/01_Base.jl:2980,2983,2993`.
  No caller-side `isnothing(w) && !isnothing(...)` guard exists anywhere in `src/`, so corollary 1
  holds. `cov(::GeneralCovariance, ...)` resolves before it dispatches.
- `Impute` is in `[weakdeps]`. `src/` calls `Impute` in no code path; the only method that calls it
  lives in `ext/PortfolioOptimisersImputeExt.jl`. The seam has an identity method on `Nothing` and
  a throwing fallback on `Any`.
- The similarity types are declared only in `src/09_Distance/04_Similarity.jl`. The phylogeny layer
  reads them and includes after the distance layer.
- Both new test files name decisions, not functions: "Resolver contract", "FeaturePrior is a pure
  addition", "Nesting order does not matter".

### B1-1 — the formatter pass reintroduced the escaped-quote corruption `CLAUDE.md` forbids

`CLAUDE.md` says: *"Do not run JuliaFormatter directly over `src/` — it has corrupted escaped
quotes inside jldoctest blocks."* Commit `33de810633` "Formatter pass" did exactly that.

Counted lines in `src/` holding a `\"`:

| Ref | Lines |
| --- | --- |
| `c975293745` (before) | 4 |
| `33de810633` (the formatter pass) | 125 |
| `HEAD` | 177, over 33 files |

The 4 at the base are genuine escapes inside ordinary `"..."` strings. The other 173 are inside
`"""..."""` docstrings, where `\"` renders as `"`. **The change is therefore behaviour-neutral**:
the doctests still read `Dict("A" => 0.1)` after rendering, which is why CI stayed green. One case
is worse to read than the rest — `src/08_Moments/01_Base_Moments.jl:1368` now spells a triple-quote
opener as `\"""`.

The defect is in the source, not in the output. Two costs follow:

1. Every affected docstring is harder to read and to edit.
2. The commit is 31 files of noise that a reviewer must diff with `-w` to see through.

There is no double escaping yet (`\\"` count is 0), so the damage has not compounded.

Recommendation: revert the escaping with a targeted pass over the 33 files, and keep the rest of
the formatter commit. Do not tick this box as "no behaviour change, skip it" — the box asks the
wrong question, and the answer to the right one is that the commit violates a stated project rule.

### B1-2 — no test gates the "never export an abstract type" rule, and six slipped through

This is a **PR-wide** finding. It surfaces in B1 and is not confined to it.

`CLAUDE.md`: *"Never export an abstract type unless explicitly told to."* Exported abstract types,
counted by parsing every `abstract type` declaration and every `export` list in `src/`:

| Ref | Abstract types | Exported |
| --- | --- | --- |
| `c975293745` (base) | 215 | **7** |
| `HEAD` | 240 | **13** |

Per block, the running set changes as follows.

| Block | Added | Withdrawn |
| --- | --- | --- |
| B1 | `AbstractFeatureMatrixEstimator` | — |
| B2 | `AbstractPhylogenyFeatureAlgorithm` | — |
| B3 | `AbstractConstraintSpace` | — |
| B4 | `AbstractCentralityPolarity`, `AbstractFeatureValue`, `AbstractSeparationAlgorithm`, `AbstractSeparationDecayAlgorithm` | — |
| B5 | `AbstractNonNegativeSimilarityMatrixAlgorithm`, `AbstractSimilarityMatrixAlgorithm` | `AbstractCentralityPolarity`, `AbstractSeparationAlgorithm`, `AbstractSeparationDecayAlgorithm` |
| B6-B11 | — | — |

Net: **six new exported abstract types**, in five different blocks. The exported abstract surface
almost doubles.

Two corrections to the review plan follow:

1. The plan's table of names "added and then withdrawn" lists two abstract exports for B4 and B5.
   There are **three** — it misses `AbstractCentralityPolarity`.
2. The plan's global box "No abstract type gains an export" is stated once and checked nowhere.
   Six names fail it.

**The root cause is that the catalogue gate cannot see an abstract type.**
`test/test_26_docs.jl` gates two populations:

- `leaf_types(...)` collects estimators and algorithms, and it skips abstract types by
  construction: `if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers`.
- "every exported function is accounted for" keeps only `getfield(PO, n) isa Function`.

An exported abstract type is in neither population, so it passes both gates silently. That is
precisely how six of them landed without CI noticing.

Recommendation: add a census to `test_26_docs.jl` in the shape of ADR 0060's — the exported
abstract types must equal a written allow-list, so adding one is a deliberate edit to that list and
never an accident. Then decide the six on their merits: each is an open extension point, which is
an argument for documenting the abstract type, not necessarily for exporting it.

---

## Block 2 — a feature matrix is data; the prior rework  (`b7944d6fe5..a0eda3861c`)

### Boxes — Block 2

| Box | Verdict |
| --- | --- |
| ADR 0046 documents every dropped field | PASS |
| The four flat factor fields collapse into one `fpr` | PASS |
| `z_sq` arrives in B1 and leaves here; nothing outside the carrier read it | PASS |
| ADR 0045: no estimator gained a feature-matrix field | PASS |
| "An estimator never holds a Result" is enforced by a type bound | PASS |
| The eight new test files test the decisions | PASS |

Evidence:

- Six of the seven dropped fields are named in ADR 0046. The seventh, `z_sq`, is documented in
  **ADR 0045's amendment of 2026-07-29**, which is the amend-never-rewrite pattern `CLAUDE.md`
  asks for. All seven are gone from `src/` as fields.
- `33eb953184` enforces the rule by narrowing a union alias, not by a runtime check:
  `NwE_PlM_ClE_Cl = Union{AbstractNetworkEstimator, PhylogenyResult{<:AbstractMatrix}, ClE_Cl}`
  becomes `NwE_ClE = Union{AbstractNetworkEstimator, AbstractClustersEstimator}`. A precomputed
  `PhylogenyResult` in a `pl` slot is now a `MethodError` at construction. `port_opt_view` on a
  `PhylogenyResult` is deleted with it, which is right — the commit message shows the slice was
  measured to be wrong, not merely unneeded (an 8-asset MST cut to 3 assets keeps 2 of 14 edge
  entries and still re-validates).
- Parsing every struct in `src/`, only three types carry `Z`: `PricesResult`, `ReturnsResult` and
  `LowOrderPrior`. All three are Results. No estimator holds a feature matrix.
- `LowOrderPrior.fpr::Option{<:LowOrderPrior}` and `HighOrderPrior.fpr` are type-bounded to the
  carrier, not to an estimator.
- The eight new test files name decisions rather than functions — "a pure forward is the identity",
  "a drop is spelled", "the collapse is convex, not an un-normalised sum", "dims is ignored on the
  routed path", "a subproblem measures its own neighbourhood, by refitting".

### B2-1 — an accuracy note on the plan, not on the code (no action needed on `dev`) — **CLOSED 2026-08-19**

The plan's API line reads `LowOrderPrior −f_mu −f_sigma −f_w` and `HighOrderPrior −f_kt −f_sk −f_V`.
Those six names are removed as **fields** but survive as **virtual reads** through
`@forward_properties` (`src/13_Prior/01_Base_Prior.jl:1088-1096`), so `pr.f_mu` still answers, and
answers `nothing` when there is no factor block — which is what the flat field did. Seven more
reads come free with the nesting (`f_ens`, `f_kld`, `f_ow`, and `f_D2`/`f_L2`/`f_S2`/`f_skmp` on
the high order carrier). None of the seven exists at base.

The break is therefore narrower than the plan states: it bites a caller who **constructs** with
`f_mu = ...`, not one who reads `pr.f_mu`. Worth correcting in the release notes, since the
migration burden is what a user will read them for.

**Closed 2026-08-19.** Verified in the REPL rather than only read off the source:

| Check | Result |
| --- | --- |
| `pr.f_mu` on a carrier with no factor block | `nothing` |
| `pr.f_mu` on a `FactorPrior` carrier | the 2-vector, `== f_mu` |
| `hop.f_kt`, `hop.f_sk`, `hop.f_V`, `hop.f_mu` | all answer; `f_mu` forwards through `pr` |
| `hasproperty` for `f_ens`, `f_kld`, `f_ow`, `f_D2`, `f_L2`, `f_S2`, `f_skmp` | all `true` |
| `LowOrderPrior(; ..., f_mu = ...)` | `MethodError`, *got unsupported keyword argument "f_mu"* |
| `hasproperty(pr, :z_sq)` | `false` — the one real read break, and it never shipped |
| `LowOrderPrior(; ..., fpr = ...)` with no `rr` | `ArgumentError`, the factor block is together-or-neither |

The togetherness invariant is respelled, not changed: at base the constructor already needed `rr`
with `f_mu`/`f_sigma` (`01_Base_Prior.jl:722-726` at `c975293745`); at `HEAD` it needs `rr` with
`fpr`.

The correction is written up in **`.review/release_notes_0.29.0.md`**, which is the seed for the
0.29.0 notes. The plan's Block 2 API line and its `[B2]` box are corrected too.

---

## Block 3 — constraint generation and the prior carrier  (`a0eda3861c..aa1d6d0b6a`)

### Boxes — Block 3

| Box | Verdict |
| --- | --- |
| ADR 0047: the linearity test inspects the expression, not the constraint's type | **The box mis-states the ADR — see B3-1. The code is right.** |
| `AssetSets` is removed; the replacement covers every former use | PASS |
| The generator handles the empty and the single-factor cases | PASS |
| `test_02b` covers a re-based and a non-re-based constraint | PASS |

Evidence:

- `AssetSets` has **zero** occurrences in `src/`. `UniverseSets` replaces it at
  `src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl:455` and gains `fkey`/`ufkey`
  beside the renamed `xkey`/`uxkey`.
- Empty: `@argcheck(!isempty(lce), IsEmptyError("lce cannot be empty"))`, line 173. Single factor:
  covered at `test/test_02b_factor_exposure_constraints.jl:50` — *"A single factor: the row is that
  factor's loadings column."*
- `test_02b` is 625 lines over 23 testsets and covers both directions, including
  "`lcse` is the only slot that admits a re-basis", "Vectors, and mixing bases", "Reduced loadings
  are not what a constraint reads", and "A stated basis is refused where the universe is replaced".
- The `FactorSpace` docstring carries an explicit warning that a **precomputed** `re` goes silently
  stale inside a cross-validation fold, and names the two spellings that do move. That hazard is
  the one a reviewer would otherwise have to find.

### B3-1 — the plan's first B3 box contradicts ADR 0047

The box reads: *"the linearity test inspects the expression, not the constraint's type."*
ADR 0047 decides the opposite, deliberately:

> the re-basis is declared by a wrapper type rather than by a field, so that constraints which are
> not linear forms **cannot represent one**.

There is **no runtime linearity test anywhere in `src/`** — grep for `is_linear`, `linear_form`,
`islinear` returns nothing. The mechanism is the type system: `FactorSpace` is admitted only by
`lcse`, and `LinearConstraintEstimator` is deliberately left with no space field so that `gcarde`
and `sgcarde` keep admitting it alone. The ADR argues this at length against the alternative of a
`space` field validated in the `JuMPOptimiser` constructor.

The box appears to have been written from ADR **0054**'s phrasing ("a degeneracy guard tests the
expression, not the term type"), which is a different decision in a different block.

Correct the box to read: *"the re-basis is unrepresentable on a non-linear-form estimator, rather
than validated"*, and then it passes — one `AbstractConstraintSpace` subtype exists (`FactorSpace`),
and only `lcse` accepts it.

Note for B1-2: `AbstractConstraintSpace` is the abstract type this block exports. It has exactly
one concrete member and no documented "define your own member" protocol, which weakens the case for
exporting it.

---

## Block 4 — phylogeny, networks and the custom value algorithm  (`aa1d6d0b6a..30ee2ea9b3`)

### Boxes — Block 4

| Box | Verdict |
| --- | --- |
| Two abstract types gain exports here | **Three, not two — see B1-2 and B4-1** |
| ADR 0048: the two removed neighbourhood types are fully replaced | PASS |
| `813bb63089` custom value algorithm interface | PASS |
| `3a185c3218` removed no public method silently | PASS, with a transient regression — see B4-2 |
| The +1,939 test lines cover the 14 closed issues | PASS |

Evidence:

- The block closes exactly 14 issues (#185 #197 #200 #202 #205 #206 #207 #208 #211 #212 #236 #237
  #238 #246) and adds 1,939 test lines over 16 files, which is what the plan states.
- **The retirement gate is a real equivalence test, not a paraphrase.**
  `test/test_12d_phylogeny_features.jl:24-40` asserts
  `Zb == Matrix{Float64}(phylogeny_matrix(NTE, rd.X).X) + I` — that is the retired
  `BinaryNeighbourhood`'s own implementation — and separately asserts that the *budget* does the
  cutting, not the decay (`all(==(1), separation_decay.(Ref(NoDecay()), 0:10, 2))` while
  `any(iszero, Zb)`). The graded case is pinned the same way. Only two textual mentions of the
  retired names survive, both explanatory.
- `813bb63089` adds `AbstractCustomValue <: AbstractAlgorithm` and
  `CustomExpectedReturnsValueAlgorithm`, and **exports neither**. Its doctest spells the supertype
  module-qualified (`PortfolioOptimisers.CustomExpectedReturnsValueAlgorithm`). This is the
  convention `CLAUDE.md` states, applied correctly — which is the best evidence that the six
  exported abstract types in B1-2 are accidents rather than decisions.

### B4-1 — the plan's withdrawal table misses `AbstractCentralityPolarity`

B4 exports **four** abstract types, not two: `AbstractCentralityPolarity`, `AbstractFeatureValue`,
`AbstractSeparationAlgorithm`, `AbstractSeparationDecayAlgorithm`. B5 withdraws three of them; the
plan's table names only two. `AbstractFeatureValue` is never withdrawn and survives to `HEAD` with
exactly one concrete member (`Scale`). Full accounting in B1-2.

On the box's question — "decide whether one block of an unwanted export is acceptable, or whether
the withdrawal folds forward" — it is moot for the three that B5 withdraws, since the PR merges as
one commit and no release ever carries them. It is **not** moot for `AbstractFeatureValue`, which
ships.

### B4-2 — `3a185c3218` dropped the informative error, and the next commit put it back

`3a185c3218` replaced `feature_target!`'s `if !isa(target, Pair) throw(ArgumentError(...)) end`
branch with dispatch on `::Pair` and `::AbstractVector{<:Pair}`. That deleted the
`feature_grammar_msg` path: a malformed target became a `MethodError` naming eight arguments
instead of a grammar message.

The very next commit, `86151923e9` "Bring back informative error", restores it as an untyped
fallback method (`src/12_ConstraintGeneration/06_AssetSetsMatrix.jl:748-752`), which is the better
shape — the check now lives in the dispatch table rather than in a branch. The regression does not
survive the block, so **no action is needed**; it is recorded because a reviewer reading commit by
commit will hit it and should not raise it.

`feature_targets!` was deleted by the same commit. It was never exported and never declared
`public`, so no public method was removed.

---

## Block 5 — distance and phylogeny, second pass  (`30ee2ea9b3..5fcba0c08f`)

### Boxes — Block 5

| Box | Verdict |
| --- | --- |
| ADR 0049 restricts non-negativity to the PMFG, not DBHT nor unrelated consumers | **The box mis-states the ADR — see B5-1. The code is right.** |
| The withdrawal of B4's abstract exports is complete | PASS — all six of B1–B5 are withdrawn at `HEAD`, `AbstractFeatureValue` included, and `test_43` gates the allow-list. |
| The five removed exports have no remaining documented caller | **False premise — see B5-2** |
| `ov` means the same thing on all three centrality types | PASS, and it is five types, not three — see below |

Evidence:

- `ov::Option{TopologyOnly}` is the field on **five** centrality types, not the three the API
  note names: `BetweennessCentrality`, `ClosenessCentrality`, `EigenvectorCentrality`,
  `RadialityCentrality` and `StressCentrality`. All five gain it in this block — none carries it at
  `30ee2ea9b3` — with one type, one default (`nothing`), and one read, `centrality_polarity`'s
  dispatch: a `Nothing` keeps the type's own polarity and a `TopologyOnly` answers `nothing`. The resolution is centralised rather than read inline, and the docstring says why:
  three algorithms carry no `ov` field at all, so `ct.ov` cannot be written for them.
- ADR 0049 is the strongest ADR in the PR. It names the two unsafe DBHT aggregations with line
  numbers (`DirectHb` at `04_DBHT.jl:942-948`, `BubbleMember` at `:1084-1086`), shows why the third
  (`BubbleCluster8s`) is safe (`all_cont = 3 * (size - 2)` is a count, so the ratio is a mean), and
  states plainly that the bound it applies is **wider than its own provenance** and why that trade
  was taken.

### B5-1 — the box inverts ADR 0049

The box says non-negativity *"is not applied to DBHT, which needs it for its own reason."*
ADR 0049 decides the opposite: **the requirement is DBHT's**, and the bound is applied at all four
`PMFG_T2s` callers, DBHT included. The ADR's own heading reads *"The requirement is DBHT's, not the
PMFG's, and the bound is deliberately wider."*

What the code does, which is what the box should be checking:

| Bound | Sites |
| --- | --- |
| `AbstractNonNegativeSimilarityMatrixAlgorithm` (narrow) | `DBHT.sim`, `LoGo.sim`, `calc_weighted_adjacency_graph` |
| `AbstractSimilarityMatrixAlgorithm` (wide) | `FeatureDistance.sim`, `distance_to_similarity`, `assert_similarity_domain` |

So unrelated consumers do keep the wide bound and `AngularSimilarity` stays usable there, which is
the half of the box that is meaningful, and it passes.

This is the third box in the plan that restates an ADR's title rather than its decision — see also
B3-1 and B5-2. Recommend re-deriving the remaining blocks' boxes from the ADR bodies.

**Applied 2026-08-19.** The box is re-worded to check the placement above and ticked, with a fourth
narrow site the table missed: `NetworkEstimator.alg`, bound through `Tree_SimMat`
(`06_Phylogeny.jl:1056`). Verified by running each refusal — the keyword route raises a `TypeError`
naming the bound and the positional route a `MethodError`, while `FeatureDistance(; sim =
AngularSimilarity())` constructs and `assert_similarity_domain` accepts it. ADR 0049's Decision
paragraph calls the refusal a `MethodError`; on the keyword route it is a `TypeError`. The decision
holds; only the word is loose.

### B5-2 — B5 removes three exports, not five, and none of them is a similarity type

The plan's API note says *"5 exports removed, among them `ExponentialSimilarity`,
`GeneralExponentialSimilarity`, `MaximumDistanceSimilarity`."* Computed by parsing every `export`
statement in `src/` at both endpoints:

- **Removed (3):** `AbstractCentralityPolarity`, `AbstractSeparationAlgorithm`,
  `AbstractSeparationDecayAlgorithm` — all abstract types, all withdrawals of B4 additions.
- **Added (6):** `AbstractNonNegativeSimilarityMatrixAlgorithm`, `AbstractSimilarityMatrixAlgorithm`,
  `HopCountQuantile`, `PathLengthQuantile`, `TopologyOnly`, `resolve_separation`.

The three similarity types the plan names as removed are **still exported at `HEAD`**
(`src/09_Distance/04_Similarity.jl:487-489`). The block's only edit to that statement prepends the
two abstract types to a multi-line `export`; the three concrete names simply moved to the second
line.

That is almost certainly how the error arose — a line-oriented read of the diff shows
`-export MaximumDistanceSimilarity, ExponentialSimilarity, GeneralExponentialSimilarity,` and
misses that the following line is unchanged continuation. Any export accounting in this review must
parse multi-line `export` statements, not diff lines. The per-block export tables in this report are
computed that way.

So this box has nothing to check, and B5 removes no user-facing name at all.

### B5-3 — the two unsafe aggregations are unreachable with a negative, and the zero route crashes

Investigated after the block closed, because ADR 0049 rests its bound on the two aggregations and
the plan never asked whether they are live. Two questions: is a negative weight reachable, and is
the residue noise.

**A negative is unreachable at `HEAD`, twice over.** `DBHTs` is the only caller of the bubble
machinery — `DirectHb`, `BubbleCluster8s`, `BubbleMember` and `HierarchyConstruct4s` have no other
caller in `src/`, and none of the four is exported or `public`. Every one of them reads the same
`Rpm`, and `Rpm = PMFG_T2s(S)[1]` (`04_DBHT.jl:1449`). `PMFG_T2s` refuses a `NaN` and a negative
before it returns (`:146-149`), and its output weights are `W`'s own entries
(`A = W ⊙ ((A + A') .== 1)`, `:221`). So the ADR's type bound and `assert_similarity_domain` are
message quality, and `PMFG_T2s`'s own check is the load-bearing guard. Reproduced: `DBHTs` on an
`S` with one `-0.3` raises `DomainError: all(x -> x >= 0, W) must hold. Got minimum(W) => -0.3`.

**The arithmetic hazard is real at the function level.** `BubbleMember` called with a bubble whose
internal weights are all exactly zero — legal non-negative input — divides `0/0`, and `argmax`
selects the resulting `NaN`, so the vertex is assigned to the empty bubble over one carrying
`2.4` of weight. The failure needs no negative at all. `DirectHb` has the matching shape: it writes
the losing sum into `Hc`, and `Sep = iszero.(sum(Hc; dims = 2))` reads an exact zero as a
separating bubble.

**The zero route reaches something worse first, and it is pre-existing.** ADR 0049 admits an exact
zero by decision, under *Non-negative, not positive*, justified by `LogDistance` mapping an exactly
zero correlation to `Inf` and `exp(-Inf)` to `0`. On that route a zero weight is not a weak edge:
`A = W ⊙ ((A + A') .== 1)` makes it an **absent** edge. Measured on a 12-asset fixture whose
correlation matrix has exact zeros (orthogonal ±1 columns), the PMFG keeps **21 of its 30** edges,
and the run dies with `BoundsError: attempt to access 7×4 Matrix{Float64} at index [9, 4]` inside
`turn_into_Hclust_merges` (`04_DBHT.jl:1389`). Reproduced through the public API:
`clusterise(ClustersEstimator(; de = Distance(; alg = LogDistance()), alg = DBHT(; sim = ExponentialSimilarity())), X)`.
Isolated to the zero, not to the `Inf`: replacing the infinite distances with `20.0` still crashes,
while replacing the zero similarities with `exp(-20)` runs clean.

**Recommendation.** Change no DBHT arithmetic — it follows the reference implementation, and the
negative it is charged with cannot arrive. Two smaller moves, in order of value:

1. Amend ADR 0049 under *Non-negative, not positive*. The clause is arithmetically right about the
   gain argmax and wrong about the graph: a zero weight removes an edge. The route the clause
   names to justify admitting zero is the route that breaks the PMFG.
2. If a guard is wanted, put it where the structure is consumed rather than in the aggregations —
   after `Rpm = PMFG_T2s(S)[1]`, assert the edge count is `3N - 6` and raise a `DomainError`
   naming the configuration. That turns a `BoundsError` about a matrix index into a message, and
   it covers all four `PMFG_T2s` callers, not only DBHT.

Neither is a B5 blocker. The crash predates the PR, and no shipped default reaches it: an exactly
zero sample correlation does not occur in the repo's own fixtures, and ADR 0049 already records
that `Denoise()` produces none on `SP500`.

---

## Block 6 — risk measures and uncertainty sets  (`5fcba0c08f..81893f41aa`)

### Boxes — Block 6

| Box | Verdict |
| --- | --- |
| `factory` and `concrete_typed_array` lose their exports | **False premise — they do not. See B6-1.** |
| ADR 0051: single resolution point, no resolved value cached on an estimator | PASS |
| ADR 0050: `val` and `mu` are not two names for one thing | PASS |
| `Kurtosis` and `Skewness` gain `pe`; the default preserves behaviour | PASS |
| `d9d4e51126` bumps the version mid-branch, superseded by the final version | **FAILS — see B6-2. It is the final version, and it is wrong.** |

Evidence:

- All seven claimed field additions are present: `BoxUncertaintySet.val`,
  `EllipsoidalUncertaintySet.val`, `L1UncertaintySet.mu`, `SignedL1UncertaintySet.mu`,
  `Kurtosis.pe`, `Skewness.pe`, `LowOrderPrior.o_X`.
- ADR 0050 distinguishes the two nouns on a stated principle: the ℓ1 family is mean-only (its `ucs`
  and `sigma_ucs` throw), so `mu` is precise there; `BoxUncertaintySet` and
  `EllipsoidalUncertaintySet` serve both the mean and the covariance axis, so `val` is the neutral
  noun. The read is inside the dispatching method, so no `isa` chain appears.
- ADR 0051's "an Estimator never holds a Result" is enforced by a **type bound**, and a deliberately
  narrow one: `pe::Option{<:AbstractPriorEstimator}` on `Kurtosis`, `Skewness` and
  `VarianceSkewKurtosis`, where every other `pe` in the library is `PrE_Pr` and would admit a
  precomputed Result. All three default to `nothing`, so previous behaviour is preserved.
- `resolve_deferred_quantities` is a method family, one per measure type — dispatch, not a branch.
  The "single resolution point" claim has one documented exception, and `test_09e` pins it by name:
  *"`MedianAbsoluteDeviation` keeps two resolution points"*. `test_09e` is 20 testsets, every one
  named for a decision.

### B6-1 — nothing loses an export in B6

Computed by parsing every `export` statement at both endpoints: **B6 removes zero exports and adds
zero exports.** `factory`, `concrete_typed_array`, `MaxValue`, `MinValue`, `ModeValue`, `ProdValue`
and `SumValue` are all still exported at `HEAD` — `src/02_Tools.jl:2713-2714` — and all five types
still exist (`src/02_Tools.jl:2362`, `:2504`, …).

This is part of a wider problem with the plan's API notes; the full ledger is in the summary
section **P-1** at the end of this report.

### B6-2 — the branch ships version 0.28.0, which is already in the registry

This is a **release blocker**, and it is the plan's own "Done when" item.

| Ref | `Project.toml` version |
| --- | --- |
| `main` | 0.27.1 |
| B1 – B5 endpoints | 0.27.1 |
| `3b17054a64` (B6) | 0.27.1 |
| **`d9d4e51126` (B6)** | **0.28.0** |
| B7 – B11 endpoints, `HEAD` | 0.28.0 |

The bump to 0.28.0 happens once, in B6, and is **never superseded** — `HEAD` still reads
`version = "0.28.0"`.

Meanwhile `v0.28.0` is an existing release tag pointing at `9adac7735b`
("Initialize first element of W_list to minimum value"), and `git merge-base --is-ancestor v0.28.0
HEAD` confirms that commit is already an ancestor of `dev`, absorbed by the empty merge
`9abbaf0b1b` that B11 describes.

So the branch would publish 0.28.0 a second time, over a different tree. The plan's own conclusion
is the fix: *"this PR changes public APIs in ten of the eleven blocks, so the next version is
**0.29.0**."* Set `version = "0.29.0"` before merge.

Note the B6 box as written would let this pass — it asserts the bump "is superseded by the PR's
final version", and nothing supersedes it. Re-word the box to check the value at `HEAD`.

---

## Block 7 — the return expression and the frontier sweep  (`81893f41aa..a06db73367`)

### Boxes — Block 7

| Box | Verdict |
| --- | --- |
| ADR 0053: term-level and expression-level weights compose once, not twice | PASS |
| ADR 0054: the degeneracy guard tests the expression, not the term type | PASS |
| `lb` becomes `settings` on both return types; the migration is documented | PASS |
| Four commits only widen CI tolerances, each a host-sensitivity fix | **Partly. Two are not host sensitivity, and there is a fifth in B8 — see B7-1.** |
| The six new `test_18*` files cover multiplicity, the sweep cap, the no-return case | PASS |
| Export `HierarchicalOptimiser` removed | **False premise. It is still exported at `HEAD`.** See P-1. |

Evidence:

- `lb` moved off both return types into a new `JuMPReturnsSettings` carrier holding
  `scale`, `lb`, `rte`, `fee`, `mic`. `ArithmeticReturn` is now `settings`, `ucs`, `mu`;
  `LogarithmicReturn` is `settings`, `w`. `NoReturn` is new and carries `settings` alone.
- ADR 0054 is the strongest-argued decision in this block. The guard is two predicates on
  **state**, and the return one is deliberately fused rather than a disjunction, with the
  counterexample written out: `[NoReturn(), ArithmeticReturn(; settings = JuMPReturnsSettings(;
  rte = false))]` has every term out of `:ret`, yet `all(isa NoReturn) || all(!rte)` is `false`.
  `test_18q` pins exactly that distinction — "the return predicate is fused" beside "the risk
  predicate composes".
- The six new files are exactly the six the plan names, and each maps to a decision:
  `test_18p_combination_weight.jl` to ADR 0053, `test_18q_degeneracy_guard.jl` to ADR 0054.

### B7-1 — the tolerance accounting is off by one commit and by two categories

Reading all four commits the box names, plus a sweep of every `rtol` change from B7 to `HEAD`:

**`570c73b688` is not a host-sensitivity fix.** Its own comment says why:

> `res_m1` is a lone `scale = 50` measure, so the model and the barrier both drop the combination
> weight and the sub-problems solve on numbers 50 times smaller. That shifts the solver's
> conditioning by about 1e-5 against the vector runs, which keep the weight.

That is a numerical consequence of `03fb5c9656` "Drop a lone return term's combination weight" —
ADR 0053's own change, three commits earlier in the same block. It is a legitimate widening, but it
belongs in a different category from the other three: **the PR changed the numbers, and the
tolerance follows.** Classifying it as host sensitivity hides that the ADR 0053 change moved
results at the 1e-4 level.

**`edbad72c63` is not a widening at all — it is a tightening.** It deletes the
`JULIA_NUMERICS_DRIFT` / `@test_skip success` gate and asserts against regenerated references.
That is strictly better than skipping. The residual risk is inherent and unavoidable: references
regenerated from the current tree cannot detect a regression introduced by the same tree. The
commit attributes the drift to Julia 1.12.7 (#333), which is plausible and not checkable from the
diff.

**There is a fifth tolerance change, and it is in B8, not B7.** `91f588dc47` — the B8 endpoint —
edits the same ladder in `test/test18_setup.jl`:

| Case | Before | After |
| --- | --- | --- |
| `i == 12` | 5e-4 | 1e-3 |
| `i == 42` | 0.1 | **0.5** |

`i == 42` is the loosest assertion in the suite, and the comment added with it is candid about why:

> 42 needs the widest of them because its model is marginal: the runner logs a **solve failure and
> falls back, and the fallback lands on a different vertex.**

At `rtol = 0.5` that case no longer tests the optimiser's answer — it asserts that two different
vertices agree within 50%. It cannot fail on anything short of a gross error. The maintainer's own
policy is that a failed solve is not a defect, and I am not calling this one; but the test should
say what it is. Better shapes than a 50% tolerance: assert `retcode`, or assert the objective value
rather than the weight vector, or mark the case broken with the issue number.

Counting all lines in `test/test18_setup.jl` whose tolerance is 0.05 or looser: **9 at the base of
the PR, 11 at `HEAD`.** The loosening is real but small, and both new entries are commented.

Neither B7's box nor any B8 box covers `91f588dc47`'s tolerance edit. Add it to B8.

---

## Block 8 — maintainability, ergonomics and security  (`a06db73367..91f588dc47`)

### Boxes — Block 8

| Box | Verdict |
| --- | --- |
| ADR 0055: HRP and Schur pin the branch order the same way | PASS |
| ADR 0056: the setup action floats on `@latest`; no toolchain pin reintroduced | PASS |
| The security fixes add no new load-time code path | PASS |
| The consolidations in `ce5718009a` are behaviour-preserving | Not fully verifiable statically — see B8-2 |
| Deleted `field_dict` / `arg_dict` entries have no remaining user | **PASS, and better than asked — see B8-1** |
| *(missing box)* Four exported aliases are renamed with no shim | **See B8-3** |
| *(missing box)* `91f588dc47` widens two more tolerances | See B7-1 |

Evidence:

- **ADR 0055.** `HierarchicalRiskParity` and `SchurComplementHierarchicalRiskParity` carry the
  *same* comment at the same two sites each — "No `branchorder`: recursive bisection splits
  `clr.res.order`, so the leaf permutation is the algorithm's input and must stay `:optimal`
  (ADR 0055)" — and the same docstring paragraph stating that a passed `branchorder` is absorbed by
  `kwargs` and ignored. Pinned the same way, and the swallowing is documented as deliberate.
- **ADR 0056.** Eight `julia-actions/setup-julia@latest` uses across the workflows and **zero**
  `julia-version:` keys anywhere in `.github/workflows/`. No pin was reintroduced.
- **Security.** The `__init__` / Preferences.jl channel is **not new**: `function __init__` and
  `using Preferences` both date to `508839674b` (release v0.24.0), and `Preferences` is a
  dependency in `Project.toml` at both the PR base and `HEAD`. B8 adds no load-time path. (The
  later `relaxed_preferences_msg` warning arrives in B10's `ce9ab55e32` — a warning on the existing
  channel, not a new one. Noted under B10.)

### B8-1 — the docstring dictionaries are now exactly balanced

Parsing every `arg_dict` / `field_dict` / `ret_dict` key defined in `src/01_Base.jl` against every
`…[:key]` reference in the repository:

| Ref | `arg_dict` keys | referenced | orphans | dangling |
| --- | --- | --- | --- | --- |
| `c975293745` (base) | — | — | **44** | 0 |
| `a06db73367` (B7 end) | — | — | 42 | 0 |
| **`91f588dc47` (B8 end)** | 594 | 594 | **0** | 0 |
| `HEAD` | 594 | 594 | **0** | 0 |

B8 did not merely avoid leaving a dangling reference — it cleared a **pre-existing backlog of 44
orphaned entries** and landed the dictionary exactly balanced. Nothing is referenced that is not
defined, at every endpoint. This is the strongest single result in the review, and it matters
because `field_dict` keys render lazily: a dangling key fails only when its docstring is rendered.

One residue, **pre-existing and not this PR's**: `ret_dict` has 9 keys with no `ret_dict[:key]`
user — `alg`, `algw`, `cev`, `cskew`, `cskewV`, `kte`, `mev`, `sk`, `vev`. The count is 9 at the
base and 9 at `HEAD`, so nothing regressed. Of the nine, only `kte` is reachable symbolically
through an `@windowed_estimator` `forward` entry; the other eight look dead. `CLAUDE.md`'s rule
("delete entries that lose their last user") applies to `ret_dict` by the same argument. A small
follow-up, not a merge blocker.

### B8-2 — what I could and could not verify in `ce5718009a`

`ce5718009a` plus `e9b478b7f7` and `91f588dc47` come to 151 files, +4,192 / −3,545. Behaviour
preservation across a consolidation of that size is what a test suite establishes, not what a
reading establishes, and the block endpoint was green. I verified the things the tests cannot see:

- the public API delta (exports and exported-type fields) — see the ledger in **P-1**;
- the docstring-dictionary balance above;
- that no abstract type gained an export in this block;
- that the tolerance ladder changed, which the plan's boxes do not cover (B7-1).

I did **not** re-derive each of the 20 source areas by hand. If more assurance is wanted for this
box, the cheapest high-value target is the set of 44 `arg_dict` deletions: confirm each deleted
key's docstring still renders, since that is the failure mode CI does not catch.

### B8-3 — four exported aliases are renamed with no deprecation, and no box covers it

`91f588dc47`/`ce5718009a` rename four exported aliases:

| Old | New | Target |
| --- | --- | --- |
| `MaxSR` | `MaxRa` | `MaximumRatio` |
| `RVaR` | `RLVaR` | `RelativisticValueatRisk` |
| `RVaR_RG` | `RLVaR_RG` | `RelativisticValueatRiskRange` |
| `R_RDaR` | `R_RLDaR` | `RelativeRelativisticDrawdownatRisk` |

All four old names have **zero** occurrences in `src/` — there is no shim, consistent with the
library's practice (ADR 0044's clean break, and `src` has never carried an `@deprecate`). The
renames themselves look right: `RLVaR` disambiguates *relativistic* from *relative*, which `RVaR`
did not, and `MaxRa` matches `MaximumRatio` rather than naming the Sharpe ratio specifically.

The problem is only that **the plan gives B8 no API line at all**, so these four breaking renames
are invisible to the review and would reach the release notes only by accident. They are four of
the six export removals in the whole PR.

---

## Block 9 — architecture review, the range measure  (`91f588dc47..9237475ea1`)

### Boxes — Block 9

| Box | Verdict |
| --- | --- |
| ADR 0057: every non-fused range declares `range_tails`; every fused range declares none | PASS at runtime, ungated by a test — see B9-1 |
| The delegation does not double-count the base measure | **PASS, verified** |

Evidence:

- `range_tails` has a **throwing fallback** on `AbstractBaseRiskMeasure`
  (`src/19_RiskMeasures/01_Base_RiskMeasures.jl:227`) whose message states the rule exactly:
  *"Only a range risk measure that is the sum of two point measures decomposes; a measure that
  fuses its two tails into one formulation declares none."* Seven concrete methods declare tails.
- **No double-counting.** `set_range_risk_constraints!`
  (`20_RiskMeasureConstraints/01_BaseRiskConstraints.jl:499`) reads the two tails, builds each with
  `loss = true` / `loss = false`, sums them, and registers the sum. Only the composite reaches the
  objective and the bound, because **all seven** `range_tails` methods construct their tails with
  `RiskMeasureSettings(; rke = false)` and no upper bound. I checked all seven individually; the
  only `range_tails` body without `rke = false` is the throwing fallback, which has no tails.
  Each tail also builds under its own `nested_index`, so a range nested in a range stays
  collision-free.

### B9-1 — ADR 0057's invariant has no census, where ADR 0060's does

The fallback enforces one direction: a non-fused range that forgets `range_tails` throws the first
time it is used. It does **not** enforce the other: a *fused* range that wrongly declares
`range_tails` would be decomposed and double-counted, and nothing objects.

`grep -rl range_tails test/` returns nothing — the invariant has no test at all. Compare ADR 0060
in the very next block, which adds `test_41_constructor_docstring_drift.jl` as a census precisely so
that its invariant cannot drift. The same shape would suit ADR 0057: enumerate the range risk
measures, assert that each one either declares `range_tails` or is on a written fused list.

Low severity — the seven declarations are correct today — but it is the cheapest gap to close in
the late blocks, and the PR already contains the pattern to copy.

---

## Block 10 — architecture review seams; entropy pooling  (`9237475ea1..5f99feb948`)

### Boxes — Block 10

| Box | Verdict |
| --- | --- |
| ADR 0060: `test_41`'s census covers every constructor | **PASS, and it is anti-vacuous — see below** |
| ADR 0064: the entropy pooling split preserves every dispatch | **PASS, verified by census** |
| ADR 0058: the dims guard and the orientation are one call, at one site | **PASS, verified and test-gated** |
| ADR 0067: the fold loop is one seam; `fd7c9324fe` found no second one | PASS |
| The export `UniverseSets`, added in B3, is removed here | **FALSE — it is exported at `HEAD`. See P-1.** |
| The three removed types have no remaining caller | **False premise — one type was removed, not three. See P-1.** |
| `test_40` and `test_42` gate what their ADRs claim | PASS |

### What I verified, and how

**ADR 0060 / `test_41`.** The census is complete *and* protected against silently doing nothing.
It walks every `src/` file, collects every keyword-taking definition and every documented
`# Constructors` block, and checks both directions — a documented signature with no definition
(`unmatched`) and one that disagrees with the code (`mismatched`). Crucially it asserts **floors**:
`length(files) >= 190`, `sum(length, values(real)) >= 1000`, `length(docs) >= 300`, with the comment
*"An extractor that silently stops finding anything must not pass."* That is the failure mode most
census tests have, and this one is armoured against it.

**ADR 0064 / the entropy pooling split.** I diffed the old single file against the three that
replace it, comparing `(function name, first argument type)` pairs. Old: 20 pairs. New: 47. Exactly
three old pairs are absent, and all three are accounted for:

- `CVaREntropyPooling(::Tuple)` ×2 — **renamed** to `ConditionalValueatRiskEntropyPooling`, same
  two signatures, matching the export ledger.
- `prior(::EntropyPoolingPrior{<:Any, …})` — **refactored**, not lost: the parametric constraint is
  replaced by an algorithm-dispatched family, `ep_prior(::H0_EntropyPooling)` and
  `ep_prior(::StagedEP)`. That *is* ADR 0064's decision ("the prior dispatches on its algorithm").

No dispatch was dropped.

**ADR 0058 / the dims guard.** Total, and gated. `dims_oriented` is the one call; there are 45 call
sites, and exactly **one** hand-rolled `dims == 2 ? transpose(...)` remains in `src/` — inside
`dims_oriented` itself (`src/01_Base.jl:3576`). `test_08d` closes it with a census named
*"no leaf spells the dims guard or the orientation by hand"*.

**ADR 0067 / the fold loop.** One `fold_loop`, no residual hand-rolled fold iteration anywhere in
`src/`. `fd7c9324fe`'s audit found no second seam. What it did find is recorded in an **Amendment
(2026-08-19)** to ADR 0067 rather than a rewrite, which is the pattern `CLAUDE.md` requires, and the
amendment is unusually honest: a JET-measured runtime dispatch caused by passing the element type as
a keyword *value* (constant propagation lost across one forwarding hop), fixed by making it a
positional `::Type{ElT}`, with the note that *"The gain is a clean JET report, not a faster
backtest"* because the lost dispatch ran once per CV call, not once per fold.

**`test_40`.** A reflective census over `methods(optimise)` that **names no optimiser**, so a
shortcut written next year is covered the day it is written. It asserts exactly two things about
every shortcut: one type parameter pinned to `Nothing`, and that parameter at the position of the
field `fb`. The comment records the real defect this caught — `22_DiscreteFiniteAllocation.jl` wrote
four parameters for a five-field estimator, so `Nothing` landed on `wf`, the method could never
match, and the miscount was silent.

**ADR 0065 (spot-checked).** `densify` is at the seam, and its docstring documents the precise
silent-wrong-answer it prevents: without densification `cov(::SimpleCovariance, X, w; mean = mu)`
resolves to `Statistics.covm(x, xmean, y, ymean, vardim)` — the cross-covariance of `X` against the
weight vector — returning an `N × 1` matrix in place of `N × N` **and raising nothing**. That is the
most dangerous defect fixed in this PR.

**ADR 0069 (spot-checked).** The integer CVaR window is ascending, and the docs say why it must be:
*"the monotonicity constraint makes the marked set a suffix of the ascending order, which is what
makes the expression the CVaR rather than the mean of an arbitrary subset of probability α."*
`test_12h` is 655 lines over 14 testsets covering levels, formulations, groups and guards.

### B10-1 — scope I did not cover

Twelve ADRs land in this block. I verified 0058, 0060, 0064, 0067 in depth and spot-checked 0065 and
0069. I did **not** individually verify **0059, 0061, 0062, 0063, 0066, 0068**. The block's boxes do
not ask about those six, but the plan's instruction to "read all twelve ADRs first as a block" does,
and a reviewer closing B10 should say which six are still open. None of the six is implicated in any
finding above.

`relaxed_preferences_msg` — the `@warn` on load-time preferences that widen a guard — arrives here,
in `ce9ab55e32`, not in B8. It is a warning on the pre-existing `__init__` channel, not a new
load-time path, so B8's security box is unaffected.

---

## Block 11 — the move back to Documenter  (`5f99feb948..30407a2b57`)

### Boxes — Block 11

| Box | Verdict |
| --- | --- |
| `docs/package.json` is deleted and nothing calls it | **PASS** |
| The Documenter move builds the docs green | Not checked — requires a docs build, which is the maintainer's to run |
| The regenerated examples and notebooks are generated files | PASS |

Evidence:

- `docs/package.json` does not exist, and there is **no** reference to `npm`, `node`, `yarn` or
  `vitepress` anywhere in `.github/workflows/`. The removal is complete.
- Outside the generated trees, the block deletes six research harness files and four review report
  HTML pages, and makes one nine-line change to `test/test_26_docs.jl`.

### B11-1 — the one real code change in B11 is a good catch, and worth naming

The `test_26_docs.jl` change is not cosmetic:

```julia
-            if !isabstracttype(T)
+            if !isabstracttype(T) && parentmodule(T) === PortfolioOptimisers
```

The comment explains it: the runner gives each test file its own module but **not** its own process,
so an estimator declared by another test file stays in the worker and `subtypes` finds it here.
Which files share a worker changes from run to run, so without the filter the catalogue census is a
**scheduling flake** — it would fail intermittently, on a different name each time, for a reason
having nothing to do with the change under test.

That is exactly the class of failure that gets dismissed as "CI being flaky" and left to rot. Fixing
it in the docs-infrastructure block is right, and the box list should record it rather than treating
B11 as nine lines of docs plumbing.

---

## Summary findings

## P-1 — the plan's API notes are unreliable; here is the verified ledger

The plan's per-block API lines drove eight checkboxes. Several are wrong, and three boxes would have
been ticked on a false premise. The cause looks mechanical: a line-oriented read of a diff cannot
see that a multi-line `export` statement's later lines are unchanged continuation. I fell into the
same trap once and corrected it by parsing the statements instead.

Everything below is computed by parsing every `export` statement and every `struct` declaration in
`src/` at each block endpoint, not by reading diff lines.

### Claims that are wrong

| Plan says | Actually |
| --- | --- |
| B5: "5 exports removed, among them `ExponentialSimilarity`, `GeneralExponentialSimilarity`, `MaximumDistanceSimilarity`" | B5 removes **3** exports, all abstract types. All three similarity types are still exported (`src/09_Distance/04_Similarity.jl:487-489`). |
| B6: "5 types removed (`MaxValue`, `MinValue`, `ModeValue`, `ProdValue`, `SumValue`) and 6 exports with them, including `concrete_typed_array` and `factory`" | B6 removes **zero** exports and **zero** types. All seven names are exported at `HEAD` (`src/02_Tools.jl:2713-2714`). |
| B7: "Export `HierarchicalOptimiser` removed" | Still exported at `HEAD`. B7 removes **zero** exports. |
| B10: "3 types removed (`CVaREntropyPooling`, `LinearConstraintEstimator`, `UniformValues`)" | Only **`CVaREntropyPooling`** was removed, and it was *renamed* to `ConditionalValueatRiskEntropyPooling`. `LinearConstraintEstimator` (`02_LinearConstraintGeneration.jl:1791`) and `UniformValues` (`:862`) both exist and are exported. |
| B10: "4 exports removed (`LinearConstraint`, `LinearConstraintEstimator`, `PartialLinearConstraint`, `UniverseSets`)" | **One** export removed. All four named types are exported at `HEAD` (`:2038-2040`). |
| B10 box: "The export `UniverseSets`, added in B3, is removed here. That is intended." | `UniverseSets` **is exported at `HEAD`**. The withdrawal table's last-but-one row is wrong too. |
| B6: "`SubsetResamplingResult` −`scale`" (stated under B10) | `scale` was removed from **`SubsetResampling`**, the estimator, not from the Result. |
| B4: "Two abstract types gain exports here" | **Four** do. B5 withdraws three. See B1-2. |

### Claims that hold

`AssetSets` really is removed in B3, type and export. Every struct-field addition the plan names in
B1, B2, B6 and B7 is present.

### The verified ledger, base `c975293745` → `HEAD`

Exported names: **705 → 768**.

**Exports removed — six, all of them:**

| Name | Block | Disposition |
| --- | --- | --- |
| `AssetSets` | B3 | replaced by `UniverseSets` |
| `MaxSR` | B8 | renamed `MaxRa` |
| `RVaR` | B8 | renamed `RLVaR` |
| `RVaR_RG` | B8 | renamed `RLVaR_RG` |
| `R_RDaR` | B8 | renamed `R_RLDaR` |
| `CVaREntropyPooling` | B10 | renamed `ConditionalValueatRiskEntropyPooling` |

**Types that no longer exist: two** — `AssetSets`, `CVaREntropyPooling`. Nothing else.

**Exports added: 69.** Six are abstract types (B1-2). The rest are the feature/distance family, the
separation and decay families, the tail-view family, the new Result types, and the renamed aliases.

**44 exported types change fields.** The four the release notes most need, because a caller cannot
discover them from a `MethodError`:

| Type | Removed | Added |
| --- | --- | --- |
| `JuMPOptimiser`, `HierarchicalOptimiser`, `NestedClustered` | `cle_pr` | `x_src`, `z_src` |
| `ArithmeticReturn`, `LogarithmicReturn` | `lb` | `settings` |
| `LowOrderPrior` | `f_mu`, `f_sigma`, `f_w` | `o_X`, `fpr`, `Z` |
| `EntropyPoolingPrior` | `var_alpha`, `cvar_alpha`, `ds_opt`, `dm_opt` | `evar_views` |

The full 44-row table is in `.review/api_ledger.txt`.

**Recommendation.** Regenerate the plan's API notes from the parser rather than the diff, and use
the ledger as the seed for the 0.29.0 release notes. Eight boxes across five blocks depend on it.
The prior-carrier row is written already — see `.review/release_notes_0.29.0.md` (B2-1).

## P-2 — the two blocking items

1. **Set `version = "0.29.0"`.** `Project.toml` reads `0.28.0` at `HEAD`, and `v0.28.0` is an
   existing tag at `9adac7735b`, already an ancestor of `dev`. See B6-2.
2. **Decide the six exported abstract types.** Base 7 → `HEAD` 13. One
   (`AbstractNonNegativeSimilarityMatrixAlgorithm`) is explicitly sanctioned in ADR 0049; the other
   five are not, and `AbstractPhylogenyFeatureAlgorithm` appears in no ADR at all. Nothing gates
   the rule. See B1-2.

## P-3 — three boxes restate an ADR's title instead of its decision

B3-1 (ADR 0047), B5-1 (ADR 0049) and B6's version box each check something the ADR decided against,
or something no longer true. In all three cases the **code is right and the box is wrong**. Re-derive
the boxes for any block reopened after fixes from the ADR bodies, not their titles.

## What this PR does well

Worth recording, because a review that only lists defects misrepresents the work:

- **ADR 0049** names the two unsafe DBHT aggregations by line number, proves the third safe, and
  states plainly that the bound it applies is wider than its own provenance and why.
- **ADR 0065** documents a silent wrong answer — an `N × 1` cross-covariance returned in place of an
  `N × N` covariance, raising nothing — and fixes it at the seam.
- **`test_40` and `test_41` are censuses that name nothing**, so they cover code not yet written, and
  `test_41` asserts floors so a broken extractor cannot pass vacuously.
- **The `arg_dict` / `field_dict` pair is exactly balanced at `HEAD`** — 594 defined, 594
  referenced, zero orphans, zero dangling — having cleared a pre-existing backlog of 44.
- **ADR 0067's amendment** measures its own change honestly and says the gain is a clean JET report
  rather than a faster backtest.
- Test names throughout describe decisions ("the collapse is convex, not an un-normalised sum",
  "a drop is spelled", "dims is ignored on the routed path"), which is what makes the suite reviewable.
