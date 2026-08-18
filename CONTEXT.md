# PortfolioOptimisers.jl — Domain Glossary

The library is organised as a workflow:

> data → moments → prior → optimisation → post-processing

("Pipeline" is reserved for the reified workflow Estimator below, not this informal organisation.)

with a small set of cross-cutting abstractions (Estimator / Algorithm / Result / Factory) that everything is built from. This glossary follows that order. Concept families list their concrete variants as one-line bullets.

## 1. Core Abstractions

**Estimator**
A configuration object that encodes a statistical or mathematical method along with its hyperparameters. Estimators are the primary input to high-level user-facing functions (`optimise`, `prior`, `factory`, etc.). Sometimes callable (functor) when it makes sense, but that is not required. Estimators may compose other estimators when the inner estimator is also independently useful.

**Algorithm**
A type used inside an Estimator to select or modify its computational behaviour. Dispatched on internally. Not intended to be used directly by the user on their own — always consumed through an Estimator.

**Result**
A plain data struct that holds the computed outputs of a function applied to an Estimator. Never callable. Passed downstream as inputs to further computation.

**An Estimator never holds a Result internally.**
A load-bearing rule, not a stylistic one, and it applies to every Estimator in the library. An Estimator says *how to compute for whatever input it is given*; a Result is *the answer for the input it was computed on*. A slot admitting either produces an object that is configuration on the outside and data on the inside, which is the shape the surrounding machinery cannot reason about: Factory cannot reconfigure it, and every guard written to reject a precomputed input misses it, because the outermost type is an Estimator and the data sits one level down.
Where a caller genuinely has precomputed structure, it goes in the **Result** the corresponding function returns — that is what the Result type is for, and the slot that consumes results already exists and is already guarded. Enforce it with the **type bound on the field**, so the shape is unconstructible rather than policed at run time; that removes the runtime check along with the mistake. On record (ADR 0045): `SemiDefinitePhylogenyEstimator`, `IntegerPhylogenyEstimator` and `CentralityEstimator` once took a `PhylogenyResult` or `Clusters` in `pl`, and now bound it to sources only.

**Choice Surface**
The set of things a caller picks when specifying a problem: the concrete leaf Estimators and Algorithms. Results are not part of it — they are what comes back, never what is chosen, so a caller writes `MeanRisk` but never writes `MeanRiskResult`. The distinction is implied by the three definitions above but worth stating, because it decides membership questions that would otherwise be settled case by case: which types a user-facing inventory must cover, and which are outputs that merely need to be reachable from one.

**Factory**
A generic configuration mechanism for immutable structs — they may be Estimator, Result, or Algorithm structs. Because all structs are immutable and the library relies heavily on composition, `factory` is the standard way to propagate runtime-computed values (moments from a Prior Result, observation weights, previous portfolio weights, etc.) down through a composed struct tree. It takes a struct and runtime data and returns a new, fully-configured struct of the same type.

**Deferred Quantity** (`DeferredQuantity` / `MuSlot` / `SigmaSlot` / `KtSlot` / `SkSlot`)
A moment Estimator standing where a prior-derived value goes. Four quantities defer — `mu`, `sigma`, `kt`, `sk` — so the slot admits the value itself or the Estimator that computes it, and **Factory** resolves the second form against the optimisation's own Prior. The struct that reaches the kernel therefore always holds a plain value. `MedianAbsoluteDeviation.mu` already carried this widening for its centring target; the rest generalise it. The field aliases are `MuSlot`, `SigmaSlot`, `KtSlot` and `SkSlot`; `DeferredQuantity` names the dynamic half alone, which is the four moment-estimator families plus the Prior Estimator, which computes every quantity at once. Two slots need their own bound rather than one of the four: `MedAbsDevMu` is `MuSlot` widened by the centring strategies, and `ArithRetMu` — `ArithmeticReturn`'s, the one widened slot that is not a Risk Measure's — is `MuSlot` narrowed by the `VecScalar` its consumer has no method for.
It exists because a stated value cannot follow a refit. A Cross-Validation fold and a Meta-optimiser's subproblem both refit the Prior, and a **View** is taken *before* Factory runs — so a stated matrix crosses that boundary as the whole universe's answer, while a Deferred Quantity crosses unresolved and computes on the subset. The choice is **which thing stays put**: a stated value holds the numbers still, a Deferred Quantity holds the method still.
A measure that defers **two or more independent quantities** names one Prior Estimator in `pe` instead, the **fan-out** form: one fit supplies every quantity the measure does not state. The count is of *deferrable* slots, not of prior-derived fields — a derived slot travels with its source rather than being fanned out separately, so `Variance` (`sigma` with `chol`) and `NegativeSkewness` (`sk` with `V`) take no `pe` and widen the source slot alone. It is also the only form that reaches the factor returns, because no moment estimator takes them and a Prior Estimator does. Precedence runs: a stated field wins, `pe` fills the rest, the Prior supplies what neither names. **Nothing refuses** — a caller may state some quantities and defer others, so keeping them consistent is the caller's job.
Resolution happens at **three** entry points, not one, and once at each: Factory, the `JuMP` risk-constraint builders (which never call Factory on a measure), and the value-level `expected_risk`. Which slots a type defers is a **per-type declaration** and never a walk over its fields, because an Estimator in a slot does not mean a deferred slot — `SimpleVariance` is a `DeferredQuantity` and stands legitimately in `Skewness.ve`. The fit reads `original_X` (§3.6) and the Prior's observation weights, neither of which a kernel can see, so a value-level call given a bare returns matrix **refuses** a Deferred Quantity instead of resolving it under a different rule.
*Avoid*: reading this `pe` as the optimiser's. A Risk Measure is an Estimator, so its `pe` admits an Estimator alone — narrower than every other `pe` in the library, which admits a precomputed Prior Result too. An optimiser and a measure inside it may name different priors, and each is used where it sits.
*Avoid*: deferring a quantity while stating the one derived from it. `chol` and `V` never defer — they arrive with the pair `sigma` and `sk` resolve to — so stating one without its source is refused at construction, rather than silently paired with a matrix the caller never saw.

**View**
The sub-selection counterpart to Factory. Where Factory propagates runtime *values* down a composed struct tree, a View propagates an index *selection*: it restricts an Estimator, Algorithm, or Result (or an array of them) to a subset of assets — or, for returns data, observations — and returns a new struct of the same type with every data-bearing field and composed child consistently sub-selected. Used wherever the library operates on part of the problem rather than the whole: meta-optimisers (Subset Resampling, Nested Clustered), Cross-Validation, and windowed moment estimators. Like Factory, it relies on composition — each struct declares which of its fields participate, and the selection is threaded recursively down the tree. Unlike Factory, View is primarily an *internal* mechanism — driven by the meta-optimisers and Cross-Validation rather than called directly by everyday callers — so its entry point `port_opt_view` is marked `public`, not exported.
Extension authors implementing a new composed estimator should define a `port_opt_view` method for it, or tag data-bearing fields with `@vprop` to have the method generated automatically.
`ReturnsResult` is Viewed through the same entry point: `port_opt_view(rd, i)` selects assets, while `port_opt_view(rd, i, j, k)` selects observations, assets, and factors — the one place in the family where the first index is not the asset index.

**Propagation Channel**
One generated method of `@propagatable`, and the field tag that opts a field into it. Three channels exist: `factory` (`@fprop` recurses into a sub-estimator, `@wprop` replaces an observation-weights slot), `view` (`@vprop`, the **View** counterpart), and `prior` (`@pprop` selects a moment off the Prior Result, `@cprop` selects a threaded context value such as a solver). A field may carry tags of more than one channel, and each channel reads only its own; where a field carries two tags of one channel, the channel's declared precedence decides which acts. The tag set, the field transform of each tag, and each channel's precedence are one table in `src/02_Tools.jl`, so a new channel is a row rather than an edit at seven sites (ADR 0061).

**Pipeline**
An Estimator that reifies an end-to-end workflow — price preprocessing, prices-to-returns, returns preprocessing, prior, phylogeny, uncertainty sets, constraint generation, optimisation — as an ordered list of optionally-named steps executed left-to-right over a Pipeline Context. Steps are ordinary Estimators; the step's family determines which context slot it reads and writes (custom steps use an explicit wrapper). Executed with `fit` (producing a result that carries every step's fitted Result) and evaluated out-of-sample with `predict`.
The Pipeline — data preparation included — is the unit that Cross-Validation splits (on contiguous input-time windows) and hyperparameter tuning searches; tuning lenses address steps by name or index and may swap entire estimators. Slots computed by pipeline steps override the optimisation step's corresponding internal configuration; slots with no step are computed internally by the optimiser as usual, so every stage is optional. Pipelines may nest as steps of other Pipelines; wrapping a Pipeline in a Meta-optimiser is deliberately unsupported (a Meta-optimiser may instead *be* the optimisation step).
*Avoid*: Workflow, Workbench (GUI-era synonyms); using "pipeline" for the library's informal stage ordering.

**Pipeline Context**
The accumulating blackboard threaded through a Pipeline's steps: a set of coarse typed slots — prices, returns, prior, phylogeny, uncertainty, constraints, weights — where each step reads the slots it needs and writes the slot its family produces. Heterogeneous slots (uncertainty, constraints) hold collections whose elements are routed to their optimiser targets by Result type. Internal machinery, not user-facing.

**Routing Target**
The destination a computed Pipeline Context slot is delivered to inside the optimisation step: `pe`, `cle`, `wb`, `lcse`, `cte`, `ple`, `lt`, `st`, `slt`, `sst`, `sglt`, `sgst`, `smtx`, `sgmtx`, `rkb`, `mu_ucs`, `sigma_ucs`. Targets are finer than slots and address a different audience — a *slot* is pipeline-author vocabulary naming a stage of the workflow, a *target* is optimiser-author vocabulary naming a destination. The Pipeline owns the fan-out (splitting the uncertainty pair, placing each constraint result); the optimiser owns where a target lands, so a field rename stays local instead of breaking the Pipeline at run time. Most targets are named after the field they land in, making the routing rule derived rather than declared — a target lands in the like-named field of whichever optimiser has one. `mu_ucs` and `sigma_ucs` carry policy and name no field; `rkb` names a field one level down, at `rba.rkb`. A target an optimiser cannot receive is ignored when losing it cannot change the answer (`pe`, `cle`), an error otherwise. Which target a constraint step writes is declared by its estimator family, and where a family names several — six fields can hold a Threshold — the step itself names one, so the choice is recorded before anything runs rather than guessed from the Result afterwards. Internal machinery.

**Data Slot**
The two Pipeline Context slots — prices and returns — whose write *changes the asset universe*, as opposed to a *derived slot* (prior, phylogeny, uncertainty, constraints, weights) computed from the data. Writing a data slot makes every slot derived from it stale: a prior or constraint fitted on one universe does not match a later, different one, so a Pipeline rejects at construction any ordering that would strand such a result. The distinction is the whole content of the invalidation rule — only data slots invalidate, and they invalidate every later slot except the terminal *weights* slot, which is the workflow's output and which nothing derives from. Because nothing derives from weights, an optimisation step must be the last step of a Pipeline.

**Preprocessing Estimator**
The family of Estimators that transform price or returns data inside a Pipeline (prices-to-returns conversion, missing-data filtering, imputation, asset selection). Fitting one on training data produces a Result carrying any fitted state — imputation parameters, thresholds, and crucially the selected asset universe — which is then *applied* to unseen data so train and test are transformed consistently. Stateless steps carry no state and applying them is just running them.

**Holdout Split**
The evaluation protocol that reserves the tail of the time-ordered observations as a test window and trains on the head. It exists in two forms: a free function (`train_test_split`) that cuts price- or returns-level data into a train/test pair, and a step form (`TrainTestSplit`) that must be the *first* step of a Pipeline, so every fitted step downstream sees the training window alone. The held-out window is fitted state — the split's Result carries both windows — and replaying a fitted split on unseen data is a pass-through, so predicting on future windows still works. One evaluation protocol per call: a Pipeline containing a holdout step cannot also be handed to Cross-Validation.
Sizes are given as row counts (integers) or fractions of the observations (floats in (0, 1)); giving one side makes the other its complement.
*Avoid*: Validation Split (nothing is tuned on the held-out window).

**Embargo**
The deliberate gap between the training and test windows of a Holdout Split when both sizes are given and sum to less than the number of observations: train comes from the head, test from the tail, and the middle rows belong to *neither* window. Overlapping windows are rejected outright.

**Asset Selector**
The Preprocessing Estimator subfamily that restricts the *asset universe* from returns data — dropping constant columns, keeping the best or worst assets by a risk measure, pruning redundant ones. A selector answers one question on the training window, *which asset columns survive?*, and that answer is its fitted state: applying it to an unseen window replays the fitted universe rather than re-deciding it. Selectors restrict columns only; dropping observations is a price-level concern (`MissingDataFilter`), because a fitted transformation cannot choose which rows of an unseen window to drop without breaking the weights/returns alignment. Every selector implements one method, `select_assets`, and shares a single `AssetSelectorResult`.
*Avoid*: Filter, Screen (used only by `MissingDataFilter`, which drops and never selects, and by the `ZeroVarianceFilter` alias, which is a pure drop).

**Selection Rule**
The Algorithm consumed by a `ScoreSelector` that turns per-asset scores into a keep-mask. *Literal* rules (`ThresholdRule`) compare raw scores against absolute bounds and ignore orientation. *Ordinal* rules (`RankRule`, `QuantileRule`) consult `bigger_is_better` — so "best" is lowest risk for a risk measure and highest value for a return measure — and take counts or fractions **from each end**, not positions, which is what makes "drop the worst 5" expressible without knowing the universe size.

**Redundancy Algorithm**
The Algorithm consumed by a `RedundancySelector`, deciding which assets duplicate information others already carry. It returns a *keep-mask*, not a partition: `PairwiseCorrelation` drops one asset at a time and may keep two members of the same correlated blob, which partition-then-keep-the-best cannot express. `CorrelationComponents` and `ClusterGroups` do partition, and share a `groups_argbest` helper. The selector's `score` chooses the survivor of each redundancy group; absent one, the correlation algorithms fall back on each asset's summary correlation to the rest of the universe.
Note that `PairwiseCorrelation` and `CorrelationComponents` give different answers on the same input: the former guarantees no surviving pair exceeds the threshold, the latter reads correlation transitively and keeps one representative per connected blob.
`ClusterGroups` is the only one that reaches a Distance Estimator — the other two carry a `StatsBase.CovarianceEstimator` — so it is the only selector that can be driven by a **Feature Matrix** (§2), which it reads straight off the data carrier. That widens the implicit contract an `AbstractReturnsResult` must satisfy to reach a selector, from `{nx, X}` to `{nx, X, Z}`.

**Trust-neither tie policy**
When two assets are indistinguishable under the criterion being applied, the library keeps *neither*. A tied block straddling a rank cut is excluded entirely (so `RankRule(; best = 20)` may return fewer than 20), and `find_uncorrelated_indices` removes both members of an exactly-tied correlated pair. The alternative — breaking ties by column index — would make the result depend on column order, which is not a property of the data.

**Vector-to-Scalar Reducers**
Small reusable Algorithms that collapse a vector of reals to a scalar, reused throughout the library: `MinValue`, `MaxValue`, `MeanValue`, `MedianValue`, `ModeValue`, `StdValue`, `VarValue`, `SumValue`, `ProdValue`, `StandardisedValue` (weighted mean ÷ weighted std). Most accept optional observation weights.

**LxNorm error family**
LxNorm errors are used as constraints, targets for risk measures, and in entropy pooling of multiple
conditional value at risk views: `L1Norm`, `L2Norm`, `SquaredL2Norm`, `LpNorm`, `LInfNorm`.

**FullMoment vs SemiMoment**
A pervasive Algorithm distinction in moment estimation: `FullMoment` includes all deviations; `SemiMoment` includes only observations below a target (downside-only). Drives the split between symmetric and downside risk/moment measures.

## 2. Data

**ReturnsResult**
The central data structure carrying all return series through the library. Fields: `X` (asset returns matrix, observations × assets), `F` (factor returns matrix), `B` (benchmark returns matrix), `nx`/`nf`/`nb` (asset/factor/benchmark names), `ts` (timestamps), `iv` (implied volatility), `ivpa` (implied volatility premium), `nz`/`Z` (Feature Matrix and its names). Produced by `prices_to_returns` from raw price data. Every Prior Estimator and Optimisation Estimator consumes a `ReturnsResult`.

**PricesResult**
The container of aligned, time-indexed *price-level* series — asset prices plus optional factor, benchmark, and implied-volatility series — the prices-level mirror of `ReturnsResult`. The input to price Preprocessing Estimators and to prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline Cross-Validation. Like `ReturnsResult`, user-constructible yet classified as a Result (produced by filtering steps); `FiniteAllocationInput` remains the glossary's only data-as-Estimator deviation.

**Feature Matrix** (`Z` / `nz`)
An **assets × features** matrix of per-asset quantities that are *not* return series — a sector or industry taxonomy, a fundamentals or ESG panel, fitted factor loadings, a graph neighbourhood. One row per asset, one column per measured quantity; `nz` names the columns. Time-varying features add a leading observation axis (`obs × assets × features`), the two shapes distinguished by `ndims` with no wrapper type. Its point is being **exogenous**: it lets the clustering and network stack see structure the returns do not encode, via a Feature Distance.
**It is data, not estimator configuration** (ADR 0045) — it is carried on `ReturnsResult`/`PricesResult` as user-supplied data and on `LowOrderPrior` as derived data, so `port_opt_view` slices it wherever it slices the returns; an estimator-held matrix would survive an asset subset still describing the full universe. A carried matrix is canonically assets-major and the constructor rejects any other orientation; only the raw-matrix `distance(de, Z; dims)` entry point lets `dims` retarget from `X` to `Z`.
Which carrier is read is chosen by `z_src::Symbol = :data` — the counterpart to `x_src` (ADR 0044), defaulting the other way because between two real sources, one hand-typed, explicit outranks derived. The two do different things under subsetting: `:data` **slices**, `:prior` **refits** on the subproblem's own returns.
A **pre-prior** site has no such choice: asset preselection through `ClusterGroups` runs before any prior exists, reads `rd.Z` off the data carrier, and carries **no** `z_src` field at all — the absence being the statement, since a knob with one legal position is not a knob. It says so with its own diagnostic, `:data_only`, because `:neither`'s remedy ("use a `FeaturePrior`") is actively wrong there (ADR 0045, amended).
When `nz == nx` — compared by *name*, not by axis length — the features **are** the assets, the matrix is square, and an asset view slices **both** axes. That is the one shape in which distance does not commute with an asset view, and it exists on the **data** carrier only: the prior carrier is nameless and states nothing about squareness, because a derived square matrix is *refit* on a subproblem's universe rather than sliced down to it (ADR 0045, amended).
Across a meta-optimiser's synthetic-asset boundary the matrix is neither sliced nor refit but **collapsed**: `collapse_feature_matrix` aggregates the real assets' features onto the clusters or sub-portfolios the outer problem allocates over. Features are **intensive**, like `iv` and `ivpa`, so the collapse is a convex combination through `synthetic_asset_weights`; an *extensive* feature wanting a weighted sum is unsupported, because the divisor depends on the inner solve. Cross-validation collapses per fold, which is why a static matrix arrives at the outer problem time-varying (ADR 0045, amended).
Under cross-validation the collapse happens **not in the fold but at the assembly seam**: `rebuild_returns_result` recomputes it from the original, unsliced `rd.Z` and the fold's weights, making the exact call `prepare_outer_rd` makes. The criterion is **path-consistency** — `cv` is execution control (ADR 0030), so toggling it must not change what the outer problem is measured on — and it is what lets the square shape survive a fold at all, since the second contraction needs every synthetic asset's weights at once and a fold holds one. The fold's rows are recovered from its timestamps rather than stored, which is why `ts` **keys** the observation axis and must be unique (ADR 0045, amended).
*Avoid*: confusing a Feature with a **Characteristic Vector** (§3.9) — near-synonyms held deliberately apart, but **not by count**: an optimisation may hold several Characteristic Vectors, one per set-bearing return term. A Characteristic Vector is a per-asset quantity an ℓ1 Uncertainty Set is built around, entering the objective as `mu'w`; a Feature is *a coordinate of an asset* in a matrix whose rows are turned into a distance. They also sit on opposite sides of ADR 0045: a Feature Matrix is data and is sliced by a view, a Characteristic Vector rides on an estimator and is resolved per term at solve time. Also avoid reading "feature" as *factor*: factor returns are `F`/`nf`, and factor loadings become features only by passing through the `RegressionFeatures` producer.

**Feature Matrix Estimator** (*producer*)
Turns something the library already computes, or a classification the caller supplies, into a Feature Matrix, running inside the wrapping `FeaturePrior`, which attaches the result to the prior it wraps. `RegressionFeatures` (factor loadings — reads `rr.L`, the reduced-dimension coordinate system, not `rr.M`), `AssetSetsFeatures` (exogenous taxonomy memberships, rectangular), `PhylogenyFeatures` (a square neighbourhood matrix under `Proximity`'s decay — the only producer whose feature axis *is* the asset axis). A literal matrix passed as `FeaturePrior(; ze = Z)` is the identity producer.
*Avoid*: a producer that clusters the returns and re-encodes the clustering. Cluster memberships from nested cuts recode the cophenetic distance of the hierarchy that produced them, so clustering the result returns that hierarchy; a single partition is worse still, giving a distance matrix with at most two distinct values (ADR 0045).

**Feature Program** (`AssetSetsFeatures.vals`, graded form)
An ordered, **last-wins** list of authored edges resolving into a Feature Matrix over the axis a Universe Sets declares at `zkey` — `AssetSetsFeatures`' second contract, dispatched on `vals`' element type (ADR 0045, amended). Where the group-name-key form *stacks partitions* and derives its axis, a Program *writes cells* into an axis the caller declared. Its productions: a **diagonal** entry (`"nx_sector" => 2.0`) writes each selected asset's own membership; a **row-scope** entry (`"AC" => [targets]`) opens a scope of assets and then names its columns explicitly. Row selectors are assets, groups, or a taxonomy key optionally restricted to one of its values; targets are always **fully qualified**, because there is no ambient scope — `"UK"` is both a country and a ticker, and which one a caller means must be written, not inferred.
A bare `Number` **sets** a cell absolutely; a `Scale` **scales** the cell's **natural value** — the key's own datum for a numeric taxonomy key, `1.0`-if-member otherwise. That referent is forced rather than chosen: at the top level there is no accumulated value to scale, so every write stays a pure overwrite and the ordering rule survives the marker. It follows that scaling a *cross* edge gives zero. The marker family (`AbstractFeatureValue`) is open.
Node names are **bare** (`"Tech"`, `"esg"`, `"A"`), the opposite of the key form's qualified `"<key>=<group>"`. The accepted cost: a *nested* taxonomy with a repeated value is inexpressible in a Program, since both levels land on the one node — the key form stays the tool for that case.
*Avoid*: expecting the key form's equal-row-norm identity (`cos = shared / L`) to hold on a Program. It holds only where every key is a partition, and no normalisation could restore it — the identity needs `0`/`1` entries, not merely equal norms. Also avoid reading `strict` as a structural check: it governs **names** only. An all-zero row and a one-column matrix are both legal, and a malformed entry throws regardless of it.

**Implied Volatility**
A forward-looking estimate of how much an asset's price is expected to fluctuate, derived from current options contract prices using models such as Black-Scholes. Stored as `iv`. Not a historical measurement.

**Implied Volatility Premium (VRP)**
The difference between Implied Volatility (market expectation) and realised volatility (historical outcome). Typically positive because investors pay a premium to hedge downside risk. Stored as `ivpa`.

## 3. Statistics

**Windowed Estimator**
A moment estimator that restricts an inner moment estimator to a sub-window of the observations — the most recent *n*, or an explicit set of indices — and rebinds observation weights to that window before delegating. The inner estimator's semantics are untouched: windowing decides *which observations are seen*, never how the moment is computed from them, so any moment kind can be windowed and a windowed estimator is substitutable wherever its inner one is. The family spans the moment kinds (expected returns, covariance, variance, coskewness, cokurtosis), one type each, because each answers a different generic. Distinct from a Cross-Validation fold, which restricts observations to evaluate a strategy; a Windowed Estimator restricts them because older observations are held to be less informative about the present.

### 3.1 Expected Returns (Moments)

**Expected Returns Estimator**
Computes a per-asset mean-return vector. Variants:

- **SimpleExpectedReturns**: sample mean, with optional observation weights.
- **ShrunkExpectedReturns**: shrinks the sample mean toward a target. Targets: `GrandMean`, `VolatilityWeighted`. Algorithms: `JamesStein`, `BayesStein`, `BodnarOkhrinParolya`.
- **EquilibriumExpectedReturns**: implied (equilibrium) returns via reverse optimisation, Π = λ·Σ·w_eq — parameterised by a covariance estimator, equilibrium weights `w`, and risk-aversion `l`. Serves as the Black-Litterman equilibrium anchor (classified as a shrinkage estimator).
- **ExcessExpectedReturns**: returns net of a reference/risk-free rate.
- **MedianExpectedReturns**: (weighted) per-asset median.
- **StandardDeviationExpectedReturns** / **VarianceExpectedReturns**: return the asset standard deviations / variances (used where a "return" slot should carry dispersion).
- **CustomValueExpectedReturns**: user-supplied per-asset values.
- **WindowedExpectedReturns**: the Windowed Estimator for expected returns.

### 3.2 Covariance & Variance (Moments)

**Covariance Estimator**
Computes an asset covariance (and correlation) matrix. Core wrappers: `Covariance` (flexible container), `GeneralCovariance` (wraps any `StatsBase.CovarianceEstimator` + weights), `SimpleVariance`, `CorrelationCovariance` (returns correlation as both cov and corr). Robust / specialised families:

- **Gerber** (`Gerber0/1/2`): co-movement counting that ignores small noise-level moves below a threshold.
- **GerberIQ** (Gerber Information Quality): Gerber with temporal lookback/decay and asset-volatility-scaled thresholds (`BasicGerberIQ`, `PartialGerberIQ`, `FullGerberIQ`; tuners `ExpGerberIQDecay`, `AssetVolatilityGerberIQScaler`).
- **Smyth-Broby** (`SmythBroby0/1/2`, `…Gerber*`, `…Count*`): Gerber-family extensions weighting co-movements by magnitude and/or vote counts.
- **DistanceCovariance**: covariance derived from a distance metric between assets.
- **LowerTailDependenceCovariance**: dependence in the joint lower tail (crash co-movement).
- **RankCovariance**: rank-correlation based — `KendallCovariance` (τ), `SpearmanCovariance` (ρ).
- **MutualInfoCovariance**: dependence via mutual information; uses a histogram binning Algorithm (`Knuth`, `FreedmanDiaconis`, `Scott`, `HacineGharbiRavier`).
- **RegimeAdjustedExpWeightedVariance/Covariance**: online exponentially-weighted estimators that rescale by a detected market regime state (methods `LogRegimeAdjusted`, `FirstMomentRegimeAdjusted`, `RootMeanSquaredAdjusted`; covariance targets `MahalanobisTarget`, `DiagonalTarget`, `PortfolioTarget`).
- **ImpliedVolatility** (covariance): scales covariance using implied volatility, optionally regressing realised on implied (`ImpliedVolatilityRegression`) or applying a premium factor (`ImpliedVolatilityPremium`).
- **DenoiseCovariance / DetoneCovariance / ProcessedCovariance**: wrap another covariance estimator and apply denoising / detoning / custom matrix processing plus posdef projection.
- **PortfolioOptimisersCovariance**: composite estimator bundling covariance estimation with post-processing.
- **Windowed** variants (`WindowedCovariance`, `WindowedVariance`): the Windowed Estimators for covariance and variance.

### 3.3 Higher-Order Moments

**Coskewness** / **Cokurtosis**
Third- and fourth-order co-moment tensors (with `FullMoment`/`SemiMoment` variants and the Windowed Estimators `WindowedCoskewness`, `WindowedCokurtosis`). Feed high-order priors and higher-moment risk measures.

### 3.4 Regression (factor modelling)

**Regression Estimator**
Builds a factor model mapping factor returns to asset returns; underpins factor priors. Families:

- **StepwiseRegression**: greedy feature selection — `ForwardSelection` (selection) or `BackwardElimination` (elimination), driven by a criterion: `PValue`, `AIC`, `AICC`, `BIC`, `RSquared`, `AdjustedRSquared`.
- **DimensionReductionRegression**: regression on reduced factors — targets `PCA`, `PPCA`.
- **Regression target** models: `LinearModel`, `GeneralisedLinearModel` (GLM).

### 3.5 Matrix Processing

**Denoising**
Removes statistical noise from a covariance/correlation matrix via spectral thresholding (Random Matrix Theory). Algorithms: `SpectralDenoise` (zero smallest eigenvalues), `FixedDenoise` (replace with constant), `ShrunkDenoise` (shrink). Requires the effective sample ratio `q = observations / assets`.

**Detoning**
Removes the largest `n` principal components (market modes) from a correlation matrix, isolating asset-specific correlation.

**Posdef**
Projects a matrix to the nearest positive definite matrix; used when a matrix is ill-conditioned or non-PD after denoising/detoning.

**Matrix Processing**
A composing estimator (`MatrixProcessing`) that applies a sequence of post-processing steps — Posdef projection, Denoising, Detoning, and a custom algorithm — to a covariance/correlation matrix. Each step has its own estimator field (`pdm`/`dn`/`dt`/`alg`); the `order` field is a tuple (or vector) of step symbols (`:pdm`, `:dn`, `:dt`, `:alg`) naming which steps run and in what sequence, applied left to right.

### 3.6 Prior

**Prior**
The full statistical summary of asset returns that feeds an optimiser. A Prior Estimator bundles Moment Estimators and may adjust them using user assumptions. "Prior" derives from "prior statistics", not Bayesian inference. Structural axes:

- **Low-order vs High-order**: low-order = mean + covariance (`LowOrderPrior`); high-order = also coskewness/cokurtosis (`HighOrderPrior`).
- **Data dependency** (`_A` / `_F` / `_AF`): asset-only, factor-required, or asset-and-factor. Determines whether `F` must be present.

Concrete estimators:

- **EmpiricalPrior**: moments computed directly from returns.
- **FactorPrior** / **HighOrderFactorPriorEstimator**: moments reconstructed through a factor model (regression).
- **HighOrderPriorEstimator**: empirical high-order prior.
- **Black-Litterman family**: blends market-equilibrium priors with investor views — `BlackLittermanViews` (views container), `BlackLittermanPrior`, `BayesianBlackLittermanPrior`, `FactorBlackLittermanPrior`, `AugmentedBlackLittermanPrior`.
- **EntropyPoolingPrior**: re-weights scenario probabilities to satisfy views with minimal relative entropy. Stepwise schemes `H0/H1/H2_EntropyPooling` (bias vs speed trade-off); optimisers `LogEntropyPooling`, `ExpEntropyPooling`, `CVaREntropyPooling`, backed by `OptimEntropyPooling` or `JuMPEntropyPooling`.
- **OpinionPoolingPrior**: consensus across multiple priors — `LinearOpinionPooling`, `LogarithmicOpinionPooling`.

**Risk-Free Shift**
The `rf` field of the four Black-Litterman prior estimators. A Black-Litterman update is written in **excess** returns, so `rf` names one round trip: off before the update, back on after it. `remove_rf` is the only site that takes it off and `apply_rf` the only site that puts it back, and three rules follow (ADR 0063). The rate is added **exactly once**, to the asset expected returns the estimator returns. A prior is **isolated**: the round trip returns the wrapped prior's mean to the scale it arrived on, so a rate an inner estimator applied internally is never undone, and the rate that comes back on is the one that came off. The rate belongs to the **asset axis**: the factor block of a result never carries it, so where the rate came off a factor mean nothing puts it back, and `FactorBlackLittermanPrior` shifts *after* the Factor Lift, because shifting before it would project the rate through the loadings as `M * (rf * 1)` rather than add it.

That last rule has a consequence worth naming, because it is the one place the round trip is not exact. `FactorBlackLittermanPrior` takes the rate off the *factor* axis and adds it on the *asset* axis, so the two moves are not inverses: writing `s` for the row sums of the loadings, its answer moves by `rf * (1 - s)`, and cancels only for an asset whose loadings sum to one. `AugmentedBlackLittermanPrior` has no such crossing — its augmented prior spans both axes, and the rate comes off and goes back on the same asset rows.

Only `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` take the rate off, and only where `l` is `nothing`. They are the two members that can build an Equilibrium Mean instead, and `equilibrium_mu` returns a risk premium, which is an excess return already. `BlackLittermanPrior` and `BayesianBlackLittermanPrior` have no equilibrium branch, so they take the wrapped mean on the scale it is given.

**Factor Lift**
The hop from the factor axis to the asset axis: fit the loadings, rebuild `X` as `F * transpose(M) .+ transpose(b)`, project the factor moments through `M`, and — when `rsd` is set — add the diagonal residual block. It is one algorithm with one home, split into `factor_reconstruction` (the loadings and the reconstruction) and `factor_lift` (the projection and the residual block), because `FactorBlackLittermanPrior` needs the reconstruction *before* it has the moments to project: its views land on the factor distribution.

`FactorPrior` and `FactorBlackLittermanPrior` call both; the only thing that differs between them is which factor moments they hand to the second. `AugmentedBlackLittermanPrior` calls the first alone — its asset moments come out of the augmented system rather than out of a projection, which is the same reason it merges two priors rather than forwarding one (ADR 0046).

A consumer that needs to *undo* the residual block reads `factor_residual_config(pe)` rather than the estimator's fields. Only the two lifting estimators carry `ve` and `mp.pdm`, and the slot that holds them is bounded by a union that admits far more, so a field access reaches past the type bound. The declaration is per-type and has no default: the two owners report `(; ve, pdm, rsd)`, a wrapper forwards its wrapped estimator's answer, an estimator that adds no residual block declares `nothing` in an explicit method, and a type that declares nothing throws. An estimator forwards whichever estimator its moments come from, however many priors it also consults for something else, so a pooling estimator is a wrapper here too. A silent `nothing` fallback would read the same as a correct answer, so a forgotten method would drop a residual block the covariance really carries.

*Avoid*: reading the Factor Lift as a **Forward** (ADR 0046). A forward carries a wrapped Result along *its own* axis and keeps every field it does not name; the lift crosses axes, so almost every field changes meaning and nothing forwards by default. The factor block a lifting estimator builds *is* an ordinary forward, and goes through `forward_prior`.

**Original Returns Matrix**
The returns matrix the caller supplied, as distinct from the one a Prior Result asserts. They differ on exactly three routes: `FactorPrior`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` each overwrite `X` with the reconstruction `F * transpose(M) .+ transpose(b)`, so their `X` is a *posterior* matrix. Everywhere else `X` already is the caller's returns.

Stored as `LowOrderPrior.o_X`: the original when `X` is not it, and `nothing` otherwise. `HighOrderPrior` reads it through its `forward(pr)` block and declares nothing of its own. `EntropyPoolingPrior` and `OpinionPoolingPrior` forward it, so it survives a pooling wrapper over a factor prior.

*Read it as `pr.original_X`*, a computed property that is always a matrix — `o_X` when there is one, `X` when there is not. The field is storage and answers a different question: `isnothing(pr.o_X)` is how to ask whether this carrier reconstructed `X`. Two names rather than one always-populated field, because `forward_prior` rebuilds through the keyword constructor with every field named, where a `nothing` is inert and a matrix would go stale.

The two are not interchangeable. The reconstruction spans only the factors, so it has rank `size(F, 2)` and carries no residual. A **Deferred Quantity** (§1) refits a moment on the sample and therefore reads `original_X`: fitting a covariance estimator on the reconstruction returns a singular matrix whenever there are more assets than factors.

Two readers, and the split is by what the consumer does with the matrix. A consumer that **evaluates** the return distribution reads `X`, because the reconstruction *is* the distribution a factor prior asserts. A consumer that **decomposes** risk into a factor part and a residual part reads `original_X`, because a matrix with no residual can only attribute noise to the second: `factor_risk_contribution` handed a factor prior reported a *negative* idiosyncratic share. So the factor attribution reads `original_X`, and the price is that under a factor prior its parts sum to the risk on the caller's returns rather than to `expected_risk(r, w, pr)`.

Loadings follow the same rule, and for the same reason. The attribution prefers `pr.rr` over a refit, so the loadings are the ones fitted on `original_X` and the pair is matched by construction. A precomputed `Regression` still outranks both; a regression *estimator* does not.

`o_X` requires `rr`, so `forward_prior` binds the two: naming `rr` obliges the caller to name `o_X`, and dropping the factor block drops the original with it. Every estimator that overwrites `X` today projects a factor prior through regression loadings, so the loadings are always in hand. This is a present-tense constraint rather than a law of the domain, and ADR 0046's amendment records what would relax it.

### 3.7 Distance

**Distance Matrix**
A symmetric, zero-diagonal matrix where larger values mean less relatedness. Usually derived from correlation, but any matrix satisfying that convention qualifies. The fundamental input to Phylogeny.

**Distance Estimator**
Converts correlation/returns into a distance. Algorithms: `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `VariationInfoDistance` (variation of information), `CanonicalDistance`. `Distance` is the configurable container (supports a generalised power distance); `DistanceDistance` computes a distance-of-distances.

**Feature Distance** (`FeatureDistance`)
The one Distance Estimator that measures something other than returns: it applies a metric to the **rows** of a Feature Matrix (§2), so the resulting hierarchy expresses exogenous structure. Configuration only — `metric`, `alg`, `sim`; the matrix itself is data and arrives from its carrier (ADR 0045). The metric family is **open**: any `Distances.SemiMetric`, including user-defined ones, defaulting to `AngularDist` (`acos(clamp(1 - CosineDist, -1, 1))/π`, a true metric where `CosineDist` is not). `alg` is an `AbstractFeatureCollapseAlgorithm` deciding how a *window* of time-varying features becomes one distance matrix — `LastObservation` (default), `AggregateFeatures`, `AggregateDistances` (both taking `MeanCollapse`/`MedianCollapse`), `StackObservations` — and is inert, not an error, for a static matrix.
*Avoid*: assuming metric choice is free of the data. Every metric other than `AngularDist` and `CorrDist` is scale-sensitive, so heterogeneous features need standardising; the Ruzicka `Jaccard`, `BrayCurtis` and `ChiSqDist` require a non-negative matrix, checked in the kernel rather than at construction because the metric is unknown when the carrier is built.

**Similarity Matrix Algorithm** (`AbstractSimilarityMatrixAlgorithm`)
Fills `cor_and_dist`'s `S` slot by transforming a Distance Matrix — `distance_to_similarity(sim; D)`. Members: `MaximumDistanceSimilarity`, `ExponentialSimilarity`, `GeneralExponentialSimilarity`, `ComplementSimilarity` (`1 .- D`), `AngularSimilarity` (`cos.(π .* D)`). Lives in the Distance layer, not in DBHT, because a Feature Distance must name it. `FeatureDistance.sim` is non-optional and defaulted from the metric by `default_similarity`, so the resolved choice prints on the object. `DBHT` has a `sim` field of the same type doing the same job; when both are set DBHT's wins, because `clusterise` overwrites `S` immediately after `cor_and_dist` returns.

**Non-Negative Similarity Matrix Algorithm** (`AbstractNonNegativeSimilarityMatrixAlgorithm`)
The narrower family admitted on the PMFG path. `Tree_SimMat`, `DBHT.sim` and `LoGo.sim` are bounded by it, so a similarity that can go negative fails at **construction** rather than inside `PMFG_T2s`. Membership is a **claim a subtype makes**, not one the library verifies: the contract quantifies over every admissible distance matrix, so no probe can decide it, and `PMFG_T2s`'s own non-negativity check stays as the backstop against a lying extension. The requirement is **DBHT's, not the PMFG's** — `DirectHb` sums unnormalised weights over different-sized bubbles and detects a separator by an exact zero, and `BubbleMember` divides by a signed total, while `PMFG_T2s` itself runs on signed input — and the bound is deliberately wider than that provenance, covering the three callers that reach only structure (ADR 0049).
The criterion is **not** algebraic form. Every similarity algorithm is a strictly decreasing function of `D`, so "monotone transform of a distance versus honest inverse of a specific metric" does not predict membership. It is whether the range can go **below zero over the member's own admissible distances**, which is a statement about a declared domain: `ExponentialSimilarity` and `GeneralExponentialSimilarity` always hold, `MaximumDistanceSimilarity` holds on a finite `D` (`ceil(Inf^2) - Inf^2` is `NaN`, and it is the default of both `DBHT` and `LoGo`), `ComplementSimilarity` holds on `D <= 1`. `AngularSimilarity` is excluded **permanently**, and not for want of a precondition: it is negative wherever `ρ < 0`, which is ordinary data. `assert_similarity_domain(sim, de, D)` enforces the two domains at the five PMFG entry points, **interface-scoped rather than member-wide**, because the same negative value is documented, tested behaviour on the `FeatureDistance` path.
*Avoid*: reading the domain rule as "an unbounded metric is refused". It is `D <= 1`. `CosineDist` and `CorrDist` are bounded — by 2, not by 1 — and are refused whenever they exceed 1, which `CorrDist` does at every negative correlation; `DistanceDistance`'s `Euclidean` default puts most of its entries above 1.
*Avoid*: reading a non-negative similarity as a correct one. `ComplementSimilarity` is the honest inverse of `CosineDist`, `Jaccard`, `BrayCurtis` and `CorrDist` and of nothing else. Paired with `SimpleDistance` it satisfies `D <= 1`, stays non-negative, and reports `0.29` where the correlation is `0.003`. Nothing catches that on any path, because `0.706` is a perfectly legal bounded distance.
*Avoid*: reading `ComplementSimilarity` as bounded on the Feature Distance path. There it is unbounded below whenever `D` is (a `Euclidean` distance of 7 gives `S = -6`), and `plot_clusters` silently clips rather than rejecting.

### 3.8 Phylogeny

**Phylogeny**
The characterisation of asset relationships derived from a Distance Matrix. The biological metaphor is intentional: related assets cluster like species on an evolutionary tree. Three sub-concepts:

- **Clustering**: groups related assets into a dendrogram. `ClustersEstimator` with `HClustAlgorithm` (hierarchical linkage), `DBHT` (Direct Bubble Hierarchical Tree — PMFG-based, root methods `UniqueRoot`/`EqualRoot`, similarity transforms, `LoGo` sparse inverse), or `KMeansAlgorithm` (non-hierarchical). Optimal cluster count via `OptimalNumberClusters` using `SecondOrderDifference` or `SilhouetteScore`.
- **Network**: assets as graph nodes/edges. `NetworkEstimator` builds a minimum spanning tree — `KruskalTree`, `BoruvkaTree`, `PrimTree`.
  Its `sep` is a **Separation** (below): the rule saying how far apart two assets sit in the resulting graph, and how far is too far.
  The structure is built once, as a `SimpleWeightedGraph`, by `calc_weighted_adjacency_graph`; `calc_weighted_adjacency` is `adjacency_matrix` of it and `calc_adjacency` is the round trip through `SimpleGraph` that binarises it. Each branch carries **the quantity that selected its own edges** — distances on a tree, since `calc_mst` minimises them; similarities on a PMFG, since `PMFG_T2s` maximises gain over them — so the two run in opposite directions and the polarity is recovered by dispatch on `alg` rather than by a tag. `calc_distance_weighted_graph` is the fourth name: the same structure with distances on *both* branches, which is what a shortest path needs. The first three take two entry points, the two-argument one taking the selecting quantity itself, because `clusterise` already holds it and re-deriving it there costs 98% of that function's runtime.
  *Avoid*: reading a weighted adjacency as authoritative weights. Both consumers that walk the structure — `PathLength` and weighted centrality — take the *edge set* and supply their own weights, so the chain is a graph constructor rather than a supplier of the one true weighting.
  A distance matrix and a weighted graph disagree about `0`: in the distance codomain it is the *floor* — as close as two assets get — while `SimpleWeightedGraph` reserves it to mean *absent* and refuses a zero-weight edge. So `graph_weight_matrix` moves each off-diagonal zero to `nextfloat(zero(·))` before the graph is built, the nearest value the representation can carry.
  It is a **repair, not a rejection**, because a zero is not a symptom of bad data: `SimpleAbsoluteDistance` and `LogDistance` are defined on `abs(rho)`, so an exactly anti-correlated pair — a long/short leg, an inverse ETF — sits at distance zero and is genuinely maximally related, and the square-root algorithms reach zero from the other side when their `clamp!` maps `rho >= 1` down. Left unrepaired, the constructor deletes precisely the edge the tree most wants and the pair comes out non-adjacent, which is why `clusterise` (reading `D` directly) and `phylogeny_matrix` (reading only the graph) could disagree about the same pair. Negative and `NaN` entries are rejected instead — they have no nearest representable value, and both are *unsound* rather than merely wrong downstream. `Inf` is left alone: it is the honest `LogDistance` between uncorrelated assets, and a spanning tree simply takes those edges last.
- **Centrality**: which assets are most influential — `BetweennessCentrality`, `ClosenessCentrality`, `DegreeCentrality`, `EigenvectorCentrality`, `KatzCentrality`, `Pagerank`, `RadialityCentrality`, `StressCentrality`.

`PhylogenyResult` carries the resulting matrix/vector. Required input to clustering-based Optimisation Estimators and to phylogeny constraints.
*Avoid*: expecting a precomputed `PhylogenyResult` to follow a subproblem. A phylogeny is not **separable** over the asset universe — every entry states something about the whole graph, so no slice of it is the phylogeny of the slice, and the subgraph of a spanning tree is routinely disconnected while still being symmetric with a zero diagonal, so a slice re-validates and travels on silently. Nothing can reach that state now: no estimator holds one (see the *Avoid* below), so a `PhylogenyResult` only ever exists where the caller put it, and the subsetting machinery never sees it (ADR 0045).
*Avoid*: putting one in an estimator's `pl` slot — it is no longer possible. `SemiDefinitePhylogenyEstimator`, `IntegerPhylogenyEstimator` and `CentralityEstimator` all bound `pl` by `NwE_ClE` (a *source*: `NetworkEstimator` or `ClustersEstimator`), because an estimator holding a precomputed result is configuration on the outside and data on the inside, which is how it slipped past every guard written for results. `Clusters` is excluded for the same reason as `PhylogenyResult`. Precomputed structure goes in the constraint **result** — `SemiDefinitePhylogeny`/`IntegerPhylogeny`, whose `A` takes a `PhylogenyResult` or a matrix — which is exactly what `phylogeny_constraints(estimator, X)` returns. Removing the shape removed the runtime checks with it: the only one left is `assert_external_optimiser` rejecting a precomputed constraint result in `ple`, which cannot be typed away because `ple` legitimately accepts a result outside a meta-optimiser.

**Phylogeny Features** (`PhylogenyFeatures` / `phylogeny_features`)
The reverse direction: a phylogeny reused as a **Feature Matrix** (§2) rather than consumed as one. An `assets × assets` neighbourhood matrix *is* an `assets × features` matrix whose features are "how close is asset *k*", so the structure a Network Estimator or a Clusters Estimator builds can drive a Feature Distance. `pl` is bound by `NwE_ClE` — both source kinds, both **estimators**, so both refit per fold and per subproblem.
`AbstractPhylogenyFeatureAlgorithm` says how separations become scores on a *graph* source: `Proximity`, which carries a **Separation Decay** (below); on a *partition* source there is no separation structure and `alg` is inert. It has exactly **one** member, and that is an open extension point rather than a taxonomy — the two knobs that vary in practice are the **Separation** (on the source `NetworkEstimator`, which pairs are related) and the Separation Decay (on `Proximity`, how strongly), so a second member would have to be a rule that is *not* a decayed separation at all. **Self is always included**, so the diagonal is just the zero-hop value: excluding it measures *structural equivalence* rather than proximity — the two non-adjacent endpoints of a three-node path come out identical — and, under an asset view that isolates every selected vertex, leaves zero rows that the zero-vector convention declares mutually identical.
*Avoid*: reading the square shape as cosmetic. A subproblem is measured against its own neighbourhood structure rather than the universe's; because both sources refit, that structure is always the subproblem's own — which is why the carrier needs no squareness flag to say so. This producer is **endogenous** — both sources derive from the returns — and what a graph source buys was measured: a different merge order and different weights from a correlation distance, but identical cuts at `k = 2` and `k = 3`.
*Avoid*: reaching for a partition source by default. A `ClustersEstimator` is admitted but carries much less than a graph: its matrix is co-membership, so the distance depends only on cluster membership and size — on a seven-asset universe clustered `[1, 1, 1, 2, 2, 3, 3]` the whole distance matrix takes three distinct values. Clustering a re-encoded clustering largely returns the clustering.

**Separation** (`AbstractSeparationAlgorithm` / `separation_graph` / `separation_matrix` / `separation_budget` / `resolve_separation`)
How far apart two assets sit in a Network, **and** how far is too far. An open family carried by `NetworkEstimator.sep`, with two members: `HopCount(; n = 1)`, the number of graph edges on the shortest path, budgeted in edges; and `PathLength(; dmax = nothing)`, the summed distance along that path, budgeted in the Distance Estimator's units. Four kernels: `separation_graph(sep, nte, X)` builds the structure the member measures over, `separation_matrix(sep, g)` reads the dense `assets × assets` separations off it, `separation_budget(sep, nte, d)` resolves the budget, and `resolve_separation(sep, nte, X, g)` replaces a **Separation Budget Rule** (below) by the value it computes.

The measuring kernels take **the structure, not the producer**: `separation_matrix(sep, g)` and `resolve_separation(sep, nte, X, g)` are the interfaces and their estimator-taking forms are wrappers, exactly as `calc_weighted_adjacency_graph` has two entry points and for the same reason. A consumer builds one graph per call and hands it to both, because a consumer that resolves a rule *and* measures the separations through two estimator-taking kernels derives the structure twice — `98%` of `clusterise`'s runtime under `VariationInfoDistance`. `separation_graph` gives `HopCount` a **binarised** graph, since a power of a weighted adjacency matrix sums products of distances rather than counting walks, and gives `PathLength` `calc_distance_weighted_graph`'s; only `HopCount` has a graph-taking entry point, because a graph carries no polarity tag and a path over the PMFG's similarities returns an answer instead of raising. The graph entry is also the only way to reach a **disconnected** structure,
every structure a shipped estimator builds being connected.
The two questions are one type because they share a **unit**: a budget stated apart from the rule that measures it has no interpretation, which is why `n` moved off `NetworkEstimator` onto `HopCount` and why a member measuring something continuous carries a budget in its own units instead. The family sits on the Network Estimator rather than on any consumer because *every* consumer of a graph needs to know which pairs it relates — `phylogeny_matrix` and the phylogeny constraints, both `clusterise` methods, `Proximity` — and the constraint path receives nothing but the estimator, so a rule living on the feature producer would be structurally invisible to it.
`HopCount.n` stays an `Integer` rather than widening to a `Real`: three readers use `0:(nte.sep.n)` as a **matrix-power count**, where `0:1.5` silently drops a power instead of failing — which is why those three readers admit `HopCount` alone and refuse a `PathLength` loudly. The family name is deliberately unqualified — a taxonomy depth is a separation too — so a non-graph member would arrive without a re-parenting.
`separation_budget`'s third argument is the separation **matrix** rather than a scalar diameter, because finding the largest finite entry *is* an `assets²` reduction and `HopCount` must not pay for one it ignores. `PathLength` does the reduction itself: `dmax = nothing` means the observed diameter, and a chosen `dmax` is **clamped** to it. The clamp truncates nothing — no pair sits beyond the diameter — so it is a scale-top correction visible only through `LinearDecay`, the one decay that reads the budget.
*Avoid*: reading an unreachable pair's entry as a number. `separation_matrix` passes the underlying routine's sentinel through unrepaired (`typemax(Int)` for `HopCount`, `Inf` for `PathLength` over `Float64` weights), so a consumer asks `is_related(sep, d, dmax)` — reachable, and no further than the budget — instead of writing the comparison out, and keeps the *evaluation* of anything it scores the entry with inside a short-circuiting branch. `is_reachable(sep, d)` is the sentinel half on its own, for a population that has no budget: the diameter reduction and `separation_quantile`. Neither is `isfinite`, which is `true` of every `Integer`. Both sentinels punish an unguarded read, for opposite reasons: `ReciprocalDecay` overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError`, while `LinearDecay` at `Inf` returns `-Inf`, which sorts every unreachable pair below every reachable one without raising anything.
*Avoid*: taking a shortest path over a PMFG's own weights. Those weights are **similarities**, so the path minimises total similarity and prefers the route through the weakest links. `PathLength` therefore reads `calc_distance_weighted_graph`, which re-weights the PMFG's structure with `D` — legitimate because every similarity algorithm is a strictly decreasing function of `D`, so `D` is the selecting quantity's monotone preimage rather than a foreign quantity. The backwards answer correlates `0.95` to `0.97` with the right one, so nothing about it looks wrong.

**Separation Budget Rule** (`HopCountAlgorithm` / `PathLengthAlgorithm` / `resolve_separation`)
A callable standing where a **Separation**'s budget number goes, called as `rule(nte, X, g; dims, kwargs...)` when the data is in hand — `g` being the structure the consumer already built, so a rule pays for a traversal and never for a second graph; `nte` and `X` are inert for what ships and are the channel to what a graph does not carry. Two families, one per unit, because their return obligations differ: a `HopCountAlgorithm` in `HopCount.n` must answer with an `Integer`, a `PathLengthAlgorithm` in `PathLength.dmax` with a `Number`. A bare `Function` is admitted in either field under the same obligation. One member each: `HopCountQuantile(; q = 0.25)` and `PathLengthQuantile(; q = 0.25)`, placing the budget at the `q`-quantile of the observed separations. The field aliases are `HopCountValue` and `Option{PathLengthValue}`; the dispatch aliases naming the dynamic halves are `HopCountRule` and `PathLengthRule`.
It exists because a budget cannot always be named in advance — a cross-validation fold and a meta optimiser's subproblem both refit the graph, so a number tuned once is applied to graphs it was never tuned for. The choice a rule makes is **which quantity stays put**: a stated `dmax` holds the *radius* still and lets the related-pair count move with the graph, a quantile rule holds the *count* still and lets the radius move. Neither is stable in both senses, because the graph is refitted either way. Measured over four folds of one year, a `dmax` fixed at the whole-sample quarter-quantile relates `84`, `110`, `96` and `110` of `380` pairs while the rule relates `96` in every fold.
`nothing` is deliberately outside `PathLengthValue` and reached only through the `Option`: it means the observed diameter, which is a budget the caller *states* rather than one a rule computes, so `resolve_separation`'s check is a plain `isa(dmax, Number)`.
*Avoid*: expecting a rule to be checked when it is stored. A functor's return type is not part of its signature, so the obligation is a run-time check inside `resolve_separation`, which then feeds the answer back through the ordinary constructor so it meets the same validation a stated budget meets. Storing `HopCount(; n = rule)` where the rule will answer `0` is legal; resolving it throws.
*Avoid*: resolving inside `separation_budget`. That kernel takes the separation **matrix** rather than the structure, deliberately, so it is the one place a rule *cannot* be answered — it refuses an unresolved member instead of returning a function. Every shipped consumer resolves first: `phylogeny_matrix`, both `clusterise` methods, and `phylogeny_features` for `Proximity`.
*Avoid*: asking `HopCountQuantile` for a precise share. `q` is continuous and a hop count is an integer, so the rule rounds to a shell; on a twenty-asset graph `q = 0.1`, `0.2` and `0.25` all resolve to `n = 2`. `PathLengthQuantile` has no such step and lands on `q` to within one pair. That is the sharpest practical difference between the two separations.

**Separation Decay** (`AbstractSeparationDecayAlgorithm` / `separation_decay`)
The rule turning a **Separation** (above) into a score. An open family applied by `separation_decay(dk, d, dmax)`: `LinearDecay` (the default, `dmax + 1 - d`), `ExponentialDecay` (`exp(-rate * d)`), `ReciprocalDecay` (`inv((1 + d)^power)`) and `NoDecay` (`1`). It is *not* the exponentially-weighted moment estimators' `decay`, which is a smoothing constant over observations; the field on `Proximity` is documented under `field_dict[:sdecay]` for the same reason `HopCount.n` is documented under `:ntn`. Its supertype is `AbstractAlgorithm` rather than `AbstractPhylogenyAlgorithm` — the phylogeny base file is a home forced by include order, not a claim, so a decay over taxonomy depth or loading magnitude would arrive as a *member* rather than a re-parenting.
The argument is a **real** separation, not an integer hop count, so one family serves a hop count and a weighted path length alike. The contract — defined for all `d >= 0`, `f(0) > 0` and maximal, monotone non-increasing, never assumed to reach zero — is *enforced*, by a probing `assert_separation_decay` fallback on the abstract type that the shipped members override to a no-op. Opt-out, not opt-in: `Proximity`'s diagonal comes out of the decay at `d = 0`, so a member that is not maximal there silently produces the structural-equivalence matrix the included diagonal exists to prevent.
*Avoid*: expressing a budget as a fall-off. **Truncation and decay are two knobs**: an exponential never reaches zero, so the reach stays with the Separation's budget and the decay only shapes the values inside it. `NoDecay` is the sharpest case of the split and the one most easily misread: no *fall-off* is not no *truncation*, so `Proximity(; decay = NoDecay())` is the neighbourhood **indicator** the retired `BinaryNeighbourhood` produced, not a matrix of ones. It is also the only shipped member that emits zero inside the budget — nowhere, since it is constant — so a zero in a Phylogeny Features matrix still means *unreachable-or-out-of-budget* and nothing else.
*Avoid*: putting a top-of-scale on a member. The budget arrives as `separation_decay`'s third argument and members may ignore it — only `LinearDecay` reads it. A free `f(0)` below `dmax + 1` goes negative and needs a `max(0, ·)` floor back, i.e. a second truncation hiding inside the decay; above it, rows flatten toward all-ones, where `AngularDist` degenerates. `ExponentialDecay`'s `rate` buys the same contrast without either hazard.

**Polarity** (`AbstractCentralityPolarity` / `centrality_polarity`)
Which of the two opposite quantities a Centrality algorithm's edge weights must be: `DistancePolarity` (small is close) or `SimilarityPolarity` (large is close). It is a fact about the algorithm's own mathematics rather than about the graph — on one and the same PMFG, closeness wants the distances and eigenvector centrality wants the similarities — so it is declared per algorithm by `centrality_polarity` and read by `centrality_graph`, the one site where the source and the algorithm are both in scope. The fallback on `AbstractCentralityAlgorithm` declares `nothing`, so a new member is unweighted until it opts in, which is the safe default: a wrong polarity does not raise, it reverses the ordering the algorithm is reading.
A declaration can be **withdrawn** for one instance. `TopologyOnly` in the algorithm's `ov` field asks for the centrality over the network's **topology alone**, so `centrality_polarity` answers `nothing` and `centrality_graph` takes the plain-graph route it already builds for the three unweightable algorithms — a redirect, never a new estimator. The override therefore belongs to the *algorithm*, not to the estimator, and `centrality_polarity` answers the **effective** polarity rather than the declared one. Only the five members that declare one carry `ov`, so `DegreeCentrality(; ov = TopologyOnly())` is a `MethodError` with no check written; and because `ct` is positional on every centrality surface, a configured algorithm reaches all of them and `CentralityEstimator` stays a pure bundle of `pl` and `ct`.
Polarity selects **which** weights an algorithm receives and never gates **whether** the call succeeds. Weightedness is a property of the *source*, not of the request — a caller names a **configured algorithm** and never asks *for* weights, and an unweightable pairing has not been handed a request it cannot serve. The one request there is runs *away* from the weights, and every source honours it, so `TopologyOnly` adds no case to the list. Five cases therefore run on the plain graph and none of them raises: a weightless source (a `ClustersEstimator`, a precomputed `Clusters`, a precomputed `PhylogenyResult`), `DegreeCentrality`, `Pagerank`, `KatzCentrality`, and `EigenvectorCentrality` on a tree branch. The last three are `Graphs.jl`'s own line, and `KatzCentrality` needs the route as real code rather than as an absent check — `Graphs.katz_centrality` throws an `InexactError` on a weighted graph.
*Avoid*: reading the override as a two-way switch, or spelling it `unweighted::Bool`. Forcing a polarity *onto* an algorithm would **succeed** rather than raise, since `calc_distance_weighted_graph` carries distances on both branches, and the algorithm would read a distance where it needs a similarity and reverse its own ordering in silence — polarity correctness is not a runtime property, so nothing could catch it. The codomain is closed at two states, which is still not a boolean: `unweighted::Bool` makes a claim *about the graph*, which is not the caller's to make, while a named request asks *what to read*, which is.
*Avoid*: expecting the Separation to be read on a weighted route. Those routes take the structure itself, because a closure is built by summing matrix powers and a power of a weighted matrix sums *products* of distances. So `NetworkEstimator.sep` is live on the unweighted routes and inert on the weighted ones — on a tree that splits four and four, since `EigenvectorCentrality` declares a polarity and still runs unweighted there. At the default `HopCount(; n = 1)` the two agree, since the closure of a graph at one hop is the graph. An `ov = TopologyOnly()` moves an algorithm to the unweighted route, and so puts `sep` back in play for it — which is why the override is a **choice** rather than a fold-stability simplification, and why no shipped default moves to it: it trades the estimated weights for a second knob, and under a bare `PathLength` that knob is the observed diameter.
*Avoid*: passing a weight matrix through a centrality algorithm's `args`. It is a second, undeclared channel, refused by `assert_centrality_args`. `Graphs.closeness_centrality` took one in its second positional slot and silently overrode the polarity; `Graphs.betweenness_centrality` takes one in its *third*, so a matrix in `args` bound to `vs` and overflowed the stack. The `StackOverflowError` is catchable, so what a matrix there cost was the call rather than the session.
*Avoid*: filling a Tree algorithm's `args` or `kwargs` at all. The same two splat fields sit on `KruskalTree`, `BoruvkaTree` and `PrimTree`, and there **every** channel they reach is a weighting channel, so `assert_tree_args` refuses a matrix or a vector in `args` and `minimize` in `kwargs`. Unlike the centrality case the calls succeeded, which made this the worse of the two: `distmx` is the second positional argument of all three `Graphs.jl` functions and `kruskal_mst` takes a `weight_vector` there as well, so a correctly sized one replaced the distances the `NetworkEstimator` built and returned a legitimate-looking tree of the wrong graph; `minimize = false` returned a **maximum** spanning tree while `calc_weighted_adjacency_graph` and `SimilarityPolarity` both still read the branch as a minimising one.
*Avoid*: reading the `args` rule as a rule about `kwargs` too. A keyword binds by **name**, so it cannot reach the positional slot the guard was written for, and the centrality family declares no matrix-valued keyword — a matrix in `kwargs` fails closed at the call with a `MethodError` or a `TypeError` and needs no guard. The tree family is the exception, and only because `minimize` names a genuine channel rather than because a matrix could get through.

### 3.9 Uncertainty Sets

**Uncertainty Set**
A robust optimisation construct used by JuMP-based optimisers: it is a **neighbourhood of a specific quantity** — a mean vector or a covariance matrix — and the optimiser protects against the worst case within it. Always a robust-optimisation concept. Shapes and constructors:

- **Box** (`BoxUncertaintySet`): independent per-parameter bounds.
- **Ellipsoidal** (`EllipsoidalUncertaintySet`, `Mu…`/`Sigma…` classes): joint confidence region; scaling parameter `k` from `NormalKUncertaintyAlgorithm`, `GeneralKUncertaintyAlgorithm`, or `ChiSqKUncertaintyAlgorithm`.
- **ℓ1 / cross-polytope** (`L1UncertaintySet`): one error budget `eps` shared across all assets and both signs, optionally scaled per asset by `sd`; worst case `mu'w − eps·‖sd ⊙ w‖∞`. **Mean-only** — it bounds a *characteristic* vector and has no covariance analogue.
- **Signed ℓ1** (`SignedL1UncertaintySet`): a budget per error sign (`ep`, `em`); worst case `mu'w − ep·[maxᵢ(−sdᵢwᵢ)]₊ − em·[maxᵢ(sdᵢwᵢ)]₊`. Not the joint set with `ep == em` — the joint set shares one budget across signs (`max(t₊,t₋)`), this one spends a budget per sign (`ep·t₊ + em·t₋`); they agree only when `w` is single-signed.
- **Estimators**: `DeltaUncertaintySet` (delta bounds), `NormalUncertaintySet` (normality assumption), `ARCHUncertaintySet` (bootstrap for time series — `StationaryBootstrap`, `CircularBootstrap`, `MovingBootstrap`), `CharacteristicUncertaintySet` (the ℓ1 family; shape via `L1UncertaintySetAlgorithm`/`SignedL1UncertaintySetAlgorithm`, whose `scaled` flag selects equal-weight vs inverse-volatility behaviour).

**A set carries the quantity it is a neighbourhood of** — `mu` on the ℓ1 pair, `val` on the two that serve both axes. A fitted set records the fit its radius or shape matrix was calibrated on, and **that quantity is the central estimate** the optimiser is centred on. The Prior is only the fallback, used when the set names nothing, which is how a hand-built set behaves. Precedence is the carried quantity, then the returns estimator's own field, then the Prior (ADR 0050). The box covariance route is the one exception, and it names no centre at all: its worst case `tr(A_u Σ_u) − tr(A_l Σ_l)` is built from the bounds alone.
*Avoid*: describing the Prior as the central estimate on a set-bearing route, or fitting the centre at the call site. A call-site fit cannot survive a **pre-built** set crossing the boundary, because by then the fit is gone — which is why the quantity rides on the set.

**ucs Triple** (`ucs` / `mu_ucs` / `sigma_ucs`)
The three ways to ask an Uncertainty Set estimator for its sets: `ucs` returns the mean and covariance sets as a pair, `mu_ucs` returns the mean half, `sigma_ucs` the covariance half. Conceptually `ucs` *is* the mean half alongside the covariance half — but for sampling-based estimators the pair shares one random draw threaded through both halves, so `ucs`'s covariance half can differ numerically from a standalone `sigma_ucs` call that re-seeds (deliberate; not a drift). ℓ1 sets are mean-only: their `ucs`/`sigma_ucs` throw (ADR 0035).
*Avoid*: assuming `ucs(ue) == (mu_ucs(ue), sigma_ucs(ue))` element-for-element under a fixed seed.

**Characteristic Vector**
The per-asset quantity an ℓ1 uncertainty set is built around. Usually the expected return, but the construction is indifferent: a prior built on `StandardDeviationExpectedReturns` ranks on volatility instead. The radius `eps` decides how many assets the resulting portfolio holds — small radius, one asset; moderate, a quintile; large, all of them equally — which is what makes the *quintile* and *1/N* heuristics exact solutions of a robust optimisation problem rather than folk wisdom (ADR 0032). There is deliberately no quintile optimiser: the portfolios are a `MeanRisk` recipe over this set. An optimisation may hold **several** — one per set-bearing return term, each carried by its own set (ADR 0050, ADR 0052).
*Avoid*: using *characteristic* for a column of a **Feature Matrix** (§2), and vice versa — and in particular avoid telling them apart **by count**, which multiplicity broke: *k* return terms carry *k* Characteristic Vectors, which is the shape of a matrix of *k* columns. They are held apart by **use** — a Characteristic Vector is a per-asset quantity this family builds an uncertainty budget around and which enters the objective as `mu'w` (`CharacteristicUncertaintySet`), while a Feature is a coordinate of an asset and the matrix's rows become a distance — and by **side**: features are data and are sliced by a view, characteristics ride on estimators and are resolved per term at solve time, so *k* of them never accumulate into a Feature Matrix. `Characteristic…` naming was the first candidate for the feature work and was rejected on exactly this collision (ADR 0045).

**Radius Calibration** (`ActiveAssetsUncertaintyAlgorithm`)
The conversion from "how many assets should I hold?" to the opaque radius `eps` that produces it, by inverting the closed forms. A *calibration, not a constraint*: exact only for the bare budget-and-sign problem the closed forms assume, so any further constraint may move the realised count. `card` remains the tool for a hard cardinality bound.
*Avoid*: reading `active` as a promise.

## 4. Optimisation

**Optimisation Estimator**
Produces portfolio weights from a Prior Result and zero or more constraints/risk measures. Primary split: **JuMP-based** (require a solver), **Non-JuMP-based** (naive + clustering), and **Meta-optimisers** (wrap others). Finite Allocation is a separate post-processing class.

### 4.1 Naive (Non-JuMP)

- **EqualWeighted**: `1/N` across assets.
- **InverseVolatility**: weights inversely proportional to asset volatility.
- **RandomWeighted**: random feasible weights (baseline/benchmark).

### 4.2 Clustering (Non-JuMP)

- **HierarchicalRiskParity (HRP)**: recursive bisection of the cluster tree, allocating by risk.
- **HierarchicalEqualRiskContribution (HERC)**: equalises risk contribution across clusters then within.
- **SchurComplementHierarchicalRiskParity (SCHRP)**: HRP augmented with a Schur-complement step (`MonotonicSchurComplement` / `NonMonotonicSchurComplement`).

`HierarchicalOptimiser` is the shared configuration for these.

**Branch order.** `branchorder` sets the dendrogram's **leaf permutation** only; it never changes
the merge tree or the cluster memberships. HRP and SCHRP bisect that permutation, so it is their
algorithmic input: they pin the optimal ordering and declare no `branchorder` keyword. HERC and
NCO read cut memberships, which are invariant under branch order, so for them the keyword is an
optional cost dial. See ADR 0055.

### 4.3 JuMP-based

- **MeanRisk**: efficient-frontier trade-off between expected return and risk (Markowitz-style). The classic objective optimiser.
- **RiskBudgeting (RB)**: targets prescribed risk *contributions* per asset/factor rather than a return/risk trade-off. `AssetRiskBudgeting` / `FactorRiskBudgeting`; formulations `LogRiskBudgeting` (log-barrier), `MixedIntegerRiskBudgeting`.
- **RelaxedRiskBudgeting (RRB)**: convex relaxation — `BasicRelaxedRiskBudgeting`, `RegularisedRelaxedRiskBudgeting`, `RegularisedPenalisedRelaxedRiskBudgeting`.
- **FactorRiskContribution (FRC)**: optimises/targets risk contributions attributed to factors.
- **NearOptimalCentering (NOC)**: finds a robust interior point near the efficient frontier (`ConstrainedNearOptimalCentering` / `UnconstrainedNearOptimalCentering`).

**JuMP Returns Estimator**
One term of the objective's **return expression**. A JuMP optimiser takes one or a vector of them (`ArithmeticReturn`, dot-product; `LogarithmicReturn`, geometric), and the model-global return expression is their **weighted sum** — there is no Scalariser on this side, a deliberate asymmetry with the Risk Measures (ADR 0052). Each term carries its own settings (`JuMPReturnsSettings`, in a `settings` field placed first): its weight in the sum, its own lower bound, whether it enters the expression at all, and which charges are netted off it. That weight is a **Combination Weight** (§5): **inert when the term is alone**, since one term is not a sum, so the singular route drops it (ADR 0053; ADR 0052 amended).
So a term may **bind a bound while contributing nothing** to the objective, and a term that is not in return units simply stays out of the sum. When *every* term is out — by the `NoReturn` sentinel type, by `rte`, or by any mix of the two — the expression is degenerate, and the guards that refuse it test that **state** rather than the term's type (ADR 0054). A set-bearing term is centred on the Characteristic Vector its Uncertainty Set carries (§3.9, ADR 0050).
*Avoid*: reading a vector of terms as a vector of characteristics. A `LogarithmicReturn` term holds **no** per-asset quantity at all, and an excluded term contributes none — the plural noun here is the **term**, not the characteristic (ADR 0052).

**Objective Functions**: `MinimumRisk`, `MaximumUtility`, `MaximumRatio` (Sharpe-type), `MaximumReturn`.

**Solver**
Wraps a single external backend (Clarabel, HiGHS, …): solver module, settings, name, and `check_sol` kwargs passed to JuMP's `is_solved_and_feasible` to validate a solution at runtime.

**JuMPOptimiser**
The shared JuMP model configuration: one or more `Solver`s (fallback chains) plus JuMP-level settings. The execution environment consumed by all JuMP-based optimisers.

**Model State**
The in-flight state of the JuMP model as it is being built — the shared variables, expressions, and scales that successive constraint/risk builders read and write. Reached only through its named interface, never by naming an entry directly; see Per-Build Risk State for which entries a nested build must keep to itself.
An entry is disambiguated on **two independent axes**, and the interface resolves both: the **prefix** separates one *build* from another, so a nested build cannot alias the build enclosing it; the **index** separates one *measure instance* from another inside a single build, so two of the same risk measure in one vector get their own scratch. The axes are orthogonal — a nested build's third CVaR is prefix *and* index — and neither substitutes for the other: seeding the index with the prefix spells the same fact twice, and it was that redundancy, not a collision, that kept the index outside the seam-lock (ADR 0037, amended). Both compose: `nested_prefix` on the build axis, `nested_index` on the measure axis, the latter for a composite measure that builds its parts in the *same* build.

**Per-Build Risk State**
The part of Model State that belongs to *one* risk build rather than to the model as a whole, and so must not be shared with a build nested inside it. Two things qualify: anything that is a function of the weights being optimised — a nested build (risk tracking) shifts those weights, so the inner and outer answers genuinely differ — and any presence flag that gates a formulation choice for the build that set it. Everything else is a pure function of the Prior, identical in both builds, and is correctly shared: treating it as per-build would break sharing rather than protect it. The distinction decides correctness, not tidiness: getting it wrong makes a nested build silently read its parent's state.

**Model Assembly**
The fixed sequence in which a single-JuMP-model Optimisation Estimator's constraint and risk builders run to turn an empty model into a fully-constrained one — the steps between shaping the weight variables and setting the objective. Shared by MeanRisk, Risk Budgeting, Relaxed Risk Budgeting, Factor Risk Contribution, and constrained Near Optimal Centering; the per-optimiser parts (how weights are shaped, the objective, the solve) sit outside it. Unconstrained Near Optimal Centering has a middle of its own — the four steps of the sequence it needs, dispatched through the same named seam — and "unconstrained" names the builders it omits rather than the absence of a middle (ADR 0008, amendments 2 and 3). Distinct from Model State: Model State is the data the builders read and write, Model Assembly is the ordering of the builders themselves.

**Objective Penalty**
The accumulator through which every *soft* contribution reaches the objective — built-in regularisation, soft Turnover and Tracking, and user-supplied Custom Objective Terms alike. It exists because the objective's optimisation *sense* varies (some objectives are minimised, some maximised, and a ratio objective is either depending on the risk measure), so contributions are expressed once, sense-free, and the sense-correct factor is applied centrally when the accumulator is folded into the objective. A contribution always worsens the objective; a reward is a negative contribution. Distinct from a hard constraint, which bounds the feasible region rather than pricing a preference.

**Custom Term** (the extension point)
The user-facing escape hatch for expressing a preference the library does not already name: a Custom Objective Term prices one (through the Objective Penalty), a Custom Constraint mandates one. Both are supplied as inputs to the JuMPOptimiser and both are reached during Model Assembly, receiving the model under construction, the outer Optimisation Estimator, and the processed problem data. A term that names itself as one but supplies no builder is an error, not a no-op — on every path that reaches the hook, which unconstrained Near Optimal Centering does through its anchor sub-problems rather than through its centring solve (ADR 0036, amended).

### 4.4 Constraints

**Universe Sets**
A user-defined mapping of names to named groups (sectors, countries), groups, or unique-member groups. Always user-defined, though code can generate them (e.g. from Phylogeny). The foundational input to nearly all Constraint Generation.
It declares **every axis it carries**: `xkey`/`uxkey` for assets (defaults `"nx"`/`"ux"`), `fkey`/`ufkey` for factors (`"nf"`/`"uf"`), and `zkey` for features (`"nz"`). Assets are the *primary* axis — required, and the one a View slices; the factor and feature axes are optional, and a consumer that needs one and does not find it fails at the point of need rather than at construction. A key prefixed with an axis key is a **partition** of that axis (same length as its universe); a key prefixed with the axis's unique key is the partition's distinct entries. Anything else is a plain group, expanded by name and **axis-blind** — which is why factor groups need no machinery of their own, and why a factor constraint naming an asset group degrades to unknown-name warnings rather than to an error.
The feature axis is the exception that proves the prefix rule: it has **no** prefix convention and no unique-entry sibling, because nothing is written *over* it. A graded Feature Program's taxonomy keys are `xkey`-prefixed and asset-length, and its columns are named out of the flat list at `zkey` directly. Its only rule is `allunique`.
*Declaring* an axis is what lets a View leave its entries alone, but for two different reasons. Factors pass through because an asset index is *meaningless* on them (before ADR 0047 a factor universe was carried in the *asset* key, so the exemption had to be maintained by hand, per field, by omitting an annotation). Features pass through because the axis is **authored** — some of its nodes *are* assets — which is what makes `size(Z, 2)` fold-invariant, at the accepted cost of a dropped asset's node surviving as an all-zero column (ADR 0045, amended).
A universe is an **ordered** declaration: position is the only link between a name and a column of the data, so a universe that lists the right names in the wrong order attaches every constraint, bound and group to the wrong column and still solves. Each axis is therefore checked against the corresponding axis of the returns data — name for name, in order — where both sides exist, and a disagreement is an error rather than a silently different answer.

**Constraint Space** (ADR 0047)
The basis a constraint's names resolve in. Assets are not a member of this family: the asset basis is the *absence* of a re-basis, spelled `LinearConstraintEstimator`. `FactorSpace` is the only member today, and `ExposureConstraintEstimator` is the wrapper that declares one — it decorates whatever `lcse` already accepts, so parsing and group expansion are inherited rather than re-implemented.
**A space carries the *source* of its basis, not the basis itself.** `FactorSpace.re` names where the loadings come from, with the precedence `resolve_factor_regression` already fixes for every factor consumer: a precomputed regression wins, then the prior's own `rr` (which `re === nothing` always resolves to), then a refit from the returns. The third arm is what makes a factor mandate legal on a prior that carries no factor block. A **stated** basis is asset-indexed data written before the universe was known, so it is *viewed* wherever the universe is sliced and *refused* where it is replaced — a Nested Clustered outer solve writes cluster names over the asset universe, and no slice of asset loadings follows that. An **estimator** is refused nowhere: it refits against whatever universe it is handed.
**A constraint can be re-based iff it is a linear form in `w`**, and the boundary is a property of the **mechanism** rather than of any list of families: the re-basis rewrites a row and leaves the model be, so a constraint reaching the model through its own variables is outside it even where the factor quantity is well defined. Cardinality and Threshold rows index the binary held-indicators, not `w`, so a projected row is neither integral nor an index into them; Weight Bounds are a per-asset box, and a factor box is a linear constraint rather than a bound; Turnover and Tracking Error are norm forms, each declaring its own variables and cones — re-basable in mathematics, not by rewriting a row; a Fee is priced per *traded position* and charged against the return, so it is not a constraint on `w` at all. Their exclusion is a property of the constraint, not a gap — and is enforced by *type*: `ExposureConstraintEstimator` decorates `lcse` alone, so an illegal space cannot be written down.
*Avoid*: reading a Constraint Space as a change to the **problem**. It is a change of coordinates on the same feasible region — the projection happens during Constraint Generation and what reaches the optimiser is an ordinary asset-space linear constraint.

**Factor Exposure Constraint**
A linear constraint on a portfolio's factor weights `w_f = Mᵀw`, written in factor names and projected through the loadings. Under a dimension-reducing regression `M` and `L` are the two sides of one projection, each recoverable from the other, so which one a consumer reads is decided by what it is doing rather than by convention: a *risk decomposition* reads `L`, the orthogonal reduced basis its covariance was estimated in, while a *constraint* reads `M`, because a constraint is written down and only `M`'s columns carry names a user can put in an equation.
It is the one constraint that **cannot** be precomputed by hand without going stale: the loadings are refit per fold, so a row computed once against a full-sample `M` describes an exposure the fold's model does not have. Requesting one against a prior that carries no regression is an error, never a dropped row — dropping every row of a factor mandate yields a feasible portfolio with none of the requested exposure.
*Not* every factor mandate is one of these. Tracking a factor needs no re-basis at all: `ReturnsTracking` takes a benchmark *return series*, and a factor's return series is a column of `F`, so it is passed directly.

**Constraint Generation**
User-facing utilities converting high-level specifications into the numeric form optimisers need (linear systems or bounds). Built on Universe Sets. Concepts:

- **LinearConstraint / LinearConstraintEstimator**: parses equations into `A·w ≤/= B` (`PartialLinearConstraint`, `ParsingResult`).
- **ExposureConstraintEstimator**: re-bases a linear constraint into another Constraint Space.
- **WeightBounds**: per-asset lower/upper bounds (scalar = same for all; vector = per asset). `UniformValues` gives `1/N…1`. Group-level sum bounds are linear constraints, not Weight Bounds.
- **RiskBudget**: per-name/group risk-contribution targets. Which axis the names live on is a property of the *algorithm*, not of the sets: `AssetRiskBudgeting` budgets the asset weights and resolves against `xkey`, `FactorRiskBudgeting` budgets the factor weights `w1` and resolves against `fkey`. The factor budget is indexed by the columns of `L` — the reduced basis the risk decomposition works in — so under a dimension-reducing regression it is shorter than the factor universe, and the mismatch is reported against `L` rather than silently mis-attributed.
- **Phylogeny constraints**: `SemiDefinitePhylogeny` (SDP coupling) and `IntegerPhylogeny` (cardinality/grouping); `CentralityConstraint` constrains by centrality.
- **Threshold**: buy-in thresholds (minimum non-zero position size).
- **AssetSetsMatrixEstimator**: builds binary group-membership matrices.

**JuMP Constraints**
The layer adding numeric constraint data into a JuMP model. Includes budget constraints (`BudgetRange` = sum-of-weights interval, `BudgetCosts` = linear transaction costs, `BudgetMarketImpact` = power-law impact), `LpRegularisation`, plus Turnover, Tracking, and Fees constraints.

**Net vs Gross Budget**
`bgt` is the *net* exposure `1ᵀw`; `sbgt` the short side; `gbgt` the *gross* exposure (leverage) `‖w‖₁`. `bgt` and `sbgt` constrain net and gross only **together** — pinning both gives `1ᵀw = bgt` and `‖w‖₁ = bgt + 2·sbgt` — so `gbgt` exists for the combination they cannot reach: gross pinned with net free (the market-neutral portfolio). A number pins it, a `BudgetRange` bounds it (a leverage cap). It requires bounds admitting shorts, and is rejected when `bgt` and `sbgt` already determine the gross exposure.

**Exact vs Bounded Budgets** (`xbgt`, `miprb`)
The long/short variables `lw`/`sw` are *upper bounds* on the parts of `w`, not equal to them, so by default every budget built on them **bounds** the realised exposure rather than pinning it: `sbgt = 0.3` means *at most* 30% short. The objective normally pushes against the budget, making the bound tight, but it is not the same problem. Pinning the decomposition needs a per-asset *sign* bit and the big-M constraints keyed on it, so the budgets hold exactly — turning a linear program into a mixed-integer one, which is why it is opt-in. (The MIP complementarity used by threshold/fee constraints forces the long-xor-short *sign pattern* but does **not** close this slack; pinning emits that pair *and* the two constraints that close it.)

Two callers want this, and they are one request routed through the **head contract** (`decomposition_contract`, in Model State): an explicit `xbgt` under a head whose `lw`/`sw` merely *bound* `w` (`PartsBoundWeights`), and risk budgeting's head, which makes `w = lw − sw` an *identity* (`WeightsFromParts`) and so asks for the pin by construction with no flag — the mechanism the old `miprb_flag` named. Both route to `set_exact_budget_constraints!`, which reads the contract back off the model and emits what it needs (the identity needs only the pin-pair; the bound needs two more constraints to close the slack).

The sign bit need not be new. When the model already runs `short_mip_threshold_constraints` for a threshold / fee / cardinality feature, its `ilb`/`isb` already mean long and short and the exact-budget constraints reuse them — `i_mip = ilb + isb` stays the *held* bit that `card` counts. Only when *nothing else* consumes a held indicator does pinning declare its own: `sign_mip_constraints` adds a single sign bit `xb` per asset (`N` binaries, not the `2N` of a full long-short split), because the sign split is total — `w_i = 0` satisfies both `w_i ≥ 0` and `w_i ≤ 0` — so the inactive third state a held builder carries is dead weight. The long-only `mip_constraints` cannot serve it either: its `ib` means *held* and carries no sign, and a sign bit and a held bit are different bits. `xbgt` is ignored when the weight bounds admit no shorts, and is applied by `set_mip_constraints!` rather than `set_weight_constraints!`, since the budgets are built before the binaries that pin them exist.

**Time-Dependent Input (Schedule)**
An optimiser input whose value changes across the folds of a cross-validation scheme instead of being fixed for the whole horizon. Historically "time-dependent constraint" — constraint inputs were the first to vary — but the concept spans every **problem-definition** input.
An input may be time-dependent **iff it is problem definition** — what is being solved: priors, returns, risk measures, objectives, constraints, weight finalisers, universe sets, warm starts, fallbacks, the meta-optimisers' inner/outer optimisers — never **execution control** — how it is solved: solvers, JuMP scaling factors, RNGs and seeds, strictness flags, a meta's own inner cross-validation scheme. The widened constructor signatures encode the criterion and are the admissibility table.
Expressed by storing a per-fold sequence of values — or a callable (a function, or a functor struct that can additionally declare its previous-weights needs as inspectable data) computing the value from the fold's Time-Dependent Context — *directly in the optimiser input it varies*, so an input holds either a static value or a schedule, never both, and the input's position names what varies. Schedules never nest, but an estimator swapped in by a schedule may carry schedules in its own inputs — those resolve against the same fold's context after the swap.
Entry *i* corresponds to fold *i* of the consuming scheme's split enumeration; the machinery imposes no ordering of its own, so where that enumeration is not a timeline (combinatorial splits, randomised paths) keying entries to time is the user's responsibility. For an input that itself accepts a vector of constraints, entry *i* is fold *i*'s whole vector (a vector of vectors); there is no separate "vector of schedules" — to vary individual entries within a constraint vector, build the fold's vector inside a callable.
A time-dependent input participates only where folds exist and is inert everywhere else — a plain (fold-less) optimisation, including a meta-optimiser's full-window solve, resolves the schedule to its own explicit `default` if it carries one, else to the input's static default; a *required* input with neither (the optimiser-valued positions) fails closed with a structured error.
Which fold loop consumes a schedule is chosen by its `bind`: the default `:outermost` binds it to the outermost fold loop (so an inner estimator's schedule under an outer backtest is sized to the outer folds), while `:nearest` binds it to the nearest enclosing loop — inside a meta-optimiser's inner estimators, the meta's own cross-validation leg, which consumes it (sized to the inner folds) even under an outer backtest. Entitlement to `:nearest` is per *input*, not per host: an input a host hands across its own inner fold loop belongs to that inner loop, whatever loop reaches the host. Either way a schedule must be sized to the folds of the loop that consumes it.

**Schedule of Optimisers**
A Time-Dependent Input whose per-fold values are whole optimisers — so the *strategy itself* varies over the backtest (e.g. regime switching). Lives in two positions: handed to cross-validation directly as the optimiser (fold *i* runs entry *i*), or held in an optimiser-valued input (a fallback, a meta-optimiser's inner/outer optimisers, a Pipeline's optimisation step). Entries may mix estimators and precomputed Optimisation Results — a fold optimises or predicts depending on what its entry is — except under asset-subsampling cross-validation, where a solved result has no asset-subset view and is rejected. Optimiser positions are required inputs, so a schedule there must state its fold-less `default` explicitly. `:nearest` binding is legal only where an inner fold loop actually consumes the position (NestedClustered's inner optimiser as a whole; an individual Stacking candidate) and is rejected at construction everywhere else, fallbacks included.

**Time-Dependent Context**
The per-fold information handed to a function-form Time-Dependent Input: the fold's position in the scheme's split enumeration (which indexes the fold index vectors, so a function can always identify its own training and test windows), the fold count, the fold's data (the possibly asset-viewed returns; for Pipeline fold loops, the raw pre-preprocessing input), and — only when previous weights are threaded — the previous fold's weights.

**Weight Finaliser**
Post-solve adjustment forcing weights into the feasible region: `IterativeWeightFinaliser` (projection) or `JuMPWeightFinaliser` (re-solve).

**Turnover**
Measures/constrains change relative to a reference. Dynamic (vs previous weights, needs them at runtime) or fixed (vs static target). Soft penalty or hard constraint.

**Tracking Error**
Measures/constrains closeness to a benchmark/target. `WeightsTracking` (vs a reference weight vector — needs a weight vector, not necessarily *previous*) or `ReturnsTracking` (vs a benchmark return series). Return tracking formulations: LxNorm error family (norm-based errors). Risk tracking formulations: `IndependentVariableTracking` / `DependentVariableTracking` (variable). Soft penalty or hard constraint.

### 4.5 Meta-optimisers

Wrap and compose other optimisers, using Cross-Validation to estimate inner out-of-sample returns:

- **NestedClustered (NCO)**: optimise within clusters, then across clusters.
- **Stacking**: combine several base optimisers' weights via a meta-optimiser.
- **SubsetResampling**: average optimisations over many asset/observation subsets for robustness.

**Sub-Portfolio Enumeration** (`SubPortfolioUniverse`, ADR 0066). NCO and Stacking run the *same* module: one inner problem per **sub-portfolio**, one predicted return series per sub-portfolio, and an outer optimiser over a synthetic universe with one asset per sub-portfolio. They differ in one thing, and this names it — what a sub-portfolio *is*, and what it *sees*. NCO enumerates cluster index sets and views the inner optimiser, the Prior Result and the Fees onto each (`ClusterUniverse`); Stacking enumerates its inner optimisers, each seeing the whole universe (`FullUniverse`). Everything else — the fold-less path, the two cross-validated paths, the feature collapse, and the zero-padding at the outer collapse — is written once and reads the enumeration back. *Avoid*: dispatching a new meta-optimiser's prediction on the host's own type parameters. `predict_outer_returns` dispatches on `cv`, which is what a custom cross-validation scheme overloads.

### 4.6 Cross-Validation

Used for (1) hyperparameter tuning and (2) out-of-sample evaluation; also supplies inner out-of-sample returns to meta-optimisers. Axes: **optimisation vs non-optimisation**, **sequential (time-aware) vs non-sequential**. Schemes:

- **KFold**: k-fold with optional purging/embargoing.
- **CombinatorialCrossValidation**: all train/test combinations (multi-path).
- **WalkForward**: `IndexWalkForward` / `DateWalkForward` for time series (purging, windowing; `DateAdjusterEstimator` for date alignment).
- **MultipleRandomised**: random asset subsets × rolling windows.
- **Search**: `GridSearchCrossValidation`, `RandomisedSearchCrossValidation` (scorers `HighestMeanScore`, `NearestQuantilePrediction`; `PredictionScorer` vs `PopulationScorer`).

Result types: `PredictionResult`, `MultiPeriodPredictionResult`, `PopulationPredictionResult`, `PredictionReturnsResult`.

### 4.7 Finite Allocation (post-processing)

Discretises continuous weights into whole shares for a fixed cash budget (real-world: no fractional shares).

- **DiscreteAllocation**: MIP-based exact allocation (needs a MIP solver).
- **GreedyAllocation**: heuristic greedy rounding.

**FiniteAllocationInput**
The problem data fed to a Finite Allocation optimiser: target weights, asset prices, cash budget, and optional time horizon / fees. Shared by both `DiscreteAllocation` and `GreedyAllocation`, and passed as the single positional argument to `optimise`. Subtypes `AbstractEstimator` — deliberately treated as a configuration object (the primary input to `optimise`) rather than as a computed output, keeping the `Result` tree reserved for outputs and staying clear of the `plot_*`/`OptimisationResult` dispatch surface. This makes it the one pure-data struct classified as an Estimator (the `WeightBounds`/`RiskBudget` precedent puts data under the Result tree); the deviation is intentional so the allocation *inputs* never collide with allocation *results*.
*Avoid*: FiniteAllocation (that is the family), AllocationProblem, AllocationInput.

## 5. Risk Measures

**Risk Measure**
Quantifies portfolio risk. Three-way split by legal usage: **Optimisation** (has a JuMP formulation; usable as objective/constraint), **Non-Optimisation** (analysis only; may be negative or non-formulable), **Hierarchical** (no JuMP formulation but valid as a clustering risk proxy).

**Risk Input Kind**
A classification of a risk measure *orthogonal* to the legal-usage split above: what the measure consumes when its expected risk is evaluated. Three kinds — **net-returns** (a function of the portfolio's net-return series), **weights-returns-fees** (needs the weights, the asset-return matrix, and fees together — the moment families and tracking), and **weights-only** (a function of the weights alone — the variance/standard-deviation family). A measure sits on this axis independently of whether it is Optimisation, Non-Optimisation, or Hierarchical, so the two axes are stated separately.

**Precomputed-returns contract**
The rule for evaluating a risk measure on an *already-reduced* net-return series the caller holds directly, with no weights to apply (e.g. an out-of-sample portfolio return series, or a single asset's return column). Well-defined exactly when the measure's result is a function of the series alone: the net-returns measures, and the moment families whose target is weight-independent. It is undefined for weights-only measures and for moment measures carrying a per-asset target — these need the weights the bare series no longer carries — and asking for it there is a defined error, never a silent wrong number.

The `XatRisk` naming uses "X" as shorthand for "Value" or "Drawdown" — the same family applied to returns or to drawdowns. "Relative" variants are the hierarchical drawdown forms; "Range" variants penalise the gap between upper and lower tails; "Distributionally Robust (DR)" variants optimise against worst-case scenario distributions.

**Range Tails**
The two point measures a Range variant is the *sum* of: the **loss** tail on the net portfolio returns at `alpha`, and the **gain** tail on their negation at the range's second level (`beta`, `w2`, …). A range declares them once, through `range_tails`, and both consumers read them from there — the model builder (`set_range_risk_constraints!`, which calls each tail's own builder through the `loss::Bool` seam) and the measure's own value-level functor. The tails carry `rke = false`: the bound and the objective contribution belong to the range as a whole (ADR 0057). `GenericValueatRiskRange` is the member whose tails are *given* by the caller rather than derived from a second level. *Avoid*: reading every Range as decomposable — `OrderedWeightsArrayRange` under `ExactOrderedWeightsArray` and `ValueatRiskRange` under `DistributionValueatRisk` **fuse** their tails into one shared formulation, so they declare none.

**Risk Series**
The per-observation series a conic tail measure *reduces*, and the sibling axis to Range Tails: a range is its base measure applied twice along the **tail** axis, a drawdown measure is its returns twin applied once along the **series** axis. Two series exist — the **net portfolio returns**, and the **negated drawdown path** `-dd[2:T+1]`. Both are signed as *returns* (a loss is a negative entry), which is what lets one builder body encode both twins under the substitution `net_X ↦ -dd[2:T+1]`. A builder declares neither series itself: it asks `risk_series` with `NetReturnsRiskSeries()` or `DrawdownRiskSeries()` and receives `(series, T)` (ADR 0059). Only the returns series has a gain tail — a run-up is a different recurrence, not this one negated — so `risk_series` takes `loss::Bool` on that marker alone and **no drawdown measure can be range-composed**. *Avoid*: reading the drawdown twin as a separate formulation — six pairs (MIP VaR/DaR, CVaR/CDaR, DR-CVaR/DR-CDaR, EVaR/EDaR, RLVaR/RLDaR, PNVaR/PNDaR) are one programme each, and a cone fixed on one side is wrong on the other until it is fixed at the shared kernel.

### Settings & combination

- **RiskMeasureSettings / HierarchicalRiskMeasureSettings / MaxRiskMeasureSettings**: per-measure config (scaling, bounds, direction).
- **Scalariser**: combines multiple risk measures into one scalar — `SumScalariser`, `MaxScalariser`, `MinScalariser`, `LogSumExpScalariser`. Risk side only: several return terms are always a weighted sum, and there is no return-side Scalariser (§4.3, ADR 0052).
- **Degeneracy Guard** (both axes): the refusal an optimiser or objective makes when the expression it reads is *identically zero*. It tests the **state of the expression, not the type of the term** (ADR 0054), because two routes reach that state on each axis — the sentinel type (`NoReturn`, `NoRisk`) and the inclusion flag (`rte`, `rke`). The return predicate is **fused** over both routes, since a vector may mix them; the risk predicate **composes**, because its type half quantifies with `any` and its state half with `all`. That `all` is what keeps a **constraint-only** element expressible — one that binds a bound without entering the objective. The guards have two homes: a *formulation* guard
(does this optimiser need the term at all?) sits at the constructor, and an *objective* guard sits at the shared model-build seam, the only site that sees every objective-carrying optimiser. *Avoid*: widening the predicate for the clustering optimisers — they never reach the JuMP builders, so `rke` is inert there and they keep the type test.
- **Combination Weight** (`settings.scale`, both axes): the weight an element carries inside a combination of *several* elements. One element is not a combination, so the weight is **inert on a lone element** — the singular routes drop it before the expression is built, on the risk axis and the return axis alike (ADR 0053). The elements combined are risk measures (`:risk_vec`), return terms (`:ret_vec`), sub-portfolios (`Stacking`), or — in Schur Complement HRP — whole **portfolios**, which is why the noun is "combination" and not "aggregate of measures". Bounds (`ub`/`lb`) bind on an element's own expression, *before* the weight; `rte`/`rke`, `fee` and `mic` are not weights and survive the drop. *Avoid*: reading `scale` as an absolute gain on one element — it only ever expresses a ratio between siblings. Schur is inert by **renormalisation** rather than omission, so it is not a template for the other sites. `Stacking` is the one site where an *optimiser* chooses the combination, so the weight multiplies the outer answer and the coefficients are rescaled to the total that optimiser chose (`combination_weights`); it is inert by that rescale, as Schur is, and the outer problem never sees the weight (ADR 0053 amended). *Avoid*: adding one to a site whose elements are drawn at **random** — `SubsetResampling` carried one and it was removed, because a per-element weight needs elements the caller can name (ADR 0053 amendment 2).
- **Frontier / FrontierBoundEstimator**: efficient-frontier point count and bound transforms (`LinearBound`, `SquareRootBound`, `SquaredBound`).
- **Frontier Sweep**: the sequence of solves a bound-swept optimiser runs, and the seam that runs it (ADR 0062). A swept entry — a risk measure's `ub`, a return term's `lb` — registers one JuMP **parameter** and one bound constraint, so the model is assembled once and a point changes only parameter values. The sweep is a **product**: every swept entry on both axes joins the same `Iterators.product`, so `k` bounds of `N` points cost `N^k` solves (`assert_frontier_sweep_cap` is the ceiling on that product, not on any one `N`). The flat order is **return-outer, risk-inner**, and it is load-bearing: `NearOptimalCentering` solves its anchors as one `MeanRisk` sweep over the same frontiers and pairs anchor `i` with point `i`. `MeanRisk` needs no per-point hook; `NearOptimalCentering` uses one to move `noc_rk`/`noc_rt`. *Avoid*: writing a parameter loop or a collect tail at a call site — `set_ret_frontier_parameters!`, `set_risk_frontier_parameters!` and `frontier_sweep!` are the seam, and they are what keep the polarity `d` and the homogenisation `k` from drifting between the two heads.

### Dispersion / moment measures

- **Variance**, **StandardDeviation**, **UncertaintySetVariance** (robust). Optimisation formulations: `QuadRiskExpr`, `SOCRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`.
- **LowOrderMoment** (`FirstLowerMoment`, `MeanAbsoluteDeviation`, `SecondMoment`, `EvenMoment`) and **HighOrderMoment** (`ThirdLowerMoment`, `FourthMoment`, `StandardisedHighOrderMoment`).
- **Kurtosis**, **Skewness**, **NegativeSkewness**, **VarianceSkewKurtosis** (composite), **ThirdCentralMoment**.
- **MedianAbsoluteDeviation (MAD)**: centred by `MedianCentering` or `MeanCentering`.

### Quantile / tail measures (XatRisk)

- **Value-at-Risk (VaR)**: `ValueatRisk` (formulations `MIPValueatRisk`, `DistributionValueatRisk`), `ValueatRiskRange`; drawdown forms `DrawdownatRisk`, `RelativeDrawdownatRisk`.
- **Conditional (CVaR / Expected Shortfall)**: `ConditionalValueatRisk`, `…Range`, DR forms, drawdown `ConditionalDrawdownatRisk` (CDaR) and relatives.
- **Entropic (EVaR)**: `EntropicValueatRisk`, `…Range`, `EntropicDrawdownatRisk` (EDaR), relatives.
- **Relativistic (RLVaR)**: `RelativisticValueatRisk`, `…Range`, `RelativisticDrawdownatRisk` (RLDaR), relatives.
- **Power Norm**: `PowerNormValueatRisk` (PNVaR), `…Range`, `PowerNormDrawdownatRisk`, relatives.

**OWA (Ordered Weights Array)**
Weighted sum of *sorted* return realisations; weights generated by an Algorithm composing L-moments (`MaximumEntropy` with `ExponentialConeEntropy`/`RelativeEntropy`, `MinimumSquaredDistance`, `MinimumSumSquares`, `NormalisedConstantRelativeRiskAversion`). Formulations `ExactOrderedWeightsArray` (LP) / `ApproxOrderedWeightsArray` (p-norm). `OrderedWeightsArray`, `OrderedWeightsArrayRange`. Flexible enough to approximate CVaR and higher-moment criteria.

### Drawdown measures

- **AverageDrawdown**, **UlcerIndex** (RMS drawdown), **MaximumDrawdown** — each with a `Relative…` hierarchical form.

### Other measures

- **BrownianDistanceVariance (BDVar)**: distance-covariance-based dispersion (`NormOneConeBrownianDistanceVariance`, `IneqBrownianDistanceVariance`).
- **WorstRealisation**: worst single-period loss.
- **NoRisk**: contributes *no* risk term at all. Exists so a risk-taking optimiser can express a problem that genuinely has none — an objective ignoring risk would otherwise still build the default `Variance`, dragging cone constraints into what is a linear program (the robust best-characteristic portfolios; the global maximum return portfolio). Only coherent under an objective that never consults risk, so `MeanRisk` rejects it with `MinimumRisk` and `MaximumRatio`, and the optimisers whose formulation *is* their risk measure reject it outright.
- **Range**: spread between best and worst realisations.
- **TurnoverRiskMeasure**: turnover expressed as a risk quantity.
- **TrackingRiskMeasure**: benchmark deviation measured as a **norm** of the portfolio-vs-benchmark difference.
- **RiskTrackingRiskMeasure**: benchmark deviation measured through a configurable **risk measure `r`** applied to the portfolio-vs-benchmark difference (a `WeightsTracking` benchmark + risk measure + `VariableTracking` algorithm) — i.e. tracking the *risk* relative to a benchmark rather than the raw norm.
- **EqualRisk**: enforces equal risk contributions (hierarchical).
- **RiskRatio** / **NonOptimisationRiskRatio**: ratio-form measures for hierarchical use.

### Non-optimisation (analysis) measures

- **ExpectedReturn**: scalar portfolio expected return given weights (reporting). Distinct from the per-asset mean vector in the Prior.
- **MeanReturn**, **ThirdCentralMoment**: reporting statistics.
- **ExpectedReturnRiskRatio** / **MeanReturnRiskRatio**: configurable return-to-risk ratio — subsumes Sharpe, Calmar, Sortino, etc. via choice of risk measure.

## 6. Post-Processing

**Net Returns**
Portfolio returns adjusted for fees and turnover costs; computed before drawdowns.

**Drawdowns**
Peak-to-trough declines computed from Net Returns; the input series to drawdown-based risk measures.

**Fees**
Composite of holding/trading costs, each proportional to the relevant weight unless fixed: **long** (≈ management), **short** (≈ borrowing), **turnover** (≈ commission), and **fixed** (constant charge whenever the weight is non-zero). Feed Net Returns and can act as optimisation constraints.

**Finite Allocation**
See §4.7 — discretising weights into whole shares within a cash budget.

## 7. Errors & Status

**PortfolioOptimisersError**
Base for library exceptions (`IsNothingError`, `IsEmptyError`, `IsNonFiniteError`, …) raised by argument validation.

**OptimisationReturnCode**
Status of an optimisation: `OptimisationSuccess` / `OptimisationFailure`.
