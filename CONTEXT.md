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
A moment Estimator standing where a prior-derived value goes. Four quantities defer — `mu`, `sigma`, `kt`, `sk` — so the slot admits the value itself or the Estimator that computes it, and **Factory** resolves the second form against the optimisation's own Prior. The struct that reaches the kernel therefore always holds a plain value. `MedianAbsoluteDeviation.mu` already carried this widening for its centring target; the rest generalise it. The field aliases are `MuSlot`, `SigmaSlot`, `KtSlot` and `SkSlot`; `DeferredQuantity` names the dynamic half alone, which is the four moment-estimator families and nothing else.
It exists because a stated value cannot follow a refit. A Cross-Validation fold and a Meta-optimiser's subproblem both refit the Prior, and a **View** is taken *before* Factory runs — so a stated matrix crosses that boundary as the whole universe's answer, while a Deferred Quantity crosses unresolved and computes on the subset. The choice is **which thing stays put**: a stated value holds the numbers still, a Deferred Quantity holds the method still.
A measure may instead name one Prior Estimator in `pe`, the **fan-out** form, which supplies every quantity the measure does not state. It is also the only form that reaches the factor returns, because no moment estimator takes them and a Prior Estimator does. Precedence runs: a stated field wins, `pe` fills the rest, the Prior supplies what neither names. **Nothing refuses** — a caller may state some quantities and defer others, so keeping them consistent is the caller's job.
*Avoid*: reading this `pe` as the optimiser's. A Risk Measure is an Estimator, so its `pe` admits an Estimator alone — narrower than every other `pe` in the library, which admits a precomputed Prior Result too. An optimiser and a measure inside it may name different priors, and each is used where it sits.
*Avoid*: deferring a quantity while stating the one derived from it. `chol` and `V` never defer — they arrive with the pair `sigma` and `sk` resolve to — so stating one without its source is refused at construction, rather than silently paired with a matrix the caller never saw.

**View**
The sub-selection counterpart to Factory. Where Factory propagates runtime *values* down a composed struct tree, a View propagates an index *selection*: it restricts an Estimator, Algorithm, or Result (or an array of them) to a subset of assets — or, for returns data, observations — and returns a new struct of the same type with every data-bearing field and composed child consistently sub-selected. Used wherever the library operates on part of the problem rather than the whole: meta-optimisers (Subset Resampling, Nested Clustered), Cross-Validation, and windowed moment estimators. Like Factory, it relies on composition — each struct declares which of its fields participate, and the selection is threaded recursively down the tree. Unlike Factory, View is primarily an *internal* mechanism — driven by the meta-optimisers and Cross-Validation rather than called directly by everyday callers — so its entry point `port_opt_view` is marked `public`, not exported.
Extension authors implementing a new composed estimator should define a `port_opt_view` method for it, or tag data-bearing fields with `@vprop` to have the method generated automatically.
`ReturnsResult` is Viewed through the same entry point: `port_opt_view(rd, i)` selects assets, while `port_opt_view(rd, i, j, k)` selects observations, assets, and factors — the one place in the family where the first index is not the asset index.

**Pipeline**
An Estimator that reifies an end-to-end workflow — price preprocessing, prices-to-returns, returns preprocessing, prior, phylogeny, uncertainty sets, constraint generation, optimisation — as an ordered list of optionally-named steps executed left-to-right over a Pipeline Context. Steps are ordinary Estimators; the step's family determines which context slot it reads and writes (custom steps use an explicit wrapper). Executed with `fit` (producing a result that carries every step's fitted Result) and evaluated out-of-sample with `predict`.
The Pipeline — data preparation included — is the unit that Cross-Validation splits (on contiguous input-time windows) and hyperparameter tuning searches; tuning lenses address steps by name or index and may swap entire estimators. Slots computed by pipeline steps override the optimisation step's corresponding internal configuration; slots with no step are computed internally by the optimiser as usual, so every stage is optional. Pipelines may nest as steps of other Pipelines; wrapping a Pipeline in a Meta-optimiser is deliberately unsupported (a Meta-optimiser may instead *be* the optimisation step).
*Avoid*: Workflow, Workbench (GUI-era synonyms); using "pipeline" for the library's informal stage ordering.

**Pipeline Context**
The accumulating blackboard threaded through a Pipeline's steps: a set of coarse typed slots — prices, returns, prior, phylogeny, uncertainty, constraints, weights — where each step reads the slots it needs and writes the slot its family produces. Heterogeneous slots (uncertainty, constraints) hold collections whose elements are routed to their optimiser targets by Result type. Internal machinery, not user-facing.

**Routing Target**
The destination a computed Pipeline Context slot is delivered to inside the optimisation step: `pe`, `cle`, `wb`, `lcse`, `ple`, `mu_ucs`, `sigma_ucs`. Targets are finer than slots and address a different audience — a *slot* is pipeline-author vocabulary naming a stage of the workflow, a *target* is optimiser-author vocabulary naming a destination, and nobody writing a Pipeline ever names one. The Pipeline owns the fan-out (splitting the uncertainty pair, grouping constraint results by Result type); the optimiser owns where a target lands, so a field rename stays local instead of breaking the Pipeline at run time. Five targets are named after the field they land in, making the routing rule derived rather than declared — a target lands in the like-named field of whichever optimiser has one. The two exceptions carry policy and name no field. A target an optimiser cannot receive is ignored when losing it cannot change the answer (`pe`, `cle`), an error otherwise. Internal machinery.

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
*Avoid*: confusing a Feature with a **Characteristic Vector** (§3.9) — near-synonyms held deliberately apart. A Characteristic Vector is *one* per-asset quantity that an ℓ1 Uncertainty Set is built around, and stays reserved for that family; a Feature is *one column of many* in a matrix whose rows are turned into a distance. Also avoid reading "feature" as *factor*: factor returns are `F`/`nf`, and factor loadings become features only by passing through the `RegressionFeatures` producer.

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

**Original Returns Matrix**
The returns matrix the caller supplied, as distinct from the one a Prior Result asserts. They differ on exactly three routes: `FactorPrior`, `FactorBlackLittermanPrior` and `AugmentedBlackLittermanPrior` each overwrite `X` with the reconstruction `F * transpose(M) .+ transpose(b)`, so their `X` is a *posterior* matrix. Everywhere else `X` already is the caller's returns.

Stored as `LowOrderPrior.o_X`: the original when `X` is not it, and `nothing` otherwise. `HighOrderPrior` reads it through its `forward(pr)` block and declares nothing of its own. `EntropyPoolingPrior` and `OpinionPoolingPrior` forward it, so it survives a pooling wrapper over a factor prior.

*Read it as `pr.original_X`*, a computed property that is always a matrix — `o_X` when there is one, `X` when there is not. The field is storage and answers a different question: `isnothing(pr.o_X)` is how to ask whether this carrier reconstructed `X`. Two names rather than one always-populated field, because `forward_prior` rebuilds through the keyword constructor with every field named, where a `nothing` is inert and a matrix would go stale.

The two are not interchangeable. The reconstruction spans only the factors, so it has rank `size(F, 2)` and carries no residual. A **Deferred Quantity** (§1) refits a moment on the sample and therefore reads `original_X`: fitting a covariance estimator on the reconstruction returns a singular matrix whenever there are more assets than factors. Consumers that evaluate portfolio returns keep reading `X`, because the reconstruction *is* the return distribution a factor prior asserts.

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

**Separation** (`AbstractSeparationAlgorithm` / `separation_matrix` / `separation_budget` / `resolve_separation`)
How far apart two assets sit in a Network, **and** how far is too far. An open family carried by `NetworkEstimator.sep`, with two members: `HopCount(; n = 1)`, the number of graph edges on the shortest path, budgeted in edges; and `PathLength(; dmax = nothing)`, the summed distance along that path, budgeted in the Distance Estimator's units. Three kernels: `separation_matrix(sep, nte, X)` returns the dense `assets × assets` separations, `separation_budget(sep, nte, d)` resolves the budget, and `resolve_separation(sep, nte, X)` replaces a **Separation Budget Rule** (below) by the value it computes.
The two questions are one type because they share a **unit**: a budget stated apart from the rule that measures it has no interpretation, which is why `n` moved off `NetworkEstimator` onto `HopCount` and why a member measuring something continuous carries a budget in its own units instead. The family sits on the Network Estimator rather than on any consumer because *every* consumer of a graph needs to know which pairs it relates — `phylogeny_matrix` and the phylogeny constraints, both `clusterise` methods, `Proximity` — and the constraint path receives nothing but the estimator, so a rule living on the feature producer would be structurally invisible to it.
`HopCount.n` stays an `Integer` rather than widening to a `Real`: three readers use `0:(nte.sep.n)` as a **matrix-power count**, where `0:1.5` silently drops a power instead of failing — which is why those three readers admit `HopCount` alone and refuse a `PathLength` loudly. The family name is deliberately unqualified — a taxonomy depth is a separation too — so a non-graph member would arrive without a re-parenting.
`separation_budget`'s third argument is the separation **matrix** rather than a scalar diameter, because finding the largest finite entry *is* an `assets²` reduction and `HopCount` must not pay for one it ignores. `PathLength` does the reduction itself: `dmax = nothing` means the observed diameter, and a chosen `dmax` is **clamped** to it. The clamp truncates nothing — no pair sits beyond the diameter — so it is a scale-top correction visible only through `LinearDecay`, the one decay that reads the budget.
*Avoid*: reading an unreachable pair's entry as a number. `separation_matrix` passes the underlying routine's sentinel through unrepaired (`typemax(Int)` for `HopCount`, `Inf` for `PathLength` over `Float64` weights), so a consumer compares against the budget **first**, and that comparison must short-circuit. Both sentinels punish an unguarded read, for opposite reasons: `ReciprocalDecay` overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError`, while `LinearDecay` at `Inf` returns `-Inf`, which sorts every unreachable pair below every reachable one without raising anything.
*Avoid*: taking a shortest path over a PMFG's own weights. Those weights are **similarities**, so the path minimises total similarity and prefers the route through the weakest links. `PathLength` therefore reads `calc_distance_weighted_graph`, which re-weights the PMFG's structure with `D` — legitimate because every similarity algorithm is a strictly decreasing function of `D`, so `D` is the selecting quantity's monotone preimage rather than a foreign quantity. The backwards answer correlates `0.95` to `0.97` with the right one, so nothing about it looks wrong.

**Separation Budget Rule** (`HopCountAlgorithm` / `PathLengthAlgorithm` / `resolve_separation`)
A callable standing where a **Separation**'s budget number goes, called as `rule(nte, X; dims, kwargs...)` when the data is in hand. Two families, one per unit, because their return obligations differ: a `HopCountAlgorithm` in `HopCount.n` must answer with an `Integer`, a `PathLengthAlgorithm` in `PathLength.dmax` with a `Number`. A bare `Function` is admitted in either field under the same obligation. One member each: `HopCountQuantile(; q = 0.25)` and `PathLengthQuantile(; q = 0.25)`, placing the budget at the `q`-quantile of the observed separations. The field aliases are `HopCountValue` and `Option{PathLengthValue}`; the dispatch aliases naming the dynamic halves are `HopCountRule` and `PathLengthRule`.
It exists because a budget cannot always be named in advance — a cross-validation fold and a meta optimiser's subproblem both refit the graph, so a number tuned once is applied to graphs it was never tuned for. The choice a rule makes is **which quantity stays put**: a stated `dmax` holds the *radius* still and lets the related-pair count move with the graph, a quantile rule holds the *count* still and lets the radius move. Neither is stable in both senses, because the graph is refitted either way. Measured over four folds of one year, a `dmax` fixed at the whole-sample quarter-quantile relates `84`, `110`, `96` and `110` of `380` pairs while the rule relates `96` in every fold.
`nothing` is deliberately outside `PathLengthValue` and reached only through the `Option`: it means the observed diameter, which is a budget the caller *states* rather than one a rule computes, so `resolve_separation`'s check is a plain `isa(dmax, Number)`.
*Avoid*: expecting a rule to be checked when it is stored. A functor's return type is not part of its signature, so the obligation is a run-time check inside `resolve_separation`, which then feeds the answer back through the ordinary constructor so it meets the same validation a stated budget meets. Storing `HopCount(; n = rule)` where the rule will answer `0` is legal; resolving it throws.
*Avoid*: resolving inside `separation_budget`. That kernel takes the separation **matrix** rather than `X`, deliberately, so it is the one place a rule *cannot* be answered — it refuses an unresolved member instead of returning a function. Every shipped consumer resolves first: `phylogeny_matrix`, both `clusterise` methods, and `phylogeny_features` for `Proximity`.
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
*Avoid*: passing a weight matrix through a centrality algorithm's `args`. It is a second, undeclared channel, refused by `assert_centrality_args`. `Graphs.closeness_centrality` took one in its second positional slot and silently overrode the polarity; `Graphs.betweenness_centrality` takes one in its *third*, so a matrix in `args` bound to `vs` and overflowed the stack.

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
The per-asset quantity an ℓ1 uncertainty set is built around. Usually the expected return, but the construction is indifferent: a prior built on `StandardDeviationExpectedReturns` ranks on volatility instead. The radius `eps` decides how many assets the resulting portfolio holds — small radius, one asset; moderate, a quintile; large, all of them equally — which is what makes the *quintile* and *1/N* heuristics exact solutions of a robust optimisation problem rather than folk wisdom (ADR 0032). There is deliberately no quintile optimiser: the portfolios are a `MeanRisk` recipe over this set.
*Avoid*: using *characteristic* for a column of a **Feature Matrix** (§2), and vice versa. The two are near-synonyms held apart on purpose: a Characteristic Vector is a *single* per-asset quantity that this family builds an uncertainty budget around, and the word stays reserved for it (`CharacteristicUncertaintySet`); a Feature is *one column of many*, and the matrix's rows become a distance. `Characteristic…` naming was the first candidate for the feature work and was rejected on exactly this collision (ADR 0045).

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

### 4.3 JuMP-based

- **MeanRisk**: efficient-frontier trade-off between expected return and risk (Markowitz-style). The classic objective optimiser.
- **RiskBudgeting (RB)**: targets prescribed risk *contributions* per asset/factor rather than a return/risk trade-off. `AssetRiskBudgeting` / `FactorRiskBudgeting`; formulations `LogRiskBudgeting` (log-barrier), `MixedIntegerRiskBudgeting`.
- **RelaxedRiskBudgeting (RRB)**: convex relaxation — `BasicRelaxedRiskBudgeting`, `RegularisedRelaxedRiskBudgeting`, `RegularisedPenalisedRelaxedRiskBudgeting`.
- **FactorRiskContribution (FRC)**: optimises/targets risk contributions attributed to factors.
- **NearOptimalCentering (NOC)**: finds a robust interior point near the efficient frontier (`ConstrainedNearOptimalCentering` / `UnconstrainedNearOptimalCentering`).

**Returns estimators** (objective inputs): `ArithmeticReturn` (dot-product), `LogarithmicReturn` (geometric).

**Objective Functions**: `MinimumRisk`, `MaximumUtility`, `MaximumRatio` (Sharpe-type), `MaximumReturn`.

**Solver**
Wraps a single external backend (Clarabel, HiGHS, …): solver module, settings, name, and `check_sol` kwargs passed to JuMP's `is_solved_and_feasible` to validate a solution at runtime.

**JuMPOptimiser**
The shared JuMP model configuration: one or more `Solver`s (fallback chains) plus JuMP-level settings. The execution environment consumed by all JuMP-based optimisers.

**Model State**
The in-flight state of the JuMP model as it is being built — the shared variables, expressions, and scales that successive constraint/risk builders read and write. Reached only through its named interface, never by naming an entry directly; see Per-Build Risk State for which entries a nested build must keep to itself.

**Per-Build Risk State**
The part of Model State that belongs to *one* risk build rather than to the model as a whole, and so must not be shared with a build nested inside it. Two things qualify: anything that is a function of the weights being optimised — a nested build (risk tracking) shifts those weights, so the inner and outer answers genuinely differ — and any presence flag that gates a formulation choice for the build that set it. Everything else is a pure function of the Prior, identical in both builds, and is correctly shared: treating it as per-build would break sharing rather than protect it. The distinction decides correctness, not tidiness: getting it wrong makes a nested build silently read its parent's state.

**Model Assembly**
The fixed sequence in which a single-JuMP-model Optimisation Estimator's constraint and risk builders run to turn an empty model into a fully-constrained one — the steps between shaping the weight variables and setting the objective. Shared by MeanRisk, Risk Budgeting, Relaxed Risk Budgeting, Factor Risk Contribution, and constrained Near Optimal Centering; the per-optimiser parts (how weights are shaped, the objective, the solve) sit outside it. Distinct from Model State: Model State is the data the builders read and write, Model Assembly is the ordering of the builders themselves.

**Objective Penalty**
The accumulator through which every *soft* contribution reaches the objective — built-in regularisation, soft Turnover and Tracking, and user-supplied Custom Objective Terms alike. It exists because the objective's optimisation *sense* varies (some objectives are minimised, some maximised, and a ratio objective is either depending on the risk measure), so contributions are expressed once, sense-free, and the sense-correct factor is applied centrally when the accumulator is folded into the objective. A contribution always worsens the objective; a reward is a negative contribution. Distinct from a hard constraint, which bounds the feasible region rather than pricing a preference.

**Custom Term** (the extension point)
The user-facing escape hatch for expressing a preference the library does not already name: a Custom Objective Term prices one (through the Objective Penalty), a Custom Constraint mandates one. Both are supplied as inputs to the JuMPOptimiser and both are reached during Model Assembly, receiving the model under construction, the outer Optimisation Estimator, and the processed problem data. A term that names itself as one but supplies no builder is an error, not a no-op.

### 4.4 Constraints

**Universe Sets**
A user-defined mapping of names to named groups (sectors, countries), groups, or unique-member groups. Always user-defined, though code can generate them (e.g. from Phylogeny). The foundational input to nearly all Constraint Generation.
It declares **every axis it carries**: `xkey`/`uxkey` for assets (defaults `"nx"`/`"ux"`), `fkey`/`ufkey` for factors (`"nf"`/`"uf"`), and `zkey` for features (`"nz"`). Assets are the *primary* axis — required, and the one a View slices; the factor and feature axes are optional, and a consumer that needs one and does not find it fails at the point of need rather than at construction. A key prefixed with an axis key is a **partition** of that axis (same length as its universe); a key prefixed with the axis's unique key is the partition's distinct entries. Anything else is a plain group, expanded by name and **axis-blind** — which is why factor groups need no machinery of their own, and why a factor constraint naming an asset group degrades to unknown-name warnings rather than to an error.
The feature axis is the exception that proves the prefix rule: it has **no** prefix convention and no unique-entry sibling, because nothing is written *over* it. A graded Feature Program's taxonomy keys are `xkey`-prefixed and asset-length, and its columns are named out of the flat list at `zkey` directly. Its only rule is `allunique`.
*Declaring* an axis is what lets a View leave its entries alone, but for two different reasons. Factors pass through because an asset index is *meaningless* on them (before ADR 0047 a factor universe was carried in the *asset* key, so the exemption had to be maintained by hand, per field, by omitting an annotation). Features pass through because the axis is **authored** — some of its nodes *are* assets — which is what makes `size(Z, 2)` fold-invariant, at the accepted cost of a dropped asset's node surviving as an all-zero column (ADR 0045, amended).
A universe is an **ordered** declaration: position is the only link between a name and a column of the data, so a universe that lists the right names in the wrong order attaches every constraint, bound and group to the wrong column and still solves. Each axis is therefore checked against the corresponding axis of the returns data — name for name, in order — where both sides exist, and a disagreement is an error rather than a silently different answer.

**Constraint Space** (ADR 0047)
The basis a constraint's names resolve in. Assets are not a member of this family: the asset basis is the *absence* of a re-basis, spelled `LinearConstraintEstimator`. `FactorSpace` is the only member today, and `ExposureConstraintEstimator` is the wrapper that declares one — it decorates whatever `lcse` already accepts, so parsing and group expansion are inherited rather than re-implemented.
**A constraint can be re-based iff it is a linear form in `w`.** Cardinality and Threshold rows index the binary held-indicators, not `w`, so a projected row is neither integral nor an index into them; Weight Bounds are a per-asset box, and a factor box is a linear constraint rather than a bound. Their exclusion is a property of the constraint, not a gap — and it is enforced by *type*: those slots admit only the unmarked estimator, so an illegal space cannot be written down.
*Avoid*: reading a Constraint Space as a change to the **problem**. It is a change of coordinates on the same feasible region — the projection happens during Constraint Generation and what reaches the optimiser is an ordinary asset-space linear constraint.

**Factor Exposure Constraint**
A linear constraint on a portfolio's factor weights `w_f = Mᵀw`, written in factor names and projected through the loadings. Under a dimension-reducing regression `M` and `L` are the two sides of one projection, each recoverable from the other, so which one a consumer reads is decided by what it is doing rather than by convention: a *risk decomposition* reads `L`, the orthogonal reduced basis its covariance was estimated in, while a *constraint* reads `M`, because a constraint is written down and only `M`'s columns carry names a user can put in an equation.
It is the one constraint that **cannot** be precomputed by hand without going stale: the loadings are refit per fold, so a row computed once against a full-sample `M` describes an exposure the fold's model does not have. Requesting one against a prior that carries no regression is an error, never a dropped row — dropping every row of a factor mandate yields a feasible portfolio with none of the requested exposure.

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

### Settings & combination

- **RiskMeasureSettings / HierarchicalRiskMeasureSettings / MaxRiskMeasureSettings**: per-measure config (scaling, bounds, direction).
- **Scalariser**: combines multiple risk measures into one scalar — `SumScalariser`, `MaxScalariser`, `MinScalariser`, `LogSumExpScalariser`.
- **Frontier / FrontierBoundEstimator**: efficient-frontier point count and bound transforms (`LinearBound`, `SquareRootBound`, `SquaredBound`).

### Dispersion / moment measures

- **Variance**, **StandardDeviation**, **UncertaintySetVariance** (robust). Optimisation formulations: `QuadRiskExpr`, `SOCRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`.
- **LowOrderMoment** (`FirstLowerMoment`, `MeanAbsoluteDeviation`, `SecondMoment`, `EvenMoment`) and **HighOrderMoment** (`ThirdLowerMoment`, `FourthMoment`, `StandardisedHighOrderMoment`).
- **Kurtosis**, **Skewness**, **NegativeSkewness**, **VarianceSkewKurtosis** (composite), **ThirdCentralMoment**.
- **MedianAbsoluteDeviation (MAD)**: centred by `MedianCentering` or `MeanCentering`.

### Quantile / tail measures (XatRisk)

- **Value-at-Risk (VaR)**: `ValueatRisk` (formulations `MIPValueatRisk`, `DistributionValueatRisk`), `ValueatRiskRange`; drawdown forms `DrawdownatRisk`, `RelativeDrawdownatRisk`.
- **Conditional (CVaR / Expected Shortfall)**: `ConditionalValueatRisk`, `…Range`, DR forms, drawdown `ConditionalDrawdownatRisk` (CDaR) and relatives.
- **Entropic (EVaR)**: `EntropicValueatRisk`, `…Range`, `EntropicDrawdownatRisk` (EDaR), relatives.
- **Relativistic (RVaR)**: `RelativisticValueatRisk`, `…Range`, `RelativisticDrawdownatRisk` (RLDaR), relatives.
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
