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

**Factory**
A generic configuration mechanism for immutable structs — they may be Estimator, Result, or Algorithm structs. Because all structs are immutable and the library relies heavily on composition, `factory` is the standard way to propagate runtime-computed values (moments from a Prior Result, observation weights, previous portfolio weights, etc.) down through a composed struct tree. It takes a struct and runtime data and returns a new, fully-configured struct of the same type.

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
The central data structure carrying all return series through the library. Fields: `X` (asset returns matrix, observations × assets), `F` (factor returns matrix), `B` (benchmark returns matrix), `nx`/`nf`/`nb` (asset/factor/benchmark names), `ts` (timestamps), `iv` (implied volatility), `ivpa` (implied volatility premium). Produced by `prices_to_returns` from raw price data. Every Prior Estimator and Optimisation Estimator consumes a `ReturnsResult`.

**PricesResult**
The container of aligned, time-indexed *price-level* series — asset prices plus optional factor, benchmark, and implied-volatility series — the prices-level mirror of `ReturnsResult`. The input to price Preprocessing Estimators and to prices-to-returns conversion, and the type that defines timestamp-window slicing for pipeline Cross-Validation. Like `ReturnsResult`, user-constructible yet classified as a Result (produced by filtering steps); `FiniteAllocationInput` remains the glossary's only data-as-Estimator deviation.

**Implied Volatility**
A forward-looking estimate of how much an asset's price is expected to fluctuate, derived from current options contract prices using models such as Black-Scholes. Stored as `iv`. Not a historical measurement.

**Implied Volatility Premium (VRP)**
The difference between Implied Volatility (market expectation) and realised volatility (historical outcome). Typically positive because investors pay a premium to hedge downside risk. Stored as `ivpa`.

## 3. Statistics

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
- **WindowedExpectedReturns**: restricts computation to a rolling or indexed observation window.

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
- **Windowed** variants (`WindowedCovariance`, `WindowedVariance`): rolling/indexed observation window.

### 3.3 Higher-Order Moments

**Coskewness** / **Cokurtosis**
Third- and fourth-order co-moment tensors (with `FullMoment`/`SemiMoment` variants and windowed forms `WindowedCoskewness`, `WindowedCokurtosis`). Feed high-order priors and higher-moment risk measures.

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

### 3.7 Distance

**Distance Matrix**
A symmetric, zero-diagonal matrix where larger values mean less relatedness. Usually derived from correlation, but any matrix satisfying that convention qualifies. The fundamental input to Phylogeny.

**Distance Estimator**
Converts correlation/returns into a distance. Algorithms: `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `VariationInfoDistance` (variation of information), `CanonicalDistance`. `Distance` is the configurable container (supports a generalised power distance); `DistanceDistance` computes a distance-of-distances.

### 3.8 Phylogeny

**Phylogeny**
The characterisation of asset relationships derived from a Distance Matrix. The biological metaphor is intentional: related assets cluster like species on an evolutionary tree. Three sub-concepts:

- **Clustering**: groups related assets into a dendrogram. `ClustersEstimator` with `HClustAlgorithm` (hierarchical linkage), `DBHT` (Direct Bubble Hierarchical Tree — PMFG-based, root methods `UniqueRoot`/`EqualRoot`, similarity transforms, `LoGo` sparse inverse), or `KMeansAlgorithm` (non-hierarchical). Optimal cluster count via `OptimalNumberClusters` using `SecondOrderDifference` or `SilhouetteScore`.
- **Network**: assets as graph nodes/edges. `NetworkEstimator` builds a minimum spanning tree — `KruskalTree`, `BoruvkaTree`, `PrimTree`.
- **Centrality**: which assets are most influential — `BetweennessCentrality`, `ClosenessCentrality`, `DegreeCentrality`, `EigenvectorCentrality`, `KatzCentrality`, `Pagerank`, `RadialityCentrality`, `StressCentrality`.

`PhylogenyResult` carries the resulting matrix/vector. Required input to clustering-based Optimisation Estimators and to phylogeny constraints.

### 3.9 Uncertainty Sets

**Uncertainty Set**
A robust optimisation construct that sits alongside a Prior in JuMP-based optimisers: the Prior gives the central estimate (mean, covariance) and the Uncertainty Set bounds how far off it might be; the optimiser then protects against the worst case within the set. Always a robust-optimisation concept. Shapes and constructors:

- **Box** (`BoxUncertaintySet`): independent per-parameter bounds.
- **Ellipsoidal** (`EllipsoidalUncertaintySet`, `Mu…`/`Sigma…` classes): joint confidence region; scaling parameter `k` from `NormalKUncertaintyAlgorithm`, `GeneralKUncertaintyAlgorithm`, or `ChiSqKUncertaintyAlgorithm`.
- **ℓ1 / cross-polytope** (`L1UncertaintySet`): one error budget `eps` shared across all assets and both signs, optionally scaled per asset by `sd`; worst case `mu'w − eps·‖sd ⊙ w‖∞`. **Mean-only** — it bounds a *characteristic* vector and has no covariance analogue.
- **Signed ℓ1** (`SignedL1UncertaintySet`): a budget per error sign (`ep`, `em`); worst case `mu'w − ep·[maxᵢ(−sdᵢwᵢ)]₊ − em·[maxᵢ(sdᵢwᵢ)]₊`. Not the joint set with `ep == em` — the joint set shares one budget across signs (`max(t₊,t₋)`), this one spends a budget per sign (`ep·t₊ + em·t₋`); they agree only when `w` is single-signed.
- **Estimators**: `DeltaUncertaintySet` (delta bounds), `NormalUncertaintySet` (normality assumption), `ARCHUncertaintySet` (bootstrap for time series — `StationaryBootstrap`, `CircularBootstrap`, `MovingBootstrap`), `CharacteristicUncertaintySet` (the ℓ1 family; shape via `L1UncertaintySetAlgorithm`/`SignedL1UncertaintySetAlgorithm`, whose `scaled` flag selects equal-weight vs inverse-volatility behaviour).

**Characteristic Vector**
The per-asset quantity an ℓ1 uncertainty set is built around. Usually the expected return, but the construction is indifferent: a prior built on `StandardDeviationExpectedReturns` ranks on volatility instead. The radius `eps` decides how many assets the resulting portfolio holds — small radius, one asset; moderate, a quintile; large, all of them equally — which is what makes the *quintile* and *1/N* heuristics exact solutions of a robust optimisation problem rather than folk wisdom (ADR 0032). There is deliberately no quintile optimiser: the portfolios are a `MeanRisk` recipe over this set.

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
The in-flight state of the JuMP model as it is being built — the shared variables, expressions, and scales that successive constraint/risk builders read and write. Accessed through a named interface rather than raw model keys.

**Model Assembly**
The fixed sequence in which a single-JuMP-model Optimisation Estimator's constraint and risk builders run to turn an empty model into a fully-constrained one — the steps between shaping the weight variables and setting the objective. Shared by MeanRisk, Risk Budgeting, Relaxed Risk Budgeting, Factor Risk Contribution, and constrained Near Optimal Centering; the per-optimiser parts (how weights are shaped, the objective, the solve) sit outside it. Distinct from Model State: Model State is the data the builders read and write, Model Assembly is the ordering of the builders themselves.

### 4.4 Constraints

**Asset Sets**
A user-defined mapping of assets to named groups (sectors, countries), groups of assets, or unique-member groups. Always user-defined, though code can generate them (e.g. from Phylogeny). The foundational input to nearly all Constraint Generation.

**Constraint Generation**
User-facing utilities converting high-level specifications into the numeric form optimisers need (linear systems or bounds). Built on Asset Sets. Concepts:

- **LinearConstraint / LinearConstraintEstimator**: parses equations into `A·w ≤/= B` (`PartialLinearConstraint`, `ParsingResult`).
- **WeightBounds**: per-asset lower/upper bounds (scalar = same for all; vector = per asset). `UniformValues` gives `1/N…1`. Group-level sum bounds are linear constraints, not Weight Bounds.
- **RiskBudget**: per-asset/group risk-contribution targets.
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
An input may be time-dependent **iff it is problem definition** — what is being solved: priors, returns, risk measures, objectives, constraints, weight finalisers, asset sets, warm starts, fallbacks, the meta-optimisers' inner/outer optimisers — never **execution control** — how it is solved: solvers, JuMP scaling factors, RNGs and seeds, strictness flags, a meta's own inner cross-validation scheme. The widened constructor signatures encode the criterion and are the admissibility table.
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
- **Relativistic (RVaR)**: `RelativisticValueatRisk`, `…Range`, `RelativisticDrawdownatRisk` (RDaR), relatives.
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
