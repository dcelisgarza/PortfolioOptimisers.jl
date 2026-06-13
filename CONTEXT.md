# PortfolioOptimisers.jl — Domain Glossary

The library is organised as a pipeline:

> data → moments → prior → optimisation → post-processing

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
The sub-selection counterpart to Factory. Where Factory propagates runtime *values* down a composed struct tree, a View propagates an index *selection*: it restricts an Estimator, Algorithm, or Result (or an array of them) to a subset of assets — or, for returns data, observations — and returns a new struct of the same type with every data-bearing field and composed child consistently sub-selected. Used wherever the library operates on part of the problem rather than the whole: meta-optimisers (Subset Resampling, Nested Clustered), Cross-Validation, and windowed moment estimators. Like Factory, it relies on composition — each struct declares which of its fields participate, and the selection is threaded recursively down the tree.

**Vector-to-Scalar Reducers**
Small reusable Algorithms that collapse a vector of reals to a scalar, reused throughout the library: `MinValue`, `MaxValue`, `MeanValue`, `MedianValue`, `ModeValue`, `StdValue`, `VarValue`, `SumValue`, `ProdValue`, `StandardisedValue` (weighted mean ÷ weighted std). Most accept optional observation weights.

**Full vs Semi**
A pervasive Algorithm distinction in moment estimation: `Full` includes all deviations; `Semi` includes only observations below a target (downside-only). Drives the split between symmetric and downside risk/moment measures.

## 2. Data

**ReturnsResult**
The central data structure carrying all return series through the library. Fields: `X` (asset returns matrix, observations × assets), `F` (factor returns matrix), `B` (benchmark returns matrix), `nx`/`nf`/`nb` (asset/factor/benchmark names), `ts` (timestamps), `iv` (implied volatility), `ivpa` (implied volatility premium). Produced by `prices_to_returns` from raw price data. Every Prior Estimator and Optimisation Estimator consumes a `ReturnsResult`.

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
Third- and fourth-order co-moment tensors (with `Full`/`Semi` variants and windowed forms `WindowedCoskewness`, `WindowedCokurtosis`). Feed high-order priors and higher-moment risk measures.

### 3.4 Regression (factor modelling)

**Regression Estimator**
Builds a factor model mapping factor returns to asset returns; underpins factor priors. Families:

- **StepwiseRegression**: greedy feature selection — `Forward` (selection) or `Backward` (elimination), driven by a criterion: `PValue`, `AIC`, `AICC`, `BIC`, `RSquared`, `AdjustedRSquared`.
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
- **Estimators**: `DeltaUncertaintySet` (delta bounds), `NormalUncertaintySet` (normality assumption), `ARCHUncertaintySet` (bootstrap for time series — `StationaryBootstrap`, `CircularBootstrap`, `MovingBootstrap`).

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

**Weight Finaliser**
Post-solve adjustment forcing weights into the feasible region: `IterativeWeightFinaliser` (projection) or `JuMPWeightFinaliser` (re-solve).

**Turnover**
Measures/constrains change relative to a reference. Dynamic (vs previous weights, needs them at runtime) or fixed (vs static target). Soft penalty or hard constraint.

**Tracking Error**
Measures/constrains closeness to a benchmark/target. `WeightsTracking` (vs a reference weight vector — needs a weight vector, not necessarily *previous*) or `ReturnsTracking` (vs a benchmark return series). Return formulations: `L1Tracking`, `L2Tracking`, `SquaredL2Tracking`, `LpTracking`, `LInfTracking` (norms) and `IndependentVariableTracking` / `DependentVariableTracking` (variable). Soft penalty or hard constraint.

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
- **Relativistic (RVaR)**: `RelativisticValueatRisk`, `…Range`, `RelativisticDrawdownatRisk` (RDDaR), relatives.
- **Power Norm**: `PowerNormValueatRisk` (PNVaR), `…Range`, `PowerNormDrawdownatRisk`, relatives.

**OWA (Ordered Weights Array)**
Weighted sum of *sorted* return realisations; weights generated by an Algorithm composing L-moments (`MaximumEntropy` with `ExponentialConeEntropy`/`RelativeEntropy`, `MinimumSquaredDistance`, `MinimumSumSquares`, `NormalisedConstantRelativeRiskAversion`). Formulations `ExactOrderedWeightsArray` (LP) / `ApproxOrderedWeightsArray` (p-norm). `OrderedWeightsArray`, `OrderedWeightsArrayRange`. Flexible enough to approximate CVaR and higher-moment criteria.

### Drawdown measures

- **AverageDrawdown**, **UlcerIndex** (RMS drawdown), **MaximumDrawdown** — each with a `Relative…` hierarchical form.

### Other measures

- **BrownianDistanceVariance (BDVar)**: distance-covariance-based dispersion (`NormOneConeBrownianDistanceVariance`, `IneqBrownianDistanceVariance`).
- **WorstRealisation**: worst single-period loss.
- **Range**: spread between best and worst realisations.
- **TurnoverRiskMeasure**: turnover expressed as a risk quantity.
- **TrackingRiskMeasure**: benchmark deviation measured as a **norm** of the portfolio-vs-benchmark difference.
- **RiskTrackingRiskMeasure**: benchmark deviation measured through a configurable **risk measure `r`** applied to the portfolio-vs-benchmark difference (a `WeightsTracking` benchmark + risk measure + `VariableTracking` algorithm) — i.e. tracking the *risk* relative to a benchmark rather than the raw norm.
- **EqualRiskMeasure**: enforces equal risk contributions (hierarchical).
- **RiskRatioRiskMeasure** / **NonOptimisationRiskRatioRiskMeasure**: ratio-form measures for hierarchical use.

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
