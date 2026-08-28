# PortfolioOptimisers.jl — Domain Glossary

A Julia portfolio-optimisation library. The domain is organised as a workflow:

> data → moments → prior → optimisation → post-processing

with a small set of cross-cutting abstractions (Estimator / Algorithm / Result / Factory) that everything is built from. This file is a glossary and nothing else: it names the concepts and fixes the words for them. Decisions and their reasoning live in `docs/adr/`.

## 1. Core Abstractions

**Estimator**
A configuration object encoding a statistical or mathematical method together with its hyperparameters, and the primary input to the library's user-facing functions. An Estimator says *how to compute for whatever input it is given*, and so never holds a Result.

**Algorithm**
A type held inside an Estimator that selects or modifies its computational behaviour. Consumed through an Estimator, never used on its own.

**Result**
A plain data struct holding the computed output of a function applied to an Estimator: *the answer for the input it was computed on*. Never callable.

**Choice Surface**
The set of things a caller picks when specifying a problem: the concrete leaf Estimators and Algorithms. Results are what comes back, never what is chosen.

**Factory**
The mechanism that propagates runtime-computed values down a composed struct tree, returning a new, fully-configured struct of the same type.

**Deferred Quantity**
A moment Estimator standing where a prior-derived value goes, so the slot admits either the value or the method that computes it. Factory resolves it against the optimisation's own Prior.

**View**
An index selection propagated down a composed struct tree, restricting an Estimator, Algorithm or Result to a subset of assets or observations.
*Avoid*: using it for a **Tail View** (§3.6), which is a statement about the distribution rather than a selection.

**Propagation Channel**
One generated method of `@propagatable`, and the field tag that opts a field into it. Three exist: factory, view, and prior.

**Pipeline**
An Estimator that reifies an end-to-end workflow as an ordered list of optionally-named steps, executed left to right over a Pipeline Context.
*Avoid*: Workflow, Workbench; and using "pipeline" for the library's informal stage ordering.

**Pipeline Context**
The accumulating blackboard threaded through a Pipeline's steps: a set of coarse typed slots that each step reads from and writes to.

**Routing Target**
The destination a computed Pipeline Context slot is delivered to inside the optimisation step. Finer than a slot: a slot names a stage of the workflow, a target names a destination field.

**Data Slot**
A Pipeline Context slot whose write changes the asset universe — prices and returns — as opposed to a *derived slot* (prior, phylogeny, uncertainty, constraints, weights) computed from the data.

**Preprocessing Estimator**
The family of Estimators that transform price or returns data inside a Pipeline. Fitting one yields a Result carrying its fitted state, which is then applied to unseen data.

**Holdout Split**
The evaluation protocol that reserves the tail of the time-ordered observations as a test window and trains on the head.
*Avoid*: Validation Split (nothing is tuned on the held-out window).

**Embargo**
The deliberate gap between the training and test windows of a Holdout Split, belonging to neither.

**Asset Selector**
The Preprocessing Estimator subfamily that restricts the asset universe from returns data. Its fitted state is the set of surviving asset columns.
*Avoid*: Filter, Screen (reserved for `MissingDataFilter`, which drops and never selects).

**Selection Rule**
The Algorithm consumed by a `ScoreSelector` that turns per-asset scores into a keep-mask. *Literal* rules compare raw scores against absolute bounds; *ordinal* rules consult orientation and take counts or fractions from each end.

**Redundancy Algorithm**
The Algorithm consumed by a `RedundancySelector`, deciding which assets duplicate information others already carry. It returns a keep-mask, not a partition.

**Trust-neither tie policy**
The convention that when two assets are indistinguishable under the criterion being applied, the library keeps neither.

**Vector-to-Scalar Reducers**
Small reusable Algorithms collapsing a vector of reals to a scalar: `MinValue`, `MaxValue`, `MeanValue`, `MedianValue`, `ModeValue`, `StdValue`, `VarValue`, `SumValue`, `ProdValue`, `StandardisedValue`.

**LxNorm error family**
The norm forms used as constraints, as risk-measure targets, and in entropy pooling: `L1Norm`, `L2Norm`, `SquaredL2Norm`, `LpNorm`, `LInfNorm`.

**FullMoment vs SemiMoment**
The pervasive Algorithm distinction in moment estimation: `FullMoment` includes all deviations, `SemiMoment` only those below a target. It drives the split between symmetric and downside measures.

## 2. Data

**ReturnsResult**
The central data structure carrying all return series through the library: asset, factor and benchmark returns, their names, timestamps, implied volatility, and the Feature Matrix. Produced by `prices_to_returns`.

**PricesResult**
The container of aligned, time-indexed price-level series: the prices-level mirror of `ReturnsResult`, and the input to price Preprocessing Estimators.

**Feature Matrix** (`Z` / `nz`)
An assets × features matrix of per-asset quantities that are *not* return series — a sector taxonomy, a fundamentals or ESG panel, factor loadings, a graph neighbourhood. It is exogenous data, letting the clustering and network stack see structure the returns do not encode.
*Avoid*: Characteristic (see **Characteristic Vector**, §3.9); and reading "feature" as *factor*, which is a return series.

**Feature Matrix Estimator**
A producer turning something the library already computes, or a classification the caller supplies, into a Feature Matrix: `RegressionFeatures`, `AssetSetsFeatures`, `PhylogenyFeatures`.

**Feature Program**
An ordered, last-wins list of authored edges resolving into a Feature Matrix. Where the group-key form stacks partitions and derives its axis, a Program writes cells into an axis the caller declared.

**Implied Volatility**
A forward-looking estimate of an asset's expected price fluctuation, derived from current option prices. Not a historical measurement.

**Implied Volatility Premium (VRP)**
The difference between Implied Volatility and realised volatility, typically positive because investors pay to hedge downside risk.

## 3. Statistics

**Windowed Estimator**
A moment estimator restricting an inner moment estimator to a sub-window of the observations, then delegating. Windowing decides *which observations are seen*, never how the moment is computed from them.

### 3.1 Expected Returns (Moments)

**Expected Returns Estimator**
Computes a per-asset mean-return vector. Variants:

- **SimpleExpectedReturns**: sample mean, with optional observation weights.
- **ShrunkExpectedReturns**: shrinks the sample mean toward a target. Targets: `GrandMean`, `VolatilityWeighted`, `MeanSquaredError`. Algorithms: `JamesStein`, `BayesStein`, `BodnarOkhrinParolya`. The target and the algorithm are independent choices, so the estimator has nine combinations.
- **EquilibriumExpectedReturns**: implied (equilibrium) returns via reverse optimisation, Π = λ·Σ·w_eq. The Black-Litterman equilibrium anchor.
- **ExcessExpectedReturns**: returns net of a reference/risk-free rate.
- **MedianExpectedReturns**: (weighted) per-asset median.
- **StandardDeviationExpectedReturns** / **VarianceExpectedReturns**: the asset standard deviations / variances, where a "return" slot should carry dispersion.
- **CustomValueExpectedReturns**: user-supplied per-asset values.
- **WindowedExpectedReturns**: the Windowed Estimator for expected returns.

### 3.2 Covariance & Variance (Moments)

**Covariance Estimator**
Computes an asset covariance (and correlation) matrix. Core wrappers: `Covariance`, `GeneralCovariance`, `SimpleVariance`, `CorrelationCovariance`. Robust and specialised families:

- **Gerber** (`Gerber0/1/2`): co-movement counting that ignores small noise-level moves below a threshold.
- **GerberIQ**: Gerber with temporal lookback/decay and asset-volatility-scaled thresholds (`BasicGerberIQ`, `PartialGerberIQ`, `FullGerberIQ`).
- **Smyth-Broby** (`SmythBroby0/1/2`, `…Gerber*`, `…Count*`): Gerber-family extensions weighting co-movements by magnitude and/or vote counts.
- **DistanceCovariance**: covariance derived from a distance metric between assets.
- **LowerTailDependenceCovariance**: dependence in the joint lower tail (crash co-movement).
- **RankCovariance**: rank-correlation based — `KendallCovariance` (τ), `SpearmanCovariance` (ρ).
- **MutualInfoCovariance**: dependence via mutual information, over a histogram binning Algorithm.
- **RegimeAdjustedExpWeightedVariance/Covariance**: online exponentially-weighted estimators rescaled by a detected market regime state.
- **ImpliedVolatility** (covariance): scales covariance using implied volatility.
- **DenoiseCovariance / DetoneCovariance / ProcessedCovariance**: wrap another covariance estimator and apply matrix processing.
- **PortfolioOptimisersCovariance**: composite estimator bundling covariance estimation with post-processing.
- **WindowedCovariance** / **WindowedVariance**: the Windowed Estimators for covariance and variance.

### 3.3 Higher-Order Moments

**Coskewness** / **Cokurtosis**
Third- and fourth-order co-moment tensors, with `FullMoment`/`SemiMoment` and Windowed variants. They feed high-order priors and higher-moment risk measures.

### 3.4 Regression (factor modelling)

**Regression Estimator**
Builds a factor model mapping factor returns to asset returns, underpinning factor priors. Families:

- **StepwiseRegression**: greedy feature selection — `ForwardSelection` or `BackwardElimination`, driven by a criterion (`PValue`, or one of the symbols `:aic`, `:aicc`, `:bic`, `:r2`, `:adjr2`).
- **DimensionReductionRegression**: regression on reduced factors — targets `PCA`, `PPCA`.
- **Regression target** models: `LinearModel`, `GeneralisedLinearModel` (GLM).

### 3.5 Matrix Processing

**Denoising**
Removal of statistical noise from a covariance/correlation matrix by spectral thresholding (Random Matrix Theory). Algorithms: `SpectralDenoise`, `FixedDenoise`, `ShrunkDenoise`.

**Detoning**
Removal of the largest principal components (market modes) from a correlation matrix, isolating asset-specific correlation.

**Posdef**
Projection of a matrix to the nearest positive definite matrix.

**Matrix Processing**
The composing estimator applying a sequence of post-processing steps — Posdef, Denoising, Detoning, and a custom algorithm — to a covariance/correlation matrix.

### 3.6 Prior

**Prior**
The full statistical summary of asset returns that feeds an optimiser. A Prior Estimator bundles Moment Estimators and may adjust them using user assumptions. The name derives from "prior statistics", not from Bayesian inference.

Structural axes: **low-order** is mean plus covariance, **high-order** adds coskewness and cokurtosis; the **data dependency** suffixes `_A` / `_F` / `_AF` say whether factor returns are required. Concrete estimators:

- **EmpiricalPrior**: moments computed directly from returns.
- **FactorPrior** / **HighOrderFactorPriorEstimator**: moments reconstructed through a factor model.
- **HighOrderPriorEstimator**: empirical high-order prior.
- **Black-Litterman family**: blends market-equilibrium priors with investor views — `BlackLittermanPrior`, `BayesianBlackLittermanPrior`, `FactorBlackLittermanPrior`, `AugmentedBlackLittermanPrior`, with `BlackLittermanViews` as the views container.
- **EntropyPoolingPrior**: re-weights scenario probabilities to satisfy views with minimal relative entropy.
- **MeucciEntropyPoolingPrior**: the earlier entropy pooling estimator, whose CVaR route root-finds the Value at Risk level rather than writing a formulation.
- **OpinionPoolingPrior**: consensus across several priors — `LinearOpinionPooling`, `LogarithmicOpinionPooling`.

**Group Pair View**
An entropy pooling correlation or covariance view whose two sides are groups rather than single assets, `"(gA, gB) == 0.35"`. The two groups must be of equal length, and the view **spans** one asset pair per position: it emits one constraint row per pair, and a `prior(gA, gB)` reference inside it resolves to that pair's own prior value. See ADR 0079.
*Avoid*: reading it as a statement about one summary of the correlation block; no such aggregate is defined.

**Tail View**
A view on a quantile risk measure of the posterior — CVaR or EVaR — as opposed to a view on a moment. It is the one view family that is not a linear function of the posterior probabilities.
*Avoid*: confusing it with a **View** (§1), which is the index-selection mechanism.

**Risk-Free Shift**
The one place a Black-Litterman prior reads its `rf` field. The update blends the prior mean against the view returns, so it runs on the total-return scale those are written on. A prior mean taken from a wrapped estimator is on that scale already and is left alone; an equilibrium mean from `EquilibriumExpectedReturns` is a bare risk premium, and the rate converts it *before* the update. A member with no equilibrium branch has nothing to convert, and adds the rate to the posterior asset mean instead.
*Avoid*: calling it a round trip. Nothing subtracts the rate, and the update is affine rather than a translation, so a conversion and its inverse around the update would not cancel.

**Factor Lift**
The hop from the factor axis to the asset axis: fit the loadings, rebuild the returns through them, project the factor moments, and optionally add a residual block.
*Avoid*: calling it a **Forward**, which carries a wrapped Result along its own axis instead of crossing axes.

**Original Returns Matrix**
The returns matrix the caller supplied, as distinct from the one a Prior Result asserts. The two differ only where a factor prior overwrites the returns with its reconstruction.

### 3.7 Distance

**Distance Matrix**
A symmetric, zero-diagonal matrix in which larger values mean less relatedness. The fundamental input to Phylogeny.

**Distance Estimator**
Converts correlation or returns into a distance: `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `VariationInfoDistance`, `CanonicalDistance`. `Distance` is the configurable container and `DistanceDistance` computes a distance-of-distances.

**Feature Distance**
The one Distance Estimator measuring something other than returns: it applies a metric to the rows of a Feature Matrix, so the resulting hierarchy expresses exogenous structure.

**Similarity Matrix Algorithm**
The transform turning a Distance Matrix into a similarity matrix: `MaximumDistanceSimilarity`, `ExponentialSimilarity`, `GeneralExponentialSimilarity`, `ComplementSimilarity`, `AngularSimilarity`.

**Non-Negative Similarity Matrix Algorithm**
The narrower similarity family admitted on the PMFG path, whose members cannot go negative over their own admissible distances.

### 3.8 Phylogeny

**Phylogeny**
The characterisation of asset relationships derived from a Distance Matrix. The biological metaphor is intentional: related assets cluster like species on an evolutionary tree. It has three sub-concepts:

- **Clustering**: groups related assets into a dendrogram — `ClustersEstimator` with `HClustAlgorithm`, `DBHT` (PMFG-based, with the `LoGo` sparse inverse) or `KMeansAlgorithm`, and `OptimalNumberClusters` for the cluster count.
- **Network**: assets as graph nodes and edges. `NetworkEstimator` builds a minimum spanning tree — `KruskalTree`, `BoruvkaTree`, `PrimTree`.
- **Centrality**: which assets are most influential — `BetweennessCentrality`, `ClosenessCentrality`, `DegreeCentrality`, `EigenvectorCentrality`, `KatzCentrality`, `Pagerank`, `RadialityCentrality`, `StressCentrality`.

`PhylogenyResult` carries the resulting matrix or vector.

**Phylogeny Features**
The reverse direction: a phylogeny reused as a Feature Matrix rather than consumed as one, since an assets × assets neighbourhood matrix is an assets × features matrix.

**Separation**
How far apart two assets sit in a Network, and how far is too far. Two members: `HopCount`, the number of edges on the shortest path, and `PathLength`, the summed distance along it.

**Separation Budget Rule**
A callable standing where a Separation's budget number goes, computing it from the data instead of stating it in advance: `HopCountQuantile`, `PathLengthQuantile`.

**Separation Decay**
The rule turning a Separation into a score: `LinearDecay`, `ExponentialDecay`, `ReciprocalDecay`, `NoDecay`.
*Avoid*: confusing it with the exponentially-weighted moment estimators' `decay`, which is a smoothing constant over observations.

**Polarity**
Which of the two opposite quantities a Centrality algorithm's edge weights must be: `DistancePolarity` (small is close) or `SimilarityPolarity` (large is close). It is a fact about the algorithm's mathematics, not about the graph.

### 3.9 Uncertainty Sets

**Uncertainty Set**
A neighbourhood of a specific quantity — a mean vector or a covariance matrix — within which a robust optimiser protects against the worst case. Shapes: **Box** (per-parameter bounds), **Ellipsoidal** (a joint confidence region), and the mean-only **ℓ1** and **Signed ℓ1** cross-polytopes. Constructors: `DeltaUncertaintySet`, `NormalUncertaintySet`, `ARCHUncertaintySet`, `CharacteristicUncertaintySet`.

**ucs Triple**
The three ways to ask an Uncertainty Set estimator for its sets: `ucs` for the mean and covariance sets as a pair, `mu_ucs` for the mean half, `sigma_ucs` for the covariance half.

**Characteristic Vector**
The per-asset quantity an ℓ1 uncertainty set is built around, usually the expected return, entering the objective as `mu'w`.
*Avoid*: Feature (§2), which is a coordinate of an asset in a matrix whose rows become a distance.

**Radius Calibration**
The conversion from "how many assets should I hold?" to the radius that produces it. It is a calibration, not a constraint, so a further constraint may move the realised count.

**Activation Ladder**
The non-decreasing sequence of radii at which each successive asset of a ranking joins the ℓ1 solution, one rung per asset. A Radius Calibration reads it as a bracket: a radius strictly between the `q`-th rung and the next activates exactly `q` assets. The long-only ladder ranks the characteristic on its own; the paired ladder adds the ladder of the reversed, negated ranking, so one rung activates a long/short pair.

**Ambiguity Set**
A neighbourhood of a whole distribution, rather than of one of its moments: Wasserstein (the data moves), Gelbrich (the moments are wrong), or divergence (the probabilities are wrong). It is a **reading of machinery that already exists**, not an object — no estimator constructs one.
*Avoid*: Uncertainty Set (above), which is an object a caller builds and passes.

**Ambiguity Radius**
The size of an Ambiguity Set, in the units of the return data. For a Wasserstein or a Gelbrich ball it enters the model as the coefficient of a dual-norm penalty on the weights, so the same number is spelled `val` on a Regularisation Estimator and `r` on a distributionally robust risk measure.

**Calibration Rule**
A value in a slot that computes its own number from the Prior instead of stating one, resolved by Factory on the clustering route and by the risk-constraint route inside a JuMP build, so the containing type's constructor validates the result. Both routes hand the rule the same effective solver. A stated number holds the quantity still across a refit; a rule holds whatever the rule is defined in terms of still, and lets the quantity move.
*Avoid*: Radius Calibration (above), which is one specific conversion rather than the mechanism.

**Calibration Role**
The type that places a Calibration Rule in the slot of one quantity, and names the quantity: the end of the distribution a tail probability or a deformation parameter addresses, the Ambiguity Radius, or the Esfahani-Kuhn tail weight. The rule itself lives in the role's `alg` field. A role is an Estimator and the rule it carries is an Algorithm, so a role placed inside another role's `alg` field is refused at construction by that field's bound.
*Avoid*: Calibration Rule (above), which is the rule a role carries rather than the placement of it.

## 4. Optimisation

**Optimisation Estimator**
Produces portfolio weights from a Prior Result and zero or more constraints and risk measures. The primary split is **JuMP-based** (requires a solver), **Non-JuMP-based** (naive and clustering), and **Meta-optimisers** (wrap others).

### 4.1 Naive (Non-JuMP)

- **EqualWeighted**: `1/N` across assets.
- **InverseVolatility**: weights inversely proportional to asset volatility.
- **RandomWeighted**: random feasible weights, as a baseline.

### 4.2 Clustering (Non-JuMP)

- **HierarchicalRiskParity (HRP)**: recursive bisection of the cluster tree, allocating by risk.
- **HierarchicalEqualRiskContribution (HERC)**: equalises risk contribution across clusters, then within them.
- **SchurComplementHierarchicalRiskParity (SCHRP)**: HRP augmented with a Schur-complement step.

`HierarchicalOptimiser` is the shared configuration for these.

**Branch order**
The dendrogram's leaf permutation. It never changes the merge tree or the cluster memberships.

### 4.3 JuMP-based

- **MeanRisk**: the efficient-frontier trade-off between expected return and risk. The classic objective optimiser.
- **RiskBudgeting (RB)**: targets prescribed risk *contributions* per asset or factor rather than a return/risk trade-off.
- **RelaxedRiskBudgeting (RRB)**: a convex relaxation of Risk Budgeting.
- **FactorRiskContribution (FRC)**: optimises or targets risk contributions attributed to factors.
- **NearOptimalCentering (NOC)**: finds a robust interior point near the efficient frontier.

**JuMP Returns Estimator**
One term of the objective's return expression: `ArithmeticReturn` (dot-product) or `LogarithmicReturn` (geometric). The model-global return expression is their weighted sum.

**Objective Functions**
`MinimumRisk`, `MaximumUtility`, `MaximumRatio` (Sharpe-type), `MaximumReturn`.

**Solver**
A wrapper around a single external backend (Clarabel, HiGHS, …): its module, settings, name, and the checks that validate a returned solution.

**JuMPOptimiser**
The shared JuMP model configuration: one or more `Solver`s as a fallback chain, plus JuMP-level settings.

**Model State**
The in-flight state of the JuMP model as it is built: the shared variables, expressions and scales that successive constraint and risk builders read and write.

**Per-Build Risk State**
The part of Model State belonging to one risk build rather than to the model as a whole, and so not shared with a build nested inside it.

**Model Assembly**
The fixed sequence in which an Optimisation Estimator's constraint and risk builders run to turn an empty model into a fully-constrained one. Distinct from Model State, which is the data those builders read and write.

**Objective Penalty**
The accumulator through which every *soft* contribution reaches the objective: regularisation, soft Turnover and Tracking, and Custom Objective Terms. A contribution always worsens the objective, so a reward is a negative contribution.

**Custom Term**
The user-facing extension point for a preference the library does not name: a Custom Objective Term prices one, a Custom Constraint mandates one.

### 4.4 Constraints

**Universe Sets**
A user-defined mapping of names to named groups (sectors, countries), groups, or unique-member groups, declaring every axis it carries: assets, factors and features. The foundational input to nearly all Constraint Generation.

**Universe Prefix Grammar**
The five key prefixes a Universe Sets carries, and the rule each one declares. `xkey` names the asset universe and is the one mandatory axis; an `xkey`-prefixed key is a partition of it and has the length of the asset universe. `uxkey` prefixes a unique-entry variant, which names the `xkey`-prefixed partition it draws from. `fkey` and `ufkey` mean the same on the factor axis, which is optional and is demanded at the point of need. `zkey` names the declared feature axis and carries no unique-entry sibling and no length rule, because nothing is partitioned over that axis. No prefix may be a prefix of another, which is what makes a key resolve to exactly one axis. A key matching none of them is a plain group: expanded by name and axis-blind.

**Constraint Space**
The basis a constraint's names resolve in. Assets are the absence of a re-basis; `FactorSpace` is the only member, declared by the `ExposureConstraintEstimator` wrapper.

**Factor Exposure Constraint**
A linear constraint on a portfolio's factor weights, written in factor names and projected through the loadings.

**Constraint Generation**
The user-facing utilities converting high-level specifications into the numeric form optimisers need. Built on Universe Sets:

- **LinearConstraint / LinearConstraintEstimator**: parses equations into `A·w ≤/= B`.
- **ExposureConstraintEstimator**: re-bases a linear constraint into another Constraint Space.
- **WeightBounds**: per-asset lower and upper bounds. Group-level sum bounds are linear constraints, not Weight Bounds.
- **RiskBudget**: per-name or per-group risk-contribution targets.
- **Phylogeny constraints**: `SemiDefinitePhylogeny` (SDP coupling) and `IntegerPhylogeny` (cardinality and grouping); `CentralityConstraint` constrains by centrality.
- **Threshold**: buy-in thresholds, the minimum non-zero position size.
- **AssetSetsMatrixEstimator**: builds binary group-membership matrices.

**JuMP Constraints**
The layer adding numeric constraint data into a JuMP model: the budget constraints, `LpRegularisation`, and the Turnover, Tracking and Fees constraints.

**Net vs Gross Budget**
`bgt` is the net exposure `1ᵀw`, `sbgt` the short side, and `gbgt` the gross exposure or leverage `‖w‖₁`. `gbgt` exists for the combination the other two cannot reach: gross pinned with net free.

**Exact vs Bounded Budgets**
The distinction between a budget that *bounds* the realised exposure and one that *pins* it. Pinning needs a per-asset sign bit, which makes the problem mixed-integer, so it is opt-in.

**Time-Dependent Input (Schedule)**
An optimiser input whose value changes across the folds of a cross-validation scheme instead of being fixed for the whole horizon. Only **problem definition** may vary this way, never **execution control**.

**Schedule of Optimisers**
A Time-Dependent Input whose per-fold values are whole optimisers, so the strategy itself varies over the backtest.

**Time-Dependent Context**
The per-fold information handed to a function-form Time-Dependent Input: the fold's position and count, its data, and the previous fold's weights where those are threaded.

**Time-Dependent Callable**
A struct, rather than a bare function, that computes a Time-Dependent Input's per-fold value from the Time-Dependent Context. Its parameters are data, and it declares a previous-weights need as a trait. A member declares what it returns: a constraint value, or an optimiser.

**Weight Finaliser**
A post-solve adjustment forcing weights into the feasible region: `IterativeWeightFinaliser` (projection) or `JuMPWeightFinaliser` (re-solve).

**Turnover**
The change in the portfolio relative to a reference, either the previous weights or a static target. Usable as a soft penalty or a hard constraint.

**Tracking Error**
The closeness of the portfolio to a benchmark or target: `WeightsTracking` against a reference weight vector, `ReturnsTracking` against a benchmark return series.

### 4.5 Meta-optimisers

Wrap and compose other optimisers, using Cross-Validation to estimate inner out-of-sample returns:

- **NestedClustered (NCO)**: optimise within clusters, then across clusters.
- **Stacking**: combine several base optimisers' weights through a meta-optimiser.
- **SubsetResampling**: average optimisations over many asset or observation subsets.

**Sub-Portfolio**
The unit a meta-optimiser solves one inner problem for, and one asset of the synthetic universe its outer optimiser allocates over. NCO's sub-portfolios are cluster index sets; Stacking's are its inner optimisers.

### 4.6 Cross-Validation

Used for hyperparameter tuning and for out-of-sample evaluation, and to supply inner out-of-sample returns to meta-optimisers. The axes are **optimisation vs non-optimisation** and **sequential (time-aware) vs non-sequential**. Schemes:

- **KFold**: k-fold with optional purging and embargoing.
- **CombinatorialCrossValidation**: all train/test combinations, giving multiple paths.
- **WalkForward**: `IndexWalkForward` / `DateWalkForward` for time series.
- **MultipleRandomised**: random asset subsets crossed with rolling windows.
- **Search**: `GridSearchCrossValidation`, `RandomisedSearchCrossValidation`.

**Fold Loop**
The one loop every cross-validation entry point runs. Per fold it resolves the estimator, views the data, swaps the Time-Dependent Inputs, and hands the callback one Fold.

**Fold**
The record the Fold Loop hands its callback: the fold's index and count, its already-resolved estimator, its already-viewed data, and its own training and test index vectors.

### 4.7 Finite Allocation (post-processing)

Discretises continuous weights into whole shares for a fixed cash budget, since real markets have no fractional shares.

- **DiscreteAllocation**: MIP-based exact allocation.
- **GreedyAllocation**: heuristic greedy rounding.

**FiniteAllocationInput**
The problem data fed to a Finite Allocation optimiser: target weights, asset prices, cash budget, and optional horizon and fees.
*Avoid*: FiniteAllocation (that is the family), AllocationProblem, AllocationInput.

## 5. Risk Measures

**Risk Measure**
Quantifies portfolio risk. The three-way split by legal usage is **Optimisation** (has a JuMP formulation, so it can serve as objective or constraint), **Non-Optimisation** (analysis only) and **Hierarchical** (no JuMP formulation, but valid as a clustering risk proxy).

**Risk Input Kind**
A classification orthogonal to legal usage: what a measure consumes when its expected risk is evaluated. The three kinds are **net-returns**, **weights-returns-fees** and **weights-only**.

**Precomputed-returns contract**
The rule for evaluating a risk measure on an already-reduced net-return series the caller holds directly, with no weights to apply. It is well defined only where the result is a function of the series alone.

**XatRisk**
The naming convention in which "X" stands for "Value" or "Drawdown", the same family applied to returns or to drawdowns. *Relative* variants are the hierarchical drawdown forms, *Range* variants penalise the gap between the two tails, and *Distributionally Robust (DR)* variants optimise against worst-case scenario distributions.

**Range Tails**
The two point measures a Range variant is the sum of: the **loss** tail on the net portfolio returns, and the **gain** tail on their negation at the range's second level.

**Negated Upper Tail**
The sign convention that lets a Range be a sum. The gain tail is the base measure applied to the *negated* returns, so it is reported on the same sign convention as the loss tail and the two add rather than subtract.

**Risk Series**
The per-observation series a conic tail measure reduces. Two exist: the net portfolio returns, and the negated drawdown path. Both are signed as returns, so a loss is a negative entry.

### Settings & combination

- **Risk Measure Settings**: the per-measure configuration of scaling, bounds and direction — `RiskMeasureSettings`, `HierarchicalRiskMeasureSettings`, `MaxRiskMeasureSettings`.
- **Scalariser**: combines several risk measures into one scalar — `SumScalariser`, `MaxScalariser`, `MinScalariser`, `LogSumExpScalariser`. There is no return-side counterpart: several return terms are always a weighted sum.
- **Degeneracy Guard**: the refusal an optimiser or objective makes when the expression it reads is identically zero.
- **Combination Weight**: the weight an element carries inside a combination of *several* elements, and inert on a lone element. The elements combined are risk measures, return terms, sub-portfolios, or whole portfolios.
- **Frontier / FrontierBoundEstimator**: the efficient-frontier point count and the bound transforms `LinearBound`, `SquareRootBound`, `SquaredBound`.
- **Frontier Sweep**: the sequence of solves a bound-swept optimiser runs. It is a product: every swept entry on both axes joins the same enumeration.

### Dispersion / moment measures

- **Variance**, **StandardDeviation**, **UncertaintySetVariance** (robust).
- **LowOrderMoment** (`FirstLowerMoment`, `MeanAbsoluteDeviation`, `SecondMoment`, `EvenMoment`) and **HighOrderMoment** (`ThirdLowerMoment`, `FourthMoment`, `StandardisedHighOrderMoment`).
- **Kurtosis**, **Skewness**, **NegativeSkewness**, **VarianceSkewKurtosis** (composite), **ThirdCentralMoment**.
- **MedianAbsoluteDeviation (MAD)**: centred by `MedianCentering` or `MeanCentering`.

### Quantile / tail measures (XatRisk)

- **Value-at-Risk (VaR)**: `ValueatRisk`, `ValueatRiskRange`; drawdown forms `DrawdownatRisk`, `RelativeDrawdownatRisk`.
- **Conditional (CVaR / Expected Shortfall)**: `ConditionalValueatRisk`, its Range and DR forms, and the drawdown `ConditionalDrawdownatRisk` (CDaR).
- **Entropic (EVaR)**: `EntropicValueatRisk`, its Range form, and `EntropicDrawdownatRisk` (EDaR).
- **Relativistic (RLVaR)**: `RelativisticValueatRisk`, its Range form, and `RelativisticDrawdownatRisk` (RLDaR).
- **Power Norm**: `PowerNormValueatRisk` (PNVaR), its Range form, and `PowerNormDrawdownatRisk`.

**OWA (Ordered Weights Array)**
A weighted sum of *sorted* return realisations, with weights generated by an Algorithm composing L-moments. Flexible enough to approximate CVaR and higher-moment criteria.

### Drawdown measures

- **AverageDrawdown**, **UlcerIndex** (RMS drawdown), **MaximumDrawdown** — each with a `Relative…` hierarchical form.

### Other measures

- **BrownianDistanceVariance (BDVar)**: distance-covariance-based dispersion.
- **WorstRealisation**: the worst single-period loss.
- **NoRisk**: contributes no risk term at all, so a risk-taking optimiser can express a problem that genuinely has none.
- **Range**: the spread between the best and worst realisations.
- **TurnoverRiskMeasure**: turnover expressed as a risk quantity.
- **TrackingRiskMeasure**: benchmark deviation measured as a *norm* of the portfolio-versus-benchmark difference.
- **RiskTrackingRiskMeasure**: benchmark deviation measured through a configurable *risk measure* applied to that difference.
- **EqualRisk**: enforces equal risk contributions (hierarchical).
- **RiskRatio** / **NonOptimisationRiskRatio**: ratio-form measures for hierarchical use.

### Non-optimisation (analysis) measures

- **ExpectedReturn**: the scalar portfolio expected return given weights. Distinct from the per-asset mean vector in the Prior.
- **MeanReturn**, **ThirdCentralMoment**: reporting statistics.
- **ExpectedReturnRiskRatio** / **MeanReturnRiskRatio**: a configurable return-to-risk ratio, subsuming Sharpe, Calmar and Sortino through the choice of risk measure.

## 6. Post-Processing

**Net Returns**
Portfolio returns adjusted for fees and turnover costs. Computed before drawdowns.

**Drawdowns**
Peak-to-trough declines computed from Net Returns, and the input series to drawdown-based risk measures.

**Fees**
The composite of holding and trading costs: **long** (management), **short** (borrowing), **turnover** (commission) and **fixed** (a constant charge on any non-zero weight).

**Finite Allocation**
See §4.7: the discretisation of weights into whole shares within a cash budget.

## 7. Errors & Status

**PortfolioOptimisersError**
The base for library exceptions raised by argument validation.

**OptimisationReturnCode**
The status of an optimisation: `OptimisationSuccess` or `OptimisationFailure`.
