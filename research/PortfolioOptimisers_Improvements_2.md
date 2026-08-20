# PortfolioOptimisers.jl Implementation Plan (from v0.28.0)

## Current State Assessment

The `dev` branch at v0.28.0 is already one of the most comprehensive open-source portfolio optimization libraries. It includes:

- **Hierarchical methods:** HRP, HERC, NCO (generalized with inner/outer risk measures and cross-validation)
- **Meta-optimization:** Stacking, Subset Resampling, Nested Clustered Optimization
- **Advanced priors:** Black-Litterman (5 variants), Entropy Pooling, Opinion Pooling, Feature Priors, High-Order Factor Priors
- **Uncertainty sets:** Delta, Normal, Bootstrap, L1
- **Risk measures:** Variance, moment-based, VaR, CVaR, EVaR, RLVaR, power norms, drawdowns, tracking, turnover, ratio measures, equal risk, MAD, Brownian distance variance, variance-skewness-kurtosis composite
- **Constraints:** MIP indicators, SDP, cardinality, threshold, turnover, fees, tracking error, regularisation (L1/L2/Lp/L∞), weight norms, integer phylogeny
- **Cross-validation:** KFold, combinatorially purged, walk-forward, multiple randomised, grid search, randomised search
- **Pipelines:** Bespoke pipeline system with `StepExecution`, `PredictionCV`, `SearchCrossValidation`
- **Finite allocation:** Discrete MIP and greedy algorithms
- **Moments:** Gerber, Smyth-Broby, distance, lower tail dependence, rank, mutual information, regime-adjusted exponential weighted, coskewness, cokurtosis, implied volatility
- **Clustering/Phylogeny:** Hierarchical, DBHT, non-hierarchical

The following features are **confirmed absent** and represent the highest-value additions.

---

## Phase 1: Exact & Robust Methods

*Target: v0.29.0 – v0.30.0 | 2–3 months*

### 1.1 Critical Line Algorithm (CLA)

**Gap:** All mean-variance problems are solved via generic QP solvers. There is no exact analytical frontier tracer.

**Design:**

- New file: `20_Optimisation/25_CriticalLineAlgorithm.jl`
- Type: `CriticalLineAlgorithm <: AbstractOptimiser`
- Inputs: any `AbstractPrior` (extracts μ, Σ), `WeightBounds`, optional linear equality/inequality matrices
- Output: `CLAFrontierResult <: AbstractResult` containing:
  - `turning_points::Vector{Vector{Float64}}` — exact corner portfolios
  - `lambdas::Vector{Float64}` — risk-aversion parameters
  - `free_sets::Vector{Set{Int}}` — active set history
  - `frontier_risk::Vector{Float64}`, `frontier_return::Vector{Float64}`

**Integration:**

- Reuse existing `WeightBounds` and `PortfolioOptimisersCovariance` via the prior system.
- CLA results should be plottable via the existing plotting extension (frontier curves, weight transitions).
- A `factory` method should convert a CLA result into a standard `MeanRisk` estimator at any specific lambda.

**Why this fits:** Self-contained, no new dependencies, complements `MeanRisk` with exact solutions for long-only/box-constrained problems.

**Effort:** Medium

---

### 1.2 Stress Testing & Scenario Matrix Framework

**Gap:** No dedicated module for systematic robustness assessment. DRO exists but requires the user to define uncertainty sets manually.

**Design:**

- New directory: `26_StressTesting/`
- `ScenarioMatrix <: AbstractPrior` — wraps a collection of perturbed priors.
- `StressTest <: AbstractEstimator` — runs any existing estimator across all scenarios.
- `StressTestResult` — aggregates:
  - Weight instability (average turnover between scenario solutions)
  - Risk measure drift per scenario
  - Return degradation
  - Probability of constraint violation

**Scenario types:**

- `HistoricalScenario` — replay 2008, 2020, 2022 from built-in or user data
- `PerturbationScenario` — correlation spikes, volatility shocks, mean shifts
- `MonteCarloScenario` — sample from fat-tailed distributions

**Integration:**

- Scenarios feed directly into existing `DistributionallyRobust` estimators.
- Reuse existing `CrossValidation` scoring infrastructure for metric aggregation.
- Plotting extension: tornado charts, scenario comparison plots.

**Why this fits:** Builds naturally on the prior system and existing DRO infrastructure.

**Effort:** Medium

---

### 1.3 Tax-Aware Optimization

**Gap:** No tax logic exists. `Fees.jl` handles transaction costs but not tax consequences.

**Design:**

- New directory: `27_TaxAware/`
- `TaxLot` — cost basis, acquisition date, shares
- `TaxAwarePrior <: AbstractPrior` — wraps an existing prior plus tax lot metadata
- `TaxConstraint <: AbstractConstraint` — wash-sale lookback, short-term/long-term rate differential
- `TaxAwareMeanRisk <: AbstractEstimator` — modifies objective to maximize after-tax utility
- `TaxAwareDiscreteAllocation` — extends `DiscreteFiniteAllocation` to consider tax lots when rounding

**Key rules to model:**

- Wash-sale: 30-day window before/after sale
- STCG/LTCG rate differential
- Loss harvesting: offset gains, $3k ordinary income cap, indefinite carryforward
- Gain/loss netting hierarchy

**Integration:**

- Natural extension of `Fees.jl` and `DiscreteFiniteAllocation`.
- Time-dependent constraints (already exist) can enforce wash-sale windows.

**Why this fits:** No open-source library does this well. Massive practical differentiation.

**Effort:** High

---

## Phase 2: Dynamic & Sequential Extensions

*Target: v0.31.0 – v0.33.0 | 4–5 months*

### 2.1 Multi-Stage Stochastic Programming (MSP)

**Gap:** All optimizers are single-period. No scenario-tree-based sequential decision making.

**Design:**

- New directory: `28_StochasticProgramming/`
- `ScenarioTree` — nodes with parent/child relationships, stage labels, and per-node priors
- `MultiStageStochastic <: AbstractEstimator`
- Non-anticipativity constraints enforced via JuMP
- Reuses existing `Solver` infrastructure; MSP is just a larger JuMP model

**Solver strategy:**

- For moderate trees: direct JuMP formulation
- For large trees: Benders decomposition or progressive hedging (can leverage `StructuredSolvers.jl` ecosystem if available)

**Integration:**

- Each tree node uses existing `AbstractPrior` and `AbstractRiskMeasure`.
- Time-dependent constraints already support stage-wise turnover and budget.

**Effort:** Very High

---

### 2.2 Online / Sequential Portfolio Selection

**Gap:** All methods are batch. No regret-minimizing streaming algorithms.

**Design:**

- New directory: `29_OnlinePortfolio/`
- `OnlinePortfolioSelector <: AbstractEstimator`
- Implementations:
  - `UniversalPortfolio` (Cover, 1991)
  - `OnlineNewtonStep`
  - `FollowTheRegularizedLeader`
- `StreamingReturns` wrapper around `ReturnsResult` that yields batches sequentially
- `OnlineResult` — time-varying weights and wealth trajectory

**Integration:**

- Can be evaluated using existing walk-forward cross-validation.
- Plotting extension: wealth trajectories, cumulative regret.

**Effort:** High

---

### 2.3 Market Impact & Optimal Execution

**Gap:** `Fees.jl` has proportional costs but no market impact models.

**Design:**

- New directory: `30_MarketImpact/`
- `MarketImpactModel <: AbstractModel`
  - `LinearImpact`
  - `SquareRootImpact` (square-root law)
  - `AlmgrenChriss` — temporary + permanent impact
- `ExecutionSchedule <: AbstractEstimator` — optimizes trade trajectory over a horizon
- Extends `DiscreteFiniteAllocation` or works as a post-processing step on `MeanRisk` results

**Integration:**

- Uses existing time-dependent constraints for participation rate limits.
- Natural bridge between portfolio construction (`MeanRisk`) and execution.

**Effort:** High

---

## Phase 3: Advanced Dependence & Tail Risk

*Target: v0.34.0 – v0.35.0 | 3–4 months*

### 3.1 Copula-Based Tail Dependence Optimization

**Gap:** `LowerTailDependenceCovariance` exists but there is no full copula-based risk measure or optimizer.

**Design:**

- New directory: `31_CopulaRisk/`
- `CopulaPrior <: AbstractPrior` — marginal models + copula (Gaussian, t-copula, vine)
- `CopulaRiskMeasure <: AbstractRiskMeasure` — CVaR, EVaR, etc. computed from copula simulations
- `VineCopulaPrior` — pair-copula constructions for high dimensions

**Dependency:** `Copulas.jl` as a weak dependency via package extension (`PortfolioOptimisersCopulaExt`).

**Integration:**

- Copula priors feed into any existing `MeanRisk` or `RiskBudgeting` estimator.
- Risk measures use the same `factory` pattern.

**Effort:** Very High

---

### 3.2 ESG & Sustainability Multi-Objective Optimization

**Gap:** No ESG constraints or sustainability objectives exist.

**Design:**

- New directory: `32_ESG/`
- `ESGConstraint <: AbstractConstraint` — score bounds, carbon budgets, exclusion lists
- `ESGPrior <: AbstractPrior` — augments empirical prior with ESG score vectors
- `TriObjective` or `EpsilonConstraint` wrapper for return-risk-sustainability Pareto surfaces
- Plotting extension: 3D efficient surfaces or 2D slices

**Integration:**

- ESG constraints plug into existing `JuMPOptimiser` constraint generation.
- Can be combined with `BlackLittermanPrior` for ESG-themed views.

**Effort:** Medium

---

## Phase 4: Explainability, Performance & Research Frontiers

*Target: v0.36.0+ | 6+ months*

### 4.1 Interpretability & Attribution

**Gap:** No post-hoc explanation of why a portfolio has specific weights.

**Design:**

- New directory: `33_Attribution/`
- `PortfolioAttribution` — computes marginal impact of each constraint and risk measure
- Methods:
  - Constraint removal attribution: optimize with/without each constraint, measure weight delta
  - Risk measure Shapley values: game-theoretic attribution of weight contributions to each risk term
- Output: waterfall charts, constraint impact rankings

**Integration:**

- Works with any existing `AbstractResult`.
- Plotting extension: attribution waterfall charts.

**Effort:** Medium

---

### 4.2 GPU-Accelerated Large-Scale Estimation

**Gap:** All covariance, shrinkage, and eigenvalue cleaning is CPU-only.

**Design:**

- Package extension: `PortfolioOptimisersCUDAExt`
- GPU-accelerated implementations for:
  - `SimpleCovariance`
  - `ShrunkExpectedReturns`
  - `NearestCorrelationMatrix` (Newton method)
  - `Posdef` matrix processing
- Opt-in via `use_gpu=true` flag or separate `GPUCovariance` type

**Integration:**

- Keep CPU as default. GPU types dispatch to CUDA kernels.
- Target: 1000+ assets where CPU becomes a bottleneck.

**Effort:** Medium-High

---

### 4.3 Reinforcement Learning Environments

**Gap:** No RL interface for sequential portfolio decisions.

**Design:**

- Package extension: `PortfolioOptimisersRLExt`
- `PortfolioEnv` following `CommonRLInterface` or `ReinforcementLearning.jl` API
- State: current weights, recent returns, risk measures
- Action: target weight adjustments
- Reward: risk-adjusted return minus transaction costs

**Integration:**

- Uses existing estimators as the environment "physics."
- Enables research into learned dynamic rebalancing policies.

**Effort:** High

---

### 4.4 Rough Volatility & Fractional Models

**Gap:** All return/covariance models assume standard Brownian motion or exponential weighting.

**Design:**

- New directory: `34_FractionalModels/`
- `FractionalBrownianPrior <: AbstractPrior` — Hurst parameter estimation
- `RoughHestonCovariance` — covariance forecasting using rough volatility
- `fBMForecast` — fractional Brownian motion return forecasts

**Integration:**

- Replaces standard covariance in any existing estimator.

**Effort:** Very High

---

### 4.5 Causal Portfolio Optimization

**Gap:** All dependence is correlational.

**Design:**

- Package extension: `PortfolioOptimisersCausalExt`
- `CausalPrior <: AbstractPrior` — stores a DAG from causal discovery
- `CausalDiversificationConstraint` — prevents concentration in assets with common causal ancestors
- `CausalRiskMeasure` — measures risk along causal pathways

**Dependency:** `CausalInference.jl` or `StructuralCausalModels.jl` as weak dependency.

**Effort:** Very High

---

### 4.6 Quantum & Quantum-Inspired Solvers

**Gap:** No quantum computing interfaces.

**Design:**

- Package extension: `PortfolioOptimisersQuantumExt`
- `QuantumSolver <: AbstractSolver` wrapper
- Start with simulated annealing on GPU
- Optional: D-Wave Ocean SDK bindings for cardinality-constrained problems

**Target:** Discrete allocation and cardinality problems where exact MIP is intractable.

**Effort:** High

---

## Priority Matrix

| Feature | Practical Impact | Differentiation | Effort | Phase |
| --- | --- | --- | --- | --- |
| CLA | Medium | Low (exactness) | Medium | 1 |
| Stress Testing | High | Medium | Medium | 1 |
| Tax-Aware Optimization | Very High | Very High | High | 1 |
| ESG Multi-Objective | High | Medium | Medium | 3 |
| Multi-Stage Stochastic | Very High | High | Very High | 2 |
| Online / Sequential | High | High | High | 2 |
| Market Impact | High | Medium | High | 2 |
| Copula Tail Dependence | High | Very High | Very High | 3 |
| Interpretability | High | Medium | Medium | 4 |
| GPU Acceleration | Medium | Medium | Medium-High | 4 |
| RL Environments | Medium | High | High | 4 |
| Rough Volatility | Medium | Very High | Very High | 4 |
| Causal Optimization | Medium | Very High | Very High | 4 |
| Quantum Solvers | Low | Very High | High | 4 |

---

## Suggested First PRs (from current dev)

1. **CLA** — Pure Julia, no new dependencies, fits existing `AbstractOptimiser` pattern. Good reference for a non-JuMP solver path.
2. **Stress Testing** — Builds directly on existing priors, DRO, and CV scoring. Demonstrates composability.
3. **Tax-Aware Discrete Allocation** — Extends `DiscreteFiniteAllocation` with practical real-world logic. No other open-source library does this.

These three establish the pattern for Phase 1 while requiring minimal architectural changes to the v0.28.0 codebase.
