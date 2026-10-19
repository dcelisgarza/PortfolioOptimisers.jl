# PortfolioOptimisers.jl — Corrected Feature Comparison and Research Roadmap

**Baseline:** `dev` branch of `dcelisgarza/PortfolioOptimisers.jl`
**Research date:** 2026-08-14

## Executive summary

> What do other serious portfolio libraries do that PortfolioOptimisers does not yet do, or what do they do in a way that could be materially improved in PortfolioOptimisers?

The strongest remaining opportunities are:

1. synthetic-data/synthetic-prior machinery;
2. a more general moment-estimation *compatibility/interface* layer rather than more moment estimators;
3. sklearn-compatible regression integration;
4. stronger distributionally robust uncertainty sets;
5. liquidity modelling;
6. frozen-position portfolio state;
7. model/portfolio uncertainty propagation;
8. robustness and model-disagreement analysis;
9. regime-conditioned and synthetic scenario machinery;
10. stability diagnostics and provenance.

The last group is where the library could go beyond feature parity and develop genuinely distinctive functionality.

---

# 3. Corrected comparison with the oracle

The useful comparison with the oracle is now about individual capabilities.

| The oracle's capability | PortfolioOptimisers `dev` | Assessment |
| --- | --- | --- |
| Synthetic prior/data | Not equivalent | **Missing** |
| General moment-estimator compatibility | Partial | **Improvement** |
| sklearn regressors | Not equivalent | **Missing** |
| Synthetic scenario/stress framework | Not yet a comparable general facility | **Opportunity** |

### The key lesson

Do not copy the oracle's architecture.

Use the existing PortfolioOptimisers Pipeline and add the missing statistical capabilities as native Julia components.

---

# 4. Synthetic priors / synthetic data

This remains one of the clearest gaps derived from the oracle.

The useful concept is not simply "generate random returns." It is:

```text
historical observations
        ↓
fit generative/distributional model
        ↓
generate synthetic observations
        ↓
run existing moment/risk/optimisation machinery
```

Potential applications:

- small samples;
- tail enrichment;
- stress scenarios;
- Monte Carlo estimation;
- robust optimisation;
- uncertainty estimation.

A particularly attractive design would make synthetic data another prior/data-generating component that can be inserted into an existing Pipeline.

### Stronger version

Support conditional synthetic generation:

```text
historical data
     ↓
regime/factor/characteristic model
     ↓
conditional generator
     ↓
synthetic paths
```

This would go substantially beyond a simple synthetic-data transformer.

**Priority: very high.**

---

# 6. Frozen positions and portfolio state

This is another strong OptimalPortfolios idea.

A real portfolio often cannot trade every asset at every rebalance.

A useful abstraction is:

```text
asset status
    ├── tradable
    ├── frozen
    ├── unavailable
    └── new
```

The optimiser then operates on the tradable portion while respecting the current state.

This becomes especially important when:

- illiquid assets rebalance less frequently;
- private assets are frozen;
- tax constraints prevent sales;
- a position temporarily cannot be traded;
- asset universes change.

OptimalPortfolios also addresses the subtle case where frozen positions drift and cause a group constraint to become infeasible, relaxing that bound for the rebalance with an audit trail. citeturn0search0

That suggests a broader PortfolioOptimisers abstraction:

> **portfolio state should be explicit rather than treating optimisation as a stateless function of returns.**

**Priority: high.**

---

# 9. Moment estimation: the gap is interface/compatibility, not estimators

This is an important correction.

PortfolioOptimisers already has a large moments subsystem. The missing opportunity is closer to the oracle's concept of a unified moment-estimation interface with explicit compatibility between estimators and downstream portfolio models.

The useful abstraction is:

```text
moment estimator
      ↓
produces:
    μ?
    Σ?
    coskewness?
    cokurtosis?
      ↓
downstream model checks requirements
```

For example:

```text
Estimator              μ    Σ    S    K
-----------------------------------------
Sample                 ✓    ✓
EWMA                   ✓    ✓
Factor                 ✓    ✓
HigherMoment           ✓    ✓    ✓    ✓
```

The framework should be able to tell the user that an estimator is or is not compatible with a particular optimisation/risk expression.

This would be a **capability/trait layer**, not another collection of estimators.

**Priority: high.**

---

# 10. sklearn-compatible regression models

This remains a real gap and could be unusually useful because the current moments subsystem already contains regression machinery.

The target should not be "copy sklearn."

The target should be interoperability:

```text
characteristics / features
          ↓
regression estimator
          ↓
conditional expected returns / factor structure
          ↓
existing moments / priors
          ↓
existing optimisation
```

Potential models:

- Ridge;
- Lasso;
- Elastic Net;
- robust regression;
- tree-based regressors;
- arbitrary sklearn regressors through an extension.

The extension boundary is important: the core package should not necessarily depend on scikit-learn.

**Priority: high.**

---

# 12. Distributionally robust uncertainty sets

This is where PortfolioOpt.jl provides a useful comparison.

PortfolioOpt explicitly models:

- moment uncertainty;
- Wasserstein ambiguity sets;
- budget sets;
- elliptical sets;
- distributional uncertainty.

PortfolioOptimisers already has several uncertainty-set types, including delta, normal, bootstrap and L1 sets. citeturn5view1turn0search1

The worthwhile question is therefore:

> Should PortfolioOptimisers add richer distributional ambiguity sets, especially Wasserstein-style sets?

This would complement the existing uncertainty framework rather than duplicate it.

Potentially useful additions:

- Wasserstein balls;
- phi-divergence ambiguity sets;
- moment-based ambiguity sets;
- support + moment hybrid sets.

**Priority: high research value.**

---

# 14. Higher moments: the real remaining opportunity

Oracle 2's continued development around factor-model-based coskewness/cokurtosis is useful evidence that the difficult problem is no longer "can we calculate skewness?"

The difficult problem is:

> **How do we make high-order moments computationally tractable in high dimensions?**

PortfolioOptimisers already has higher moments. The next-level improvements could therefore include:

- low-rank factor representations;
- tensor decompositions;
- sparse higher moments;
- scalable large-N formulations.

**Priority: medium-high research priority.**

---

# 15. Liquidity

This remains one of the clearest practical gaps across the comparison.

A richer implementation model could include:

```text
ADV
+
participation rate
+
minimum trade size
+
liquidity state
```

Possible abstractions:

```julia
LiquidityConstraint(...)
```

This would materially improve real-world portfolio construction.

**Priority: high.**

---

# 16. Model and portfolio uncertainty

This is the strongest genuinely new idea.

The missing layer is **propagation of the library's uncertainties into the portfolio itself**.

Instead of:

```julia
weights
```

returning only one deterministic answer, analyse:

```text
estimator perturbation
model perturbation
constraint perturbation
regime perturbation
       ↓
many plausible portfolios
       ↓
distribution of weights / returns / risks
```

For example:

```text
Asset       median    5%      95%
A            8.2%     2.1%    14.7%
B            6.4%     4.8%     9.1%
C            4.1%     0.0%     9.8%
```

This is substantially more informative than a single optimum.

**Priority: very high / strategic.**

---

# 17. Model disagreement

A particularly interesting form of portfolio uncertainty is **disagreement among reasonable models**.

Suppose:

```text
MeanRisk          → portfolio A
HRP               → portfolio B
HERC              → portfolio C
RiskBudgeting     → portfolio D
BlackLitterman    → portfolio E
```

Measure the dispersion.

Possible outputs:

- pairwise weight distance;
- weight variance;
- portfolio return dispersion;
- risk dispersion;
- rank stability;
- consensus weights.

Then:

```text
low disagreement
    → higher model confidence

high disagreement
    → model uncertainty
```

This could lead to a genuinely novel PortfolioOptimisers concept:

> **optimiser disagreement as a portfolio diagnostic or uncertainty measure.**

**Priority: very high / strategic.**

---

# 18. Robustness analysis and a robustness frontier

The next logical layer is to perturb assumptions systematically.

Dimensions include:

- lookback;
- expected-return estimator;
- covariance estimator;
- prior;
- shrinkage;
- risk aversion;
- constraints;
- optimiser;
- regime.

Then quantify how much the portfolio changes.

This supports a three-dimensional decision concept:

```text
return
risk
robustness
```

A portfolio with slightly lower expected performance but dramatically higher stability may be preferable.

This is an especially natural extension of PortfolioOptimisers because the package already contains the underlying estimation and optimisation components needed to generate the perturbations.

**Priority: very high / strategic.**

---

# 19. Synthetic scenarios + regimes

Once synthetic priors exist, the next step is conditional scenario generation.

Instead of:

```text
historical returns
    ↓
fit distribution
    ↓
sample
```

support:

```text
historical returns
    ↓
identify regimes / factors
    ↓
fit conditional models
    ↓
simulate paths
```

Then support explicit scenarios:

- equity crash;
- rate shock;
- volatility spike;
- correlation breakdown;
- factor reversal;
- liquidity shock.

This can feed existing:

- CVaR;
- drawdown;
- worst-realisation;
- robust optimisation;
- scenario constraints.

**Priority: high.**

---

# 20. Regime-aware estimation

The current moments subsystem already contains regime-adjusted estimators, so this is **not** a blank-slate feature.

The opportunity is to generalise the regime concept across the package:

```text
regime
  ↓
expected returns
  ↓
covariance
  ↓
factor exposures
  ↓
uncertainty
  ↓
optimisation
```

Potential future capabilities:

- regime-conditional priors;
- regime probabilities;
- regime-conditional uncertainty;
- regime-weighted optimisation;
- worst-case regime optimisation;
- regime-aware synthetic paths.

**Priority: medium-high.**

---

# 21. Portfolio state and dynamic optimisation

A full dynamic architecture would represent:

```text
state_t
  +
information_t
  ↓
estimation_t
  ↓
optimisation_t
  ↓
trades_t
  ↓
state_{t+1}
```

The state can contain:

- cash;
- frozen assets;
- tradability;
- previous trades;
- liquidity;
- costs;
- constraints.

This is a much stronger concept than simply adding a backtester.

**Priority: high, but larger scope.**

---

# 23. Attribution and diagnostics

Rather than adding isolated reporting functions, build a coherent diagnostic layer.

### Stability

- bootstrap dispersion;
- model disagreement;
- regime sensitivity;
- constraint sensitivity.

This would turn the package's large amount of underlying machinery into a much more inspectable research object.

**Priority: medium-high.**

---

# 24. Portfolio provenance

A small but valuable addition would be to make every portfolio result able to describe its construction.

For example:

```text
data
  ↓
preprocessing
  ↓
pre-selection
  ↓
prior
  ↓
moment estimator
  ↓
uncertainty set
  ↓
risk measure
  ↓
objective
  ↓
constraints
  ↓
solver
  ↓
post-processing
```

The result should ideally retain enough metadata to answer:

> How exactly was this portfolio produced?

This is especially useful for research reproducibility and auditability.

**Priority: medium.**

---

# 25. Revised priority matrix

| Feature | Current status | Value | Differentiation | Priority |
| --- | --- | ---: | ---: | ---: |
| Synthetic prior/data | Missing | 5 | 4 | **Very high** |
| Moment compatibility interface | Partial | 5 | 4 | **Very high** |
| sklearn regression integration | Missing | 4 | 4 | **High** |
| Distributionally robust sets | Partial | 5 | 5 | **High** |
| Frozen portfolio state | Missing/improvable | 5 | 4 | **High** |
| Liquidity constraints | Limited/extension opportunity | 5 | 5 | **High** |
| High-dimensional higher moments | Existing; improve scalability | 4 | 4 | Medium-high |
| Regime-aware unified framework | Partial | 5 | 5 | Medium-high |
| Synthetic regime scenarios | Missing | 5 | 5 | **High** |
| Portfolio uncertainty | Missing as a unified layer | 5 | 5 | **Strategic** |
| Model disagreement | Missing | 5 | 5 | **Strategic** |
| Robustness analysis | Missing as unified layer | 5 | 5 | **Strategic** |
| Robustness frontier | New idea | 5 | 5 | **Strategic** |
| Portfolio provenance | Missing/limited | 3 | 4 | Medium |
| Declarative PortfolioSpec | New idea | 4 | 4 | Later |

---

# 27. Recommended development sequence

## Phase 1 — Finish the known gaps

1. Synthetic prior/data.
2. Moment-estimator capability/compatibility interface.
3. sklearn regression integration.

## Phase 2 — Make rolling construction production-grade

1. Explicit portfolio state.
2. Frozen positions.
3. Liquidity constraints.

## Phase 3 — Exploit the existing statistical breadth

 1. Conditional synthetic scenarios.
 2. Regime-aware pipelines.
 3. Scalable factor/high-dimensional higher moments.
 4. Portfolio provenance.

## Phase 4 — Build the distinctive research layer

 1. Portfolio uncertainty propagation.
 2. Model uncertainty.
 3. Optimiser disagreement.
 4. Robustness analysis.
 5. Robustness frontier.
 6. Model ensembles.
 7. Regime stress testing.

---

# 28. The strongest strategic direction

After removing the features that are already implemented, the picture becomes much clearer.

PortfolioOptimisers should not try to become a Julia clone of the oracle, oracle 2 or OptimalPortfolios.

It already has the broad mathematical foundation.

The opportunity is to make the package answer a harder question:

> **How confident should I be in this portfolio, given uncertainty in the data, statistical model, market regime, constraints and implementation?**

That leads naturally to:

```text
                         DATA
                           |
              +------------+------------+
              |                         |
       Characteristics              Returns
              |                         |
              +------------+------------+
                           |
                      ESTIMATION
                           |
          +----------------+----------------+
          |                |                |
         μ                 Σ          Higher moments
          |                |                |
          +----------------+----------------+
                           |
                     UNCERTAINTY
                           |
          +----------------+----------------+
          |                |                |
      Sampling           Model            Regime
          |                |                |
          +----------------+----------------+
                           |
                      OPTIMISATION
                           |
                    PORTFOLIO STATE
                           |
          +----------------+----------------+
          |                |                |
       Liquidity         Costs           Drift
          |                |                |
          +----------------+----------------+
                           |
                 OUT-OF-SAMPLE / ONLINE
                           |
                           ↓
                  MODEL UPDATE
```

The existing Pipeline is the natural glue.

The key new layer would sit around it:

```text
candidate models
      ↓
portfolio distribution
      ↓
uncertainty / disagreement
      ↓
robustness
      ↓
decision
```

That is where I think PortfolioOptimisers could become genuinely distinctive.

---

# 31. Full synthesis: comparison findings + original ideas

The comparison with other libraries should not replace the broader set of ideas developed during the original review. The useful outcome is to separate them into three categories:

1. **Confirmed gaps / concrete improvements**, supported by comparison with other libraries.
2. **Extensions of machinery PortfolioOptimisers already has**, where the package is not missing the concept but could make it more general, scalable, or easier to use.
3. **Original strategic ideas**, which are not simply copies of another package and could become differentiating features.

## 31.1 Confirmed gaps and near-term improvements

These are the strongest concrete items from the comparison:

- synthetic-data priors and conditional synthetic scenarios;
- explicit rolling portfolio state;
- frozen positions;
- a formal moment-estimator capability/compatibility layer;
- sklearn regression interoperability;
- richer distributionally robust uncertainty sets;
- liquidity constraints.

These should form the core of the conventional roadmap.

---

## 31.2 Extensions of existing PortfolioOptimisers machinery

Several of the most interesting ideas do **not** require adding an entirely new subsystem.

### Existing moments → scalable higher-order models

Higher moments already exist. The next step is:

```text
high-order moments
       ↓
factor structure / low rank
       ↓
scalable estimation
       ↓
scalable optimisation
```

This could make coskewness/cokurtosis useful at asset universes where dense tensors become impractical.

### Existing uncertainty sets → uncertainty propagation

Bootstrap/delta/normal/etc. uncertainty sets already exist. The missing higher-level layer is:

```text
uncertain input
      ↓
portfolio distribution
```

rather than merely:

```text
uncertain input
      ↓
robust optimisation constraint
```

### Existing regime-adjusted moments → regime-aware portfolio models

Regime-aware moments already exist. The opportunity is to make regimes first-class across priors, synthetic data, optimisation, and stress testing.

---

# 32. Original ideas that go beyond the library comparison

The following ideas came from considering what PortfolioOptimisers could become after the obvious gaps are filled.

## 32.1 Portfolio uncertainty as a first-class result

Today the natural result of optimisation is a portfolio/weight vector.

A future result could optionally contain:

```text
point estimate
+
sampling uncertainty
+
parameter uncertainty
+
model uncertainty
+
regime uncertainty
+
constraint uncertainty
```

For example:

```text
Asset    Weight    5%      95%
A         8.2%     2.1%    14.7%
B         6.4%     4.8%     9.1%
C         4.1%     0.0%     9.8%
```

The important conceptual shift is:

> the optimiser returns a distribution over plausible portfolios, not merely one optimum.

This could be implemented by repeatedly evaluating an existing Pipeline under resampled/perturbed inputs, so it builds on existing architecture rather than replacing it.

---

## 32.2 Model disagreement

Run several reasonable portfolio construction processes:

```text
Pipeline A → portfolio A
Pipeline B → portfolio B
Pipeline C → portfolio C
Pipeline D → portfolio D
```

Then measure:

- pairwise weight distance;
- average weight variance;
- portfolio-return dispersion;
- risk dispersion;
- rank stability;
- consensus weights.

This creates a new diagnostic:

> **How much does the portfolio depend on the modelling choice?**

That is a different question from statistical confidence.

---

## 32.3 Robustness analysis

A general robustness API could perturb:

- lookback windows;
- expected-return estimators;
- covariance estimators;
- priors;
- shrinkage;
- uncertainty-set radii;
- risk aversion;
- constraints;
- optimisers;
- regimes.

Conceptually:

```julia
robustness(result; perturbations=...)
```

could return distributions of:

- weights;
- expected return;
- risk;
- turnover;
- factor exposure.

This would turn the large collection of existing estimators into a systematic research tool.

---

## 32.4 Robustness frontier

Traditional portfolio analysis asks:

```text
return ↔ risk
```

A stronger research tool asks:

```text
return ↔ risk ↔ robustness
```

A portfolio can have a superior in-sample Sharpe ratio while being much more sensitive to modelling assumptions.

This suggests reporting a frontier of portfolios trading off:

- expected performance;
- risk;
- sensitivity to assumptions.

This is a potentially distinctive research contribution.

---

## 32.5 Model ensembles

Once model disagreement can be measured, it becomes possible to construct portfolios from multiple models.

For example:

```text
                    +--> MeanRisk
                    |
Pipeline search ----+--> HRP
                    |
                    +--> RiskBudgeting
                    |
                    +--> Black-Litterman
                    |
                    +--> Characteristics model
                             |
                             ↓
                         Ensemble
```

Models could be weighted by:

- validation performance;
- robustness;
- stability;
- Bayesian/model probabilities;
- inverse disagreement.

This turns the Pipeline architecture into a **portfolio model ensemble framework**.

---

## 32.6 Synthetic scenario laboratory

The synthetic-prior feature could eventually become much more than a prior estimator.

A scenario laboratory could support:

```text
historical data
      ↓
distribution / copula / factor / regime model
      ↓
synthetic paths
      ↓
scenario transformation
      ↓
portfolio evaluation
```

Scenario transformations could include:

- volatility multiplication;
- correlation shocks;
- factor shocks;
- characteristic reversals;
- rate shocks;
- equity crashes;
- liquidity shocks.

The same scenario object could then be consumed by existing risk measures and optimisation machinery.

---

## 32.7 Portfolio experiment objects

A useful research abstraction could be an experiment containing:

```text
data
pipeline
parameter grid
resampling scheme
validation scheme
random seeds
results
diagnostics
```

Then an experiment could answer:

> Which modelling assumptions actually matter for this portfolio?

This is a natural complement to Pipeline and cross-validation.

---

## 32.8 Provenance and reproducibility

Every fitted portfolio could retain enough information to reconstruct its construction:

```text
data source / version
preprocessing
pre-selection
prior
moment estimator
uncertainty set
risk measure
objective
constraints
solver
parameters
random seeds
```

This becomes increasingly important once the package supports synthetic data, model ensembles and uncertainty propagation.

---

## 32.9 Automatic compatibility checking

The existing multiple-dispatch architecture could support a useful diagnostic layer that checks a Pipeline before execution.

For example:

```text
RiskMeasure X
requires:
    covariance
    expected returns

MomentEstimator Y
provides:
    covariance
    coskewness
```

The system could detect:

- missing required estimates;
- incompatible moment representations;
- unsupported uncertainty sets;
- solver incompatibilities;
- incompatible constraint types;
- data-shape problems.

This would turn many runtime errors into useful model-construction diagnostics.

---

# 33. Final combined roadmap

The complete recommendation is therefore:

## Tier 1 — close genuine gaps

1. Synthetic priors.
2. Moment-estimator capability/compatibility.
3. sklearn regression interoperability.

## Tier 2 — production-grade portfolio state

1. Frozen positions.
2. Liquidity constraints.
3. Dynamic/multi-period optimisation.

## Tier 3 — exploit existing statistical machinery

 1. Conditional synthetic scenarios.
 2. Regime-aware priors and stress testing.
 3. Scalable factor/high-dimensional higher moments.
 4. Portfolio provenance.
 5. Automatic compatibility checking.

## Tier 4 — differentiated research functionality

 1. Portfolio uncertainty propagation.
 2. Model uncertainty.
 3. Model/optimiser disagreement.
 4. Robustness analysis.
 5. Robustness frontier.
 6. Portfolio model ensembles.
 7. Synthetic scenario laboratory.
 8. Portfolio experiment/research objects.

---

# 34. The central thesis

The most important conclusion after combining both analyses is:

> **PortfolioOptimisers does not need more isolated portfolio-optimisation features nearly as much as it needs to become better at expressing uncertainty around the entire portfolio-construction process.**

The existing architecture already gives it most of the raw ingredients:

```text
moments
priors
uncertainty sets
risk measures
constraints
optimisers
pipelines
cross-validation
resampling
factors
regimes
```

The next layer is:

```text
             MANY PLAUSIBLE MODELS
                      |
                      ↓
             MANY PLAUSIBLE PORTFOLIOS
                      |
          +-----------+-----------+
          |           |           |
       risk        weights     exposures
          |           |           |
          +-----------+-----------+
                      |
                disagreement
                      |
                  robustness
                      |
                    decision
```

That would position PortfolioOptimisers differently from a simple "Julia equivalent" of another library.

The compelling end-state is a framework where the user can ask not only:

> "What portfolio does this model produce?"

but:

> "How stable is this portfolio across reasonable statistical models, samples, regimes, assumptions and implementation constraints?"

That is the direction most likely to turn the existing breadth of PortfolioOptimisers into a coherent and genuinely differentiated research platform.

# Sources checked

## PortfolioOptimisers.jl

- `dev` branch: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev>
- `src` tree: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src>
- Moments: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/05_Moments>
- Priors: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/10_Prior>
- Uncertainty sets: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/11_UncertaintySets>
- Constraint generation: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/09_ConstraintGeneration>
- Pre-selection: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/dev/src/20_AssetSelection.jl>
- Risk measures: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/16_RiskMeasures>
- Pipeline: <https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/dev/src/21_Pipeline>

## The oracle

- Repository
- Model selection / online learning
- Pre-selection
- Priors / synthetic data

## OptimalPortfolios

- Repository and current documentation: <https://github.com/ArturSepp/OptimalPortfolios>

Its current implementation is particularly relevant for NaN-aware rolling construction, frozen positions, drift-aware turnover and production-oriented portfolio state. citeturn0search0

## Oracle 2

- Repository
- Changelog
- Documentation

Its current feature set confirms that factor risk contribution, graph constraints, higher moments, integer constraints, attribution and uncertainty-set tooling are all established capabilities in the competing ecosystem. citeturn1search0turn1search1

## PortfolioOpt.jl

- Repository: <https://github.com/andrewrosemberg/PortfolioOpt.jl>

Its explicit treatment of moment uncertainty, Wasserstein ambiguity and robust uncertainty sets makes it a useful reference for the next generation of PortfolioOptimisers uncertainty machinery. citeturn0search1
