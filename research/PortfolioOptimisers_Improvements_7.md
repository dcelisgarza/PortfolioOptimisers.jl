# PortfolioOptimisers.jl feature expansion report

## 1. Executive summary

This review was grounded in the current implementation and docs of the `dev` branch, especially the optimizer and risk stack in [src/17_Optimisation/05_JuMP/04_MeanRisk.jl](../src/17_Optimisation/05_JuMP/04_MeanRisk.jl), the uncertainty-set machinery in [src/11_UncertaintySets/03_NormalUncertaintySets.jl](../src/11_UncertaintySets/03_NormalUncertaintySets.jl), the Black-Litterman prior implementation in [src/10_Prior/05_BlackLitterman/02_BlackLittermanPrior.jl](../src/10_Prior/05_BlackLitterman/02_BlackLittermanPrior.jl), the risk-budgeting abstractions in [src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl](../src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl), and the time-dependent CV mechanics shown in [examples/5_validation_tuning/04_Time_Dependent_Constraints.jl](../examples/5_validation_tuning/04_Time_Dependent_Constraints.jl).

The gaps that remain are in the move from static single-period portfolio design toward practical portfolio engineering workflows:

1. multi-period rebalancing over a horizon,
2. distributionally robust optimization as a first-class type with a bootstrap-calibrated radius,
3. regime-aware priors,
4. explainability and attribution diagnostics.

These ideas fit naturally into the library’s current design because the package already exposes the right abstraction boundaries: `JuMPOptimiser`, `AbstractLowOrderPriorEstimator`, `AbstractUncertaintySetEstimator`, `RiskBudgetingAlgorithm`, and `TimeDependent` schedules.

---

## 3. Priority feature ideas

## 3.1 Multi-period rebalancing with transaction costs and turnover penalties

### Why it fits

The library already has:

- turnover constraints in [src/12_Turnover.jl](../src/12_Turnover.jl),
- tracking/risk-budget machinery in [src/15_Tracking.jl](../src/15_Tracking.jl) and [src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl](../src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl),
- time-dependent fold logic in [examples/5_validation_tuning/04_Time_Dependent_Constraints.jl](../examples/5_validation_tuning/04_Time_Dependent_Constraints.jl).

This is the most natural and highest-value extension.

### Mathematical formulation

Let:

- $T$ = number of rebalancing dates,
- $N$ = number of assets,
- $w_t \in \mathbb{R}^N$ = portfolio weights at date $t$,
- $\mu_t \in \mathbb{R}^N$ = expected return vector at date $t$,
- $\Sigma_t \in \mathbb{S}_{+}^{N}$ = covariance matrix at date $t$,
- $w_{t-1}$ = previous portfolio weights before rebalancing,
- $\ell, u \in \mathbb{R}^N$ = lower and upper weight bounds,
- $\tau_t \ge 0$ = turnover budget at date $t$,
- $\rho(\cdot)$ = chosen portfolio risk measure,
- $\lambda_\text{risk}, \lambda_\text{return}, \lambda_\text{tc}, \lambda_\text{turn} \ge 0$ = objective weights.

For a sequence of rebalances $t = 1, \ldots, T$,

$$
\min_{\{w_t\}_{t=1}^T}
\sum_{t=1}^{T}
\left[\lambda_\text{risk}\, \rho(w_t; \Sigma_t)
- \lambda_\text{return}\, \mu_t^\top w_t
+ \lambda_\text{tc}\, \|w_t - w_{t-1}\|_1
+ \lambda_\text{turn}\, \|w_t - w_{t-1}\|_2^2\right]
$$

subject to

$$
\mathbf{1}^\top w_t = 1,\quad
\ell \le w_t \le u,\quad
w_0 = w_{\text{prev}},\quad
\|w_t - w_{t-1}\|_1 \le \tau_t.
$$

This formulation captures:

- dynamic rebalancing,
- turnover constraints,
- transaction costs,
- spread or implementation shortfall effects,
- position drift over time.

### Good library shape

This could be implemented as a new optimizer estimator, e.g.:

```julia
@concrete struct MultiPeriodRebalance <: RiskJuMPOptimisationEstimator
    opt
    r
    obj
    turnover
    transaction_costs
    wi
    w_prev
end
```

where `turnover` and `transaction_costs` are estimators or structured objects that behave like existing `TurnoverEstimator` and fee components in [src/12_Turnover.jl](../src/12_Turnover.jl) and [src/13_Fees.jl](../src/13_Fees.jl).

### Implementation sketch

```julia
using LinearAlgebra, Statistics

abstract type AbstractRebalancingEstimator <: AbstractEstimator end

@concrete struct RebalanceCost <: AbstractRebalancingEstimator
    cost_per_trade::Float64
    max_turnover::Float64
    transaction_costs::Vector{Float64}
end

@concrete struct MultiPeriodRebalance <: RiskJuMPOptimisationEstimator
    opt::JuMPOptimiser
    r::AbstractBaseRiskMeasure
    obj::ObjectiveFunction
    turnover::RebalanceCost
    w_prev::Vector{Float64}
    horizon::Int
end

function add_rebalance_constraints!(model, w, w_prev, rc::RebalanceCost)
    @constraint(model, sum(abs.(w .- w_prev)) <= rc.max_turnover)
    @constraint(model, sum(max.(0.0, w .- w_prev)) <= rc.cost_per_trade)
    return nothing
end

function optimise_multi_period(model, returns, w_prev; rc::RebalanceCost)
    T = size(returns, 1)
    N = size(returns, 2)

    @variable(model, w[1:T, 1:N])
    @constraint(model, [t in 1:T], sum(w[t, :]) == 1.0)
    @constraint(model, [t in 1:T], w[t, :] .>= 0.0)

    for t in 1:T
        if t == 1
            add_rebalance_constraints!(model, w[t, :], w_prev, rc)
        else
            add_rebalance_constraints!(model, w[t, :], w[t-1, :], rc)
        end
    end

    @objective(model, Min,
        sum(0.5 * dot(w[t, :], cov(returns)) * dot(w[t, :], ones(N)) for t in 1:T)
        - 0.2 * sum(dot(mean(returns, dims = 1)[:], w[t, :]) for t in 1:T))
    return model
end
```

This would sit naturally alongside the current `MeanRisk` and `RiskBudgeting` optimizers.

---

## 3.2 Distributionally robust and uncertainty-aware optimization

### Why it fits

The library already has uncertainty-set abstractions, and the package is structured for robust estimation and optimization. The dynamic is already there, especially in [src/11_UncertaintySets/03_NormalUncertaintySets.jl](../src/11_UncertaintySets/03_NormalUncertaintySets.jl), but the strongest next move is more direct robust optimization as a first-class optimizer feature.

### Library-shape idea

Add a `WassersteinDROUncertaintySet` or a `DROMeanRisk` estimator built into the same abstraction family as `NormalUncertaintySet` and `BlackLittermanPrior`.

Possible types:

```julia
@concrete struct WassersteinDROUncertaintySet <: AbstractUncertaintySetEstimator
    pe
    epsilon::Float64
    q::Float64
    n_sim::Int
end
```

Then route it into `JuMPOptimiser` via a new objective or constraint builder that is aware of ambiguity sets.

### How to compute $\epsilon$ in practice

For a sample of historical returns $\{r_t\}_{t=1}^T$, a good default is not to hard-code a single constant but to estimate the radius from the sample itself. The cleanest, library-friendly approach is a bootstrap quantile rule:

$$
\epsilon = \mathrm{Quantile}_{1-\delta}
\left\{ W_1(\hat P^{(b)}, \hat P) \right\}_{b=1}^{B},
$$

where:

- $\hat P$ is the empirical return distribution,
- $\hat P^{(b)}$ is a bootstrap resample of the same return sample,
- $B$ is the number of bootstrap draws,
- $\delta \in (0, 1)$ is the tail probability or confidence level,
- $W_1(\cdot, \cdot)$ is the 1-Wasserstein distance.

This means the ambiguity radius expands just enough to cover the empirical distributional drift one expects with probability $1-\delta$.

---

## 3.3 Regime-aware and state-dependent portfolio models

### Why it fits

The library already contains clustering and phylogeny ideas, plus prior structures and dynamic schedules. That means regime dependence is a strong fit for the existing architecture.

### Mathematical formulation

Let:

- $K$ = number of market regimes,
- $s_t \in \{1, \ldots, K\}$ = regime indicator at time $t$,
- $r_t \in \mathbb{R}^N$ = asset return vector at time $t$,
- $\mu_k \in \mathbb{R}^N$ = expected return vector in regime $k$,
- $\Sigma_k \in \mathbb{S}_{+}^{N}$ = covariance matrix in regime $k$,
- $w_t \in \mathbb{R}^N$ = portfolio weights chosen at time $t$,
- $\mathcal{W}_{s_t}$ = feasible set of weights in regime $s_t$,
- $\lambda \ge 0$ = risk-aversion parameter.

Suppose a market is in hidden state $s_t \in \{1, \ldots, K\}$. Then the return model becomes

$$
r_t \mid s_t = k \sim \mathcal{N}(\mu_k, \Sigma_k),
$$

and the optimization solves

$$
\min_{w_t} \; w_t^\top \Sigma_{s_t} w_t - \lambda \mu_{s_t}^\top w_t
\quad \text{s.t. } w_t \in \mathcal{W}_{s_t}.
$$

This is more realistic than using a single unconditional covariance matrix, and it is directly compatible with the existing “prior + covariance + uncertainty set” architecture.

### Good library shape

Add a `RegimeAwarePrior` or `RegimeAdjustedExpectedReturns` estimator that wraps a `LowOrderPrior` and switches parameters by regime.

```julia
@concrete struct RegimeAwarePrior <: AbstractLowOrderPriorEstimator_AF
    pe
    regime_model
    transition_matrix
end
```

This would compose naturally with `BlackLittermanPrior` and `NormalUncertaintySet`.

### Implementation sketch

```julia
using StatsBase, LinearAlgebra

@concrete struct RegimeAwarePrior <: AbstractLowOrderPriorEstimator
    pe
    regime_model
    transition_matrix::Matrix{Float64}
    min_regime_size::Int
end

function regime_expected_returns(X, regime_labels)
    K = length(unique(regime_labels))
    mu_list = Vector{Vector{Float64}}()
    sigma_list = Vector{Matrix{Float64}}()

    for k in 1:K
        idx = findall(==(k), regime_labels)
        Xk = X[idx, :]
        push!(mu_list, mean(Xk, dims = 1)[:])
        push!(sigma_list, cov(Xk))
    end
    return mu_list, sigma_list
end

function regime_weighted_optimiser(X, regime_labels, mu_list, sigma_list; solver = Clp.Optimizer)
    T, N = size(X)
    K = length(mu_list)
    model = Model(solver)
    @variable(model, w[1:N] >= 0)
    @constraint(model, sum(w) == 1.0)

    # regime-conditioned objective: expected return weighted by transition probabilities
    regime_weights = ones(K) ./ K
    @objective(model, Max,
        sum(regime_weights[k] * dot(mu_list[k], w) for k in 1:K)
        - 0.5 * sum(regime_weights[k] * dot(w, sigma_list[k] * w) for k in 1:K))

    optimize!(model)
    return value.(w)
end
```

This integrates directly with the existing prior + covariance architecture and gives a realistic regime-aware optimizer with minimal new machinery.

## 3.4 Explainability and attribution diagnostics

### Priority expansion ideas

#### 1. Rebalance attribution and drift decomposition

A natural extension is to decompose the change in portfolio risk during a rebalance into:

- weight drift from the previous allocation,
- change in covariance or factor exposures,
- change in idiosyncratic contribution,
- transaction-cost-driven drag.

For a previous portfolio $w_{t-1}$ and a new portfolio $w_t$, define the risk delta as

$$
\Delta \rho_t = \rho(w_t) - \rho(w_{t-1}).
$$

Then decompose it into components:

$$
\Delta \rho_t = \Delta \rho_{\text{allocation}} + \Delta \rho_{\text{covariance}} + \Delta \rho_{\text{cost}} + \Delta \rho_{\text{residual}}.
$$

This is very useful for PM dashboards and explainable rebalancing reports.

#### 2. Benchmark-vs-active attribution

The missing layer is a clean decomposition of active risk into contributions by asset, factor, and sector relative to a benchmark $w_b$.

Define active risk as

$$
\rho_{\text{active}}(w) = \rho(w - w_b),
$$

and then compute

$$
RC_i^{\text{active}} = (w_i - w_{b,i}) \frac{\partial \rho(w - w_b)}{\partial w_i}.
$$

This would let users answer “what changed the benchmark gap?” in a way that is consistent with the existing risk-contribution infrastructure.

#### 3. Grouped and nested factor attribution

A stronger product feature would be a multi-level attribution tree over the existing asset and factor contributions:

- sector contribution,
- cluster contribution for hierarchical portfolios.

A useful decomposition is

$$
\rho(w) = \sum_{g \in \mathcal{G}} \rho_g(w) + \rho_{\text{idio}}(w),
$$

where $\mathcal{G}$ denotes sector, factor, or cluster groups.

#### 4. Scenario and stress attribution

Risk explanations become much more actionable when each contribution is tied to a scenario or stress event. For scenario set $\mathcal{S}$,

$$
\rho(w) = \sum_{s \in \mathcal{S}} \pi_s \rho_s(w),
$$

and one can attribute change in portfolio risk to:

- equity drawdown scenarios,
- rate shock scenarios,
- credit spread scenarios,
- macro regime transitions.

This is especially useful when the library is used in tooling for risk control or portfolio review.

#### 5. Time-varying attribution for multi-period workflows

Once multi-period rebalancing is added, the attribution layer should also track how risk contributions change over time. For horizons $t = 1, \ldots, T$, one would compute:

$$
\rho_t(w_t) - \rho_{t-1}(w_{t-1}),
$$

and attribute the change to:

- active allocation drift,
- regime-switch effects,
- turnover and implementation shortfall,
- risk-model changes.

#### 6. Robustness and model-drift attribution

A very practical missing feature is attribution of portfolio risk changes between a baseline model and a robust or uncertainty-aware variant. For example:

$$
\Delta \rho = \rho_{\text{robust}}(w) - \rho_{\text{base}}(w)
$$

can be decomposed into changes from:

- revised expected returns,
- wider uncertainty sets,
- skewed covariance estimates,
- factor-loading revisions.

This would turn robust optimization from a black-box replacement into a transparent model-selection tool.

### Good library shape

The extension should not replace the existing risk models; it should sit on top of them as a result-reporting and diagnostics layer. A natural API pattern is:

```julia
result = optimise(MyEstimator(...), rd)
report = attribution_report(result; benchmark = b, groups = sectors, scenarios = stress_paths)
```

A concrete implementation can be built as a lightweight result wrapper:

```julia
@concrete struct AttributionReport
    asset_rc::Vector{Float64}
    factor_rc::Dict{Symbol, Float64}
    benchmark_rc::Vector{Float64}
    group_rc::Dict{Symbol, Float64}
    turnover_drag::Vector{Float64}
    scenario_rc::Dict{Symbol, Float64}
end

function attribution_report(result, w, wb, factors, groups, scenarios; X = nothing)
    asset_rc = risk_contribution(Variance(), w, result.X)
    benchmark_rc = risk_contribution(Variance(), w .- wb, result.X)

    factor_rc = Dict{Symbol, Float64}()
    for (k, beta_k) in pairs(factors)
        factor_rc[k] = dot(beta_k, w) * sqrt(dot(beta_k, cov(result.X) * beta_k))
    end

    group_rc = Dict{Symbol, Float64}()
    for (g, idx) in pairs(groups)
        group_rc[g] = sum(asset_rc[idx])
    end

    scenario_rc = Dict{Symbol, Float64}()
    for (name, s) in pairs(scenarios)
        scenario_rc[name] = dot(s, asset_rc)
    end

    return AttributionReport(asset_rc, factor_rc, benchmark_rc, group_rc,
                            abs.(w .- wb), scenario_rc)
end
```

This would extend the existing infrastructure without disrupting the optimizer interfaces already used by the library.

---

## 3.5 Comparison with other libraries and practice

### What the strongest existing libraries already do well

The most relevant comparators are not “portfolio libraries” in the abstract, but optimizer stacks that are already strong in either execution-aware modeling, factor decomposition, or robust optimization.

#### 1. PyPortfolioOpt

What is worth adapting here:

- make the “results + attribution + diagnostics” story easier for non-experts.

#### 2. CVXPortfolio

CVXPortfolio is especially strong on multi-period, transaction-cost-aware, and control-oriented optimization. It treats portfolio construction as a dynamic decision process and includes time-varying constraints, turnover, and execution-aware design.

Why it matters here:

- it is a strong precedent for using convex optimization in practical portfolio workflows,
- it demonstrates the need for dynamic decision models rather than static one-period optimizers,
- it handles implementation details in a way that is close to how a real allocator thinks.

How to adapt it here:

- keep the current risk-measure abstraction,
- add mathematically explicit time-step models with `TimeDependent` schedules,
- allow execution penalties and trade-amount constraints to be added as modular extensions rather than bespoke solver hacks.

#### 3. oracle 2 and factor-based library ecosystems

oracle 2 is a particularly good reference for factor models, risk decomposition, and style attribution. The idea that a portfolio can be understood as a combination of factor risk, sector risk, and idiosyncratic risk is central to modern portfolio risk reporting.

The next step is to broaden the existing factor risk contribution into a full decomposition suite:

- sector and style attribution,
- benchmark-relative factor attribution,
- hierarchical cluster attribution,
- macro regime attribution.

#### 5. Robust optimization and DRO literature

The robust optimization literature (Wasserstein DRO, distributional uncertainty, ambiguity set optimization) is mature and practically important. It is not enough to have a robust covariance estimate; real users want to understand how uncertainty in the estimation problem changes the portfolio decision.

The barrier is mostly interface design and user-facing reporting, not pure mathematics.

### Why this is useful for this project

The main opportunity is to turn the library's capabilities into a portfolio toolkit that is both mathematically expressive and operationally interpretable. That means blending the best of:

- dynamic execution-aware modeling from CVXPortfolio,
- factor decomposition from oracle 2 and factor-analytics work,
- robust optimization from modern operations-research practice.

### Suggested sources

1. PyPortfolioOpt documentation and source — <https://pyportfolioopt.readthedocs.io/>
2. CVXPortfolio: A convex optimization approach to portfolio construction.
3. Meucci, A. (2005), Risk and Asset Allocation.
4. Black, F. and Litterman, B. (1992), Global Asset Allocation.
5. Delage, E. and Ye, Y. (2010), Distributionally Robust Optimization Under Moment Uncertainty.
6. Rockafellar, R. T. and Uryasev, S. (2000), Optimization of Conditional Value-at-Risk.

---

## 3.6 Good additions to the library, and how to adapt them here

### 3.6.3 Multi-period attribution and scenario decomposition

Once a multi-period optimizer exists, the next useful step is a time-series explanation of where risk changes came from.

#### Why it would be nice to have

- helps explain performance drift,
- connects the optimizer to a reporting workflow,
- makes model selection and regime shifts interpretable.

#### Mathematical formulation

For $t = 1, \ldots, T$, define the risk delta as

$$
\Delta \rho_t = \rho(w_t) - \rho(w_{t-1}).
$$

Then decompose into a sum of orthogonal parts:

$$
\Delta \rho_t = \Delta \rho_t^{\text{allocation}} + \Delta \rho_t^{\text{covariance}}
+ \Delta \rho_t^{\text{turnover}} + \Delta \rho_t^{\text{scenario}}.
$$

#### Standalone implementation

```julia
"""
    portfolio_risk_delta(w_new::Vector{<:Real}, w_old::Vector{<:Real}, Sigma::AbstractMatrix)

Estimate the change in variance risk between two consecutive allocations.

# Variables
- `w_new`: new portfolio weights.
- `w_old`: previous portfolio weights.
- `Sigma`: covariance matrix of returns.
"""
function portfolio_risk_delta(w_new::Vector{<:Real}, w_old::Vector{<:Real},
                             Sigma::AbstractMatrix)
    return dot(w_new, Sigma * w_new) - dot(w_old, Sigma * w_old)
end
```

This is deliberately simple and designed to be extended to a richer decomposition interface later.

#### How to adapt it here

- build it as a result-reporting layer over the optimizer outputs,
- provide scenario dictionaries for `stress`, `macro`, `factor`, and `benchmark` decompositions,
- keep it modular so it can be used without altering core optimizers.

---

### 3.6.5 Extension-package strategy

Not every feature should be forced into the main package. Some additions are better as their own extension or companion package if they need special data or a niche solver stack.

Good candidates for a companion extension package are:

- `PortfolioOptimisersExecution.jl` for execution-aware trading, market impact, and slippage models,
- `PortfolioOptimisersFactorModels.jl` for richer factor attribution, macro regimes, and style decomposition,
- `PortfolioOptimisersRobust.jl` for ambiguity-set models, DRO, and calibration utilities,
- `PortfolioOptimisersDiagnostics.jl` for risk-attribution reports, dashboards, and scenario decomposition.

This keeps the core library focused while allowing richer, domain-specific tools to evolve independently.

---

## 4. Concrete implementation plan for inclusion in the library

## 4.1 Proposed new modules and types

The implementation should stay consistent with the package style, which already uses fine-grained files, submodule-like conceptual groupings, and typed estimators. I would recommend adding new files in the same spirit as the existing structure:

- `src/17_Optimisation/18_MultiPeriodRebalance.jl`
- `src/11_UncertaintySets/06_WassersteinDRO.jl`
- `src/10_Prior/14_RegimeAwarePrior.jl`

This is consistent with the numerical grouping already visible in the package root file [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl).

## 4.2 Minimal, incremental roadmap

### Phase 1: multi-period rebalancing

- Add `MultiPeriodRebalance` estimator.
- Reuse current `JuMPOptimiser`.
- Validate with walk-forward examples.

### Phase 2: robust optimization extension

- Add `WassersteinDROUncertaintySet` or `RobustRiskBudgeting`.
- Reuse `NormalUncertaintySet` and `JuMPOptimiser` integration points.
- Add a cross-validation stress test to compare robust vs non-robust outcomes.

### Phase 3: regime-aware priors

- Add `RegimeAwarePrior`.
- Keep output objects consistent with existing result patterns.
- Add tests and examples to match the style of the existing docs.

### Phase 4: explainability layer

- Add `portfolio_attribution` result types.
- Return them through optimisation results, as the library already does for risk-budgeting and other processed outputs.

---

## 5. Recommended API sketches

The package’s current style is strongly type-based and composable. I would keep the API consistent with that style.

```julia
# Multi-period, turnover-aware optimizer
res = optimise(
    MultiPeriodRebalance(
        opt = JuMPOptimiser(; slv = slv),
        r = Variance(),
        obj = MinimumRisk(),
        turnover = TurnoverEstimator(; w = w_prev, val = 0.05),
        transaction_costs = FeesEstimator(; val = 0.0005),
    ),
    rd,
)
```

```julia
# Robust and distributionally robust possibility
ue = WassersteinDROUncertaintySet(; pe = EmpiricalPrior(), epsilon = 0.05)
res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv, ucs = ue), r = Variance()), rd)
```

```julia
# Regime-aware returns
pr = RegimeAwarePrior(; pe = EmpiricalPrior(), regime_model = HierarchicalClustering())
res = optimise(MeanRisk(; opt = JuMPOptimiser(; slv = slv), r = Variance()), rd, pr)
```

---

## 6. Sources and references

### Library references used in this review

- [README.md](../README.md)
- [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl)
- [src/17_Optimisation/05_JuMP/04_MeanRisk.jl](../src/17_Optimisation/05_JuMP/04_MeanRisk.jl)
- [src/11_UncertaintySets/03_NormalUncertaintySets.jl](../src/11_UncertaintySets/03_NormalUncertaintySets.jl)
- [src/10_Prior/05_BlackLitterman/02_BlackLittermanPrior.jl](../src/10_Prior/05_BlackLitterman/02_BlackLittermanPrior.jl)
- [src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl](../src/17_Optimisation/05_JuMP/07_RiskBudgeting.jl)
- [src/12_Turnover.jl](../src/12_Turnover.jl)
- [src/15_Tracking.jl](../src/15_Tracking.jl)
- [examples/5_validation_tuning/04_Time_Dependent_Constraints.jl](../examples/5_validation_tuning/04_Time_Dependent_Constraints.jl)

### Literature references

1. Markowitz, H. (1952). “Portfolio Selection.” Journal of Finance.
2. Black, F., & Litterman, B. (1992). “Global Asset Allocation.” Goldman Sachs.
3. Delage, E., & Ye, Y. (2010). “Distributionally Robust Optimization Under Moment Uncertainty.” Mathematics of Operations Research.
4. Pflug, G. C., & Wozabal, D. (2007). “Ambiguity in Portfolio Selection.” Quantitative Finance.
5. Rockafellar, R. T., & Uryasev, S. (2000). “Optimization of Conditional Value-at-Risk.” Journal of Risk.
6. Meucci, A. (2005). Risk and Asset Allocation.
7. Fabozzi, F. J., Kolm, P. N., Pachamanova, D. A., & Focardi, S. M. (2007). Robust Portfolio Optimization and Management.

---

## 7. Final recommendation

The strongest contribution to this library would be a cohesive “portfolio engineering” layer built on top of the existing optimizer foundations:

1. multi-period rebalancing,
2. robust objectives under parameter uncertainty,
3. richer risk attribution and explainability.

This approach respects the project’s current architecture, extends it with real-world portfolio constraints, and preserves the library’s composable, type-driven style.

The next best step is not to create a totally separate object model. It is to extend the existing abstractions already present in `JuMPOptimiser`, risk budgets, priors, and time-dependent constraints into a coherent robust multi-period optimization workflow.
