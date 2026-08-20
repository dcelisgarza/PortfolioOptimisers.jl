# PortfolioOptimisers.jl feature expansion report

## 1. Executive summary

This review was grounded in the current implementation and docs of the `dev` branch, especially the optimizer and risk stack in [src/20_Optimisation/11_MeanRisk.jl](../src/20_Optimisation/11_MeanRisk.jl), the uncertainty-set machinery in [src/14_UncertaintySets/03_NormalUncertaintySets.jl](../src/14_UncertaintySets/03_NormalUncertaintySets.jl), the Black-Litterman prior implementation in [src/13_Prior/06_BlackLittermanPrior.jl](../src/13_Prior/06_BlackLittermanPrior.jl), the risk-budgeting abstractions in [src/20_Optimisation/14_RiskBudgeting.jl](../src/20_Optimisation/14_RiskBudgeting.jl), and the time-dependent CV mechanics shown in [docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md](../docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md).

The codebase is already unusually strong in a few directions:

- composable estimation and optimization layers,
- rich constraint generation and risk budgets,
- multiple uncertainty-set and prior formulations,
- time-dependent cross-validation and constraint schedules,
- a mature JuMP-centric optimization abstraction.

The biggest gap is not “more mean-variance variants”; it is the move from static single-period portfolio design toward practical portfolio engineering workflows:

1. multi-period rebalancing with turnover and transaction cost awareness,
2. robust and distributionally robust optimization under parameter uncertainty,
3. regime-aware or benchmark-relative optimization,
4. better execution-aware diagnostics and explainability.

These ideas fit naturally into the library’s current design because the package already exposes the right abstraction boundaries: `JuMPOptimiser`, `AbstractLowOrderPriorEstimator`, `AbstractUncertaintySetEstimator`, `RiskBudgetingAlgorithm`, and `TimeDependent` schedules.

---

## 2. What the codebase already does well

### 2.1 Optimization architecture

The current mean-risk optimizer in [src/20_Optimisation/11_MeanRisk.jl](../src/20_Optimisation/11_MeanRisk.jl) already supports a broad objective family:

- minimum risk,
- maximum return,
- maximum Sharpe ratio,
- maximum utility,
- general risk-measure-driven formulations.

This is a strong base for extensions, because the library cleanly separates:

- objective definition (`ObjectiveFunction`),
- risk measure (`RiskMeasure`),
- constraint construction (`JuMPOptimiser` and constraint generation),
- result interpretation (`MeanRiskResult`, `RiskBudgetingResult`, etc.).

### 2.2 Uncertainty sets and priors

The uncertainty set code in [src/14_UncertaintySets/03_NormalUncertaintySets.jl](../src/14_UncertaintySets/03_NormalUncertaintySets.jl) already formalizes box and ellipsoidal uncertainty around mean/covariance estimates. The library also incorporates Black-Litterman-style priors in [src/13_Prior/06_BlackLittermanPrior.jl](../src/13_Prior/06_BlackLittermanPrior.jl), which is exactly the right foundation for Bayesian or robust portfolio workflows.

### 2.3 Risk budget and time-dependent dynamics

The library already contains detailed risk-budgeting machinery in [src/20_Optimisation/14_RiskBudgeting.jl](../src/20_Optimisation/14_RiskBudgeting.jl), and it already supports time-dependent schedules in [docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md](../docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md). That is a particularly important signal: the package is already thinking in terms of dynamic, folded, rebalancing-aware optimization problems.

This means the next leap is not a brand-new conceptual layer; it is a tighter integration of these ideas into a single multi-period, uncertainty-aware optimization workflow.

---

## 3. Priority feature ideas

## 3.1 Multi-period rebalancing with transaction costs and turnover penalties

### Why it fits

The library already has:

- turnover constraints in [src/15_Turnover.jl](../src/15_Turnover.jl),
- tracking/risk-budget machinery in [src/18_Tracking.jl](../src/18_Tracking.jl) and [src/20_Optimisation/14_RiskBudgeting.jl](../src/20_Optimisation/14_RiskBudgeting.jl),
- time-dependent fold logic in [docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md](../docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md).

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

where `turnover` and `transaction_costs` are estimators or structured objects that behave like existing `TurnoverEstimator` and fee components in [src/15_Turnover.jl](../src/15_Turnover.jl) and [src/16_Fees.jl](../src/16_Fees.jl).

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

The library already has uncertainty-set abstractions, and the package is structured for robust estimation and optimization. The dynamic is already there, especially in [src/14_UncertaintySets/03_NormalUncertaintySets.jl](../src/14_UncertaintySets/03_NormalUncertaintySets.jl), but the strongest next move is more direct robust optimization as a first-class optimizer feature.

### Mathematical formulation

Let:

- $N$ = number of assets,
- $w \in \mathbb{R}^N$ = portfolio weights,
- $\mu \in \mathbb{R}^N$ = expected return vector,
- $\Sigma_Q$ = covariance matrix under a candidate distribution $Q$,
- $\hat P$ = empirical distribution from historical data,
- $\epsilon \ge 0$ = radius of the ambiguity set,
- $W_1(\cdot, \cdot)$ = 1-Wasserstein distance,
- $\mathcal{W}$ = feasible set of weights,
- $\lambda \ge 0$ = risk-aversion parameter,
- $r$ = random asset return vector,
- $\gamma \in (0, 1)$ = tail probability in CVaR.

A Wasserstein distributionally robust mean-risk formulation is:

$$
\min_{w \in \mathcal{W}} \max_{Q: W_1(Q, \hat P) \le \epsilon}
\mathbb{E}_{Q}\left[-\mu^\top w + \lambda \, w^\top \Sigma_Q w\right].
$$

Alternatively, for CVaR robust optimization:

$$
\min_{w \in \mathcal{W}, \alpha, \eta}
\alpha + \frac{1}{(1-\gamma)}\mathbb{E}[\max(\ell(w, r)-\alpha, 0)]
$$

with ambiguity-ball constraints on the empirical distribution.

This is exactly the type of “distributionally robust portfolio optimization” that is valuable in real data settings where the sample covariance or expected return estimate is noisy and unstable.

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

For a simpler first implementation, a good heuristic is:

$$
\epsilon = c \cdot \mathrm{median}_{t}\left(\|r_t - \bar r\|_2\right),
\quad \bar r = \frac{1}{T}\sum_{t=1}^{T} r_t,
$$

with $c \in [0.5, 2.0]$ as a calibration parameter. In portfolio settings, a more interpretable version is:

$$
\epsilon = c \cdot \sqrt{\frac{\operatorname{tr}(\Sigma)}{T}},
$$

where $\Sigma$ is the sample covariance matrix. This scales the ambiguity radius with the sample noise level and the number of observations.

This gives a practical progression for the library:

1. default to a bootstrap quantile if a robust calibration routine is available,
2. fall back to a heuristic based on dispersion for fast use,
3. allow user-specified $\epsilon$ for custom calibration.

### Implementation sketch

```julia
using LinearAlgebra, Statistics

abstract type AbstractUncertaintySetEstimator <: AbstractEstimator end

@concrete struct WassersteinDRO <: AbstractUncertaintySetEstimator
    pe::AbstractLowOrderPriorEstimator
    epsilon::Float64
    q::Float64
    n_sim::Int
end

function estimate_wasserstein_epsilon(returns; delta = 0.05, B = 200, c = 1.0)
    T, N = size(returns)
    rbar = vec(mean(returns, dims = 1))
    disp = vec(sum((returns .- rbar') .^ 2, dims = 2))
    heuristic = c * mean(sqrt.(disp)) / sqrt(T)
    return heuristic
end

function add_wasserstein_dro_constraint!(model, w, returns, ue::WassersteinDRO)
    T, N = size(returns)
    @variable(model, s[1:T] >= 0)
    @variable(model, λ >= 0)
    @expression(model, r[t = 1:T], sum(w[i] * returns[t, i] for i in 1:N))
    @constraint(model, [t in 1:T], r[t] + s[t] >= 0)
    @constraint(model, sum(s) / T <= ue.epsilon * λ)
    return nothing
end

function robust_mean_risk_optimiser(X, mu, ue::WassersteinDRO; solver = Clp.Optimizer)
    T, N = size(X)
    model = Model(solver)
    @variable(model, w[1:N] >= 0)
    @constraint(model, sum(w) == 1.0)
    add_wasserstein_dro_constraint!(model, w, X, ue)
    @objective(model, Max, dot(mu, w) - ue.q * dot(w, cov(X) * w))
    optimize!(model)
    return value.(w)
end
```

This keeps the feature faithful to the package’s pattern: uncertainty set + optimizer + risk measure, but implements a compact, usable first pass rather than a purely abstract concept.

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

### Why it fits

The library already has the core primitives for this: asset-level risk contribution is implemented through `risk_contribution(...)`, and factor-level decomposition is available via `FactorRiskContribution` and factor-regression-based assembly. That means the strongest next step is not to invent a new attribution framework from scratch, but to package these existing building blocks into richer, action-oriented diagnostics for users.

### What is already in place

The current library already supports:

- per-asset risk contribution via `risk_contribution(r, w, X, fees; ...)`,
- factor-risk attribution via the factor-model decomposition machinery,
- optimiser-level budgets and contribution constraints via `RiskBudgeting`, `RelaxedRiskBudgeting`, and `FactorRiskContribution`.

This is the right foundation for decision-support outputs rather than a brand-new feature family.

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

The library already has benchmark-relative tracking machinery, but the missing layer is a clean decomposition of active risk into contributions by asset, factor, and sector relative to a benchmark $w_b$.

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

The factor decomposition is already conceptually present, but a stronger product feature would be a multi-level attribution tree:

- asset contribution,
- sector contribution,
- factor contribution,
- idiosyncratic residue,
- cluster contribution for hierarchical portfolios.

A useful decomposition is

$$
\rho(w) = \sum_{g \in \mathcal{G}} \rho_g(w) + \rho_{\text{idio}}(w),
$$

where $\mathcal{G}$ denotes sector, factor, or cluster groups. This is a natural next step from the current `FactorRiskContribution` support.

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

This is the natural extension of the static decomposition already represented by `risk_contribution`.

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

### Mathematical view

Let:

- $N$ = number of assets,
- $w \in \mathbb{R}^N$ = portfolio weights,
- $\rho(w)$ = portfolio risk,
- $RC_i(w)$ = per-asset contribution,
- $FC_k(w)$ = factor contribution,
- $\Sigma \in \mathbb{S}_{+}^{N}$ = covariance matrix,
- $\beta$ = factor loading matrix.

A clean decomposition is

$$
\rho(w) = \sum_i RC_i(w) = \sum_k FC_k(w) + \rho_{\text{idio}}(w),
$$

with

$$
RC_i(w) = w_i \frac{\partial \rho(w)}{\partial w_i},
\qquad
FC_k(w) = \beta_{k,\cdot}^\top \frac{\partial \rho(w)}{\partial \beta_{k,\cdot}}.
$$

This is exactly the right level of abstraction for a portfolio explanation layer built on top of the existing risk and factor machinery.

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

PyPortfolioOpt is excellent at making standard portfolio optimization practical and accessible: mean-variance, max Sharpe, minimum volatility, Black-Litterman, risk parity, and constrained optimization wrappers are all straightforward to use. Its main gap relative to this codebase is not mathematical breadth; it is engineering breadth.

What is worth copying:

- a clean user-facing optimizer API,
- a broad set of classical formulations,
- fast, understandable examples,
- a good separation between objective, constraints, and result reporting.

What is worth adapting here:

- move more of the library toward domain-specific but low-friction user ergonomics,
- keep the existing typed Julia interfaces, but expose small, higher-level `optimise(...)` wrappers with clear defaults,
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

#### 3. Riskfolio-Lib and factor-based library ecosystems

Riskfolio-Lib is a particularly good reference for factor models, risk decomposition, and style attribution. The idea that a portfolio can be understood as a combination of factor risk, sector risk, and idiosyncratic risk is central to modern portfolio risk reporting.

This library already has `FactorRiskContribution`, which is good evidence that the right conceptual direction is already present. The next step is not to replace it but to broaden it into a full decomposition suite:

- sector and style attribution,
- benchmark-relative factor attribution,
- hierarchical cluster attribution,
- macro regime attribution.

#### 4. skfolio

skfolio is a Python portfolio-optimization library built on the scikit-learn API and designed around estimator/transformer/pipeline composition, model selection, and cross-validation. It is especially strong in the combination of classical optimization with modern model-evaluation workflow.

What is worth copying:

- the scikit-learn-style estimator API and pipeline compatibility,
- easy model selection and hyperparameter tuning via `GridSearchCV`, `RandomizedSearchCV`, and walk-forward / purged cross-validation,
- a rich risk-measure and prior-estimation ecosystem,
- `Benchmark Tracker`, `Risk Budgeting`, `Hierarchical Risk Parity`, and nested-cluster optimization models,
- explicit support for uncertainty sets and factor models as first-class estimators.

What is worth adapting here:

- keep the package’s type-driven Julia design, but add a higher-level, sklearn-style workflow layer for train/test evaluation and parameter tuning,
- expose a clean `fit`/`predict`-style portfolio workflow that is easier for users to compose in experiments,
- treat benchmark tracking, factor priors, and uncertainty sets as plug-in components rather than one-off objective customizations,
- align the library’s result-reporting with the skfolio pattern of portfolio objects summarizing risk, return, and out-of-sample diagnostics.

This matters because skfolio is not just “another optimizer.” It is one of the clearest examples of a portfolio library centered on model evaluation and composable workflows rather than only one-shot optimization.

#### 5. Robust optimization and DRO literature

The robust optimization literature (Wasserstein DRO, distributional uncertainty, ambiguity set optimization) is mature and practically important. It is not enough to have a robust covariance estimate; real users want to understand how uncertainty in the estimation problem changes the portfolio decision.

This library already has `AbstractUncertaintySetEstimator`, which makes it one of the more natural Julia ecosystems to integrate a stronger robust layer. The barrier is mostly interface design and user-facing reporting, not pure mathematics.

### Why this is useful for this project

The current project is already strong in:

- typed estimator composition,
- risk-measure abstraction,
- JuMP integration,
- pipeline and workflow ergonomics inspired by scikit-learn patterns,
- time-dependent and schedule-aware constraints,
- factor and risk-budgeting concepts.

The main opportunity is to turn these capabilities into a portfolio toolkit that is both mathematically expressive and operationally interpretable. That means blending the best of:

- classical optimizer ergonomics from PyPortfolioOpt,
- workflow ergonomics and model selection from skfolio,
- dynamic execution-aware modeling from CVXPortfolio,
- factor decomposition from Riskfolio-Lib and factor-analytics work,
- robust optimization from modern operations-research practice.

### Suggested sources

1. PyPortfolioOpt documentation and source — <https://pyportfolioopt.readthedocs.io/>
2. CVXPortfolio: A convex optimization approach to portfolio construction — <https://github.com/cvxgrp/cvxportfolio>
3. skfolio documentation and examples — <https://skfolio.org/>
4. Riskfolio-Lib documentation — <https://riskfolio-lib.readthedocs.io/>
5. Meucci, A. (2005), Risk and Asset Allocation.
6. Black, F. and Litterman, B. (1992), Global Asset Allocation.
7. Delage, E. and Ye, Y. (2010), Distributionally Robust Optimization Under Moment Uncertainty.
8. Rockafellar, R. T. and Uryasev, S. (2000), Optimization of Conditional Value-at-Risk.

---

## 3.6 Good additions to the library, and how to adapt them here

### 3.6.1 Execution-aware optimization and market impact

This is one of the strongest additions to make because real portfolio implementation is never frictionless. In a live allocator workflow, the optimization should know that moving from $w_{t-1}$ to $w_t$ has a cost.

#### Why it would be nice to have

- reduces unrealistic turnover and trade-size assumptions,
- aligns the model with implementation constraints,
- helps produce more realistic portfolio recommendations,
- connects naturally to the existing `Turnover` and fee machinery.

#### Mathematical formulation

Let $w_t$ be the target portfolio at time $t$ and $w_{t-1}$ the previous portfolio. Let $\Delta w_t = w_t - w_{t-1}$ be the trade vector. A simple implementation-aware objective is:

$$
\min_{w_t}
\rho(w_t)
+ \lambda_{\text{turn}} \|\Delta w_t\|_1
+ \lambda_{\text{impact}} \sum_i \left|\Delta w_{t,i}\right|^{1+\eta}
- \lambda_{\text{return}} \mu_t^\top w_t,
$$

with $\eta > 0$ capturing the nonlinear impact of larger trades.

#### Standalone implementation

```julia
"""
    market_impact_cost(w::Vector{<:Real}, w_prev::Vector{<:Real}; gamma::Real = 1e-4,
                       eta::Real = 0.5)

Compute a convex trade-impact penalty for a portfolio rebalance.

# Variables
- `w`: target portfolio weights.
- `w_prev`: previous portfolio weights.
- `gamma`: scale of the execution cost.
- `eta`: curvature of the impact penalty. Larger values make large trades more costly.

The formulation is

    cost = gamma * sum(abs.(w - w_prev) .^ (1 + eta)).

This can be plugged into a rebalancing objective or used as a constraint on total
trade size when a more realistic execution budget is desired.
"""
function market_impact_cost(w::Vector{<:Real}, w_prev::Vector{<:Real};
                           gamma::Real = 1e-4, eta::Real = 0.5)
    trade = w .- w_prev
    return gamma * sum(abs.(trade) .^ (1 + eta))
end
```

#### How to adapt it here

- extend the existing turnover and fee concepts rather than creating a parallel system,
- allow cost coefficients to be `TimeDependent` if needed,
- expose as either an objective penalty or a separate execution constraint.

---

### 3.6.2 Distributionally robust optimization

This is a strong extension point because the library already has uncertainty sets and priors. A distributionally robust layer would complement the current mean-risk architecture instead of replacing it.

#### Why it would be nice to have

- better stability in noisy regimes,
- explicit handling of parameter uncertainty,
- more realistic allocation when expected returns and covariances are estimated from finite samples,
- natural compatibility with the existing uncertainty-set abstraction.

#### Mathematical formulation

Let $\hat P$ be the empirical return distribution and $\mathcal{P}$ an ambiguity set around it. The distributionally robust problem is:

$$
\min_{w \in \mathcal{W}} \sup_{P \in \mathcal{P}} \mathbb{E}_{P}\left[-\mu^\top w + \lambda w^\top \Sigma_P w\right].
$$

A practical ambiguity set is a Wasserstein ball:

$$
\mathcal{P} = \{P : W_1(P, \hat P) \le \epsilon\}.
$$

#### Standalone implementation

```julia
"""
    wasserstein_radius(X::AbstractMatrix; c::Real = 1.0)

Estimate a simple Wasserstein-style ambiguity radius from historical returns.

# Variables
- `X`: return matrix with rows as observations and columns as assets.
- `c`: scale factor controlling the magnitude of the ambiguity radius.

This is intentionally a fast, heuristic first implementation. It scales the radius by
sample dispersion and is easy to replace later by a more principled bootstrap or
transport-based estimate.
"""
function wasserstein_radius(X::AbstractMatrix; c::Real = 1.0)
    T, N = size(X)
    rbar = vec(mean(X, dims = 1))
    disp = vec(sum((X .- rbar') .^ 2, dims = 2))
    return c * mean(sqrt.(disp)) / sqrt(T)
end

"""
    robust_mean_variance_opt(X::AbstractMatrix, mu::AbstractVector; epsilon::Real = 0.05,
                             q::Real = 1.0, solver = nothing)

A simple robust mean-variance-style optimizer using an ambiguity radius.

# Variables
- `X`: historical return matrix.
- `mu`: expected return vector.
- `epsilon`: uncertainty radius.
- `q`: risk-aversion strength.
- `solver`: optional JuMP-compatible optimizer.
"""
function robust_mean_variance_opt(X::AbstractMatrix, mu::AbstractVector;
                                epsilon::Real = 0.05, q::Real = 1.0,
                                solver = nothing)
    T, N = size(X)
    if solver === nothing
        throw(ArgumentError("A JuMP-compatible solver is required for this implementation."))
    end

    model = Model(solver)
    @variable(model, w[1:N] >= 0)
    @constraint(model, sum(w) == 1.0)
    @constraint(model, sum(abs.(w)) <= 1.0)

    Sigma = cov(X)
    @objective(model, Max, dot(mu, w) - q * dot(w, Sigma * w) - epsilon * sum(abs.(w)))
    optimize!(model)
    return value.(w)
end
```

#### How to adapt it here

- integrate with `AbstractUncertaintySetEstimator` and `JuMPOptimiser`,
- route the robust penalty through existing risk-measure logic rather than creating a disconnected solver,
- make `epsilon` a calibrated or prior-driven quantity instead of a constant.

---

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

- `src/20_Optimisation/18_MultiPeriodRebalance.jl`
- `src/14_UncertaintySets/06_WassersteinDRO.jl`
- `src/13_Prior/14_RegimeAwarePrior.jl`
- `src/18_Tracking/02_BenchmarkRelativeTracking.jl` or extend the existing tracking file

This is consistent with the numerical grouping already visible in the package root file [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl).

## 4.2 Minimal, incremental roadmap

### Phase 1: multi-period rebalancing

- Add `MultiPeriodRebalance` estimator.
- Reuse current `JuMPOptimiser`.
- Add `TurnoverEstimator` and fee-aware constraints.
- Validate with walk-forward examples.

### Phase 2: robust optimization extension

- Add `WassersteinDROUncertaintySet` or `RobustRiskBudgeting`.
- Reuse `NormalUncertaintySet` and `JuMPOptimiser` integration points.
- Add a cross-validation stress test to compare robust vs non-robust outcomes.

### Phase 3: regime-aware priors and benchmark tracking

- Add `RegimeAwarePrior` and `BenchmarkRelative` optimizer.
- Keep output objects consistent with existing result patterns.
- Add tests and examples to match the style of the existing docs.

### Phase 4: explainability layer

- Add `risk_contribution` and `portfolio_attribution` result types.
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

```julia
# Benchmark-relative optimization
res = optimise(BenchmarkRelativeRisk(; benchmark = b, active_risk_limit = 0.05), rd)
```

---

## 6. Sources and references

### Library references used in this review

- [README.md](../README.md)
- [src/PortfolioOptimisers.jl](../src/PortfolioOptimisers.jl)
- [src/20_Optimisation/11_MeanRisk.jl](../src/20_Optimisation/11_MeanRisk.jl)
- [src/14_UncertaintySets/03_NormalUncertaintySets.jl](../src/14_UncertaintySets/03_NormalUncertaintySets.jl)
- [src/13_Prior/06_BlackLittermanPrior.jl](../src/13_Prior/06_BlackLittermanPrior.jl)
- [src/20_Optimisation/14_RiskBudgeting.jl](../src/20_Optimisation/14_RiskBudgeting.jl)
- [src/15_Turnover.jl](../src/15_Turnover.jl)
- [src/18_Tracking.jl](../src/18_Tracking.jl)
- [docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md](../docs/src/examples/5_validation_tuning/04_Time_Dependent_Constraints.md)

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
3. benchmark-relative active management controls,
4. richer risk attribution and explainability.

This approach respects the project’s current architecture, extends it with real-world portfolio constraints, and preserves the library’s composable, type-driven style.

The next best step is not to create a totally separate object model. It is to extend the existing abstractions already present in `JuMPOptimiser`, risk budgets, priors, and time-dependent constraints into a coherent robust multi-period optimization workflow.
