# PortfolioOptimisers.jl — Comprehensive Feature Roadmap

This document expands the earlier suggestions into a detailed, actionable roadmap for extending `PortfolioOptimisers.jl`. Each section includes the mathematical formulation (with all variables clearly defined), relevant literature, and standalone Julia implementation sketches (with extensive comments) that can be adapted into the library's architecture.

---

## 1. Deep Learning for Return & Risk Prediction

### 1.1 Motivation

Traditional moment estimation (sample mean, exponentially weighted moving average) makes strong assumptions about return distributions and fails to capture nonlinear dependencies. Deep learning models—particularly LSTMs and transformers—have demonstrated superior predictive performance for financial time series. Integrating these into the optimisation pipeline allows users to combine sophisticated predictive models with the library's robust optimisation engine.

### 1.2 Mathematical Formulation

Let $\mathbf{r}_t \in \mathbb{R}^N$ be the vector of asset returns at time $t$ (where $N$ is the number of assets). A neural network $f_\theta$ parameterised by $\theta$ maps historical features $\mathcal{H}_t$ to predicted returns:

$$\hat{\boldsymbol{\mu}}_{t+1} = f_\theta(\mathcal{H}_t)$$

Here:

- $\mathcal{H}_t$ is the history of returns and possibly other features up to time $t$.
- $\hat{\boldsymbol{\mu}}_{t+1}$ is the predicted expected return vector for the next period.
- $\theta$ denotes the trainable parameters of the neural network.

For covariance estimation, a common approach uses a GARCH-type structure where conditional variance evolves as:

$$\boldsymbol{\Sigma}_{t+1} = \boldsymbol{\omega} + \boldsymbol{\alpha} \odot \boldsymbol{\epsilon}_t^2 + \boldsymbol{\beta} \odot \boldsymbol{\Sigma}_t$$

where:

- $\boldsymbol{\Sigma}_{t+1}$ is the predicted covariance matrix for period $t+1$.
- $\boldsymbol{\omega}$ is a constant vector (or matrix) of long-run variances.
- $\boldsymbol{\alpha}$ and $\boldsymbol{\beta}$ are coefficient vectors (element‑wise multiplication $\odot$).
- $\boldsymbol{\epsilon}_t = \mathbf{r}_t - \hat{\boldsymbol{\mu}}_t$ are the residual returns.

More sophisticated approaches use a neural network $g_\phi$ to directly parameterise the covariance matrix:

$$\hat{\boldsymbol{\Sigma}}_{t+1} = g_\phi(\mathcal{H}_t)$$

The optimisation problem then becomes:

$$\begin{align}
\underset{\mathbf{w}}{\text{opt}} \quad & \mathbf{w}^\top \hat{\boldsymbol{\mu}}_{t+1} - \frac{\lambda}{2} \mathbf{w}^\top \hat{\boldsymbol{\Sigma}}_{t+1} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0
\end{align}$$

where:
- $\mathbf{w} \in \mathbb{R}^N$ is the portfolio weight vector.
- $\lambda$ is the risk aversion coefficient.
- $\mathbf{1}$ is a vector of ones; $\mathbf{1}^\top \mathbf{w}=1$ ensures full investment, and $\mathbf{w} \ge 0$ enforces no short‑selling.

### 1.3 Literature

- **LSTM for return prediction**: Multi-period portfolio selection using LSTM for return forecasting with GARCH for volatility estimation.
- **ARCHModels.jl**: Julia package for GARCH estimation.
- **Deep learning in portfolio optimisation**: JuMP integration with ML forecasts for portfolio optimisation.

### 1.4 Sample Implementation (with detailed comments)

```julia
using Flux, PortfolioOptimisers, LinearAlgebra

# Define a prior that uses a neural network to predict returns.
# T is the type of the neural network (must be a Flux Chain).
struct NeuralReturnPrior{T<:Chain} <: AbstractPrior
    model::T            # The trained neural network (e.g., LSTM)
    lookback::Int       # Number of past observations used as input
end

# Application: given historical returns X (T × N), predict next-period returns.
function (prior::NeuralReturnPrior)(X::AbstractMatrix)
    # X: T rows (time steps) × N columns (assets)
    T, N = size(X)
    # Prepare features as sliding windows of length `lookback`.
    features = prepare_features(X, prior.lookback)
    # Run the model to get predicted returns for the next period.
    μ = prior.model(features)  # This should output a vector of length N
    # For simplicity, we use the sample covariance of the most recent `lookback` days.
    Σ = cov(X[end-prior.lookback+1:end, :])
    return μ, Σ
end

# Helper function: convert a matrix of returns into sliding windows.
function prepare_features(X::AbstractMatrix, lookback::Int)
    T, N = size(X)
    # For each time step from `lookback` to T-1, extract a window of `lookback` rows.
    windows = [X[t-lookback+1:t, :] for t in lookback:T-1]   # exclude the last row as it's the target
    # Reshape to (number_of_windows, lookback, N) for LSTM input.
    # Flux's LSTM expects input as (features, sequence_length, batch_size) or (features, sequence_length, ...)
    # Here we treat each window as a sequence of length `lookback` with N features.
    # We'll permute dimensions to (N, lookback, windows) and then maybe transpose.
    # A simpler approach: use a Dense network on flattened features.
    # For demonstration, we'll just flatten:
    return hcat([reshape(w, :) for w in windows]...)   # not ideal; adapt as needed
end

# Usage with a simple LSTM (this is illustrative; actual implementation would vary)
model = Chain(
    LSTM(size(X, 2), 64),   # input size = N, hidden size = 64
    Dense(64, size(X, 2))   # output size = N
)
prior = NeuralReturnPrior(model, 60)   # use last 60 days
μ, Σ = prior(returns)
```

### 1.5 Integration Points

- **New prior type**: `NeuralPrior <: AbstractPrior`
- **Dependency**: Optional `Flux.jl` or `Lux.jl` extension
- **API**: `NeuralPrior(model, lookback; transforms=identity)`

---

## 2. Reinforcement Learning for Dynamic Portfolio Optimisation

### 2.1 Motivation

Traditional mean‑variance optimisation solves a static problem. Reinforcement learning (RL) enables dynamic, multi‑period portfolio selection where the agent learns an optimal policy that adapts to changing market conditions. This is particularly valuable for transaction cost‑aware strategies and regime‑switching environments.

### 2.2 Mathematical Formulation

The portfolio optimisation problem is framed as a Markov Decision Process (MDP):

- **State** $s_t = (\mathbf{w}_{t-1}, \mathbf{r}_{t-k:t}, \text{features}_t)$
  - $\mathbf{w}_{t-1}$: previous portfolio weights.
  - $\mathbf{r}_{t-k:t}$: returns over the last $k$ periods.
  - $\text{features}_t$: additional market features.
- **Action** $a_t = \Delta \mathbf{w}_t$: the change in portfolio weights (so $\mathbf{w}_t = \mathbf{w}_{t-1} + a_t$).
- **Reward** $R(s_t, a_t) = \mathbf{w}_t^\top \mathbf{r}_{t+1} - \gamma \|\Delta \mathbf{w}_t\|_1$ (returns minus transaction costs), where:
  - $\mathbf{r}_{t+1}$ is the realised return vector for the next period.
  - $\gamma$ is a transaction cost penalty coefficient.
  - $\|\cdot\|_1$ denotes the L1 norm (sum of absolute changes).

The objective is to learn a policy $\pi_\theta(s_t)$ (a mapping from states to actions) that maximises the expected cumulative reward:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^T \gamma^t R(s_t, a_t)\right]$$

Here $\gamma$ (the discount factor) is a scalar in $[0,1)$ that weighs future rewards.

For a risk‑adjusted objective, a common multi‑objective reward incorporates both return and risk:

$$R(s_t, a_t) = \alpha \cdot \text{return}_t - \beta \cdot \text{risk}_t - \gamma \cdot \text{transaction\_cost}_t$$

where $\alpha$, $\beta$, and $\gamma$ are user‑defined coefficients.

### 2.3 Literature

- **Online Portfolio Selection**: `OnlinePortfolioSelection.jl` implements benchmark and novel OPS algorithms.
- **Inverse RL**: Heuristic‑guided inverse reinforcement learning with multi‑objective rewards.
- **JuMP + RL**: Integration of JuMP for constrained optimisation with RL for policy learning.

### 2.4 Sample Implementation (with detailed comments)

```julia
using ReinforcementLearning, PortfolioOptimisers, LinearAlgebra

# Define an RL‑based optimiser.
# T is the type of the RL policy (must implement RLBase.AbstractPolicy).
struct RLPortfolioOptimiser{T<:AbstractPolicy} <: AbstractOptimiser
    policy::T                # The policy network or table
    transaction_cost::Float64  # Cost per unit of absolute weight change
    risk_aversion::Float64     # Risk penalty coefficient (for reward)
end

# The optimiser call: given expected returns μ, covariance Σ, and previous weights,
# produce new weights using the policy.
function (opt::RLPortfolioOptimiser)(μ::AbstractVector, Σ::AbstractMatrix,
                                      prev_weights::AbstractVector)
    # Construct the state as a tuple; in practice, you'd encode this into a feature vector.
    state = (weights=prev_weights, μ=μ, Σ=Σ)
    # Ask the policy for an action (weight change).
    Δw = RLBase.action(opt.policy, state)
    # Apply the action and ensure weights stay non‑negative and sum to 1.
    w = clamp.(prev_weights + Δw, 0, 1)
    w ./= sum(w)   # Normalise to sum to 1
    return w
end

# Define the environment for training the RL agent.
struct PortfolioEnv <: AbstractEnv
    returns::Matrix{Float64}       # historical returns (T × N)
    transaction_cost::Float64
    risk_aversion::Float64
    t::Int                         # current time step
end

# The environment step: given state s and action a, produce reward and next state.
function (env::PortfolioEnv)(s, a)
    # s is a tuple (weights, μ, Σ) – simplified.
    w_new = s.weights + a
    w_new = clamp.(w_new, 0, 1)
    w_new ./= sum(w_new)
    # Compute reward: dot product with next returns, minus risk and transaction cost.
    r = dot(w_new, env.returns[:, env.t]) -
        env.risk_aversion * (w_new' * s.Σ * w_new) -
        env.transaction_cost * sum(abs.(a))
    return r, (weights=w_new, t=env.t+1)
end

# Usage (pseudocode)
# policy = DQNPolicy(...)  # from ReinforcementLearning.jl
# optimiser = RLPortfolioOptimiser(policy, 0.001, 1.0)
```

### 2.5 Integration Points

- **New optimiser type**: `RLPortfolioOptimiser <: AbstractOptimiser`
- **Dependency**: Optional `ReinforcementLearning.jl` extension
- **API**: `RLPortfolioOptimiser(policy; transaction_cost=0.001, risk_aversion=1.0)`

---

## 3. LLM-Enhanced Black‑Litterman Views

### 3.1 Motivation

The Black‑Litterman model addresses the sensitivity issues of traditional mean‑variance optimisation by incorporating investor views. Large Language Models (LLMs) can automatically generate these views by parsing financial news, reports, and social media sentiment. This bridges the gap between unstructured textual data and quantitative portfolio construction.

### 3.2 Mathematical Formulation

The Black‑Litterman model combines equilibrium returns $\boldsymbol{\pi}$ with investor views $\mathbf{Q}$:

$$\boldsymbol{\mu}_{BL} = \left[(\tau \boldsymbol{\Sigma})^{-1} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \boldsymbol{\Sigma})^{-1} \boldsymbol{\pi} + \mathbf{P}^\top \boldsymbol{\Omega}^{-1} \mathbf{Q}\right]$$

where:
- $\boldsymbol{\pi} \in \mathbb{R}^N$: equilibrium excess returns (e.g., from CAPM or market capitalisation weights).
- $\boldsymbol{\Sigma} \in \mathbb{R}^{N \times N}$: covariance matrix of asset returns.
- $\mathbf{P} \in \mathbb{R}^{K \times N}$: pick matrix that maps $K$ views to the $N$ assets.
- $\mathbf{Q} \in \mathbb{R}^K$: vector of expected returns for each view.
- $\boldsymbol{\Omega} \in \mathbb{R}^{K \times K}$: uncertainty matrix of the views (usually diagonal).
- $\tau$: scalar indicating the uncertainty of the prior (often set to $1/T$ or a small value).

LLM-generated sentiment scores $s_i \in [-1, 1]$ for each asset can be transformed into absolute views:

$$Q_i = \mu_{\text{market}} + s_i \cdot \sigma_i$$

where $\mu_{\text{market}}$ is the average market return and $\sigma_i$ is the asset's volatility. Alternatively, relative views can be expressed as:

$$Q_{i,j} = \mathbb{E}[r_i - r_j] = \alpha \cdot (s_i - s_j)$$

with $\alpha$ a scaling factor.

### 3.3 Literature

- **Multi‑LLM aggregation**: Aggregating sentiment from multiple LLMs enhances risk‑adjusted returns while mitigating model‑specific limitations.
- **FinBERT + Black‑Litterman**: Sentiment‑weighted views using FinBERT sentiment signals.
- **Agentic frameworks**: Unified agentic framework for regime‑aware portfolio optimisation with LLM signals.

### 3.4 Sample Implementation (with detailed comments)

```julia
using HTTP, JSON, PortfolioOptimisers

# A prior that obtains views from an LLM endpoint.
struct LLMBlackLittermanPrior <: AbstractPrior
    llm_endpoint::String      # URL of the LLM API
    view_uncertainty::Float64 # Variance of each view (assumed diagonal)
    tau::Float64              # Prior uncertainty scale
end

# Application: given returns and market caps, compute posterior mean and covariance.
function (prior::LLMBlackLittermanPrior)(returns::AbstractMatrix,
                                          market_caps::AbstractVector)
    N = size(returns, 2)

    # 1. Compute equilibrium returns (market‑cap weighted average of historical returns)
    π = market_caps ./ sum(market_caps)
    Σ = cov(returns)   # sample covariance

    # 2. Generate LLM views: returns a vector of (asset_index, sentiment_score)
    views = generate_llm_views(prior.llm_endpoint, returns)

    # 3. Construct pick matrix P (each view picks one asset) and Q (expected returns)
    K = length(views)
    P = zeros(K, N)
    Q = zeros(K)
    Ω = Diagonal(fill(prior.view_uncertainty, K))

    for (i, (idx, score)) in enumerate(views)
        P[i, idx] = 1.0
        # Transform sentiment to expected return (using market mean + sentiment * volatility)
        Q[i] = mean(π) + score * sqrt(diag(Σ)[idx])
    end

    # 4. Black‑Litterman posterior
    Σ_inv = inv(Σ)
    M = inv(prior.tau * Σ_inv + P' * inv(Ω) * P)
    μ = M * (prior.tau * Σ_inv * π + P' * inv(Ω) * Q)

    return μ, Σ   # return posterior mean and original covariance (or could adjust)
end

# Helper to call the LLM API.
function generate_llm_views(endpoint::String, returns::AbstractMatrix)
    # In practice, this would:
    # - fetch latest news/sentiment for each asset
    # - send to LLM endpoint
    # - parse returned sentiment scores
    # For demonstration, we return random scores.
    return [(i, rand(-1.0, 1.0)) for i in 1:size(returns, 2)]
end
```

### 3.5 Integration Points

- **New prior type**: `LLMBlackLittermanPrior <: AbstractPrior`
- **Configuration**: `LLMBlackLittermanPrior(; endpoint, view_uncertainty=0.1, tau=0.05)`
- **Extensions**: Support for multiple LLM aggregation schemes.

---

## 4. ESG Integration

### 4.1 Motivation

Environmental, Social, and Governance (ESG) considerations have become a critical component of modern portfolio management. Investors increasingly require portfolios that meet minimum ESG standards or optimise ESG scores alongside financial objectives.

### 4.2 Mathematical Formulation

Let $\mathbf{e} \in \mathbb{R}^N$ be the vector of ESG scores for each asset (higher = better). The optimisation problem with an ESG constraint becomes:

$$\begin{align}
\underset{\mathbf{w}}{\text{opt}} \quad & \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0 \\
& \mathbf{w}^\top \mathbf{e} \geq \bar{e} \quad \text{(minimum ESG score)}
\end{align}$$

Here $\bar{e}$ is the minimum acceptable ESG score for the portfolio.

Alternatively, ESG can be incorporated as a penalty term in the objective:

$$\underset{\mathbf{w}}{\text{opt}} \quad \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} + \phi \cdot \text{ESG\_penalty}(\mathbf{w})$$

where $\phi$ is a trade‑off parameter and $\text{ESG\_penalty}(\mathbf{w}) = -\mathbf{w}^\top \mathbf{e}$ (so we penalise low ESG).

Or with separate preferences for E, S, and G components:

$$\mathbf{w}^\top \mathbf{e}_E \geq \bar{e}_E, \quad \mathbf{w}^\top \mathbf{e}_S \geq \bar{e}_S, \quad \mathbf{w}^\top \mathbf{e}_G \geq \bar{e}_G$$

with $\mathbf{e}_E,\mathbf{e}_S,\mathbf{e}_G$ the component scores.

### 4.3 Literature

- **ESG‑constrained optimisation**: Optimal portfolio selection with affine equality constraints for ESG.
- **Binding ESG constraints**: Quadratic programming for portfolio optimisation under binding ESG constraints.
- **Performance impact**: Stricter ESG thresholds lead to reduced diversification and potential decline in risk‑adjusted returns.

### 4.4 Sample Implementation (with detailed comments)

```julia
using PortfolioOptimisers, LinearAlgebra

# Define an ESG constraint that enforces a minimum portfolio ESG score.
struct ESGConstraint <: AbstractConstraint
    esg_scores::Vector{Float64}   # per‑asset ESG scores
    min_score::Float64            # minimum acceptable portfolio score
end

# Check if a weight vector satisfies the constraint.
function (c::ESGConstraint)(w::AbstractVector)
    return dot(c.esg_scores, w) >= c.min_score
end

# Optionally, return a JuMP expression for use in optimisation.
function (c::ESGConstraint)()
    # This would return an expression to be used inside a JuMP macro.
    # Here we just illustrate the idea.
    return :(sum(esg_scores[i] * w[i] for i in 1:N) >= $min_score)
end

# Define an ESG objective component that adds an ESG term to the objective.
struct ESGObjective <: AbstractObjective
    esg_scores::Vector{Float64}
    weight::Float64   # trade‑off weight (positive = prefer higher ESG)
end

# Evaluate the ESG contribution.
function (obj::ESGObjective)(w::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix)
    # Maximising ESG score is equivalent to adding a positive linear term.
    return obj.weight * dot(obj.esg_scores, w)
end

# Usage with existing optimiser:
scores = [0.8, 0.6, 0.9, 0.3, 0.7]  # ESG scores for 5 assets
constraint = ESGConstraint(scores, 0.6)
objective = ESGObjective(scores, 0.1)

# Combine with mean‑variance objective (assuming SumObjective exists)
combined_obj = SumObjective(MeanVarianceObjective(λ=1.0), objective)
```

### 4.5 Integration Points

- **New constraint type**: `ESGConstraint <: AbstractConstraint`
- **New objective component**: `ESGObjective <: AbstractObjective`
- **API**: `ESGConstraint(scores, min_score)` and `ESGObjective(scores, weight)`

---

## 5. Scenario Generation & Stress Testing

### 5.1 Motivation

Mean‑variance optimisation relies on historical estimates that may not capture extreme events. Scenario generation enables users to test portfolio performance under hypothetical market conditions, while stress testing evaluates resilience to adverse scenarios.

### 5.2 Mathematical Formulation

Given a set of $S$ scenarios, each with return vector $\mathbf{r}^{(s)} \in \mathbb{R}^N$ and probability $p_s$, the scenario‑based portfolio optimisation is:

$$\begin{align}
\underset{\mathbf{w}}{\text{opt}} \quad & \sum_{s=1}^S p_s \cdot \mathbf{w}^\top \mathbf{r}^{(s)} - \frac{\lambda}{2} \sum_{s=1}^S p_s \cdot \left(\mathbf{w}^\top \mathbf{r}^{(s)} - \sum_{s'} p_{s'} \mathbf{w}^\top \mathbf{r}^{(s')}\right)^2 \\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0
\end{align}$$

Here:
- $\mathbf{r}^{(s)}$ is the return vector under scenario $s$.
- $p_s$ is the probability (or weight) of scenario $s$.
- The first term is the expected return, and the second is the variance of returns across scenarios.

For tail risk measures like Conditional Value‑at‑Risk (CVaR):

$$\text{CVaR}_\alpha(\mathbf{w}) = \mathbb{E}[- \mathbf{w}^\top \mathbf{r} \mid - \mathbf{w}^\top \mathbf{r} \geq \text{VaR}_\alpha]$$

where $\text{VaR}_\alpha$ is the $\alpha$‑quantile of the portfolio loss distribution.

Scenario generation methods include:
- **Historical bootstrapping**: resample from historical returns.
- **Parametric**: sample from fitted distributions (normal, t, skewed‑t).
- **Copula‑based**: capture dependence structures.
- **Regime‑switching**: generate scenarios conditional on market regimes.

### 5.3 Literature

- **TailRiskScenGen.jl**: Julia package for scenario generation with tail risk measures.
- **Scenario‑based risk**: Implementation in `PortfolioOptimisers.jl` already supports scenario‑based risk expressions.
- **Stress testing**: Backtesting and stress testing frameworks in Julia.

### 5.4 Sample Implementation (with detailed comments)

```julia
using Distributions, Copulas, PortfolioOptimisers

# Abstract type for scenario generators.
abstract type AbstractScenarioGenerator end

# Historical bootstrap: sample blocks of consecutive returns.
struct HistoricalScenarioGenerator <: AbstractScenarioGenerator
    returns::Matrix{Float64}   # T × N historical returns
    block_size::Int            # length of each block (scenario)
end

# Generate a set of scenarios.
function generate_scenarios(gen::HistoricalScenarioGenerator, n_scenarios::Int)
    T, N = size(gen.returns)
    # Randomly select starting indices for each block.
    indices = rand(1:T - gen.block_size + 1, n_scenarios)
    # Each scenario is a block of returns.
    scenarios = [gen.returns[i:i+gen.block_size-1, :] for i in indices]
    return scenarios
end

# Copula‑based generation: sample from a fitted copula.
struct CopulaScenarioGenerator <: AbstractScenarioGenerator
    returns::Matrix{Float64}
    copula::Copula          # copula object from e.g., Copulas.jl
    marginals::Vector{Distribution}  # fitted marginal distributions
end

function generate_scenarios(gen::CopulaScenarioGenerator, n_scenarios::Int)
    N = size(gen.returns, 2)
    # Sample from the copula (returns uniform [0,1]).
    U = rand(gen.copula, n_scenarios)   # size: N × n_scenarios
    # Transform to actual returns using the inverse CDFs.
    scenarios = hcat([quantile.(gen.marginals[i], U[i, :]) for i in 1:N]...)
    return scenarios
end

# Define a stress scenario.
struct StressScenario
    name::String                # e.g., "Financial Crisis"
    returns::Vector{Float64}    # asset return shocks
    probability::Float64        # probability of this scenario
end

# Apply stress by adjusting moments and re‑optimising.
function apply_stress(optimiser::AbstractOptimiser,
                      base_μ::AbstractVector,
                      base_Σ::AbstractMatrix,
                      stress::StressScenario)
    # Adjust expected returns by adding the stress shocks.
    μ_stress = base_μ + stress.returns
    # Optionally, adjust covariance (here we keep it unchanged for simplicity).
    Σ_stress = base_Σ
    return optimiser(μ_stress, Σ_stress)
end

# Usage:
gen = HistoricalScenarioGenerator(returns, 20)
scenarios = generate_scenarios(gen, 1000)
scenario_returns = mean.(scenarios)   # average return per scenario
scenario_cov = cov(hcat(scenarios...))  # covariance across scenarios
```

### 5.5 Integration Points

- **New module**: `ScenarioGeneration` with multiple generator types.
- **Integration**: Scenario‑based risk measures already exist in the library.
- **API**: `ScenarioPortfolioOptimiser(optimiser, scenario_generator)`

---

## 6. Dedicated Backtesting Framework

### 6.1 Motivation

While `PortfolioOptimisers.jl` provides optimisation capabilities, it lacks a formal backtesting framework for evaluating strategy performance over time. A dedicated backtesting module would enable walk‑forward validation, hyperparameter tuning, and performance attribution.

### 6.2 Mathematical Formulation

For a walk‑forward backtest with rebalancing frequency $\Delta t$ (e.g., monthly), at each rebalancing date $t$ we use the past $L$ observations to compute moments and optimise:

$$\mathbf{w}_{t+\Delta t} = \text{Optimise}\left(\mathcal{D}_{t-L:t}, \Theta\right)$$

where $\mathcal{D}_{t-L:t}$ is the lookback window of length $L$ and $\Theta$ are the optimisation parameters (e.g., risk aversion).

Portfolio value evolution:

$$V_{t+1} = V_t \cdot \left(1 + \mathbf{w}_t^\top \mathbf{r}_{t+1} - \text{TC}(\Delta \mathbf{w}_t)\right)$$

Here:
- $V_t$ is the portfolio value at time $t$.
- $\mathbf{r}_{t+1}$ is the vector of asset returns from $t$ to $t+1$.
- $\text{TC}(\Delta \mathbf{w}_t)$ is the transaction cost incurred when changing weights from $\mathbf{w}_{t-1}$ to $\mathbf{w}_t$.

A common transaction cost function:

$$\text{TC}(\Delta \mathbf{w}) = c_{\text{fixed}} \cdot \mathbf{1}^\top |\Delta \mathbf{w}| + c_{\text{prop}} \cdot \sum_i |\Delta w_i| \cdot |V_t|$$

where:
- $c_{\text{fixed}}$ is a fixed cost per unit of absolute weight change.
- $c_{\text{prop}}$ is a proportional cost (e.g., bid‑ask spread).

Performance metrics:
$$\text{Sharpe} = \frac{\mathbb{E}[R] - R_f}{\sigma(R)}, \quad \text{MaxDD} = \max_{t} \left(1 - \frac{V_t}{\max_{s \leq t} V_s}\right)$$

where $R$ is the portfolio return over the backtest period, $R_f$ is the risk‑free rate, and $\sigma(R)$ is the standard deviation of returns. MaxDD is the maximum drawdown.

### 6.3 Literature

- **Lucky.jl**: Reactive and async trading framework for rapid strategy deployment.
- **RiskPerf.jl**: High‑performance risk and performance analysis, avoiding temporary allocations.
- **OnlinePortfolioAnalytics.jl**: Quantitative portfolio analytics via online algorithms.

### 6.4 Sample Implementation (with detailed comments)

```julia
using DataFrames, Dates, PortfolioOptimisers

# Configuration for a backtest.
struct BacktestConfig
    lookback::Int               # number of historical days used for each optimisation
    rebalance_freq::Int         # days between rebalancing (e.g., 21 for monthly)
    transaction_cost::Float64   # proportional cost (fixed cost is omitted for simplicity)
    initial_capital::Float64
end

# Results container.
struct BacktestResult
    weights::Matrix{Float64}          # N × number_of_rebalances
    portfolio_values::Vector{Float64} # daily values
    returns::Vector{Float64}          # daily portfolio returns
    metrics::Dict{Symbol, Float64}    # Sharpe, maxDD, etc.
end

# Run the backtest.
function run_backtest(returns::Matrix{Float64},
                      optimiser_factory::Function,   # function that returns an optimiser
                      config::BacktestConfig)::BacktestResult

    T, N = size(returns)
    n_steps = T ÷ config.rebalance_freq   # number of rebalancing points

    weights = zeros(N, n_steps)
    values = zeros(n_steps + 1)   # we will expand to daily later; simplified here
    values[1] = config.initial_capital

    for step in 1:n_steps
        t = (step - 1) * config.rebalance_freq + 1   # start of lookback window
        lookback_end = min(t + config.lookback - 1, T)

        # Get historical data for the lookback window.
        hist_returns = returns[t:lookback_end, :]

        # Compute moments.
        μ = mean(hist_returns, dims=1)[:]
        Σ = cov(hist_returns)
        # Create the optimiser and solve.
        w = optimiser_factory()(μ, Σ)
        weights[:, step] = w

        # Simulate returns during the holding period.
        period_end = min(t + config.rebalance_freq - 1, T)
        period_returns = returns[t+1:period_end, :]

        for day in 1:size(period_returns, 1)
            # Apply transaction cost only on the first day after rebalancing.
            if day == 1 && step > 1
                Δw = w - weights[:, step-1]
                tc = config.transaction_cost * sum(abs.(Δw))
            else
                tc = 0.0
            end
            # Update portfolio value.
            idx = step + day   # index in values array (simplified)
            values[idx] = values[idx-1] * (1 + dot(w, period_returns[day, :]) - tc)
        end
    end

    # Compute performance metrics.
    port_returns = diff(values) ./ values[1:end-1]
    metrics = Dict(
        :sharpe => (mean(port_returns) / std(port_returns)) * sqrt(252),
        :max_drawdown => maximum(1 .- values ./ cummax(values)),
        :total_return => values[end] / values[1] - 1,
        :volatility => std(port_returns) * sqrt(252)
    )

    return BacktestResult(weights, values, port_returns, metrics)
end

# Usage:
config = BacktestConfig(
    lookback=252,
    rebalance_freq=21,
    transaction_cost=0.001,
    initial_capital=1_000_000
)
result = run_backtest(returns, () -> MeanVarianceOptimiser(λ=1.0), config)
```

### 6.5 Integration Points

- **New module**: `Backtesting` with configurable rebalancing, transaction costs, and metrics.
- **Integration**: Works with all existing optimiser types.
- **Extensions**: Parallel backtesting, cross‑validation, parameter grids.

---

## 7. Factor Model Integration

### 7.1 Motivation

Factor models like Fama‑French provide a structured approach to understanding asset returns. Deeper integration would enable factor‑based portfolio construction, risk decomposition, and performance attribution.

### 7.2 Mathematical Formulation

The Fama‑French 3‑factor model:

$$\mathbb{E}[r_i] - r_f = \beta_{i,M} \cdot \text{Mkt-RF} + \beta_{i,SMB} \cdot \text{SMB} + \beta_{i,HML} \cdot \text{HML}$$

where:
- $r_i$ is the return of asset $i$.
- $r_f$ is the risk‑free rate.
- $\text{Mkt-RF}$, $\text{SMB}$, $\text{HML}$ are the factor returns.
- $\beta_{i,M}, \beta_{i,SMB}, \beta_{i,HML}$ are the factor loadings of asset $i$.

The 5‑factor model adds profitability (RMW) and investment (CMA) factors.

For factor‑based portfolio construction, the optimisation can be framed as:

$$\begin{align}
\underset{\mathbf{w}}{\text{opt}} \quad & \mathbf{w}^\top \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w} \\
\text{s.t.} \quad & \mathbf{1}^\top \mathbf{w} = 1, \quad \mathbf{w} \geq 0 \\
& \mathbf{w}^\top \boldsymbol{\beta}_k = \beta_k^{\text{target}} \quad \text{for each factor } k
\end{align}$$

where $\boldsymbol{\beta}_k \in \mathbb{R}^N$ is the vector of loadings for factor $k$, and $\beta_k^{\text{target}}$ is the desired portfolio exposure to that factor.

### 7.3 Literature

- **FamaFrenchData.jl**: Julia package for accessing Ken French Data Library.
- **Factor models**: CAPM, Fama‑French 3 & 5‑Factor, Carhart Momentum.
- **Market regime**: HMM‑based regime estimation using Fama‑French factors.

### 7.4 Sample Implementation (with detailed comments)

```julia
using FamaFrenchData, PortfolioOptimisers, LinearAlgebra

# Extend the existing FactorPrior with automatic data fetching.
struct FactorPrior <: AbstractPrior
    factor_loadings::Matrix{Float64}   # N × K, where K is number of factors
    factor_returns::Matrix{Float64}    # T × K, historical factor returns
    factor_premiums::Vector{Float64}   # K, expected factor premiums (e.g., historical averages)
end

# Given asset returns (or just using the prior), compute expected returns and covariance.
function (prior::FactorPrior)(::AbstractMatrix)
    # Expected returns: factor_loadings * factor_premiums.
    μ = prior.factor_loadings * prior.factor_premiums

    # Covariance: factor covariance + idiosyncratic risk (diagonal).
    Σ_factor = cov(prior.factor_returns)
    # Idiosyncratic variance: variance of residuals after factor regression.
    # Here we approximate: residuals = factor_loadings * factor_returns' - mean(returns)
    # Actually we need asset returns to compute residuals; for demo we use a placeholder.
    # In practice, you'd have asset returns to compute residuals.
    # For illustration, we assume a diagonal matrix of residual variances.
    residuals = prior.factor_loadings * prior.factor_returns'  # N × T (transposed)
    # Assume residuals are mean‑zero; compute variance per asset.
    resid_var = var(residuals, dims=2)[:]   # length N
    Σ_idio = diagm(resid_var)
    Σ = prior.factor_loadings * Σ_factor * prior.factor_loadings' + Σ_idio

    return μ, Σ
end

# Constraint to enforce target factor exposures.
struct FactorConstraint <: AbstractConstraint
    factor_loadings::Matrix{Float64}   # N × K
    target_exposures::Vector{Float64}  # K
    tolerance::Float64                 # allowed deviation
end

function (c::FactorConstraint)(w::AbstractVector)
    exposures = c.factor_loadings' * w
    return all(abs.(exposures - c.target_exposures) .<= c.tolerance)
end

# Helper to fetch Fama‑French factors.
function fetch_ff_factors(which::String="3")
    # This would use FamaFrenchData.jl to download factor returns.
    # Returns (factor_returns, factor_names).
    # Example placeholder:
    return rand(100, parse(Int, which[1])), ["Mkt-RF", "SMB", "HML"][1:parse(Int, which[1])]
end
```

### 7.5 Integration Points

- **Enhance existing `FactorPrior`**: Add automatic data fetching from `FamaFrenchData.jl`.
- **New constraint**: `FactorExposureConstraint`.
- **New analytics**: Factor attribution and risk decomposition.

---

## 8. Interactive Visualisations with Makie

### 8.1 Motivation

The existing plotting extension is a good start. Moving towards interactive dashboards would significantly improve the user experience for exploration and strategy development.

### 8.2 Mathematical Formulation

Interactive visualisations should support:

1. **Efficient frontier**: Plot of risk (standard deviation) vs. return for different portfolios.
2. **Weight composition**: Treemap or stacked bar chart of portfolio weights.
3. **Risk contribution**: Breakdown of risk by asset or factor, e.g., $\text{risk contribution}_i = w_i (\boldsymbol{\Sigma} \mathbf{w})_i / (\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w})$.
4. **Time series**: Portfolio value, drawdowns, and rolling metrics.
5. **Scenario analysis**: Interactive sliders for stress testing parameters.

### 8.3 Literature

- **Makie.jl**: Interactive data visualisation ecosystem for Julia.
- **WGLMakie + JSServe**: Web‑based interactive dashboards.
- **GLMakie**: Native window interactive visualisations.

### 8.4 Sample Implementation (with detailed comments)

```julia
using GLMakie, PortfolioOptimisers

# Plot efficient frontier.
function plot_efficient_frontier(μ::AbstractVector, Σ::AbstractMatrix,
                                 points::Int=100)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Risk (σ)", ylabel="Return (μ)")

    # Generate frontier points by varying target return.
    returns = range(minimum(μ), maximum(μ), length=points)
    risks = Float64[]

    for r in returns
        try
            w = min_risk_for_return(μ, Σ, r)   # assumes such a function exists
            push!(risks, sqrt(w' * Σ * w))
        catch
            push!(risks, NaN)
        end
    end

    lines!(ax, risks, returns, color=:blue, linewidth=2)

    # Mark the maximum Sharpe portfolio.
    sharpe_w = max_sharpe(μ, Σ)
    sharpe_risk = sqrt(sharpe_w' * Σ * sharpe_w)
    sharpe_return = sharpe_w' * μ
    scatter!(ax, [sharpe_risk], [sharpe_return], color=:red, markersize=15)

    fig
end

# Interactive dashboard with sliders.
function interactive_portfolio_dashboard(returns::AbstractMatrix)
    μ = mean(returns, dims=1)[:]
    Σ = cov(returns)

    fig = Figure()

    # Layout: efficient frontier (top left), weights (bottom left), risk contribution (top right)
    ax1 = Axis(fig[1, 1], title="Efficient Frontier")
    ax2 = Axis(fig[2, 1], title="Portfolio Weights")
    ax3 = Axis(fig[1, 2], title="Risk Contribution")

    # Slider for risk aversion λ.
    sl = Slider(fig[2, 2], range=0.0:0.01:1.0, startvalue=0.5)

    on(sl.value) do λ
        # Solve mean‑variance optimisation.
        w = mean_variance_optimiser(μ, Σ, λ)   # placeholder
        # Update weight bar chart.
        barplot!(ax2, 1:length(w), w)
        # Update risk contribution bar chart.
        risk_contrib = (w .* (Σ * w)) / (w' * Σ * w)
        barplot!(ax3, 1:length(w), risk_contrib)
    end

    fig
end
```

### 8.5 Integration Points

- **Enhance existing plotting**: Move from static `Plots.jl` to interactive `Makie.jl`.
- **New module**: `PortfolioOptimisersVisualisation` or extension package.
- **Backends**: Support for `GLMakie` (desktop) and `WGLMakie` (web).

---

## 9. Summary of Integration Priorities

| Feature | Priority | Complexity | Dependencies | Estimated Effort |
|---------|----------|------------|--------------|------------------|
| **Backtesting Framework** | High | Medium | None | 2‑3 weeks |
| **ESG Integration** | High | Low | None | 1‑2 weeks |
| **Factor Model Enhancement** | High | Low | `FamaFrenchData.jl` | 1‑2 weeks |
| **Interactive Visualisations** | Medium | Medium | `Makie.jl` | 2‑3 weeks |
| **LLM Black‑Litterman** | Medium | Medium | HTTP/JSON | 2‑3 weeks |
| **Deep Learning Priors** | Medium | High | `Flux.jl`/`Lux.jl` | 3‑4 weeks |
| **RL Optimisers** | Low | High | `ReinforcementLearning.jl` | 4‑6 weeks |
| **Scenario Generation** | Medium | Medium | `Distributions.jl` | 2‑3 weeks |

---

## 10. Architectural Recommendations

### 10.1 Extension System

Consider adopting a plugin/extension system where advanced features (DL, RL, LLM) are optional dependencies:

```julia
# In Project.toml
[extras]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ReinforcementLearning = "..."

[extensions]
DeepLearningExt = ["Flux"]
RLExt = ["ReinforcementLearning"]
LLMExt = ["HTTP", "JSON"]
```

### 10.2 API Consistency

All new features should follow the existing API patterns:
- Priors: `<Name>Prior <: AbstractPrior`
- Optimisers: `<Name>Optimiser <: AbstractOptimiser`
- Constraints: `<Name>Constraint <: AbstractConstraint`

### 10.3 Documentation

Each feature should include:
- Mathematical formulation (LaTeX) with all variables defined.
- Usage examples.
- References to literature.
- Performance considerations.

---

This roadmap provides a comprehensive foundation for extending `PortfolioOptimisers.jl` with cutting‑edge features while maintaining the library's existing elegance and performance. Each section can be implemented incrementally, allowing for steady progress and community feedback.
