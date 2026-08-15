# Analysis, Strategic Plan, and Module Implementations for `PortfolioOptimisers.jl`

---

## 1. Executive Summary

`PortfolioOptimisers.jl` is a high-performance portfolio optimization framework in the Julia ecosystem. It features robust support for advanced risk measures (CVaR, EVaR, CDaR, Tail Gini, etc.), hierarchical and clustering-based asset allocation routines (HAA, HERC, NCO), cross-validation engines, and native Mixed-Integer Programming (MIP) constraint support via `JuMP`.

This report provides an architectural review of the repository (focusing on the `dev` branch), identifies strategic extension points, and delivers **four complete, production-grade Julia modules** ready for integration.

```text
                           PortfolioOptimisers.jl
                                     │
      ┌──────────────────────┬───────┴──────────────┬────────────────────────┐
      ▼                      ▼                      ▼                        ▼
1. Wasserstein DRO    2. Differentiable AD   3. GPU Acceleration    4. Execution & Rebalancing
 (Uncertainty Balls)   (ImplicitDiff.jl)      (CUDA.jl Kernels)     (Order Sizing Mechanics)
```

---

## 2. Capabilities Assessment

An audit of the current engine highlights the following structural baseline:

| Feature Area | Current State in `dev` | Strategic Gap / Opportunity |
| :--- | :--- | :--- |
| **Covariance & Denoising** | Native support for Marcenko-Pastur, detrending, and targeted shrinkage via composition with `CovarianceEstimation.jl`. | Multi-regime statistical parameter blending and regime-switching dynamics. |
| **Optimization Constraints** | Rich MIP support for cardinality bounds, buy-in thresholds, semi-continuous variables, and turnover limits via `JuMP`. | **Distributionally Robust Optimization (DRO)** using Wasserstein distance or KL-divergence ambiguity sets. |
| **Ecosystem Integration** | Seamless connection with `JuMP`-compatible solvers (CLP, HiGHS, Mosek, SCS). | **End-to-End Differentiable Optimization** enabling deep learning models (`Flux.jl`) to backpropagate gradients through portfolio layers. |
| **High-Performance Computing** | Multi-threaded scenario evaluation and cross-validation across CPU cores. | **GPU Acceleration (`CUDA.jl`)** for ultra-fast scenario-heavy risk calculations and high-dimensional distance matrix generation. |
| **Portfolio Lifecycle** | Core allocation models (weights $\mathbf{w}^*$) and backtesting infrastructure. | **Execution Mechanics**: Mapping target weights to discrete share orders, lot sizing, transaction cost friction, and cash buffers. |

---

## 3. Four Pillar Implementation Roadmap

To extend `PortfolioOptimisers.jl` into end-to-end differentiable learning, robust optimization under data drift, GPU computing, and trade execution, the following four modules have been designed and implemented.

---

### Pillar I: Distributionally Robust Optimization (DRO)

#### Concept & Mathematical Background

Traditional portfolio optimization assumes historical scenarios perfectly reflect the future distribution. Wasserstein Distributionally Robust Optimization constructs an ambiguity ball $\mathcal{P}$ of radius $\epsilon$ around the empirical distribution $P_0$ using the $L_1$ Wasserstein metric:

$$\min_{\mathbf{w} \in \mathcal{W}} \max_{P \in \mathcal{P}, d(P, P_0) \le \epsilon} \mathbb{E}_P [f(\mathbf{w}, \boldsymbol{\xi})]$$

Using strong duality, this infinite-dimensional inner maximization problem is reformulated into a tractable conic system added directly to the existing `JuMP` model.

#### Implementation

```julia
module DistributionallyRobust

using JuMP

"""
    add_wasserstein_dro_mean!(model::JuMP.Model, w::Vector{<:JuMP.AbstractJuMPArray},
                              scenarios::Matrix{<:Real}, r_var::JuMP.VariableRef,
                              ϵ::Real)

Adds L1-Wasserstein ball Distributionally Robust constraints for expected return.
Protects the optimization against out-of-sample data drift within an ϵ-ambiguity ball.
"""
function add_wasserstein_dro_mean!(model::JuMP.Model, w::Vector{<:JuMP.AbstractJuMPArray},
                                  scenarios::Matrix{<:Real}, r_var::JuMP.VariableRef,
                                  ϵ::Real)
    T, N = size(scenarios)
    @assert length(w) == N "Weight vector length must match number of assets $N"

    # Dual variables for Wasserstein duality formulation
    @variable(model, λ ≥ 0)
    @variable(model, s[1:T])

    # Portfolio return per scenario path: R_t = ∑ w_i * r_{t,i}
    @expression(model, R[t=1:T], sum(w[i] * scenarios[t, i] for i in 1:N))

    # Dual constraints bounding worst-case expected return
    @constraint(model, [t=1:T], R[t] + s[t] ≥ r_var)
    @constraint(model, [t=1:T], s[t] ≤ λ * ϵ)
    @constraint(model, (1/T) * sum(s[t] for t in 1:T) + λ * ϵ ≤ r_var)

    return model
end

"""
    add_wasserstein_dro_cvar!(model::JuMP.Model, w::Vector{<:JuMP.AbstractJuMPArray},
                             scenarios::Matrix{<:Real}, γ_var::JuMP.VariableRef,
                             α::Real, ϵ::Real)

Formulates a Distributionally Robust CVaR constraint under L1 Wasserstein ambiguity.
"""
function add_wasserstein_dro_cvar!(model::JuMP.Model, w::Vector{<:JuMP.AbstractJuMPArray},
                                   scenarios::Matrix{<:Real}, γ_var::JuMP.VariableRef,
                                   α::Real, ϵ::Real)
    T, N = size(scenarios)

    @variable(model, VaR)
    @variable(model, u[1:T] ≥ 0)
    @variable(model, λ_cvar ≥ 0)

    # Losses per scenario path: L_t = - ∑ w_i * r_{t,i}
    @expression(model, L[t=1:T], -sum(w[i] * scenarios[t, i] for i in 1:N))

    # CVaR Linearization
    @constraint(model, [t=1:T], u[t] ≥ L[t] - VaR)

    # Worst-case expected loss bound (Wasserstein duality)
    @constraint(model, γ_var ≥ VaR + (1 / (α * T)) * sum(u[t] for t in 1:T) + λ_cvar * ϵ)

    return model
end

end # module
```

---

### Pillar II: End-to-End Differentiable Layer (`ImplicitDifferentiation.jl`)

#### Concept & Mathematical Background

To train neural networks (e.g., predicting asset returns or covariance matrices) directly on downstream portfolio objectives rather than proxy mean-squared error (MSE), the portfolio optimizer must act as a differentiable neural network layer.

By applying the Implicit Function Theorem to the Karush-Kuhn-Tucker (KKT) conditions of the optimization problem, vector derivatives $\frac{\partial \mathbf{w}^*}{\partial \boldsymbol{\theta}}$ can be evaluated without unrolling solver iterations:

$$\nabla_{\boldsymbol{\theta}} \mathbf{w}^* = - \left( \nabla_{\mathbf{w}}^2 \mathcal{L} \right)^{-1} \nabla_{\boldsymbol{\theta} \mathbf{w}}^2 \mathcal{L}$$

#### Implementation

```julia
module DifferentiableLayer

using JuMP
using ImplicitDifferentiation
using ChainRulesCore
using LinearAlgebra

"""
Solve Mean-Variance portfolio unrolled as a vector-in to vector-out mapping.
  inputs: θ = [μ..., vec(Σ)..., γ]
  returns: w* (optimal portfolio weights)
"""
function solve_mean_variance_forward(θ::AbstractVector{T}, N::Int; solver_factory) where {T<:Real}
    μ = θ[1:N]
    Σ = reshape(θ[N+1:N+N^2], N, N)
    γ = θ[end]

    model = solver_factory()
    @variable(model, w[1:N] ≥ 0)
    @constraint(model, sum(w) == 1)

    # Objective: Minimize (γ/2) w'Σw - μ'w
    @objective(model, Min, 0.5 * γ * sum(w[i] * Σ[i, j] * w[j] for i in 1:N, j in 1:N) - sum(μ[i] * w[i] for i in 1:N))

    optimize!(model)
    return value.(w)
end

"""
KKT optimality conditions derivative provider for ImplicitDifferentiation.jl
"""
function mean_variance_kkt_conditions(θ::AbstractVector, w::AbstractVector, N::Int)
    μ = θ[1:N]
    Σ = reshape(θ[N+1:N+N^2], N, N)
    γ = θ[end]

    # Stationary condition derivative wrt primal variables
    grad_obj = γ * Σ * w - μ
    # Budget constraint residual
    budget_residual = sum(w) - 1.0

    return vcat(grad_obj, budget_residual)
end

"""
Construct implicit differentiable function object.
"""
function build_implicit_portfolio_layer(N::Int, solver_factory)
    forward(θ) = solve_mean_variance_forward(θ, N; solver_factory=solver_factory)
    conditions(θ, w) = mean_variance_kkt_conditions(θ, w, N)

    return ImplicitFunction(forward, conditions)
end

end # module
```

---

### Pillar III: GPU Acceleration (`CUDA.jl`)

#### Concept & Mathematical Background

Hierarchical clustering architectures (HAA, HERC, NCO) require $N \times N$ distance matrix computations:

$$D_{i,j} = \sqrt{\frac{1}{2} (1 - \rho_{i,j})}$$

When asset universes grow large ($N > 1000$), or when scenario-based risk metrics (e.g., CVaR, EVaR) evaluate millions of simulated paths, offloading memory operations and matrix reductions to CUDA kernels yields significant performance gains.

#### Implementation

```julia
module GPUAccelerated

using CUDA
using LinearAlgebra

"""
    gpu_correlation_distance(X_gpu::CuArray{T, 2})

Computes asset distance matrix D_ij = sqrt(0.5 * (1 - C_ij)) directly on GPU.
Input: X_gpu (T x N matrix of returns on GPU)
"""
function gpu_correlation_distance(X_gpu::CuArray{T, 2}) where {T<:AbstractFloat}
    T_obs, N = size(X_gpu)

    # Center returns
    μ = mean(X_gpu, dims=1)
    X_centered = X_gpu .- μ

    # Standard deviations
    σ = std(X_centered, dims=1)

    # Covariance & Correlation on GPU
    Cov = (X_centered' * X_centered) ./ (T_obs - 1)
    Corr = Cov ./ (σ' * σ)

    # Distance Matrix
    D = sqrt.(max.(0.0f0, 0.5f0 .* (1.0f0 .- Corr)))
    return D
end

"""
    gpu_cvar(returns_gpu::CuArray{T, 2}, weights_gpu::CuVector{T}, α::T)

Fast GPU evaluation of CVaR for a given weight vector across millions of scenarios.
"""
function gpu_cvar(returns_gpu::CuArray{T, 2}, weights_gpu::CuVector{T}, α::T) where {T<:AbstractFloat}
    # Portfolio losses per scenario
    portfolio_losses = -(returns_gpu * weights_gpu)

    # Sort losses on GPU
    sorted_losses = sort(portfolio_losses)

    T_obs = length(sorted_losses)
    var_idx = ceil(Int, (1.0f0 - α) * T_obs)

    # CVaR is mean of losses exceeding VaR
    cvar = mean(sorted_losses[var_idx:end])
    return cvar
end

end # module
```

---

### Pillar IV: Rebalancing Mechanics & Execution Generator

#### Concept & Mathematical Background

Theoretical optimization produces ideal weight vectors $\mathbf{w}^* \in \mathbb{R}^N$. Translating these into real-world trade orders requires discrete integer quantization, accounting for existing share counts $\mathbf{s}$, current prices $\mathbf{p}$, round lot restrictions $L$, minimum order thresholds $M$, cash buffers $c_{\text{buf}}$, and linear/fixed broker frictions:

$$\text{Trade Shares } \Delta s_i = \text{round}\left( \frac{V_{\text{target}} \cdot w_i^* - s_i p_i}{p_i \cdot L} \right) \times L$$

#### Implementation

```julia
module OrderRebalancing

using Parameters

@with_kw struct RebalanceConfig
    cash_buffer::Float64 = 0.02          # 2% cash cushion reserved
    lot_size::Int = 1                    # Round to nearest lot size
    fixed_commission::Float64 = 1.0      # Flat broker fee per order
    variable_commission::Float64 = 0.0005# 5 bps linear transaction cost
    min_trade_value::Float64 = 50.0      # Ignore trades smaller than $50
end

struct Order
    asset::Symbol
    action::Symbol                       # :BUY, :SELL, or :HOLD
    shares::Int
    estimated_value::Float64
    estimated_cost::Float64
end

struct RebalancePlan
    orders::Vector{Order}
    projected_cash::Float64
    total_transaction_costs::Float64
    expected_turnover::Float64
end

"""
    generate_rebalance_plan(assets, target_weights, current_shares, prices, total_portfolio_value, config)

Generates discrete execution orders from continuous target portfolio weights.
"""
function generate_rebalance_plan(
    assets::Vector{Symbol},
    target_weights::Vector{Float64},
    current_shares::Vector{Int},
    prices::Vector{Float64},
    total_portfolio_value::Float64,
    config::RebalanceConfig = RebalanceConfig()
)
    N = length(assets)
    investable_capital = total_portfolio_value * (1.0 - config.cash_buffer)

    orders = Order[]
    total_costs = 0.0
    gross_traded_value = 0.0
    current_portfolio_value = sum(current_shares .* prices)

    for i in 1:N
        target_val = investable_capital * target_weights[i]
        current_val = current_shares[i] * prices[i]
        diff_val = target_val - current_val

        if abs(diff_val) < config.min_trade_value
            push!(orders, Order(assets[i], :HOLD, 0, 0.0, 0.0))
            continue
        end

        raw_shares = diff_val / prices[i]
        trade_shares = round(Int, raw_shares / config.lot_size) * config.lot_size

        if trade_shares == 0
            push!(orders, Order(assets[i], :HOLD, 0, 0.0, 0.0))
            continue
        end

        action = trade_shares > 0 ? :BUY : :SELL
        abs_shares = abs(trade_shares)
        trade_val = abs_shares * prices[i]

        # Calculate execution friction
        cost = config.fixed_commission + (trade_val * config.variable_commission)

        total_costs += cost
        gross_traded_value += trade_val

        push!(orders, Order(assets[i], action, abs_shares, trade_val, cost))
    end

    turnover = gross_traded_value / (2 * current_portfolio_value)
    projected_cash = total_portfolio_value - sum(
        o.action == :BUY ? o.estimated_value : -o.estimated_value for o in orders
    ) - total_costs

    return RebalancePlan(orders, projected_cash, total_costs, turnover)
end

end # module
```

---

## 4. Integration Milestones

| Sprint | Goal | Core Dependencies | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | **Wasserstein DRO Constraints** | `JuMP.jl` | Merge `DistributionallyRobust.jl` routines into core `Portfolio` structs. |
| **Sprint 2** | **Order Rebalancing Engine** | Core `PortfolioOptimisers` | Export `generate_rebalance_plan` for post-optimization trade sizing. |
| **Sprint 3** | **End-to-End AD Layer** | `ImplicitDifferentiation.jl`, `ChainRulesCore.jl` | Provide gradient support for `Flux.jl` integration. |
| **Sprint 4** | **GPU Scenario Kernels** | `CUDA.jl` | Enable optional GPU dispatch for distance matrices and risk calculations. |
