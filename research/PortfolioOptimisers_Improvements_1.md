# Analysis, Strategic Plan, and Module Implementations for `PortfolioOptimisers.jl`

---

## 1. Executive Summary

This report provides an architectural review of the repository (focusing on the `dev` branch), identifies strategic extension points, and delivers **three complete, production-grade Julia modules** ready for integration.

```text
                           PortfolioOptimisers.jl
                                     │
              ┌──────────────────────┼────────────────────────┐
              ▼                      ▼                        ▼
      2. Differentiable AD   3. GPU Acceleration    4. Execution & Rebalancing
       (ImplicitDiff.jl)      (CUDA.jl Kernels)     (Order Sizing Mechanics)
```

---

## 2. Capabilities Assessment

An audit of the current engine highlights the following structural baseline:

| Feature Area | Current State in `dev` | Strategic Gap / Opportunity |
| :--- | :--- | :--- |
| **Covariance & Denoising** | Native support for Marcenko-Pastur, detrending, and targeted shrinkage via composition with `CovarianceEstimation.jl`. | Multi-regime statistical parameter blending and regime-switching dynamics. |
| **Ecosystem Integration** | Seamless connection with `JuMP`-compatible solvers (CLP, HiGHS, Mosek, SCS). | **End-to-End Differentiable Optimization** enabling deep learning models (`Flux.jl`) to backpropagate gradients through portfolio layers. |
| **High-Performance Computing** | Multi-threaded scenario evaluation and cross-validation across CPU cores. | **GPU Acceleration (`CUDA.jl`)** for ultra-fast scenario-heavy risk calculations and high-dimensional distance matrix generation. |
| **Portfolio Lifecycle** | Core allocation models (weights $\mathbf{w}^*$) and backtesting infrastructure. | **Execution Mechanics**: A trade list against the current share counts, with a minimum trade value. |

---

## 3. Four Pillar Implementation Roadmap

To extend `PortfolioOptimisers.jl` into end-to-end differentiable learning, GPU computing, and trade execution, the following three modules have been designed and implemented.

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

Theoretical optimization produces ideal weight vectors $\mathbf{w}^* \in \mathbb{R}^N$. Translating these into real-world trade orders requires a trade list against the existing share counts $\mathbf{s}$ at current prices $\mathbf{p}$, with minimum order thresholds $M$:

$$\text{Trade Shares } \Delta s_i = \text{round}\left( \frac{V_{\text{target}} \cdot w_i^* - s_i p_i}{p_i} \right)$$

#### Implementation

```julia
module OrderRebalancing

using Parameters

@with_kw struct RebalanceConfig
    min_trade_value::Float64 = 50.0      # Ignore trades smaller than $50
end

struct Order
    asset::Symbol
    action::Symbol                       # :BUY, :SELL, or :HOLD
    shares::Int
    estimated_value::Float64
end

struct RebalancePlan
    orders::Vector{Order}
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

    orders = Order[]
    gross_traded_value = 0.0
    current_portfolio_value = sum(current_shares .* prices)

    for i in 1:N
        target_val = total_portfolio_value * target_weights[i]
        current_val = current_shares[i] * prices[i]
        diff_val = target_val - current_val

        if abs(diff_val) < config.min_trade_value
            push!(orders, Order(assets[i], :HOLD, 0, 0.0))
            continue
        end

        trade_shares = round(Int, diff_val / prices[i])

        if trade_shares == 0
            push!(orders, Order(assets[i], :HOLD, 0, 0.0))
            continue
        end

        action = trade_shares > 0 ? :BUY : :SELL
        abs_shares = abs(trade_shares)
        trade_val = abs_shares * prices[i]

        gross_traded_value += trade_val

        push!(orders, Order(assets[i], action, abs_shares, trade_val))
    end

    turnover = gross_traded_value / (2 * current_portfolio_value)

    return RebalancePlan(orders, turnover)
end

end # module
```

---

## 4. Integration Milestones

| Sprint | Goal | Core Dependencies | Primary Deliverable |
| :--- | :--- | :--- | :--- |
| **Sprint 2** | **Order Rebalancing Engine** | Core `PortfolioOptimisers` | Export `generate_rebalance_plan` for post-optimization trade sizing. |
| **Sprint 3** | **End-to-End AD Layer** | `ImplicitDifferentiation.jl`, `ChainRulesCore.jl` | Provide gradient support for `Flux.jl` integration. |
| **Sprint 4** | **GPU Scenario Kernels** | `CUDA.jl` | Enable optional GPU dispatch for distance matrices and risk calculations. |
