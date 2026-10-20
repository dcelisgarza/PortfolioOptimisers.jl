# PortfolioOptimisers.jl – Feature Roadmap

**Status:** Living document derived from discussion of the `dev` branch (2026-08).
**Purpose:** Prioritised suggestions for future development that do **not** duplicate existing capabilities.

Sample code below is **standalone** (JuMP / plain Julia). It is illustrative only and is **not** integrated into the package API.

---

## Priority 1 – High Leverage / Practical Impact

### 1. Dedicated Rebalancing Policy Layer

**Idea**
Elevate `TimeDependent` into (or wrap it with) first-class **rebalancing policies**:

- Threshold-based (turnover, drift from target, risk-budget violation)
- Cost-aware (trade only when expected benefit > estimated transaction cost)
- Hybrid (threshold + cost)

**Standalone sample**

```julia
using Dates, LinearAlgebra

abstract type RebalancePolicy end

struct ThresholdRebalance <: RebalancePolicy
    tol::Float64            # max L1 drift from target before trade
    cost_model            # callable (w_old, w_new) -> cost
end

struct CostAwareRebalance <: RebalancePolicy
    benefit_estimator     # expected utility improvement
    cost_model
    min_net_benefit::Float64
end

"""
Simple backtest loop that applies a rebalancing policy.
`optimise_fn(rd_window) -> w` is any optimiser returning weights.
"""
function run_rebalance(policy::ThresholdRebalance, prices::AbstractMatrix,
                       dates::Vector{Date}, optimise_fn;
                       lookback::Int = 252)
    T, n = size(prices)
    w = fill(1/n, n)                       # start equal-weight
    history = Vector{Vector{Float64}}()
    turnovers = Float64[]
    costs = Float64[]

    for t in (lookback+1):T
        # current drifted weights (price relative)
        rel = prices[t, :] ./ prices[t-1, :]
        w_drifted = w .* rel
        w_drifted ./= sum(w_drifted)

        target = optimise_fn(prices[t-lookback:t-1, :])
        drift = norm(w_drifted - target, 1)

        if drift > policy.tol
            c = policy.cost_model(w_drifted, target)
            w = target
            push!(turnovers, drift)
            push!(costs, c)
        else
            w = w_drifted
            push!(turnovers, 0.0)
            push!(costs, 0.0)
        end
        push!(history, copy(w))
    end
    return history, turnovers, costs
end

# example cost model
linear_tc(w_old, w_new; bps = 5.0) = bps * 1e-4 * norm(w_new - w_old, 1)
```

---

### 2. Richer Transaction-Cost & Tax Modelling

**Idea**

- Piecewise-linear / quadratic transaction costs
- Simple tax-lot / short-term vs long-term capital-gains approximations

**Simple tax-lot sketch**

```julia
# FIFO lot accounting (standalone, not optimised)
struct Lot
    shares::Float64
    cost_basis::Float64
    date::Date
end

function realise_gains!(lots::Vector{Lot}, shares_to_sell::Float64, price::Float64, today::Date;
                        st_threshold = Day(365))
    gains_st = 0.0
    gains_lt = 0.0
    remaining = shares_to_sell
    while remaining > 1e-12 && !isempty(lots)
        lot = lots[1]
        take = min(lot.shares, remaining)
        gain = take * (price - lot.cost_basis)
        if today - lot.date < st_threshold
            gains_st += gain
        else
            gains_lt += gain
        end
        lot.shares -= take
        remaining -= take
        if lot.shares < 1e-12
            popfirst!(lots)
        end
    end
    return gains_st, gains_lt
end
```

---

### 3. Multi-Period / Rolling-Horizon Optimisation

**Idea**
Limited-horizon multi-period formulations with intermediate rebalancing and transaction costs.

**Standalone sample (2-period mean-variance + linear TC)**

```julia
using JuMP, Clarabel, LinearAlgebra

function two_period_mvo(μ1, Σ1, μ2, Σ2, w0;
                        γ = 1.0, tc = 1e-3)
    n = length(μ1)
    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, w1[1:n] >= 0)
    @variable(model, w2[1:n] >= 0)
    @variable(model, abs_t1[1:n] >= 0)
    @variable(model, abs_t2[1:n] >= 0)

    @constraint(model, sum(w1) == 1)
    @constraint(model, sum(w2) == 1)

    @constraint(model, abs_t1 .>=  w1 - w0)
    @constraint(model, abs_t1 .>= -w1 + w0)
    @constraint(model, abs_t2 .>=  w2 - w1)
    @constraint(model, abs_t2 .>= -w2 + w1)

    # expected terminal wealth utility (approx)
    # maximise μ1'w1 + μ2'w2 - γ (w1'Σ1 w1 + w2'Σ2 w2) - tc*(|Δ1|+|Δ2|)
    @objective(model, Max,
        μ1 ⋅ w1 + μ2 ⋅ w2
        - γ * (w1' * Σ1 * w1 + w2' * Σ2 * w2)
        - tc * (sum(abs_t1) + sum(abs_t2))
    )
    optimize!(model)
    return value.(w1), value.(w2)
end
```

---

## Priority 2 – Robustness & Scenario Tools

### 5. Scenario Generation & Stress Testing

**Idea**
Vine-copula (or simpler Gaussian / t) scenario generators + conditional stress sampling.

**Standalone sample (Gaussian copula scenarios + conditional stress)**

```julia
using LinearAlgebra, Distributions, Random

function gaussian_scenarios(μ, Σ, n_scen; rng = Random.default_rng())
    L = cholesky(Symmetric(Σ)).L
    Z = randn(rng, length(μ), n_scen)
    return μ .+ L * Z                       # n × n_scen
end

"""
Stress asset `idx` to a fixed return `stress_ret` and
resample the remaining assets conditionally (Gaussian case).
"""
function conditional_stress(μ, Σ, idx::Int, stress_ret; n_scen = 10_000, rng = Random.default_rng())
    n = length(μ)
    # partition
    others = setdiff(1:n, idx)
    μ1, μ2 = μ[idx], μ[others]
    Σ11 = Σ[idx, idx]
    Σ12 = Σ[idx, others]'
    Σ22 = Σ[others, others]

    # conditional mean & cov
    μ2_cond = μ2 .+ Σ12 * (stress_ret - μ1) / Σ11
    Σ22_cond = Σ22 .- Σ12 * Σ12' / Σ11

    scen = zeros(n, n_scen)
    scen[idx, :] .= stress_ret
    scen[others, :] = gaussian_scenarios(μ2_cond, Symmetric(Σ22_cond), n_scen; rng)
    return scen
end

# evaluate a portfolio under stressed scenarios
portfolio_returns(w, scen) = w' * scen          # 1 × n_scen
```

---

### 6. Uncertainty Quantification on the Weights Themselves

**Idea**
Bootstrap distributions of optimal weights, conformal intervals, selection confidence sets.

**Standalone sample (bootstrap weights)**

```julia
using Statistics, Random

function bootstrap_weights(returns::AbstractMatrix, optimise_fn;
                           B = 500, block = 0.05, rng = Random.default_rng())
    T, n = size(returns)
    W = zeros(n, B)
    for b in 1:B
        idx = rand(rng, 1:T, T)                 # residual bootstrap / iid
        W[:, b] = optimise_fn(returns[idx, :])
    end
    mean_w = vec(mean(W; dims = 2))
    lo = [quantile(W[i, :], alpha/2) for i in 1:n]
    hi = [quantile(W[i, :], 1 - alpha/2) for i in 1:n]
    return mean_w, lo, hi, W
end

# crude conformal-style interval for next-period portfolio return
function conformal_portfolio_interval(w, calibration_returns, alpha = 0.1)
    # nonconformity = -portfolio return (we want high return)
    scores = -calibration_returns * w
    q = quantile(scores, 1 - alpha)
    # prediction set: {r | -w'r ≤ q} ⇒ w'r ≥ -q
    return -q   # lower bound on portfolio return at level 1-α
end
```

---

## Priority 3 – Differentiable & Online Extensions

### 7. Differentiable Portfolio Layers / End-to-End Learning

**Idea**
Expose differentiable optimisers (via DiffOpt.jl or implicit differentiation) so that moments / hyper-parameters can be learned by gradient descent on a downstream loss.

**Standalone sketch (implicit differentiation through a QP)**

```julia
# Pseudocode – requires DiffOpt.jl or manual KKT differentiation
using JuMP, DiffOpt, Clarabel

function differentiable_mvo_layer(μ, Σ; γ = 1.0)
    model = Model(() -> DiffOpt.diff_optimizer(Clarabel.Optimizer))
    set_silent(model)
    n = length(μ)
    @variable(model, w[1:n] >= 0)
    @constraint(model, sum(w) == 1)
    @objective(model, Min, γ * w' * Σ * w - μ ⋅ w)
    optimize!(model)

    # forward: optimal weights
    ŵ = value.(w)

    # backward: ∂ŵ/∂μ, ∂ŵ/∂Σ via DiffOpt
    # (see DiffOpt.jl docs for MOI.Forward / Reverse)
    return ŵ, model
end

# training loop sketch
# for epoch in 1:E
#     μθ = network(features; θ)
#     ŵ, model = differentiable_mvo_layer(μθ, Σ)
#     loss = downstream_performance(ŵ, realised_returns)
#     grads = gradient(loss, θ)   # flows through DiffOpt
#     update!(θ, grads)
```

---

## Priority 4 – Graph & Network Extensions

### 9. Richer Graph / Network Constraints & Objectives

**Idea**
Laplacian regularisation on top of existing MST / clustering graphs.

**Standalone sample**

```julia
using JuMP, Clarabel, LinearAlgebra

"""
Mean-variance with Laplacian regularisation:
  min  w'Σw - μ'w + λ w' L w
where L is the graph Laplacian (encourages neighbouring assets to have similar weights).
"""
function mvo_laplacian(μ, Σ, L; λ = 0.1, γ = 1.0)
    n = length(μ)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:n] >= 0)
    @constraint(model, sum(w) == 1)
    @objective(model, Min, γ * w' * Σ * w - μ ⋅ w + λ * w' * L * w)
    optimize!(model)
    return value.(w)
end

# build Laplacian from adjacency
laplacian(A) = Diagonal(vec(sum(A; dims = 2))) - A
```

---

## Priority 5 – Higher-Order & Non-Convex / Advanced Risk

### 11. Automatic Differentiation Through the Whole Pipeline

**Idea**
Sensitivities of optimal weights w.r.t. μ, Σ, constraint RHS, uncertainty radii.

**Standalone sketch (finite-difference fallback + analytic for unconstrained min-var)**

```julia
function minvar_sensitivity(Σ)
    # analytic dw/dΣ for unconstrained minimum-variance
    # (for illustration; real pipelines need implicit-function theorem on KKT)
    n = size(Σ, 1)
    invΣ = inv(Σ)
    ones_n = ones(n)
    a = invΣ * ones_n
    b = ones_n' * a
    w = a / b
    # ∂w/∂Σ is a 3-tensor; here we return the Jacobian wrt vech(Σ) numerically
    return w
end

function finite_diff_sensitivity(optimise_fn, μ, Σ; ε = 1e-6)
    n = length(μ)
    w0 = optimise_fn(μ, Σ)
    Jμ = zeros(n, n)
    for i in 1:n
        μ2 = copy(μ)
        μ2[i] += ε
        Jμ[:, i] = (optimise_fn(μ2, Σ) - w0) / ε
    end
    return w0, Jμ
end
```

---

## Priority 6 – Usability & Ecosystem

### 13. Mixed-Frequency / Alternative-Data Support

**Idea**
Native handling of mixed-frequency returns and irregular observation times.

**Standalone sketch**

```julia
using Dates

struct MixedFrequencyReturns
    daily::NamedTuple          # (dates, matrix)
    weekly::NamedTuple
    monthly::NamedTuple
    alt::Dict{Symbol, Any}     # alternative data aligned by date
end

function align_to_daily(mf::MixedFrequencyReturns)
    # forward-fill lower-frequency series onto daily calendar
    # (simplified)
    D = mf.daily.dates
    weekly_aligned = forward_fill(mf.weekly.dates, mf.weekly.matrix, D)
    monthly_aligned = forward_fill(mf.monthly.dates, mf.monthly.matrix, D)
    return hcat(mf.daily.matrix, weekly_aligned, monthly_aligned)
end

function forward_fill(src_dates, src_mat, target_dates)
    # standard LOCF alignment
    ...
end
```

---

## Suggested Near-Term Ordering

1. Dedicated rebalancing policy layer + richer transaction-cost models
2. Scenario generation / stress-testing toolkit (vine-copula + conditional sampling)
3. Multi-period / rolling-horizon formulations
4. Uncertainty quantification on weights (bootstrap / conformal)
5. Differentiable layers (DiffOpt integration or native)
6. Graph-constraint primitives (Laplacian regularisation)
7. Mixed-frequency support

---

## Notes for Maintainers

- All new estimators should follow the existing immutable, compositional, early-validation design.
- Prefer adding new `AbstractEstimator` / `AbstractAlgorithm` subtypes over special-casing.
- Documentation and examples should appear together with the feature (the capability catalogue generation already helps keep this honest).
- Breaking changes remain acceptable while the package is in v0.*.
- The code samples above are **standalone illustrations** only; they are deliberately not written against the current internal API.

---

*Generated from discussion of the `dev` branch of PortfolioOptimisers.jl. Feedback and prioritisation from the maintainer will refine this list.*
