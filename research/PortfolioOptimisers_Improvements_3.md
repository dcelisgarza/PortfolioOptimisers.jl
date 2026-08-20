# PortfolioOptimisers.jl – Feature Roadmap

**Status:** Living document derived from discussion of the `dev` branch (2026-08).
**Purpose:** Prioritised suggestions for future development that do **not** duplicate existing capabilities.

Sample code below is **standalone** (JuMP / plain Julia). It is illustrative only and is **not** integrated into the package API.

---

## Already Implemented (for reference)

| Feature | Notes |
| --------- | ------- |
| OWA / L-moments | Present |
| Relaxed risk parity | Expanded to integer formulations + factor risk parity |
| Worst-case formulations | Uncertainty sets (box, ellipse, …) on μ and Σ |
| Distributionally Robust CVaR / CVaR Range / CDaR | Present (do **not** re-implement these) |
| Cardinality, threshold, conditional inclusion (MIP) + set counterparts | Present |
| HRP | Present |
| HERC | Generalised to different inner/outer risk measures |
| Schur complementary allocation | Present |
| Index tracking + weight tracking objectives/constraints | Present |
| NCO | Generalised to arbitrary inner/outer estimators + CV prediction for outer returns |
| Stacking optimiser | Present |
| Walk-forward + combinatorial purged CV | Present |
| TimeDependent | Can implement rebalancing policies (dedicated higher-level machinery still useful) |

---

## Priority 1 – High Leverage / Practical Impact

### 1. Dedicated Rebalancing Policy Layer

**Idea**
Elevate `TimeDependent` into (or wrap it with) first-class **rebalancing policies**:

- Calendar-based (daily / weekly / monthly / quarter-end)
- Threshold-based (turnover, drift from target, risk-budget violation)
- Cost-aware (trade only when expected benefit > estimated transaction cost)
- Hybrid (calendar + threshold + cost)

**Standalone sample**

```julia
using Dates, LinearAlgebra

abstract type RebalancePolicy end

struct CalendarRebalance <: RebalancePolicy
    period::Period          # e.g. Month(1), Week(1)
end

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

- Piecewise-linear / quadratic / market-impact (power-law) transaction costs
- Bid–ask spread models
- Simple tax-lot / short-term vs long-term capital-gains approximations

**Standalone sample (JuMP market-impact + linear TC)**

```julia
using JuMP, Clarabel

function portfolio_with_market_impact(μ, Σ, w0;
                                      γ = 1.0,           # risk aversion
                                      a = 1e-3,          # linear TC coeff
                                      b = 1e-4,          # market-impact coeff
                                      beta = 1.5)        # impact exponent
    n = length(μ)
    model = Model(Clarabel.Optimizer)
    set_silent(model)

    @variable(model, w[1:n] >= 0)
    @variable(model, trade[1:n])          # w - w0
    @variable(model, abs_trade[1:n] >= 0)
    @variable(model, impact[1:n] >= 0)

    @constraint(model, sum(w) == 1)
    @constraint(model, trade .== w - w0)
    @constraint(model, abs_trade .>= trade)
    @constraint(model, abs_trade .>= -trade)

    # power-law impact approximated by SOCP for beta=1.5
    # impact_i >= b * |trade_i|^1.5  (convex)
    for i in 1:n
        # t^{1.5} = t * sqrt(t) → auxiliary
        @variable(model, s >= 0)
        @constraint(model, [s; abs_trade[i]] in SecondOrderCone()) # s >= sqrt(|trade|)
        @constraint(model, impact[i] >= b * abs_trade[i] * s)
    end

    @objective(model, Max,
        μ ⋅ w - γ * w' * Σ * w
        - a * sum(abs_trade) - sum(impact)
    )
    optimize!(model)
    return value.(w), value.(abs_trade), value.(impact)
end
```

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

### 4. Wasserstein-Style Uncertainty / Ambiguity Sets

**Idea**
Wasserstein (and Gelbrich) ambiguity sets usable with mean-risk objectives — **excluding** the already-implemented DR-CVaR / CVaR-range / CDaR.

**Standalone sample (Wasserstein mean-variance style regularisation)**

A common tractable reformulation of worst-case expectation under a type-1 Wasserstein ball of radius ε around the empirical measure yields a regularised empirical problem. For mean-variance this often appears as an extra term proportional to ‖w‖_* (dual norm).

```julia
using JuMP, Clarabel, LinearAlgebra, Statistics

"""
Wasserstein-robust mean-variance (illustrative dual form).
ε = Wasserstein radius, p = 1 or 2.
"""
function wasserstein_mvo(returns::AbstractMatrix; ε = 0.01, γ = 1.0, p = 1)
    T, n = size(returns)
    μ = vec(mean(returns; dims = 1))
    Σ = cov(returns)

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:n] >= 0)
    @constraint(model, sum(w) == 1)

    # dual-norm regulariser: for p=1 → ∞-norm, for p=2 → 2-norm
    if p == 1
        @variable(model, t >= 0)
        @constraint(model, -t .<= w .<= t)          # t ≥ ‖w‖_∞
        reg = t
    else
        @variable(model, t >= 0)
        @constraint(model, [t; w] in SecondOrderCone()) # t ≥ ‖w‖₂
        reg = t
    end

    # worst-case mean ≈ μ'w - ε * ‖w‖_*
    @objective(model, Max, μ ⋅ w - ε * reg - γ * w' * Σ * w)
    optimize!(model)
    return value.(w)
end

# Gelbrich-style outer approximation (moment ball)
function gelbrich_mvo(μ, Σ; ρ = 0.05, γ = 1.0)
    n = length(μ)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    @variable(model, w[1:n] >= 0)
    @constraint(model, sum(w) == 1)

    # Gelbrich upper bound on worst-case variance involves
    # (√(w'Σw) + ρ‖w‖)²  (simplified)
    @variable(model, s >= 0)
    @constraint(model, [s; sqrt(Σ) * w] in SecondOrderCone()) # s ≥ √(w'Σw)
    @variable(model, u >= 0)
    @constraint(model, [u; w] in SecondOrderCone())
    @objective(model, Max, μ ⋅ w - γ * (s + ρ * u)^2)
    optimize!(model)
    return value.(w)
end
```

---

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

### 8. Online / Streaming Estimators & Incremental Updates

**Idea**
O(1) / O(n²) updates for mean and covariance when a new return vector arrives.

**Standalone sample**

```julia
mutable struct OnlineMeanCov
    n::Int
    t::Int
    μ::Vector{Float64}
    Σ::Matrix{Float64}          # running covariance (population or sample)
    α::Float64                  # EWMA decay (1 = equal weight)
end

function OnlineMeanCov(n; α = 0.99)
    OnlineMeanCov(n, 0, zeros(n), zeros(n, n), α)
end

function update!(o::OnlineMeanCov, x::AbstractVector)
    o.t += 1
    if o.t == 1
        o.μ .= x
        return o
    end
    # EWMA
    δ = x - o.μ
    o.μ .+= (1 - o.α) * δ
    # rank-1 covariance update (EWMA)
    o.Σ .= o.α * o.Σ + (1 - o.α) * (δ * δ')
    return o
end

# incremental minimum-variance weights (analytic)
function minvar_weights(Σ)
    n = size(Σ, 1)
    ones_n = ones(n)
    invΣ = inv(Σ)
    w = invΣ * ones_n
    w ./= sum(w)
    return w
end
```

---

## Priority 4 – Graph & Network Extensions

### 9. Richer Graph / Network Constraints & Objectives

**Idea**
Laplacian regularisation, centrality / neighbourhood constraints on top of existing MST / clustering graphs.

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

"""
Neighbourhood constraint: if (i,j) is an edge, at most one of them may be held
(or their total weight ≤ cap).  Requires MIP.
"""
function mvo_neighbourhood(μ, Σ, edges; cap = 0.05)
    n = length(μ)
    model = Model(HiGHS.Optimizer)          # MIP solver
    set_silent(model)
    @variable(model, w[1:n] >= 0)
    @variable(model, z[1:n], Bin)
    @constraint(model, sum(w) == 1)
    @constraint(model, w .<= z)             # z_i = 1 if asset i selected
    for (i, j) in edges
        @constraint(model, z[i] + z[j] <= 1)
    end
    @objective(model, Max, μ ⋅ w - w' * Σ * w)
    optimize!(model)
    return value.(w)
end

# build Laplacian from adjacency
laplacian(A) = Diagonal(vec(sum(A; dims = 2))) - A
```

---

## Priority 5 – Higher-Order & Non-Convex / Advanced Risk

### 10. Additional Higher-Order / Non-Convex Risk Measures with Good Relaxations

**Idea**
Full MVSK with SDP relaxations, etc.

**Standalone sketch (mean-variance-skewness via moment relaxation)**

```julia
# Highly simplified illustration – real MVSK needs the full coskewness tensor
# and specialised SDP / polynomial optimisation tools (e.g. SumOfSquares.jl)

using JuMP, SCS, LinearAlgebra

function rough_mvsk_sdp(μ, Σ, coskew_slice; γ2 = 1.0, γ3 = 0.1)
    # coskew_slice is a matrix approximation / marginal slice of the tensor
    n = length(μ)
    model = Model(SCS.Optimizer)
    set_silent(model)
    @variable(model, w[1:n] >= 0)
    @constraint(model, sum(w) == 1)

    # variance
    @variable(model, v >= 0)
    @constraint(model, [v; √Σ * w] in SecondOrderCone())

    # crude skewness proxy
    skew_proxy = w' * coskew_slice * w   # not the real cubic form

    @objective(model, Max, μ ⋅ w - γ2 * v + γ3 * skew_proxy)
    optimize!(model)
    return value.(w)
end
```

---

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

### 12. Batteries-Included End-to-End Pipelines

**Idea**
High-level convenience pipelines that still expose the full type system.

**Standalone sketch**

```julia
struct PortfolioPipeline
    preprocess
    select
    optimise
    allocate
    rebalance
    validate
end

function fit_predict(pipe::PortfolioPipeline, prices; dates = nothing)
    rd = pipe.preprocess(prices)
    assets = pipe.select(rd)
    rd_sel = rd[:, assets]          # simplified
    w = pipe.optimise(rd_sel)
    shares = pipe.allocate(w, prices[end, assets])
    # optional walk-forward
    if pipe.validate !== nothing
        score = pipe.validate(pipe.optimise, rd_sel)
    else
        score = nothing
    end
    return (; w, shares, score, assets)
end
```

---

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
2. Wasserstein-style (and Gelbrich) uncertainty / ambiguity sets (tractable reformulations for mean-risk objectives)
3. Scenario generation / stress-testing toolkit (vine-copula + conditional sampling)
4. Multi-period / rolling-horizon formulations
5. Uncertainty quantification on weights (bootstrap / conformal)
6. Differentiable layers (DiffOpt integration or native)
7. Graph-constraint primitives
8. Online/streaming estimators
9. Usability pipelines and mixed-frequency support

---

## Notes for Maintainers

- All new estimators should follow the existing immutable, compositional, early-validation design.
- Prefer adding new `AbstractEstimator` / `AbstractAlgorithm` subtypes over special-casing.
- Documentation and examples should appear together with the feature (the capability catalogue generation already helps keep this honest).
- Breaking changes remain acceptable while the package is in v0.*.
- The code samples above are **standalone illustrations** only; they are deliberately not written against the current internal API.

---

*Generated from discussion of the `dev` branch of PortfolioOptimisers.jl. Feedback and prioritisation from the maintainer will refine this list.*
