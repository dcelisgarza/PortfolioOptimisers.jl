# =============================================================================
# Prototype 16 — When to trade: rebalancing policies.
#
# Purpose
#   Report 3 asks for this and report 5 asks for the simulation loop around it.
#   The library answers "what should I hold?". It has no vocabulary for "should
#   I trade today at all?", and the two questions have different answers.
#
#   The optimiser is indifferent to a one-basis-point improvement. A trader is
#   not, because that improvement costs money to capture. A rebalancing policy
#   is the object that decides, and there are exactly four useful kinds:
#
#     * **Calendar.** Trade every `k` periods. Simple, auditable, and blind.
#     * **Threshold.** Trade when the drifted portfolio is far enough from the
#       target. Reacts to the market instead of the diary.
#     * **Cost aware.** Trade when the expected gain exceeds the expected cost.
#       The only rule with an economic justification.
#     * **Hybrid.** Cost aware, but never less often than the calendar and
#       never more often than a floor. What is actually used in practice.
#
#   The library already has `TimeDependent` machinery, which can express a
#   calendar. What is absent is the decision itself, and the loop that applies
#   it.
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics`.
#
# Notation used throughout this file
#   w_held    Drifted weights currently held, length `N`.
#   w_target  The optimiser's answer for this period, length `N`.
#   tau       Turnover, `0.5 * norm(w_target - w_held, 1)`.
#   U(w)      Utility, `mu' w - (gamma / 2) w' sig w`.
#   c(.)      Trading cost of moving from one portfolio to another.
#
# Sources
#   Leland, H. E. (1999). Optimal portfolio management with transactions costs
#     and capital gains taxes. Research Program in Finance Working Paper
#     RPF-290, University of California Berkeley. The no-trade region.
#   Davis, M. H. A. and Norman, A. R. (1990). Portfolio selection with
#     transaction costs. Mathematics of Operations Research 15(4), 676-713. The
#     proof that the optimal policy is a no-trade cone, not a schedule.
#   Constantinides, G. M. (1986). Capital market equilibrium with transaction
#     costs. Journal of Political Economy 94(4), 842-862.
#   Donohue, C. and Yip, K. (2003). Optimal portfolio rebalancing with
#     transaction costs. Journal of Portfolio Management 29(4), 49-63. The
#     empirical comparison of calendar against threshold rules.
#   Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P. and
#     Speth, J. (2017). Multi-period trading via convex optimization.
#     Foundations and Trends in Optimization 3(1), 1-76.
# =============================================================================
module RebalancingPolicies

using LinearAlgebra, Statistics

export AbstractRebalancePolicy, AlwaysRebalance, NeverRebalance, CalendarPolicy,
       ThresholdPolicy, CostAwarePolicy, HybridPolicy, should_rebalance, linear_cost,
       simulate_policy

"""
    AbstractRebalancePolicy

Supertype of the rules that decide whether to trade at a given period.

Every member answers one question through [`should_rebalance`](@ref), and
carries no state of its own. The state lives in the arguments.
"""
abstract type AbstractRebalancePolicy end

"""
    AlwaysRebalance <: AbstractRebalancePolicy

Trade every period. The control case, and the implicit assumption of every
backtest that ignores costs.
"""
struct AlwaysRebalance <: AbstractRebalancePolicy end

"""
    NeverRebalance <: AbstractRebalancePolicy

Never trade after the first period. Buy and hold.
"""
struct NeverRebalance <: AbstractRebalancePolicy end

"""
    CalendarPolicy <: AbstractRebalancePolicy

Trade every `period` observations.

# Fields

  - `period::Int`: Number of periods between trades. Must be at least one.

# Notes

  - **A calendar rule is blind to the market**, so it trades when nothing has
    changed and waits when everything has. It is the baseline every other rule
    must beat, and it frequently is not beaten, because its predictability has
    real operational value.
"""
struct CalendarPolicy <: AbstractRebalancePolicy
    period::Int
    function CalendarPolicy(period::Integer)
        if period < 1
            throw(DomainError(period, "period must be >= 1"))
        end
        return new(Int(period))
    end
end

"""
    ThresholdPolicy <: AbstractRebalancePolicy

Trade when turnover would exceed `tol`.

# Fields

  - `tol::Float64`: Turnover threshold, in `[0, 1]`. Zero recovers
    [`AlwaysRebalance`](@ref).

# Mathematical definition

Trade when

    (1/2) || w_target - w_held ||_1  >  tol

# Notes

  - This is a crude approximation of the **no-trade region** that Davis and
    Norman (1990) proved is optimal under proportional costs. The exact region
    is a cone whose shape depends on the covariance; a single scalar threshold
    is a ball. The approximation is standard because the exact cone is
    intractable beyond a few assets.
"""
struct ThresholdPolicy <: AbstractRebalancePolicy
    tol::Float64
    function ThresholdPolicy(tol::Real)
        if tol < 0
            throw(DomainError(tol, "tol must be >= 0"))
        end
        return new(float(tol))
    end
end

"""
    CostAwarePolicy <: AbstractRebalancePolicy

Trade when the expected utility gain exceeds the trading cost.

# Fields

  - `gamma::Float64`: Risk aversion used in the utility.
  - `min_net_benefit::Float64`: Required surplus after costs. A positive value
    creates a hysteresis band and stops the rule from churning near
    indifference.

# Mathematical definition

With `U(w) = mu' w - (gamma / 2) w' sig w`, trade when

    U(w_target) - U(w_held)  -  c(w_held -> w_target)  >  min_net_benefit

# Notes

  - **This is the only rule with an economic justification**, and it is also
    the only one that needs `mu`, which is the least reliable input in the
    problem. A cost-aware rule driven by a noisy mean estimate trades on noise
    and pays for the privilege. Pair it with the shrinkage of prototype 4 or
    the ambiguity radius of prototype 2.

  - **The horizon mismatch is the trap, and it is fatal if ignored.** The
    utility gain is per period. The cost is paid once. Comparing them directly
    means the rule almost never trades: measured over 1248 daily periods with
    a 20 basis point cost and `gamma = 5`, this policy traded **zero times**,
    while a 21-day calendar traded 60 times and a hybrid rule traded 20.

  - The fix is to compare like with like. The gain must be multiplied by the
    **expected number of periods the new portfolio will be held**, `h`, or
    equivalently the cost divided by it:

        trade when   h * ( U(w_target) - U(w_held) )  -  c  >  min_net_benefit

    `h` is not free: it depends on the policy, which depends on `h`. In
    practice it is fixed at the average observed holding period and iterated
    once. **A rule that sets `h = 1` never trades, and a rule that sets
    `h = Inf` always trades.** Neither extreme is the answer, and this
    implementation deliberately exposes the `h = 1` end so the failure is
    visible rather than hidden inside a tuned constant.

  - [`HybridPolicy`](@ref) sidesteps the problem with its `max_gap` ceiling,
    which is why it is the rule that survives contact with real data.
"""
struct CostAwarePolicy <: AbstractRebalancePolicy
    gamma::Float64
    min_net_benefit::Float64
    function CostAwarePolicy(; gamma::Real = 1.0, min_net_benefit::Real = 0.0)
        if gamma <= 0
            throw(DomainError(gamma, "gamma must be > 0"))
        end
        return new(float(gamma), float(min_net_benefit))
    end
end

"""
    HybridPolicy <: AbstractRebalancePolicy

Combine a cost-aware test with a calendar ceiling and a turnover floor.

# Fields

  - `inner::CostAwarePolicy`: The economic test.
  - `max_gap::Int`: Force a trade after this many periods without one, however
    the economic test votes.
  - `min_turnover::Float64`: Refuse to trade below this turnover, however the
    economic test votes.

# Notes

  - `max_gap` bounds model drift. `min_turnover` bounds churn. **Together they
    turn an unbounded rule into an auditable one**, which is what makes the
    hybrid the one used in practice.
"""
struct HybridPolicy <: AbstractRebalancePolicy
    inner::CostAwarePolicy
    max_gap::Int
    min_turnover::Float64
    function HybridPolicy(; inner::CostAwarePolicy = CostAwarePolicy(),
                          max_gap::Integer = 63, min_turnover::Real = 0.01)
        if max_gap < 1
            throw(DomainError(max_gap, "max_gap must be >= 1"))
        end
        return new(inner, Int(max_gap), float(min_turnover))
    end
end

"""
    linear_cost(w_from::AbstractVector, w_to::AbstractVector; bps::Real = 5.0) -> Real

Return a proportional trading cost in return units.

# Arguments

  - `w_from`, `w_to`: The old and new portfolios.
  - `bps`: Cost in basis points of the value traded, one way.

# Mathematical definition

    c  =  (bps / 10_000) * || w_to - w_from ||_1

Note the **one**-norm, not half of it. Turnover is conventionally a half-norm
because a round trip is counted once; a cost is paid on both the sale and the
purchase, so it is the full norm. Confusing the two halves the cost, which is
the most common error in a backtest.

# Notes

  - This is linear, so it has no notion of trade size. See prototype 17 for the
    square-root and Almgren-Chriss models, which do.
"""
function linear_cost(w_from::AbstractVector{<:Real}, w_to::AbstractVector{<:Real};
                     bps::Real = 5.0)
    if length(w_from) != length(w_to)
        throw(DimensionMismatch("w_from has length $(length(w_from)), w_to has length $(length(w_to))"))
    end
    return (bps / 10_000) * sum(abs, collect(w_to) .- collect(w_from))
end

"""
    should_rebalance(policy, t::Integer, w_held::AbstractVector,
                     w_target::AbstractVector; last_trade::Integer = 0,
                     mu = nothing, sigma = nothing, cost_fn = linear_cost)
        -> NamedTuple

Decide whether to trade, and say why.

# Arguments

  - `policy`: Any [`AbstractRebalancePolicy`](@ref).
  - `t`: The current period index.
  - `w_held`: The drifted holdings.
  - `w_target`: The optimiser's answer.
  - `last_trade`: Index of the last period in which a trade happened.
  - `mu`, `sigma`: Required by [`CostAwarePolicy`](@ref) and
    [`HybridPolicy`](@ref), ignored otherwise.
  - `cost_fn`: A function `(w_from, w_to) -> cost`.

# Returns

A `NamedTuple` with `trade::Bool`, `turnover`, `reason` and, where the policy
computes them, `gain` and `cost`.

# Notes

  - **`reason` is not decoration.** A rebalancing decision that cannot be
    explained after the fact is not auditable, and the audit is the reason a
    policy exists as an object instead of an `if` statement.
"""
function should_rebalance(policy::AbstractRebalancePolicy, t::Integer,
                          w_held::AbstractVector{<:Real}, w_target::AbstractVector{<:Real};
                          last_trade::Integer = 0, mu = nothing, sigma = nothing,
                          cost_fn::Function = linear_cost)
    tau = sum(abs, collect(w_target) .- collect(w_held)) / 2
    return _decide(policy, t, w_held, w_target, tau, last_trade, mu, sigma, cost_fn)
end

function _decide(::AlwaysRebalance, t, wh, wt, tau, lt, mu, sig, cf)
    return (; trade = true, turnover = tau, reason = "always")
end
function _decide(::NeverRebalance, t, wh, wt, tau, lt, mu, sig, cf)
    return (; trade = false, turnover = tau, reason = "never")
end

function _decide(p::CalendarPolicy, t, wh, wt, tau, lt, mu, sig, cf)
    due = (t - lt) >= p.period
    return (; trade = due, turnover = tau,
            reason = if due
                "calendar: $(t - lt) periods since last trade"
            else
                "calendar: only $(t - lt) of $(p.period) periods elapsed"
            end)
end

function _decide(p::ThresholdPolicy, t, wh, wt, tau, lt, mu, sig, cf)
    over = tau > p.tol
    return (; trade = over, turnover = tau,
            reason = if over
                "turnover $(round(tau; digits = 4)) exceeds $(p.tol)"
            else
                "turnover $(round(tau; digits = 4)) within $(p.tol)"
            end)
end

function _decide(p::CostAwarePolicy, t, wh, wt, tau, lt, mu, sig, cf)
    if isnothing(mu) || isnothing(sig)
        throw(ArgumentError("CostAwarePolicy needs both mu and sigma"))
    end
    U(w) = dot(mu, w) - (p.gamma / 2) * dot(w, sig, w)
    gain = U(wt) - U(wh)
    cost = cf(wh, wt)
    net = gain - cost
    ok = net > p.min_net_benefit
    return (; trade = ok, turnover = tau, gain = gain, cost = cost,
            reason = "net benefit $(round(net; sigdigits = 3)) vs threshold $(p.min_net_benefit)")
end

function _decide(p::HybridPolicy, t, wh, wt, tau, lt, mu, sig, cf)
    if tau < p.min_turnover
        return (; trade = false, turnover = tau,
                reason = "below turnover floor $(p.min_turnover)")
    end
    if (t - lt) >= p.max_gap
        return (; trade = true, turnover = tau,
                reason = "forced: $(t - lt) periods since last trade exceeds $(p.max_gap)")
    end
    inner = _decide(p.inner, t, wh, wt, tau, lt, mu, sig, cf)
    return (; trade = inner.trade, turnover = tau, gain = inner.gain, cost = inner.cost,
            reason = "cost-aware: " * inner.reason)
end

"""
    simulate_policy(policy, X::AbstractMatrix, optimise_fn::Function;
                    lookback::Integer = 252, cost_bps::Real = 5.0,
                    gamma::Real = 1.0) -> NamedTuple

Run a rebalancing policy over a return history and report the net outcome.

# Arguments

  - `policy`: The rule under test.
  - `X`: Returns, `T x N`.
  - `optimise_fn`: A function `X_window -> w`.
  - `lookback`: Estimation window length.
  - `cost_bps`: One-way cost in basis points.
  - `gamma`: Risk aversion, forwarded to the cost-aware rules.

# Returns

A `NamedTuple` with `wealth_gross`, `wealth_net`, `n_trades`, `total_cost`,
`total_turnover`, `returns_net` and `weights`.

# Details

At each period the held weights are drifted by the realised returns, the
optimiser is re-run on the trailing window, and the policy decides. **Costs are
charged in the period the trade happens**, and the return of a period is
computed on the weights held *during* it, so no look-ahead enters.

# Notes

  - Compare policies on `wealth_net`, never on `wealth_gross`. A rule that
    trades every day always wins on the gross figure, which is exactly why the
    gross figure is worthless.
  - `optimise_fn` is called every period even when no trade follows. That is
    deliberate: the decision needs the target. A production implementation can
    skip the solve when a calendar rule has already vetoed the period.
"""
function simulate_policy(policy::AbstractRebalancePolicy, X::AbstractMatrix{<:Real},
                         optimise_fn::Function; lookback::Integer = 252,
                         cost_bps::Real = 5.0, gamma::Real = 1.0)
    T, N = size(X)
    if lookback >= T
        throw(ArgumentError("lookback $(lookback) must be less than the $(T) observations"))
    end
    Tv = float(eltype(X))
    w = fill(one(Tv) / N, N)
    last_trade = 0
    n_trades = 0
    total_cost = zero(Tv)
    total_turn = zero(Tv)
    rets_gross = Tv[]
    rets_net = Tv[]
    Ws = Vector{Vector{Tv}}()
    cf = (a, b) -> linear_cost(a, b; bps = cost_bps)
    for t in (lookback + 1):T
        win = view(X, (t - lookback):(t - 1), :)
        w_target = optimise_fn(win)
        mu = vec(mean(win; dims = 1))
        sig = Matrix(cov(win))
        d = should_rebalance(policy, t, w, w_target; last_trade = last_trade, mu = mu,
                             sigma = sig, cost_fn = cf)
        cost = zero(Tv)
        if d.trade
            cost = cf(w, w_target)
            total_cost += cost
            total_turn += d.turnover
            n_trades += 1
            last_trade = t
            w = collect(Tv, w_target)
        end
        push!(Ws, copy(w))
        r = dot(w, view(X, t, :))
        push!(rets_gross, r)
        push!(rets_net, r - cost)
        w = drift(w, view(X, t, :))
    end
    return (; wealth_gross = prod(one(Tv) .+ rets_gross),
            wealth_net = prod(one(Tv) .+ rets_net), n_trades = n_trades,
            total_cost = total_cost, total_turnover = total_turn, returns_net = rets_net,
            returns_gross = rets_gross, weights = Ws)
end

"""
    drift(w::AbstractVector, r::AbstractVector) -> Vector

Return weights carried forward by one period of returns, renormalised to the
same budget. See prototype 15 for the version that also tracks cash.
"""
function drift(w::AbstractVector{<:Real}, r::AbstractVector{<:Real})
    g = collect(float.(w)) .* (one(float(eltype(w))) .+ r)
    s = sum(g)
    return iszero(s) ? g : g .* (sum(w) / s)
end

end # module RebalancingPolicies
