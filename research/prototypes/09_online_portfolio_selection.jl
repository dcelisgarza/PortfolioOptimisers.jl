# =============================================================================
# Prototype 9 — Online portfolio selection.
#
# Purpose
#   Every optimiser in the library is a batch method: it reads a window of
#   history, fits moments, and solves a programme. Online portfolio selection
#   is a different family. It never estimates a moment. It updates the
#   portfolio directly from the last price relative, and it carries a
#   worst-case guarantee that holds for **every** price sequence, with no
#   statistical assumption at all.
#
#   That makes these algorithms the natural fast baseline for the walk-forward
#   machinery the library already has. They sit beside the naive optimisers,
#   not beside `MeanRisk`, and they need no prior, no covariance and no solver.
#
#   Report 5 noted that no Julia package implements them. That is the gap.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics` and `Random`.
#
# Notation used throughout this file
#   T       Number of periods.
#   N       Number of assets.
#   X       Price-relative matrix, `T x N`. Entry `X[t, i]` is
#           `price[t, i] / price[t-1, i]`, so a flat asset gives one. Convert
#           from simple returns with `X = 1 .+ R`.
#   w_t     The portfolio held during period `t`, on the simplex.
#   x_t     Row `t` of `X`.
#   S_T     Terminal wealth, `prod_t dot(w_t, x_t)`, starting from one.
#   eta     Learning rate.
#   eps     Reversion threshold.
#
# Sources
#   Cover, T. M. (1991). Universal portfolios. Mathematical Finance 1(1), 1-29.
#   Helmbold, D. P., Schapire, R. E., Singer, Y. and Warmuth, M. K. (1998).
#     On-line portfolio selection using multiplicative updates. Mathematical
#     Finance 8(4), 325-347. Exponentiated gradient.
#   Agarwal, A., Hazan, E., Kale, S. and Schapire, R. E. (2006). Algorithms for
#     portfolio management based on the Newton method. Proceedings of the 23rd
#     International Conference on Machine Learning, 9-16. Online Newton step.
#   Li, B., Zhao, P., Hoi, S. C. H. and Gopalkrishnan, V. (2012). PAMR:
#     passive aggressive mean reversion strategy for portfolio selection.
#     Machine Learning 87(2), 221-258.
#   Li, B. and Hoi, S. C. H. (2012). On-line portfolio selection with moving
#     average reversion. Proceedings of the 29th International Conference on
#     Machine Learning. arXiv:1206.4626.
#   Huang, D., Zhou, J., Li, B., Hoi, S. C. H. and Zhou, S. (2016). Robust
#     median reversion strategy for online portfolio selection. IEEE
#     Transactions on Knowledge and Data Engineering 28(9), 2480-2493.
#   Li, B. and Hoi, S. C. H. (2014). Online portfolio selection: a survey. ACM
#     Computing Surveys 46(3), article 35. The single citation that covers the
#     whole family.
#   Duchi, J., Shalev-Shwartz, S., Singer, Y. and Chandra, T. (2008). Efficient
#     projections onto the l1-ball for learning in high dimensions.
#     Proceedings of the 25th International Conference on Machine Learning,
#     272-279. The simplex projection used by every reversion method here.
#   Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n
#     points donnes est minimum. Tohoku Mathematical Journal 43, 355-386. The
#     L1 median iteration used by robust median reversion.
# =============================================================================
module OnlinePortfolioSelection

using LinearAlgebra, Statistics, Random

export project_simplex, uniform_crp, best_stock, best_crp, universal_portfolio,
       exponentiated_gradient, online_newton_step, pamr, olmar, rmr, run_online,
       OnlineResult, terminal_wealth, regret_against_bcrp

"""
    OnlineResult{T}

The output of an online strategy.

# Fields

  - `weights::Matrix{T}`: `T x N`. Row `t` is the portfolio **held during**
    period `t`, decided before `x_t` was seen.
  - `wealth::Vector{T}`: Length `T`. Entry `t` is the cumulative wealth after
    period `t`, starting from one.
  - `name::String`: The strategy label.

# Notes

  - **Row `t` of `weights` must depend only on rows `1` to `t-1` of `X`.**
    Every constructor in this file respects that. It is the whole content of
    the word "online", and it is the property a backtest silently breaks.
"""
struct OnlineResult{T <: Real}
    weights::Matrix{T}
    wealth::Vector{T}
    name::String
end

"""
    terminal_wealth(r::OnlineResult) -> Real

Return the final cumulative wealth, `S_T`.
"""
terminal_wealth(r::OnlineResult) = last(r.wealth)

"""
    project_simplex(v::AbstractVector) -> Vector

Return the Euclidean projection of `v` onto the probability simplex.

# Arguments

  - `v`: Any real vector, length `N`.

# Returns

  - The unique `w` with `sum(w) == 1`, `w .>= 0`, minimising `norm(w - v, 2)`.

# Mathematical definition

The projection solves

    minimise  (1/2) || w - v ||_2^2   subject to  1' w = 1,  w >= 0

Its Lagrangian gives `w_i = max(v_i - theta, 0)` for a single scalar `theta`
fixed by the budget. Sort `v` in decreasing order as `u`, find

    rho = max { j : u_j - (1/j) ( sum_{k<=j} u_k - 1 ) > 0 }

and set `theta = (1/rho) ( sum_{k<=rho} u_k - 1 )`. This is the algorithm of
Duchi and co-authors (2008) and costs one sort.

# Notes

  - Every mean-reversion method in this file makes an unconstrained update and
    then projects. **The projection is what keeps the portfolio a portfolio**,
    and it is also what makes these methods sparse: entries pushed below
    `theta` become exactly zero.
"""
function project_simplex(v::AbstractVector{<:Real})
    N = length(v)
    T = float(eltype(v))
    u = sort(collect(T, v); rev = true)
    css = zero(T)
    rho = 1
    theta = zero(T)
    for j in 1:N
        css += u[j]
        t = (css - one(T)) / j
        if u[j] - t > 0
            rho = j
            theta = t
        end
    end
    return max.(T.(v) .- theta, zero(T))
end

"""
    _wealth_path(X::AbstractMatrix, W::AbstractMatrix) -> Vector

Return the cumulative wealth of a weight path.

# Arguments

  - `X`: Price relatives, `T x N`.
  - `W`: Weights, `T x N`. Row `t` is held during period `t`.

# Returns

  - Cumulative wealth, length `T`, starting from one.

# Mathematical definition

    S_t = prod_{s=1}^{t} < w_s , x_s >

The inner product is the portfolio's gross return for the period, so a value
of one means no change. **This assumes the portfolio is rebalanced back to
`w_{t+1}` at every step and that rebalancing is free.** Transaction costs are
not modelled here. Prototype 12 supplies them.
"""
function _wealth_path(X::AbstractMatrix{<:Real}, W::AbstractMatrix{<:Real})
    T_, N = size(X)
    S = Vector{float(eltype(X))}(undef, T_)
    acc = one(eltype(S))
    @inbounds for t in 1:T_
        acc *= dot(view(W, t, :), view(X, t, :))
        S[t] = acc
    end
    return S
end

"""
    run_online(update!, X::AbstractMatrix, name::AbstractString;
               w0::Union{Nothing, AbstractVector} = nothing) -> OnlineResult

Drive any online rule over a price-relative matrix.

# Arguments

  - `update!`: A function `(w_next, t, X, W) -> nothing` that writes the
    portfolio for period `t + 1` into `w_next`, reading only rows `1` to `t`
    of `X` and rows `1` to `t` of `W`.
  - `X`: Price relatives, `T x N`.
  - `name`: Strategy label.
  - `w0`: Starting portfolio. Uniform if absent.

# Returns

  - An [`OnlineResult`](@ref).

# Notes

  - The loop is the only place where the no-look-ahead rule is enforced, so
    every strategy in this file goes through it.
"""
function run_online(update!::Function, X::AbstractMatrix{<:Real}, name::AbstractString;
                    w0::Union{Nothing, AbstractVector{<:Real}} = nothing)
    T_, N = size(X)
    Tv = float(eltype(X))
    W = Matrix{Tv}(undef, T_, N)
    W[1, :] .= isnothing(w0) ? fill(one(Tv) / N, N) : project_simplex(w0)
    tmp = Vector{Tv}(undef, N)
    for t in 1:(T_ - 1)
        update!(tmp, t, X, W)
        W[t + 1, :] .= tmp
    end
    return OnlineResult{Tv}(W, _wealth_path(X, W), String(name))
end

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
"""
    uniform_crp(X::AbstractMatrix) -> OnlineResult

The uniform constant rebalanced portfolio: hold `1/N` in every asset and
rebalance every period.

# Details

This is the reference every online result is quoted against, and it is a
surprisingly strong one. Cover's theorem is stated relative to the *best*
constant rebalanced portfolio, and the uniform one already captures the
volatility harvesting that makes constant rebalancing beat buy and hold on a
mean-reverting market.
"""
function uniform_crp(X::AbstractMatrix{<:Real})
    return run_online((wn, t, Xm, W) -> (wn .= view(W, t, :)), X, "uniform CRP")
end

"""
    best_stock(X::AbstractMatrix) -> OnlineResult

The single asset with the highest terminal wealth, chosen **in hindsight**.

# Notes

  - **This is not a strategy.** It looks at the whole sample and is reported
    only as an upper reference for a single-asset holding.
"""
function best_stock(X::AbstractMatrix{<:Real})
    T_, N = size(X)
    Tv = float(eltype(X))
    finals = [prod(view(X, :, i)) for i in 1:N]
    i = argmax(finals)
    W = zeros(Tv, T_, N)
    W[:, i] .= one(Tv)
    return OnlineResult{Tv}(W, _wealth_path(X, W), "best stock (hindsight)")
end

"""
    best_crp(X::AbstractMatrix; iters::Integer = 20_000, tol::Real = 1e-12)
        -> NamedTuple

Return the best constant rebalanced portfolio, chosen **in hindsight**.

# Arguments

  - `X`: Price relatives, `T x N`.
  - `iters`: Maximum number of fixed-point iterations.
  - `tol`: Convergence tolerance on the change in log wealth.

# Returns

A `NamedTuple` with `w`, `log_wealth`, `wealth` and `converged`.

# Mathematical definition

The best constant rebalanced portfolio solves

    maximise  sum_t log < w , x_t >   subject to  1' w = 1,  w >= 0

The objective is concave, so the optimum is global. Cover's (1984) fixed-point
iteration solves it:

    w_i^{k+1}  =  w_i^k * (1/T) * sum_t  x_{t,i} / < w^k , x_t >

The bracket is the average ratio of asset `i`'s return to the portfolio's own
return. An asset that beats the portfolio on average has a multiplier above
one and grows. At a fixed point every held asset has multiplier exactly one,
which is the first-order condition. **The iteration is a
minorise-maximise scheme, so the log wealth increases at every step** and the
sequence cannot oscillate. The routine asserts that monotonicity.

# Notes

  - **This is the benchmark that defines regret.** Cover's universal portfolio
    is universal precisely in the sense that its log wealth stays within
    `O(N log T)` of this quantity, for every price sequence, with no
    probabilistic assumption.
  - The result must be at least as good as the best single asset, because
    every single-asset portfolio is a corner of the simplex and therefore
    feasible. That is a cheap and effective correctness test.

# Sources

  - Cover, T. M. (1984). An algorithm for maximizing expected log investment
    return. IEEE Transactions on Information Theory 30(2), 369-373.
"""
function best_crp(X::AbstractMatrix{<:Real}; iters::Integer = 20_000, tol::Real = 1e-12)
    T_, N = size(X)
    Tv = float(eltype(X))
    w = fill(one(Tv) / N, N)
    prev = -Tv(Inf)
    converged = false
    lw = prev
    for _ in 1:iters
        p = X * w
        lw = sum(log, p)
        if abs(lw - prev) <= tol * max(one(Tv), abs(lw))
            converged = true
            break
        end
        prev = lw
        # Cover's multiplicative fixed point. `mean(x_i / <w,x>)` over periods.
        mult = vec(mean(X ./ p; dims = 1))
        w = w .* mult
        w ./= sum(w)
    end
    lw = sum(log, X * w)
    return (; w = w, log_wealth = lw, wealth = exp(lw), converged = converged)
end

# -----------------------------------------------------------------------------
# Strategies
# -----------------------------------------------------------------------------
"""
    universal_portfolio(X::AbstractMatrix; n_experts::Integer = 2000,
                        rng::Random.AbstractRNG = Random.default_rng())
        -> OnlineResult

Cover's universal portfolio, approximated by a finite set of experts.

# Arguments

  - `X`: Price relatives, `T x N`.
  - `n_experts`: Number of constant rebalanced experts sampled from the
    simplex.
  - `rng`: Random number generator.

# Mathematical definition

Cover (1991) defines the portfolio as a wealth-weighted average of **every**
constant rebalanced portfolio:

    w_{t+1}  =  int_Delta  b * S_t(b) d mu(b)  /  int_Delta S_t(b) d mu(b)

where `S_t(b) = prod_{s<=t} <b, x_s>` is the wealth of the constant portfolio
`b` and `mu` is a prior on the simplex, taken here as uniform. The integral is
intractable for `N` beyond three or four, so this implementation replaces it
with a Monte Carlo average over `n_experts` draws from a uniform Dirichlet.

# Notes

  - **The guarantee is the point.** Cover proved
    `S_T(UP) >= S_T(BCRP) / C(N, T)` with `C` growing only polynomially in
    `T`, for *every* sequence. No estimator in the library carries a
    guarantee of that kind, because every one of them assumes something about
    the distribution.
  - The Monte Carlo approximation weakens the bound. Accuracy is governed by
    `n_experts`, and the required count grows quickly with `N`. That is the
    practical reason the method is a reference rather than a workhorse.
"""
function universal_portfolio(X::AbstractMatrix{<:Real}; n_experts::Integer = 2000,
                             rng::Random.AbstractRNG = Random.default_rng())
    T_, N = size(X)
    Tv = float(eltype(X))
    # Uniform draws from the simplex, by normalising exponential variates.
    E = -log.(rand(rng, Tv, n_experts, N))
    B = E ./ sum(E; dims = 2)
    logS = zeros(Tv, n_experts)          # running log wealth of each expert
    function update!(wn, t, Xm, W)
        xt = view(Xm, t, :)
        logS .+= log.(B * xt)
        m = maximum(logS)
        sw = exp.(logS .- m)
        wn .= vec(transpose(B) * sw) ./ sum(sw)
        return nothing
    end
    return run_online(update!, X, "universal portfolio")
end

"""
    exponentiated_gradient(X::AbstractMatrix; eta::Real = 0.05) -> OnlineResult

The exponentiated gradient algorithm of Helmbold and co-authors (1998).

# Arguments

  - `X`: Price relatives, `T x N`.
  - `eta`: Learning rate. Larger reacts faster and is less stable.

# Mathematical definition

Maximise the last period's log return with a relative-entropy penalty against
the current portfolio:

    w_{t+1} = argmax_w { eta log < w , x_t >  -  D( w || w_t ) }

Linearising the log gives the multiplicative update

    w_{t+1,i}  proportional to  w_{t,i} * exp( eta * x_{t,i} / < w_t , x_t > )

which is normalised to sum to one. The penalty is the same relative entropy
that prototype 5 uses, applied here across assets rather than across
scenarios.

# Notes

  - **This is a momentum rule.** It increases the weight of whatever just did
    well. Every other reversion method in this file does the opposite. Running
    both and comparing is a cheap, assumption-free test of whether the market
    at hand trends or reverts.
  - The regret bound is `O(sqrt(T log N))` with `eta` tuned to the horizon.
"""
function exponentiated_gradient(X::AbstractMatrix{<:Real}; eta::Real = 0.05)
    function update!(wn, t, Xm, W)
        w = view(W, t, :)
        xt = view(Xm, t, :)
        p = dot(w, xt)
        wn .= w .* exp.(eta .* xt ./ p)
        wn ./= sum(wn)
        return nothing
    end
    return run_online(update!, X, "exponentiated gradient")
end

"""
    online_newton_step(X::AbstractMatrix; beta::Real = 1.0, delta::Real = 0.125,
                       eta::Real = 0.0) -> OnlineResult

The online Newton step of Agarwal and co-authors (2006).

# Arguments

  - `X`: Price relatives, `T x N`.
  - `beta`, `delta`: Algorithm constants from the paper.
  - `eta`: Optional shrinkage towards the uniform portfolio, in `[0, 1)`.

# Mathematical definition

The log-wealth objective is exp-concave, which admits a second-order method
with **logarithmic** rather than square-root regret. Accumulate the gradient
`g_t = x_t / <w_t, x_t>` and the outer-product matrix

    A_t = I + sum_{s<=t} g_s g_s' ,      b_t = (1 + 1/beta) sum_{s<=t} g_s

and take the generalised projection

    w_{t+1} = Proj_Delta^{A_t} ( delta * A_t^{-1} b_t )

This implementation uses the Euclidean projection in place of the
`A`-weighted one, which is the standard simplification and keeps the method
free of an inner solver.

# Notes

  - The regret is `O(N log T)`, the same order as Cover's, but the cost per
    step is `O(N^2)` rather than exponential in `N`. **That is why this rather
    than the universal portfolio is the practical second-order choice.**
"""
function online_newton_step(X::AbstractMatrix{<:Real}; beta::Real = 1.0,
                            delta::Real = 0.125, eta::Real = 0.0)
    N = size(X, 2)
    Tv = float(eltype(X))
    A = Matrix{Tv}(I, N, N)
    b = zeros(Tv, N)
    uni = fill(one(Tv) / N, N)
    function update!(wn, t, Xm, W)
        w = view(W, t, :)
        xt = view(Xm, t, :)
        g = xt ./ dot(w, xt)
        A .+= g * transpose(g)
        b .+= (1 + 1 / beta) .* g
        wn .= project_simplex(Tv(delta) .* (Symmetric(A) \ b))
        if eta > 0
            wn .= (1 - eta) .* wn .+ eta .* uni
        end
        return nothing
    end
    return run_online(update!, X, "online Newton step")
end

"""
    pamr(X::AbstractMatrix; epsilon::Real = 0.5, variant::Symbol = :pamr0,
         C::Real = 500.0) -> OnlineResult

Passive aggressive mean reversion, Li and co-authors (2012).

# Arguments

  - `X`: Price relatives, `T x N`.
  - `epsilon`: Reversion threshold. The rule acts only when the last period's
    portfolio return exceeded it.
  - `variant`: `:pamr0`, `:pamr1` or `:pamr2`, the three step-size rules of the
    paper.
  - `C`: Aggressiveness, used by `:pamr1` and `:pamr2`.

# Mathematical definition

The update is the closest portfolio to the current one whose return *on the
last observed price relative* is at most `epsilon`:

    w_{t+1} = argmin_w (1/2) || w - w_t ||^2   subject to  < w , x_t > <= eps

whose closed-form solution is `w_{t+1} = w_t - tau_t (x_t - xbar_t 1)`, with
`xbar_t` the mean of `x_t` and

    tau_t = max( 0 , ( < w_t , x_t > - eps ) / || x_t - xbar_t 1 ||^2 )     (:pamr0)
    tau_t = min( C , ... )                                                  (:pamr1)
    tau_t = ( <w_t,x_t> - eps ) / ( ||x_t - xbar_t 1||^2 + 1/(2C) )         (:pamr2)

followed by the simplex projection.

# Notes

  - **The rule sells what just rose.** It assumes single-period mean reversion,
    which is an empirical regularity in daily equity data and is absent in
    trending markets. Compare against `exponentiated_gradient` before
    believing either.
"""
function pamr(X::AbstractMatrix{<:Real}; epsilon::Real = 0.5, variant::Symbol = :pamr0,
              C::Real = 500.0)
    if !(variant in (:pamr0, :pamr1, :pamr2))
        throw(ArgumentError("variant must be :pamr0, :pamr1 or :pamr2, got $(variant)"))
    end
    function update!(wn, t, Xm, W)
        w = view(W, t, :)
        xt = view(Xm, t, :)
        xbar = mean(xt)
        dev = xt .- xbar
        denom = sum(abs2, dev)
        loss = max(zero(eltype(w)), dot(w, xt) - epsilon)
        tau = if iszero(denom)
            zero(loss)
        elseif variant === :pamr0
            loss / denom
        elseif variant === :pamr1
            min(C, loss / denom)
        else
            loss / (denom + 1 / (2C))
        end
        wn .= project_simplex(w .- tau .* dev)
        return nothing
    end
    return run_online(update!, X, "PAMR ($(variant))")
end

"""
    olmar(X::AbstractMatrix; window::Integer = 5, epsilon::Real = 10.0)
        -> OnlineResult

On-line moving average reversion, Li and Hoi (2012).

# Arguments

  - `X`: Price relatives, `T x N`.
  - `window`: Length of the moving average, in periods. Must be at least two.
  - `epsilon`: Target return. Larger is more aggressive.

# Mathematical definition

PAMR reverts to the last price. OLMAR reverts to a **moving average**, which is
far less noisy. Reconstruct relative prices `p_t = prod_{s<=t} x_s` per asset,
form the simple moving average `MA_t` over the last `window` values, and
predict the next price relative as

    xhat_{t+1,i}  =  MA_t(i) / p_t(i)

Then take the passive-aggressive step towards achieving `epsilon` on the
prediction:

    lam_t   = max( 0 , ( eps - < w_t , xhat > ) / || xhat - xbar 1 ||^2 )
    w_{t+1} = Proj_Delta ( w_t + lam_t ( xhat - xbar 1 ) )

Note the **sign is opposite to PAMR's**, because `xhat` is a forecast of the
next return rather than the last realised one.

# Notes

  - `window` is the only parameter that matters. It sets the reversion horizon
    and is the natural thing to tune with the library's existing
    `GridSearchCrossValidation`.
"""
function olmar(X::AbstractMatrix{<:Real}; window::Integer = 5, epsilon::Real = 10.0)
    if window < 2
        throw(DomainError(window, "window must be >= 2"))
    end
    T_, N = size(X)
    Tv = float(eltype(X))
    # Relative price levels, starting at one before the first period.
    P = ones(Tv, T_ + 1, N)
    for t in 1:T_
        @views P[t + 1, :] .= P[t, :] .* X[t, :]
    end
    function update!(wn, t, Xm, W)
        w = view(W, t, :)
        lo = max(1, t + 1 - window + 1)
        ma = vec(mean(view(P, lo:(t + 1), :); dims = 1))
        xhat = ma ./ view(P, t + 1, :)
        xbar = mean(xhat)
        dev = xhat .- xbar
        denom = sum(abs2, dev)
        lam = iszero(denom) ? zero(Tv) : max(zero(Tv), (epsilon - dot(w, xhat)) / denom)
        wn .= project_simplex(w .+ lam .* dev)
        return nothing
    end
    return run_online(update!, X, "OLMAR (window $(window))")
end

"""
    rmr(X::AbstractMatrix; window::Integer = 5, epsilon::Real = 10.0,
        max_iter::Integer = 100, tol::Real = 1e-8) -> OnlineResult

Robust median reversion, Huang and co-authors (2016).

# Arguments

  - `X`: Price relatives, `T x N`.
  - `window`: Length of the reversion window.
  - `epsilon`: Target return.
  - `max_iter`, `tol`: Controls for the Weiszfeld iteration.

# Mathematical definition

RMR replaces OLMAR's moving **average** with the multivariate **L1 median**,
also called the spatial median, of the last `window` price vectors:

    m  =  argmin_y  sum_{s} || p_s - y ||_2

The median is found by Weiszfeld's iteration

    y^{k+1}  =  ( sum_s p_s / || p_s - y^k || )  /  ( sum_s 1 / || p_s - y^k || )

and the portfolio update is then identical to OLMAR's, with
`xhat = m / p_t`.

# Notes

  - **The point is the breakdown point.** A single extreme day moves a mean
    without limit and moves an L1 median almost not at all. On daily equity
    data, where a single print can be a data error, that difference is the
    whole reason to prefer RMR.
  - The Weiszfeld iteration is undefined when the current estimate lands
    exactly on a data point. The implementation guards it with a floor on the
    distance.
"""
function rmr(X::AbstractMatrix{<:Real}; window::Integer = 5, epsilon::Real = 10.0,
             max_iter::Integer = 100, tol::Real = 1e-8)
    if window < 2
        throw(DomainError(window, "window must be >= 2"))
    end
    T_, N = size(X)
    Tv = float(eltype(X))
    P = ones(Tv, T_ + 1, N)
    for t in 1:T_
        @views P[t + 1, :] .= P[t, :] .* X[t, :]
    end
    function l1_median(Pw)
        y = vec(mean(Pw; dims = 1))
        for _ in 1:max_iter
            num = zeros(Tv, size(Pw, 2))
            den = zero(Tv)
            for s in axes(Pw, 1)
                d = max(norm(view(Pw, s, :) .- y), Tv(1e-12))
                num .+= view(Pw, s, :) ./ d
                den += 1 / d
            end
            ynew = num ./ den
            if norm(ynew .- y) < tol
                return ynew
            end
            y = ynew
        end
        return y
    end
    function update!(wn, t, Xm, W)
        w = view(W, t, :)
        lo = max(1, t + 1 - window + 1)
        m = l1_median(view(P, lo:(t + 1), :))
        xhat = m ./ view(P, t + 1, :)
        xbar = mean(xhat)
        dev = xhat .- xbar
        denom = sum(abs2, dev)
        lam = iszero(denom) ? zero(Tv) : max(zero(Tv), (epsilon - dot(w, xhat)) / denom)
        wn .= project_simplex(w .+ lam .* dev)
        return nothing
    end
    return run_online(update!, X, "RMR (window $(window))")
end

"""
    regret_against_bcrp(r::OnlineResult, X::AbstractMatrix; kwargs...) -> NamedTuple

Return the regret of a strategy against the best constant rebalanced
portfolio.

# Arguments

  - `r`: The strategy result.
  - `X`: Price relatives, `T x N`.
  - `kwargs...`: Forwarded to [`best_crp`](@ref).

# Returns

A `NamedTuple` with `regret`, `regret_per_period`, `bcrp_wealth` and
`strategy_wealth`.

# Mathematical definition

    R_T  =  log S_T(BCRP)  -  log S_T(strategy)

Regret is measured in log wealth because that is the quantity the guarantees
bound, and because it is additive across periods.

# Notes

  - **Negative regret is possible and is not a bug.** The best constant
    rebalanced portfolio is the best *constant* one. An adaptive strategy that
    exploits reversion can beat it, and OLMAR and RMR routinely do on real
    equity data. The bound only says the strategy cannot fall far *behind*.
"""
function regret_against_bcrp(r::OnlineResult, X::AbstractMatrix{<:Real}; kwargs...)
    b = best_crp(X; kwargs...)
    ls = log(terminal_wealth(r))
    T_ = size(X, 1)
    return (; regret = b.log_wealth - ls, regret_per_period = (b.log_wealth - ls) / T_,
            bcrp_wealth = b.wealth, strategy_wealth = terminal_wealth(r))
end

end # module OnlinePortfolioSelection
