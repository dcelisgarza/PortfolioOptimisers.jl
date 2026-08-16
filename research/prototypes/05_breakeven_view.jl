# =============================================================================
# Prototype 5 — The breakeven view: how far must the world move to change the
#               decision?
#
# Purpose
#   An optimiser answers "what is the best portfolio?". Nobody can act on that
#   answer without knowing how fragile it is. This file answers a different
#   question with the machinery the library already has:
#
#       What is the smallest change to the market's probabilities that makes
#       my second choice beat my first?
#
#   The answer is a single number in nats, and it is comparable across
#   problems, universes and dates.
#
#   The construction is an **inverse entropy pooling** problem. The library
#   already solves the forward problem in
#   `src/13_Prior/10_EntropyPoolingPrior.jl`: given views, find the
#   least-distorted probabilities that satisfy them. The inverse problem is:
#   given a decision, find the least-distorted probabilities that reverse it.
#   It is the same convex program with the view supplied by the decision
#   rather than by the user.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics`, `Optim` and
#   `LogExpFunctions`, all three of which are already direct dependencies of
#   PortfolioOptimisers.jl.
#
# Notation used throughout this file
#   T       Number of scenarios (rows of the returns matrix).
#   N       Number of assets.
#   X       Returns matrix, `T x N`. Row `t` is scenario `t`.
#   p       Prior scenario probabilities, length `T`, positive and summing to
#           one. Uniform `1/T` for a plain historical sample.
#   q       Posterior scenario probabilities, length `T`.
#   A       View matrix, `K x T`. Row `k` holds the per-scenario value whose
#           `q`-expectation the `k`-th view fixes.
#   b       View targets, length `K`.
#   lam     Dual variables, length `K`.
#   c       Decision contrast, length `T`. Entry `t` is how much better
#           portfolio A does than portfolio B in scenario `t`.
#
# Sources
#   Meucci, A. (2008). Fully flexible views: theory and practice. Risk 21(10),
#     97-102. The forward entropy pooling program.
#   Meucci, A. (2010). Historical scenarios with fully flexible probabilities.
#     GARP Risk Professional, December, 40-43. The effective number of
#     scenarios.
#   Kullback, S. and Leibler, R. A. (1951). On information and sufficiency.
#     Annals of Mathematical Statistics 22(1), 79-86.
#   Csiszar, I. (1975). I-divergence geometry of probability distributions and
#     minimization problems. Annals of Probability 3(1), 146-158. The
#     information projection onto a convex set of measures, which is what
#     `breakeven_entropy` computes.
#   Hansen, L. P. and Sargent, T. J. (2008). Robustness. Princeton University
#     Press. The economics of measuring model doubt in relative entropy, and
#     the detection-error probability that makes a nat interpretable.
# =============================================================================
module BreakevenView

using LinearAlgebra, Statistics, Optim, LogExpFunctions

export entropy_pooling, relative_entropy, effective_number_of_scenarios, breakeven_entropy,
       view_entropy_cost, detection_error_probability

"""
    relative_entropy(q::AbstractVector, p::AbstractVector) -> Real

Return the Kullback-Leibler divergence `D(q || p)` in nats.

# Arguments

  - `q`, `p`: Probability vectors of the same length. Entries of `p` must be
    positive wherever `q` is positive.

# Returns

  - The scalar `sum_t q_t * log(q_t / p_t)`, which is zero when `q == p` and
    positive otherwise.

# Details

Terms with `q_t == 0` contribute nothing, by the convention `0 * log 0 = 0`.
"""
function relative_entropy(q::AbstractVector{<:Real}, p::AbstractVector{<:Real})
    if length(q) != length(p)
        throw(DimensionMismatch("q has length $(length(q)), p has length $(length(p))"))
    end
    s = zero(float(eltype(q)))
    @inbounds for t in eachindex(q)
        if q[t] > 0
            s += q[t] * log(q[t] / p[t])
        end
    end
    return s
end

"""
    effective_number_of_scenarios(q::AbstractVector) -> Real

Return Meucci's effective number of scenarios.

# Arguments

  - `q`: Probability vector, length `T`.

# Returns

  - The scalar `exp(H(q))` where `H(q) = -sum_t q_t log q_t` is the Shannon
    entropy. The value lies in `[1, T]`.

# Details

A uniform `q` gives exactly `T`. A `q` that puts all its mass on one scenario
gives `1`. Divide by `T` to read it as a fraction: **the share of the sample
that a reweighting actually still uses.** A posterior with an effective
scenario count of thirty out of two thousand is not a mild tilt, whatever its
relative entropy looks like.
"""
function effective_number_of_scenarios(q::AbstractVector{<:Real})
    h = zero(float(eltype(q)))
    @inbounds for t in eachindex(q)
        if q[t] > 0
            h -= q[t] * log(q[t])
        end
    end
    return exp(h)
end

"""
    entropy_pooling(p::AbstractVector, A::AbstractMatrix, b::AbstractVector;
                    max_iter::Integer = 500, g_tol::Real = 1e-10)
        -> NamedTuple

Solve the forward entropy pooling problem.

# Arguments

  - `p`: Prior probabilities, length `T`, positive and summing to one.
  - `A`: View matrix, `K x T`. Row `k` holds the per-scenario quantity whose
    expectation view `k` fixes. To state a view on the mean of asset `j`, pass
    the `j`-th column of `X` as a row of `A`.
  - `b`: View targets, length `K`.
  - `max_iter`: Iteration cap for the dual solver.
  - `g_tol`: Gradient tolerance for the dual solver.

# Returns

A `NamedTuple`:

  - `q`: Posterior probabilities, length `T`.
  - `lambda`: Dual variables, length `K`.
  - `divergence`: `D(q || p)` in nats.
  - `converged`: Whether the dual solver reported convergence.
  - `view_error`: `maximum(abs.(A * q - b))`, the residual of the views.

# Mathematical definition

The primal problem is the information projection of `p` onto the affine set of
measures that satisfy the views:

    minimise    sum_t q_t log(q_t / p_t)
    over        q
    subject to  sum_t q_t = 1,   q >= 0,   A q = b

It is strictly convex on the simplex, so the solution is unique when it exists.
The Lagrangian dual has the exponential-family form

    q_t(lam)  =  p_t * exp( -(A' lam)_t )  /  Z(lam)
    Z(lam)    =  sum_t p_t * exp( -(A' lam)_t )

and the dual objective to **minimise** over the unconstrained `lam` is

    L(lam)  =  log Z(lam)  +  lam' b

with gradient `grad L = b - A q(lam)`. The gradient vanishes exactly when the
views hold, which is the optimality condition. At the optimum the divergence is

    D(q* || p)  =  -L(lam*)

so the solver returns the answer and its own cost in the same number. The
routine below asserts that identity.

# Notes

  - The dual is unconstrained and `K`-dimensional, and `K` is the number of
    views, typically under ten. **The cost of the program does not grow with
    the number of scenarios**, only the cost of each function evaluation does.
    That is why entropy pooling scales to a hundred thousand scenarios where a
    primal simplex method would not.
  - An infeasible view set makes the dual unbounded below and the solver walks
    off to infinity. Check `view_error` before trusting `q`.
"""
function entropy_pooling(p::AbstractVector{<:Real}, A::AbstractMatrix{<:Real},
                         b::AbstractVector{<:Real}; max_iter::Integer = 500,
                         g_tol::Real = 1e-10)
    T = length(p)
    K = length(b)
    if size(A) != (K, T)
        throw(DimensionMismatch("A must be $(K) x $(T) to match b and p, got $(size(A))"))
    end
    if any(<=(0), p)
        throw(DomainError(minimum(p), "every prior probability must be > 0"))
    end
    logp = log.(p)

    # Posterior for a given dual point, in log space for stability.
    function posterior(lam)
        s = logp .- vec(transpose(A) * lam)
        return exp.(s .- logsumexp(s))
    end
    # Dual objective L(lam) = log Z(lam) + lam' b.
    function dual_obj(lam)
        s = logp .- vec(transpose(A) * lam)
        return logsumexp(s) + dot(lam, b)
    end
    # Dual gradient grad L = b - A q(lam).
    function dual_grad!(g, lam)
        g .= b .- A * posterior(lam)
        return g
    end

    lam0 = zeros(float(eltype(p)), K)
    res = Optim.optimize(dual_obj, dual_grad!, lam0, Optim.LBFGS(),
                         Optim.Options(; iterations = max_iter, g_tol = g_tol))
    lam = Optim.minimizer(res)
    q = posterior(lam)
    return (; q = q, lambda = lam, divergence = relative_entropy(q, p),
            converged = Optim.converged(res), view_error = maximum(abs, A * q .- b),
            dual_value = Optim.minimum(res))
end

"""
    breakeven_entropy(c::AbstractVector, p::AbstractVector; kwargs...)
        -> NamedTuple

Return the smallest relative entropy that reverses a decision.

**This is the central function of this prototype.**

# Arguments

  - `c`: Decision contrast, length `T`. Entry `t` is the amount by which the
    incumbent choice beats the challenger in scenario `t`. For two portfolios
    `w_a` and `w_b` on a returns matrix `X`, pass `c = X * (w_a - w_b)`.
  - `p`: Prior probabilities, length `T`. Uniform for a plain sample.
  - `kwargs...`: Forwarded to [`entropy_pooling`](@ref).

# Returns

A `NamedTuple`:

  - `divergence`: The breakeven entropy `delta*` in nats, or `Inf` when no
    measure on these scenarios can reverse the decision.
  - `q`: The probabilities that achieve it, length `T`. **This is the world the
    caller is betting against**, and it is a plain reweighting of scenarios the
    caller has already seen, so it can be read and argued with.
  - `ens`: The effective number of scenarios of `q`.
  - `ens_fraction`: `ens / T`.
  - `prior_edge`: `E_p[c]`, the incumbent's advantage under the prior.
  - `status`: `:already_reversed`, `:solved` or `:unreachable`.

# Mathematical definition

The problem is the information projection of `p` onto the half-space of
measures under which the incumbent no longer wins:

    delta*  =  min { D(q || p)  :  sum_t q_t c_t <= 0,  sum_t q_t = 1,  q >= 0 }

Three cases exhaust it:

 1. `E_p[c] <= 0`. The prior already prefers the challenger, so `q = p` and
    `delta* = 0`.
 2. `min_t c_t > 0`. The incumbent wins in **every** scenario, so no
    reweighting of these scenarios can reverse it and `delta* = Inf`. The
    decision is not fragile with respect to probabilities. It could still be
    fragile with respect to the scenario set itself, which is a Wasserstein
    question, not an entropy one. See prototype 2.
 3. Otherwise the constraint binds at the boundary, so the inequality may be
    replaced by the equality `E_q[c] = 0`, and the problem is exactly a
    one-view entropy pooling program. The dual is one-dimensional.

The reduction in case 3 is what makes this cheap: **one scalar convex problem
answers a question about the stability of the whole optimisation.**

# Notes

  - Read `delta*` with [`detection_error_probability`](@ref), which converts
    nats into the probability that a statistician with this sample could not
    tell the two worlds apart. A breakeven entropy that a test would reject at
    once means the decision is safe. One that no test could detect means the
    optimiser is reporting a preference it cannot support.
  - The statistic is **not** a function of the optimiser. It compares two
    portfolios, so it works for a hierarchical portfolio, a naive one, or a
    portfolio a client sent in an email.
"""
function breakeven_entropy(c::AbstractVector{<:Real}, p::AbstractVector{<:Real}; kwargs...)
    if length(c) != length(p)
        throw(DimensionMismatch("c has length $(length(c)), p has length $(length(p))"))
    end
    T = length(p)
    edge = dot(p, c)
    if edge <= 0
        return (; divergence = zero(float(edge)), q = collect(float.(p)),
                ens = effective_number_of_scenarios(p),
                ens_fraction = effective_number_of_scenarios(p) / T, prior_edge = edge,
                status = :already_reversed)
    end
    if minimum(c) > 0
        return (; divergence = Inf, q = collect(float.(p)),
                ens = effective_number_of_scenarios(p),
                ens_fraction = effective_number_of_scenarios(p) / T, prior_edge = edge,
                status = :unreachable)
    end
    A = reshape(collect(float.(c)), 1, T)
    b = [zero(float(eltype(c)))]
    sol = entropy_pooling(p, A, b; kwargs...)
    ens = effective_number_of_scenarios(sol.q)
    return (; divergence = sol.divergence, q = sol.q, ens = ens, ens_fraction = ens / T,
            prior_edge = edge, status = :solved, view_error = sol.view_error)
end

"""
    view_entropy_cost(x::AbstractVector, target::Real, p::AbstractVector; kwargs...)
        -> NamedTuple

Return the relative entropy cost of asserting that `E[x] == target`.

# Arguments

  - `x`: Per-scenario values of the quantity the view is about, length `T`. For
    a view on asset `j`, pass the `j`-th column of the returns matrix. For a
    view on a portfolio, pass `X * w`. For a view on a factor, pass the factor
    column.
  - `target`: The asserted expectation.
  - `p`: Prior probabilities, length `T`.
  - `kwargs...`: Forwarded to [`entropy_pooling`](@ref).

# Returns

A `NamedTuple` with `divergence`, `q`, `ens`, `ens_fraction` and `view_error`.

# Notes

  - This function makes views **comparable**. A caller with five candidate
    views has no way today to say which one is the bold claim. Ranking them by
    entropy cost says exactly that, in one unit, without asking the caller to
    guess a confidence.
  - The cost is zero when `target` equals the prior expectation, and it grows
    without bound as `target` approaches the largest value in `x`, because the
    posterior must concentrate on a single scenario to reach it.
"""
function view_entropy_cost(x::AbstractVector{<:Real}, target::Real,
                           p::AbstractVector{<:Real}; kwargs...)
    T = length(x)
    if length(p) != T
        throw(DimensionMismatch("x has length $(T), p has length $(length(p))"))
    end
    if !(minimum(x) < target < maximum(x))
        throw(DomainError(target,
                          "target must lie strictly inside ($(minimum(x)), $(maximum(x))), the range of the scenarios; no reweighting can reach it otherwise"))
    end
    A = reshape(collect(float.(x)), 1, T)
    sol = entropy_pooling(p, A, [float(target)]; kwargs...)
    ens = effective_number_of_scenarios(sol.q)
    return (; divergence = sol.divergence, q = sol.q, ens = ens, ens_fraction = ens / T,
            view_error = sol.view_error)
end

"""
    detection_error_probability(divergence::Real, n_obs::Integer) -> Real

Convert a relative entropy into the probability that a statistician cannot tell
the two worlds apart with `n_obs` observations.

# Arguments

  - `divergence`: Relative entropy in nats, per observation.
  - `n_obs`: Number of observations available to the test.

# Returns

  - A probability in `(0, 0.5]`. Large means the two worlds are hard to
    distinguish.

# Mathematical definition

Chernoff's bound on the error of the optimal test between two hypotheses
separated by relative entropy `D` over `n` independent observations gives the
approximation used by Hansen and Sargent (2008),

    p_detect  ~  Phi( -sqrt( n * D / 2 ) )

where `Phi` is the standard normal distribution function. It is an
approximation, and it is the standard one in the robust control literature.

# Notes

  - The reading is the point of the whole prototype. A breakeven entropy of
    `0.002` nats over `2000` observations gives about `0.16`: **a competent
    statistician with the same data would fail to reject the alternative world
    about one time in six.** A portfolio whose ranking flips inside that band
    is not a conclusion, it is a coin toss with extra steps.
"""
function detection_error_probability(divergence::Real, n_obs::Integer)
    if divergence < 0
        throw(DomainError(divergence, "divergence must be >= 0"))
    end
    if n_obs <= 0
        throw(DomainError(n_obs, "n_obs must be > 0"))
    end
    if isinf(divergence)
        return 0.0
    end
    z = -sqrt(n_obs * divergence / 2)
    return (1 + _erf(z / sqrt(2))) / 2
end

"""
    _erf(x::Real) -> Float64

Abramowitz and Stegun 7.1.26 rational approximation to the error function.
Maximum absolute error about `1.5e-7`.
"""
function _erf(x::Real)
    s = sign(x)
    z = abs(float(x))
    t = 1 / (1 + 0.3275911z)
    y = 1 -
        (((((1.061405429t - 1.453152027)t) + 1.421413741)t - 0.284496736)t + 0.254829592)t *
        exp(-z * z)
    return s * y
end

end # module BreakevenView
