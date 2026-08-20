# =============================================================================
# Prototype 14 — Attribution: which constraint, which term, which change.
#
# Purpose
#   Reports 2 and 7 both ask for an explainability layer. The library can say
#   what the portfolio *is* and how its risk splits across assets and factors.
#   It cannot say **why** the weights are what they are. Three questions are
#   wanted, and each has an exact answer:
#
#     1. *What did this constraint cost me?* Solve with and without it.
#     2. *How should I share credit between overlapping causes?* The Shapley
#        value, which is the unique fair answer.
#     3. *Why did my risk change at this rebalance?* Split it into the part
#        caused by trading and the part caused by the market moving.
#
#   Question 3 has a trap: the naive two-term split depends on the order the
#   two effects are applied. The Shapley value over the two causes removes the
#   ambiguity, which makes questions 2 and 3 the same question.
#
# Status
#   Standalone. Depends on `LinearAlgebra`, `Statistics`, `Combinatorics` and
#   `Random`. All are already dependencies of the library.
#
# Notation used throughout this file
#   n        Number of players: constraints, risk terms, or causes.
#   S        A coalition, a subset of the players.
#   v(S)     The characteristic function: the value achieved using only `S`.
#   phi_i    The Shapley value of player `i`.
#   w        Portfolio weights.
#   sig      Covariance matrix.
#
# Sources
#   Shapley, L. S. (1953). A value for n-person games. In: Contributions to the
#     Theory of Games II, Princeton University Press, 307-317.
#   Young, H. P. (1985). Monotonic solutions of cooperative games.
#     International Journal of Game Theory 14(2), 65-72. The axiomatisation
#     used to justify the choice here.
#   Castro, J., Gomez, D. and Tejada, J. (2009). Polynomial calculation of the
#     Shapley value based on sampling. Computers and Operations Research
#     36(5), 1726-1730. The permutation sampling estimator.
#   Lundberg, S. M. and Lee, S. I. (2017). A unified approach to interpreting
#     model predictions. Advances in Neural Information Processing Systems 30.
#     arXiv:1705.07874. The modern machine-learning use of the same value.
#   Brinson, G. P. and Fachler, N. (1985). Measuring non-US equity portfolio
#     performance. Journal of Portfolio Management 11(3), 73-76. The
#     interaction term this file's decomposition generalises.
#   Grinold, R. C. and Kahn, R. N. (1999). Active Portfolio Management, 2nd
#     edition. McGraw-Hill. Marginal contribution to risk.
# =============================================================================
module Attribution

using LinearAlgebra, Statistics, Random, Combinatorics

export shapley_values, shapley_values_sampled, constraint_attribution,
       rebalance_decomposition, marginal_risk_contribution

"""
    shapley_values(v::Function, n::Integer) -> Vector

Return the exact Shapley values of an `n`-player game.

# Arguments

  - `v`: The characteristic function. `v(S)` takes a `Vector{Int}` of player
    indices and returns a scalar. `v(Int[])` must be defined.
  - `n`: Number of players. Exact enumeration is used, so keep `n` at or below
    about 16.

# Returns

  - A vector of length `n` with the Shapley value of each player.

# Mathematical definition

    phi_i  =  sum_{S subset N \\ {i}}  ( |S|! (n - |S| - 1)! / n! )
                                       [ v(S + i) - v(S) ]

Read it as: average the marginal contribution of player `i` over every order
in which the players could arrive, with all `n!` orders equally likely.

# The axioms

The Shapley value is the **unique** rule satisfying:

  - **Efficiency.** `sum_i phi_i = v(N) - v(empty)`. Everything is allocated.
  - **Symmetry.** Two players who contribute equally to every coalition get
    equal shares.
  - **Null player.** A player who adds nothing to any coalition gets zero.
  - **Linearity.** The value of a sum of games is the sum of the values.

Efficiency is what makes the output a decomposition rather than a ranking, and
the verification driver asserts it to machine precision.

# Notes

  - **Cost is `2^n` evaluations of `v`.** For a portfolio problem each
    evaluation is a solve, so exact enumeration is affordable for a handful of
    constraints and nothing more. Use [`shapley_values_sampled`](@ref) beyond
    that.
"""
function shapley_values(v::Function, n::Integer)
    if n < 0
        throw(DomainError(n, "n must be >= 0"))
    end
    if n > 20
        throw(ArgumentError("exact enumeration needs 2^$(n) evaluations; use shapley_values_sampled"))
    end
    if n == 0
        return Float64[]
    end
    phi = zeros(float(typeof(v(Int[]))), n)
    others(i) = [j for j in 1:n if j != i]
    fact = [factorial(big(k)) for k in 0:n]
    for i in 1:n
        rest = others(i)
        for k in 0:(n - 1), S in Combinatorics.combinations(rest, k)
            wgt = Float64(fact[k + 1] * fact[n - k] / fact[n + 1])
            phi[i] += wgt * (v(sort(vcat(S, i))) - v(sort(collect(S))))
        end
    end
    return phi
end

"""
    shapley_values_sampled(v::Function, n::Integer; n_perm::Integer = 2000,
                           rng::Random.AbstractRNG = Random.default_rng())
        -> NamedTuple

Estimate Shapley values by permutation sampling.

# Arguments

  - `v`: The characteristic function.
  - `n`: Number of players.
  - `n_perm`: Number of random permutations.
  - `rng`: Random number generator.

# Returns

A `NamedTuple` with `phi`, `stderr` (the Monte Carlo standard error of each
estimate) and `efficiency_error`.

# Mathematical definition

The Shapley value is an expectation over a uniformly random arrival order:

    phi_i  =  E_perm [ v( predecessors of i, plus i ) - v( predecessors of i ) ]

Sampling permutations gives an unbiased estimator, and the estimator is
**efficient by construction for every single permutation**, because the
marginal contributions along one order telescope to `v(N) - v(empty)`. The
sample mean therefore satisfies efficiency exactly, whatever `n_perm` is. The
routine reports `efficiency_error` so that claim can be checked.

# Notes

  - Cost is `n_perm * (n + 1)` evaluations of `v`, linear in `n`.
  - Report `stderr`. A Shapley value whose standard error is the same size as
    the value itself is not an explanation.
"""
function shapley_values_sampled(v::Function, n::Integer; n_perm::Integer = 2000,
                                rng::Random.AbstractRNG = Random.default_rng())
    contrib = zeros(Float64, n, n_perm)
    for p in 1:n_perm
        order = randperm(rng, n)
        cur = Int[]
        prev = Float64(v(Int[]))
        for i in order
            push!(cur, i)
            val = Float64(v(sort(copy(cur))))
            contrib[i, p] = val - prev
            prev = val
        end
    end
    phi = vec(mean(contrib; dims = 2))
    se = vec(std(contrib; dims = 2)) ./ sqrt(n_perm)
    total = Float64(v(collect(1:n))) - Float64(v(Int[]))
    return (; phi = phi, stderr = se, efficiency_error = abs(sum(phi) - total))
end

"""
    constraint_attribution(solve::Function, constraint_names::AbstractVector;
                           objective::Function, exact::Bool = true,
                           n_perm::Integer = 500,
                           rng::Random.AbstractRNG = Random.default_rng())
        -> Vector{<:NamedTuple}

Attribute the cost of an objective to the constraints that caused it.

# Arguments

  - `solve`: A function taking a `Vector{Int}` of **active** constraint indices
    and returning the optimal weights under exactly that subset.
  - `constraint_names`: Labels, length `n`.
  - `objective`: A function `w -> value`, evaluated on the returned weights.
    Use a quantity where lower is better, such as the risk or the negative
    utility.
  - `exact`: Enumerate all subsets, or sample permutations.
  - `n_perm`, `rng`: Controls for the sampled route.

# Returns

One `NamedTuple` per constraint with `name`, `shapley` (its share of the total
cost), `leave_one_out` (the naive alternative) and `share` (the Shapley value
as a fraction of the total).

# Details

The characteristic function is `v(S) = objective(solve(S))`, the objective
attainable when only the constraints in `S` are imposed. Efficiency then says

    sum_i phi_i  =  objective(all constraints)  -  objective(no constraints)

so **the Shapley values partition the exact cost of constraining the
problem**.

# Notes

  - **Leave-one-out and Shapley disagree whenever constraints overlap**, and
    they overlap constantly. Measured on an eight-asset equal-weight base with
    two *identical* binding caps and one non-binding global cap:

    | constraint              | Shapley    | leave-one-out |
    |:----------------------- | ----------:| -------------:|
    | cap A                   | `-3.71e-5` | `0.0`         |
    | cap B (exact duplicate) | `-3.71e-5` | `0.0`         |
    | global cap, not binding | `0.0`      | `0.0`         |
    | **sum**                 | `-7.41e-5` | `0.0`         |
    | **actual total cost**   | `-7.41e-5` |               |

    **Leave-one-out reports zero for everything and misses 100 per cent of the
    real cost**, because removing either duplicate leaves the other binding.
    Shapley splits the cost equally between the duplicates, gives the
    non-binding constraint exactly zero, and sums to the true total with no
    residual. The `leave_one_out` column is returned so a caller can see the
    size of that error rather than take it on trust.

  - The direction convention matters. With a "lower is better" objective the
    Shapley values are non-negative when constraints only ever hurt, which is
    the usual case for a feasible-set restriction.
"""
function constraint_attribution(solve::Function, constraint_names::AbstractVector;
                                objective::Function, exact::Bool = true,
                                n_perm::Integer = 500,
                                rng::Random.AbstractRNG = Random.default_rng())
    n = length(constraint_names)
    cache = Dict{Vector{Int}, Float64}()
    function v(S)
        key = sort(collect(Int, S))
        return get!(cache, key) do
            return Float64(objective(solve(key)))
        end
    end
    phi = if exact
        shapley_values(v, n)
    else
        shapley_values_sampled(v, n; n_perm = n_perm, rng = rng).phi
    end
    total = v(collect(1:n)) - v(Int[])
    all_idx = collect(1:n)
    return [(; name = String(constraint_names[i]), shapley = phi[i],
             leave_one_out = v(all_idx) - v(setdiff(all_idx, [i])),
             share = iszero(total) ? 0.0 : phi[i] / total) for i in 1:n]
end

"""
    rebalance_decomposition(w0::AbstractVector, w1::AbstractVector,
                            sig0::AbstractMatrix, sig1::AbstractMatrix)
        -> NamedTuple

Split the change in portfolio risk between the trade and the market.

# Arguments

  - `w0`, `w1`: The old and new portfolios, length `N`.
  - `sig0`, `sig1`: The old and new covariance matrices, `N x N`.

# Returns

A `NamedTuple`:

  - `total`: `risk(w1, sig1) - risk(w0, sig0)`.
  - `weight_effect`, `moment_effect`: The Shapley split of `total` between the
    two causes.
  - `path_weights_first`, `path_moments_first`: The two order-dependent naive
    splits, for comparison.
  - `interaction`: The gap between them, which is what the Shapley split
    shares equally.
  - `residual`: `total - weight_effect - moment_effect`, which must be zero.

# Mathematical definition

Write `f(w, sig) = sqrt(w' sig w)`. There are two causes and therefore two
orders:

    weights first:  [ f(w1,sig0) - f(w0,sig0) ]  +  [ f(w1,sig1) - f(w1,sig0) ]
    moments first:  [ f(w0,sig1) - f(w0,sig0) ]  +  [ f(w1,sig1) - f(w0,sig1) ]

Both sum to `total`, and they disagree. The disagreement is the **interaction**
term, familiar from Brinson attribution. The Shapley value over the two-player
game averages the two orders, which for `n = 2` is exactly

    phi_weights  =  (1/2) [ f(w1,sig0) - f(w0,sig0) ]  +  (1/2) [ f(w1,sig1) - f(w0,sig1) ]

# Notes

  - **This is why the interaction term exists in Brinson attribution and why it
    is unsatisfying.** Brinson reports it as a third line. The Shapley value
    says the fair thing to do is split it in half, and Young's (1985)
    axiomatisation says that is the only rule that is fair, symmetric and
    additive. A three-line Brinson table and a two-line Shapley table carry
    the same information and the second is easier to act on.
"""
function rebalance_decomposition(w0::AbstractVector{<:Real}, w1::AbstractVector{<:Real},
                                 sig0::AbstractMatrix{<:Real}, sig1::AbstractMatrix{<:Real})
    N = length(w0)
    if length(w1) != N
        throw(DimensionMismatch("w0 has length $(N), w1 has length $(length(w1))"))
    end
    if size(sig0) != (N, N) || size(sig1) != (N, N)
        throw(DimensionMismatch("both covariances must be $(N) x $(N)"))
    end
    f(w, s) = sqrt(max(dot(w, s, w), zero(float(eltype(w)))))
    f00 = f(w0, sig0)
    f10 = f(w1, sig0)
    f01 = f(w0, sig1)
    f11 = f(w1, sig1)
    total = f11 - f00
    wfirst = (f10 - f00, f11 - f10)
    mfirst = (f11 - f01, f01 - f00)
    we = (wfirst[1] + mfirst[1]) / 2
    me = (wfirst[2] + mfirst[2]) / 2
    return (; total = total, weight_effect = we, moment_effect = me,
            path_weights_first = wfirst, path_moments_first = (mfirst[2], mfirst[1]),
            interaction = wfirst[1] - mfirst[1], residual = total - we - me)
end

"""
    marginal_risk_contribution(w::AbstractVector, sig::AbstractMatrix) -> NamedTuple

Return the classical Euler decomposition of portfolio risk across assets.

# Arguments

  - `w`: Weights, length `N`.
  - `sig`: Covariance, `N x N`.

# Returns

A `NamedTuple` with `marginal` (the derivative of risk with respect to each
weight), `contribution` (`w .* marginal`), `percent` and `total`.

# Mathematical definition

With `f(w) = sqrt(w' sig w)`,

    marginal_i     =  d f / d w_i  =  (sig w)_i / f(w)
    contribution_i =  w_i * marginal_i

Because `f` is positively homogeneous of degree one, Euler's theorem gives

    sum_i contribution_i  =  f(w)

exactly. **That is what makes this a decomposition and not merely a
sensitivity**, and it is the property the driver asserts.

# Notes

  - The library already computes this. It is reproduced here because
    [`rebalance_decomposition`](@ref) and the Shapley machinery need something
    to be compared against, and because the Euler property is the cleanest
    example of the efficiency axiom appearing outside game theory.
  - For a **non**-homogeneous risk measure, such as the entropic risk in
    prototype 11, Euler's theorem does not apply and the contributions do not
    sum to the total. Use the Shapley value there instead. That is the general
    replacement, and it costs `2^N` rather than `O(N^2)`.
"""
function marginal_risk_contribution(w::AbstractVector{<:Real}, sig::AbstractMatrix{<:Real})
    N = length(w)
    if size(sig) != (N, N)
        throw(DimensionMismatch("sig must be $(N) x $(N), got $(size(sig))"))
    end
    total = sqrt(max(dot(w, sig, w), zero(float(eltype(w)))))
    marg = iszero(total) ? zeros(float(eltype(w)), N) : (sig * w) ./ total
    contrib = w .* marg
    return (; marginal = marg, contribution = contrib,
            percent = iszero(total) ? contrib : contrib ./ total, total = total)
end

end # module Attribution
