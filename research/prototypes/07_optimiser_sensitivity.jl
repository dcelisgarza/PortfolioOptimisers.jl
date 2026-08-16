# =============================================================================
# Prototype 7 — Sensitivity of the optimal portfolio, by implicit differentiation.
#
# Purpose
#   Reports 1, 3 and 7 all ask for the same object under three names: a
#   "differentiable portfolio layer", "automatic differentiation through the
#   pipeline", and "sensitivity analysis". They are one thing. The derivative
#
#       d w* / d theta      for theta in { mu, sigma, b }
#
#   answers all three. With it a caller can:
#
#     1. Read which input actually moves the portfolio, and by how much.
#     2. Train a return model on realised portfolio performance rather than on
#        a proxy mean squared error, because the layer passes gradients.
#     3. Price a constraint. The multiplier on a constraint is the marginal
#        cost of tightening it, in the units of the objective.
#
#   No new dependency is needed. The derivative is a linear solve against the
#   KKT matrix the problem already forms.
#
# Status
#   Standalone. Depends on `LinearAlgebra` only. The verification driver also
#   uses finite differences, which need nothing.
#
# Notation used throughout this file
#   N       Number of assets.
#   M       Number of active equality constraints, the budget included.
#   mu      Expected returns, length `N`.
#   sigma   Covariance matrix, `N x N`, symmetric positive definite.
#   gamma   Risk-aversion parameter, positive. Larger means more risk averse.
#   A       Active constraint matrix, `M x N`.
#   b       Active constraint right-hand side, length `M`.
#   w       Portfolio weights, length `N`.
#   nu      Lagrange multipliers of the active constraints, length `M`.
#   K       The `N x N` top-left block of the inverse KKT matrix. Every
#           sensitivity in this file is built from it.
#
# Sources
#   Amos, B. and Kolter, J. Z. (2017). OptNet: differentiable optimization as a
#     layer in neural networks. Proceedings of the 34th International
#     Conference on Machine Learning, 136-145. arXiv:1703.00443. The
#     differentiable quadratic-programme layer.
#   Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S. and Kolter, J. Z.
#     (2019). Differentiable convex optimization layers. Advances in Neural
#     Information Processing Systems 32. arXiv:1910.12430.
#   Donti, P., Amos, B. and Kolter, J. Z. (2017). Task-based end-to-end model
#     learning in stochastic optimization. Advances in Neural Information
#     Processing Systems 30. arXiv:1703.04529. The argument for training a
#     return model on the decision rather than on the forecast error.
#   Fiacco, A. V. (1976). Sensitivity analysis for nonlinear programming using
#     penalty methods. Mathematical Programming 10(1), 287-311. The classical
#     statement of the implicit function theorem for a parametric programme.
#   Best, M. J. and Grauer, R. R. (1991). On the sensitivity of mean-variance
#     efficient portfolios to changes in asset means. Review of Financial
#     Studies 4(2), 315-342. The finance result this machinery quantifies.
# =============================================================================
module OptimiserSensitivity

using LinearAlgebra

export QPSolution, solve_equality_qp, kkt_inverse_block, dw_dmu, dw_db, dw_dsigma_direction,
       sensitivity_report, pullback_mu

"""
    QPSolution{T}

The solution of an equality-constrained mean-variance problem, together with
the factorised KKT system that produced it.

# Fields

  - `w::Vector{T}`: Optimal weights, length `N`.
  - `nu::Vector{T}`: Multipliers of the active constraints, length `M`. Entry
    `m` is the rate at which the objective improves when `b[m]` rises by one
    unit. **This is the price of the constraint.**
  - `K::Matrix{T}`: The `N x N` top-left block of the inverse KKT matrix.
  - `G::Matrix{T}`: The `N x M` top-right block of the inverse KKT matrix.
  - `gamma::T`: The risk-aversion parameter used.

# Notes

  - `K` and `G` are stored because every sensitivity in this file is a product
    with one of them. Forming them once costs a single `(N + M)` linear solve.
"""
struct QPSolution{T <: Real}
    w::Vector{T}
    nu::Vector{T}
    K::Matrix{T}
    G::Matrix{T}
    gamma::T
end

"""
    solve_equality_qp(mu::AbstractVector, sigma::AbstractMatrix,
                      A::AbstractMatrix, b::AbstractVector; gamma::Real = 1.0)
        -> QPSolution

Solve the equality-constrained mean-variance problem in closed form.

# Arguments

  - `mu`: Expected returns, length `N`.
  - `sigma`: Covariance, `N x N`, symmetric positive definite.
  - `A`: Active constraint matrix, `M x N`. For a plain budget constraint pass
    `ones(1, N)`.
  - `b`: Right-hand side, length `M`. For a plain budget pass `[1.0]`.
  - `gamma`: Risk aversion, positive.

# Returns

  - A [`QPSolution`](@ref).

# Mathematical definition

The problem is

    minimise    (1/2) w' sigma w  -  (1/gamma) mu' w
    over        w
    subject to  A w = b

Its Lagrangian is stationary when

    sigma w  -  (1/gamma) mu  +  A' nu  =  0
    A w                                 =  b

which is the linear system

    [ sigma   A' ] [  w ]  =  [ mu / gamma ]
    [   A      0 ] [ nu ]     [     b      ]

Write the coefficient matrix as `Z`, of size `(N + M)`. Then

    [ K   G  ]  =  Z^(-1),      K is N x N,   G is N x M

and the solution is `w = K mu / gamma + G b`.

# Notes

  - **Inequality constraints are handled by the active set.** At a
    non-degenerate optimum the inactive constraints have zero multipliers and
    do not influence the local derivative, so passing the *active* rows of the
    inequality system as equalities gives the correct sensitivity. The
    derivative is then valid in a neighbourhood in which the active set does
    not change, which is exactly the region a sensitivity is meaningful in.
  - `Z` is symmetric but indefinite, so it is a saddle-point system. `bunchkaufman`
    is the right factorisation. `lu` also works and is used here for clarity.

# Validation

  - `size(A, 2) == length(mu) == size(sigma, 1)`.
  - `size(A, 1) == length(b)`.
  - `gamma > 0`.
"""
function solve_equality_qp(mu::AbstractVector{<:Real}, sigma::AbstractMatrix{<:Real},
                           A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real};
                           gamma::Real = 1.0)
    N = length(mu)
    M = length(b)
    if size(sigma) != (N, N)
        throw(DimensionMismatch("sigma must be $(N) x $(N), got $(size(sigma))"))
    end
    if size(A) != (M, N)
        throw(DimensionMismatch("A must be $(M) x $(N), got $(size(A))"))
    end
    if !(gamma > 0)
        throw(DomainError(gamma, "gamma must be > 0"))
    end
    T = float(promote_type(eltype(mu), eltype(sigma), eltype(A), eltype(b)))
    Z = zeros(T, N + M, N + M)
    Z[1:N, 1:N] .= sigma
    Z[1:N, (N + 1):end] .= transpose(A)
    Z[(N + 1):end, 1:N] .= A
    Zinv = inv(Z)
    K = Zinv[1:N, 1:N]
    G = Zinv[1:N, (N + 1):end]
    rhs = vcat(mu ./ gamma, b)
    sol = Z \ rhs
    return QPSolution{T}(sol[1:N], sol[(N + 1):end], K, G, T(gamma))
end

"""
    kkt_inverse_block(s::QPSolution) -> Matrix

Return the block `K`, the top-left `N x N` block of the inverse KKT matrix.

# Details

`K` is the projected inverse covariance. For a single budget constraint it
equals

    K = sigma^(-1) - (sigma^(-1) 1 1' sigma^(-1)) / (1' sigma^(-1) 1)

which is the inverse covariance projected onto the subspace of weight changes
that keep the budget satisfied. **`K` is symmetric positive semi-definite and
its null space is spanned by the rows of `A`.** That is the formal statement of
an obvious fact: a change in expected returns cannot move the portfolio in a
direction that would break the budget.
"""
kkt_inverse_block(s::QPSolution) = s.K

"""
    dw_dmu(s::QPSolution) -> Matrix

Return the Jacobian of the optimal weights with respect to expected returns.

# Arguments

  - `s`: The solved problem.

# Returns

  - An `N x N` matrix whose `(i, j)` entry is `d w_i / d mu_j`.

# Mathematical definition

    d w / d mu  =  K / gamma

# Notes

  - **This matrix is the quantitative form of the Markowitz enigma.** Best and
    Grauer (1991) showed that mean-variance weights react violently to small
    changes in the means. The reason is visible here: `K` is a projected
    inverse covariance, so its largest entries scale with the reciprocal of the
    *smallest* eigenvalue of `sigma`. A near-singular covariance, which is what
    a sample covariance is when `T` is close to `N`, makes this Jacobian
    enormous.
  - The norm of this matrix is therefore a **usable diagnostic**, available
    before any backtest: it says how much the portfolio amplifies estimation
    error in the means.
"""
dw_dmu(s::QPSolution) = s.K ./ s.gamma

"""
    dw_db(s::QPSolution) -> Matrix

Return the Jacobian of the optimal weights with respect to the constraint
right-hand side.

# Arguments

  - `s`: The solved problem.

# Returns

  - An `N x M` matrix whose `(i, m)` entry is `d w_i / d b_m`.

# Mathematical definition

    d w / d b  =  G

# Notes

  - This prices a constraint in **portfolio** units rather than in objective
    units. The multiplier `s.nu[m]` says what relaxing constraint `m` is worth.
    This column says *where the money goes* when it is relaxed. A risk
    committee usually wants the second.
"""
dw_db(s::QPSolution) = s.G

"""
    dw_dsigma_direction(s::QPSolution, E::AbstractMatrix) -> Vector

Return the directional derivative of the optimal weights when the covariance
moves in the direction `E`.

# Arguments

  - `s`: The solved problem.
  - `E`: A symmetric `N x N` perturbation direction.

# Returns

  - A vector of length `N`, equal to `d/dt w*(sigma + t E)` at `t = 0`.

# Mathematical definition

Differentiate the KKT system `Z(sigma) z = c` with respect to `t`:

    (dZ/dt) z  +  Z (dz/dt)  =  0
    =>  dz/dt  =  -Z^(-1) (dZ/dt) z

Only the top-left block of `Z` depends on `sigma`, and `dZ/dt` there is `E`.
Taking the first `N` rows,

    d w / dt  =  - K E w

# Notes

  - The full derivative with respect to `sigma` is a three-index object, of
    size `N x N x N`. **A directional derivative avoids forming it**, and a
    direction is what a caller actually has: a shrinkage direction, a
    correlation shock, a single-entry bump. This is the same trick that makes
    reverse-mode differentiation cheap.
"""
function dw_dsigma_direction(s::QPSolution, E::AbstractMatrix{<:Real})
    N = length(s.w)
    if size(E) != (N, N)
        throw(DimensionMismatch("E must be $(N) x $(N), got $(size(E))"))
    end
    return -(s.K * (E * s.w))
end

"""
    pullback_mu(s::QPSolution, dl_dw::AbstractVector) -> Vector

Return the gradient of a downstream scalar loss with respect to `mu`.

**This one function makes the optimiser a trainable layer.**

# Arguments

  - `s`: The solved problem.
  - `dl_dw`: The gradient of the loss with respect to the optimal weights,
    length `N`. A training loop supplies it.

# Returns

  - The gradient with respect to `mu`, length `N`.

# Mathematical definition

By the chain rule and the symmetry of `K`,

    dL / d mu  =  (d w / d mu)' (dL / d w)  =  K' (dL / dw) / gamma
               =  K (dL / dw) / gamma

# Notes

  - The cost is one matrix-vector product, because `K` was formed when the
    problem was solved. **No solver is re-run and no iterations are unrolled.**
    That is the whole point of the implicit function theorem here.
  - This is the vector-Jacobian product that a reverse-mode framework needs.
    Wiring it to `ChainRulesCore.rrule` makes a mean-variance optimiser
    differentiable inside any Julia automatic-differentiation system. The
    library needs no dependency to expose it, because the rule can live in a
    package extension.
  - The economic argument for it is Donti, Amos and Kolter (2017): a return
    model trained to minimise forecast error is not trained to produce a good
    portfolio, and the two objectives disagree whenever the covariance is
    ill-conditioned, which is always.
"""
function pullback_mu(s::QPSolution, dl_dw::AbstractVector{<:Real})
    if length(dl_dw) != length(s.w)
        throw(DimensionMismatch("dl_dw has length $(length(dl_dw)), w has length $(length(s.w))"))
    end
    return (s.K * dl_dw) ./ s.gamma
end

"""
    sensitivity_report(s::QPSolution; asset_names = nothing) -> NamedTuple

Bundle the readable diagnostics of a solved problem.

# Arguments

  - `s`: The solved problem.
  - `asset_names`: Optional labels, length `N`.

# Returns

A `NamedTuple`:

  - `w`: The weights.
  - `constraint_prices`: `s.nu`, the marginal value of each constraint.
  - `mu_amplification`: `opnorm(dw_dmu(s))`. **The single most useful number
    here.** It is the factor by which an error in the expected returns is
    magnified into an error in the weights, measured in the 2-norm. A value
    near one is benign. A value in the hundreds means the portfolio is an
    error amplifier and needs shrinkage, a norm penalty, or a Wasserstein
    radius. See prototype 2.
  - `most_sensitive_asset`: The asset whose weight responds most to its own
    expected return, and the size of that response.
  - `condition_number`: `cond(sigma)` recovered from `K`'s spectrum, as a
    cross-check on the amplification figure.
"""
function sensitivity_report(s::QPSolution;
                            asset_names::Union{Nothing, AbstractVector} = nothing)
    J = dw_dmu(s)
    diagJ = diag(J)
    i = argmax(abs.(diagJ))
    name = isnothing(asset_names) ? "asset $(i)" : String(asset_names[i])
    lam = eigvals(Symmetric(s.K))
    pos = filter(>(sqrt(eps(eltype(lam)))), lam)
    return (; w = s.w, constraint_prices = s.nu, mu_amplification = opnorm(J),
            most_sensitive_asset = (name, diagJ[i]),
            condition_number = if isempty(pos)
                one(eltype(lam))
            else
                maximum(pos) / minimum(pos)
            end)
end

end # module OptimiserSensitivity
