# =============================================================================
# Prototype 20 — Graph structure in the objective: Laplacian regularisation.
#
# Purpose
#   Report 3 asks for Laplacian regularisation and neighbourhood constraints.
#   The library has a rich phylogeny subsystem that turns a correlation matrix
#   into a graph, and it uses that graph to build **constraints**. It does not
#   use it in the **objective**.
#
#   The distinction matters because a constraint is a wall and a penalty is a
#   slope. A phylogeny constraint says "these two may not both be held". A
#   Laplacian penalty says "these two are similar, so give them similar
#   weights unless the data argues otherwise". The second is what a caller
#   usually means, and it is a one-line addition to a quadratic objective.
#
#   The whole idea rests on one identity, which the driver verifies:
#
#       w' L w  =  (1/2) sum_{i,j}  A_ij ( w_i - w_j )^2
#
#   The left side is a quadratic form that a solver handles trivially. The
#   right side is the plain English statement "penalise weight differences
#   between connected assets, in proportion to how strongly they are
#   connected".
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics`.
#
# Notation used throughout this file
#   N       Number of assets.
#   A       Adjacency matrix, `N x N`, symmetric, non-negative, zero diagonal.
#   D       Degree matrix, `Diagonal(vec(sum(A; dims = 2)))`.
#   L       Combinatorial Laplacian, `D - A`.
#   L_sym   Symmetric normalised Laplacian, `I - D^{-1/2} A D^{-1/2}`.
#   w       Portfolio weights, length `N`.
#   lam     Penalty strength.
#
# Sources
#   Chung, F. R. K. (1997). Spectral Graph Theory. American Mathematical
#     Society. The Laplacian and its spectrum.
#   von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and
#     Computing 17(4), 395-416. The identity above, and the connected-component
#     result.
#   Mantegna, R. N. (1999). Hierarchical structure in financial markets.
#     European Physical Journal B 11(1), 193-197. The minimum spanning tree of
#     a correlation matrix, which the library already builds.
#   Tumminello, M., Aste, T., Di Matteo, T. and Mantegna, R. N. (2005). A tool
#     for filtering information in complex systems. Proceedings of the National
#     Academy of Sciences 102(30), 10421-10426. The planar maximally filtered
#     graph, also already in the library.
#   Ando, R. K. and Zhang, T. (2007). Learning on graph with Laplacian
#     regularization. Advances in Neural Information Processing Systems 19,
#     25-32. The penalty in its machine-learning form.
# =============================================================================
module GraphStructure

using LinearAlgebra, Statistics

export graph_laplacian, normalised_laplacian, laplacian_penalty, laplacian_penalty_pairwise,
       laplacian_smoothed_weights, connected_components_count, exclusion_pairs,
       spectral_profile

"""
    graph_laplacian(A::AbstractMatrix) -> Matrix

Return the combinatorial Laplacian `L = D - A`.

# Arguments

  - `A`: Adjacency matrix, `N x N`, symmetric, non-negative, with a zero
    diagonal.

# Returns

  - `L`, symmetric positive semi-definite.

# Properties

  - `L` is positive semi-definite, whatever `A` is, provided `A` is
    non-negative.
  - `L * ones(N) == 0` always, because each row sums to zero by construction.
  - **The multiplicity of the zero eigenvalue equals the number of connected
    components** of the graph. That is the single most useful spectral fact
    about `L`, and [`connected_components_count`](@ref) uses it.

# Validation

  - `A` must be square, symmetric, and non-negative.
"""
function graph_laplacian(A::AbstractMatrix{<:Real})
    N = size(A, 1)
    if size(A, 2) != N
        throw(DimensionMismatch("A must be square, got $(size(A))"))
    end
    if !isapprox(A, transpose(A); atol = 1e-12)
        throw(ArgumentError("A must be symmetric"))
    end
    if any(<(0), A)
        throw(DomainError(minimum(A),
                          "A must be non-negative; a negative edge breaks positive semi-definiteness"))
    end
    Az = copy(Matrix(float.(A)))
    for i in 1:N
        Az[i, i] = zero(eltype(Az))
    end
    return Diagonal(vec(sum(Az; dims = 2))) - Az
end

"""
    normalised_laplacian(A::AbstractMatrix) -> Matrix

Return the symmetric normalised Laplacian `L_sym = I - D^{-1/2} A D^{-1/2}`.

# Details

Isolated nodes have zero degree, so their rows are set to the identity rather
than dividing by zero.

# Notes

  - **Every eigenvalue lies in `[0, 2]`**, which makes the penalty strength
    comparable across graphs of different densities. The combinatorial
    Laplacian has no such bound: a dense graph produces large eigenvalues and
    the same `lam` then means something completely different. **Use the
    normalised form whenever `lam` is tuned on one universe and applied to
    another.**
"""
function normalised_laplacian(A::AbstractMatrix{<:Real})
    graph_laplacian(A)   # validation only: square, symmetric, non-negative
    Adiag = Matrix(float.(A)) - Diagonal(diag(float.(A)))
    deg = vec(sum(Adiag; dims = 2))
    N = length(deg)
    dinv = [deg[i] > 0 ? 1 / sqrt(deg[i]) : zero(eltype(deg)) for i in 1:N]
    Ls = Matrix{eltype(Adiag)}(I, N, N) .- (dinv .* Adiag .* transpose(dinv))
    for i in 1:N
        if deg[i] <= 0
            Ls[i, :] .= 0
            Ls[:, i] .= 0
            Ls[i, i] = one(eltype(Ls))
        end
    end
    return (Ls .+ transpose(Ls)) ./ 2
end

"""
    laplacian_penalty(w::AbstractVector, L::AbstractMatrix) -> Real

Return the quadratic form `w' L w`.
"""
function laplacian_penalty(w::AbstractVector{<:Real}, L::AbstractMatrix{<:Real})
    N = length(w)
    if size(L) != (N, N)
        throw(DimensionMismatch("L must be $(N) x $(N), got $(size(L))"))
    end
    return dot(w, L, w)
end

"""
    laplacian_penalty_pairwise(w::AbstractVector, A::AbstractMatrix) -> Real

Return `(1/2) * sum_{i,j} A_ij (w_i - w_j)^2`, computed directly from the
edges.

# Notes

  - **This must equal [`laplacian_penalty`](@ref) with `L = graph_laplacian(A)`
    exactly**, and the driver asserts it. The function exists only to make the
    identity checkable, and to give a reader the pairwise reading that the
    matrix form hides.
"""
function laplacian_penalty_pairwise(w::AbstractVector{<:Real}, A::AbstractMatrix{<:Real})
    N = length(w)
    s = zero(float(eltype(w)))
    @inbounds for i in 1:N, j in 1:N
        i == j && continue
        s += A[i, j] * (w[i] - w[j])^2
    end
    return s / 2
end

"""
    connected_components_count(A::AbstractMatrix; tol::Real = 1e-9) -> Int

Return the number of connected components, from the Laplacian spectrum.

# Mathematical definition

The multiplicity of the eigenvalue zero of `L` equals the number of connected
components. Count the eigenvalues below `tol`.

# Notes

  - **This is a diversification diagnostic, not a graph utility.** A
    correlation graph filtered at a threshold that leaves one giant component
    says the universe has no separable blocks. One that shatters into thirty
    components says the threshold is too high and the graph carries no
    information. The count is the fastest way to see which regime a filter is
    in.
"""
function connected_components_count(A::AbstractMatrix{<:Real}; tol::Real = 1e-9)
    lam = eigvals(Symmetric(graph_laplacian(A)))
    return count(<(tol), lam)
end

"""
    laplacian_smoothed_weights(mu::AbstractVector, sigma::AbstractMatrix,
                               L::AbstractMatrix; lam::Real = 0.0,
                               gamma::Real = 1.0, budget::Real = 1.0) -> Vector

Solve the budget-constrained mean-variance problem with a Laplacian penalty,
in closed form.

# Arguments

  - `mu`: Expected returns, length `N`.
  - `sigma`: Covariance, `N x N`.
  - `L`: Laplacian, `N x N`.
  - `lam`: Penalty strength. Zero recovers the unpenalised solution.
  - `gamma`: Risk aversion.
  - `budget`: Total weight.

# Returns

  - The optimal weights, length `N`.

# Mathematical definition

    minimise  (1/2) w' ( sigma + 2 lam L ) w  -  (1/gamma) mu' w
    subject to  1' w = budget

The penalty adds `2 lam L` to the covariance, so the whole problem is the
ordinary one with an **effective covariance**

    sigma_eff  =  sigma  +  2 lam L

and the solution comes from the same KKT system as prototype 7.

# Notes

  - **The Laplacian penalty is a covariance modification, not a new kind of
    objective.** That is the useful realisation: the library needs no new
    optimiser, only a way to add a matrix to the covariance before it is used.
    `sigma + 2 lam L` is still symmetric positive semi-definite, because both
    terms are.
  - As `lam -> infinity` the solution tends to the **equal-weight portfolio
    within each connected component**, because the penalty drives connected
    weights together and only the null space of `L` survives. The driver
    verifies that limit, and it is the cleanest way to see what the penalty
    does.
"""
function laplacian_smoothed_weights(mu::AbstractVector{<:Real},
                                    sigma::AbstractMatrix{<:Real},
                                    L::AbstractMatrix{<:Real}; lam::Real = 0.0,
                                    gamma::Real = 1.0, budget::Real = 1.0)
    N = length(mu)
    if size(sigma) != (N, N) || size(L) != (N, N)
        throw(DimensionMismatch("sigma and L must both be $(N) x $(N)"))
    end
    if lam < 0
        throw(DomainError(lam, "lam must be >= 0"))
    end
    Seff = Symmetric(Matrix(sigma) .+ 2 * lam .* Matrix(L))
    a = Seff \ collect(float.(mu))
    b = Seff \ ones(float(eltype(mu)), N)
    # Enforce 1'w = budget by the standard two-fund decomposition.
    nu = (dot(ones(N), a) / gamma - budget) / dot(ones(N), b)
    return a ./ gamma .- nu .* b
end

"""
    exclusion_pairs(A::AbstractMatrix, threshold::Real) -> Vector{Tuple{Int, Int}}

Return the edges whose weight exceeds `threshold`, as index pairs.

# Returns

  - Pairs `(i, j)` with `i < j` and `A[i, j] > threshold`.

# Notes

  - These are the pairs for a mutual-exclusion constraint `z_i + z_j <= 1` in a
    mixed-integer model. **The library already has the indicator machinery** in
    `src/20_Optimisation/09_JuMPConstraints/01_MIPIndicators.jl` and the
    integer phylogeny constraints, so this returns the input those need rather
    than the constraint itself.
  - **Prefer the penalty to the exclusion** unless the exclusion is a real
    mandate. An exclusion turns a convex problem into a mixed-integer one and
    buys a binary answer to a continuous question.
"""
function exclusion_pairs(A::AbstractMatrix{<:Real}, threshold::Real)
    N = size(A, 1)
    out = Tuple{Int, Int}[]
    for i in 1:N, j in (i + 1):N
        A[i, j] > threshold && push!(out, (i, j))
    end
    return out
end

"""
    spectral_profile(w::AbstractVector, L::AbstractMatrix) -> NamedTuple

Express a portfolio in the Laplacian eigenbasis.

# Returns

A `NamedTuple` with `eigenvalues`, `coefficients` (the portfolio in the
eigenbasis), `smooth_fraction` and `penalty`.

# Details

Write `w = sum_k c_k v_k` in the orthonormal eigenbasis of `L`. Then

    w' L w  =  sum_k  lambda_k  c_k^2

so the penalty is a **weighted energy**, with low-eigenvalue directions free
and high-eigenvalue directions expensive. `smooth_fraction` is the share of
`sum_k c_k^2` sitting in the lower half of the spectrum.

# Notes

  - A portfolio with a high smooth fraction respects the graph's structure. One
    with a low fraction is betting against it, deliberately or not. **Nothing
    in the library can currently say which**, and the number costs one
    eigen-decomposition.
"""
function spectral_profile(w::AbstractVector{<:Real}, L::AbstractMatrix{<:Real})
    lam, V = eigen(Symmetric(Matrix(L)))
    c = transpose(V) * collect(float.(w))
    total = sum(abs2, c)
    half = div(length(lam), 2)
    return (; eigenvalues = lam, coefficients = c,
            smooth_fraction = iszero(total) ? 1.0 : sum(abs2, view(c, 1:half)) / total,
            penalty = sum(lam .* abs2.(c)))
end

end # module GraphStructure
