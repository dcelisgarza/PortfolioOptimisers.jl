"""
    detmcd_directions(Z::MatNum) -> NTuple{6, Matrix}

Returns the eigenvectors of the six robust estimates of the correlation of the standardised data that [`DetMCD`](@ref) reads.

Section 3.2 of [hubert2012](@cite) gives the six estimates, and the code of the R package `robustbase`, version 0.99-7, computes them as this method does.

 1. The correlation of `tanh.(Z)`.
 2. The Spearman correlation of `Z`, the correlation of the ranks. A tie takes the mean rank.
 3. The correlation of the normal scores of `Z`. The score of rank ``r`` of ``n`` is ``\\Phi^{-1}((r - 1/3)/(n + 1/3))``.
 4. The spatial sign covariance: the product moment of the rows of `Z` divided by their norms. A row with a norm below `eps` keeps its values.
 5. The covariance of the ``\\lceil n/2 \\rceil`` rows of `Z` with the smallest norms, the first step of BACON.
 6. The raw orthogonalised Gnanadesikan–Kettenring scatter of [`ogk_scatter`](@ref).

With fewer observations than assets, an estimate can be singular. The eigenvectors of its zero eigenvalues are then an arbitrary basis, which the linear algebra library chooses. So the method returns the eigenvectors whose eigenvalue is not zero, with [`regular_directions`](@ref).

# Arguments

  - `Z`: The standardised data, `observations × assets`.

# Returns

  - `P::NTuple{6, Matrix}`: The eigenvectors of each estimate, one per column.

# Related

  - [`mcd_starts`](@ref)
  - [`regular_directions`](@ref)
  - [`ogk_scatter`](@ref)

# References

  - $(ref_dict[:hubert2012]) Section 3.2.
"""
function detmcd_directions(Z::MatNum)
    n = size(Z, 1)
    ev = regular_directions
    rk = stack(StatsBase.tiedrank, eachcol(Z))
    third = one(eltype(Z)) / 3
    scores = Distributions.quantile.(Distributions.Normal(), (rk .- third) ./ (n + third))
    nrm = vec(sqrt.(sum(abs2, Z; dims = 2)))
    sgn = Z ./ ifelse.(nrm .> eps(eltype(nrm)), nrm, one(eltype(nrm)))
    near = nearest_positions(nrm, cld(n, 2))
    return (ev(Statistics.cor(tanh.(Z))), ev(Statistics.cor(rk)),
            ev(Statistics.cor(scores)), ev(transpose(sgn) * sgn),
            ev(Statistics.cov(view(Z, near, :))), ev(ogk_scatter(Z)))
end
"""
    regular_directions(A::MatNum) -> Matrix

Returns the eigenvectors of a symmetric matrix whose eigenvalue is larger in absolute value than `p * eps` times the largest one, where `p` is the number of rows.

The estimate of [`ogk_scatter`](@ref) can have a negative eigenvalue. Its eigenvector is a direction of the estimate, so the function keeps it.

# Arguments

  - `A`: The symmetric matrix.

# Returns

  - `P::Matrix`: The eigenvectors, one per column. The matrix is square when `A` is regular.

# Related

  - [`detmcd_directions`](@ref)
  - [`detmcd_initset`](@ref)
"""
function regular_directions(A::MatNum)
    vals, P = LinearAlgebra.eigen(LinearAlgebra.Symmetric(A))
    keep = abs.(vals) .> length(vals) * eps(eltype(vals)) * maximum(abs, vals)
    return P[:, keep]
end
"""
    ogk_scatter(Z::MatNum) -> Matrix

Returns the raw orthogonalised Gnanadesikan–Kettenring scatter that the sixth start of [`DetMCD`](@ref) reads. Section 2.2 of [hubert2012](@cite) states it.

Entry ``(i, j)`` off the diagonal is ``(Q_n(z_i + z_j)^2 - Q_n(z_i - z_j)^2)/4``, the identity of Gnanadesikan and Kettenring with the robust scale [`qn_scale`](@ref). The diagonal is one, because the columns of `Z` are standardised.

# Arguments

  - `Z`: The standardised data, `observations × assets`.

# Returns

  - `U::Matrix`: The symmetric scatter matrix.

# Related

  - [`detmcd_directions`](@ref)
  - [`qn_scale`](@ref)

# References

  - $(ref_dict[:hubert2012]) Sections 2.2 and 3.2.
"""
function ogk_scatter(Z::MatNum)
    p = size(Z, 2)
    U = Matrix{eltype(Z)}(LinearAlgebra.I, p, p)
    for i in 2:p, j in 1:(i - 1)
        a, b = view(Z, :, i), view(Z, :, j)
        U[i, j] = U[j, i] = (qn_scale(a .+ b)^2 - qn_scale(a .- b)^2) / 4
    end
    return U
end
"""
    detmcd_initset(Z::MatNum, P::MatNum, h::Integer, fixed::Bool) -> Vector{Int}

Returns the start subset of [`DetMCD`](@ref) that one set of directions `P` gives.

Section 3.1 of [hubert2012](@cite) states the steps. The eigenvalues of a robust correlation estimate can be far from the truth, so the method takes a robust scale of the data along each direction in place of them.

# Algorithm

 1. Project the data on the directions, `Y = Z * P`. Take the robust scale ``\\lambda_k`` of each column of `Y` about its median with [`robust_column_scale`](@ref).
 2. Take the centre as the coordinatewise median of the data in the sphered coordinates ``\\mathbf{P} \\mathrm{Diag}(\\lambda)^{-1} \\mathbf{P}^\\intercal``, mapped back. When `fixed` is `true`, the centre is zero.
 3. Compute the distance of each observation from the centre in the metric ``\\mathbf{P} \\mathrm{Diag}(\\lambda)^2 \\mathbf{P}^\\intercal``. Add the squared length of the part of the observation that no direction of `P` reaches, with [`outside_directions`](@ref). Return the `h` nearest observations. The first concentration step of the search is the next step of [hubert2012](@cite).

# Arguments

  - `Z`: The standardised data.
  - `P`: The directions, one per column.
  - `h`: The size of the subset.
  - `fixed`: Whether the centre is fixed at zero.

# Returns

  - `idx::Vector{Int}`: The positions of the start subset.

# Related

  - [`mcd_starts`](@ref)
  - [`detmcd_directions`](@ref)

# References

  - $(ref_dict[:hubert2012]) Section 3.1, Equation 3.
"""
function detmcd_initset(Z::MatNum, P::MatNum, h::Integer, fixed::Bool)
    Y = Z * P
    lambda = map(y -> robust_column_scale(y .- Statistics.median(y)), eachcol(Y))
    loc = if fixed
        zeros(eltype(Z), size(Z, 2))
    else
        sqrtinv = P * LinearAlgebra.Diagonal(inv.(lambda)) * transpose(P)
        P * (lambda .* (transpose(P) * vec(Statistics.median(Z * sqrtinv; dims = 1))))
    end
    d = vec(sum(abs2, ((Z .- transpose(loc)) * P) ./ transpose(lambda); dims = 2))
    return nearest_positions(d .+ outside_directions(Z, Y, P), h)
end
"""
    outside_directions(Z::MatNum, Y::MatNum, P::MatNum) -> Union{Vector, Number}

Returns the squared length of the part of each row of `Z` that is orthogonal to the columns of `P`, or zero when `P` is square.

A singular correlation estimate gives no scale for these directions, so they take the scale one of the standardised data. The result does not depend on a basis of them.

# Arguments

  - `Z`: The standardised data.
  - `Y`: The data on the directions, `Z * P`.
  - `P`: The directions, one per column.

# Returns

  - `q::Union{Vector, Number}`: The squared length for each row, or zero.

# Related

  - [`detmcd_initset`](@ref)
  - [`regular_directions`](@ref)
"""
function outside_directions(Z::MatNum, Y::MatNum, P::MatNum)
    if size(P, 1) == size(P, 2)
        return zero(eltype(Z))
    end
    return vec(sum(abs2, Z - Y * transpose(P); dims = 2))
end
