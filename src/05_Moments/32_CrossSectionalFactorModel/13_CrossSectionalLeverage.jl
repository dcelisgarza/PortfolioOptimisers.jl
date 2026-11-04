"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the mask of the eligible assets of one observation whose leverage in the weighted design is one.

The leverage of row ``i`` is the diagonal entry ``h_{i}`` of the projection onto the column span of the design. It is one exactly when the design gives the row a direction of its own, for example when the row is the only member of a level of a one-hot family. A least-squares fit then reproduces the target of that row whatever it is, so its residual is zero by construction and its variance under the design is zero too. The mask does not depend on the scale of the target, and it does not depend on a positive weight either, because a positive weight moves no row into or out of the span of the others.

# Mathematical definition

```math
\\begin{align}
\\mathbf{D} \\mathbf{\\Pi} &= \\mathbf{Q} \\mathbf{R}\\,, &
\\mathbf{Q}_{1} &= \\mathbf{D} \\mathbf{\\Pi}_{1} \\mathbf{R}_{11}^{-1}\\,, \\\\
h_{i} &= \\sum_{j = 1}^{r} \\left(Q_{1}\\right)_{ij}^{2} + \\mathbb{1}_{\\text{intercept}} \\frac{w_{i}}{\\sum_{k} w_{k}}\\,, &
\\text{mark}_{i} &= 1 - h_{i} \\le \\sqrt{\\epsilon}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{D}``: Weighted design `sqrt.(w) .* A`.
  - ``\\mathbf{\\Pi}``, ``\\mathbf{Q}``, ``\\mathbf{R}``: Column permutation, orthonormal factor and triangular factor of the column-pivoted `QR` of ``\\mathbf{D}``.
  - ``r``: Numerical rank of the design, read as [`cross_sectional_rank`](@ref) reads it.
  - ``\\mathbf{\\Pi}_{1}``, ``\\mathbf{R}_{11}``, ``\\mathbf{Q}_{1}``: The first `r` columns of ``\\mathbf{\\Pi}``, the leading `r × r` block of ``\\mathbf{R}``, and the first `r` columns of ``\\mathbf{Q}``.
  - ``h_{i}``: Leverage of row ``i``.
  - ``w_{i}``: Weight of row ``i``.
  - ``\\epsilon``: Machine epsilon of the element type of ``\\mathbf{R}``.

The intercept puts the column `sqrt.(w)` in front of the design. Under `intercept`, `A` is centred on its weighted means, so every column of ``\\mathbf{D}`` is orthogonal to `sqrt.(w)`, and the two spans add. The projection onto `sqrt.(w)` gives each row ``w_{i} / \\sum_{k} w_{k}``, and the projection onto the span of ``\\mathbf{D}`` gives the rest.

The tolerance ``\\sqrt{\\epsilon}`` absorbs the round-off of ``h_{i}``, which is of order ``m \\epsilon`` for ``m`` rows. A row whose leverage is below one by more than that keeps a residual variance that the data can estimate.

The triangular solve for ``\\mathbf{Q}_{1}`` costs about a quarter of the flops of forming ``\\mathbf{Q}_{1}`` from the reflectors of the factorisation, and it loses about ``\\kappa(\\mathbf{R}_{11}) \\epsilon`` of ``h_{i}``. The ratio ``|R_{11} / R_{rr}|`` estimates that condition number. When the ratio is above ``\\epsilon^{-1/4}``, the loss can come near the tolerance of the mark, so the verb forms ``\\mathbf{Q}_{1}`` from the reflectors instead.

# Algorithm

 1. Take the column-pivoted `QR` of the weighted design `D`, unless the caller passes it, through [`cross_sectional_qr`](@ref). Read the numerical rank `r` of `D` off its triangular factor through [`cross_sectional_rank`](@ref), which answers zero for a design with no row or no column.
 2. At rank zero, take a leverage of zero on every row. Otherwise take ``\\mathbf{Q}_{1}`` by the triangular solve, or from the reflectors when the ratio of the diagonal of ``\\mathbf{R}_{11}`` is above ``\\epsilon^{-1/4}``, and sum the squares of each row, giving the leverage `h`.
 3. When `intercept` is `true`, add `w ./ sum(w)` to `h`.
 4. Mark each row whose `1 - h` is at most `sqrt(eps)`.

# Arguments

  - `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, already demeaned when an intercept is fitted.
  - `w::VecNum`: Cross-sectional weights of the eligible assets.
  - `intercept::Bool`: Whether the fit has an intercept.
  - `D::MatNum`: Weighted design `sqrt.(w) .* A`. A caller that weighted the design already passes it.
  - `F`: Column-pivoted `QR` of `D`, or `nothing` to take it here. The regression passes the factorisation that its solve reads, so the design is factorised once.
  - `G`: A buffer with the rows and at least the columns of `D` that receives the pivoted columns of the triangular solve, or `nothing` to gather them into a new matrix. The regression passes a buffer of its task.

# Returns

  - `mark::BitVector`: Leverage-one mask, one entry per eligible asset.

# Related

  - [`cross_sectional_regression`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function cross_sectional_leverage_one(A::MatNum, w::VecNum, intercept::Bool;
                                      D::MatNum = A .* sqrt.(w), F = nothing,
                                      G = nothing)::BitVector
    F = cross_sectional_qr(D, F)
    r = cross_sectional_rank(D; R = F.R)
    h = cross_sectional_leverage(D, F, r; G = G)
    if intercept && !isempty(w)
        h = h .+ w ./ sum(w)
    end
    return BitVector(one.(h) .- h .<= sqrt(eps(real(eltype(h)))))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the leverage of each row of a weighted design in the span of its columns, read off the column-pivoted `QR` of the design.

[`cross_sectional_leverage_one`](@ref) states the definition, the triangular solve and the guard on the condition number.

# Arguments

  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `F`: Column-pivoted `QR` of `D`.
  - `r::Integer`: Numerical rank of `D`, read by [`cross_sectional_rank`](@ref).
  - `G`: A buffer for the pivoted columns, or `nothing`, as [`cross_sectional_leverage_one`](@ref) states.

# Returns

  - `h::VecNum`: Leverage of each row, zero on every row at rank zero.

# Related

  - [`cross_sectional_leverage_one`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`cross_sectional_pivot_columns`](@ref)
"""
function cross_sectional_leverage(D::MatNum, F, r::Integer; G = nothing)::VecNum
    T = real(eltype(F.R))
    if iszero(r)
        return zeros(T, size(D, 1))
    end
    R = F.R
    Q1 = if abs(R[1, 1]) <= abs(R[r, r]) / sqrt(sqrt(eps(T)))
        # The gather is a copy of its own, so the division runs in place on it.
        LinearAlgebra.rdiv!(cross_sectional_pivot_columns(G, D, view(F.p, 1:r)),
                            LinearAlgebra.UpperTriangular(view(R, 1:r, 1:r)))
    else
        F.Q * Matrix{eltype(R)}(LinearAlgebra.I, size(D, 1), r)
    end
    return vec(sum(abs2, Q1; dims = 2))
end
"""
    cross_sectional_pivot_columns(G::Nothing, D::MatNum, p::AbstractVector{<:Integer}) -> MatNum
    cross_sectional_pivot_columns(G::MatNum, D::MatNum, p::AbstractVector{<:Integer}) -> MatNum

Return the columns `p` of the weighted design `D` as a matrix of its own, which [`cross_sectional_leverage`](@ref) divides in place.

# Algorithm

The method that Julia selects is the algorithm.

 1. No buffer: gather the columns into a new matrix, `D[:, p]`.
 2. A buffer: copy the columns into its first `length(p)` columns, and return that view. A loop over observations then gathers into one allocation.

# Arguments

  - `G`: A buffer with the rows of `D` and at least `length(p)` columns, or `nothing`.
  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `p::AbstractVector{<:Integer}`: The pivoted columns of the leading block of the `QR` of `D`.

# Returns

  - `Dp::MatNum`: The columns `p` of `D`, `eligible assets × length(p)`.

# Related

  - [`cross_sectional_leverage`](@ref)
  - [`cross_sectional_loop_buffers`](@ref)
"""
function cross_sectional_pivot_columns(::Nothing, D::MatNum,
                                       p::AbstractVector{<:Integer})::MatNum
    return D[:, p]
end
function cross_sectional_pivot_columns(G::MatNum, D::MatNum,
                                       p::AbstractVector{<:Integer})::MatNum
    Dp = view(G, :, 1:length(p))
    Dp .= view(D, :, p)
    return Dp
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the leverage-one mask that the fits of the observations write, or `nothing` when the estimator runs no leverage check.

A `BitMatrix` packs 64 entries into one word, so two threads that write rows of one word would race. So the mask is a `Matrix{Bool}`, whose entries are bytes, and the regression packs it after the loop through [`cross_sectional_bit_mask`](@ref).

# Arguments

  - `cre`: Cross-sectional regression estimator, whose `leverage` field decides.
  - `X::MatNum`: Asset returns matrix `observations × assets`, which gives the size of the mask.

# Returns

  - `h1::Option{<:Matrix{Bool}}`: All `false`, `observations × assets`, or `nothing`.

# Related

  - [`cross_sectional_mark_leverage!`](@ref)
  - [`cross_sectional_block_regression`](@ref)
"""
function cross_sectional_leverage_rows(cre::Union{CrossSectionalLinearRegression,
                                                  CrossSectionalTargetRegression},
                                       X::MatNum)::Option{<:Matrix{Bool}}
    return cre.leverage ? zeros(Bool, size(X)) : nothing
end
"""
    cross_sectional_mark_leverage!(h1::Nothing, A::MatNum, w::VecNum, intercept::Bool,
                                   t::Integer, idx::AbstractVector; D, F) -> Nothing
    cross_sectional_mark_leverage!(h1::AbstractMatrix{Bool}, A::MatNum, w::VecNum,
                                   intercept::Bool, t::Integer, idx::AbstractVector;
                                   D::MatNum, F) -> Nothing

Mark the eligible assets of leverage one of observation `t` in its row of `h1`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the fit runs no leverage check, so mark nothing.
 2. A mask: write [`cross_sectional_leverage_one`](@ref) of the observation, on the factorisation `F` of the weighted design `D`, into the entries `idx` of the row `t`.

# Arguments

  - `h1`: Leverage-one mask `observations × assets`, written in place, or `nothing`.
  - `A::MatNum`: Exposures of the eligible assets, `eligible assets × factors`, already demeaned when an intercept is fitted.
  - `w::VecNum`: Cross-sectional weights of the eligible assets.
  - `intercept::Bool`: Whether the fit has an intercept.
  - `t::Integer`: Row of the observation in `h1`.
  - `idx::AbstractVector`: Columns of the eligible assets in `h1`.
  - `D::MatNum`: Weighted design `sqrt.(w) .* A`.
  - `F`: Column-pivoted `QR` of `D`, as [`cross_sectional_shared_qr`](@ref) returns it.
  - `G`: A buffer for the pivoted columns, or `nothing`, as [`cross_sectional_leverage_one`](@ref) states.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_leverage_one`](@ref)
  - [`cross_sectional_leverage_rows`](@ref)
  - [`cross_sectional_block_regression`](@ref)
"""
function cross_sectional_mark_leverage!(::Nothing, ::MatNum, ::VecNum, ::Bool, ::Integer,
                                        ::AbstractVector; D = nothing, F = nothing,
                                        G = nothing)::Nothing
    return nothing
end
function cross_sectional_mark_leverage!(h1::AbstractMatrix{Bool}, A::MatNum, w::VecNum,
                                        intercept::Bool, t::Integer, idx::AbstractVector;
                                        D::MatNum, F, G = nothing)::Nothing
    h1[t, idx] = cross_sectional_leverage_one(A, w, intercept; D = D, F = F, G = G)
    return nothing
end
"""
    cross_sectional_zero_marked!(eps::MatNum, h1::Nothing, ls::Bool) -> Nothing
    cross_sectional_zero_marked!(eps::MatNum, h1::AbstractMatrix{Bool}, ls::Bool) -> Nothing

Write an exact zero into the residual of every pair that `h1` marks, when the fit is a least-squares fit.

The fit reproduced the return of a marked pair, so zero is its residual in exact arithmetic, and the subtraction leaves only its round-off. A fit that is not least squares keeps the residual of a marked pair, and only marks it.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: the fit marks no pair, so change nothing.
 2. A mask: when `ls` is `true`, write zero at every marked pair.

# Arguments

  - `eps::MatNum`: Residual matrix `observations × assets`, changed in place.
  - `h1`: Leverage-one mask of the fit, or `nothing`.
  - `ls::Bool`: [`cross_sectional_least_squares`](@ref) of the estimator.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_least_squares`](@ref)
  - [`cross_sectional_block_regression`](@ref)
"""
function cross_sectional_zero_marked!(::MatNum, ::Nothing, ::Bool)::Nothing
    return nothing
end
function cross_sectional_zero_marked!(eps::MatNum, h1::AbstractMatrix{Bool},
                                      ls::Bool)::Nothing
    eps[h1 .& ls] .= zero(eltype(eps))
    return nothing
end
"""
    leverage_one_nan(h1::Nothing, A::MatNum) -> MatNum
    leverage_one_nan(h1::AbstractMatrix{Bool}, A::MatNum) -> MatNum

Return a history of a cross-sectional fit with `NaN` at every pair whose leverage is one.

A pair that `h1` marks gives no observation of its idiosyncratic return. The fit reproduced its return, so its residual is zero by construction, and so is the variance that reads that residual. A reader that treats such a pair as data reads the design, not the asset. The function writes `NaN` over the pair, so every reader that skips a value that is not finite leaves the pair out. It never changes `A`.

# Algorithm

The method that Julia selects is the algorithm.

 1. `nothing`: a regression built by hand states no mask, so return `A`.
 2. A mask: check its size, and return `A` when it marks no pair. Otherwise copy `A` into the type of a square root of its elements, which holds a `NaN` and the division the readers take, and write `NaN` at every marked pair.

# Arguments

  - `h1`: The leverage-one mask of [`CrossSectionalRegression`](@ref), `observations × assets`, or `nothing`.
  - `A`: A history on the axes of `h1`, `observations × assets`.

# Validation

  - `size(h1) == size(A)`. Raises a `DimensionMismatch`.

# Returns

  - `A::MatNum`: `A` itself when no pair is marked, otherwise a copy with `NaN` at every marked pair.

# Examples

```jldoctest
julia> PortfolioOptimisers.leverage_one_nan(BitMatrix([0 1; 0 0]), [0.1 0.0; 0.3 0.4])
2×2 Matrix{Float64}:
 0.1  NaN
 0.3    0.4
```

# Related

  - [`CrossSectionalRegression`](@ref)
  - [`cross_sectional_leverage_one`](@ref)
"""
function leverage_one_nan(::Nothing, A::MatNum)::MatNum
    return A
end
function leverage_one_nan(h1::AbstractMatrix{Bool}, A::MatNum)::MatNum
    @argcheck(size(h1) == size(A),
              DimensionMismatch("h1 ($(size(h1, 1))×$(size(h1, 2))) must match the history ($(size(A, 1))×$(size(A, 2)))"))
    if !any(h1)
        return A
    end
    B = typeof(sqrt(one(real(eltype(A))))).(A)
    B[h1] .= NaN
    return B
end
"""
    cross_sectional_least_squares(cre::CrossSectionalLinearRegression) -> Bool
    cross_sectional_least_squares(cre::CrossSectionalTargetRegression) -> Bool
    cross_sectional_least_squares(tgt::LinearModel) -> Bool
    cross_sectional_least_squares(tgt::GeneralisedLinearModel) -> Bool
    cross_sectional_least_squares(tgt::AbstractRegressionTarget) -> Bool

Return whether the fit of a cross-sectional regression reproduces the return of a pair whose leverage is one.

[`cross_sectional_regression`](@ref) writes an exact zero residual at such a pair when this answers `true`. A least-squares fit projects the returns onto the span of the design, so the pair's residual is zero in exact arithmetic, and the subtraction leaves only its round-off. A generalised linear model with an identity link solves its score equations with the pair's own parameter, which also fits the pair exactly. Under any other link the fit reproduces the mean of the pair on the scale of the link, so the residual on the scale of the returns is not zero, and the verb answers `false`. So does a target the verb does not know.

# Arguments

  - `cre`: Cross-sectional regression estimator.
  - `tgt`: Regression target of a [`CrossSectionalTargetRegression`](@ref).

# Returns

  - `ls::Bool`: `true` when the residual of a leverage-one pair is zero by construction.

# Related

  - [`cross_sectional_leverage_one`](@ref)
  - [`cross_sectional_regression`](@ref)
  - [`LinearModel`](@ref)
  - [`GeneralisedLinearModel`](@ref)
"""
function cross_sectional_least_squares(::CrossSectionalLinearRegression)::Bool
    return true
end
function cross_sectional_least_squares(cre::CrossSectionalTargetRegression)::Bool
    return cross_sectional_least_squares(cre.tgt)
end
function cross_sectional_least_squares(::LinearModel)::Bool
    return true
end
function cross_sectional_least_squares(tgt::GeneralisedLinearModel)::Bool
    link = length(tgt.args) >= 2 ? tgt.args[2] : GLM.canonicallink(tgt.args[1])
    return isa(link, GLM.IdentityLink)
end
function cross_sectional_least_squares(::AbstractRegressionTarget)::Bool
    return false
end
"""
    cross_sectional_shared_qr(h1::Nothing, dc::Nothing,
                              alg::AbstractCrossSectionalSolveAlgorithm, D::MatNum,
                              Q::MatNum) -> Nothing
    cross_sectional_shared_qr(h1::Nothing, dc::Nothing,
                              alg::Union{UncheckedSolve, MinimumNormSolve,
                                         PseudoInverseFallback, RankDeficiencyRefusal,
                                         DependentColumnDrop}, D::MatNum,
                              Q::MatNum) -> LinearAlgebra.QRPivoted
    cross_sectional_shared_qr(h1, dc, alg::AbstractCrossSectionalSolveAlgorithm, D::MatNum,
                              Q::MatNum) -> LinearAlgebra.QRPivoted

Return the column-pivoted `QR` that every reader of one observation shares, or `nothing` when no reader of the regression needs it.

The leverage check writes `h1`, and the mark of the dropped columns of [`DependentColumnDrop`](@ref) writes `dc`. Both read a `QR` of the weighted design, so the regression takes it once and hands it to the solve as well. With neither mask, the regression takes it for a member of the library whose solve reads a `QR`, so that it factorises into the buffer of its task. [`NormalEquationsSolve`](@ref) reads none, and a member of a caller drops it, so both receive `nothing`.

# Algorithm

The method that Julia selects is the algorithm.

 1. Both masks are `nothing`, and the member reads no `QR`: return `nothing`.
 2. Otherwise: return [`cross_sectional_qr`](@ref) of `D` into the buffer `Q`.

# Arguments

  - `h1`: The leverage-one mask of the fit, or `nothing` when the fit runs no leverage check.
  - `dc`: The dropped-column mask of the fit, or `nothing`.
  - `alg`: The solve algorithm of the estimator.
  - `D::MatNum`: Weighted design of one observation, `eligible assets × factors`.
  - `Q::MatNum`: A buffer of the size and the element type of `D`, which receives the factors. The factorisation reads it until the next observation of the task writes it.

# Returns

  - `F::Option{<:LinearAlgebra.QRPivoted}`: The factorisation, or `nothing`.

# Related

  - [`cross_sectional_qr`](@ref)
  - [`cross_sectional_block_regression`](@ref)
  - [`cross_sectional_loop_buffers`](@ref)
"""
function cross_sectional_shared_qr(::Nothing, ::Nothing,
                                   ::AbstractCrossSectionalSolveAlgorithm, ::MatNum,
                                   ::MatNum)::Nothing
    return nothing
end
function cross_sectional_shared_qr(::Nothing, ::Nothing,
                                   ::Union{UncheckedSolve, MinimumNormSolve,
                                           PseudoInverseFallback, RankDeficiencyRefusal,
                                           DependentColumnDrop}, D::MatNum, Q::MatNum)
    return cross_sectional_qr(D, nothing, Q)
end
function cross_sectional_shared_qr(::Any, ::Any, ::AbstractCrossSectionalSolveAlgorithm,
                                   D::MatNum, Q::MatNum)
    return cross_sectional_qr(D, nothing, Q)
end
