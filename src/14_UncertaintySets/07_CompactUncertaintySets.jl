"""
$(DocStringExtensions.TYPEDEF)

Holds a worst-case variance penalty as a radius, a diagonal metric square root and a basis of the directions the penalty spares.

The set inflates the variance on every direction outside the span of its basis and on none inside it, so a portfolio built out of that span pays nothing. The consumer adds one quadratic term and `rank` free variables instead of the dense ``N \\times N`` matrix the same worst case would otherwise need, which is why the type is called compact: no lifted semidefinite block appears, and the programme stays a second-order cone programme. The basis is orthonormal over all ``N`` rows, so a row slice of it is not orthonormal, and [`port_opt_view`](@ref) re-orthonormalises the slice rather than slicing the projector.

# Mathematical definition

```math
\\begin{align}
\\underset{\\mathbf{\\Sigma} \\in U^{\\text{cpt}}_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^{\\intercal} \\mathbf{\\Sigma} \\boldsymbol{w} &= \\boldsymbol{w}^{\\intercal} \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa \\underset{\\boldsymbol{z}}{\\min} \\lVert \\mathbf{C} \\boldsymbol{w} - \\mathbf{Q} \\boldsymbol{z} \\rVert_{2}^{2} \\\\
U^{\\text{cpt}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma} \\succeq 0 \\, \\vert \\, \\mathbf{\\Sigma} \\preceq \\hat{\\mathbf{\\Sigma}} + \\kappa \\mathbf{C}^{\\intercal} (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}) \\mathbf{C} \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{cpt}}_{\\mathbf{\\Sigma}}``: Compact uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}``: Uncertain covariance.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:w_port])
  - ``\\kappa \\geq 0``: Radius, the multiplier of the penalty.
  - ``\\mathbf{C} = \\operatorname{diag}(\\boldsymbol{c})``: Diagonal metric square root, ``N \\times N``.
  - ``\\mathbf{Q}``: Basis of the spared subspace, ``N \\times r``, with orthonormal columns.
  - ``\\boldsymbol{z}``: Coefficient vector of the inner problem, ``r \\times 1``.

The two lines are one object because the inner problem is a least-squares problem whose value is ``\\lVert (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``, and ``\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}`` is symmetric and idempotent. The variational form on the right of the first line is the weaker of the two: it projects onto ``\\operatorname{col}(\\mathbf{Q})`` for **any** ``\\mathbf{Q}``, whereas the closed form of the second line needs orthonormal columns. A weight vector with ``\\mathbf{C}\\boldsymbol{w} \\in \\operatorname{col}(\\mathbf{Q})`` pays a zero penalty, and ``r = 0`` leaves ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CompactCovarianceUncertaintySet(;
        kappa::Number,
        C::VecNum,
        Q::MatNum,
        val::Option{<:MatNum} = nothing
    ) -> CompactCovarianceUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(kappa)` and `kappa >= 0`.
  - `!isempty(C)`, `all(isfinite, C)` and `all(x -> x >= 0, C)`.
  - `all(isfinite, Q)`.
  - `size(Q, 1) == length(C)`.
  - If `val` is provided: `size(val, 1) == size(val, 2) == length(C)`.

# Examples

```jldoctest
julia> CompactCovarianceUncertaintySet(; kappa = 2.0, C = [1.0, 1.0],
                                       Q = reshape([1.0, 0.0], 2, 1))
CompactCovarianceUncertaintySet
  kappa ┼ Float64: 2.0
      C ┼ Vector{Float64}: [1.0, 1.0]
      Q ┼ 2×1 Matrix{Float64}
    val ┴ nothing
```

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
  - [`ucs_variance`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct CompactCovarianceUncertaintySet <: AbstractUncertaintySetResult
    """
    Radius ``\\kappa \\geq 0``, the multiplier of the quadratic penalty. It is a user-set size rather than a quantile, and `0` disables the penalty and leaves the nominal variance.
    """
    kappa
    """
    Diagonal of the metric square root ``\\mathbf{C}``, of length ``N``, held as a vector rather than as a matrix.
    """
    C
    """
    Basis ``\\mathbf{Q}`` of the subspace the penalty spares, ``N \\times r``, with orthonormal columns. A rank of `0` is admitted and leaves the penalty ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.
    """
    Q
    """
    $(field_dict[:val_ucs])
    """
    val
    function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum, Q::MatNum,
                                             val::Option{<:MatNum})
        @argcheck(isfinite(kappa) && kappa >= zero(kappa),
                  DomainError(kappa, "kappa must be finite and >= 0"))
        @argcheck(!isempty(C), IsEmptyError("C cannot be empty"))
        @argcheck(all(isfinite, C), IsNonFiniteError("all entries of C must be finite"))
        @argcheck(all(x -> x >= zero(x), C),
                  DomainError(C, "all entries of C must be >= 0"))
        @argcheck(all(isfinite, Q), IsNonFiniteError("all entries of Q must be finite"))
        @argcheck(size(Q, 1) == length(C),
                  DimensionMismatch("Q ($(size(Q, 1)) rows) must match C ($(length(C)))"))
        if isa(val, MatNum)
            assert_matrix_issquare(val, :val)
            @argcheck(size(val, 1) == length(C),
                      DimensionMismatch("val ($(size(val, 1))) must match C ($(length(C)))"))
        end
        return new{typeof(kappa), typeof(C), typeof(Q), typeof(val)}(kappa, C, Q, val)
    end
end
function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum,
                                         Q::MatNum)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(kappa, C, Q, nothing)
end
function CompactCovarianceUncertaintySet(; kappa::Number, C::VecNum, Q::MatNum,
                                         val::Option{<:MatNum} = nothing)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(kappa, C, Q, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return an orthonormal basis of the column space of `Q`, dropping the columns a rank-revealing factorisation finds numerically dependent.

# Algorithm

 1. Return `Q` unchanged when it has no column, because a zero-rank basis is already orthonormal and a factorisation of it has no pivot to read.
 2. Take the column-pivoted `LinearAlgebra.qr` of `Q`. The magnitudes of the diagonal of its `R` are non-increasing, so they rank the columns by how much each adds to the span.
 3. Count the pivots above `maximum(size(Q)) * eps(float(real(eltype(Q)))) * abs(R[1, 1])`, giving `r`, the numerical rank. The tolerance is the one `LinearAlgebra.rank` applies to a singular value.
 4. Return the first `r` columns of the orthogonal factor, materialised as a `Matrix`.

# Arguments

  - `Q::MatNum`: Basis whose columns span the subspace, not necessarily orthonormal.

# Returns

  - `Q::MatNum`: Orthonormal basis of `col(Q)`, with one column per unit of numerical rank.

# Related

  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function orthonormalise_basis(Q::MatNum)
    if size(Q, 2) == 0
        return Q
    end
    F = LinearAlgebra.qr(Q, LinearAlgebra.ColumnNorm())
    R = F.R
    tol = maximum(size(Q)) * eps(float(real(eltype(R)))) * abs(R[1, 1])
    r = count(j -> abs(R[j, j]) > tol, axes(R, 1))
    return Matrix(F.Q)[:, 1:r]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`CompactCovarianceUncertaintySet`](@ref) restricted to the asset indices `i`, re-orthonormalising the basis it slices.

The restricted penalty is the projection onto the sliced span, and **not** the slice of the projection. The two differ: ``\\mathbf{Q}^{\\intercal}\\mathbf{Q} = \\mathbf{I}`` sums over all ``N`` rows, so the rows of one cluster satisfy ``\\mathbf{Q}_{i}^{\\intercal}\\mathbf{Q}_{i} = \\mathbf{I} - \\mathbf{Q}_{-i}^{\\intercal}\\mathbf{Q}_{-i}``, which is not the identity and makes ``\\mathbf{Q}_{i}\\mathbf{Q}_{i}^{\\intercal}`` no projector at all. The subspace itself survives the slice, so re-orthonormalising the sliced rows recovers a basis of the right span.

# Algorithm

 1. Take `view(risk_ucs.C, i)`, the diagonal of the metric square root restricted to the selected assets.
 2. Take `view(risk_ucs.Q, i, :)`, the rows of the basis the selected assets occupy, and pass it through [`orthonormalise_basis`](@ref). The rank can fall, because the columns of a slice can become dependent.
 3. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the nominal covariance restricted to the same assets on both axes, which passes a `nothing` through unchanged.
 4. Build a [`CompactCovarianceUncertaintySet`](@ref) from the three, carrying `kappa` through unchanged. The radius is a size the caller set rather than a quantile of a dimension, so the smaller universe does not recalibrate it.

# Arguments

  - `risk_ucs`: Compact covariance uncertainty set.
  - `i`: Cluster or asset index.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::CompactCovarianceUncertaintySet`: The set restricted to `i`.

# Related

  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`orthonormalise_basis`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::CompactCovarianceUncertaintySet, i,
                       args...)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(; kappa = risk_ucs.kappa,
                                           C = view(risk_ucs.C, i),
                                           Q = orthonormalise_basis(view(risk_ucs.Q, i, :)),
                                           val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
    mu_ucs(uc::CompactCovarianceUncertaintySet, args...; kwargs...)

Always throw. [`CompactCovarianceUncertaintySet`](@ref) is covariance-only.

The method is a refusal rather than a procedure, so it carries no `# Algorithm` section. It shadows the passthrough method every other Result reaches, which would otherwise hand a covariance set to a consumer of the mean.

# Arguments

  - `uc`: Compact covariance uncertainty set.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - The method always throws an `ArgumentError`. The set bounds a covariance matrix through a quadratic penalty on the weights, and no mean analogue is defined for it. The message names the fix: [`BoxUncertaintySet`](@ref), [`EllipsoidalUncertaintySet`](@ref) or [`L1UncertaintySet`](@ref) for a mean set.

# Returns

  - Never returns.

# Related

  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`mu_ucs`](@ref)
  - [`sigma_ucs`](@ref)
"""
function mu_ucs(::CompactCovarianceUncertaintySet, args...; kwargs...)
    return throw(ArgumentError("CompactCovarianceUncertaintySet is covariance-only: it holds a quadratic worst-case variance penalty, and no mean analogue is defined for it. Use BoxUncertaintySet, EllipsoidalUncertaintySet or L1UncertaintySet for a mean uncertainty set."))
end

export CompactCovarianceUncertaintySet
