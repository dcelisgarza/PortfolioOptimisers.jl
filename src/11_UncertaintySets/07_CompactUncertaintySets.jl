"""
$(DocStringExtensions.TYPEDEF)

Holds a worst-case variance penalty as a radius, a diagonal metric square root and a basis of the directions that the penalty spares.

The set adds variance on every direction outside the span of its basis, and it adds none inside the span. So a portfolio in the span pays no penalty. The consumer adds one quadratic term and one free variable for each column of the basis. It does not add the dense ``N \\times N`` matrix of the same worst case, so the model holds no lifted semidefinite block and stays a second-order cone programme. This is the reason for the name compact.

A fitted set has a basis with orthonormal columns over its ``N`` rows, and no row in `R`. A row slice of the basis does not have orthonormal columns. So [`port_opt_view`](@ref) keeps the slice, and puts a factor of the rows that it drops into `R`. The view then keeps the penalty of the full set exactly.

The set is a construction of this library. [`OrthogonalUncertaintySet`](@ref) fits it, and cites the works that it extends.

# Mathematical definition

```math
\\begin{align}
\\underset{\\mathbf{\\Sigma} \\in U^{\\text{cpt}}_{\\mathbf{\\Sigma}}}{\\max} \\boldsymbol{w}^{\\intercal} \\mathbf{\\Sigma} \\boldsymbol{w} &= \\boldsymbol{w}^{\\intercal} \\hat{\\mathbf{\\Sigma}} \\boldsymbol{w} + \\kappa \\underset{\\boldsymbol{z}}{\\min} \\left( \\lVert \\mathbf{C} \\boldsymbol{w} - \\mathbf{Q} \\boldsymbol{z} \\rVert_{2}^{2} + \\lVert \\mathbf{R} \\boldsymbol{z} \\rVert_{2}^{2} \\right) \\\\
U^{\\text{cpt}}_{\\mathbf{\\Sigma}} &= \\left\\{ \\mathbf{\\Sigma} \\succeq 0 \\, \\vert \\, \\mathbf{\\Sigma} \\preceq \\hat{\\mathbf{\\Sigma}} + \\kappa \\mathbf{C}^{\\intercal} (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}) \\mathbf{C} \\right\\}\\,.
\\end{align}
```

Where:

  - ``U^{\\text{cpt}}_{\\mathbf{\\Sigma}}``: Compact uncertainty set for the covariance matrix.
  - ``\\mathbf{\\Sigma}``: Uncertain covariance.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:w_port])
  - $(math_dict[:kappa_cpt])
  - $(math_dict[:C_cpt])
  - ``\\mathbf{Q}``: Basis of the subspace that the penalty spares, ``N \\times r``.
  - ``\\mathbf{R}``: Factor of the rows of the basis that a view dropped, ``m \\times r``, with ``m = 0`` on a fitted set. The stacked matrix ``[\\mathbf{Q}; \\mathbf{R}]`` has orthonormal columns, so ``\\mathbf{R}^{\\intercal}\\mathbf{R} = \\mathbf{I} - \\mathbf{Q}^{\\intercal}\\mathbf{Q}``.
  - ``\\boldsymbol{z}``: Coefficient vector of the inner problem, ``r \\times 1``.

The two lines state one worst case. The inner problem is a least-squares problem over the columns of ``[\\mathbf{Q}; \\mathbf{R}]``, with the target ``[\\mathbf{C}\\boldsymbol{w}; \\mathbf{0}]``. These columns are orthonormal, so the value of the inner problem is ``\\boldsymbol{w}^{\\intercal}\\mathbf{C}^{\\intercal} (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}) \\mathbf{C}\\boldsymbol{w}``. Because ``\\mathbf{Q}^{\\intercal}\\mathbf{Q} \\preceq \\mathbf{I}``, the matrix ``\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}`` is positive semidefinite. So the bound on the second line is a member of the set, and it gives the maximum on the first line.

On a fitted set, ``\\mathbf{R}`` has no row and ``\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}`` is a projector. The value is then ``\\lVert (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``. The inner problem has a value for every ``\\mathbf{Q}`` and ``\\mathbf{R}``, but it agrees with the closed form only when the stacked columns are orthonormal. A weight vector with ``\\mathbf{C}\\boldsymbol{w} = \\mathbf{Q}\\boldsymbol{z}`` and ``\\mathbf{R}\\boldsymbol{z} = \\mathbf{0}`` pays no penalty. When ``r = 0``, the penalty is ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CompactCovarianceUncertaintySet(;
        kappa::Number,
        C::VecNum,
        Q::MatNum,
        R::MatNum = zeros(eltype(Q), 0, size(Q, 2)),
        val::Option{<:MatNum} = nothing
    ) -> CompactCovarianceUncertaintySet

Keywords correspond to the struct's fields.

## Validation

  - `!(kappa isa Bool)`, else an `ArgumentError`.
  - `isfinite(kappa)` and `kappa >= 0`.
  - `!isempty(C)`, `all(isfinite, C)` and `all(x -> x >= 0, C)`.
  - `all(isfinite, Q)`.
  - `size(Q, 1) == length(C)`.
  - `all(isfinite, R)` and `size(R, 2) == size(Q, 2)`.
  - If `val` is provided: `size(val, 1) == size(val, 2) == length(C)`.

# Examples

```jldoctest
julia> CompactCovarianceUncertaintySet(; kappa = 2.0, C = [1.0, 1.0],
                                       Q = reshape([1.0, 0.0], 2, 1))
CompactCovarianceUncertaintySet
  kappa ┼ Float64: 2.0
      C ┼ Vector{Float64}: [1.0, 1.0]
      Q ┼ 2×1 Matrix{Float64}
      R ┼ 0×1 Matrix{Float64}
    val ┴ nothing
```

# Related

  - [`AbstractUncertaintySetResult`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
  - [`ucs_variance`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct CompactCovarianceUncertaintySet <: AbstractUncertaintySetResult
    """
    Radius ``\\kappa \\geq 0``, the multiplier of the quadratic penalty. A radius of `0` removes the penalty and leaves the nominal variance. A set is a Result, so the radius is always a number. When the `kappa` of an [`OrthogonalUncertaintySet`](@ref) holds an [`AbstractCompactRadiusAlgorithm`](@ref), the estimator resolves it to a number before it builds the set.
    """
    kappa
    """
    Diagonal of the metric square root ``\\mathbf{C}``, a vector of length ``N``. The set holds the diagonal and not the matrix.
    """
    C
    """
    Basis ``\\mathbf{Q}`` of the subspace that the penalty spares, ``N \\times r``. On a fitted set the columns are orthonormal. A rank of `0` is permitted, and the penalty is then ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.
    """
    Q
    """
    Factor ``\\mathbf{R}`` of the rows of the basis that a view dropped, ``m \\times r``, with ``\\mathbf{R}^{\\intercal}\\mathbf{R} = \\mathbf{I} - \\mathbf{Q}^{\\intercal}\\mathbf{Q}``. A fitted set has no row here. A view fills it. Then the penalty of the view on a portfolio is the penalty of the full set on the same portfolio, with zero weight outside the view.
    """
    R
    """
    $(field_dict[:val_ucs])
    """
    val
    function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum, Q::MatNum, R::MatNum,
                                             val::Option{<:MatNum})
        @argcheck(!isa(kappa, Bool),
                  ArgumentError("kappa is a radius, so it must be a number and not a Bool. Got\nkappa => $kappa."))
        @argcheck(isfinite(kappa) && kappa >= zero(kappa),
                  DomainError(kappa, "kappa must be finite and >= 0"))
        @argcheck(!isempty(C), IsEmptyError("C cannot be empty"))
        @argcheck(all(isfinite, C), IsNonFiniteError("all entries of C must be finite"))
        @argcheck(all(x -> x >= zero(x), C),
                  DomainError(C, "all entries of C must be >= 0"))
        @argcheck(all(isfinite, Q), IsNonFiniteError("all entries of Q must be finite"))
        @argcheck(size(Q, 1) == length(C),
                  DimensionMismatch("Q ($(size(Q, 1)) rows) must match C ($(length(C)))"))
        @argcheck(all(isfinite, R), IsNonFiniteError("all entries of R must be finite"))
        @argcheck(size(R, 2) == size(Q, 2),
                  DimensionMismatch("R ($(size(R, 2)) columns) must match Q ($(size(Q, 2)))"))
        if isa(val, MatNum)
            assert_matrix_issquare(val, :val)
            @argcheck(size(val, 1) == length(C),
                      DimensionMismatch("val ($(size(val, 1))) must match C ($(length(C)))"))
        end
        return new{typeof(kappa), typeof(C), typeof(Q), typeof(R), typeof(val)}(kappa, C, Q,
                                                                                R, val)
    end
end
function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum, Q::MatNum,
                                         val::Option{<:MatNum} = nothing)::CompactCovarianceUncertaintySet
    # The four positional arguments of a released version, `(kappa, C, Q, val)`, keep their
    # meaning: a fitted set has no row in `R` (ADR 0189).
    return CompactCovarianceUncertaintySet(kappa, C, Q, zeros(eltype(Q), 0, size(Q, 2)),
                                           val)
end
function CompactCovarianceUncertaintySet(; kappa::Number, C::VecNum, Q::MatNum,
                                         R::MatNum = zeros(eltype(Q), 0, size(Q, 2)),
                                         val::Option{<:MatNum} = nothing)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(kappa, C, Q, R, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`CompactCovarianceUncertaintySet`](@ref) restricted to the asset indices `i`, as the projection of the set onto the selected assets and not as a refit.

A portfolio of the selected assets is a portfolio of the full universe with zero weight on the other assets. Its worst-case variance under the set is ``\\boldsymbol{w}_{i}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{ii} \\boldsymbol{w}_{i} + \\kappa \\boldsymbol{w}_{i}^{\\intercal} \\mathbf{C}_{i} (\\mathbf{I} - \\mathbf{Q}_{i}\\mathbf{Q}_{i}^{\\intercal}) \\mathbf{C}_{i} \\boldsymbol{w}_{i}``. The view states that value. So the worst case of a cluster portfolio does not depend on whether the caller views the set first.

The sliced rows ``\\mathbf{Q}_{i}`` do not have orthonormal columns, because ``\\mathbf{Q}_{i}^{\\intercal}\\mathbf{Q}_{i} = \\mathbf{I} - \\mathbf{Q}_{-i}^{\\intercal}\\mathbf{Q}_{-i}``. The view keeps them, and puts a triangular factor of the dropped rows ``\\mathbf{Q}_{-i}`` into `R`. Then ``[\\mathbf{Q}_{i}; \\mathbf{R}]`` has orthonormal columns, and the inner problem of the set gives the value above.

A slice with new orthonormal columns gives ``\\mathbf{C}_{i} (\\mathbf{I} - \\mathbf{P}_{i}) \\mathbf{C}_{i}`` instead, where ``\\mathbf{P}_{i}`` is the projector onto ``\\operatorname{col}(\\mathbf{Q}_{i})``. That is the set of a fit on the selected assets alone, and it spares portfolios that the full set does not spare. A caller who wants that set fits the estimator on the viewed prior.

The radius passes through unchanged. The projection of a set keeps its radius, so the radius stays exact. A set is a Result, so it holds the radius as a number and holds no rule to run again.

# Algorithm

 1. Mark the rows of the basis that `i` drops and that are not zero. A zero row adds nothing to the Gram matrix. The rows outside an Investable Mask are zero, so the view at the mask of a set that [`expand_investable_ucs`](@ref) wrote recovers the fitted set exactly.
 2. Stack the marked rows of `risk_ucs.Q` on `risk_ucs.R`, the factor of the rows that an earlier view dropped. When no row is marked, keep `risk_ucs.R` as `R`. Otherwise take the triangular factor of the `LinearAlgebra.qr` of the stack as `R`. The Gram matrix of `R` is the Gram matrix of the stack, so a view of a view stays exact.
 3. Take `view(risk_ucs.C, i)` and `view(risk_ucs.Q, i, :)`, the diagonal of the metric square root and the rows of the basis at the selected assets.
 4. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the nominal covariance restricted to the same assets on both axes. A `nothing` passes through unchanged.
 5. Build a [`CompactCovarianceUncertaintySet`](@ref) from the four, with `kappa` unchanged.

# Arguments

  - `risk_ucs`: Compact covariance uncertainty set.
  - `i`: Indices of the assets that the view keeps.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `risk_ucs::CompactCovarianceUncertaintySet`: The set restricted to `i`.

# Related

  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`OrthogonalUncertaintySet`](@ref)
  - [`port_opt_view`](@ref)
"""
function port_opt_view(risk_ucs::CompactCovarianceUncertaintySet, i,
                       args...)::CompactCovarianceUncertaintySet
    Q = risk_ucs.Q
    # A zero row adds nothing to the Gram matrix, so only a dropped row that is not zero
    # enters `R`. The rows outside an Investable Mask are zero, so the view at the mask
    # recovers the fitted set exactly.
    dropped = vec(any(!iszero, Q; dims = 2))
    view(dropped, i) .= false
    R = if any(dropped)
        Matrix(LinearAlgebra.qr(vcat(view(Q, dropped, :), risk_ucs.R)).R)
    else
        risk_ucs.R
    end
    return CompactCovarianceUncertaintySet(; kappa = risk_ucs.kappa,
                                           C = view(risk_ucs.C, i), Q = view(Q, i, :),
                                           R = R,
                                           val = nothing_scalar_array_view(risk_ucs.val, i))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Write a [`CompactCovarianceUncertaintySet`](@ref) fitted on the Investable Mask back onto the full asset universe.

The expansion is the inverse of [`port_opt_view`](@ref). The view leaves a zero row of the basis out of `R`, because a zero row adds nothing to the Gram matrix. So the view at the mask recovers the fitted set. The rows outside the mask also hold a zero of `C`, so the set states nothing about an asset that the prior could not estimate. The nominal covariance is the full `pr.sigma`, with its `NaN` frame. The estimator calibrated the set on this prior, and the prior covers the full universe.

# Algorithm

 1. Allocate a zero frame of `length(imsk)` rows and `size(set.Q, 2)` columns. Write `set.Q` at the rows that the mask keeps. This frame is `Q`.
 2. Allocate a zero vector of `length(imsk)` entries. Write `set.C` at the same rows. This vector is `C`.
 3. Build a [`CompactCovarianceUncertaintySet`](@ref) from the two, with `kappa` and `R` unchanged and `pr.sigma` as `val`. A zero row adds nothing to the Gram matrix of the basis, so `R` stays the factor of the rows that a view dropped.

# Arguments

  - `set`: Compact covariance uncertainty set fitted on the reduced prior.
  - `imsk`: The Investable Mask of the full prior.
  - $(arg_dict[:pr])

# Returns

  - `set::CompactCovarianceUncertaintySet`: The set on the full asset universe.

# Related

  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`investable_ucs_reduction`](@ref)
  - [`port_opt_view`](@ref)
"""
function expand_investable_ucs(set::CompactCovarianceUncertaintySet, imsk::BitVector,
                               pr::AbstractPriorResult)::CompactCovarianceUncertaintySet
    N = length(imsk)
    Q = zeros(eltype(set.Q), N, size(set.Q, 2))
    Q[imsk, :] = set.Q
    C = zeros(eltype(set.C), N)
    C[imsk] = set.C
    return CompactCovarianceUncertaintySet(; kappa = set.kappa, C = C, Q = Q, R = set.R,
                                           val = pr.sigma)
end
"""
    mu_ucs(uc::CompactCovarianceUncertaintySet, args...; kwargs...)

Always throw. [`CompactCovarianceUncertaintySet`](@ref) is a set for the covariance only.

The method refuses and runs no procedure, so it has no `# Algorithm` section. It replaces the passthrough method that every other Result reaches. That method gives the set to the consumer of the mean, which cannot read a covariance set.

# Arguments

  - `uc`: Compact covariance uncertainty set.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - The method always throws an `ArgumentError`. The set bounds a covariance matrix through a quadratic penalty on the weights, and the library defines no mean analogue for it. The message names the sets to use for the mean, [`BoxUncertaintySet`](@ref), [`EllipsoidalUncertaintySet`](@ref) and [`L1UncertaintySet`](@ref).

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
