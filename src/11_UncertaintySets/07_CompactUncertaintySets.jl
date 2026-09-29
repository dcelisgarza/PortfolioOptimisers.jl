"""
$(DocStringExtensions.TYPEDEF)

Holds a worst-case variance penalty as a radius, a diagonal metric square root and a basis of the directions the penalty spares.

The set inflates the variance on every direction outside the span of its basis and on none inside it, so a portfolio built out of that span pays nothing. The consumer adds one quadratic term and `rank` free variables instead of the dense ``N \\times N`` matrix the same worst case would otherwise need, which is why the type is called compact: no lifted semidefinite block appears, and the programme stays a second-order cone programme. A fitted set has a basis orthonormal over its ``N`` rows and no row in `R`. A row slice of the basis is not orthonormal, so [`port_opt_view`](@ref) keeps the slice and puts the rows it drops into `R`, which keeps the penalty of the full set exactly.

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
  - ``\\kappa \\geq 0``: Radius, the multiplier of the penalty.
  - ``\\mathbf{C} = \\operatorname{diag}(\\boldsymbol{c})``: Diagonal metric square root, ``N \\times N``.
  - ``\\mathbf{Q}``: Basis of the spared subspace, ``N \\times r``.
  - ``\\mathbf{R}``: Factor of the rows a view dropped from the basis, ``m \\times r``, with ``m = 0`` on a fitted set. The stacked matrix ``[\\mathbf{Q}; \\mathbf{R}]`` has orthonormal columns, so ``\\mathbf{R}^{\\intercal}\\mathbf{R} = \\mathbf{I} - \\mathbf{Q}^{\\intercal}\\mathbf{Q}``.
  - ``\\boldsymbol{z}``: Coefficient vector of the inner problem, ``r \\times 1``.

The two lines are one object because the inner problem is a least-squares problem over the orthonormal columns of ``[\\mathbf{Q}; \\mathbf{R}]`` with the target ``[\\mathbf{C}\\boldsymbol{w}; \\mathbf{0}]``, so its value is ``\\boldsymbol{w}^{\\intercal}\\mathbf{C}^{\\intercal} (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}) \\mathbf{C}\\boldsymbol{w}``. On a fitted set ``\\mathbf{R}`` has no row, ``\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal}`` is a projector, and the value is ``\\lVert (\\mathbf{I} - \\mathbf{Q}\\mathbf{Q}^{\\intercal})\\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``. The variational form on the right of the first line is the weaker of the two: it is defined for **any** ``\\mathbf{Q}`` and ``\\mathbf{R}``, whereas the closed form of the second line needs orthonormal stacked columns. A weight vector with ``\\mathbf{C}\\boldsymbol{w} = \\mathbf{Q}\\boldsymbol{z}`` and ``\\mathbf{R}\\boldsymbol{z} = \\mathbf{0}`` pays a zero penalty, and ``r = 0`` leaves ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.

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
  - [`UncertaintySetVariance`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
  - [`ucs_variance`](@ref)
  - [`port_opt_view`](@ref)
"""
@concrete struct CompactCovarianceUncertaintySet <: AbstractUncertaintySetResult
    """
    Radius ``\\kappa \\geq 0``, the multiplier of the quadratic penalty, and `0` disables the penalty and leaves the nominal variance. A set is a Result, so this is always a number: an [`OrthogonalUncertaintySet`](@ref) whose own `kappa` held an [`AbstractCompactRadiusAlgorithm`](@ref) resolved it before building the set.
    """
    kappa
    """
    Diagonal of the metric square root ``\\mathbf{C}``, of length ``N``, held as a vector rather than as a matrix.
    """
    C
    """
    Basis ``\\mathbf{Q}`` of the subspace the penalty spares, ``N \\times r``, with orthonormal columns on a fitted set. A rank of `0` is admitted and leaves the penalty ``\\kappa \\lVert \\mathbf{C}\\boldsymbol{w} \\rVert_{2}^{2}``.
    """
    Q
    """
    Factor ``\\mathbf{R}`` of the rows of the basis that a view dropped, ``m \\times r``, with ``\\mathbf{R}^{\\intercal}\\mathbf{R} = \\mathbf{I} - \\mathbf{Q}^{\\intercal}\\mathbf{Q}``. A fitted set has no row here. A view fills it, so that the penalty of the view is the penalty of the full set on a portfolio that holds nothing outside the view.
    """
    R
    """
    $(field_dict[:val_ucs])
    """
    val
    function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum, Q::MatNum, R::MatNum,
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
function CompactCovarianceUncertaintySet(kappa::Number, C::VecNum,
                                         Q::MatNum)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(kappa, C, Q, zeros(eltype(Q), 0, size(Q, 2)),
                                           nothing)
end
function CompactCovarianceUncertaintySet(; kappa::Number, C::VecNum, Q::MatNum,
                                         R::MatNum = zeros(eltype(Q), 0, size(Q, 2)),
                                         val::Option{<:MatNum} = nothing)::CompactCovarianceUncertaintySet
    return CompactCovarianceUncertaintySet(kappa, C, Q, R, val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a view of a [`CompactCovarianceUncertaintySet`](@ref) restricted to the asset indices `i`: the projection of the set onto the selected assets, and not a refit.

A portfolio of the selected assets is a portfolio of the full universe that holds nothing elsewhere, and its worst-case variance under the set is ``\\boldsymbol{w}_{i}^{\\intercal} \\hat{\\mathbf{\\Sigma}}_{ii} \\boldsymbol{w}_{i} + \\kappa \\boldsymbol{w}_{i}^{\\intercal} \\mathbf{C}_{i} (\\mathbf{I} - \\mathbf{Q}_{i}\\mathbf{Q}_{i}^{\\intercal}) \\mathbf{C}_{i} \\boldsymbol{w}_{i}``. The view states exactly that value, so the worst case of a cluster portfolio does not depend on whether the set is viewed first (ADR 0189).

The sliced rows ``\\mathbf{Q}_{i}`` are not orthonormal, because ``\\mathbf{Q}_{i}^{\\intercal}\\mathbf{Q}_{i} = \\mathbf{I} - \\mathbf{Q}_{-i}^{\\intercal}\\mathbf{Q}_{-i}``. The view keeps them as they are, and puts a triangular factor of the dropped rows ``\\mathbf{Q}_{-i}`` into `R`. Then ``[\\mathbf{Q}_{i}; \\mathbf{R}]`` has orthonormal columns, and the inner problem of the set gives the value above. A re-orthonormalised slice would give ``\\mathbf{C}_{i} (\\mathbf{I} - \\mathbf{P}_{i}) \\mathbf{C}_{i}`` instead, with ``\\mathbf{P}_{i}`` the projector onto ``\\operatorname{col}(\\mathbf{Q}_{i})``. That is the set a fit on the selected assets alone gives, and it spares portfolios the full set does not spare. A caller who wants that set fits the estimator on the viewed prior.

The radius passes through unchanged, and it stays exact: the projection of a set keeps its radius. A set is a Result, so it carries the radius as a **number** and holds no rule that could be run again.

# Algorithm

 1. Mark the rows of the basis that `i` drops and that are not zero. A zero row adds nothing to the Gram matrix, and the rows outside an Investable Mask are zero, so the view at the mask of a set that [`expand_investable_ucs`](@ref) wrote recovers the fitted set exactly.
 2. Stack the marked rows of `risk_ucs.Q` on `risk_ucs.R`, the factor of the rows an earlier view dropped. When no row is marked, keep `risk_ucs.R`. Otherwise take the triangular factor of the `LinearAlgebra.qr` of the stack, giving `R`. Its Gram matrix is the Gram matrix of the stack, so a view of a view stays exact.
 3. Take `view(risk_ucs.C, i)` and `view(risk_ucs.Q, i, :)`, the diagonal of the metric square root and the rows of the basis the selected assets occupy.
 4. Take `nothing_scalar_array_view(risk_ucs.val, i)`, the nominal covariance restricted to the same assets on both axes, which passes a `nothing` through unchanged.
 5. Build a [`CompactCovarianceUncertaintySet`](@ref) from the four, carrying `kappa` through unchanged.

# Arguments

  - `risk_ucs`: Compact covariance uncertainty set.
  - `i`: Cluster or asset index.
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

The expansion is the inverse of [`port_opt_view`](@ref): the view leaves a zero row of the basis out of `R`, because it adds nothing to the Gram matrix, so the view at the mask recovers the fitted set. The rows outside the mask carry a zero of `C` too, so the set states nothing about an asset the prior could not estimate, and the nominal covariance is the full `pr.sigma`, `NaN` frame and all, because the set is a neighbourhood of the prior it was calibrated on and that prior lives on the full universe.

# Algorithm

 1. Allocate a zero frame of `length(imsk)` rows and `size(set.Q, 2)` columns, and write `set.Q` at the rows the mask keeps, giving `Q`.
 2. Allocate a zero vector of `length(imsk)` entries and write `set.C` at the same rows, giving `C`.
 3. Build a [`CompactCovarianceUncertaintySet`](@ref) from the two, carrying `kappa` and `R` through unchanged and `pr.sigma` as `val`. A zero row adds nothing to the Gram matrix of the basis, so `R` stays the factor of the rows a view dropped.

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
