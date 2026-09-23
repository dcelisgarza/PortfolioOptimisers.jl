"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that build a covariance shrinkage target from the matrix being shrunk.

Each rule is a target of the taxonomy of Schäfer and Strimmer that is positive definite, so a geodesic reaches it. Their target with perfect positive correlation has rank one and is not a member. [`DiagonalTarget`](@ref) is a rule of the same kind, and it is a member of [`GeodesicShrinkageTarget`](@ref) instead, because it is also a [`RegimeAdjustedTarget`](@ref).

# Interfaces

In order to implement a new target rule, subtype `AbstractCovarianceShrinkageTarget` and implement the following method.

## `shrinkage_target`

  - `shrinkage_target(tgt::MyShrinkageTarget, sigma::MatNum) -> MatNum`: Builds the target matrix from the matrix being shrunk.

### Arguments

  - `tgt`: The target rule.
  - $(arg_dict[:sigrho])

### Returns

  - `target::MatNum`: A positive definite matrix of the size of `sigma`.

### Examples

```jldoctest
julia> struct MyShrinkageTarget <: PortfolioOptimisers.AbstractCovarianceShrinkageTarget end

julia> function PortfolioOptimisers.shrinkage_target(::MyShrinkageTarget,
                                                     sigma::PortfolioOptimisers.MatNum)
           return 2 * PortfolioOptimisers.shrinkage_target(IdentityTarget(), sigma)
       end

julia> ce = GeodesicShrinkageCovariance(; tgt = MyShrinkageTarget(), alpha = 1);

julia> cov(ce, [1.0 2.0; 2.0 1.0; 0.0 0.5])
2×2 Matrix{Float64}:
 2.0  0.0
 0.0  2.0
```

# Related

  - [`IdentityTarget`](@ref)
  - [`ScaledIdentityTarget`](@ref)
  - [`CommonCovarianceTarget`](@ref)
  - [`ConstantCorrelationTarget`](@ref)
  - [`GeodesicShrinkageTarget`](@ref)
  - [`shrinkage_target`](@ref)

# References

  - $(ref_dict[:schaferstrimmer2005])
"""
abstract type AbstractCovarianceShrinkageTarget <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Targets the identity matrix, which ignores the scale of the matrix being shrunk.

# Mathematical definition

```math
\\begin{align}
\\mathbf{T} &= \\mathbf{I}\\,.
\\end{align}
```

Where:

  - $(math_dict[:T_shrink_target])
  - ``\\mathbf{I}``: Identity matrix.

# Examples

```jldoctest
julia> IdentityTarget()
IdentityTarget()
```

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`ScaledIdentityTarget`](@ref): the same shape at the scale of the average variance.
  - [`GeodesicShrinkageCovariance`](@ref)

# References

  - $(ref_dict[:schaferstrimmer2005]) Table 2, target A.
"""
struct IdentityTarget <: AbstractCovarianceShrinkageTarget end
"""
$(DocStringExtensions.TYPEDEF)

Targets the identity matrix scaled by the average variance, the target with condition number one.

The target has the trace of the matrix being shrunk. It commutes with every matrix, so a geodesic towards it keeps the eigenvectors of the start and moves only the eigenvalues.

# Mathematical definition

```math
\\begin{align}
\\mathbf{T} &= \\bar{v} \\, \\mathbf{I}\\,, \\\\
\\bar{v} &= \\frac{\\operatorname{tr}(\\hat{\\mathbf{\\Sigma}})}{N}\\,.
\\end{align}
```

Where:

  - $(math_dict[:T_shrink_target])
  - ``\\bar{v}``: Average variance.
  - ``\\mathbf{I}``: Identity matrix.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:N])

# Examples

```jldoctest
julia> ScaledIdentityTarget()
ScaledIdentityTarget()
```

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`IdentityTarget`](@ref)
  - [`GeodesicShrinkageCovariance`](@ref)

# References

  - $(ref_dict[:schaferstrimmer2005]) Table 2, target B.
"""
struct ScaledIdentityTarget <: AbstractCovarianceShrinkageTarget end
"""
$(DocStringExtensions.TYPEDEF)

Targets a matrix with the average variance on the diagonal and the average covariance everywhere else.

# Mathematical definition

```math
\\begin{align}
T_{i,\\,j} &= \\begin{cases} \\bar{v} & i = j\\,, \\\\ \\bar{c} & i \\neq j\\,, \\end{cases} \\\\
\\bar{v} &= \\frac{1}{N} \\sum_{i = 1}^{N} \\hat{\\Sigma}_{i,\\,i}\\,, \\\\
\\bar{c} &= \\frac{1}{N (N - 1)} \\sum_{i \\neq j} \\hat{\\Sigma}_{i,\\,j}\\,.
\\end{align}
```

Where:

  - ``T_{i,\\,j}``: Entry of the shrinkage target.
  - ``\\bar{v}``: Average variance.
  - ``\\bar{c}``: Average covariance of the distinct pairs.
  - ``\\hat{\\Sigma}_{i,\\,j}``: Entry of the matrix being shrunk.
  - $(math_dict[:N])

The target is positive definite when ``-\\bar{v} / (N - 1) < \\bar{c} < \\bar{v}``, which holds for the covariance of assets that are not all perfectly correlated.

# Examples

```jldoctest
julia> CommonCovarianceTarget()
CommonCovarianceTarget()
```

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`ConstantCorrelationTarget`](@ref): keeps the variances and averages the correlations.
  - [`GeodesicShrinkageCovariance`](@ref)

# References

  - $(ref_dict[:schaferstrimmer2005]) Table 2, target C.
"""
struct CommonCovarianceTarget <: AbstractCovarianceShrinkageTarget end
"""
$(DocStringExtensions.TYPEDEF)

Targets a matrix that keeps the variances and gives every pair of assets the average correlation.

# Mathematical definition

```math
\\begin{align}
T_{i,\\,j} &= \\begin{cases} \\hat{\\Sigma}_{i,\\,i} & i = j\\,, \\\\ \\bar{r} \\sqrt{\\hat{\\Sigma}_{i,\\,i} \\, \\hat{\\Sigma}_{j,\\,j}} & i \\neq j\\,, \\end{cases} \\\\
\\bar{r} &= \\frac{1}{N (N - 1)} \\sum_{i \\neq j} \\frac{\\hat{\\Sigma}_{i,\\,j}}{\\sqrt{\\hat{\\Sigma}_{i,\\,i} \\, \\hat{\\Sigma}_{j,\\,j}}}\\,.
\\end{align}
```

Where:

  - ``T_{i,\\,j}``: Entry of the shrinkage target.
  - ``\\bar{r}``: Average correlation of the distinct pairs.
  - ``\\hat{\\Sigma}_{i,\\,j}``: Entry of the matrix being shrunk.
  - $(math_dict[:N])

The target is positive definite when every variance is positive and ``-1 / (N - 1) < \\bar{r} < 1``.

# Examples

```jldoctest
julia> ConstantCorrelationTarget()
ConstantCorrelationTarget()
```

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`CommonCovarianceTarget`](@ref)
  - [`GeodesicShrinkageCovariance`](@ref)

# References

  - $(ref_dict[:schaferstrimmer2005]) Table 2, target F.
"""
struct ConstantCorrelationTarget <: AbstractCovarianceShrinkageTarget end
"""
    const GeodesicShrinkageTarget = Union{<:AbstractCovarianceShrinkageTarget,
                                          <:DiagonalTarget, <:MatNum}

Groups the targets a [`GeodesicShrinkageCovariance`](@ref) accepts: the rules that build the target from the matrix being shrunk, and a fixed positive definite matrix.

The members share no supertype. [`DiagonalTarget`](@ref) is also a [`RegimeAdjustedTarget`](@ref), and a matrix is data, so the field is bounded by this union and [`shrinkage_target`](@ref) dispatches on its members.

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`DiagonalTarget`](@ref)
  - [`MatNum`](@ref)
"""
const GeodesicShrinkageTarget = Union{<:AbstractCovarianceShrinkageTarget, <:DiagonalTarget,
                                      <:MatNum}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the target matrix of a covariance shrinkage from the matrix being shrunk.

Each rule states its target under `# Mathematical definition`. A fixed matrix is returned as it is, after a check that its size is the size of `sigma`.

# Arguments

  - `tgt`: The target.
      + `::IdentityTarget`: the identity matrix, as a `Diagonal`.
      + `::ScaledIdentityTarget`: the average variance of `sigma` times the identity matrix, as a `Diagonal`.
      + `::CommonCovarianceTarget`: the average variance on the diagonal and the average covariance elsewhere.
      + `::ConstantCorrelationTarget`: the variances of `sigma` and the average correlation of its pairs.
      + `::DiagonalTarget`: the diagonal of `sigma`, as a `Diagonal`.
      + `::MatNum`: the matrix itself.
  - $(arg_dict[:sigrho])

# Validation

  - A matrix `tgt` has the size of `sigma`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `target::MatNum`: The target matrix.

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`GeodesicShrinkageTarget`](@ref)
"""
function shrinkage_target(::IdentityTarget, sigma::MatNum)
    return LinearAlgebra.Diagonal(fill(one(eltype(sigma)), size(sigma, 1)))
end
function shrinkage_target(::ScaledIdentityTarget, sigma::MatNum)
    n = size(sigma, 1)
    return LinearAlgebra.Diagonal(fill(LinearAlgebra.tr(sigma) / n, n))
end
function shrinkage_target(::CommonCovarianceTarget, sigma::MatNum)
    n = size(sigma, 1)
    v = LinearAlgebra.tr(sigma) / n
    c = n > 1 ? (sum(sigma) - LinearAlgebra.tr(sigma)) / (n * (n - 1)) : zero(v)
    tgt_mat = fill(c, n, n)
    tgt_mat[LinearAlgebra.diagind(tgt_mat)] .= v
    return tgt_mat
end
function shrinkage_target(::ConstantCorrelationTarget, sigma::MatNum)
    n = size(sigma, 1)
    sd = sqrt.(LinearAlgebra.diag(sigma))
    rho = sigma ./ (sd * transpose(sd))
    r = n > 1 ? (sum(rho) - LinearAlgebra.tr(rho)) / (n * (n - 1)) : zero(eltype(rho))
    tgt_mat = r * (sd * transpose(sd))
    tgt_mat[LinearAlgebra.diagind(tgt_mat)] .= LinearAlgebra.diag(sigma)
    return tgt_mat
end
function shrinkage_target(::DiagonalTarget, sigma::MatNum)
    return LinearAlgebra.Diagonal(LinearAlgebra.diag(sigma))
end
function shrinkage_target(tgt::MatNum, sigma::MatNum)
    @argcheck(size(tgt) == size(sigma),
              DimensionMismatch("the shrinkage target has size $(size(tgt)) and the matrix it shrinks has size $(size(sigma)). A fixed target names one asset universe, and a fit over a sample with gaps shrinks the block of the assets it could estimate, which is smaller when an asset has no observation. State a target of the size of the matrix, or use a target rule, which is built from the matrix."))
    return tgt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Checks a covariance shrinkage target at construction.

A target rule needs no check, because it is built from the matrix it shrinks. A fixed matrix must be a valid target for every matrix it meets, so it is checked once here.

# Arguments

  - `tgt`: The target.

# Validation

  - A matrix `tgt` is square. A `DimensionMismatch` is thrown otherwise.
  - A matrix `tgt` is finite, symmetric within the default tolerance of `isapprox`, and positive definite. A `DomainError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`shrinkage_target`](@ref)
"""
function assert_shrinkage_target(::Union{<:AbstractCovarianceShrinkageTarget,
                                         <:DiagonalTarget})::Nothing
    return nothing
end
function assert_shrinkage_target(tgt::MatNum)::Nothing
    assert_matrix_issquare(tgt, :tgt)
    assert_finite(tgt, :tgt)
    @argcheck(isapprox(tgt, transpose(tgt)),
              DomainError(tgt,
                          "`tgt` is not symmetric. A covariance target is symmetric, and the shrinkage reads the upper triangle of the target alone, so an asymmetric matrix states two targets. State a symmetric matrix."))
    @argcheck(LinearAlgebra.isposdef(LinearAlgebra.Symmetric(tgt)),
              DomainError(tgt,
                          "`tgt` is not positive definite. The shrinkage computes the geodesic from the target end, which exists only when that end is positive definite. State a positive definite matrix, or repair it with `posdef`."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks a covariance matrix towards a target along the geodesic of the positive definite matrices.

Linear shrinkage averages the matrix and its target entry by entry. This estimator follows the shortest path between them under the affine-invariant metric instead, so the estimate is positive definite at every intensity when both ends are positive definite. Towards a target that commutes with the start the eigenvalues are interpolated geometrically: towards [`ScaledIdentityTarget`](@ref), each eigenvalue ``\\lambda_i`` moves to ``\\lambda_i^{1 - \\alpha} \\bar{v}^{\\alpha}``, where linear shrinkage gives ``(1 - \\alpha) \\lambda_i + \\alpha \\bar{v}``. The trace is not preserved at an intermediate intensity.

A zero eigenvalue of the start stays zero at every intensity below one, so the shrinkage does not repair a singular matrix. That is what `pdm` is for: it repairs the matrix `ce` computes before the shrinkage runs. `cor` returns the correlation matrix of the shrunk covariance.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mathbf{\\Sigma}}_{\\alpha} &= \\hat{\\mathbf{\\Sigma}}^{1/2} \\left(\\hat{\\mathbf{\\Sigma}}^{-1/2} \\, \\mathbf{T} \\, \\hat{\\mathbf{\\Sigma}}^{-1/2}\\right)^{\\alpha} \\hat{\\mathbf{\\Sigma}}^{1/2} \\\\
&= \\mathbf{T}^{1/2} \\left(\\mathbf{T}^{-1/2} \\, \\hat{\\mathbf{\\Sigma}} \\, \\mathbf{T}^{-1/2}\\right)^{1 - \\alpha} \\mathbf{T}^{1/2}\\,.
\\end{align}
```

The second form needs only ``\\mathbf{T}`` to be positive definite. It is the continuous extension of the first to a positive semidefinite ``\\hat{\\mathbf{\\Sigma}}``. Two targets give closed forms:

```math
\\begin{align}
\\mathbf{T} = \\bar{v} \\mathbf{I} &\\implies \\hat{\\mathbf{\\Sigma}}_{\\alpha} = \\mathbf{V} \\operatorname{Diag}\\left(\\lambda_i^{1 - \\alpha} \\bar{v}^{\\alpha}\\right) \\mathbf{V}^\\intercal\\,, \\\\
\\mathbf{T} = \\mathbf{D} &\\implies \\hat{\\mathbf{\\Sigma}}_{\\alpha} = \\mathbf{D}^{1/2} \\, \\mathbf{R}^{1 - \\alpha} \\, \\mathbf{D}^{1/2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\mathbf{\\Sigma}}_{\\alpha}``: Shrunk covariance matrix.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T_shrink_target])
  - ``\\alpha \\in [0, 1]``: Shrinkage intensity, the fraction of the geodesic distance from ``\\hat{\\mathbf{\\Sigma}}`` to ``\\mathbf{T}`` that the estimate travels.
  - ``\\bar{v} = \\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}) / N``: Average variance.
  - ``\\mathbf{V}``, ``\\lambda_i``: Eigenvectors and eigenvalues of ``\\hat{\\mathbf{\\Sigma}}``.
  - ``\\mathbf{D} = \\operatorname{Diag}(\\hat{\\mathbf{\\Sigma}})``: Diagonal matrix of the variances.
  - ``\\mathbf{R} = \\mathbf{D}^{-1/2} \\hat{\\mathbf{\\Sigma}} \\mathbf{D}^{-1/2}``: Correlation matrix of ``\\hat{\\mathbf{\\Sigma}}``.
  - $(math_dict[:N])

The geodesic distance from ``\\hat{\\mathbf{\\Sigma}}`` to ``\\hat{\\mathbf{\\Sigma}}_{\\alpha}`` is ``\\alpha`` times the distance from ``\\hat{\\mathbf{\\Sigma}}`` to ``\\mathbf{T}``. Towards ``\\bar{v} \\mathbf{I}``, the condition number of ``\\hat{\\mathbf{\\Sigma}}_{\\alpha}`` is ``\\kappa(\\hat{\\mathbf{\\Sigma}})^{1 - \\alpha}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeodesicShrinkageCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
        tgt::GeodesicShrinkageTarget = ScaledIdentityTarget(),
        alpha::Number = 0.1
    ) -> GeodesicShrinkageCovariance

Keywords correspond to the struct's fields.

## Validation

  - `0 <= alpha <= 1`.
  - A matrix `tgt` is square, finite, symmetric and positive definite, as [`assert_shrinkage_target`](@ref) states.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

`GeodesicShrinkageCovariance` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `ce` recurses through [`port_opt_view`](@ref) with every argument.
  - A matrix `tgt` is sliced on both axes, because it is an asset-by-asset matrix. A target rule passes through unchanged, because it is built from the viewed matrix.

# Examples

```jldoctest
julia> GeodesicShrinkageCovariance()
GeodesicShrinkageCovariance
     ce ┼ Covariance
        │    me ┼ SimpleExpectedReturns
        │       │   w ┴ nothing
        │    ce ┼ GeneralCovariance
        │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
        │       │    w ┴ nothing
        │   alg ┼ FullMoment()
        │     w ┴ nothing
    pdm ┼ Posdef
        │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
    tgt ┼ ScaledIdentityTarget()
  alpha ┴ Float64: 0.1

julia> X = [0.01 0.02 0.0; -0.02 0.01 0.01; 0.03 -0.01 0.02; 0.0 0.01 -0.01];

julia> cond(cov(GeodesicShrinkageCovariance(; alpha = 0.5), X)) ≈ sqrt(cond(cov(Covariance(), X)))
true
```

# Related

  - [`AbstractCovarianceShrinkageTarget`](@ref)
  - [`GeodesicShrinkageTarget`](@ref)
  - [`Covariance`](@ref)
  - [`Posdef`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:bhatia2007]) Chapter 6.
  - $(ref_dict[:musolas2021]) Section 2.
"""
@propagatable @concrete struct GeodesicShrinkageCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce]) It computes the matrix that the estimator shrinks.
    """
    @fprop ce
    """
    $(field_dict[:opdm]) It repairs the matrix `ce` computes before the shrinkage runs.
    """
    pdm
    """
    $(field_dict[:mutgt]) A rule of [`GeodesicShrinkageTarget`](@ref), or a fixed positive definite matrix.
    """
    tgt
    """
    Shrinkage intensity, the fraction of the geodesic from the matrix to the target that the estimate travels. `0` returns the matrix unchanged and `1` returns the target.
    """
    alpha
    function GeodesicShrinkageCovariance(ce::StatsBase.CovarianceEstimator,
                                         pdm::Option{<:AbstractPosdefEstimator},
                                         tgt::GeodesicShrinkageTarget, alpha::Number)
        assert_shrinkage_target(tgt)
        assert_closed_unit_interval(alpha, :alpha)
        return new{typeof(ce), typeof(pdm), typeof(tgt), typeof(alpha)}(ce, pdm, tgt, alpha)
    end
end
function GeodesicShrinkageCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                                     pdm::Option{<:AbstractPosdefEstimator} = Posdef(),
                                     tgt::GeodesicShrinkageTarget = ScaledIdentityTarget(),
                                     alpha::Number = 0.1)::GeodesicShrinkageCovariance
    return GeodesicShrinkageCovariance(ce, pdm, tgt, alpha)
end
"""
    port_opt_view(ce::GeodesicShrinkageCovariance, i, args...) -> GeodesicShrinkageCovariance

Restricts a [`GeodesicShrinkageCovariance`](@ref) to the assets `i`.

A fixed target is an asset-by-asset matrix, so it is sliced on both axes, as a covariance matrix is. A target rule is built from the matrix it shrinks, so it needs no slice.

# Algorithm

 1. View `ce.ce` with [`port_opt_view`](@ref) and every argument.
 2. Slice a matrix `ce.tgt` with [`nothing_scalar_array_view`](@ref), and keep a target rule as it is.
 3. Rebuild the estimator with the viewed fields, `ce.pdm` and `ce.alpha`.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - `i`: Indices of the selected assets.
  - `args...`: Additional positional arguments, forwarded to the view of `ce.ce`.

# Returns

  - `ce::GeodesicShrinkageCovariance`: The estimator restricted to the assets `i`.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`port_opt_view`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(ce::GeodesicShrinkageCovariance, i,
                       args...)::GeodesicShrinkageCovariance
    tgt = isa(ce.tgt, AbstractMatrix) ? nothing_scalar_array_view(ce.tgt, i) : ce.tgt
    return GeodesicShrinkageCovariance(port_opt_view(ce.ce, i, args...), ce.pdm, tgt,
                                       ce.alpha)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Raises the eigenvalues of a positive semidefinite matrix to a power, reading an eigenvalue inside the round-off tolerance as zero.

An eigendecomposition of a singular positive semidefinite matrix returns its zero eigenvalues as small numbers of either sign. The tolerance is the one a symmetric eigensolver resolves: the length of `vals` times the machine epsilon of its element type times its largest magnitude. A power below one magnifies such a number, `(1e-17)^0.5` is about `3e-9`, so an eigenvalue inside the tolerance is set to zero before the power, and a zero eigenvalue of the start stays zero. An eigenvalue below the negative of the tolerance belongs to an indefinite matrix, which no geodesic reaches.

# Arguments

  - `vals`: Eigenvalues of a symmetric matrix.
  - `p`: The power, positive.

# Validation

  - `minimum(vals)` is not below the negative of the tolerance. A `DomainError` is thrown otherwise.

# Returns

  - `vals_p::VecNum`: The eigenvalues raised to `p`, with every eigenvalue inside the tolerance at zero.

# Related

  - [`geodesic_point`](@ref)
"""
function geodesic_power(vals::VecNum, p::Number)
    tol = length(vals) * eps(eltype(vals)) * maximum(abs, vals)
    lo = minimum(vals)
    @argcheck(lo >= -tol,
              DomainError(lo,
                          "the matrix to shrink has the eigenvalue $lo, below the round-off tolerance $(-tol), so it is not positive semidefinite and no geodesic reaches it. An available-case covariance can be indefinite. Repair it with the `pdm` field of `GeodesicShrinkageCovariance`, which holds `Posdef()` by default."))
    return ifelse.(vals .> tol, vals, zero(eltype(vals))) .^ p
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the point at intensity `alpha` on the geodesic from `sigma` to the target of `tgt`.

# Algorithm

For [`ScaledIdentityTarget`](@ref):

 1. Take the average variance `v` from the trace of `sigma`.
 2. At `alpha = 1`, return `v` times the identity matrix, as a `Diagonal`.
 3. Eigendecompose `sigma`, giving `vals` and `vecs`.
 4. Raise `vals` to `1 - alpha` with [`geodesic_power`](@ref) and multiply them by `v ^ alpha`.
 5. Rebuild the matrix from `vecs` and the new eigenvalues, and return its symmetric part.

For every other target:

 1. Build the target matrix `tgt_mat` with [`shrinkage_target`](@ref).
 2. At `alpha = 1`, return a copy of `tgt_mat`.
 3. Factor `tgt_mat` by Cholesky into `chol`, with lower factor `L`. Refuse a factorisation that fails. A `Diagonal` target factors into `Diagonal` factors, so its whitening only rescales the rows and columns of `sigma`.
 4. Eigendecompose the whitened matrix `L \\ sigma / L'`, giving `vals` and `vecs`.
 5. Take the basis `basis = L * vecs`, raise `vals` to `1 - alpha` with [`geodesic_power`](@ref), rebuild the matrix from `basis` and the new eigenvalues, and return its symmetric part.

# Arguments

  - `tgt`: The target.
  - $(arg_dict[:sigrho])
  - `alpha`: Shrinkage intensity, strictly positive.

# Validation

  - The target matrix is positive definite. A `DomainError` is thrown otherwise.
  - `sigma` is positive semidefinite, as [`geodesic_power`](@ref) states.

# Returns

  - `sigma_alpha::MatNum`: A new matrix, the point on the geodesic.

# Related

  - [`geodesic_shrinkage!`](@ref)
  - [`GeodesicShrinkageCovariance`](@ref)
"""
function geodesic_point(::ScaledIdentityTarget, sigma::MatNum, alpha::Number)
    n = size(sigma, 1)
    v = LinearAlgebra.tr(sigma) / n
    if isone(alpha)
        return LinearAlgebra.Diagonal(fill(v, n))
    end
    vals, vecs = LinearAlgebra.eigen(LinearAlgebra.Symmetric(sigma))
    vals = geodesic_power(vals, 1 - alpha) * v^alpha
    sigma_alpha = vecs * LinearAlgebra.Diagonal(vals) * transpose(vecs)
    return (sigma_alpha + transpose(sigma_alpha)) / 2
end
function geodesic_point(tgt::GeodesicShrinkageTarget, sigma::MatNum, alpha::Number)
    tgt_mat = shrinkage_target(tgt, sigma)
    if isone(alpha)
        return copy(tgt_mat)
    end
    chol = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(tgt_mat); check = false)
    @argcheck(LinearAlgebra.issuccess(chol),
              DomainError(LinearAlgebra.diag(tgt_mat),
                          "the shrinkage target is not positive definite, so no geodesic reaches it. A target rule built from a matrix with a zero variance is the usual cause: an asset whose returns are constant has a zero variance. Remove the asset, or use `ScaledIdentityTarget`, whose scale is the average variance."))
    vals, vecs = LinearAlgebra.eigen(LinearAlgebra.Symmetric(chol.L \ sigma / chol.U))
    basis = chol.L * vecs
    sigma_alpha = basis *
                  LinearAlgebra.Diagonal(geodesic_power(vals, 1 - alpha)) *
                  transpose(basis)
    return (sigma_alpha + transpose(sigma_alpha)) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Repairs and shrinks the finite block of a covariance matrix in place, and leaves the frame of `NaN` around it alone.

An inner estimator that reads the gaps of a sample answers `NaN` for an asset it could not estimate. The repair and the shrinkage have no answer for a `NaN`, so they run on the block of the assets whose variance is finite, and the frame keeps its `NaN` rows and columns.

# Algorithm

 1. Take the block `blk` as the assets whose variance on the diagonal of `sigma` is finite. Return `sigma` when `blk` holds no asset.
 2. Refuse a non-finite entry inside the block with [`assert_finite_block`](@ref).
 3. Copy the block out as `block`, or take `sigma` itself when `blk` holds every asset.
 4. Repair `block` with [`posdef!`](@ref) under `ce.pdm`.
 5. When `ce.alpha` is positive, write the point at `ce.alpha` on the geodesic into `block`, with [`geodesic_point`](@ref).
 6. Write `block` back into `sigma`, and return `sigma`.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - $(arg_dict[:sigrho])

# Validation

  - The block of `sigma` is finite. An `IsNonFiniteError` is thrown otherwise.
  - Every check of [`geodesic_point`](@ref) applies to the block.

# Returns

  - `sigma::MatNum`: The input matrix, whose block was repaired and shrunk in place.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`geodesic_point`](@ref)
  - [`matrix_processing_block!`](@ref): the same rule for a matrix processing estimator.
"""
function geodesic_shrinkage!(ce::GeodesicShrinkageCovariance, sigma::MatNum)
    blk = isfinite.(LinearAlgebra.diag(sigma))
    if !any(blk)
        return sigma
    end
    assert_finite_block(view(sigma, blk, blk))
    whole = all(blk)
    block = whole ? sigma : sigma[blk, blk]
    posdef!(ce.pdm, block)
    if !iszero(ce.alpha)
        block .= geodesic_point(ce.tgt, block, ce.alpha)
    end
    if !whole
        sigma[blk, blk] = block
    end
    return sigma
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rescales a covariance matrix to its correlation matrix in place, and keeps the `NaN` of an asset whose variance is not finite.

`StatsBase.cov2cor!` writes one on the whole diagonal, which would turn the `NaN` variance of an asset outside the Coverage Universe into a unit correlation with itself. The entries off the diagonal of such an asset are already `NaN`, so only the diagonal needs it back.

# Algorithm

 1. Take the standard deviations `s` from the diagonal of `sigma`.
 2. Rescale `sigma` with `StatsBase.cov2cor!` and `s`.
 3. Write `s[i]` back onto the diagonal wherever `s[i]` is not finite.

# Arguments

  - $(arg_dict[:sigrho])

# Returns

  - `rho::MatNum`: The input matrix, rescaled in place to a correlation matrix.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
"""
function frame_cov2cor!(sigma::MatNum)
    s = sqrt.(LinearAlgebra.diag(sigma))
    StatsBase.cov2cor!(sigma, s)
    for i in findall(!isfinite, s)
        sigma[i, i] = s[i]
    end
    return sigma
end
"""
    Statistics.cov(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)
    Statistics.cor(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)

Computes the covariance matrix with `ce.ce`, and shrinks it towards `ce.tgt` along the geodesic of the positive definite matrices.

[`GeodesicShrinkageCovariance`](@ref) states the mathematics. The estimator forwards the sample and its keywords to `ce.ce` untouched, so an active mask reaches an inner estimator that reads one, and a gap is the inner estimator's to keep or to refuse.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute `sigma` with `Statistics.cov(library_covariance_estimator(ce.ce), X; kwargs...)`.
 3. Copy `sigma` into a new `Matrix`, because step 4 writes in place and the inner estimator can return an immutable matrix.
 4. Repair and shrink the finite block of `sigma` with [`geodesic_shrinkage!`](@ref).
 5. `cov` returns `sigma`. `cor` rescales it to a correlation matrix with [`frame_cov2cor!`](@ref) and returns it, so the correlation is the one of the shrunk covariance.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to `ce.ce`.

# Validation

  - $(val_dict[:dims])
  - Every check of [`geodesic_shrinkage!`](@ref) applies.

# Returns

  - `sigma::MatNum`: The shrunk covariance matrix, or `rho`, its correlation matrix.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`geodesic_shrinkage!`](@ref)
"""
function Statistics.cov(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
    X = dims_oriented(dims, X)
    sigma = Matrix(Statistics.cov(library_covariance_estimator(ce.ce), X; kwargs...))
    return geodesic_shrinkage!(ce, sigma)
end
function Statistics.cor(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
    sigma = Statistics.cov(ce, X; dims = dims, kwargs...)
    return frame_cov2cor!(sigma)
end
"""
    Statistics.cov(ce::GeodesicShrinkageCovariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...) -> MatNum
    Statistics.cor(ce::GeodesicShrinkageCovariance, X::MatNum,
                   pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...) -> MatNum

Forwards the Asset Panel to `ce.ce`, then repairs and shrinks the finite block of the frame it gets back.

This is the estimator's override of the reduce-and-expand root. The inner estimator owns the reduction, because it alone knows whether it reads the gaps of the sample, and this method owns the shrinkage. An asset outside the Coverage Universe keeps its `NaN` row and column.

# Algorithm

 1. Orient `X` to `observations × assets` with [`dims_oriented`](@ref).
 2. Compute `sigma` with the inner estimator, forwarding `pnl` as its third positional argument.
 3. Copy `sigma` into a new `Matrix`, because step 4 writes in place and the inner estimator can return an immutable matrix.
 4. Repair and shrink the finite block of `sigma` with [`geodesic_shrinkage!`](@ref).
 5. `cov` returns `sigma`. `cor` rescales it to a correlation matrix with [`frame_cov2cor!`](@ref) and returns it.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to `ce.ce`.

# Validation

  - $(val_dict[:dims])
  - Every check of [`geodesic_shrinkage!`](@ref) applies.

# Returns

  - `sigma::MatNum`: The shrunk covariance matrix, or `rho`, its correlation matrix, on the full asset universe.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`geodesic_shrinkage!`](@ref)
  - [`coverage_reduction`](@ref)
"""
function Statistics.cov(ce::GeodesicShrinkageCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
    sigma = Matrix(Statistics.cov(library_covariance_estimator(ce.ce), X, pnl; dims = 1,
                                  kwargs...))
    return geodesic_shrinkage!(ce, sigma)
end
function Statistics.cor(ce::GeodesicShrinkageCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    sigma = Statistics.cov(ce, X, pnl; dims = dims, kwargs...)
    return frame_cov2cor!(sigma)
end
"""
    gap_fill_value(ce::GeodesicShrinkageCovariance) -> Number

Answer what `ce.ce` answers, because the estimator forwards the sample and its keywords untouched.

The shrinkage reads the matrix `ce.ce` returns and no cell of the sample, so a gap is the inner estimator's to keep or to lose.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.

# Returns

  - `fv::Number`: [`gap_fill_value`](@ref) of `ce.ce`.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`gap_fill_value`](@ref)
"""
function gap_fill_value(ce::GeodesicShrinkageCovariance)
    return gap_fill_value(ce.ce)
end
"""
    partial_fit!(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)
    partial_fit!(ce::GeodesicShrinkageCovariance, x::VecNum; kwargs...)

Folds observations into a [`GeodesicShrinkageCovariance`](@ref) by forwarding them to `ce.ce`.

The shrinkage reads the matrix `ce.ce` returns and no observation, so the estimator keeps no state of its own and folds exactly when `ce.ce` does. The read-out applies the shrinkage to the folded matrix.

# Algorithm

 1. Rebind `ce.ce` to the estimator [`partial_fit!`](@ref) gives, with `Accessors.@reset`.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - $(arg_dict[:X])
  - `x`: One observation, whose entries are the assets.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments, forwarded to `ce.ce`.

# Returns

  - `ce`: The estimator, with `ce.ce` rebound to the estimator carrying the state after the last observation.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)
    return Accessors.@reset ce.ce = partial_fit!(ce.ce, X; dims = dims, kwargs...)
end
function partial_fit!(ce::GeodesicShrinkageCovariance, x::VecNum; kwargs...)
    return Accessors.@reset ce.ce = partial_fit!(ce.ce, x; kwargs...)
end
"""
    Statistics.cov(ce::GeodesicShrinkageCovariance; kwargs...)
    Statistics.cor(ce::GeodesicShrinkageCovariance; kwargs...)

Reads the shrunk covariance, or its correlation, of a folded [`GeodesicShrinkageCovariance`](@ref).

`ce.ce` answers its own folded matrix, and the shrinkage runs on it as it runs on a batch fit, so this method answers what a batch fit over the same observations answers.

# Algorithm

 1. Read the folded matrix `sigma` of `ce.ce` with the one-argument `Statistics.cov`.
 2. Copy `sigma` into a new `Matrix`, because step 3 writes in place and the inner estimator can return an immutable matrix.
 3. Repair and shrink the finite block of `sigma` with [`geodesic_shrinkage!`](@ref).
 4. `cov` returns `sigma`. `cor` rescales it to a correlation matrix with [`frame_cov2cor!`](@ref) and returns it.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.
  - `kwargs...`: Additional keyword arguments, forwarded to `ce.ce`.

# Validation

  - `ce.ce` carries a partial-fit state. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: The shrunk covariance matrix, or `rho`, its correlation matrix.

# Related

  - [`GeodesicShrinkageCovariance`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.cov(ce::GeodesicShrinkageCovariance; kwargs...)
    sigma = Matrix(Statistics.cov(ce.ce; kwargs...))
    return geodesic_shrinkage!(ce, sigma)
end
function Statistics.cor(ce::GeodesicShrinkageCovariance; kwargs...)
    sigma = Statistics.cov(ce; kwargs...)
    return frame_cov2cor!(sigma)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`GeodesicShrinkageCovariance`](@ref) method of [`supports_partial_fit`](@ref).

The estimator folds by forwarding to `ce.ce`, so it folds exactly when `ce.ce` does.

# Arguments

  - `ce`: The geodesic shrinkage covariance estimator.

# Returns

  - `folds::Bool`: `true` when [`partial_fit!`](@ref) folds `ce.ce`.

# Related

  - [`supports_partial_fit`](@ref)
  - [`GeodesicShrinkageCovariance`](@ref)
"""
function supports_partial_fit(ce::GeodesicShrinkageCovariance)
    return supports_partial_fit(ce.ce)
end

export IdentityTarget, ScaledIdentityTarget, CommonCovarianceTarget,
       ConstantCorrelationTarget, GeodesicShrinkageCovariance
