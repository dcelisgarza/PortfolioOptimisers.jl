"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that find the shrinkage intensity of a robust scatter from the data.

A shrunk robust scatter is a blend of the scatter of the data and of a target. The intensity is the weight of the target, in `[0, 1]`. A member of this family finds it when the fit runs. The slot also takes a number, the intensity itself.

# Interfaces

A subtype implements one method for [`TylerShrinkage`](@ref):

  - `PortfolioOptimisers.tyler_intensity(rule, alg::AbstractTylerShrinkageAlgorithm, W::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Number`: Returns the intensity that the form `alg` reads, for the centred data `W`, `observations × assets`, in the coordinates that turn the target into the identity matrix.

A rule that [`MCDShrinkage`](@ref) can also hold subtypes [`AbstractSubsetShrinkageIntensity`](@ref).

# Related

  - [`AbstractSubsetShrinkageIntensity`](@ref)
  - [`OracleIntensity`](@ref)
  - [`AsymptoticIntensity`](@ref)
  - [`TylerShrinkage`](@ref)
"""
abstract type AbstractRobustShrinkageIntensity <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the shrinkage rules that can also read the scatter of one subset of the observations.

[`MCDShrinkage`](@ref) takes these rules, because the minimum covariance determinant shrinks the scatter of a subset. [`TylerShrinkage`](@ref) takes them too.

# Interfaces

A subtype implements the method of [`AbstractRobustShrinkageIntensity`](@ref), and one more:

  - `PortfolioOptimisers.mcd_intensity(rule, W::MatNum, idx::AbstractVector{<:Integer}, c::Number, fixed::Bool) -> Number`: Returns the intensity for the observations `idx` of the data `W`, in the coordinates that turn the target into the identity matrix. `c` is the factor of the scatter of the subset. When `fixed` is `true`, the centre is zero.

# Related

  - [`ConditionNumberIntensity`](@ref)
  - [`SphericityIntensity`](@ref)
  - [`MCDShrinkage`](@ref)
"""
abstract type AbstractSubsetShrinkageIntensity <: AbstractRobustShrinkageIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Takes the smallest shrinkage intensity at which the condition number of the scatter is not above a bound.

Section 2.4 of [boudt2020](@cite) gives the rule for the shrunk minimum covariance determinant. The condition number is the one of the scatter in the coordinates that turn the target into the identity matrix. A scatter that meets the bound gets no shrinkage. For a subset the rule has a closed form. This library extends the rule to [`TylerShrinkage`](@ref), where a bisection over the fits of the shape finds the intensity, at the pilot centre of the fit.

# Mathematical definition

```math
\\begin{align}
\\rho &= \\max\\left(0, \\frac{\\lambda_{\\max} - \\kappa \\lambda_{\\min}}{\\kappa - 1 + \\lambda_{\\max} - \\kappa \\lambda_{\\min}}\\right)\\,.
\\end{align}
```

Where:

  - ``\\rho``: The shrinkage intensity.
  - ``\\kappa``: The bound of the condition number, `max_cond`.
  - ``\\lambda_{\\min}``, ``\\lambda_{\\max}``: The smallest and the largest eigenvalue of the scatter of the subset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConditionNumberIntensity(;
        max_cond::Number = 50
    ) -> ConditionNumberIntensity

Keywords correspond to the struct's fields.

## Validation

  - `max_cond > 1`.

# Examples

```jldoctest
julia> ConditionNumberIntensity()
ConditionNumberIntensity
  max_cond ┴ Int64: 50
```

# Related

  - [`AbstractSubsetShrinkageIntensity`](@ref)
  - [`MCDShrinkage`](@ref)
  - [`TylerShrinkage`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 2.4, Equation 17.
"""
@concrete struct ConditionNumberIntensity <: AbstractSubsetShrinkageIntensity
    """
    $(field_dict[:max_cond])
    """
    max_cond
    function ConditionNumberIntensity(max_cond::Number)
        @argcheck(max_cond > one(max_cond),
                  DomainError(max_cond, "`max_cond` must be larger than 1, got $max_cond"))
        return new{typeof(max_cond)}(max_cond)
    end
end
function ConditionNumberIntensity(; max_cond::Number = 50)::ConditionNumberIntensity
    return ConditionNumberIntensity(max_cond)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the shrinkage intensity of the smallest mean squared error, from a robust estimate of the sphericity of the scatter.

The sphericity ``\\gamma`` is the mean of the squares of the eigenvalues over the square of their mean. It is one when the scatter is a multiple of the target. [ollila2019](@cite) estimates it from the spatial signs of the data, so an outlier has the influence of one direction and not of its size. This library takes the signs about the coordinatewise median, and that work takes them about the spatial median.

For [`MCDShrinkage`](@ref) the intensity is the optimum at the normal model, with the count of the subset. The signs come from every observation, because a subset of the nearest observations is more spherical than the data. For [`TylerShrinkage`](@ref) it is the optimum of [ollila2021](@cite) for the shape of Tyler's M-estimator, which holds over the elliptical distributions.

# Mathematical definition

```math
\\begin{align}
\\hat{\\gamma} &= \\frac{n}{n - 1} \\left(p \\, \\mathrm{tr}(\\mathbf{S}^2) - \\frac{p}{n}\\right)\\,, \\\\
\\rho_{\\mathrm{MCD}} &= \\frac{(\\hat{\\gamma} + p)/(h - 1)}{\\hat{\\gamma} - 1 + (\\hat{\\gamma} + p)/(h - 1)}\\,, \\\\
\\rho_{\\mathrm{Tyler}} &= 1 - \\frac{n (\\hat{\\gamma} - 1)}{n (\\hat{\\gamma} - 1)(1 - 1/n) + (2\\hat{\\gamma} + p)(p - 1)/(p + 2)}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{S}``: The spatial sign covariance, the mean of the product moments of the unit vectors of the observations.
  - ``\\hat{\\gamma}``: The sphericity, clamped into ``[1, p]``.
  - ``n``: The number of observations, or the effective count of the weights.
  - ``h``: The number of observations of the subset. A fixed centre gives ``h`` in place of ``h - 1``.
  - ``p``: The number of assets.

# Examples

```jldoctest
julia> SphericityIntensity()
SphericityIntensity()
```

# Related

  - [`AbstractSubsetShrinkageIntensity`](@ref)
  - [`MCDShrinkage`](@ref)
  - [`TylerShrinkage`](@ref)

# References

  - $(ref_dict[:ollila2019]) Equations 23 and 25, and Section IV-A.
  - $(ref_dict[:ollila2021]) Corollary 2.
"""
struct SphericityIntensity <: AbstractSubsetShrinkageIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Takes the oracle intensity of the work that gives the form of the shrunk Tyler shape, with an estimate in place of the unknown scatter.

Each form of [`TylerShrinkage`](@ref) has its own oracle, the intensity of the smallest mean squared error when the weights of the fixed point read the true scatter.

  - [`TraceNormalisedShrinkage`](@ref): Equation 13 of [chen2011](@cite), with the normalised sample covariance of its Equation 14.
  - [`PenalisedShrinkage`](@ref): Theorem 4 of [ollila2014](@cite) in the real case, with the Tyler shape when the sample gives one. Otherwise, the pilot is the penalised shape at the middle of the interval of the intensities at which it exists. That work leaves the choice of that pilot open.

# Mathematical definition

```math
\\begin{align}
\\rho_{\\mathrm{trace}} &= \\frac{p^2 + (1 - 2/p) \\, \\mathrm{tr}(\\mathbf{R}^2)}{p^2 - np - 2n + (n + 1 + 2(n - 1)/p) \\, \\mathrm{tr}(\\mathbf{R}^2)}\\,, \\\\
\\rho_{\\mathrm{penalised}} &= \\frac{p - 2 + p \\, \\mathrm{tr}(\\mathbf{V})}{p - 2 + p \\, \\mathrm{tr}(\\mathbf{V}) + n (p + 2) (\\mathrm{tr}(\\mathbf{V}^{-2})/p - 1)}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{R}``: ``p`` times the spatial sign covariance of the data.
  - ``\\mathbf{V}``: The pilot shape, with ``\\mathrm{tr}(\\mathbf{V}^{-1}) = p``.
  - ``n``: The number of observations, or the effective count of the weights.
  - ``p``: The number of assets.

The rule clamps the intensity into ``[0, 1]``.

# Examples

```jldoctest
julia> OracleIntensity()
OracleIntensity()
```

# Related

  - [`AbstractRobustShrinkageIntensity`](@ref)
  - [`TylerShrinkage`](@ref)

# References

  - $(ref_dict[:chen2011]) Theorem 2, Equations 13 and 14.
  - $(ref_dict[:ollila2014]) Section V-A, Theorem 4.
"""
struct OracleIntensity <: AbstractRobustShrinkageIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Takes the intensity of the smallest mean squared error when the numbers of observations and of assets grow together.

Proposition 2 of [couillet2014](@cite) gives an equation in the intensity for each form of [`TylerShrinkage`](@ref), and shows that both forms then reach the same error. The rule finds the smallest intensity that reaches the value of the equation by a bisection over the fits of the shape, at the pilot centre of the fit.

# Mathematical definition

```math
\\begin{align}
\\frac{\\rho \\, a(\\rho)}{1 - \\rho + \\rho \\, a(\\rho)} &= r\\,, & a(\\rho) &= \\frac{1}{n} \\sum_{t=1}^{n} \\frac{\\boldsymbol{x}_t^\\intercal \\mathbf{V}(\\rho)^{-1} \\boldsymbol{x}_t}{\\lVert \\boldsymbol{x}_t \\rVert^2}\\,, \\\\
\\frac{\\rho}{\\mathrm{tr}(\\mathbf{V}(\\rho))/p} &= r\\,, & r &= \\frac{p/n}{\\mathrm{tr}(\\mathbf{R}^2)/p - 1}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}(\\rho)``: The shape at the intensity ``\\rho``. The first equation holds for [`TraceNormalisedShrinkage`](@ref), and the second for [`PenalisedShrinkage`](@ref).
  - ``\\mathbf{R}``: ``p`` times the spatial sign covariance of the data.
  - ``n``: The number of observations, or the effective count of the weights.
  - ``p``: The number of assets.

A value ``r`` of one or more gives the intensity one.

# Examples

```jldoctest
julia> AsymptoticIntensity()
AsymptoticIntensity()
```

# Related

  - [`AbstractRobustShrinkageIntensity`](@ref)
  - [`TylerShrinkage`](@ref)

# References

  - $(ref_dict[:couillet2014]) Propositions 1 and 2.
"""
struct AsymptoticIntensity <: AbstractRobustShrinkageIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the forms of the fixed point of a shrunk Tyler shape.

The two members give the same set of shapes. They read the intensity on different scales. The shape of [`TraceNormalisedShrinkage`](@ref) at the intensity ``\\rho`` is the shape of [`PenalisedShrinkage`](@ref) at the intensity ``1 - (1 - \\rho)/t``, where ``t`` is the mean of the diagonal of the blend before its normalisation.

# Interfaces

A subtype implements one method, and a method of each rule of the intensity that it takes:

  - `PortfolioOptimisers.shrink_shape!(alg, V::MatNum, rho::Number) -> MatNum`: Blends the step `V` of the shape equation with the identity matrix in place.

# Related

  - [`TraceNormalisedShrinkage`](@ref)
  - [`PenalisedShrinkage`](@ref)
  - [`TylerShrinkage`](@ref)
"""
abstract type AbstractTylerShrinkageAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Blends each step of the shape equation with the target, and scales the blend to a fixed trace.

[chen2011](@cite) shows that the iteration has one limit for each intensity in `(0, 1)`, for any number of observations.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\mathbf{V}} &= (1 - \\rho) \\frac{p}{n} \\sum_{t=1}^{n} \\frac{\\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal}{\\boldsymbol{x}_t^\\intercal \\mathbf{V}^{-1} \\boldsymbol{x}_t} + \\rho \\, \\mathbf{I}\\,, \\\\
\\mathbf{V} &= p \\, \\tilde{\\mathbf{V}} / \\mathrm{tr}(\\tilde{\\mathbf{V}})\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}``: The shape, in the coordinates that turn the target into the identity matrix.
  - ``\\rho``: The shrinkage intensity.
  - ``\\boldsymbol{x}_t``: The centred observation ``t``, in the same coordinates.
  - ``n``: The number of observations.
  - ``p``: The number of assets.

# Examples

```jldoctest
julia> TraceNormalisedShrinkage()
TraceNormalisedShrinkage()
```

# Related

  - [`AbstractTylerShrinkageAlgorithm`](@ref)
  - [`PenalisedShrinkage`](@ref)

# References

  - $(ref_dict[:chen2011]) Equations 8 and 9, Theorem 1.
"""
struct TraceNormalisedShrinkage <: AbstractTylerShrinkageAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Blends each step of the shape equation with the target, with no normalisation, so the shape is the minimum of a penalised cost.

[ollila2014](@cite) derives the form from a penalty on the trace of the inverse shape. Its Theorem 3 shows that the shape exists in general position when the intensity is above ``1 - n/p``. [sun2014](@cite) derives the same form from a Kullback-Leibler penalty.

# Mathematical definition

```math
\\begin{align}
\\mathbf{V} &= (1 - \\rho) \\frac{p}{n} \\sum_{t=1}^{n} \\frac{\\boldsymbol{x}_t \\boldsymbol{x}_t^\\intercal}{\\boldsymbol{x}_t^\\intercal \\mathbf{V}^{-1} \\boldsymbol{x}_t} + \\rho \\, \\mathbf{I}\\,.
\\end{align}
```

Where:

  - ``\\mathbf{V}``: The shape, in the coordinates that turn the target into the identity matrix.
  - ``\\rho``: The shrinkage intensity.
  - ``\\boldsymbol{x}_t``: The centred observation ``t``, in the same coordinates.
  - ``n``: The number of observations.
  - ``p``: The number of assets.

# Examples

```jldoctest
julia> PenalisedShrinkage()
PenalisedShrinkage()
```

# Related

  - [`AbstractTylerShrinkageAlgorithm`](@ref)
  - [`TraceNormalisedShrinkage`](@ref)

# References

  - $(ref_dict[:ollila2014]) Section V, Theorem 3.
  - $(ref_dict[:sun2014]) Section III-B, Equation 21.
"""
struct PenalisedShrinkage <: AbstractTylerShrinkageAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the scatter of each subset of an [`MCDCovariance`](@ref) to a target, so the fit exists with fewer observations than assets.

[`MCDCovariance`](@ref) holds it in the field `shr`. The fit is then the minimum regularised covariance determinant of [boudt2020](@cite): the subset whose shrunk scatter has the smallest determinant. The target does not change in the search.

The target `tgt` is a covariance matrix in the units of the data. A rule of [`CovarianceShrinkageTarget`](@ref) reads a robust pilot: the robust variances of the assets on the diagonal, and the sine transform of the Kendall correlations off it. The default [`DiagonalTarget`](@ref) is the diagonal of the robust variances, the identity target of [boudt2020](@cite) on its standardised data. [`ConstantCorrelationTarget`](@ref) is its equicorrelation target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MCDShrinkage(;
        tgt::CovarianceShrinkageTarget = DiagonalTarget(),
        intensity::Union{Number, AbstractSubsetShrinkageIntensity} = ConditionNumberIntensity()
    ) -> MCDShrinkage

Keywords correspond to the struct's fields.

## Validation

  - A matrix `tgt` is square, finite, symmetric and positive definite, as [`assert_shrinkage_target`](@ref) states.
  - A number `intensity` is in `[0, 1]`.

## View parameters

[`port_opt_view`](@ref) slices a matrix `tgt` on both axes, and keeps a target rule as it is.

# Examples

```jldoctest
julia> MCDShrinkage(; intensity = 0.2)
MCDShrinkage
        tgt ┼ DiagonalTarget()
  intensity ┴ Float64: 0.2
```

# Related

  - [`MCDCovariance`](@ref)
  - [`ConditionNumberIntensity`](@ref)
  - [`SphericityIntensity`](@ref)
  - [`CovarianceShrinkageTarget`](@ref)
  - [`TylerShrinkage`](@ref)

# References

  - $(ref_dict[:boudt2020]) Sections 2.2 and 2.4.
"""
@concrete struct MCDShrinkage <: AbstractEstimator
    """
    $(field_dict[:mutgt]) A rule of [`CovarianceShrinkageTarget`](@ref), or a fixed positive definite matrix in the units of the data.
    """
    tgt
    """
    $(field_dict[:rob_intensity])
    """
    intensity
    function MCDShrinkage(tgt::CovarianceShrinkageTarget,
                          intensity::Union{Number, AbstractSubsetShrinkageIntensity})
        assert_shrinkage_target(tgt)
        assert_closed_unit_interval(intensity, :intensity)
        return new{typeof(tgt), typeof(intensity)}(tgt, intensity)
    end
end
function MCDShrinkage(; tgt::CovarianceShrinkageTarget = DiagonalTarget(),
                      intensity::Union{Number, AbstractSubsetShrinkageIntensity} = ConditionNumberIntensity())::MCDShrinkage
    return MCDShrinkage(tgt, intensity)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the shape of a [`TylerCovariance`](@ref) to a target, so the fit exists with fewer observations than assets.

[`TylerCovariance`](@ref) holds it in the field `shr`. The form `alg` states the fixed point, and `intensity` the weight of the target in it. The number, or the rule, reads the scale of that form, see [`AbstractTylerShrinkageAlgorithm`](@ref).

The target `tgt` is a covariance matrix in the units of the data, and its scale has no effect. A rule of [`CovarianceShrinkageTarget`](@ref) reads a robust pilot: the squares of the median absolute deviations of the assets on the diagonal, and the correlations of the spatial sign covariance off it. The default [`DiagonalTarget`](@ref) does not depend on the units of the assets. [`ScaledIdentityTarget`](@ref) gives the target of [chen2011](@cite) and of [ollila2014](@cite), which has one variance for each asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TylerShrinkage(;
        alg::AbstractTylerShrinkageAlgorithm = TraceNormalisedShrinkage(),
        tgt::CovarianceShrinkageTarget = DiagonalTarget(),
        intensity::Union{Number, AbstractRobustShrinkageIntensity} = OracleIntensity()
    ) -> TylerShrinkage

Keywords correspond to the struct's fields.

## Validation

  - A matrix `tgt` is square, finite, symmetric and positive definite, as [`assert_shrinkage_target`](@ref) states.
  - A number `intensity` is in `[0, 1]`.

## View parameters

[`port_opt_view`](@ref) slices a matrix `tgt` on both axes, and keeps a target rule as it is.

# Examples

```jldoctest
julia> TylerShrinkage()
TylerShrinkage
        alg ┼ TraceNormalisedShrinkage()
        tgt ┼ DiagonalTarget()
  intensity ┴ OracleIntensity()
```

# Related

  - [`TylerCovariance`](@ref)
  - [`TraceNormalisedShrinkage`](@ref)
  - [`PenalisedShrinkage`](@ref)
  - [`OracleIntensity`](@ref)
  - [`AsymptoticIntensity`](@ref)
  - [`SphericityIntensity`](@ref)
  - [`ConditionNumberIntensity`](@ref)
  - [`CovarianceShrinkageTarget`](@ref)
  - [`MCDShrinkage`](@ref)

# References

  - $(ref_dict[:chen2011]) Section III.
  - $(ref_dict[:ollila2014]) Section V.
"""
@concrete struct TylerShrinkage <: AbstractEstimator
    """
    $(field_dict[:tyler_shr_alg])
    """
    alg
    """
    $(field_dict[:mutgt]) A rule of [`CovarianceShrinkageTarget`](@ref), or a fixed positive definite matrix in the units of the data.
    """
    tgt
    """
    $(field_dict[:rob_intensity])
    """
    intensity
    function TylerShrinkage(alg::AbstractTylerShrinkageAlgorithm,
                            tgt::CovarianceShrinkageTarget,
                            intensity::Union{Number, AbstractRobustShrinkageIntensity})
        assert_shrinkage_target(tgt)
        assert_closed_unit_interval(intensity, :intensity)
        return new{typeof(alg), typeof(tgt), typeof(intensity)}(alg, tgt, intensity)
    end
end
function TylerShrinkage(; alg::AbstractTylerShrinkageAlgorithm = TraceNormalisedShrinkage(),
                        tgt::CovarianceShrinkageTarget = DiagonalTarget(),
                        intensity::Union{Number, AbstractRobustShrinkageIntensity} = OracleIntensity())::TylerShrinkage
    return TylerShrinkage(alg, tgt, intensity)
end
"""
    port_opt_view(shr::Union{MCDShrinkage, TylerShrinkage}, i, args...)

Restricts the shrinkage of a robust covariance estimator to the assets `i`.

A fixed target is an asset-by-asset matrix, so the method slices it on both axes. [`shrinkage_target`](@ref) builds the target of a rule from the pilot of the data, so a rule needs no slice.

# Arguments

  - `shr`: The shrinkage.
  - `i`: Indices of the selected assets.
  - `args...`: Not read.

# Returns

  - `shr::Union{MCDShrinkage, TylerShrinkage}`: The shrinkage restricted to the assets `i`.

# Related

  - [`MCDShrinkage`](@ref)
  - [`TylerShrinkage`](@ref)
  - [`nothing_scalar_array_view`](@ref)
"""
function port_opt_view(shr::Union{MCDShrinkage, TylerShrinkage}, i, args...)
    if isa(shr.tgt, AbstractMatrix)
        return rebuild_struct(shr, (; tgt = nothing_scalar_array_view(shr.tgt, i)))
    end
    return shr
end
"""
    robust_target(tgt::CovarianceShrinkageTarget, pilot::MatNum) -> MatNum

Builds the target of a shrunk robust scatter from the robust pilot, and refuses a target that is not positive definite.

# Arguments

  - `tgt`: The target rule, or a fixed matrix.
  - `pilot`: The robust pilot covariance, in the units of the data.

# Validation

  - The target is positive definite. Otherwise, the function raises a `DomainError`.

# Returns

  - `target::MatNum`: The target.

# Related

  - [`shrinkage_target`](@ref)
  - [`MCDShrinkage`](@ref)
  - [`TylerShrinkage`](@ref)
"""
function robust_target(tgt::CovarianceShrinkageTarget, pilot::MatNum)
    T = shrinkage_target(tgt, pilot)
    @argcheck(LinearAlgebra.isposdef(LinearAlgebra.Symmetric(T)),
              DomainError(tgt,
                          "the shrinkage target that the rule builds from the robust pilot is not positive definite. Use `DiagonalTarget` or `ScaledIdentityTarget`"))
    return T
end
"""
    symmetric_root(T::LinearAlgebra.Diagonal) -> LinearAlgebra.Diagonal
    symmetric_root(T::MatNum) -> AbstractMatrix

Returns the symmetric square root of a positive definite matrix.

The data times the inverse of the root have the identity matrix as their target. The symmetric root does not depend on the order of the assets.

# Arguments

  - `T`: The target.

# Returns

  - `root::AbstractMatrix`: The symmetric square root.

# Related

  - [`robust_target`](@ref)
"""
function symmetric_root(T::LinearAlgebra.Diagonal)
    return sqrt(T)
end
function symmetric_root(T::MatNum)
    return sqrt(LinearAlgebra.Symmetric(T))
end
"""
    spatial_sign_scatter(W::MatNum, wt::VecNum) -> Matrix

Returns the number of assets times the weighted spatial sign covariance of the centred data `W`.

The unit vector of each observation has the weight of the observation. An observation at the centre has no direction and leaves the sum. The trace of the result is the number of assets.

# Arguments

  - `W`: The centred data, `observations × assets`.
  - `wt`: The weight of each observation.

# Returns

  - `R::Matrix`: The scatter, exactly symmetric.

# Related

  - [`sphericity`](@ref)
  - [`OracleIntensity`](@ref)
"""
function spatial_sign_scatter(W::MatNum, wt::VecNum)
    d2 = vec(sum(abs2, W; dims = 2))
    return tyler_shape_step(W, d2, wt, away_from_centre(d2))
end
"""
    sphericity(R::MatNum, n::Number) -> Number

Returns the estimate of the sphericity of a scatter from its [`spatial_sign_scatter`](@ref), clamped into ``[1, p]``.

# Arguments

  - `R`: The spatial sign scatter, of trace ``p``.
  - `n`: The number of observations, or the effective count of the weights.

# Returns

  - `gamma::Number`: The sphericity.

# Related

  - [`SphericityIntensity`](@ref)

# References

  - $(ref_dict[:ollila2019]) Section IV-A.
"""
function sphericity(R::MatNum, n::Number)
    p = size(R, 1)
    return clamp(n / (n - 1) * (sum(abs2, R) / p - p / n), one(eltype(R)), p)
end
"""
    robust_outliers(intensity::Nothing, d2::VecNum, p::Integer, q::Number) -> BitVector
    robust_outliers(intensity::Number, d2::VecNum, p::Integer, q::Number) -> BitVector

Flags the observations of a robust fit whose squared distance is above the cut-off of the probability `q`.

The method of a fit with no shrinkage compares each squared distance with the quantile `q` of the ``\\chi^2_p`` distribution. The distances of a shrunk scatter do not have that distribution, because the shrinkage changes the metric. Section 6 of [boudt2020](@cite) states that the approximation fails when ``p`` is large in relation to ``n``. A squared distance in another metric is a weighted sum of ``\\chi^2_1`` variables, and its cube root is almost normal. So the method of a shrunk fit compares each cube root with the quantile `q` of the normal distribution that has the median and the robust scale of the cube roots. This cut-off is a rule of this library.

# Arguments

  - `intensity`: The shrinkage intensity of the fit, or `nothing`.
  - `d2`: The squared distance of each observation.
  - `p`: The number of assets.
  - `q`: The probability of the cut-off.

# Returns

  - `outlier::BitVector`: The flags.

# Related

  - [`RobustCovarianceResult`](@ref)
  - [`robust_column_scale`](@ref)

# References

  - $(ref_dict[:boudt2020]) Section 6.
"""
function robust_outliers(::Nothing, d2::VecNum, p::Integer, q::Number)
    return d2 .> Distributions.quantile(Distributions.Chisq(p), q)
end
function robust_outliers(::Number, d2::VecNum, ::Integer, q::Number)
    y = cbrt.(d2)
    y .-= Statistics.median(y)
    # The normal distribution is symmetric, so this is its quantile `q`.
    z = -Distributions.cquantile(Distributions.Normal(), q)
    return y .> z * robust_column_scale(y)
end

export ConditionNumberIntensity, SphericityIntensity, OracleIntensity, AsymptoticIntensity,
       TraceNormalisedShrinkage, PenalisedShrinkage, MCDShrinkage, TylerShrinkage
public AbstractRobustShrinkageIntensity, AbstractSubsetShrinkageIntensity,
       AbstractTylerShrinkageAlgorithm
