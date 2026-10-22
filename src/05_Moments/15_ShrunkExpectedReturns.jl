"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrunk expected returns estimators.

A subtype computes the sample expected returns and moves them toward a shrinkage target.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`EquilibriumExpectedReturns`](@ref)
  - [`ExcessExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage algorithms for expected returns estimation.

A subtype holds a shrinkage target and selects the formula that combines the sample expected returns with that target.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage targets used in expected returns estimation.

A subtype selects a method of [`target_mean`](@ref), which returns a vector whose entries all hold the same value.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
  - [`target_mean`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the grand mean of the sample expected returns.

Every element of the target holds the same value, so a shrinkage estimator pulls each asset toward the average of the whole universe. The three targets are each a multiple of the vector of ones, and only the multiplier separates them.

# Mathematical definition

```math
\\begin{align}
b_j &= \\bar{\\mu} = \\frac{1}{N} \\sum_{i=1}^{N} \\hat{\\mu}_i\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:N])

This target is the only one of the three that does not read the covariance matrix.

Equation 4.141 of [meucci2005](@cite) states this target as the scenario-dependent grand mean. Section 3.3.1 of [fengpalomar2016](@cite) lists it and credits [meucci2005](@cite) with it. The method computes it from the same sample as the mean that it shrinks.

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects does these steps.

 1. Take the unweighted mean of `mu` as `val`.
 2. Return the constant range that repeats `val` `length(mu)` times.

# Constructors

    GrandMean() -> GrandMean

# Examples

```jldoctest
julia> GrandMean()
GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:meucci2005]) Section 4.4, Equation 4.141.
"""
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the inverse-covariance-weighted mean of the sample expected returns.

The weights come from the inverse covariance matrix. Under a diagonal covariance matrix, each weight is proportional to the reciprocal of the variance of its asset, so an asset with a larger variance gets a smaller weight. The name says volatility, but the form reads the whole inverse covariance matrix. An off-diagonal entry of the covariance matrix also changes the target.

# Mathematical definition

```math
\\begin{align}
b_j &= \\bar{\\mu}_{\\mathrm{vol}} = \\frac{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}}{\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{1}}\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:Sigma_hat])
  - ``\\boldsymbol{1}``: ``N \\times 1`` vector of ones.
  - $(math_dict[:N])

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects does these steps.

 1. When `isigma` is `nothing`, solve `sigma \\ LinearAlgebra.I` as `isigma`. A caller that already has the inverse passes it, so the solve runs at most once for each estimate.
 2. When `mu` has one row, flatten it with `vec`, so that the product `isigma * mu` is defined.
 3. Divide the sum of `isigma * mu` by the sum of `isigma` as `val`. The sum of the matrix-vector product is the numerator ``\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}``. The sum of the matrix is the denominator ``\\boldsymbol{1}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{1}``.
 4. Return the constant range that repeats `val` `length(mu)` times.

This target is the only one of the three that inverts the covariance matrix, so a singular covariance matrix raises a `LinearAlgebra.SingularException` here and nowhere else in [`target_mean`](@ref).

Equation 14 of [jorion1986](@cite) defines this quotient as the grand mean ``Y_0``. It notes that the quotient is the average return of the minimum variance portfolio. Equation 4.142 of [meucci2005](@cite) states it as the volatility-weighted grand mean and credits [jorion1986](@cite) with it. The method computes it from the same sample as the mean that it shrinks.

# Constructors

    VolatilityWeighted() -> VolatilityWeighted

# Examples

```jldoctest
julia> VolatilityWeighted()
VolatilityWeighted()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:jorion1986]) Section IV, Equation 14.
  - $(ref_dict[:meucci2005]) Section 4.4, Equation 4.142.
"""
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Fills the shrinkage target with the trace of the covariance matrix divided by the number of observations.

Every element of the target holds the same value. The target reads only the covariance matrix, and not the sample expected returns. A shift of the mean of every asset by one constant moves the other two targets by that constant, and leaves this target as it is.

# Mathematical definition

```math
\\begin{align}
b_j &= \\frac{\\mathrm{tr}(\\hat{\\mathbf{\\Sigma}})}{T}\\,, \\quad j = 1, \\ldots, N\\,.
\\end{align}
```

Where:

  - $(math_dict[:b_j_shrink_tgt])
  - ``\\mathrm{tr}(\\cdot)``: Matrix trace operator.
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

# Algorithm

The branch of [`target_mean`](@ref) that this tag selects does these steps.

 1. Divide the trace of `sigma` by `T` as `val`. Only this branch needs the keyword `T`.
 2. Return the constant range that repeats `val` `length(mu)` times.

Equation 4.136 of [meucci2005](@cite) states that ``\\mathrm{tr}(\\mathbf{\\Sigma}) / T`` is the error ``\\mathbb{E}\\lVert \\hat{\\boldsymbol{\\mu}} - \\boldsymbol{\\mu} \\rVert_2^2`` of the sample mean. Equation 3.29 of [fengpalomar2016](@cite) restates it. Equation 3.43 of [cajas2025](@cite) uses that value as a shrinkage target. The search for the original source found no earlier work that does so, so the target follows [cajas2025](@cite).

# Constructors

    MeanSquaredError() -> MeanSquaredError

# Examples

```jldoctest
julia> MeanSquaredError()
MeanSquaredError()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:meucci2005]) Section 4.4, Equation 4.136.
  - $(ref_dict[:fengpalomar2016]) Section 3.3.1, Equation 3.29.
  - $(ref_dict[:cajas2025]) Equation 3.43. $(ref_dict[:no_original_source])
"""
struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Chooses the interval that holds the shrinkage intensity of a blend of the sample expected returns and the target.

[`JamesStein`](@ref) and [`BayesStein`](@ref) hold a member of this family in their field `intensity`. Each computes the blend ``(1 - \\alpha) \\hat{\\boldsymbol{\\mu}} + \\alpha \\boldsymbol{b}``. The member takes the raw intensity ``\\alpha`` of the formula and returns the intensity that the blend uses. [`UnitIntervalIntensity`](@ref) keeps the blend on the segment that joins the sample mean and the target. [`NonNegativeIntensity`](@ref) keeps it on the ray that starts at the sample mean and goes through the target. [`UnclampedIntensity`](@ref) returns the raw intensity.

# Interfaces

To add a rule for the shrinkage intensity, subtype `AbstractShrunkExpectedReturnsIntensity`, keep its parameters in the fields of the struct, and add this method:

## `shrinkage_intensity`

  - `shrinkage_intensity(rule::MyShrunkExpectedReturnsIntensity, alpha::Number) -> Number`: Returns the intensity that the blend uses.

### Arguments

  - `rule`: The member of the family.
  - `alpha`: The raw intensity of the formula of the algorithm.

### Returns

  - `alpha::Number`: The intensity that the blend uses.

# Related

  - [`UnitIntervalIntensity`](@ref)
  - [`NonNegativeIntensity`](@ref)
  - [`UnclampedIntensity`](@ref)
  - [`shrinkage_intensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsIntensity <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Clamps the shrinkage intensity into ``[0, 1]``.

The blend then lies on the segment that joins the sample expected returns and the target. An intensity below zero gives the sample mean, and an intensity above one gives the target. This is the default rule of [`JamesStein`](@ref) and [`BayesStein`](@ref).

# Mathematical definition

```math
\\begin{align}
\\alpha^{\\prime} &= \\min\\left(\\max\\left(\\alpha,\\, 0\\right),\\, 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\alpha``: The raw intensity of the formula of the algorithm.
  - ``\\alpha^{\\prime}``: The intensity that the blend uses.

Equation 4.139 of [meucci2005](@cite) gives the James-Stein intensity. The text after it constrains the weight to the interval ``(0, 1)``. This rule includes the two ends of that interval, so the blend can be equal to the sample mean or to the target.

# Examples

```jldoctest
julia> PortfolioOptimisers.shrinkage_intensity(UnitIntervalIntensity(), -2.5)
0.0

julia> PortfolioOptimisers.shrinkage_intensity(UnitIntervalIntensity(), 1.5)
1.0
```

# Related

  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`NonNegativeIntensity`](@ref)
  - [`UnclampedIntensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)

# References

  - $(ref_dict[:meucci2005]) Section 4.4, Equations 4.138 and 4.139.
"""
struct UnitIntervalIntensity <: AbstractShrunkExpectedReturnsIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Clamps the shrinkage intensity at zero from below.

An intensity below zero gives the sample expected returns. The rule keeps an intensity above one, so the blend can go past the target.

# Mathematical definition

```math
\\begin{align}
\\alpha^{\\prime} &= \\max\\left(\\alpha,\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\alpha``: The raw intensity of the formula of the algorithm.
  - ``\\alpha^{\\prime}``: The intensity that the blend uses.

Equation 3.31 of [fengpalomar2016](@cite) takes the James-Stein intensity "if it is positive, otherwise it is zero". The James-Stein intensity is negative exactly when ``\\mathrm{tr}(\\hat{\\mathbf{\\Sigma}}) < 2 \\lambda_{\\max}``. Theorem 1 of [bock1975](@cite) shows that, under this condition, the only minimax estimator of its class is the sample mean. The text after Theorem 5.7 of [lehmanncasella1998](@cite) states the same.

# Examples

```jldoctest
julia> PortfolioOptimisers.shrinkage_intensity(NonNegativeIntensity(), -2.5)
0.0

julia> PortfolioOptimisers.shrinkage_intensity(NonNegativeIntensity(), 1.5)
1.5
```

# Related

  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`UnitIntervalIntensity`](@ref)
  - [`UnclampedIntensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)

# References

  - $(ref_dict[:fengpalomar2016]) Section 3.3.1, Equation 3.31.
  - $(ref_dict[:bock1975]) Theorem 1.
  - $(ref_dict[:lehmanncasella1998]) Section 5.5, Theorem 5.7.
"""
struct NonNegativeIntensity <: AbstractShrunkExpectedReturnsIntensity end
"""
$(DocStringExtensions.TYPEDEF)

Returns the shrinkage intensity of the formula unchanged.

This rule does not keep the intensity inside ``[0, 1]``, so the blend can move away from the target or go past it. The move can be large. A covariance matrix whose largest eigenvalue is more than half of its trace gives a negative James-Stein intensity. A small distance between the sample mean and the target makes the magnitude of that intensity large.

# Mathematical definition

```math
\\begin{align}
\\alpha^{\\prime} &= \\alpha\\,.
\\end{align}
```

Where:

  - ``\\alpha``: The raw intensity of the formula of the algorithm.
  - ``\\alpha^{\\prime}``: The intensity that the blend uses.

Equation 6 of [james1961](@cite) does not clamp its factor.

# Examples

```jldoctest
julia> PortfolioOptimisers.shrinkage_intensity(UnclampedIntensity(), -2.5)
-2.5
```

# Related

  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`UnitIntervalIntensity`](@ref)
  - [`NonNegativeIntensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)

# References

  - $(ref_dict[:james1961]) Section 2, Equation 6.
"""
struct UnclampedIntensity <: AbstractShrunkExpectedReturnsIntensity end
"""
    shrinkage_intensity(rule::UnitIntervalIntensity, alpha::Number) -> Number
    shrinkage_intensity(rule::NonNegativeIntensity, alpha::Number) -> Number
    shrinkage_intensity(rule::UnclampedIntensity, alpha::Number) -> Number

Return the intensity that a blend of the sample expected returns and the target uses, from the raw intensity `alpha`.

[`UnitIntervalIntensity`](@ref), [`NonNegativeIntensity`](@ref) and [`UnclampedIntensity`](@ref) state their formulas. A `NaN` stays a `NaN` under each rule.

# Arguments

  - `rule`: The member of [`AbstractShrunkExpectedReturnsIntensity`](@ref).
  - `alpha`: The raw intensity of the formula of the algorithm.

# Returns

  - `alpha::Number`: The intensity that the blend uses.

# Related

  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
"""
function shrinkage_intensity(::UnitIntervalIntensity, alpha::Number)
    return clamp(alpha, zero(alpha), one(alpha))
end
function shrinkage_intensity(::NonNegativeIntensity, alpha::Number)
    return max(alpha, zero(alpha))
end
function shrinkage_intensity(::UnclampedIntensity, alpha::Number)
    return alpha
end
"""
$(DocStringExtensions.TYPEDEF)

Blends the sample expected returns with the target under an intensity read off the covariance eigenvalues.

The intensity is the trace of the covariance matrix minus twice its largest eigenvalue, divided by the number of observations and by the squared distance between the sample mean and the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JamesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean(),
        intensity::AbstractShrunkExpectedReturnsIntensity = UnitIntervalIntensity()
    ) -> JamesStein

Keywords correspond to the struct's fields. The default `intensity` keeps the blend on the segment that joins the sample mean and the target. The intensity of the formula is negative when the largest eigenvalue of the covariance matrix is more than half of its trace. [`UnclampedIntensity`](@ref) keeps that negative value.

# Examples

```jldoctest
julia> JamesStein()
JamesStein
        tgt ┼ GrandMean()
  intensity ┴ UnitIntervalIntensity()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:james1961]) Section 2, Equation 6.
  - $(ref_dict[:bock1975]) Theorems 1 and 2.
  - $(ref_dict[:meucci2005]) Section 4.4, Equations 4.138 and 4.139.
  - $(ref_dict[:lehmanncasella1998]) Section 5.5, Theorem 5.7.
"""
@concrete struct JamesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    """
    $(field_dict[:mu_intensity])
    """
    intensity
    function JamesStein(tgt::AbstractShrunkExpectedReturnsTarget,
                        intensity::AbstractShrunkExpectedReturnsIntensity)
        return new{typeof(tgt), typeof(intensity)}(tgt, intensity)
    end
end
function JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean(),
                    intensity::AbstractShrunkExpectedReturnsIntensity = UnitIntervalIntensity())::JamesStein
    return JamesStein(tgt, intensity)
end
"""
$(DocStringExtensions.TYPEDEF)

Blends the sample expected returns with the target under an empirical Bayes intensity.

The intensity falls as the number of observations increases, and as the distance between the sample mean and the target increases. The inverse covariance matrix weights that distance. A short sample therefore moves the estimate further toward the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BayesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean(),
        intensity::AbstractShrunkExpectedReturnsIntensity = UnitIntervalIntensity()
    ) -> BayesStein

Keywords correspond to the struct's fields. The intensity of the formula lies in ``(0, 1]`` when the covariance matrix is positive definite. The field `intensity` therefore changes the result only for a covariance estimator that returns an indefinite matrix.

# Examples

```jldoctest
julia> BayesStein()
BayesStein
        tgt ┼ GrandMean()
  intensity ┴ UnitIntervalIntensity()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`AbstractShrunkExpectedReturnsIntensity`](@ref)
  - [`JamesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:jorion1986]) Section IV, Equations 14, 17 and 18.
"""
@concrete struct BayesStein <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    """
    $(field_dict[:mu_intensity])
    """
    intensity
    function BayesStein(tgt::AbstractShrunkExpectedReturnsTarget,
                        intensity::AbstractShrunkExpectedReturnsIntensity)
        return new{typeof(tgt), typeof(intensity)}(tgt, intensity)
    end
end
function BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean(),
                    intensity::AbstractShrunkExpectedReturnsIntensity = UnitIntervalIntensity())::BayesStein
    return BayesStein(tgt, intensity)
end
"""
$(DocStringExtensions.TYPEDEF)

Combines the sample expected returns and the target under two coefficients from random matrix theory.

The method computes the two coefficients separately, and neither is a convex weight. The result is therefore not a blend of the sample mean and the target. The algorithm is for a universe whose number of assets is a large fraction of its number of observations. It needs more observations than assets.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BodnarOkhrinParolya(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BodnarOkhrinParolya

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> BodnarOkhrinParolya()
BodnarOkhrinParolya
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}, X::MatNum; dims::Int = 1, kwargs...)`](@ref)

# References

  - $(ref_dict[:bodnar2019]) Section 3, Theorem 3, Equations 6 and 7.
"""
@concrete struct BodnarOkhrinParolya <: AbstractShrunkExpectedReturnsAlgorithm
    """
    $(field_dict[:mutgt])
    """
    tgt
    function BodnarOkhrinParolya(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BodnarOkhrinParolya(;
                             tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BodnarOkhrinParolya
    return BodnarOkhrinParolya(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinks the sample expected returns toward a target chosen by the shrinkage algorithm.

It holds a mean estimator, a covariance estimator and a shrinkage algorithm. The shrinkage algorithm holds the target.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShrunkExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein()
    ) -> ShrunkExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `me`: Recursively updated via [`factory`](@ref).
  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `me`: Recursively viewed via [`port_opt_view`](@ref).
  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `alg`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> ShrunkExpectedReturns()
ShrunkExpectedReturns
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┼ FullMoment()
      │      │     w ┴ nothing
      │   mp ┼ MatrixProcessing
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
  alg ┼ JamesStein
      │         tgt ┼ GrandMean()
      │   intensity ┴ UnitIntervalIntensity()
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`target_mean`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct ShrunkExpectedReturns <:
                               AbstractShrunkExpectedReturnsEstimator
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:me_shrink_alg])
    """
    @vprop alg
    function ShrunkExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                   ce::StatsBase.CovarianceEstimator,
                                   alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())::ShrunkExpectedReturns
    return ShrunkExpectedReturns(me, ce, alg)
end
"""
    target_mean(
        tgt::GrandMean,
        mu::ArrNum,
        sigma::MatNum,
        args...;
        kwargs...
    ) -> StepRangeLen

    target_mean(
        tgt::VolatilityWeighted,
        mu::ArrNum,
        sigma::MatNum,
        isigma::Option{<:MatNum} = nothing;
        kwargs...
    ) -> StepRangeLen

    target_mean(
        tgt::MeanSquaredError,
        mu::ArrNum,
        sigma::MatNum,
        args...;
        T::Integer,
        kwargs...
    ) -> StepRangeLen

Compute the shrinkage target vector for expected returns estimation.

[`JamesStein`](@ref), [`BayesStein`](@ref) and [`BodnarOkhrinParolya`](@ref) all call `target_mean`, so the library computes each target in one place.

Every element of the returned vector holds the same value, so the function returns a `StepRangeLen` and not a dense vector.

# Algorithm

Julia selects the method from the type of `tgt`. The docstring of each target type states the closed form of its method. Every method computes one scalar `val` and returns the constant range `range(val, val; length = length(mu))`.

 1. If `tgt` is a [`GrandMean`](@ref), take the unweighted mean of `mu`. This method does not read `sigma`, `isigma` or `T`.
 2. If `tgt` is a [`VolatilityWeighted`](@ref), solve for `isigma` when the caller did not pass it. Flatten `mu` when it has one row. Then divide the sum of `isigma * mu` by the sum of `isigma`.
 3. If `tgt` is a [`MeanSquaredError`](@ref), divide the trace of `sigma` by the keyword `T`. This method does not read the values of `mu` or `isigma`. It reads only the length of `mu`.

Each method accepts, through `args...` and `kwargs...`, the arguments that only the other methods read. One call therefore works for all three targets.

# Arguments

  - `tgt`: The shrinkage target type.

      + `tgt::GrandMean`: Fills the vector with the mean of `mu`.
      + `tgt::VolatilityWeighted`: Fills the vector with the inverse-covariance-weighted mean of `mu`.
      + `tgt::MeanSquaredError`: Fills the vector with the trace of `sigma` divided by `T`.

  - `mu`: The array of the sample expected returns, a vector or a matrix with one row or one column.

  - $(arg_dict[:sigma])

  - `isigma`: The inverse covariance matrix. The [`VolatilityWeighted`](@ref) method takes it as a positional argument. If it is `nothing`, that method computes `sigma \\ LinearAlgebra.I`. The other two methods accept it in `args...` and do not read it.

  - `T`: The number of observations. The [`MeanSquaredError`](@ref) method needs it as a keyword. The other two methods accept it in `kwargs...` and do not read it.

  - `kwargs...`: More keyword arguments. The function does not read them.

# Returns

  - `b::StepRangeLen`: The shrinkage target vector, of length `length(mu)`.

# Related

  - [`GrandMean`](@ref): the closed form of the branch of step 1.
  - [`VolatilityWeighted`](@ref): the closed form of the branch of step 2.
  - [`MeanSquaredError`](@ref): the closed form of the branch of step 3.
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`ArrNum`](@ref)
  - [`MatNum`](@ref)
  - [`Option`](@ref)

# References

  - $(ref_dict[:meucci2005]) Section 4.4, Equations 4.136, 4.141 and 4.142.
  - $(ref_dict[:jorion1986]) Section IV, Equation 14.
  - $(ref_dict[:cajas2025]) Equation 3.43. $(ref_dict[:no_original_source])
"""
function target_mean(::GrandMean, mu::ArrNum, sigma::MatNum, args...; kwargs...)
    val = Statistics.mean(mu)
    return range(val, val; length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::ArrNum, sigma::MatNum,
                     isigma::Option{<:MatNum} = nothing; kwargs...)
    if isnothing(isigma)
        isigma = sigma \ LinearAlgebra.I
    end
    if isone(size(mu, 1))
        mu = vec(mu)
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(val, val; length = length(mu))
end
function target_mean(::MeanSquaredError, mu::ArrNum, sigma::MatNum, args...; T::Integer,
                     kwargs...)
    val = LinearAlgebra.tr(sigma) / T
    return range(val, val; length = length(mu))
end
"""
    Statistics.mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute the shrunk expected returns of the estimator `me`.

The method moves the sample expected returns toward a target. The aim is a smaller estimation error of the mean when the number of assets is large compared with the number of observations. [`JamesStein`](@ref) and [`BayesStein`](@ref) return `(1 - alpha) * mu + alpha * b`. The member of [`AbstractShrunkExpectedReturnsIntensity`](@ref) in their field `intensity` sets the interval of `alpha`. The default [`UnitIntervalIntensity`](@ref) clamps `alpha` into ``[0, 1]``, so the result lies on the segment that joins the sample mean and the target. Under [`UnclampedIntensity`](@ref), the result can lie outside that segment. [`BodnarOkhrinParolya`](@ref) computes its two coefficients separately. They do not sum to one, and no rule clamps them.

# Mathematical definition

James-Stein shrinkage of the sample expected returns toward the target:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}}_{JS} &= (1 - \\alpha^{\\prime})\\, \\hat{\\boldsymbol{\\mu}} + \\alpha^{\\prime}\\, \\boldsymbol{b}\\,, \\\\
\\alpha &= \\frac{N \\bar{\\lambda} - 2 \\lambda_{\\max}}{T \\, \\lVert \\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b} \\rVert_2^2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}_{JS}``: James-Stein shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:alpha_shrink_mu])
  - ``\\alpha^{\\prime}``: The intensity that the member in `intensity` returns from ``\\alpha``. It is ``\\min(\\max(\\alpha, 0), 1)`` under the default [`UnitIntervalIntensity`](@ref).
  - ``\\bar{\\lambda}``: Mean eigenvalue of the covariance matrix.
  - ``\\lambda_{\\max}``: Maximum eigenvalue of the covariance matrix.
  - $(math_dict[:T])
  - $(math_dict[:N])

The form has these limits.

  - ``N \\bar{\\lambda}`` is the trace of the covariance matrix. Thus ``N \\bar{\\lambda} \\leq 2 \\lambda_{\\max}`` when ``N \\leq 2``. For any ``N``, it also holds when the largest eigenvalue is at least half of the trace, as with one dominant factor. The intensity ``\\alpha`` is then not positive.
  - Under that condition, [`UnitIntervalIntensity`](@ref) and [`NonNegativeIntensity`](@ref) return the sample mean. Under [`UnclampedIntensity`](@ref), the blend moves away from the target. Theorem 1 of [bock1975](@cite) gives the same condition, ``\\mathrm{tr}(\\mathbf{D}) \\leq 2 d_L``. Under it, the only minimax estimator of its class is the sample mean.
  - The denominator is zero when the target is equal to the sample mean. [`GrandMean`](@ref) and [`VolatilityWeighted`](@ref) are both equal to the sample mean at ``N = 1``, so a one-asset sample raises a `DomainError` under either of them. [`MeanSquaredError`](@ref) does not read the sample mean, so it stays finite there.

Equation 6 of [james1961](@cite) shrinks a normal mean with an identity covariance matrix toward the origin by the factor ``1 - (N - 2) / \\lVert \\boldsymbol{x} \\rVert_2^2``. It does not clamp the factor. Theorem 2 of [bock1975](@cite) extends the minimax property to a known covariance matrix ``\\mathbf{D}`` that is not the identity. The factor is ``1 - c / (\\boldsymbol{x}^\\intercal \\mathbf{D}^{-1} \\boldsymbol{x})``, with ``0 \\leq c \\leq 2 (\\mathrm{tr}(\\mathbf{D}) / d_L - 2)``, where ``d_L`` is the largest eigenvalue of ``\\mathbf{D}``.

The intensity above divides by the squared Euclidean norm and not by the ``\\mathbf{D}^{-1}`` quadratic form, so it is not the form of [bock1975](@cite). It is Equation 4.139 of [meucci2005](@cite), which sets the weight of Equation 4.138. That equation writes ``N \\bar{\\lambda} - 2 \\lambda_{\\max}`` for ``\\mathrm{tr}(\\mathbf{\\Sigma}) - 2 \\lambda_{\\max}``. Equation 3.31 of [fengpalomar2016](@cite) restates it.

  - [meucci2005](@cite) constrains the weight to the interval ``(0, 1)``. [`UnitIntervalIntensity`](@ref) follows this rule.
  - [fengpalomar2016](@cite) sets a negative intensity to zero. [`NonNegativeIntensity`](@ref) follows this rule.
  - [meucci2005](@cite) replaces the known covariance matrix by an estimate. This method does the same with the sample covariance matrix. [`GrandMean`](@ref) and [`VolatilityWeighted`](@ref) read the target off the same sample.

Theorem 5.7 of [lehmanncasella1998](@cite) states the Euclidean form ``(1 - c(\\lVert \\boldsymbol{x} \\rVert_2^2) / \\lVert \\boldsymbol{x} \\rVert_2^2) \\boldsymbol{x}`` for a known covariance matrix. It requires that ``c \\geq 0`` and that ``c`` is nondecreasing. It notes that the estimator differs from ``\\boldsymbol{x}`` only when ``\\mathrm{tr}(\\mathbf{\\Sigma}) > 2 \\lambda_{\\max}``. The covariance matrix of the sample mean is ``\\hat{\\mathbf{\\Sigma}} / T``, so the constant of this method is ``c = (\\mathrm{tr}(\\hat{\\mathbf{\\Sigma}}) - 2 \\lambda_{\\max}) / T``.

The theorem prints the upper bound ``2 \\mathrm{tr}(\\mathbf{\\Sigma}) / \\lambda_{\\max} - 4``. This bound does not scale with ``\\mathbf{\\Sigma}`` as ``c`` does. Its Equation 5.15 writes the factor ``(c + 4) \\boldsymbol{x}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{x} / \\lVert \\boldsymbol{x} \\rVert_2^2``. Stein's identity gives ``c + 4 \\boldsymbol{x}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{x} / \\lVert \\boldsymbol{x} \\rVert_2^2``. With that term, Equation 5.16 gives the bound ``0 \\leq c \\leq 2 (\\mathrm{tr}(\\mathbf{\\Sigma}) - 2 \\lambda_{\\max})``. The constant of this method is the midpoint of that interval, which minimises the bound on the risk.

# Algorithm

 1. Compute the sample expected returns `mu` with `me.me`.
 2. Compute the covariance matrix `sigma` with `me.ce`.
 3. Read `T` and `N` off `size(X)`. Swap them when `dims` is `2`.
 4. Compute the shrinkage target `b` with [`target_mean`](@ref). Transpose it into a row when `dims` is `1`.
 5. Compute the eigenvalues `evals` of `sigma`.
 6. Subtract `b` from `mu` as `mb`. Compute the intensity `alpha` from `evals`, `mb`, `N` and `T`.
 7. Pass `alpha` through [`shrinkage_intensity`](@ref) with the member in `me.alg.intensity`.
 8. Return the blend `(1 - alpha) * mu + alpha * b`.

# Arguments

  - `me`: The shrunk expected returns estimator.

      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.

  - $(arg_dict[:X])

  - $(arg_dict[:dims])

  - `kwargs...`: More keyword arguments. The method passes them to the mean estimator and to the covariance estimator.

# Validation

  - `!iszero(dot(mu - b, mu - b))`. The [`JamesStein`](@ref) intensity divides by this value. It is exactly zero when the target is equal to the sample mean. [`GrandMean`](@ref) and [`VolatilityWeighted`](@ref) are both equal to the sample mean at ``N = 1``. The other two overloads state their own checks.

# Returns

  - $(ret_dict[:mu])

# Related

  - [`JamesStein`](@ref): the tag that selects this method.
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
  - [`ArrNum`](@ref)
  - [`MatNum`](@ref)

# References

  - $(ref_dict[:james1961]) Section 2, Equation 6.
  - $(ref_dict[:bock1975]) Theorems 1 and 2.
  - $(ref_dict[:meucci2005]) Section 4.4, Equations 4.138 and 4.139.
  - $(ref_dict[:lehmanncasella1998]) Section 5.5, Theorem 5.7.
  - $(ref_dict[:fengpalomar2016]) Section 3.3.1, Equation 3.31.
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    b = target_mean(me.alg.tgt, mu, sigma; T = T)
    if flag
        b = transpose(b)
    end
    evals = LinearAlgebra.eigvals(sigma)
    mb = mu - b
    mb2 = LinearAlgebra.dot(mb, mb)
    @argcheck(!iszero(mb2),
              DomainError(mb2,
                          "the James-Stein intensity divides by `dot(mu - b, mu - b)`, which is exactly zero because the $(nameof(typeof(me.alg.tgt))) target equals the sample mean. GrandMean and VolatilityWeighted both reduce to the sample mean at N == 1, got N = $N"))
    alpha = (N * Statistics.mean(evals) - 2 * maximum(evals)) / mb2 / T
    alpha = shrinkage_intensity(me.alg.intensity, alpha)
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BayesStein`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Shrinks the sample expected returns toward the target under the empirical Bayes intensity of [jorion1986](@cite). The intensity reads the inverse covariance matrix.

# Mathematical definition

Bayes-Stein shrinkage of the sample expected returns toward the target:

```math
\\begin{align}
\\alpha &= \\frac{N + 2}{(N + 2) + T \\, (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} (\\hat{\\boldsymbol{\\mu}} - \\boldsymbol{b})}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BS} &= (1 - \\alpha)\\hat{\\boldsymbol{\\mu}} + \\alpha \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_shrink_mu])
  - ``\\hat{\\boldsymbol{\\mu}}_{BS}``: Bayes-Stein shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

The form has these properties.

  - The quadratic form is non-negative when ``\\hat{\\mathbf{\\Sigma}}`` is positive definite, so ``\\alpha`` then lies in ``(0, 1]``. Of the three algorithms, only this one gives a convex weight without a clamp. It returns the target exactly when the quadratic form is zero.
  - When a covariance estimator returns an indefinite matrix, ``\\alpha`` can lie outside ``(0, 1]``. The member in `intensity` then sets the interval, as for [`JamesStein`](@ref). Under the default [`UnitIntervalIntensity`](@ref), the blend stays on the segment.
  - The quadratic form uses the inverse of the covariance matrix that `me.ce` returns. Equation 18 of [jorion1986](@cite) puts the bias-corrected matrix ``\\bar{\\mathbf{\\Sigma}} = \\frac{T-1}{T-N-2} \\mathbf{S}`` in the quadratic form. ``\\mathbf{S}`` is the sample covariance matrix that divides by ``T - 1``.
  - Equation 3.44 of [cajas2025](@cite) writes the factor ``\\frac{T-1}{T-N-1}`` instead. This method uses ``\\hat{\\mathbf{\\Sigma}}`` and applies neither factor. Both factors are more than one. Each of them makes ``\\alpha`` larger when it multiplies the matrix.

Equation 17 of [jorion1986](@cite) is this intensity. Equation 14 of that work gives its target, the minimum variance grand mean of [`VolatilityWeighted`](@ref). Equation 3.44 of [cajas2025](@cite) lets the target take the same values as for [`JamesStein`](@ref). The use of [`GrandMean`](@ref) and [`MeanSquaredError`](@ref) here follows [cajas2025](@cite) alone.

# Algorithm

 1. Compute the sample expected returns `mu` with `me.me`.
 2. Compute the covariance matrix `sigma` with `me.ce`.
 3. Read `T` and `N` off `size(X)`. Swap them when `dims` is `2`.
 4. Solve `sigma \\ LinearAlgebra.I` as `isigma`. Pass it to [`target_mean`](@ref), so that the [`VolatilityWeighted`](@ref) branch does not solve a second time.
 5. Compute the shrinkage target `b`. Transpose it into a row when `dims` is `1`.
 6. Flatten `mu - b` with `vec` as `mb`. Compute the intensity `alpha` from `mb`, `isigma`, `N` and `T`.
 7. Pass `alpha` through [`shrinkage_intensity`](@ref) with the member in `me.alg.intensity`.
 8. Return the blend `(1 - alpha) * mu + alpha * b`.

# Related

  - [`BayesStein`](@ref): the tag that selects this method.
  - [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): the arguments and the return value that the three overloads share.
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:jorion1986]) Section IV, Equations 14, 17 and 18.
  - $(ref_dict[:cajas2025]) Section 3.4.1.2, Equation 3.44. $(ref_dict[:no_original_source])
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * LinearAlgebra.dot(mb, isigma, mb))
    alpha = shrinkage_intensity(me.alg.intensity, alpha)
    return (one(alpha) - alpha) * mu + alpha * b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`BodnarOkhrinParolya`](@ref) overload of [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref). Combines the sample expected returns and the target under the two coefficients of [bodnar2019](@cite). It needs ``T > N``. The term ``N/(T-N)`` is undefined at ``T = N`` and changes sign below it, so a square or wide returns matrix raises a `DomainError`.

# Mathematical definition

The method computes three products of the sample mean and the target, weighted by the inverse covariance matrix:

```math
\\begin{align}
u &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\hat{\\boldsymbol{\\mu}}\\,, \\\\
v &= \\boldsymbol{b}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,, \\\\
w &= \\hat{\\boldsymbol{\\mu}}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{b}\\,.
\\end{align}
```

The two coefficients and the combination follow from them:

```math
\\begin{align}
\\alpha &= \\frac{(u - N/(T-N))v - w^2}{uv - w^2}\\,, \\\\
\\beta &= \\frac{(1-\\alpha) w}{v}\\,, \\\\
\\hat{\\boldsymbol{\\mu}}_{BOP} &= \\alpha \\hat{\\boldsymbol{\\mu}} + \\beta \\boldsymbol{b}\\,.
\\end{align}
```

Where:

  - ``u``, ``v``, ``w``: The three products above.
  - ``\\alpha``, ``\\beta``: The coefficients of the sample mean and of the target.
  - ``\\hat{\\boldsymbol{\\mu}}_{BOP}``: Bodnar-Okhrin-Parolya shrunk expected returns.
  - $(math_dict[:mu_hat_shrink])
  - $(math_dict[:b_shrink_tgt])
  - $(math_dict[:Sigma_hat])
  - $(math_dict[:T])
  - $(math_dict[:N])

The form has these properties, which the other two algorithms do not have.

  - The method computes ``\\alpha`` and ``\\beta`` separately, and they do not sum to one. The result is therefore not a point on the segment that joins ``\\hat{\\boldsymbol{\\mu}}`` and ``\\boldsymbol{b}``.
  - The ``w^2`` terms cancel, so ``\\alpha = 1 - \\frac{N}{T-N} \\frac{v}{uv - w^2}``. Thus ``\\alpha < 1`` always, and ``\\alpha < 0`` exactly when ``\\frac{N}{T-N} v > uv - w^2``. The combination then moves away from the sample mean.
  - ``uv - w^2`` is a Cauchy-Schwarz gap in the inner product ``\\langle \\boldsymbol{x}, \\boldsymbol{y} \\rangle = \\boldsymbol{x}^\\intercal \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{y}``. It is zero exactly when the target is a multiple of the sample mean. At ``N = 1`` every vector is such a multiple, so a one-asset sample raises a `DomainError` under all three targets.
  - ``\\beta \\boldsymbol{b}`` is the projection of ``(1-\\alpha) \\hat{\\boldsymbol{\\mu}}`` onto the line through ``\\boldsymbol{b}`` in the same inner product. It depends on the direction of the target and not on its scale.
  - Every target of this file is a multiple of the vector of ones, ``\\boldsymbol{b} = c \\boldsymbol{1}``, so ``v`` scales with ``c^2`` and ``w`` with ``c``. The factor cancels in ``\\alpha`` and in ``\\beta \\boldsymbol{b}``, so the three targets return the same estimate on one sample.

The coefficient ``\\alpha`` is Equation 6 of [bodnar2019](@cite). The coefficient ``\\beta`` is Equation 7, which divides by the quadratic form ``v`` of the target. Equation 3.45 of [cajas2025](@cite) divides by ``u`` instead. That quotient does not minimise the quadratic loss from which [bodnar2019](@cite) derives both coefficients, so this method follows [bodnar2019](@cite).

This method differs from Section 3 of [bodnar2019](@cite) in three ways.

  - That section computes the coefficients from the sample covariance matrix that divides by the number of observations. This method uses the matrix that `me.ce` returns, which divides by ``T - 1`` under the default estimator.
  - That section takes a target that does not depend on the sample. The three targets of this file read the same sample as the mean.
  - That section also gives a suboptimal rule that uses a generalised inverse for ``N / T > 1``. This method does not implement it.

# Algorithm

 1. Compute the sample expected returns `mu` with `me.me`.
 2. Compute the covariance matrix `sigma` with `me.ce`.
 3. Read `T` and `N` off `size(X)`. Swap them when `dims` is `2`.
 4. Solve `sigma \\ LinearAlgebra.I` as `isigma`. Pass it to [`target_mean`](@ref), so that the [`VolatilityWeighted`](@ref) branch does not solve a second time.
 5. Compute the shrinkage target `b`. Transpose it into a row when `dims` is `1`.
 6. When `dims` is `1`, flatten `mu` and `b` with `vec` as `vm` and `vb`. When `dims` is `2`, use `mu` and `b` as they are.
 7. Compute the three products `u`, `v` and `w` from `vm`, `vb` and `isigma`.
 8. Compute `alpha` from `u`, `v`, `w`, `N` and `T`. Then compute `beta` from `alpha`, `w` and `v`.
 9. Return the combination `alpha * mu + beta * b`.

# Validation

  - `T > N`. This is the condition that [bodnar2019](@cite) publishes for the estimator. The term ``N/(T-N)`` is undefined at ``T = N`` and negative below it.
  - `!iszero(u * v - w^2)`. Both coefficients divide by this Cauchy-Schwarz gap. It is exactly zero when the target is a multiple of the sample mean. Every vector is such a multiple at ``N = 1``.

# Related

  - [`BodnarOkhrinParolya`](@ref): the tag that selects this method.
  - [`mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)`](@ref): the arguments and the return value that the three overloads share.
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)

# References

  - $(ref_dict[:bodnar2019]) Section 3, Theorem 3, Equations 6 and 7.
  - $(ref_dict[:cajas2025]) Equation 3.45.
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::MatNum; dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(library_covariance_estimator(me.ce), X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    @argcheck(T > N,
              DomainError((T, N),
                          "the Bodnar-Okhrin-Parolya coefficients contain `N / (T - N)`, so they need more observations than assets. The term is undefined at T == N and negative below it, got T = $T, N = $N"))
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
        vm = vec(mu)
        vb = vec(b)
    else
        vm = mu
        vb = b
    end
    u = LinearAlgebra.dot(vm, isigma, vm)
    v = LinearAlgebra.dot(vb, isigma, vb)
    w = LinearAlgebra.dot(vm, isigma, vb)
    gap = u * v - w^2
    @argcheck(!iszero(gap),
              DomainError(gap,
                          "the Bodnar-Okhrin-Parolya coefficients divide by the Cauchy-Schwarz gap `u * v - w^2`, which is exactly zero because the $(nameof(typeof(me.alg.tgt))) target is a multiple of the sample mean. Every vector is such a multiple at N == 1, got N = $N"))
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= gap
    beta = (one(alpha) - alpha) * w / v
    return alpha * mu + beta * b
end

export GrandMean, VolatilityWeighted, MeanSquaredError, UnitIntervalIntensity,
       NonNegativeIntensity, UnclampedIntensity, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
public AbstractShrunkExpectedReturnsIntensity, shrinkage_intensity
